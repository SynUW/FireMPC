#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MOD11A1 + MYD11A1 per-pixel merging script (parallel + tiled + geographically aligned).
Outputs 6 bands: 1 LST_Day_1km, 2 LST_Night_1km, 3 Emis_31, 4 Emis_32, 5 QC_Day, 6 QC_Night.

Merging rules (per pixel):
- If only one side is valid (not NoData), pick that side.
- If both sides are invalid, output NoData.
- If both sides are valid:
  * LST_Day: score by QC_Day; the better-scoring side wins; QC_Day takes the winner's pixel.
  * LST_Night: score by QC_Night; same as above.
  * Emis_31 / 32: score by emissivity error (by default, take the "better" emissivity error and mandatory / data_quality from QC_Day / QC_Night on that source as the score; configurable).
- Score tuples (smaller is better):
  LST: (mandatory, lst_error, data_quality, emis_error)
  Emis: (mandatory, emis_error, data_quality)

Geometry and pixel alignment:
- Target grid: take mod.crs as CRS (fall back to myd.crs); take the finer pixel size; take the UNION extent.
- Use WarpedVRT to read both sides on the same grid block-by-block, avoiding whole-image reprojection memory blow-ups.
- QC bands use nearest-neighbor resampling; numeric bands (LST / Emis) use bilinear.

Parallelism:
- Multi-process per date (parsed from filenames); each task is itself tiled internally.
"""

import re
import sys
import math
import shutil
import argparse
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import rasterio
from rasterio import windows
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds, calculate_default_transform, Resampling
from rasterio.errors import NotGeoreferencedWarning
import warnings

warnings.simplefilter("ignore", NotGeoreferencedWarning)

DATE_RE = re.compile(r"(?P<y>\d{4})[-_]?((?P<m>\d{2})[-_]?)(?P<d>\d{2})")

def parse_date_key(p: Path) -> str | None:
    m = DATE_RE.search(p.stem)
    if not m:
        return None
    y, mth, d = m.group("y"), m.group("m"), m.group("d")
    return f"{y}_{mth}_{d}"

def bitslice(arr, start, end):
    width = end - start + 1
    return (arr >> start) & ((1 << width) - 1)

def qc_fields(qc):
    return dict(
        mandatory   = bitslice(qc, 0, 1),
        data_quality= bitslice(qc, 2, 3),
        emis_error  = bitslice(qc, 4, 5),
        lst_error   = bitslice(qc, 6, 7),
    )

def score_lst(qc_arr):
    q = qc_fields(qc_arr)
    # (mandatory, lst_error, data_quality, emis_error)
    return np.stack([
        q["mandatory"], q["lst_error"], q["data_quality"], q["emis_error"]
    ], axis=0)

def score_emis(qc_day_arr, qc_night_arr, mode="union"):
    """
    Return the (mandatory, emis_error, data_quality) triple used for emissivity scoring.
    mode:
      - "union": for each pixel, pick the better (smaller-valued) mandatory / emis_error / data_quality between day and night
      - "day": use day only
      - "night": use night only
    """
    qd = qc_fields(qc_day_arr)
    qn = qc_fields(qc_night_arr)
    if mode == "day":
        mand = qd["mandatory"]; emis = qd["emis_error"]; dq = qd["data_quality"]
    elif mode == "night":
        mand = qn["mandatory"]; emis = qn["emis_error"]; dq = qn["data_quality"]
    else:
        mand = np.minimum(qd["mandatory"], qn["mandatory"])
        emis = np.minimum(qd["emis_error"], qn["emis_error"])
        dq   = np.minimum(qd["data_quality"], qn["data_quality"])
    return np.stack([mand, emis, dq], axis=0)

def lexicographic_choose(a_score, b_score, prefer_a=True):
    """
    a_score, b_score: shape=(K,H,W); smaller score is better.
    Returns a bool mask: True picks A, False picks B; ties are resolved by prefer_a.
    """
    # Compare dimension by dimension
    choose_a = np.zeros(a_score.shape[1:], dtype=bool)
    undecided = np.ones_like(choose_a, dtype=bool)
    for k in range(a_score.shape[0]):
        ak = a_score[k]; bk = b_score[k]
        better_a = ak < bk
        better_b = ak > bk
        choose_a |= (better_a & undecided)
        undecided &= ~(better_a | better_b)
        # Loop until no pixel is left undecided
        if not undecided.any():
            break
    # Ties: by preference
    if undecided.any():
        if prefer_a:
            choose_a |= undecided
        # else: default False, leaving to B
    return choose_a

def read_nodata_per_band(src):
    nds = []
    for i in range(1, src.count + 1):
        nds.append(src.nodatavals[i-1])
    return nds

def pick_target_grid(mod_src, myd_src):
    """
    Select the target grid: prefer mod's CRS; take the finer resolution; take the union extent.
    If mod has no CRS, use myd's.
    """
    crs = mod_src.crs or myd_src.crs
    if crs is None:
        raise ValueError("Both inputs lack a valid CRS.")

    # Resolution (smaller absolute value = finer)
    def res_of(src):
        a = src.transform
        return (abs(a.a), abs(a.e))  # (xres, yres); note yres is negative, so take absolute value
    mod_res = res_of(mod_src)
    myd_res = res_of(myd_src)
    xres = min(mod_res[0], myd_res[0])
    yres = min(mod_res[1], myd_res[1])

    # union bounds (in crs)
    mod_bounds = transform_bounds(mod_src.crs, crs, *mod_src.bounds, densify_pts=0) if mod_src.crs != crs else mod_src.bounds
    myd_bounds = transform_bounds(myd_src.crs, crs, *myd_src.bounds, densify_pts=0) if myd_src.crs != crs else myd_src.bounds
    xmin = min(mod_bounds[0], myd_bounds[0])
    ymin = min(mod_bounds[1], myd_bounds[1])
    xmax = max(mod_bounds[2], myd_bounds[2])
    ymax = max(mod_bounds[3], myd_bounds[3])

    # Compute output affine, width, and height
    dst_transform, width, height = calculate_default_transform(
        crs, crs,  # Only used for width/height computation; pass the same CRS
        int((xmax - xmin) / xres), int((ymax - ymin) / yres),
        left=xmin, bottom=ymin, right=xmax, top=ymax,
        resolution=(xres, yres)
    )
    # Fix: the calculate_default_transform call above is only a quick generator; constructing directly is simpler
    from affine import Affine
    dst_transform = Affine(xres, 0.0, xmin, 0.0, -yres, ymax)
    width  = int(math.ceil((xmax - xmin) / xres))
    height = int(math.ceil((ymax - ymin) / yres))

    return crs, dst_transform, width, height, (xres, yres)

def band_types(src):
    return [np.dtype(dt) for dt in src.dtypes]

def is_nan_nodata(nd):
    return isinstance(nd, float) and np.isnan(nd)

def valid_mask(arr, nd):
    if is_nan_nodata(nd):
        return ~np.isnan(arr)
    return arr != nd

def write_profile_template(crs, transform, width, height, dtypes, compress="LZW"):
    # Output 6 bands; per-band dtype follows the input LST/Emis preference (QC is integer)
    profile = dict(
        driver="GTiff",
        count=6,
        crs=crs,
        transform=transform,
        width=width,
        height=height,
        tiled=True,
        blockxsize=min(1024, width),
        blockysize=min(1024, height),
        compress=compress,
        bigtiff="IF_SAFER"
    )
    # If per-band dtypes are needed, cast per band after writing; here we assume both sides share dtypes and write per the first side (mod)
    profile["dtype"] = dtypes[0].name
    return profile

def choose_and_write(date_key, mod_path, myd_path, out_dir, prefer, emis_qc_mode,
                     chunk, compress):
    out_path = out_dir / f"{date_key}.tif"

    # Only one side exists: copy directly (without changing the grid or metadata)
    if (mod_path is None) ^ (myd_path is None):
        src_path = mod_path or myd_path
        shutil.copy2(src_path, out_path)
        return True, "copied"

    # Both sides exist: merge
    with rasterio.open(mod_path) as mod, rasterio.open(myd_path) as myd:
        if mod.count < 6 or myd.count < 6:
            raise ValueError("Input must have 6 bands (LST_Day, LST_Night, Emis_31, Emis_32, QC_Day, QC_Night).")

        # Target grid (CRS prefers mod, finer resolution, union extent)
        crs, dst_transform, width, height, _ = pick_target_grid(mod, myd)

        # Build the aligned VRT: bilinear for LST/Emis, nearest for QC
        vrt_opts_float = dict(crs=crs, transform=dst_transform, width=width, height=height,
                              resampling=Resampling.bilinear, add_alpha=False)
        vrt_opts_near  = dict(crs=crs, transform=dst_transform, width=width, height=height,
                              resampling=Resampling.nearest, add_alpha=False)

        # Per-band nodata
        mod_nd = read_nodata_per_band(mod)
        myd_nd = read_nodata_per_band(myd)

        # Target output dtype: reuse mod's first band (typically int16 or float32)
        dtypes = band_types(mod)
        profile = write_profile_template(crs, dst_transform, width, height, dtypes, compress=compress)

        # Unified per-band output nodata: NaN if source is float, otherwise -32768
        out_nd = []
        for i in range(6):
            dt = dtypes[i] if i < len(dtypes) else np.dtype("float32")
            if np.issubdtype(dt, np.floating):
                out_nd.append(np.nan)
            else:
                out_nd.append(np.int64(-32768))

        with rasterio.open(out_path, "w", **profile) as dst, \
             WarpedVRT(mod, **vrt_opts_float) as mod_vrt_f, WarpedVRT(myd, **vrt_opts_float) as myd_vrt_f, \
             WarpedVRT(mod, **vrt_opts_near)  as mod_vrt_n, WarpedVRT(myd, **vrt_opts_near)  as myd_vrt_n:

            # Block iteration
            for row_off in range(0, height, chunk):
                h = min(chunk, height - row_off)
                for col_off in range(0, width, chunk):
                    w = min(chunk, width - col_off)
                    win = windows.Window(col_off, row_off, w, h)

                    # Read numeric bands (bilinear)
                    m_day   = mod_vrt_f.read(1, window=win)
                    m_nite  = mod_vrt_f.read(2, window=win)
                    m_e31   = mod_vrt_f.read(3, window=win)
                    m_e32   = mod_vrt_f.read(4, window=win)

                    y_day   = myd_vrt_f.read(1, window=win)
                    y_nite  = myd_vrt_f.read(2, window=win)
                    y_e31   = myd_vrt_f.read(3, window=win)
                    y_e32   = myd_vrt_f.read(4, window=win)

                    # Read QC (nearest neighbor)
                    m_qd    = mod_vrt_n.read(5, window=win).astype(np.uint16, copy=False)
                    m_qn    = mod_vrt_n.read(6, window=win).astype(np.uint16, copy=False)
                    y_qd    = myd_vrt_n.read(5, window=win).astype(np.uint16, copy=False)
                    y_qn    = myd_vrt_n.read(6, window=win).astype(np.uint16, copy=False)

                    # Valid mask (based on GeoTIFF nodata; nodata also propagates after resampling)
                    mv_day  = valid_mask(m_day,  mod_nd[0])
                    mv_nite = valid_mask(m_nite, mod_nd[1])
                    mv_e31  = valid_mask(m_e31,  mod_nd[2])
                    mv_e32  = valid_mask(m_e32,  mod_nd[3])

                    yv_day  = valid_mask(y_day,  myd_nd[0])
                    yv_nite = valid_mask(y_nite, myd_nd[1])
                    yv_e31  = valid_mask(y_e31,  myd_nd[3-1])  # index 2
                    yv_e32  = valid_mask(y_e32,  myd_nd[4-1])  # index 3

                    # --- LST_Day selection ---
                    s_m_day = score_lst(m_qd)
                    s_y_day = score_lst(y_qd)
                    # Score-compare only when both sides are valid
                    both = mv_day & yv_day
                    choose_mod = np.zeros(both.shape, dtype=bool)
                    if both.any():
                        choose_mod[both] = lexicographic_choose(
                            s_m_day[:, both], s_y_day[:, both], prefer_a=(prefer=="mod")
                        )

                    out_day = np.empty_like(m_day)
                    # Fill with NoData by default
                    if is_nan_nodata(out_nd[0]):
                        out_day[:] = np.nan
                    else:
                        out_day[:] = out_nd[0]

                    # One side valid
                    only_m = mv_day & ~yv_day
                    only_y = ~mv_day & yv_day
                    out_day[only_m] = m_day[only_m]
                    out_day[only_y] = y_day[only_y]
                    # Both sides valid: by score
                    aidx = both &  choose_mod
                    bidx = both & ~choose_mod
                    out_day[aidx] = m_day[aidx]
                    out_day[bidx] = y_day[bidx]

                    # Corresponding QC_Day: follows the source of the LST_Day winner
                    out_qd = np.empty_like(m_qd)
                    out_qd[only_m] = m_qd[only_m]
                    out_qd[only_y] = y_qd[only_y]
                    out_qd[aidx]   = m_qd[aidx]
                    out_qd[bidx]   = y_qd[bidx]

                    # --- LST_Night selection ---
                    s_m_nite = score_lst(m_qn)
                    s_y_nite = score_lst(y_qn)
                    both = mv_nite & yv_nite
                    choose_mod = np.zeros(both.shape, dtype=bool)
                    if both.any():
                        choose_mod[both] = lexicographic_choose(
                            s_m_nite[:, both], s_y_nite[:, both], prefer_a=(prefer=="mod")
                        )

                    out_nite = np.empty_like(m_nite)
                    if is_nan_nodata(out_nd[1]):
                        out_nite[:] = np.nan
                    else:
                        out_nite[:] = out_nd[1]

                    only_m = mv_nite & ~yv_nite
                    only_y = ~mv_nite & yv_nite
                    out_nite[only_m] = m_nite[only_m]
                    out_nite[only_y] = y_nite[only_y]
                    aidx = both &  choose_mod
                    bidx = both & ~choose_mod
                    out_nite[aidx] = m_nite[aidx]
                    out_nite[bidx] = y_nite[bidx]

                    # Corresponding QC_Night: follows the source of the LST_Night winner
                    out_qn = np.empty_like(m_qn)
                    out_qn[only_m] = m_qn[only_m]
                    out_qn[only_y] = y_qn[only_y]
                    out_qn[aidx]   = m_qn[aidx]
                    out_qn[bidx]   = y_qn[bidx]

                    # --- Emis selection (default: union) ---
                    s_m_emis = score_emis(m_qd, m_qn, mode=emis_qc_mode)
                    s_y_emis = score_emis(y_qd, y_qn, mode=emis_qc_mode)

                    # Emis_31
                    both = mv_e31 & yv_e31
                    choose_mod = np.zeros(both.shape, dtype=bool)
                    if both.any():
                        choose_mod[both] = lexicographic_choose(
                            s_m_emis[:, both], s_y_emis[:, both], prefer_a=(prefer=="mod")
                        )
                    out_e31 = np.empty_like(m_e31)
                    if is_nan_nodata(out_nd[2]):
                        out_e31[:] = np.nan
                    else:
                        out_e31[:] = out_nd[2]
                    only_m = mv_e31 & ~yv_e31
                    only_y = ~mv_e31 & yv_e31
                    out_e31[only_m] = m_e31[only_m]
                    out_e31[only_y] = y_e31[only_y]
                    aidx = both &  choose_mod
                    bidx = both & ~choose_mod
                    out_e31[aidx] = m_e31[aidx]
                    out_e31[bidx] = y_e31[bidx]

                    # Emis_32
                    both = mv_e32 & yv_e32
                    choose_mod = np.zeros(both.shape, dtype=bool)
                    if both.any():
                        choose_mod[both] = lexicographic_choose(
                            s_m_emis[:, both], s_y_emis[:, both], prefer_a=(prefer=="mod")
                        )
                    out_e32 = np.empty_like(m_e32)
                    if is_nan_nodata(out_nd[3]):
                        out_e32[:] = np.nan
                    else:
                        out_e32[:] = out_nd[3]
                    only_m = mv_e32 & ~yv_e32
                    only_y = ~mv_e32 & yv_e32
                    out_e32[only_m] = m_e32[only_m]
                    out_e32[only_y] = y_e32[only_y]
                    aidx = both &  choose_mod
                    bidx = both & ~choose_mod
                    out_e32[aidx] = m_e32[aidx]
                    out_e32[bidx] = y_e32[bidx]

                    # Write block
                    dst.write(out_day,  1, window=win)
                    dst.write(out_nite, 2, window=win)
                    dst.write(out_e31,  3, window=win)
                    dst.write(out_e32,  4, window=win)
                    dst.write(out_qd,   5, window=win)
                    dst.write(out_qn,   6, window=win)

            # Set per-band nodata (single value for GTiff)
            for b in range(1, 7):
                nd = out_nd[b-1]
                if not is_nan_nodata(nd):
                    dst.update_tags(b, nodata=str(int(nd)))
                else:
                    # For NaN, GTiff has no explicit nodata value; most raster software recognizes NaN as missing
                    pass
    return True, "merged"

def build_pairs(mod_dir: Path, myd_dir: Path):
    mod_map, myd_map = {}, {}
    for p in sorted(mod_dir.glob("*.tif")):
        k = parse_date_key(p)
        if k: mod_map[k] = p
    for p in sorted(myd_dir.glob("*.tif")):
        k = parse_date_key(p)
        if k: myd_map[k] = p
    keys = sorted(set(mod_map) | set(myd_map))
    pairs = [(k, mod_map.get(k), myd_map.get(k)) for k in keys]
    return pairs

def main():
    ap = argparse.ArgumentParser(description="MOD11A1 / MYD11A1 per-pixel merge (parallel + tiled + geographically aligned, keeping QC)")
    ap.add_argument("--mod", type=Path, 
                    default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MOD11A1_QAapplied', help="MOD11A1 directory (filenames include the date)")
    ap.add_argument("--myd", type=Path, 
                    default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MYD11A1_QAapplied', help="MYD11A1 directory (filenames include the date)")
    ap.add_argument("--out", type=Path, 
                    default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MCD11A1_QAapplied', help="Output directory")
    ap.add_argument("--prefer", choices=["mod", "myd"], default="mod", help="Tie-break when scores are exactly equal (default: mod)")
    ap.add_argument("--emis-qc-mode", choices=["union", "day", "night"], default="union",
                    help="Which QC to use for Emis scoring (default: union -- take the better of day/night)")
    ap.add_argument("--chunk", type=int, default=1024, help="Block size (pixels), default 1024")
    ap.add_argument("--workers", type=int, default=8, help="Number of parallel processes (default: 8)")
    ap.add_argument("--compress", default="LZW", help="Output compression (default: LZW)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    pairs = build_pairs(args.mod, args.myd)
    if not pairs:
        print("No date-matched filenames parsed from the input directories.", file=sys.stderr); sys.exit(1)

    task = partial(
        choose_and_write,
        out_dir=args.out, prefer=args.prefer,
        emis_qc_mode=args.emis_qc_mode,
        chunk=args.chunk, compress=args.compress
    )

    ok, fail = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(task, k, mp, yp): k for (k, mp, yp) in pairs}
        for fut in as_completed(futs):
            k = futs[fut]
            try:
                _, msg = fut.result()
                ok += 1
            except Exception as e:
                fail += 1
                print(f"[{k}] FAILED: {e}", file=sys.stderr)
    print(f"All done. OK={ok}, FAILED={fail}")

if __name__ == "__main__":
    main()
