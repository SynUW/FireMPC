#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2-in-1: MOD11A1 / MYD11A1 masking + MCD11A1 merge (keeps the original masking standard; merges by QC-score selection).

Input band order (6 bands):
  1: LST_Day_1km
  2: LST_Night_1km
  3: Emis_31
  4: Emis_32
  5: QC_Day
  6: QC_Night

Masking (original standard):
  Default strategy (conservative): "mask only especially-low quality; keep medium quality".
  - Mandatory QA (bits 0-1): {2, 3} -> mask
  - LST error (bits 6-7): > 3 K (value=3) -> mask; <= 3 K (0/1/2/3) -> keep (default max=3)
  - Emissivity error (bits 4-5): > 0.04 (value=3) -> mask; <= 0.04 (0/1/2/3) -> keep (default max=3)
  - Data quality (bits 2-3): 0 = good, 1 = other -> keep; 2/3 = TBD -> mask (can be relaxed via flags)

Merging:
  One side valid -> take that side; both invalid -> NoData; both valid -> pick by QC score:
  - LST_Day: score by QC_Day (mandatory, lst_error, data_quality, emis_error)
  - LST_Night: score by QC_Night (mandatory, lst_error, data_quality, emis_error)
  - Emis_31 / 32: score by emissivity error (mandatory, emis_error, data_quality)
  Ties broken by --prefer (default: mod).

NoData: uniformly -32765 across all bands.
"""

import re
import math
import shutil
import argparse
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np
import rasterio
from rasterio import windows
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds, Resampling
from rasterio.errors import NotGeoreferencedWarning
import warnings
warnings.simplefilter("ignore", NotGeoreferencedWarning)

# -------------------- General --------------------
DATE_RE = re.compile(r"(?P<y>\d{4})[-_]?((?P<m>\d{2})[-_]?)(?P<d>\d{2})")
TILE_RE = re.compile(r"_tile(?P<tile>\d+)", re.IGNORECASE)

def parse_date_tile_key(p: Path) -> str | None:
    """Parse filename and return key in 'YYYY_MM_DD_tileXX' format"""
    m_date = DATE_RE.search(p.stem)
    if not m_date: return None
    
    date_str = f"{m_date.group('y')}_{m_date.group('m')}_{m_date.group('d')}"
    
    # Try to extract tile number
    m_tile = TILE_RE.search(p.stem)
    if m_tile:
        tile_str = m_tile.group('tile')
        return f"{date_str}_tile{tile_str}"
    else:
        # If no tile info, return date (backward compatible)
        return date_str

def bitslice(arr, start, end):
    """Extract inclusive bit slice [start, end]."""
    width = end - start + 1
    mask = (1 << width) - 1
    return (arr >> start) & mask

def print_progress(prefix: str, ok: int, total: int, fail: int):
    print(f"\r{prefix}: OK={ok}/{total}, FAIL={fail}", end="", flush=True)

def finish_progress():
    print("", flush=True)

# -------------------- QC field parsing --------------------
def qc_fields(qc_arr):
    """
    Parse MOD11A1 QC_Day / QC_Night bit fields into a dict:
    0-1: mandatory
    2-3: data_quality
    4-5: emis_error
    6-7: lst_error
    """
    return dict(
        mandatory = bitslice(qc_arr, 0, 1),
        data_quality = bitslice(qc_arr, 2, 3),
        emis_error = bitslice(qc_arr, 4, 5),
        lst_error = bitslice(qc_arr, 6, 7),
    )

# -------------------- Mask criteria (keeping original standards) --------------------
def build_bad_mask_from_qc(
    qc,
    mask_if_mandatory_23=True,
    mask_if_dq_tbd=True,
    dq_keep_1=True,
    max_keep_lst_error=3,     # Keep 0..3; >3 does not exist, so effectively keep all
    max_keep_emis_error=3,    # Keep 0..3; >3 does not exist, so effectively keep all
):
    """
    Returns a boolean bad_mask: True means "quality is especially low and should be masked".
    """
    q = qc_fields(qc)

    bad = np.zeros(qc.shape, dtype=bool)

    # 1) mandatory QA: 2/3 -> always mask (not produced)
    if mask_if_mandatory_23:
        bad |= np.isin(q["mandatory"], [2, 3])

    # 2) data quality: 0=good, 1=other quality (keep), 2/3=TBD -> mask by default
    if mask_if_dq_tbd:
        bad |= np.isin(q["data_quality"], [2, 3])
    if not dq_keep_1:
        # For stricter filtering, you can also mask data_quality==1
        bad |= (q["data_quality"] == 1)

    # 3) LST error: > max_keep_lst_error -> mask; <= max_keep_lst_error -> keep
    bad |= (q["lst_error"] > max_keep_lst_error)

    # 4) Emissivity error: > max_keep_emis_error -> mask; <= max_keep_emis_error -> keep
    bad |= (q["emis_error"] > max_keep_emis_error)

    return bad

# -------------------- Scoring system (used for merging) --------------------
def score_lst(qc_arr):
    """LST score: (mandatory, lst_error, data_quality, emis_error)"""
    q = qc_fields(qc_arr)
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
    choose_a = np.zeros(a_score.shape[1:], dtype=bool)
    undecided = np.ones_like(choose_a, dtype=bool)
    for k in range(a_score.shape[0]):
        ak = a_score[k]; bk = b_score[k]
        better_a = ak < bk
        better_b = ak > bk
        choose_a |= (better_a & undecided)
        undecided &= ~(better_a | better_b)
        if not undecided.any():
            break
    if undecided.any():
        if prefer_a:
            choose_a |= undecided
    return choose_a

# -------------------- Geometric alignment --------------------
def pick_target_grid(mod_src, myd_src):
    """Select the target grid: prefer mod's CRS; take the finer resolution; take the UNION extent."""
    crs = mod_src.crs or myd_src.crs
    if crs is None:
        raise ValueError("Both inputs lack a valid CRS.")

    def res_of(src):
        a = src.transform
        return (abs(a.a), abs(a.e))
    mod_res = res_of(mod_src)
    myd_res = res_of(myd_src)
    xres = min(mod_res[0], myd_res[0])
    yres = min(mod_res[1], myd_res[1])

    mod_bounds = transform_bounds(mod_src.crs, crs, *mod_src.bounds, densify_pts=0) if mod_src.crs != crs else mod_src.bounds
    myd_bounds = transform_bounds(myd_src.crs, crs, *myd_src.bounds, densify_pts=0) if myd_src.crs != crs else myd_src.bounds
    xmin = min(mod_bounds[0], myd_bounds[0])
    ymin = min(mod_bounds[1], myd_bounds[1])
    xmax = max(mod_bounds[2], myd_bounds[2])
    ymax = max(mod_bounds[3], myd_bounds[3])

    from affine import Affine
    dst_transform = Affine(xres, 0.0, xmin, 0.0, -yres, ymax)
    width  = int(math.ceil((xmax - xmin) / xres))
    height = int(math.ceil((ymax - ymin) / yres))

    return crs, dst_transform, width, height

def ensure_valid_blocksize(b, max_size=1024, min_size=16):
    b = max(min(b, max_size), min_size)
    return (b // 16) * 16 or 16

# -------------------- Mask (single file) --------------------
def process_single_file(
    path_in: Path,
    path_out: Path,
    out_nodata: float,
    overwrite: bool,
    allow_mandatory_23: bool,
    allow_dq_tbd: bool,
    mask_dq1: bool,
    max_keep_lst_error: int,
    max_keep_emis_error: int,
    emis_qc_mode_mask: str,  # emis_qc_mode used in the masking stage
    compress: str,
):
    """Function to process a single file (used for parallel processing)"""
    if path_out.exists() and not overwrite:
        return True, path_in.name, None

    try:
        with rasterio.open(path_in) as src:
            if src.count < 6:
                return False, path_in.name, f"6 bands required (LST_Day, LST_Night, Emis_31, Emis_32, QC_Day, QC_Night)."

            # Read band
            lst_day  = src.read(1)
            lst_nite = src.read(2)
            emis31   = src.read(3)
            emis32   = src.read(4)
            qc_day   = src.read(5)
            qc_nite  = src.read(6)

            # Build the "especially-low-quality" mask from QC
            bad_day  = build_bad_mask_from_qc(
                qc_day,
                mask_if_mandatory_23=not allow_mandatory_23,
                mask_if_dq_tbd=not allow_dq_tbd,
                dq_keep_1=not mask_dq1,
                max_keep_lst_error=max_keep_lst_error,
                max_keep_emis_error=max_keep_emis_error,
            )
            bad_nite = build_bad_mask_from_qc(
                qc_nite,
                mask_if_mandatory_23=not allow_mandatory_23,
                mask_if_dq_tbd=not allow_dq_tbd,
                dq_keep_1=not mask_dq1,
                max_keep_lst_error=max_keep_lst_error,
                max_keep_emis_error=max_keep_emis_error,
            )

            # 3/4) Emis_31 / Emis_32: based on emis_qc_mode_mask
            if emis_qc_mode_mask == "day":
                bad_emis = bad_day
            elif emis_qc_mode_mask == "night":
                bad_emis = bad_nite
            else:  # "union" (default): mask if either day or night is severely bad
                bad_emis = (bad_day | bad_nite)

            profile = src.profile.copy()
            bx = ensure_valid_blocksize(profile.get("blockxsize", 512))
            by = ensure_valid_blocksize(profile.get("blockysize", 512))
            profile.update(
                compress=compress,
                tiled=True,
                blockxsize=bx, blockysize=by,
                bigtiff="IF_SAFER",
            )

            # Use out_nodata uniformly
            path_out.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(path_out, "w", **profile) as dst:
                # 1) LST_Day: apply NoData based on bad_day
                arr = lst_day.astype(np.float32, copy=False)
                arr[bad_day] = out_nodata
                dst.write(arr, 1)
                dst.update_tags(1, nodata=str(int(out_nodata)))

                # 2) LST_Night: apply NoData based on bad_nite
                arr = lst_nite.astype(np.float32, copy=False)
                arr[bad_nite] = out_nodata
                dst.write(arr, 2)
                dst.update_tags(2, nodata=str(int(out_nodata)))

                # 3) Emis_31
                arr = emis31.astype(np.float32, copy=False)
                arr[bad_emis] = out_nodata
                dst.write(arr, 3)
                dst.update_tags(3, nodata=str(int(out_nodata)))

                # 4) Emis_32
                arr = emis32.astype(np.float32, copy=False)
                arr[bad_emis] = out_nodata
                dst.write(arr, 4)
                dst.update_tags(4, nodata=str(int(out_nodata)))

                # 5/6) Write QA back as-is
                dst.write(qc_day.astype(np.float32), 5)
                dst.write(qc_nite.astype(np.float32), 6)

        return True, path_in.name, None

    except Exception as e:
        return False, path_in.name, str(e)

# -------------------- Merge (single date+tile) --------------------
def combine_one(
    date_tile_key: str,
    mod_path: Path | None,
    myd_path: Path | None,
    out_dir: Path,
    prefer_mod: bool,
    emis_qc_mode_combine: str,  # emis_qc_mode used in the merging stage
    chunk: int,
    compress: str,
    out_nodata: float,
    overwrite: bool,
):
    """Merge MOD and MYD files for a single date+tile"""
    # Output filename: MCD11A1_YYYY_MM_DD_tileXX.tif
    out_filename = f"MCD11A1_{date_tile_key}.tif"
    out_path = out_dir / out_filename

    if out_path.exists() and not overwrite:
        return True, "skipped"

    # Only one side exists: copy directly
    if (mod_path is None) ^ (myd_path is None):
        shutil.copy2(mod_path or myd_path, out_path)
        return True, "copied"

    # Both sides exist: merge
    with rasterio.open(mod_path) as mod, rasterio.open(myd_path) as myd:
        if mod.count < 6 or myd.count < 6:
            raise ValueError("Input must have 6 bands (LST_Day, LST_Night, Emis_31, Emis_32, QC_Day, QC_Night).")

        crs, dst_transform, width, height = pick_target_grid(mod, myd)

        # VRT: bilinear for LST/Emis, nearest neighbor for QC
        vrt_opts_float = dict(crs=crs, transform=dst_transform, width=width, height=height,
                              resampling=Resampling.bilinear, add_alpha=False)
        vrt_opts_near  = dict(crs=crs, transform=dst_transform, width=width, height=height,
                              resampling=Resampling.nearest, add_alpha=False)

        # Unified output nodata
        out_nd = out_nodata

        bx = ensure_valid_blocksize(min(chunk, 1024))
        by = ensure_valid_blocksize(min(chunk, 1024))
        profile = dict(
            driver="GTiff", count=6, crs=crs, transform=dst_transform,
            width=width, height=height, dtype='float32',
            tiled=True, blockxsize=bx, blockysize=by,
            compress=compress, bigtiff="IF_SAFER",
        )

        with rasterio.open(out_path, "w", **profile) as dst, \
             WarpedVRT(mod, **vrt_opts_float) as mod_vrt_f, WarpedVRT(myd, **vrt_opts_float) as myd_vrt_f, \
             WarpedVRT(mod, **vrt_opts_near)  as mod_vrt_n, WarpedVRT(myd, **vrt_opts_near)  as myd_vrt_n:

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

                    # Valid mask (based on the unified nodata)
                    mv_day  = (m_day != out_nd)
                    mv_nite = (m_nite != out_nd)
                    mv_e31  = (m_e31 != out_nd)
                    mv_e32  = (m_e32 != out_nd)

                    yv_day  = (y_day != out_nd)
                    yv_nite = (y_nite != out_nd)
                    yv_e31  = (y_e31 != out_nd)
                    yv_e32  = (y_e32 != out_nd)

                    # --- LST_Day selection ---
                    s_m_day = score_lst(m_qd)
                    s_y_day = score_lst(y_qd)
                    both = mv_day & yv_day
                    choose_mod = np.zeros(both.shape, dtype=bool)
                    if both.any():
                        choose_mod[both] = lexicographic_choose(
                            s_m_day[:, both], s_y_day[:, both], prefer_a=prefer_mod
                        )

                    out_day = np.full_like(m_day, out_nd)
                    only_m = mv_day & ~yv_day
                    only_y = ~mv_day & yv_day
                    out_day[only_m] = m_day[only_m]
                    out_day[only_y] = y_day[only_y]
                    aidx = both & choose_mod
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
                            s_m_nite[:, both], s_y_nite[:, both], prefer_a=prefer_mod
                        )

                    out_nite = np.full_like(m_nite, out_nd)
                    only_m = mv_nite & ~yv_nite
                    only_y = ~mv_nite & yv_nite
                    out_nite[only_m] = m_nite[only_m]
                    out_nite[only_y] = y_nite[only_y]
                    aidx = both & choose_mod
                    bidx = both & ~choose_mod
                    out_nite[aidx] = m_nite[aidx]
                    out_nite[bidx] = y_nite[bidx]

                    # Corresponding QC_Night: follows the source of the LST_Night winner
                    out_qn = np.empty_like(m_qn)
                    out_qn[only_m] = m_qn[only_m]
                    out_qn[only_y] = y_qn[only_y]
                    out_qn[aidx]   = m_qn[aidx]
                    out_qn[bidx]   = y_qn[bidx]

                    # --- Emis selection (using emis_qc_mode_combine) ---
                    s_m_emis = score_emis(m_qd, m_qn, mode=emis_qc_mode_combine)
                    s_y_emis = score_emis(y_qd, y_qn, mode=emis_qc_mode_combine)

                    # Emis_31
                    both = mv_e31 & yv_e31
                    choose_mod = np.zeros(both.shape, dtype=bool)
                    if both.any():
                        choose_mod[both] = lexicographic_choose(
                            s_m_emis[:, both], s_y_emis[:, both], prefer_a=prefer_mod
                        )
                    out_e31 = np.full_like(m_e31, out_nd)
                    only_m = mv_e31 & ~yv_e31
                    only_y = ~mv_e31 & yv_e31
                    out_e31[only_m] = m_e31[only_m]
                    out_e31[only_y] = y_e31[only_y]
                    aidx = both & choose_mod
                    bidx = both & ~choose_mod
                    out_e31[aidx] = m_e31[aidx]
                    out_e31[bidx] = y_e31[bidx]

                    # Emis_32
                    both = mv_e32 & yv_e32
                    choose_mod = np.zeros(both.shape, dtype=bool)
                    if both.any():
                        choose_mod[both] = lexicographic_choose(
                            s_m_emis[:, both], s_y_emis[:, both], prefer_a=prefer_mod
                        )
                    out_e32 = np.full_like(m_e32, out_nd)
                    only_m = mv_e32 & ~yv_e32
                    only_y = ~mv_e32 & yv_e32
                    out_e32[only_m] = m_e32[only_m]
                    out_e32[only_y] = y_e32[only_y]
                    aidx = both & choose_mod
                    bidx = both & ~choose_mod
                    out_e32[aidx] = m_e32[aidx]
                    out_e32[bidx] = y_e32[bidx]

                    # Write block
                    dst.write(out_day,  1, window=win)
                    dst.write(out_nite, 2, window=win)
                    dst.write(out_e31,  3, window=win)
                    dst.write(out_e32,  4, window=win)
                    dst.write(out_qd.astype(np.float32),   5, window=win)
                    dst.write(out_qn.astype(np.float32),   6, window=win)

            # Set per-band nodata
            for b in range(1, 7):
                dst.update_tags(b, nodata=str(int(out_nd)))

    return True, "merged"

# -------------------- Directory batch processing --------------------
def list_tifs(d: Path):
    return sorted([p for p in d.iterdir() if p.suffix.lower() in (".tif", ".tiff")])

def build_pairs(mod_dir: Path, myd_dir: Path):
    """Build pairs: match MOD and MYD files by date and tile"""
    mod_map, myd_map = {}, {}
    for p in list_tifs(mod_dir):
        k = parse_date_tile_key(p)
        if k: mod_map[k] = p
    for p in list_tifs(myd_dir):
        k = parse_date_tile_key(p)
        if k: myd_map[k] = p
    keys = sorted(set(mod_map) | set(myd_map))
    return [(k, mod_map.get(k), myd_map.get(k)) for k in keys]

def mask_dir(in_dir: Path, out_dir: Path, out_nodata: float, args):
    """Mask an entire directory"""
    out_dir.mkdir(parents=True, exist_ok=True)
    tifs = list_tifs(in_dir)
    total = len(tifs)
    prefix = f"[MASK] {in_dir.name} → {out_dir.name}"
    print_progress(prefix, 0, total, 0)

    tasks = [
        (
            p,
            out_dir / p.name,  # Keep the original filename
            out_nodata,
            args.overwrite,
            args.allow_mandatory_23,
            args.allow_dq_tbd,
            args.mask_dq1,
            args.max_keep_lst_error,
            args.max_keep_emis_error,
            args.emis_qc_mode_mask,
            args.compress,
        )
        for p in tifs
    ]

    ok, fail = 0, 0
    num_processes = max(1, cpu_count() // 4)
    with ProcessPoolExecutor(max_workers=num_processes) as ex:
        futs = {ex.submit(process_single_file, *task): task for task in tasks}
        for fut in as_completed(futs):
            try:
                success, fn, error_msg = fut.result()
                if success:
                    ok += 1
                else:
                    fail += 1
                    if error_msg:
                        print(f"\nError {fn}: {error_msg}")
            except Exception as e:
                fail += 1
                task = futs[fut]
                print(f"\nException {task[0].name}: {e}")
            finally:
                print_progress(prefix, ok, total, fail)
    finish_progress()

def combine_dirs(mod_masked: Path, myd_masked: Path, out_dir: Path, out_nodata: float, args):
    """Merge two directories"""
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = build_pairs(mod_masked, myd_masked)
    total = len(pairs)
    prefix = f"[COMBINE] {mod_masked.name}+{myd_masked.name} → {out_dir.name}"
    print_progress(prefix, 0, total, 0)

    task = partial(
        combine_one,
        out_dir=out_dir,
        prefer_mod=(args.prefer=="mod"),
        emis_qc_mode_combine=args.emis_qc_mode_combine,
        chunk=args.chunk,
        compress=args.compress,
        out_nodata=out_nodata,
        overwrite=args.overwrite
    )
    ok, fail = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(task, k, mp, yp): k for (k, mp, yp) in pairs}
        for fut in as_completed(futs):
            try:
                fut.result(); ok += 1
            except Exception as e:
                fail += 1
                print(f"\nException: {e}")
            finally:
                print_progress(prefix, ok, total, fail)
    finish_progress()

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="MOD11A1 / MYD11A1 masking + MCD11A1 merge (keeps the original masking standard; merges by QC-score selection)")
    ap.add_argument("--mod_in",  type=Path, default='/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MOD11A1', help="MOD11A1 input directory")
    ap.add_argument("--myd_in",  type=Path, default='/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MYD11A1', help="MYD11A1 input directory")
    ap.add_argument("--mod_out", type=Path, default='/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MOD11A1_QAapplied', help="MOD11A1 masked-output directory")
    ap.add_argument("--myd_out", type=Path, default='/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MYD11A1_QAapplied', help="MYD11A1 masked-output directory")
    ap.add_argument("--mcd_out", type=Path, default='/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/MCD11A1_QAapplied', help="MCD11A1 merge-output directory")

    # Merge / performance
    ap.add_argument("--prefer", choices=["mod","myd"], default="mod", help="Tie-break priority when scores are equal (default: mod)")
    ap.add_argument("--chunk", type=int, default=1024, help="Block size in pixels (default: 1024)")
    ap.add_argument("--workers", type=int, default=8, help="Number of parallel processes (default: 8)")
    ap.add_argument("--compress", default="LZW", help="Output compression (default: LZW)")

    # NoData
    ap.add_argument("--out-nodata", type=float, default=-32765, help="Unified NoData value (default: -32765)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output")

    # ---------- Mask parameters (keep original standards) ----------
    ap.add_argument("--allow-mandatory-23", action="store_true",
                    help="Loosen: do not mask due to mandatory=2/3 (not produced) (generally not recommended)")
    ap.add_argument("--allow-dq-tbd", action="store_true",
                    help="Loosen: do not mask DataQuality=2/3 (TBD)")
    ap.add_argument("--mask-dq1", action="store_true",
                    help="Tighten: also mask DataQuality=1 (Other quality) (not masked by default to retain medium quality)")
    ap.add_argument("--max-keep-lst-error", type=int, default=3, choices=[0,1,2,3],
                    help="Maximum LST error level allowed to be kept (0..3); default 3 (keep all)")
    ap.add_argument("--max-keep-emis-error", type=int, default=3, choices=[0,1,2,3],
                    help="Maximum Emissivity error level allowed to be kept (0..3); default 3 (keep all)")

    ap.add_argument("--emis-qc-mode-mask", choices=["union", "day", "night"], default="union",
                    help="Masking stage: which QC to use for the emissivity bands (default: union)")
    ap.add_argument("--emis-qc-mode-combine", choices=["union", "day", "night"], default="union",
                    help="Merging stage: which QC to use for the Emis score (default: union)")

    args = ap.parse_args()

    # Mask
    mask_dir(args.mod_in, args.mod_out, args.out_nodata, args)
    mask_dir(args.myd_in, args.myd_out, args.out_nodata, args)

    # Merge
    combine_dirs(args.mod_out, args.myd_out, args.mcd_out, args.out_nodata, args)

if __name__ == "__main__":
    main()
