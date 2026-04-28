#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2-in-1: MOD09CMG / MYD09CMG masking + MCD09CMG merge (very lenient QA, progress bar, parallel + tiled + geographically aligned).

Input band order (7):
 1: Coarse_Resolution_Brightness_Temperature_Band_20
 2: Coarse_Resolution_Brightness_Temperature_Band_21
 3: Coarse_Resolution_Brightness_Temperature_Band_31
 4: Coarse_Resolution_Brightness_Temperature_Band_32
 5: Coarse_Resolution_QA            (uint32, up to bit 31)
 6: Coarse_Resolution_State_QA      (uint16, up to bit 15)
 7: Coarse_Resolution_Internal_CM   (uint16, up to bit 15)

Masking (very lenient):
 A pixel is treated as invalid (masked) only when all four conditions are met:
   A) MODLAND in {2, 3}
   B) State.cloud_state == 1 (Cloudy)
   C) State.internal_cloud == 1
   D) ICM.cloudy == 1
Everything else is kept (including Mixed=2, Assumed clear=3, internal-cloud-only, etc.).

Merging:
 One side valid -> take that side; both invalid -> NoData; both valid -> pick by mild scoring:
   cloud_state: clear(0) < assumed(3) < mixed(2) < cloudy(1)
   ICM.cloudy: 0 < 1
   internal_cloud: 0 < 1
   MODLAND: 0 < 1 < 2 < 3
 Ties broken by --prefer (default: mod).

NoData: data bands are uniformly written as -32768 (override with --out-nodata); QA bands are written back as-is with no nodata set.
"""

import re
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
    width = end - start + 1
    return (arr >> start) & ((1 << width) - 1)

def get_bit(arr, bit):
    return (arr >> bit) & 1

def ensure_valid_blocksize(b, max_size=1024, min_size=16):
    b = max(min(b, max_size), min_size)
    return (b // 16) * 16 or 16

def print_progress(prefix: str, ok: int, total: int, fail: int):
    print(f"\r{prefix}: OK={ok}/{total}, FAIL={fail}", end="", flush=True)

def finish_progress():
    print("", flush=True)

# -------------------- QA parsing --------------------
def qa_cr_fields(qa_u32: np.ndarray):
    # Coarse_Resolution_QA: up to bit 31 (uint32)
    return dict(
        modland = bitslice(qa_u32, 0, 1),  # 0 ideal,1 less-than-ideal,2 cloud,3 other fail
        # Other bits are not used for masking, only for mild scoring
    )

def qa_state_fields(qa_u16: np.ndarray):
    # Coarse_Resolution_State_QA: up to bit 15 (uint16)
    return dict(
        cloud_state    = bitslice(qa_u16, 0, 1),  # 0 clear, 1 cloudy, 2 mixed, 3 assumed clear
        internal_cloud = get_bit(qa_u16, 10),
    )

def qa_icm_fields(qa_u16: np.ndarray):
    # Coarse_Resolution_Internal_CM: up to bit 15 (uint16)
    return dict(
        cloudy = get_bit(qa_u16, 0),
    )

# -------------------- Masking (very lenient: mask only when all four are true) --------------------
def build_invalid_cmg_ultra_relaxed(qa_cr_u32, qa_state_u16, qa_icm_u16):
    q1 = qa_cr_fields(qa_cr_u32)
    q2 = qa_state_fields(qa_state_u16)
    q3 = qa_icm_fields(qa_icm_u16)

    bad_modland = np.isin(q1["modland"], [2, 3])
    return bad_modland & (q2["cloud_state"] == 1) & (q2["internal_cloud"] == 1) & (q3["cloudy"] == 1)

# -------------------- Scoring (mild) --------------------
def cmg_quality_score_soft(qa_cr_u32, qa_state_u16, qa_icm_u16):
    q1 = qa_cr_fields(qa_cr_u32)
    q2 = qa_state_fields(qa_state_u16)
    q3 = qa_icm_fields(qa_icm_u16)

    cs = q2["cloud_state"].astype(np.uint8)
    cs_map = np.zeros_like(cs, dtype=np.uint8)
    cs_map[cs == 0] = 0          # clear
    cs_map[cs == 3] = 1          # assumed clear
    cs_map[cs == 2] = 2          # mixed
    cs_map[cs == 1] = 3          # cloudy (worst)

    return np.stack([
        cs_map,                                   # Most important
        q3["cloudy"].astype(np.uint8),            # ICM cloud
        q2["internal_cloud"].astype(np.uint8),    # Internal cloud
        qa_cr_fields(qa_cr_u32)["modland"].astype(np.uint8),  # MODLAND 0<1<2<3
    ], axis=0)

def lexicographic_choose(a_score, b_score, prefer_a=True):
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
    if undecided.any() and prefer_a:
        choose_a |= undecided
    return choose_a

# -------------------- Geometric alignment (UNION + finer res) --------------------
def pick_target_grid(src_a, src_b):
    crs = src_a.crs or src_b.crs
    if crs is None:
        raise ValueError("Missing a valid CRS")
    def res_of(src):
        a = src.transform
        return (abs(a.a), abs(a.e))
    ra = res_of(src_a); rb = res_of(src_b)
    xres = min(ra[0], rb[0]); yres = min(ra[1], rb[1])

    a_bounds = transform_bounds(src_a.crs, crs, *src_a.bounds, densify_pts=0) if src_a.crs != crs else src_a.bounds
    b_bounds = transform_bounds(src_b.crs, crs, *src_b.bounds, densify_pts=0) if src_b.crs != crs else src_b.bounds
    xmin = min(a_bounds[0], b_bounds[0]); ymin = min(a_bounds[1], b_bounds[1])
    xmax = max(a_bounds[2], b_bounds[2]); ymax = max(a_bounds[3], b_bounds[3])

    from affine import Affine
    width  = int(math.ceil((xmax - xmin) / xres))
    height = int(math.ceil((ymax - ymin) / yres))
    transform = Affine(xres, 0, xmin, 0, -yres, ymax)
    return crs, transform, width, height

# -------------------- Mask (single file) --------------------
def mask_one(src_path: Path, dst_path: Path, out_nodata: float, args):
    if dst_path.exists() and not args.overwrite:
        return True  # Skip
    with rasterio.open(src_path) as src:
        if src.count < 7:
            raise ValueError(f"{src.name}: 7 bands required (4 BT + 3 QA).")

        bt20 = src.read(1); bt21 = src.read(2); bt31 = src.read(3); bt32 = src.read(4)
        qa_cr = src.read(5).astype(np.uint32, copy=False)   # 32-bit
        qa_st = src.read(6).astype(np.uint16, copy=False)   # 16-bit
        qa_ic = src.read(7).astype(np.uint16, copy=False)   # 16-bit

        invalid = build_invalid_cmg_ultra_relaxed(qa_cr, qa_st, qa_ic)

        profile = src.profile.copy()
        bx = ensure_valid_blocksize(profile.get("blockxsize", 512))
        by = ensure_valid_blocksize(profile.get("blockysize", 512))
        profile.update(
            compress=args.compress,
            tiled=True,
            blockxsize=bx, blockysize=by,
            bigtiff="IF_SAFER",
            nodata=float(out_nodata),  # Set nodata at creation time
        )

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            # Data bands: unified out_nodata, with nodata tag set
            for i, arr in enumerate([bt20, bt21, bt31, bt32], start=1):
                out = arr.copy()
                out[invalid] = out_nodata
                dst.write(out, i)
            # Write QA bands back as-is
            dst.write(qa_cr, 5)
            dst.write(qa_st, 6)
            dst.write(qa_ic, 7)
    return True

# -------------------- Merge (single date+tile) --------------------
def combine_one(date_tile_key, mod_path, myd_path, out_dir, prefer_mod: bool, chunk, compress, out_nodata: float, overwrite: bool):
    """Merge MOD and MYD files for a single date+tile"""
    # date_tile_key format: 'YYYY_MM_DD_tileXX' or 'YYYY_MM_DD'
    out_path = out_dir / f"{date_tile_key}.tif"
    if out_path.exists() and not overwrite:
        return True, "skipped"

    # Only one side exists: copy directly
    if (mod_path is None) ^ (myd_path is None):
        shutil.copy2(mod_path or myd_path, out_path)
        return True, "copied"

    with rasterio.open(mod_path) as mod, rasterio.open(myd_path) as myd:
        if mod.count < 7 or myd.count < 7:
            raise ValueError("7 bands required (4 BT + 3 QA).")

        crs, transform, width, height = pick_target_grid(mod, myd)

        # VRT: bilinear for numeric bands, nearest neighbor for QA
        vrt_f_opts = dict(crs=crs, transform=transform, width=width, height=height, resampling=Resampling.bilinear)
        vrt_n_opts = dict(crs=crs, transform=transform, width=width, height=height, resampling=Resampling.nearest)

        # Use float32 uniformly to ensure out_nodata (e.g. -32765) can be represented
        out_dtype = 'float32'
        bx = ensure_valid_blocksize(min(chunk, 1024))
        by = ensure_valid_blocksize(min(chunk, 1024))
        profile = dict(
            driver="GTiff", count=7, crs=crs, transform=transform,
            width=width, height=height, dtype=out_dtype,
            tiled=True, blockxsize=bx, blockysize=by,
            compress=compress, bigtiff="IF_SAFER",
            nodata=float(out_nodata)
        )

        with rasterio.open(out_path, "w", **profile) as dst, \
             WarpedVRT(mod, **vrt_f_opts) as m_vrt_f, WarpedVRT(myd, **vrt_f_opts) as y_vrt_f, \
             WarpedVRT(mod, **vrt_n_opts) as m_vrt_n, WarpedVRT(myd, **vrt_n_opts) as y_vrt_n:

            for row_off in range(0, height, chunk):
                h = min(chunk, height - row_off)
                for col_off in range(0, width, chunk):
                    w = min(chunk, width - col_off)
                    win = windows.Window(col_off, row_off, w, h)

                    # Data band
                    m20 = m_vrt_f.read(1, window=win); m21 = m_vrt_f.read(2, window=win)
                    m31 = m_vrt_f.read(3, window=win); m32 = m_vrt_f.read(4, window=win)
                    y20 = y_vrt_f.read(1, window=win); y21 = y_vrt_f.read(2, window=win)
                    y31 = y_vrt_f.read(3, window=win); y32 = y_vrt_f.read(4, window=win)

                    # QA (nearest neighbor)
                    m_q1 = m_vrt_n.read(5, window=win).astype(np.uint32, copy=False)
                    m_q2 = m_vrt_n.read(6, window=win).astype(np.uint16, copy=False)
                    m_q3 = m_vrt_n.read(7, window=win).astype(np.uint16, copy=False)
                    y_q1 = y_vrt_n.read(5, window=win).astype(np.uint32, copy=False)
                    y_q2 = y_vrt_n.read(6, window=win).astype(np.uint16, copy=False)
                    y_q3 = y_vrt_n.read(7, window=win).astype(np.uint16, copy=False)

                    # Validity (masking stage has already written -32768)
                    mv20 = (m20 != out_nodata); mv21 = (m21 != out_nodata)
                    mv31 = (m31 != out_nodata); mv32 = (m32 != out_nodata)
                    yv20 = (y20 != out_nodata); yv21 = (y21 != out_nodata)
                    yv31 = (y31 != out_nodata); yv32 = (y32 != out_nodata)

                    # Scoring (mild)
                    s_m = cmg_quality_score_soft(m_q1, m_q2, m_q3)
                    s_y = cmg_quality_score_soft(y_q1, y_q2, y_q3)
                    choose_mod_global = lexicographic_choose(s_m, s_y, prefer_a=prefer_mod)

                    def choose_value(m_arr, y_arr, mv, yv):
                        both = mv & yv
                        out = np.empty_like(m_arr); out[:] = out_nodata
                        only_m = mv & ~yv; only_y = ~mv & yv
                        out[only_m] = m_arr[only_m]
                        out[only_y] = y_arr[only_y]
                        aidx = both &  choose_mod_global
                        bidx = both & ~choose_mod_global
                        out[aidx] = m_arr[aidx]
                        out[bidx] = y_arr[bidx]
                        return out, aidx, bidx

                    out20, a20, b20 = choose_value(m20, y20, mv20, yv20)
                    out21, a21, b21 = choose_value(m21, y21, mv21, yv21)
                    out31, a31, b31 = choose_value(m31, y31, mv31, yv31)
                    out32, a32, b32 = choose_value(m32, y32, mv32, yv32)

                    # QA source: majority vote (ties broken by prefer)
                    m_votes = (a20 + a21 + a31 + a32).astype(np.int16)
                    y_votes = (b20 + b21 + b31 + b32).astype(np.int16)
                    choose_mod_qa = (m_votes > y_votes) | ((m_votes == y_votes) & prefer_mod)

                    out_q1 = np.where(choose_mod_qa, m_q1, y_q1).astype(np.uint32, copy=False)
                    out_q2 = np.where(choose_mod_qa, m_q2, y_q2).astype(np.uint16, copy=False)
                    out_q3 = np.where(choose_mod_qa, m_q3, y_q3).astype(np.uint16, copy=False)

                    # Write block
                    dst.write(out20, 1, window=win)
                    dst.write(out21, 2, window=win)
                    dst.write(out31, 3, window=win)
                    dst.write(out32, 4, window=win)
                    dst.write(out_q1, 5, window=win)
                    dst.write(out_q2, 6, window=win)
                    dst.write(out_q3, 7, window=win)

            # rasterio already sets nodata in the profile; no need to update per-band tags

    return True, "merged"

# -------------------- Directory batch processing + numeric progress bar --------------------
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
    out_dir.mkdir(parents=True, exist_ok=True)
    tifs = list_tifs(in_dir)
    total = len(tifs)
    prefix = f"[MASK] {in_dir.name} \u2192 {out_dir.name}"
    print_progress(prefix, 0, total, 0)

    task = partial(mask_one, out_nodata=out_nodata, args=args)
    ok, fail = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(task, p, out_dir / p.name): p.name for p in tifs}
        for fut in as_completed(futs):
            try:
                fut.result(); ok += 1
            except Exception:
                fail += 1
            finally:
                print_progress(prefix, ok, total, fail)
    finish_progress()

def combine_dirs(mod_masked: Path, myd_masked: Path, out_dir: Path, out_nodata: float, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = build_pairs(mod_masked, myd_masked)
    total = len(pairs)
    prefix = f"[COMBINE] {mod_masked.name}+{myd_masked.name} \u2192 {out_dir.name}"
    print_progress(prefix, 0, total, 0)

    task = partial(
        combine_one, out_dir=out_dir,
        prefer_mod=(args.prefer=="mod"),
        chunk=args.chunk, compress=args.compress,
        out_nodata=out_nodata, overwrite=args.overwrite
    )
    ok, fail = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(task, k, mp, yp): k for (k, mp, yp) in pairs}
        for fut in as_completed(futs):
            try:
                fut.result(); ok += 1
            except Exception:
                fail += 1
            finally:
                print_progress(prefix, ok, total, fail)
    finish_progress()

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="MOD09CMG / MYD09CMG masking + MCD09CMG merge (very lenient: only mask when MODLAND, State, and ICM all flag cloud)")
    ap.add_argument("--mod_in",  type=Path, default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MOD09CMG", help="MOD09CMG input directory (7 bands)")
    ap.add_argument("--myd_in",  type=Path, default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MYD09CMG", help="MYD09CMG input directory (7 bands)")
    ap.add_argument("--mod_out", type=Path, default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MOD09CMG_QAapplied", help="MOD09CMG masked-output directory")
    ap.add_argument("--myd_out", type=Path, default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MYD09CMG_QAapplied", help="MYD09CMG masked-output directory")
    ap.add_argument("--mcd_out", type=Path, default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MCD09CMG_QAapplied", help="MCD09CMG merge-output directory")

    # Merge / performance
    ap.add_argument("--prefer", choices=["mod","myd"], default="mod", help="Tie-break priority when scores are equal (default: mod)")
    ap.add_argument("--chunk", type=int, default=1024, help="Block size in pixels (default: 1024)")
    ap.add_argument("--workers", type=int, default=8, help="Number of parallel processes (default: 8)")
    ap.add_argument("--compress", default="LZW", help="Output compression (default: LZW)")

    # NoData & overwrite
    ap.add_argument("--out-nodata", type=float, default=-32765, help="Unified NoData value for data bands (default: -32765)")
    ap.add_argument("--overwirte", dest="overwrite", action="store_true", help="Overwrite existing outputs (process from scratch)")
    ap.add_argument("--overwrite", dest="overwrite", action="store_true", help="Overwrite existing outputs (process from scratch)")

    args = ap.parse_args()

    # Mask
    mask_dir(args.mod_in, args.mod_out, args.out_nodata, args)
    mask_dir(args.myd_in, args.myd_out, args.out_nodata, args)

    # Merge
    combine_dirs(args.mod_out, args.myd_out, args.mcd_out, args.out_nodata, args)

if __name__ == "__main__":
    main()
