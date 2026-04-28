#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2-in-1: MOD16A2 / MYD16A2 masking + MCD16A2 merge (uses the ET_QC masking standard; merges by QC-score selection).

Input band order (5 bands):
  1: ET (Evapotranspiration)
  2: LT (Latent heat flux)
  3: PET (Potential evapotranspiration)
  4: PLE (Potential latent heat flux)
  5: ET_QC (Quality control)

Masking criteria:
  - Bit 0 (MODLAND_QC): keep all (not checked)
  - Bit 1 (Sensor): not set (keep all)
  - Bit 2 (Dead detector): not set (keep all)
  - Bits 3-4 (Cloud state): drop 1 (keep 0, 2, 3 only)
  - Bits 5-7 (SCF_QC): drop 4 (keep 0, 1, 2, 3 only)

Merging:
  One side valid -> take that side; both invalid -> NoData; both valid -> pick by QC score:
  Priority (high to low):
    1. MODLAND_QC: 0 < 1
    2. Cloud state: 0 (clear) < 3 (assumed clear) < 2 (mixed) < 1 (cloudy)
    3. SCF_QC: 0 < 1 < 2 < 3 < 4
    4. Dead detector: 0 < 1
  Ties broken by --prefer (default: mod).

NoData: uniformly -32765 across all data bands.
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
def qc_fields_et(qc_arr):
    """
    Parse MOD16A2 ET_QC bit fields into a dict:
    0: MODLAND_QC
    1: Sensor
    2: Dead detector
    3-4: Cloud state
    5-7: SCF_QC
    """
    return dict(
        modland_qc = bitslice(qc_arr, 0, 0),
        sensor = bitslice(qc_arr, 1, 1),
        dead_detector = bitslice(qc_arr, 2, 2),
        cloud_state = bitslice(qc_arr, 3, 4),
        scf_qc = bitslice(qc_arr, 5, 7),
    )

# -------------------- Mask criteria --------------------
def build_bad_mask_from_et_qc(qc):
    """
    Returns a boolean bad_mask: True means "low quality and should be masked".
    Masking criteria:
      - Bits 3-4 (Cloud state): drop 1 (keep 0, 2, 3 only)
      - Bits 5-7 (SCF_QC): drop 4 (keep 0, 1, 2, 3 only)
    """
    q = qc_fields_et(qc)
    bad = np.zeros(qc.shape, dtype=bool)
    
    # Cloud state: drop 1 (cloudy)
    bad |= (q["cloud_state"] == 1)
    
    # SCF_QC: drop 4 (pixel not produced)
    bad |= (q["scf_qc"] == 4)
    
    return bad

# -------------------- Scoring system (used for merging) --------------------
def mod16a2_quality_score(qc_arr):
    """
    Quality-scoring system based on ET_QC.
    Returns a score array; each dimension is one priority; smaller is better.
    """
    q = qc_fields_et(qc_arr)
    
    # 1. MODLAND_QC: 0 < 1
    modland = q["modland_qc"].astype(np.uint8)
    
    # 2. Cloud state: 0(clear) < 3(assumed clear) < 2(mixed) < 1(cloudy)
    cloud_state = q["cloud_state"].astype(np.uint8)
    cs_map = np.zeros_like(cloud_state, dtype=np.uint8)
    cs_map[cloud_state == 0] = 0  # clear (best)
    cs_map[cloud_state == 3] = 1  # assumed clear
    cs_map[cloud_state == 2] = 2  # mixed
    cs_map[cloud_state == 1] = 3  # cloudy (worst)
    
    # 3. SCF_QC: 0 < 1 < 2 < 3 < 4
    scf_qc = q["scf_qc"].astype(np.uint8)
    
    # 4. Dead detector: 0 < 1
    dead_detector = q["dead_detector"].astype(np.uint8)
    
    return np.stack([
        modland,        # Priority 1: most important
        cs_map,         # Priority 2
        scf_qc,         # Priority 3
        dead_detector,  # Priority 4
    ], axis=0)

def lexicographic_choose(a_score, b_score, prefer_a=True):
    """Lexicographic selection: compare dimension by dimension; smaller is better"""
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
    compress: str,
):
    """Function to process a single file (used for parallel processing)"""
    if path_out.exists() and not overwrite:
        return True, path_in.name, None

    try:
        with rasterio.open(path_in) as src:
            if src.count < 5:
                return False, path_in.name, f"5 bands required (ET, LT, PET, PLE, ET_QC)."

            # Read band
            et  = src.read(1)
            lt  = src.read(2)
            pet = src.read(3)
            ple = src.read(4)
            qc  = src.read(5)

            # Build mask from QC
            bad = build_bad_mask_from_et_qc(qc)

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
                # Data bands: apply NoData per bad-mask
                for i, arr in enumerate([et, lt, pet, ple], start=1):
                    out_arr = arr.astype(np.float32, copy=False)
                    out_arr[bad] = out_nodata
                    dst.write(out_arr, i)
                    dst.update_tags(i, nodata=str(int(out_nodata)))

                # Write ET_QC back as-is
                dst.write(qc.astype(np.float32), 5)

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
    chunk: int,
    compress: str,
    out_nodata: float,
    overwrite: bool,
):
    """Merge MOD and MYD files for a single date+tile"""
    # Output filename: MCD16A2_YYYY_MM_DD_tileXX.tif
    out_filename = f"MCD16A2_{date_tile_key}.tif"
    out_path = out_dir / out_filename

    if out_path.exists() and not overwrite:
        return True, "skipped"

    # Only one side exists: copy directly
    if (mod_path is None) ^ (myd_path is None):
        shutil.copy2(mod_path or myd_path, out_path)
        return True, "copied"

    # Both sides exist: merge
    with rasterio.open(mod_path) as mod, rasterio.open(myd_path) as myd:
        if mod.count < 5 or myd.count < 5:
            raise ValueError("Input must have 5 bands (ET, LT, PET, PLE, ET_QC).")

        crs, dst_transform, width, height = pick_target_grid(mod, myd)

        # VRT: bilinear for data bands, nearest neighbor for QC
        vrt_opts_float = dict(crs=crs, transform=dst_transform, width=width, height=height,
                              resampling=Resampling.bilinear, add_alpha=False)
        vrt_opts_near  = dict(crs=crs, transform=dst_transform, width=width, height=height,
                              resampling=Resampling.nearest, add_alpha=False)

        bx = ensure_valid_blocksize(min(chunk, 1024))
        by = ensure_valid_blocksize(min(chunk, 1024))
        profile = dict(
            driver="GTiff", count=5, crs=crs, transform=dst_transform,
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

                    # Read data bands (bilinear)
                    m_et  = mod_vrt_f.read(1, window=win)
                    m_lt  = mod_vrt_f.read(2, window=win)
                    m_pet = mod_vrt_f.read(3, window=win)
                    m_ple = mod_vrt_f.read(4, window=win)

                    y_et  = myd_vrt_f.read(1, window=win)
                    y_lt  = myd_vrt_f.read(2, window=win)
                    y_pet = myd_vrt_f.read(3, window=win)
                    y_ple = myd_vrt_f.read(4, window=win)

                    # Read QC (nearest neighbor)
                    m_qc = mod_vrt_n.read(5, window=win).astype(np.uint16, copy=False)
                    y_qc = myd_vrt_n.read(5, window=win).astype(np.uint16, copy=False)

                    # Valid mask (based on the unified nodata)
                    mv_et  = (m_et != out_nodata)
                    mv_lt  = (m_lt != out_nodata)
                    mv_pet = (m_pet != out_nodata)
                    mv_ple = (m_ple != out_nodata)

                    yv_et  = (y_et != out_nodata)
                    yv_lt  = (y_lt != out_nodata)
                    yv_pet = (y_pet != out_nodata)
                    yv_ple = (y_ple != out_nodata)

                    # Scoring
                    s_m = mod16a2_quality_score(m_qc)
                    s_y = mod16a2_quality_score(y_qc)
                    choose_mod_global = lexicographic_choose(s_m, s_y, prefer_a=prefer_mod)

                    # Select data (all data bands use the same selection logic)
                    def choose_value(m_arr, y_arr, mv, yv):
                        both = mv & yv
                        out = np.full_like(m_arr, out_nodata)
                        only_m = mv & ~yv
                        only_y = ~mv & yv
                        out[only_m] = m_arr[only_m]
                        out[only_y] = y_arr[only_y]
                        aidx = both & choose_mod_global
                        bidx = both & ~choose_mod_global
                        out[aidx] = m_arr[aidx]
                        out[bidx] = y_arr[bidx]
                        return out, aidx, bidx

                    out_et, a_et, b_et = choose_value(m_et, y_et, mv_et, yv_et)
                    out_lt, a_lt, b_lt = choose_value(m_lt, y_lt, mv_lt, yv_lt)
                    out_pet, a_pet, b_pet = choose_value(m_pet, y_pet, mv_pet, yv_pet)
                    out_ple, a_ple, b_ple = choose_value(m_ple, y_ple, mv_ple, yv_ple)

                    # QC source: majority vote (ties broken by prefer)
                    m_votes = (a_et + a_lt + a_pet + a_ple).astype(np.int16)
                    y_votes = (b_et + b_lt + b_pet + b_ple).astype(np.int16)
                    choose_mod_qc = (m_votes > y_votes) | ((m_votes == y_votes) & prefer_mod)

                    out_qc = np.where(choose_mod_qc, m_qc, y_qc).astype(np.float32, copy=False)

                    # Write block
                    dst.write(out_et,  1, window=win)
                    dst.write(out_lt,  2, window=win)
                    dst.write(out_pet, 3, window=win)
                    dst.write(out_ple, 4, window=win)
                    dst.write(out_qc,  5, window=win)

            # Set per-band nodata
            for b in range(1, 5):  # Data band
                dst.update_tags(b, nodata=str(int(out_nodata)))

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
    ap = argparse.ArgumentParser(description="MOD16A2 / MYD16A2 masking + MCD16A2 merge (uses the ET_QC masking standard; merges by QC-score selection)")
    ap.add_argument("--mod_in",  type=Path, required=True, help="MOD16A2 input directory")
    ap.add_argument("--myd_in",  type=Path, required=True, help="MYD16A2 input directory")
    ap.add_argument("--mod_out", type=Path, required=True, help="MOD16A2 masked-output directory")
    ap.add_argument("--myd_out", type=Path, required=True, help="MYD16A2 masked-output directory")
    ap.add_argument("--mcd_out", type=Path, required=True, help="MCD16A2 merge-output directory")

    # Merge / performance
    ap.add_argument("--prefer", choices=["mod","myd"], default="mod", help="Tie-break priority when scores are equal (default: mod)")
    ap.add_argument("--chunk", type=int, default=1024, help="Block size in pixels (default: 1024)")
    ap.add_argument("--workers", type=int, default=8, help="Number of parallel processes (default: 8)")
    ap.add_argument("--compress", default="LZW", help="Output compression (default: LZW)")

    # NoData
    ap.add_argument("--out-nodata", type=float, default=-32765, help="Unified NoData value (default: -32765)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output")

    args = ap.parse_args()

    # Mask
    mask_dir(args.mod_in, args.mod_out, args.out_nodata, args)
    mask_dir(args.myd_in, args.myd_out, args.out_nodata, args)

    # Merge
    combine_dirs(args.mod_out, args.myd_out, args.mcd_out, args.out_nodata, args)

if __name__ == "__main__":
    main()

