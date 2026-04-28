#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2-in-1: MOD09A1 / MYD09A1 masking + MCD09A1 merge (keeps the original masking standards; merges by quality-score selection).

Input band order:
  - Bands 1..(count-2): reflectance bands (typically 4 bands: b01, b02, b03, b07)
  - Band (count-1): QA (32-bit)
  - Band (count):   StateQA (16-bit)

Masking (keeps original standards):
  - Uses build_good_mask_mod09a1, preserving all original masking parameters

Merging:
  One side valid -> take that side; both invalid -> NoData; both valid -> pick by quality score:
  Priority (high to low):
    1. MODLAND QA: 0 < 1 < 2 < 3
    2. Cloud state: 0 (clear) < 3 (assumed clear) < 2 (mixed) < 1 (cloudy)
    3. Internal cloud: 0 < 1
    4. Cloud shadow: 0 < 1
    5. Adjacent to cloud: 0 < 1
    6. MOD35 snow/ice: 0 < 1
    7. Internal snow: 0 < 1
    8. Aerosol: 0 < 1 < 2 < 3
    9. Cirrus: 0 < 1 < 2 < 3
    10. Atmospheric correction: 1 < 0
    11. Adjacency correction: 1 < 0
  Ties are broken by --prefer (default mod).

NoData: data bands are uniformly written as -32768 (override with --out-nodata); QA bands are written back as-is with no nodata set.
"""

import os
import re
import math
import shutil
import argparse
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Set

import numpy as np
import rasterio
from rasterio import windows
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds, Resampling
from rasterio.errors import NotGeoreferencedWarning
import warnings
warnings.simplefilter("ignore", NotGeoreferencedWarning)

# Initialize GDAL (used in the masking stage)
from osgeo import gdal
gdal.UseExceptions()

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

def bit_get(arr: np.ndarray, shift: int, width: int = 1) -> np.ndarray:
    mask = (1 << width) - 1
    return (arr >> shift) & mask

def parse_set_ints(s: str) -> Set[int]:
    out = set()
    for part in s.split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out

def ensure_valid_blocksize(b, max_size=1024, min_size=16):
    b = max(min(b, max_size), min_size)
    return (b // 16) * 16 or 16

def print_progress(prefix: str, ok: int, total: int, fail: int):
    print(f"\r{prefix}: OK={ok}/{total}, FAIL={fail}", end="", flush=True)

def finish_progress():
    print("", flush=True)

# -------------------- Masking (keep original standards) --------------------
def build_good_mask_mod09a1(
    qa: np.ndarray,
    stateqa: np.ndarray,
    *,
    # --- QA ---
    require_modland_ideal: bool = True,
    require_atcorr_performed: bool = True,
    require_adjcorr_performed: bool = False,
    use_band_quality: bool = False,
    allow_band_quality_values: Set[int] = None,  # e.g., {0} or {0,7}
    # --- StateQA ---
    allow_cloud_states: Set[int] = None,         # e.g., {0,3}
    require_no_cloud_shadow: bool = True,
    require_internal_cloud_free: bool = True,
    require_not_adjacent_cloud: bool = False,
    require_no_snow_ice: bool = False,           # MOD35 snow/ice flag bit12
    require_internal_no_snow: bool = False,      # internal snow mask bit15
    require_no_fire_flag: bool = False,          # internal fire flag bit11
    use_aerosol: bool = False,
    allow_aerosol_qty: Set[int] = None,          # bits6-7
    use_cirrus: bool = False,
    allow_cirrus: Set[int] = None,               # bits8-9
) -> np.ndarray:
    """
    Return a boolean mask: True = keep pixel.
    Original masking standards preserved.
    """

    if allow_band_quality_values is None:
        allow_band_quality_values = {0}
    if allow_cloud_states is None:
        allow_cloud_states = {0, 3}
    if allow_aerosol_qty is None:
        allow_aerosol_qty = {0, 1, 2, 3}
    if allow_cirrus is None:
        allow_cirrus = {0, 1, 2, 3}

    qa_u32 = qa.astype(np.uint32, copy=False)
    st_u16 = stateqa.astype(np.uint16, copy=False)

    good = np.ones(qa_u32.shape, dtype=bool)

    # ---------------- QA (32-bit) ----------------
    # Bits 0-1: MODLAND QA
    if require_modland_ideal:
        modland = bit_get(qa_u32, 0, 2)  # 0..3
        good &= np.isin(modland, [0, 1])

    # Bits 2-29: band data quality (4 bits per band 1..7)
    # band1: bits2-5, band2: bits6-9, ..., band7: bits26-29
    if use_band_quality:
        for b in range(7):
            shift = 2 + 4 * b
            bq = bit_get(qa_u32, shift, 4)
            good &= np.isin(bq, list(allow_band_quality_values))

    # Bit 30: Atmospheric correction performed
    if require_atcorr_performed:
        atcorr = bit_get(qa_u32, 30, 1)
        good &= (atcorr == 1)

    # Bit 31: Adjacency correction performed
    if require_adjcorr_performed:
        adjcorr = bit_get(qa_u32, 31, 1)
        good &= (adjcorr == 1)

    # ---------------- StateQA (16-bit) ----------------
    # Bits 0-1: Cloud state
    cloud_state = bit_get(st_u16, 0, 2)
    good &= np.isin(cloud_state, list(allow_cloud_states))

    # Bit 2: Cloud shadow
    if require_no_cloud_shadow:
        shadow = bit_get(st_u16, 2, 1)
        good &= (shadow == 0)

    # Bit 10: Internal cloud algorithm flag
    if require_internal_cloud_free:
        icloud = bit_get(st_u16, 10, 1)
        good &= (icloud == 0)

    # Bit 13: Pixel is adjacent to cloud
    if require_not_adjacent_cloud:
        adj = bit_get(st_u16, 13, 1)
        good &= (adj == 0)

    # Bit 12: MOD35 snow/ice flag
    if require_no_snow_ice:
        snow = bit_get(st_u16, 12, 1)
        good &= (snow == 0)

    # Bit 15: Internal snow mask
    if require_internal_no_snow:
        snow2 = bit_get(st_u16, 15, 1)
        good &= (snow2 == 0)

    # Bit 11: Internal fire algorithm flag
    if require_no_fire_flag:
        fire = bit_get(st_u16, 11, 1)
        good &= (fire == 0)

    # Bits 6-7: Aerosol quantity
    if use_aerosol:
        aerosol = bit_get(st_u16, 6, 2)
        good &= np.isin(aerosol, list(allow_aerosol_qty))

    # Bits 8-9: Cirrus detected
    if use_cirrus:
        cirrus = bit_get(st_u16, 8, 2)
        good &= np.isin(cirrus, list(allow_cirrus))

    return good

# -------------------- Scoring system (quality-metric based) --------------------
def mod09a1_quality_score(qa_u32: np.ndarray, stateqa_u16: np.ndarray):
    """
    Quality-scoring system based on QA and StateQA.
    Returns a score array; each dimension is one priority; smaller is better.
    """
    # 1. MODLAND QA (0 best, 1 next, 2 and 3 worst)
    modland = bit_get(qa_u32, 0, 2).astype(np.uint8)
    
    # 2. Cloud state: 0(clear) < 3(assumed clear) < 2(mixed) < 1(cloudy)
    cloud_state = bit_get(stateqa_u16, 0, 2).astype(np.uint8)
    cs_map = np.zeros_like(cloud_state, dtype=np.uint8)
    cs_map[cloud_state == 0] = 0  # clear (best)
    cs_map[cloud_state == 3] = 1  # assumed clear
    cs_map[cloud_state == 2] = 2  # mixed
    cs_map[cloud_state == 1] = 3  # cloudy (worst)
    
    # 3. Internal cloud: 0 < 1
    internal_cloud = bit_get(stateqa_u16, 10, 1).astype(np.uint8)
    
    # 4. Cloud shadow: 0 < 1
    cloud_shadow = bit_get(stateqa_u16, 2, 1).astype(np.uint8)
    
    # 5. Adjacent to cloud: 0 < 1
    adjacent_cloud = bit_get(stateqa_u16, 13, 1).astype(np.uint8)
    
    # 6. MOD35 snow/ice: 0 < 1
    snow_ice = bit_get(stateqa_u16, 12, 1).astype(np.uint8)
    
    # 7. Internal snow: 0 < 1
    internal_snow = bit_get(stateqa_u16, 15, 1).astype(np.uint8)
    
    # 8. Aerosol: 0 < 1 < 2 < 3
    aerosol = bit_get(stateqa_u16, 6, 2).astype(np.uint8)
    
    # 9. Cirrus: 0 < 1 < 2 < 3
    cirrus = bit_get(stateqa_u16, 8, 2).astype(np.uint8)
    
    # 10. Atmospheric correction: 1 < 0 (reversed; 1 is best)
    atcorr = bit_get(qa_u32, 30, 1).astype(np.uint8)
    atcorr_map = 1 - atcorr  # 1->0 (best), 0->1 (worst)
    
    # 11. Adjacency correction: 1 < 0 (reversed; 1 is best)
    adjcorr = bit_get(qa_u32, 31, 1).astype(np.uint8)
    adjcorr_map = 1 - adjcorr  # 1->0 (best), 0->1 (worst)
    
    return np.stack([
        modland,           # Priority 1: most important
        cs_map,            # Priority 2
        internal_cloud,    # Priority 3
        cloud_shadow,      # Priority 4
        adjacent_cloud,    # Priority 5
        snow_ice,          # Priority 6
        internal_snow,     # Priority 7
        aerosol,           # Priority 8
        cirrus,            # Priority 9
        atcorr_map,        # Priority 10
        adjcorr_map,       # Priority 11
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

# -------------------- Masking (single file, using GDAL) --------------------
def process_single_file(
    fn: str,
    input_dir: str,
    output_dir: str,
    nodata: float,
    overwrite: bool,
    require_modland_ideal: bool,
    require_atcorr_performed: bool,
    require_adjcorr_performed: bool,
    use_band_quality: bool,
    allow_band_quality_values: Set[int],
    allow_cloud_states: Set[int],
    require_no_cloud_shadow: bool,
    require_internal_cloud_free: bool,
    require_not_adjacent_cloud: bool,
    require_no_snow_ice: bool,
    require_internal_no_snow: bool,
    require_no_fire_flag: bool,
    use_aerosol: bool,
    allow_aerosol_qty: Set[int],
    use_cirrus: bool,
    allow_cirrus: Set[int],
):
    """Function to process a single file (used for parallel processing)"""
    in_path = os.path.join(input_dir, fn)
    out_path = os.path.join(output_dir, fn)

    if (not overwrite) and os.path.exists(out_path):
        return True, fn, None

    try:
        # Open input file with GDAL
        src_ds = gdal.Open(in_path, gdal.GA_ReadOnly)
        if src_ds is None:
            return False, fn, f"Cannot open file: {in_path}"
        
        nb = src_ds.RasterCount
        if nb < 3:
            src_ds = None
            return False, fn, f"band count={nb}, expected >= 3 (refl..., QA, StateQA)"
        
        width = src_ds.RasterXSize
        height = src_ds.RasterYSize
        geotransform = src_ds.GetGeoTransform()
        projection = src_ds.GetProjection()
        
        # Read QA and StateQA bands (the last two bands)
        qa_band = src_ds.GetRasterBand(nb - 1)
        stateqa_band = src_ds.GetRasterBand(nb)
        
        qa = qa_band.ReadAsArray()
        stateqa = stateqa_band.ReadAsArray()
        
        # Build mask
        good = build_good_mask_mod09a1(
            qa, stateqa,
            require_modland_ideal=require_modland_ideal,
            require_atcorr_performed=require_atcorr_performed,
            require_adjcorr_performed=require_adjcorr_performed,
            use_band_quality=use_band_quality,
            allow_band_quality_values=allow_band_quality_values,
            allow_cloud_states=allow_cloud_states,
            require_no_cloud_shadow=require_no_cloud_shadow,
            require_internal_cloud_free=require_internal_cloud_free,
            require_not_adjacent_cloud=require_not_adjacent_cloud,
            require_no_snow_ice=require_no_snow_ice,
            require_internal_no_snow=require_internal_no_snow,
            require_no_fire_flag=require_no_fire_flag,
            use_aerosol=use_aerosol,
            allow_aerosol_qty=allow_aerosol_qty,
            use_cirrus=use_cirrus,
            allow_cirrus=allow_cirrus,
        )
        
        # Read reflectance bands (1 to nb-2)
        refl_bands = []
        src_nodata = None
        for band_idx in range(1, nb - 1):
            band = src_ds.GetRasterBand(band_idx)
            if src_nodata is None:
                src_nodata = band.GetNoDataValue()
            refl_bands.append(band.ReadAsArray())
        
        # Combine into an array of shape (nb-2, H, W)
        refl = np.array(refl_bands)
        
        # Handle source NoData values
        if src_nodata is not None:
            valid_src = np.all(refl != src_nodata, axis=0)
        else:
            valid_src = np.ones(good.shape, dtype=bool)
        
        keep = good & valid_src
        
        # Apply mask: set bad pixels to NoData
        refl_out = refl.astype(np.float32)
        refl_out[:, ~keep] = float(nodata)
        
        # Create output file with GDAL
        driver = gdal.GetDriverByName('GTiff')
        
        # Compute block size: must be a multiple of 16
        blockxsize = 256
        blockysize = 256
        if width < blockxsize:
            blockxsize = ((width + 15) // 16) * 16
        if height < blockysize:
            blockysize = ((height + 15) // 16) * 16
        
        creation_options = [
            'COMPRESS=LZW',
            'TILED=YES',
            f'BLOCKXSIZE={blockxsize}',
            f'BLOCKYSIZE={blockysize}',
            'BIGTIFF=IF_SAFER',
            'NUM_THREADS=ALL_CPUS'
        ]
        
        out_ds = driver.Create(
            out_path,
            width,
            height,
            nb,  # All bands
            gdal.GDT_Float32,
            options=creation_options
        )
        
        if out_ds is None:
            src_ds = None
            return False, fn, f"Cannot create output file: {out_path}"
        
        # Set georeferencing
        out_ds.SetGeoTransform(geotransform)
        out_ds.SetProjection(projection)
        
        # Write reflectance bands
        for band_idx in range(1, nb - 1):
            out_band = out_ds.GetRasterBand(band_idx)
            out_band.SetNoDataValue(float(nodata))
            out_band.WriteArray(refl_out[band_idx - 1])
        
        # Write QA and StateQA bands (keep raw values, cast to float32)
        out_qa_band = out_ds.GetRasterBand(nb - 1)
        # Do not set NoData on the QA band
        out_qa_band.WriteArray(qa.astype(np.float32))
        
        out_stateqa_band = out_ds.GetRasterBand(nb)
        # Do not set NoData on the StateQA band
        out_stateqa_band.WriteArray(stateqa.astype(np.float32))
        
        # Flush cache
        out_ds.FlushCache()
        
        # Clean up resources
        out_ds = None
        src_ds = None
        
        return True, fn, None
        
    except Exception as e:
        return False, fn, str(e)

# -------------------- Merge (single date + tile, using rasterio) --------------------
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
        if mod.count < 3 or myd.count < 3:
            raise ValueError("At least 3 bands are required (refl..., QA, StateQA).")
        
        # Ensure band counts match
        if mod.count != myd.count:
            raise ValueError(f"MOD and MYD band counts differ: MOD={mod.count}, MYD={myd.count}")

        nb = mod.count
        n_refl = nb - 2  # Number of reflectance bands

        crs, transform, width, height = pick_target_grid(mod, myd)

        # VRT: bilinear for reflectance, nearest neighbor for QA
        vrt_f_opts = dict(crs=crs, transform=transform, width=width, height=height, resampling=Resampling.bilinear)
        vrt_n_opts = dict(crs=crs, transform=transform, width=width, height=height, resampling=Resampling.nearest)

        # Use float32 uniformly
        out_dtype = 'float32'
        bx = ensure_valid_blocksize(min(chunk, 1024))
        by = ensure_valid_blocksize(min(chunk, 1024))
        profile = dict(
            driver="GTiff", count=nb, crs=crs, transform=transform,
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

                    # Read reflectance bands (bilinear)
                    m_refl = []
                    y_refl = []
                    for b in range(1, n_refl + 1):
                        m_refl.append(m_vrt_f.read(b, window=win))
                        y_refl.append(y_vrt_f.read(b, window=win))
                    m_refl = np.array(m_refl)  # shape: (n_refl, H, W)
                    y_refl = np.array(y_refl)

                    # Read QA and StateQA (nearest neighbor)
                    m_qa = m_vrt_n.read(nb - 1, window=win).astype(np.uint32, copy=False)
                    m_stateqa = m_vrt_n.read(nb, window=win).astype(np.uint16, copy=False)
                    y_qa = y_vrt_n.read(nb - 1, window=win).astype(np.uint32, copy=False)
                    y_stateqa = y_vrt_n.read(nb, window=win).astype(np.uint16, copy=False)

                    # Validity (masking stage has already written NoData)
                    m_valid = np.ones((h, w), dtype=bool)
                    y_valid = np.ones((h, w), dtype=bool)
                    for b in range(n_refl):
                        m_valid &= (m_refl[b] != out_nodata)
                        y_valid &= (y_refl[b] != out_nodata)

                    # Scoring
                    s_m = mod09a1_quality_score(m_qa, m_stateqa)
                    s_y = mod09a1_quality_score(y_qa, y_stateqa)
                    choose_mod_global = lexicographic_choose(s_m, s_y, prefer_a=prefer_mod)

                    # Select data
                    def choose_value(m_arr, y_arr, mv, yv):
                        both = mv & yv
                        out = np.empty_like(m_arr); out[:] = out_nodata
                        only_m = mv & ~yv; only_y = ~mv & yv
                        out[only_m] = m_arr[only_m]
                        out[only_y] = y_arr[only_y]
                        aidx = both & choose_mod_global
                        bidx = both & ~choose_mod_global
                        out[aidx] = m_arr[aidx]
                        out[bidx] = y_arr[bidx]
                        return out, aidx, bidx

                    # Process each reflectance band
                    out_refl = []
                    m_votes = np.zeros((h, w), dtype=np.int16)
                    y_votes = np.zeros((h, w), dtype=np.int16)
                    for b in range(n_refl):
                        out_b, a_b, b_b = choose_value(m_refl[b], y_refl[b], m_valid, y_valid)
                        out_refl.append(out_b)
                        m_votes += a_b.astype(np.int16)
                        y_votes += b_b.astype(np.int16)

                    # QA source: majority vote (ties broken by prefer)
                    choose_mod_qa = (m_votes > y_votes) | ((m_votes == y_votes) & prefer_mod)

                    out_qa = np.where(choose_mod_qa, m_qa, y_qa).astype(np.uint32, copy=False)
                    out_stateqa = np.where(choose_mod_qa, m_stateqa, y_stateqa).astype(np.uint16, copy=False)

                    # Write block
                    for b in range(n_refl):
                        dst.write(out_refl[b], b + 1, window=win)
                    dst.write(out_qa, nb - 1, window=win)
                    dst.write(out_stateqa, nb, window=win)

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
    out_dir.mkdir(parents=True, exist_ok=True)
    tifs = [p for p in list_tifs(in_dir)]
    total = len(tifs)
    prefix = f"[MASK] {in_dir.name} → {out_dir.name}"
    print_progress(prefix, 0, total, 0)

    # Prepare task parameters
    allow_cloud_states = parse_set_ints(args.allow_cloud_states)
    allow_band_quality_values = parse_set_ints(args.allow_band_quality_values)
    allow_aerosol_qty = parse_set_ints(args.allow_aerosol_qty)
    allow_cirrus = parse_set_ints(args.allow_cirrus)

    tasks = [
        (
            p.name,
            str(in_dir),
            str(out_dir),
            out_nodata,
            args.overwrite,
            args.require_modland_ideal,
            args.require_atcorr_performed,
            args.require_adjcorr_performed,
            args.use_band_quality,
            allow_band_quality_values,
            allow_cloud_states,
            args.require_no_cloud_shadow,
            args.require_internal_cloud_free,
            args.require_not_adjacent_cloud,
            args.require_no_snow_ice,
            args.require_internal_no_snow,
            args.require_no_fire_flag,
            args.use_aerosol,
            allow_aerosol_qty,
            args.use_cirrus,
            allow_cirrus,
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
                print(f"\nException {task[0]}: {e}")
            finally:
                print_progress(prefix, ok, total, fail)
    finish_progress()

def combine_dirs(mod_masked: Path, myd_masked: Path, out_dir: Path, out_nodata: float, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = build_pairs(mod_masked, myd_masked)
    total = len(pairs)
    prefix = f"[COMBINE] {mod_masked.name}+{myd_masked.name} → {out_dir.name}"
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
            except Exception as e:
                fail += 1
                print(f"\nException: {e}")
            finally:
                print_progress(prefix, ok, total, fail)
    finish_progress()

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="MOD09A1 / MYD09A1 masking + MCD09A1 merge (keeps the original masking standard; merges by quality-score selection)")
    ap.add_argument("--mod_in",  type=Path, default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MOD09A1", help="MOD09A1 input directory")
    ap.add_argument("--myd_in",  type=Path, default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MYD09A1", help="MYD09A1 input directory")
    ap.add_argument("--mod_out", type=Path, default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MOD09A1_QAapplied", help="MOD09A1 masked-output directory")
    ap.add_argument("--myd_out", type=Path, default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MYD09A1_QAapplied", help="MYD09A1 masked-output directory")
    ap.add_argument("--mcd_out", type=Path, default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MCD09A1_QAapplied", help="MCD09A1 merge-output directory")

    # Merge / performance
    ap.add_argument("--prefer", choices=["mod","myd"], default="mod", help="Tie-break priority when scores are equal (default: mod)")
    ap.add_argument("--chunk", type=int, default=1024, help="Block size in pixels (default: 1024)")
    ap.add_argument("--workers", type=int, default=8, help="Number of parallel processes (default: 8)")
    ap.add_argument("--compress", default="LZW", help="Output compression (default: LZW)")

    # NoData & overwrite
    ap.add_argument("--out-nodata", type=float, default=-32768, help="Unified NoData value for data bands (default: -32768)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs (process from scratch)")

    # ---------- Mask parameters (keep original standards) ----------
    ap.add_argument("--require_modland_ideal", action="store_true", default=True,
                    help="Require MODLAND QA bits 0-1 == 0 or 1 (enabled by default)")
    ap.add_argument("--no_require_modland_ideal", action="store_false", dest="require_modland_ideal",
                    help="Do not require MODLAND ideal")

    ap.add_argument("--require_atcorr_performed", action="store_true", default=False,
                    help="Require QA bit30 = 1 (atmospheric correction performed, off by default)")
    ap.add_argument("--no_require_atcorr_performed", action="store_false", dest="require_atcorr_performed",
                    help="Do not require the atmospheric-correction bit")

    ap.add_argument("--allow_cloud_states", type=str, default="0,3",
                    help="Allowed cloud states (StateQA bits 0-1); default 0, 3 (clear / assumed clear)")

    ap.add_argument("--require_no_cloud_shadow", action="store_true", default=False,
                    help="Require StateQA bit2 = 0 (no cloud shadow, off by default)")
    ap.add_argument("--no_require_no_cloud_shadow", action="store_false", dest="require_no_cloud_shadow")

    ap.add_argument("--require_internal_cloud_free", action="store_true", default=False,
                    help="Require StateQA bit10 = 0 (internal cloud = 0, off by default)")
    ap.add_argument("--no_require_internal_cloud_free", action="store_false", dest="require_internal_cloud_free")

    ap.add_argument("--require_adjcorr_performed", action="store_true", default=False,
                    help="Require QA bit31 = 1 (adjacency correction performed, off by default)")

    ap.add_argument("--use_band_quality", action="store_true", default=False,
                    help="Enable per-band data-quality filtering (QA bits 2-29, off by default)")
    ap.add_argument("--allow_band_quality_values", type=str, default="0",
                    help="Effective when --use_band_quality is enabled: allowed 4-bit band-quality values; default 0 (highest quality)")

    ap.add_argument("--require_not_adjacent_cloud", action="store_true", default=False,
                    help="Require StateQA bit13 = 0 (not adjacent to cloud, off by default)")

    ap.add_argument("--require_no_snow_ice", action="store_true", default=False,
                    help="Require StateQA bit12 = 0 (MOD35 snow/ice = 0, off by default)")
    ap.add_argument("--require_internal_no_snow", action="store_true", default=False,
                    help="Require StateQA bit15 = 0 (internal snow mask = 0, off by default)")

    ap.add_argument("--require_no_fire_flag", action="store_true", default=False,
                    help="Require StateQA bit11 = 0 (internal fire flag = 0, off by default)")

    ap.add_argument("--use_aerosol", action="store_true", default=False,
                    help="Enable aerosol-quantity filtering (StateQA bits 6-7, off by default)")
    ap.add_argument("--allow_aerosol_qty", type=str, default="0,1,2,3",
                    help="Allowed aerosol-quantity values; default: allow all")

    ap.add_argument("--use_cirrus", action="store_true", default=False,
                    help="Enable cirrus filtering (StateQA bits 8-9, off by default)")
    ap.add_argument("--allow_cirrus", type=str, default="0,1,2,3",
                    help="Allowed cirrus values; default: allow all")

    args = ap.parse_args()

    # Mask
    mask_dir(args.mod_in, args.mod_out, args.out_nodata, args)
    mask_dir(args.myd_in, args.myd_out, args.out_nodata, args)

    # Merge
    combine_dirs(args.mod_out, args.myd_out, args.mcd_out, args.out_nodata, args)

if __name__ == "__main__":
    main()
