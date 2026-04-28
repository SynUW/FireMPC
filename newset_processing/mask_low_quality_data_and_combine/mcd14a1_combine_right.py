#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOD14A1 + MYD14A1 same-day composite (fire detection; NoData read directly from TIFF)
-------------------------------------------------
Logic:
1. Inputs: MOD14A1 (Terra, AM) and MYD14A1 (Aqua, PM); band order: ['FireMask', 'MaxFRP', 'QA'].
2. FireMask composite:
   - One side fire (7-9), the other non-fire -> keep the fire pixel.
   - Both fire -> keep the higher confidence (9 > 8 > 7).
   - Both non-fire -> per the following rules:
     * 5 + 5 -> output 5 (Land + Land -> Land)
     * Both 4 -> output 4 (Cloud + Cloud -> Cloud)
     * Both 2 or 1 -> output 2 (Not processed + Not processed -> Not processed)
     * Both 3 -> output 3 (Water + Water -> Water)
     * 3 + 4 -> output 3 (Water + Cloud -> Water)
     * 3 + 1 -> output 3 (Water + Not processed -> Water)
     * 3 + 2 -> output 3 (Water + Not processed -> Water)
     * Otherwise -> output 6 (Unknown)
   - Undefined values (0 or others) -> keep raw value.
   - Preserve raw classes; no binarization.
3. MaxFRP composite:
   - Both fire -> take max.
   - One side fire -> take that value.
   - Non-fire -> output 0.
4. QA composite:
   - Land/water state (bits 0-1): Land(2) > Coast(1) > Water(0) > Missing(3).
   - Day/night (bit 2): when they differ, keep Day(1).
5. Output:
   - Keep all pixels; do not drop.
   - NoData is read directly from the input TIFF's first band (prefer MOD's; fall back to MYD's if MOD is absent).

"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
from osgeo import gdal
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# ----------------- NoData loading -----------------
def get_nodata_from_tiff(tiff_path):
    """Read the NoData value from a TIFF file's first band"""
    if not tiff_path or not os.path.exists(tiff_path):
        return None
    try:
        ds = gdal.Open(tiff_path, gdal.GA_ReadOnly)
        if ds is None:
            return None
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        ds = None
        return nodata
    except Exception:
        return None

def valid_mask_int(arr, nodata_value):
    """Validity check (integer array) based on NoData. nodata_value = None means everything is valid."""
    if nodata_value is None:
        return np.ones_like(arr, dtype=bool)
    return arr != nodata_value

# ---------- Utility functions ----------
def parse_date_tile_key(fname):
    """Parse the filename and return a key 'YYYY_MM_DD_tileXX' (when tile is present) or 'YYYY_MM_DD' otherwise"""
    # Extract date
    m_date = re.search(r"(\d{4})[-_](\d{2})[-_](\d{2})", fname)
    if not m_date:
        return None
    
    date_str = f"{m_date.group(1)}_{m_date.group(2)}_{m_date.group(3)}"
    
    # Try to extract tile number
    m_tile = re.search(r"_tile(\d+)", fname, re.IGNORECASE)
    if m_tile:
        tile_str = m_tile.group(1)
        return f"{date_str}_tile{tile_str}"
    else:
        # If no tile info, return date (backward compatible)
        return date_str

def load_band(ds, band_idx, xoff=0, yoff=0, xsize=None, ysize=None):
    return ds.GetRasterBand(band_idx).ReadAsArray(xoff, yoff, xsize, ysize)

# ---------- Vectorized priority / merging (logic matches the original; better performance) ----------
# Valid FireMask categories (possible values include 0..9); build a 0..255 priority table for fully vectorized comparison
_FIRE_PRIORITY = np.full(256, -999, dtype=np.int16)
# Confidence-rated fire pixels: 9 > 8 > 7
_FIRE_PRIORITY[9] = 6
_FIRE_PRIORITY[8] = 5
_FIRE_PRIORITY[7] = 4
# Non-fire priority: Cloud(4) > Water(3) > Land(5) > Unknown(6) > Not processed(2/1)
_FIRE_PRIORITY[4] = 3
_FIRE_PRIORITY[3] = 2
_FIRE_PRIORITY[5] = 1
_FIRE_PRIORITY[6] = 0
_FIRE_PRIORITY[2] = -1
_FIRE_PRIORITY[1] = -2
# Earlier code included 0; we treat it as low priority (if 0 never appears in your data, this can be ignored)
_FIRE_PRIORITY[0] = -3

_FIRE_CLASSES = np.array([7, 8, 9], dtype=np.int16)

def choose_firemask_vector(mod_fm, myd_fm):
    """
    New FireMask composite logic:
    
    Fire cases (unchanged):
      - One side fire (7-9), the other non-fire -> keep the fire pixel.
      - Both fire -> keep the higher confidence (9 > 8 > 7).
    
    Non-fire cases (new logic):
      - 5 + 5 -> output 5 (Land + Land -> Land)
      - Both 4 -> output 4 (Cloud + Cloud -> Cloud)
      - Both 2 or 1 (Not processed) -> output 2 (Not processed + Not processed -> Not processed)
      - Both 3 -> output 3 (Water + Water -> Water)
      - 3 + 4 -> output 3 (Water + Cloud -> Water)
      - 3 + 1 -> output 3 (Water + Not processed -> Water)
      - 3 + 2 -> output 3 (Water + Not processed -> Water)
      - Otherwise -> output 6 (Unknown)
    
    Note: undefined values (0 or others) fall under "otherwise" and output 6 (Unknown).
    """
    out = np.empty_like(mod_fm, dtype=mod_fm.dtype)
    
    # Decide whether it is a fire pixel
    mod_is_fire = np.isin(mod_fm, _FIRE_CLASSES)
    myd_is_fire = np.isin(myd_fm, _FIRE_CLASSES)
    
    # Case 1: one fire, the other non-fire -> keep the fire pixel
    only_mod_fire = mod_is_fire & (~myd_is_fire)
    only_myd_fire = (~mod_is_fire) & myd_is_fire
    if np.any(only_mod_fire):
        out[only_mod_fire] = mod_fm[only_mod_fire]
    if np.any(only_myd_fire):
        out[only_myd_fire] = myd_fm[only_myd_fire]
    
    # Case 2: both fire -> keep the higher confidence (9 > 8 > 7)
    both_fire = mod_is_fire & myd_is_fire
    if np.any(both_fire):
        pri_mod = _FIRE_PRIORITY[mod_fm[both_fire]]
        pri_myd = _FIRE_PRIORITY[myd_fm[both_fire]]
        out[both_fire] = np.where(pri_mod >= pri_myd, mod_fm[both_fire], myd_fm[both_fire])
    
    # Case 3: both non-fire -> new logic
    both_non_fire = (~mod_is_fire) & (~myd_is_fire)
    if np.any(both_non_fire):
        m = mod_fm[both_non_fire]
        y = myd_fm[both_non_fire]
        result = np.empty_like(m)
        
        # 5 + 5 -> output 5
        mask_5_5 = (m == 5) & (y == 5)
        result[mask_5_5] = 5
        
        # Both 4 -> output 4
        mask_4_4 = (m == 4) & (y == 4)
        result[mask_4_4] = 4
        
        # Both 2 or 1 (Not processed) -> output 2
        mask_notproc = ((m == 1) | (m == 2)) & ((y == 1) | (y == 2))
        result[mask_notproc] = 2
        
        # Both 3 -> output 3
        mask_3_3 = (m == 3) & (y == 3)
        result[mask_3_3] = 3
        
        # 3 + 4 -> output 3
        mask_3_4 = ((m == 3) & (y == 4)) | ((m == 4) & (y == 3))
        result[mask_3_4] = 3
        
        # 3 + 1 -> output 3
        mask_3_1 = ((m == 3) & (y == 1)) | ((m == 1) & (y == 3))
        result[mask_3_1] = 3
        
        # 3 + 2 -> output 3
        mask_3_2 = ((m == 3) & (y == 2)) | ((m == 2) & (y == 3))
        result[mask_3_2] = 3
        
        # Otherwise -> output 6
        handled = mask_5_5 | mask_4_4 | mask_notproc | mask_3_3 | mask_3_4 | mask_3_1 | mask_3_2
        result[~handled] = 6
        
        out[both_non_fire] = result
    
    return out

def combine_qa_vector(mod_qa, myd_qa):
    """
    Equivalent to the original combine_qa (logic preserved):
      - Bits 0-1: land/water (Land(2) > Coast(1) > Water(0) > Missing(3))
      - Bit 2: day/night (prefer Day(1))
    Pure vectorized implementation, avoiding the Python loop inside np.vectorize.
    """
    # Bits 0-1
    lw_mod = (mod_qa & 0b11).astype(np.uint8)
    lw_myd = (myd_qa & 0b11).astype(np.uint8)

    # Priority: Land(2)=3 > Coast(1)=2 > Water(0)=1 > Missing(3)=0
    _LW_PRI = np.array([1, 2, 3, 0], dtype=np.int8)  # Index is 0/1/2/3
    pri_mod = _LW_PRI[lw_mod]
    pri_myd = _LW_PRI[lw_myd]

    lw_out = np.where(pri_mod >= pri_myd, lw_mod, lw_myd).astype(np.uint8)

    # Bit 2: day/night (prefer Day=1)
    dn_mod = ((mod_qa >> 2) & 1).astype(np.uint8)
    dn_myd = ((myd_qa >> 2) & 1).astype(np.uint8)
    dn_out = ((dn_mod == 1) | (dn_myd == 1)).astype(np.uint8)  # If any side is day -> 1, otherwise 0

    return ((dn_out << 2) | lw_out).astype(mod_qa.dtype)

# ---------- Merging logic ----------
def combine_day(mod_path, myd_path, out_path, block_size=512, output_nodata=None):
    ds_mod = gdal.Open(mod_path, gdal.GA_ReadOnly)
    if ds_mod is None:
        raise ValueError(f"Cannot open MOD file: {mod_path}")
    
    ds_myd = gdal.Open(myd_path if myd_path else mod_path, gdal.GA_ReadOnly)
    if ds_myd is None:
        raise ValueError(f"Cannot open MYD file: {myd_path if myd_path else mod_path}")
    
    driver = gdal.GetDriverByName("GTiff")
    if driver is None:
        raise ValueError("Cannot create the GTiff driver")

    xsize = ds_mod.RasterXSize
    ysize = ds_mod.RasterYSize
    bands = ds_mod.RasterCount  # 3: FireMask, MaxFRP, QA

    # Read NoData directly from the TIFF file
    nodata_mod = ds_mod.GetRasterBand(1).GetNoDataValue()
    nodata_myd = ds_myd.GetRasterBand(1).GetNoDataValue() if ds_myd else None

    # Output NoData: prefer MOD; otherwise MYD; otherwise -32768 (do not use 0 to avoid clashing with valid categories)
    if output_nodata is None:
        if nodata_mod is not None:
            output_nodata = nodata_mod
        elif nodata_myd is not None:
            output_nodata = nodata_myd
        else:
            output_nodata = -32768

    out_ds = driver.Create(out_path, xsize, ysize, bands, gdal.GDT_Int16,
                           options=["COMPRESS=LZW", "TILED=YES",
                                    "BLOCKXSIZE=512", "BLOCKYSIZE=512"])
    out_ds.SetProjection(ds_mod.GetProjection())
    out_ds.SetGeoTransform(ds_mod.GetGeoTransform())

    # -- Per the spec: ensure output NoData matches the input (using output_nodata determined above)
    #    All three bands share the same NoData value
    for b in range(1, bands + 1):
        out_ds.GetRasterBand(b).SetNoDataValue(int(output_nodata))

    for y in range(0, ysize, block_size):
        rows = min(block_size, ysize - y)
        for x in range(0, xsize, block_size):
            cols = min(block_size, xsize - x)

            # Read three bands
            mod_fm = load_band(ds_mod, 1, x, y, cols, rows).astype(np.int16)
            mod_frp = load_band(ds_mod, 2, x, y, cols, rows).astype(np.int16)
            mod_qa = load_band(ds_mod, 3, x, y, cols, rows).astype(np.int16)

            myd_fm = load_band(ds_myd, 1, x, y, cols, rows).astype(np.int16)
            myd_frp = load_band(ds_myd, 2, x, y, cols, rows).astype(np.int16)
            myd_qa = load_band(ds_myd, 3, x, y, cols, rows).astype(np.int16)

            # Validity (based on FireMask's Excel NoData)
            mod_valid = valid_mask_int(mod_fm, nodata_mod)
            myd_valid = valid_mask_int(myd_fm, nodata_myd)

            only_mod = mod_valid & (~myd_valid)
            only_myd = myd_valid & (~mod_valid)
            both_valid = mod_valid & myd_valid
            none_valid = (~mod_valid) & (~myd_valid)

            # Initialize output (uniformly filled with output_nodata)
            fire_out = np.full_like(mod_fm, int(output_nodata), dtype=np.int16)
            frp_out  = np.full_like(mod_frp, int(output_nodata), dtype=np.int16)
            qa_out   = np.full_like(mod_qa, int(output_nodata), dtype=np.int16)

            # -------- FireMask merge --------
            if np.any(only_mod):
                fire_out[only_mod] = mod_fm[only_mod]
            if np.any(only_myd):
                fire_out[only_myd] = myd_fm[only_myd]
            if np.any(both_valid):
                fire_out[both_valid] = choose_firemask_vector(mod_fm[both_valid], myd_fm[both_valid])
            # none_valid stays as output_nodata

            # -------- MaxFRP merge (keeping your rule: non-fire -> 0, not NoData) --------
            # only_mod
            if np.any(only_mod):
                fire_mod_only = np.isin(mod_fm[only_mod], _FIRE_CLASSES)
                # Fire pixel -> take source FRP; non-fire pixel -> 0
                frp_out[only_mod] = np.where(fire_mod_only, mod_frp[only_mod], 0).astype(np.int16)
            # only_myd
            if np.any(only_myd):
                fire_myd_only = np.isin(myd_fm[only_myd], _FIRE_CLASSES)
                frp_out[only_myd] = np.where(fire_myd_only, myd_frp[only_myd], 0).astype(np.int16)
            # both_valid
            if np.any(both_valid):
                fire_mod_b = np.isin(mod_fm[both_valid], _FIRE_CLASSES)
                fire_myd_b = np.isin(myd_fm[both_valid], _FIRE_CLASSES)
                both_fire  = fire_mod_b & fire_myd_b
                only_mfire = fire_mod_b & (~fire_myd_b)
                only_afire = (~fire_mod_b) & fire_myd_b
                neither    = (~fire_mod_b) & (~fire_myd_b)

                # Fill 0 first (non-fire = 0)
                block = np.zeros_like(mod_frp[both_valid], dtype=np.int16)
                # Both fire -> take max
                if np.any(both_fire):
                    block[both_fire] = np.maximum(mod_frp[both_valid][both_fire],
                                                  myd_frp[both_valid][both_fire]).astype(np.int16)
                # Only one side fire -> take that value
                if np.any(only_mfire):
                    block[only_mfire] = mod_frp[both_valid][only_mfire]
                if np.any(only_afire):
                    block[only_afire] = myd_frp[both_valid][only_afire]
                # neither already is 0
                frp_out[both_valid] = block

            # none_valid: keep output_nodata (the non-fire=0 rule only applies to pixels with valid input)

            # -------- QA merge (keep original logic: land/water priority, day preferred over night) --------
            if np.any(only_mod):
                qa_out[only_mod] = mod_qa[only_mod]
            if np.any(only_myd):
                qa_out[only_myd] = myd_qa[only_myd]
            if np.any(both_valid):
                qa_out[both_valid] = combine_qa_vector(mod_qa[both_valid], myd_qa[both_valid])

            # Write out
            out_ds.GetRasterBand(1).WriteArray(fire_out, x, y)
            out_ds.GetRasterBand(2).WriteArray(frp_out,  x, y)
            out_ds.GetRasterBand(3).WriteArray(qa_out,   x, y)

    ds_mod = None
    ds_myd = None
    out_ds = None

# ---------- Batch processing ----------
def process_single_task(task):
    d, mod_path, myd_path, out_dir, block_size, output_nodata = task
    try:
        out_path = os.path.join(out_dir, f"MCD14A1_{d}.tif")
        if mod_path and myd_path:
            if not os.path.exists(mod_path):
                raise FileNotFoundError(f"MOD file does not exist: {mod_path}")
            if not os.path.exists(myd_path):
                raise FileNotFoundError(f"MYD file does not exist: {myd_path}")
            combine_day(mod_path, myd_path, out_path, block_size, output_nodata=output_nodata)
        else:
            # Only one side exists -> copy directly
            src = mod_path if mod_path else myd_path
            if not os.path.exists(src):
                raise FileNotFoundError(f"Source file does not exist: {src}")
            gdal.Translate(out_path, src, creationOptions=["COMPRESS=LZW", "TILED=YES",
                                                           "BLOCKXSIZE=512", "BLOCKYSIZE=512"])
        return True
    except Exception as e:
        print(f"\nProcessing {d} failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False

def process_all(mod_dir, myd_dir, out_dir, workers=4, block_size=512, output_nodata=None):
    os.makedirs(out_dir, exist_ok=True)

    # List date + tile
    print(f"Scanning files...")
    mod_files = {parse_date_tile_key(f): os.path.join(mod_dir, f)
                 for f in os.listdir(mod_dir)} if mod_dir and os.path.exists(mod_dir) else {}
    mod_files = {k: v for k, v in mod_files.items() if k and v.lower().endswith(".tif")}
    print(f"Found {len(mod_files)} MOD14A1 files")
    
    myd_files = {parse_date_tile_key(f): os.path.join(myd_dir, f)
                 for f in os.listdir(myd_dir)} if myd_dir and os.path.exists(myd_dir) else {}
    myd_files = {k: v for k, v in myd_files.items() if k and v.lower().endswith(".tif")}
    print(f"Found {len(myd_files)} MYD14A1 files")

    all_dates = sorted(set(mod_files.keys()) | set(myd_files.keys()))
    print(f"Total found {len(all_dates)} unique dates")
    
    # Quickly check existing output files (list all outputs at once, avoiding per-file checks)
    print(f"Checking existing output files...", flush=True)
    existing_outputs = set()
    if os.path.exists(out_dir):
        for f in os.listdir(out_dir):
            if f.lower().endswith(".tif") and f.startswith("MCD14A1_"):
                # Extract the date + tile key
                key = f[8:-4]  # Strip the "MCD14A1_" prefix and the ".tif" suffix
                existing_outputs.add(key)
    print(f"Found {len(existing_outputs)} existing output files", flush=True)
    
    # Build the task list (only items that need processing)
    print(f"Building task list...", flush=True)
    tasks = []
    for d in all_dates:
        mod_path = mod_files.get(d)
        myd_path = myd_files.get(d)
        if not mod_path and not myd_path:
            continue
        if d in existing_outputs:
            continue
        tasks.append((d, mod_path, myd_path, out_dir, block_size, output_nodata))

    print(f"Need to process {len(tasks)} tasks (skipped {len(existing_outputs)} existing files)", flush=True)
    if not tasks:
        print("No tasks to process (everything may already exist).", flush=True)
        return

    if workers <= 1:
        print(f"Starting sequential processing {len(tasks)} tasks...", flush=True)
        for task in tqdm(tasks, desc="Process files"):
            process_single_task(task)
    else:
        print(f"Using {workers} processes processing in parallel {len(tasks)} tasks...", flush=True)
        print("Starting the process pool...", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            print("Process pool started; beginning processing...", flush=True)
            list(tqdm(ex.map(process_single_task, tasks),
                      total=len(tasks), desc="Process files"))

# ----------------- CLI -----------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MOD14A1 + MYD14A1 composite (fire detection; NoData read directly from TIFF)")
    parser.add_argument("--mod_dir", default='/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MOD14A1', help="MOD14A1 folder path")
    parser.add_argument("--myd_dir", default='/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MOD14A1', help="MYD14A1 folder path")
    parser.add_argument("--out_dir", default='/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/MCD14A1_combined', help="Output folder path")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel processes")
    parser.add_argument("--block_size", type=int, default=512, help="Block size")
    parser.add_argument("--output-nodata", type=int, default=None,
                        help="Optional: force the output NoData; defaults to reading the input TIFF's first band, or -32768 if missing")
    args = parser.parse_args()

    process_all(args.mod_dir, args.myd_dir, args.out_dir,
                workers=args.workers, block_size=args.block_size,
                output_nodata=args.output_nodata)
