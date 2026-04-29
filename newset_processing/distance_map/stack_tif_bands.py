#!/usr/bin/env python3
"""
Stack the GeoTIFFs in a folder along the band axis into a single multi-band image and report the band order.

Features:
- Collect TIFs from the input directory, sorted by filename (optionally natural sort).
- Strictly validate that all inputs have consistent size, projection, and affine.
- Write all bands of each input TIF in order; multi-band files are flattened and appended.
- Output is a single multi-band GTiff with the first input's projection and affine.
- Unify NoData to -9999: source-band NoData (including NaN) is mapped to -9999 before writing, and all output bands are set to NoData=-9999. If the first file's dtype is unsigned integer, the output dtype is upgraded to Float32 to hold -9999.
- Also writes a sidecar mapping file (.bands.txt) listing band_index -> source_file [band_index].
- If dtypes differ, casts to the output dtype with a warning.
- Supports LZW compression, tiling, and BIGTIFF.

Example:
python3 stack_tif_bands.py \
  --input-dir /path/to/src \
  --output /path/to/stacked.tif \
  --pattern "*.tif" \
  --natural-sort \
  --save-map
"""

import os
import re
import glob
import logging
import argparse
from typing import List, Tuple

import numpy as np
from osgeo import gdal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('stack_tif_bands')

TARGET_NODATA = -32768.0

def list_tifs(folder: str, pattern: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, pattern)))

def natural_key(s: str):
    # Natural-sort key: sort numeric parts as integers
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def read_info(ds) -> Tuple[int, int, int, str, tuple, int]:
    w = ds.RasterXSize
    h = ds.RasterYSize
    b = ds.RasterCount
    prj = ds.GetProjection()
    gt = ds.GetGeoTransform()
    dtype = ds.GetRasterBand(1).DataType
    return w, h, b, prj, gt, dtype

def dtype_name(dt: int) -> str:
    return gdal.GetDataTypeName(dt)

# Helper: GDAL data type to numpy dtype
_numpy_from_gdal = {
    gdal.GDT_Byte: np.uint8,
    gdal.GDT_UInt16: np.uint16,
    gdal.GDT_Int16: np.int16,
    gdal.GDT_UInt32: np.uint32,
    gdal.GDT_Int32: np.int32,
    gdal.GDT_Float32: np.float32,
    gdal.GDT_Float64: np.float64,
}

def gdal_array_type_to_numpy(dt: int):
    return _numpy_from_gdal.get(dt, np.float32)

def is_unsigned(dt: int) -> bool:
    return dt in (gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_UInt32)

def choose_out_dtype(ref_dt: int) -> int:
    # If the first is unsigned integer, use Float32 to accommodate -9999; otherwise reuse the first dtype
    return gdal.GDT_Float32 if is_unsigned(ref_dt) else ref_dt

def map_src_nodata_to_target(arr: np.ndarray, src_nd) -> np.ndarray:
    out = arr
    if src_nd is None:
        return out
    try:
        # Treat NaN as NoData
        if isinstance(src_nd, float) and np.isnan(src_nd):
            mask = np.isnan(out)
        else:
            mask = (out == src_nd)
        if np.any(mask):
            out = out.copy()
            out[mask] = TARGET_NODATA
    except Exception:
        pass
    return out

def main():
    ap = argparse.ArgumentParser(description='Stack TIFs along the band axis into a multi-band image and emit the band order (unified NoData = -9999)')
    ap.add_argument('--input-dir', default='/mnt/raid/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x/distance_map_interpolated_margin_nodata_norm', help='Input directory')
    ap.add_argument('--output', default='/mnt/raid/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x/distance_maps_materials/distance_map_inter_norm_stack.tif', help='Output multi-band TIF path')
    ap.add_argument('--pattern', default='*.tif', help='File match pattern, default *.tif')
    ap.add_argument('--natural-sort', action='store_true', help='Use natural sort (numbers sorted by numeric value)')
    ap.add_argument('--save-map', action='store_true', help='Save the band mapping to a .bands.txt sidecar file')
    args = ap.parse_args()

    files = list_tifs(args.input_dir, args.pattern)
    if not files:
        logger.error(f'No matching TIFs in input directory: {args.input_dir}/{args.pattern}')
        return
    files = sorted(files, key=natural_key if args.natural_sort else str.lower)

    # Open the first file as reference
    ref_ds = gdal.Open(files[0], gdal.GA_ReadOnly)
    if ref_ds is None:
        logger.error(f'Cannot open reference file: {files[0]}')
        return
    ref_w, ref_h, ref_bands, ref_prj, ref_gt, ref_dtype = read_info(ref_ds)

    # Validate consistency and tally total bands
    total_bands = 0
    for fp in files:
        ds = gdal.Open(fp, gdal.GA_ReadOnly)
        if ds is None:
            logger.error(f'Cannot open: {fp}')
            return
        w, h, b, prj, gt, dt = read_info(ds)
        ds = None
        if (w != ref_w) or (h != ref_h):
            logger.error(f'Size mismatch: {os.path.basename(fp)} ({w}x{h}) != ({ref_w}x{ref_h})')
            return
        if prj != ref_prj:
            logger.error(f'Projection mismatch: {os.path.basename(fp)}')
            return
        if gt != ref_gt:
            logger.error(f'Affine mismatch: {os.path.basename(fp)}')
            return
        total_bands += b

    out_dtype = choose_out_dtype(ref_dtype)
    if out_dtype != ref_dtype:
        logger.info(f'Using Float32 output dtype to accommodate the unified NoData {TARGET_NODATA} (first input dtype was {dtype_name(ref_dtype)}）')

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        args.output,
        ref_w,
        ref_h,
        total_bands,
        out_dtype,
        options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256']
    )
    out_ds.SetProjection(ref_prj)
    out_ds.SetGeoTransform(ref_gt)

    # Record the mapping
    mapping_lines: List[str] = []
    out_band_idx = 1

    for fp in files:
        ds = gdal.Open(fp, gdal.GA_ReadOnly)
        w, h, b, prj, gt, dt = read_info(ds)
        if dt != out_dtype:
            logger.warning(f'Data type mismatch: {os.path.basename(fp)} {dtype_name(dt)} -> cast to {dtype_name(out_dtype)}')
        for bidx in range(1, b + 1):
            band = ds.GetRasterBand(bidx)
            arr = band.ReadAsArray()
            if arr is None:
                ds = None
                out_ds = None
                logger.error(f'Read failed: {fp} band {bidx}')
                return
            # Map source NoData to the unified NoData
            src_nd = band.GetNoDataValue()
            arr = map_src_nodata_to_target(arr, src_nd)
            # Unify dtype
            if band.DataType != out_dtype:
                arr = arr.astype(gdal_array_type_to_numpy(out_dtype))
            out_band = out_ds.GetRasterBand(out_band_idx)
            out_band.WriteArray(arr)
            # Set the target NoData uniformly
            out_band.SetNoDataValue(TARGET_NODATA)
            mapping_lines.append(f'Band {out_band_idx} <- {os.path.basename(fp)} [band {bidx}]')
            out_band_idx += 1
        ds = None

    out_ds.FlushCache()
    out_ds = None

    # Save mapping
    if args.save_map:
        map_path = args.output + '.bands.txt'
        with open(map_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(mapping_lines) + '\n')
        logger.info(f'Mapping saved: {map_path}')

    # Print the mapping to the console (briefly)
    logger.info(f'Total bands: {total_bands} -> {os.path.basename(args.output)}; unified NoData={TARGET_NODATA}')
    if len(mapping_lines) <= 20:
        for line in mapping_lines:
            logger.info(line)
    else:
        for line in mapping_lines[:10]:
            logger.info(line)
        logger.info('...')
        for line in mapping_lines[-10:]:
            logger.info(line)

if __name__ == '__main__':
    main()