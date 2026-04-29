#!/usr/bin/env python3
"""
Single-file NoData unification tool:
- If any band has NoData at a pixel, set every band at that pixel to the target NoData (default -9999).
- Output TIF preserves input size / projection / affine; band count is unchanged; every output band's NoData is set to the target value.
- If the input dtype is an unsigned integer (Byte / UInt16 / UInt32), the output is upgraded to Float32 to accommodate -9999; otherwise the first input band's dtype is reused.
- Block-wise processing, LZW compression, tiled, BIGTIFF.

Example:
python3 homo_nodata.py \
  --src /path/to/in.tif \
  --dst /path/to/out.tif \
  --target-nodata -9999 \
  --block-size 1024
"""

import os
import logging
import argparse
from typing import Tuple

import numpy as np
from osgeo import gdal

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('homo_nodata_single')


def read_info(ds) -> Tuple[int, int, int, str, tuple, int]:
    w = ds.RasterXSize
    h = ds.RasterYSize
    b = ds.RasterCount
    prj = ds.GetProjection()
    gt = ds.GetGeoTransform()
    dtype = ds.GetRasterBand(1).DataType
    return w, h, b, prj, gt, dtype

_numpy_from_gdal = {
    gdal.GDT_Byte: np.uint8,
    gdal.GDT_UInt16: np.uint16,
    gdal.GDT_Int16: np.int16,
    gdal.GDT_UInt32: np.uint32,
    gdal.GDT_Int32: np.int32,
    gdal.GDT_Float32: np.float32,
    gdal.GDT_Float64: np.float64,
}


def npdtype(dt: int):
    return _numpy_from_gdal.get(dt, np.float32)


def is_unsigned(dt: int) -> bool:
    return dt in (gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_UInt32)


def choose_out_dtype(ref_dt: int) -> int:
    # When the target NoData is -9999 and the input is an unsigned integer, upgrade to Float32
    return gdal.GDT_Float32 if is_unsigned(ref_dt) else ref_dt


def band_nodata_mask(arr: np.ndarray, nd_val) -> np.ndarray:
    if nd_val is None:
        return np.zeros(arr.shape, dtype=bool)
    if isinstance(nd_val, float) and np.isnan(nd_val):
        return np.isnan(arr)
    return arr == nd_val


def homogenize_single(src_path: str, dst_path: str, target_nd: float, block_size: int) -> bool:
    ds = gdal.Open(src_path, gdal.GA_ReadOnly)
    if ds is None:
        logger.error(f'Cannot open: {src_path}')
        return False
    width, height, bands, prj, gt, ref_dt = read_info(ds)
    out_dt = choose_out_dtype(ref_dt)
    out_np = npdtype(out_dt)

    os.makedirs(os.path.dirname(dst_path) or '.', exist_ok=True)
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        dst_path, width, height, bands, out_dt,
        options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256']
    )
    out_ds.SetProjection(prj)
    out_ds.SetGeoTransform(gt)
    for b in range(1, bands + 1):
        out_ds.GetRasterBand(b).SetNoDataValue(float(target_nd))

    # Pre-read per-band NoData
    band_nd = [ds.GetRasterBand(b).GetNoDataValue() for b in range(1, bands + 1)]

    # Block-wise
    for y in range(0, height, block_size):
        rows = min(block_size, height - y)
        for x in range(0, width, block_size):
            cols = min(block_size, width - x)
            blocks = []
            masks = []
            for b in range(1, bands + 1):
                arr = ds.GetRasterBand(b).ReadAsArray(x, y, cols, rows)
                if arr is None:
                    continue
                blocks.append(arr)
                masks.append(band_nodata_mask(arr, band_nd[b - 1]))
            if not blocks:
                continue
            any_nd = np.zeros(blocks[0].shape, dtype=bool)
            for m in masks:
                any_nd |= m
            for idx in range(bands):
                arr = blocks[idx].astype(out_np, copy=False)
                if np.any(any_nd):
                    arr = arr.copy()
                    arr[any_nd] = target_nd
                out_ds.GetRasterBand(idx + 1).WriteArray(arr, xoff=x, yoff=y)

    out_ds.FlushCache()
    ds = None
    out_ds = None
    return True


def main():
    ap = argparse.ArgumentParser(description='Single-file NoData unification: if any band has NoData at a pixel, set every band at that pixel to the target NoData')
    ap.add_argument('--src', default='/mnt/raid/user/wildfire_dataset/self_built_materials/all_data_masked/DEM_and_distance_map/Topo_and_distance_map_stack.tif', help='Input multi-band TIF')
    ap.add_argument('--dst', default='/mnt/raid/user/wildfire_dataset/self_built_materials/all_data_masked/DEM_and_distance_map/Topo_and_distance_map_stack_nodata_homogenized.tif', help='Output TIF path')
    ap.add_argument('--target-nodata', type=float, default=-9999.0, help='Unified output NoData value, default -9999')
    ap.add_argument('--block-size', type=int, default=1024, help='Block-wise processing size (pixels)')
    args = ap.parse_args()

    ok = homogenize_single(args.src, args.dst, args.target_nodata, args.block_size)
    if ok:
        logger.info(f'Done: {os.path.basename(args.src)} -> {args.dst} (NoData={args.target_nodata})')
    else:
        logger.error('Processing failed')


if __name__ == '__main__':
    main()