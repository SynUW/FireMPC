#!/usr/bin/env python3
"""
Compute NDVI and EVI from MODIS multi-spectral reflectance and output a two-band GeoTIFF:
- Input images contain 4 bands (MODIS Band1, 2, 3, 7), named like yyyy_mm_dd.tif.
- MODIS band mapping: Band1=RED(620-670nm), Band2=NIR(841-876nm), Band3=BLUE(459-479nm), Band4=SWIR(2105-2155nm)
- Output two bands: Band1=NDVI, Band2=EVI. Saved to a new directory with the same filenames as the input.
- Configurable reflectance scale and offset: out = in * scale_factor + offset (default scale_factor=0.0001, offset=0).
- NoData pixels do not participate in computation; pixels with a zero denominator are set to NoData.
- Both NDVI and EVI are clipped to [-1, 1].
- Projection, resolution and size are preserved; LZW compression, tiled, BIGTIFF supported; block-wise parallel processing.

NDVI = (NIR - RED) / (NIR + RED)
EVI  = G * (NIR - RED) / (NIR + C1*RED - C2*BLUE + L)
Defaults: G=2.5, L=1.0, C1=6.0, C2=7.5

Example:
python3 compute_ndvi_evi.py \
  --input-dir /path/to/reflectance \
  --output-dir /path/to/indices \
  --scale-factor 0.0001 --offset 0.0 \
  --max-workers 16 --block-size 1024
"""

import os
import glob
import logging
import argparse
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from osgeo import gdal
import re

# Initialize GDAL exception handling
gdal.UseExceptions()

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('compute_ndvi_evi')

def list_tifs(folder: str, pattern: str) -> List[str]:
    # Match files in yyyy_mm_dd.tif format
    all_files = glob.glob(os.path.join(folder, "*.tif")) + glob.glob(os.path.join(folder, "*.TIF"))
    matching_files = []
    for f in all_files:
        basename = os.path.basename(f)
        # Check whether the filename matches the yyyy_mm_dd.tif format
        if re.match(r'^\d{4}_\d{1,2}_\d{1,2}\.tif$', basename, re.IGNORECASE):
            matching_files.append(f)
    return sorted(matching_files)


def read_info(ds) -> Tuple[int, int, int, str, tuple, int]:
    w = ds.RasterXSize
    h = ds.RasterYSize
    b = ds.RasterCount
    prj = ds.GetProjection()
    gt = ds.GetGeoTransform()
    dtype = ds.GetRasterBand(1).DataType
    return w, h, b, prj, gt, dtype


def process_one(
    src_path: str,
    dst_path: str,
    scale_factor: float,
    offset: float,
    evi_G: float,
    evi_L: float,
    evi_C1: float,
    evi_C2: float,
    block_size: int,
    default_nodata: float,
    overwrite: bool,
    input_mode: str = "auto",
) -> Tuple[str, bool]:
    try:
        # If the destination exists: skip by default; --overwrite forces overwrite
        if os.path.exists(dst_path):
            if not overwrite:
                return src_path, True
            try:
                os.remove(dst_path)
            except Exception:
                pass
        ds = gdal.Open(src_path, gdal.GA_ReadOnly)
        if ds is None:
            logger.error(f'Cannot open file: {src_path}')
            return src_path, False
        width, height, bands, projection, geotransform, _ = read_info(ds)
        if bands < 4:
            logger.error(f'Insufficient band count (>= 4 required); skipping: {src_path}')
            return src_path, False

        # Band mapping (MODIS): Band1=RED, Band2=NIR, Band3=BLUE, Band4=SWIR
        # For NDVI/EVI computation: RED=Band1, NIR=Band2, BLUE=Band3
        band_red = ds.GetRasterBand(1)
        band_nir = ds.GetRasterBand(2)
        band_blue = ds.GetRasterBand(3)

        # dtype strategy (input_mode):
        # - auto: decide based on the input dtype
        # - uint8: treat the input as uint8, first divide by 255.0 (no scale multiplied for integer)
        # - integer: treat the input as integer, cast directly to float (no scale multiplied)
        # - float: treat the input as float, apply scale_factor + offset
        dt = band_red.DataType
        if input_mode not in ("auto", "uint8", "integer", "float"):
            input_mode = "auto"
        if input_mode == "auto":
            is_integer_input = dt in (
                gdal.GDT_Byte,
                gdal.GDT_UInt16,
                gdal.GDT_Int16,
                gdal.GDT_UInt32,
                gdal.GDT_Int32,
            )
            is_uint8_input = (dt == gdal.GDT_Byte)
        else:
            is_integer_input = (input_mode in ("uint8", "integer"))
            is_uint8_input = (input_mode == "uint8")

        nd_red = band_red.GetNoDataValue()
        nd_nir = band_nir.GetNoDataValue()
        nd_blue = band_blue.GetNoDataValue()

        # Output dataset (2 bands)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(
            dst_path,
            width,
            height,
            2,
            gdal.GDT_Float32,
            options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256']
        )
        out_ds.SetProjection(projection)
        out_ds.SetGeoTransform(geotransform)

        nodata_out = default_nodata
        out_ds.GetRasterBand(1).SetNoDataValue(nodata_out)  # NDVI
        out_ds.GetRasterBand(2).SetNoDataValue(nodata_out)  # EVI

        for y in range(0, height, block_size):
            rows = min(block_size, height - y)
            for x in range(0, width, block_size):
                cols = min(block_size, width - x)
                r = band_red.ReadAsArray(x, y, cols, rows)
                n = band_nir.ReadAsArray(x, y, cols, rows)
                b = band_blue.ReadAsArray(x, y, cols, rows)
                if r is None or n is None or b is None:
                    continue

                # Valid mask (input NoData does not participate in computation)
                valid = np.ones(r.shape, dtype=bool)
                if nd_red is not None:
                    valid &= (r != nd_red)
                if nd_nir is not None:
                    valid &= (n != nd_nir)
                if nd_blue is not None:
                    valid &= (b != nd_blue)

                # Scale + offset to physical reflectance
                if is_integer_input:
                    rf = r.astype(np.float32)
                    nf = n.astype(np.float32)
                    bf = b.astype(np.float32)
                    if is_uint8_input:
                        rf = rf / np.float32(255.0)
                        nf = nf / np.float32(255.0)
                        bf = bf / np.float32(255.0)
                else:
                    rf = r.astype(np.float32) * scale_factor + offset
                    nf = n.astype(np.float32) * scale_factor + offset
                    bf = b.astype(np.float32) * scale_factor + offset

                # Compute NDVI and EVI; overflow / invalid / zero-denominator pixels remain NoData
                with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                    # NDVI
                    ndvi = np.full(r.shape, nodata_out, dtype=np.float32)
                    num_ndvi = (nf - rf).astype(np.float64)
                    denom_ndvi = (nf + rf).astype(np.float64)
                    mask_ndvi = valid & (denom_ndvi != 0) & np.isfinite(num_ndvi) & np.isfinite(denom_ndvi)
                    if np.any(mask_ndvi):
                        ndvi_val = (num_ndvi[mask_ndvi] / denom_ndvi[mask_ndvi]).astype(np.float32)
                        # Write only finite values
                        finite_mask = np.isfinite(ndvi_val)
                        ndvi[mask_ndvi] = ndvi[mask_ndvi]  # no-op to keep shape
                        ndvi_idx = np.where(mask_ndvi)
                        # Clip NDVI to [-1, 1]
                        ndvi_val_clipped = np.clip(ndvi_val[finite_mask], -1.0, 1.0)
                        ndvi[ndvi_idx[0][finite_mask], ndvi_idx[1][finite_mask]] = ndvi_val_clipped

                    # EVI
                    evi = np.full(r.shape, nodata_out, dtype=np.float32)
                    num_evi = (nf - rf).astype(np.float64)
                    denom_evi = (nf.astype(np.float64) + evi_C1 * rf.astype(np.float64) - evi_C2 * bf.astype(np.float64) + evi_L)
                    mask_evi = valid & (denom_evi != 0) & np.isfinite(num_evi) & np.isfinite(denom_evi)
                    if np.any(mask_evi):
                        evi_val = (evi_G * num_evi[mask_evi] / denom_evi[mask_evi]).astype(np.float32)
                        finite_mask_evi = np.isfinite(evi_val)
                        evi[mask_evi] = evi[mask_evi]
                        evi_idx = np.where(mask_evi)
                        # Clip EVI to [-1, 1]
                        evi_val_clipped = np.clip(evi_val[finite_mask_evi], -1.0, 1.0)
                        evi[evi_idx[0][finite_mask_evi], evi_idx[1][finite_mask_evi]] = evi_val_clipped

                out_ds.GetRasterBand(1).WriteArray(ndvi, xoff=x, yoff=y)
                out_ds.GetRasterBand(2).WriteArray(evi, xoff=x, yoff=y)

        out_ds.FlushCache()
        ds = None
        out_ds = None
        return src_path, True
    except Exception as e:
        logger.error(f'Processing failed {src_path}: {e}')
        return src_path, False


def build_output_name(src_basename: str) -> str:
    # Use the input filename directly for consistency
    return src_basename


def main():
    ap = argparse.ArgumentParser(description='Compute NDVI and EVI from MODIS multi-spectral reflectance and output a two-band GeoTIFF')
    ap.add_argument('--input-dir', default='/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled_5xdownsampled_merged_unifiedEntireCan_scaledInt/MCD09GA_b1237_QAapplied_spatialFilled_downsampled_merged', help='Input directory (containing yyyy_mm_dd.tif files)')
    ap.add_argument('--output-dir', default='/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled_5xdownsampled_merged_unifiedEntireCan_scaledInt/NDVIEVI_QAapplied_spatialFilled_downsampled_merged_float', help='Output directory (will be created)')
    ap.add_argument('--pattern', default='*.tif', help='Input match pattern, default: yyyy_mm_dd.tif format')
    ap.add_argument('--scale-factor', type=float, default=0.0001, help='Reflectance scale factor (raw * scale_factor + offset)')
    ap.add_argument('--offset', type=float, default=0, help='Reflectance offset')
    ap.add_argument('--default-nodata', type=float, default=-32768, help='Output NoData value')
    ap.add_argument('--evi-G', type=float, default=2.5)
    ap.add_argument('--evi-L', type=float, default=1.0)
    ap.add_argument('--evi-C1', type=float, default=6.0)
    ap.add_argument('--evi-C2', type=float, default=7.5)
    ap.add_argument('--block-size', type=int, default=1024, help='Block-wise processing size (pixels)')
    ap.add_argument('--max-workers', type=int, default=16, help='Number of parallel processes')
    ap.add_argument('--overwrite', action='store_true', help='Overwrite existing output files (default: skip existing)')

    args = ap.parse_args()

    tifs = list_tifs(args.input_dir, args.pattern)
    if not tifs:
        logger.error(f'No matching TIFs in input directory: {args.input_dir}/{args.pattern}')
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Pre-build the task list (resume mode: skip existing by default; --overwrite recomputes all)
    tasks: List[Tuple[str, str]] = []
    skipped = 0
    for src in tifs:
        base = os.path.basename(src)
        out_name = build_output_name(base)
        dst = os.path.join(args.output_dir, out_name)
        if (not args.overwrite) and os.path.exists(dst):
            skipped += 1
            continue
        tasks.append((src, dst))

    total_all = len(tifs)
    total_run = len(tasks)
    done = 0
    ok_cnt = 0

    def progress():
        print(f"\rProgress {done}/{total_run} success: {ok_cnt} skipped: {skipped}/{total_all}", end='', flush=True)

    if total_run == 0:
        logger.info(f'All exist; skipping {total_all} files')
        return

    # Print only once: by default assume input dtype is consistent within the same folder
    # If you suspect mixed dtypes, change process_one's input_mode back to auto (per-file adaptive, no printing)
    input_mode = "auto"
    try:
        probe_path = tasks[0][0] if tasks else tifs[0]
        pds = gdal.Open(probe_path, gdal.GA_ReadOnly)
        if pds is not None:
            pdt = pds.GetRasterBand(1).DataType
            dt_name = gdal.GetDataTypeName(pdt)
            if pdt == gdal.GDT_Byte:
                input_mode = "uint8"
                logger.info(f"input dtype={dt_name}: integer; do not multiply by scale; first divide by 255 (uint8), then compute NDVI/EVI (printed once only)")
            elif pdt in (gdal.GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32):
                input_mode = "integer"
                logger.info(f"input dtype={dt_name}: integer; do not multiply by scale; cast directly to float and compute NDVI/EVI (printed once only)")
            else:
                input_mode = "float"
                logger.info(f"input dtype={dt_name}: with scale_factor={args.scale_factor} offset={args.offset} compute reflectance, then NDVI/EVI (printed once only)")
        pds = None
    except Exception:
        input_mode = "auto"
        logger.info("Input dtype probe failed: fall back to auto strategy (no per-file printing)")

    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = []
        for src, dst in tasks:
            futs.append(ex.submit(
                process_one,
                src,
                dst,
                args.scale_factor,
                args.offset,
                args.evi_G,
                args.evi_L,
                args.evi_C1,
                args.evi_C2,
                args.block_size,
                args.default_nodata,
                args.overwrite,
                input_mode,
            ))
        for f in as_completed(futs):
            _, ok = f.result()
            done += 1
            if ok:
                ok_cnt += 1
            progress()

    print()
    logger.info(f'Done: newly computed {ok_cnt}/{total_run}; skipped {skipped}/{total_all} -> {args.output_dir}')


if __name__ == '__main__':
    main()