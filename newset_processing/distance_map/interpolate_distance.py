#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interpolate invalid values in a distance raster, filling only invalid pixels.
By default uses GDAL's FillNodata algorithm (follows neighboring-pixel patterns, with smoothing iterations);
supports batch processing of folders, multithreading, and compressed output.

Dependencies:
  - GDAL Python bindings: pip install GDAL  or use the system's osgeo.gdal
  - rasterio (optional, for more robust reads/writes; not required)

Example:
  python interpolate_distance_nodata.py \
      --input-folder /path/to/in_tifs \
      --output-folder /path/to/out_tifs \
      --nodata -9999 \
      --search-radius 50 \
      --smooth-iterations 2 \
      --threads 8 \
      --compress ZSTD \
      --blocksize 512
"""

import os
import sys
import math
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from osgeo import gdal
except ImportError as e:
    print("GDAL Python bindings (osgeo.gdal) not found. Install with: pip install GDAL, or use the system GDAL.")
    raise

# Optional: prefer rasterio for output (easier compression/block-size config); fall back to GDAL if absent
try:
    import rasterio
    from rasterio.enums import Resampling
    RASTERIO_OK = True
except Exception:
    RASTERIO_OK = False


def parse_args():
    ap = argparse.ArgumentParser(description="Interpolate the NoData regions of a distance raster (fill only invalid pixels)")
    ap.add_argument("--input-folder", default='/mnt/raid/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip/distance_map', help="Input folder containing .tif files to interpolate")
    ap.add_argument("--output-folder", default='/mnt/raid/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip/distance_map_interpolated_margin_nodata', help="Output folder")
    ap.add_argument("--nodata", default='-32768',
                    help="Invalid value (e.g. -9999, -32768, or 'nan')")
    ap.add_argument("--search-radius", type=float, default=50.0,
                    help="Maximum search radius (pixels); maxSearchDist for GDAL FillNodata; default 50")
    ap.add_argument("--smooth-iterations", type=int, default=2,
                    help="Smoothing iterations (smoothingIterations); default 2; 0 disables smoothing")
    ap.add_argument("--threads", type=int, default=8, help="Number of parallel threads, default 8")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite if the output already exists")
    ap.add_argument("--compress", default="ZSTD", choices=["NONE", "LZW", "ZSTD", "DEFLATE"],
                    help="Output compression codec, default ZSTD")
    ap.add_argument("--blocksize", type=int, default=512, help="Tile block size, default 512")
    ap.add_argument("--keep-dtype", action="store_true",
                    help="Preserve the original dtype if possible (default). When disabled, write uniformly as float32")
    ap.add_argument("--gdal-only", action="store_true",
                    help="Force writes via GDAL only (do not depend on rasterio)")
    return ap.parse_args()


def str_to_nodata(nodata_str: str):
    """Parse the command-line nodata string into a number or NaN."""
    if nodata_str.lower() == "nan":
        return float("nan")
    try:
        return float(nodata_str)
    except ValueError:
        raise ValueError(f"Cannot parse nodata='{nodata_str}'; please enter a number or 'nan'.")


def build_creation_options(compress: str, blocksize: int):
    co = []
    if compress and compress.upper() != "NONE":
        co.append(f"COMPRESS={compress.upper()}")
        # ZSTD: a quality level is recommended
        if compress.upper() == "ZSTD":
            co.append("ZSTD_LEVEL=9")
        if compress.upper() == "DEFLATE":
            co.append("PREDICTOR=2")
    if blocksize:
        co.append(f"TILED=YES")
        co.append(f"BLOCKXSIZE={blocksize}")
        co.append(f"BLOCKYSIZE={blocksize}")
    return co


def fill_nodata_gdal(in_path: Path, out_path: Path, user_nodata, max_search_dist: float,
                     smooth_iters: int, keep_dtype: bool, creation_options):
    """
    Use GDAL.FillNodata to fill only the nodata pixels, preserving georeferencing, projection, and metadata as much as possible.
    """
    # Open the source data with GDAL
    src_ds = gdal.Open(str(in_path), gdal.GA_ReadOnly)
    if src_ds is None:
        raise RuntimeError(f"Cannot open input image: {in_path}")

    band = src_ds.GetRasterBand(1)
    xsize = src_ds.RasterXSize
    ysize = src_ds.RasterYSize
    geotransform = src_ds.GetGeoTransform(can_return_null=True)
    projection = src_ds.GetProjectionRef()
    src_nodata = band.GetNoDataValue()

    # Decide which nodata to use: prefer the user's value; if the user said NaN and the source already has nodata, keep the source's value
    if isinstance(user_nodata, float) and math.isnan(user_nodata):
        # User input NaN: use the source's nodata if present; otherwise use the NaN flow (cast to float32 and treat NaN as nodata)
        use_nan = True if src_nodata is None else False
        nodata_value = src_nodata if src_nodata is not None else None
    else:
        use_nan = False
        nodata_value = float(user_nodata)

    # Determine output dtype
    gdal_dtype = band.DataType
    if not keep_dtype:
        # Uniform float32 (GDAL: gdal.GDT_Float32)
        gdal_dtype = gdal.GDT_Float32

    # Create output
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(str(out_path), xsize, ysize, 1, gdal_dtype, creation_options)
    if out_ds is None:
        raise RuntimeError(f"Cannot create output image: {out_path}")

    if geotransform:
        out_ds.SetGeoTransform(geotransform)
    if projection:
        out_ds.SetProjection(projection)

    out_band = out_ds.GetRasterBand(1)

    # Read the entire image (GDAL FillNodata needs to operate on the destination band)
    arr = band.ReadAsArray()
    if arr is None:
        out_ds = None
        src_ds = None
        raise RuntimeError(f"Failed to read data: {in_path}")

    import numpy as np

    # Write data to the output band and uniformly set nodata
    if use_nan:
        # User specified nan; if the source has no nodata, treat the user's "invalid pixels" as np.nan
        # Case 1: source has nodata; use the source's nodata as the band's nodata
        if nodata_value is not None:
            out_band.SetNoDataValue(nodata_value)
            arr_f = arr.astype("float32", copy=False)
        else:
            # Case 2: source has no nodata; use NaN as a placeholder (note: band nodata cannot be set to NaN directly)
            # Use NaN temporarily for algorithm recognition; after filling, write back (without setting a band nodata value)
            arr_f = arr.astype("float32", copy=False)
            # In this branch the user said nan, so we'd need to detect originally-missing-nodata, which is not possible
            # Therefore, only existing NaN (if the source data has it) is handled here; otherwise FillNodata cannot identify "which pixels are invalid".
            # For safety: if the source has no nodata and no NaN, no filling is needed (just write directly)
            if not np.isnan(arr_f).any():
                # No NaN, nothing to fill
                out_band.WriteArray(arr_f)
                out_band.FlushCache()
                out_ds = None
                src_ds = None
                return
    else:
        # User specified an explicit numeric nodata
        out_band.SetNoDataValue(nodata_value)
        arr_f = arr.astype("float32", copy=False)
        # Keep nodata pixels as the nodata value (if the source's nodata differs, unify to the user's nodata)
        if src_nodata is not None and src_nodata != nodata_value:
            arr_f[arr == src_nodata] = nodata_value

    out_band.WriteArray(arr_f)
    out_band.FlushCache()

    # Fill with GDAL.FillNodata
    # Interpolate only the nodata pixels of the destination band; maskBand is optional and not supplied here.
    # maxSearchDist unit: pixels; smoothingIterations: number of smoothing iterations
    gdal.ErrorReset()
    err = gdal.FillNodata(targetBand=out_band,
                          maskBand=None,
                          maxSearchDist=max_search_dist,
                          smoothingIterations=smooth_iters)
    if err != 0:
        # gdal.FillNodata returns 0 on success, non-zero is failure
        out_ds = None
        src_ds = None
        raise RuntimeError(f"GDAL.FillNodata failed (code {err}）：{in_path}")

    out_band.FlushCache()
    out_ds.FlushCache()

    # Cleanup
    out_ds = None
    src_ds = None


def write_with_rasterio_like(in_path: Path, tmp_filled_path: Path, out_path: Path,
                             compress: str, blocksize: int, keep_dtype: bool):
    """
    If rasterio is enabled, the GDAL-filled result can be re-written normalized (to ensure consistent compression, tiling, etc.).
    The implementation here is: read tmp_filled_path and write as-is to out_path (with compression / block size).
    """
    if not RASTERIO_OK:
        # Rename (or copy) the temp file as the final output directly
        if out_path.exists():
            out_path.unlink()
        tmp_filled_path.replace(out_path)
        return

    import numpy as np
    co = {"compress": None} if compress == "NONE" else {"compress": compress.lower()}
    blocksize = int(blocksize) if blocksize else 512

    with rasterio.open(tmp_filled_path, "r") as src:
        meta = src.meta.copy()
        # Force tiling
        meta.update(
            tiled=True,
            blockxsize=blocksize,
            blockysize=blocksize,
            **co
        )
        if not keep_dtype:
            meta.update(dtype="float32")
        # Delete old file
        if out_path.exists():
            out_path.unlink()
        with rasterio.open(out_path, "w", **meta) as dst:
            for ji, window in dst.block_windows(1):
                data = src.read(1, window=window)
                if not keep_dtype:
                    data = data.astype("float32", copy=False)
                dst.write(data, 1, window=window)
            # Inherit nodata
            nd = src.nodata
            if nd is not None:
                dst.nodata = nd


def process_one(in_path: Path,
                out_path: Path,
                nodata_value,
                search_radius: float,
                smooth_iterations: int,
                keep_dtype: bool,
                creation_options,
                use_rasterio_writer: bool):
    """
    Process a single file:
      1) Fill with GDAL.FillNodata (NoData pixels only)
      2) Optionally: write a normalized compressed / blocked output via rasterio
    """
    tmp_path = out_path.with_suffix(".tmp.tif")

    # Run filling on a temp file first
    # Using GDAL's CreateCopy then FillNodata is slow here; do Create->Write->Fill directly.
    # For simplicity, write to tmp_path first, then normalize to out_path as needed.
    # -- For convenience: treat tmp as the final output, then "normalize copy / rewrite" at the end.
    fill_nodata_gdal(in_path=in_path,
                     out_path=tmp_path,
                     user_nodata=nodata_value,
                     max_search_dist=search_radius,
                     smooth_iters=smooth_iterations,
                     keep_dtype=keep_dtype,
                     creation_options=creation_options)

    # "Normalized write" as needed
    write_with_rasterio_like(in_path=in_path,
                             tmp_filled_path=tmp_path,
                             out_path=out_path,
                             compress=creation_options_dict(creation_options).get("COMPRESS", "NONE"),
                             blocksize=int(creation_options_dict(creation_options).get("BLOCKXSIZE", 512)),
                             keep_dtype=keep_dtype)

    # Clean up tmp (the rasterio-write branch replaces it internally, so it no longer exists; the GDAL-only branch renames directly)
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except Exception:
            pass


def creation_options_dict(co_list):
    """Convert a GDAL creation-options list to a dict for easier reading."""
    d = {}
    for kv in co_list:
        if "=" in kv:
            k, v = kv.split("=", 1)
            d[k.upper()] = v
        else:
            d[kv.upper()] = True
    return d


def main():
    args = parse_args()
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    nodata_value = str_to_nodata(args.nodata)
    creation_options = build_creation_options(args.compress, args.blocksize)

    # Iterate over input files
    tifs = sorted([p for p in input_folder.glob("*.tif")])
    if not tifs:
        print(f"No .tif found in the input directory: {input_folder}")
        sys.exit(1)

    tasks = []
    for tif in tifs:
        out_path = output_folder / tif.name
        if out_path.exists() and not args.overwrite:
            continue
        tasks.append((tif, out_path))

    total = len(tasks)
    skipped = len(tifs) - total
    print(f"Pending: {total} items; skipped (existing and not overwritten): {skipped} items")

    use_rasterio_writer = (RASTERIO_OK and not args.gdal_only)

    ok = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futures = []
        for in_path, out_path in tasks:
            futures.append(ex.submit(
                process_one,
                in_path,
                out_path,
                nodata_value,
                args.search_radius,
                args.smooth_iterations,
                args.keep_dtype,
                creation_options,
                use_rasterio_writer
            ))

        for f in as_completed(futures):
            try:
                f.result()
                ok += 1
                if (ok + fail) % 10 == 0:
                    print(f"[Progress] succeeded: {ok} | failed: {fail} / {total}")
            except Exception as e:
                fail += 1
                print(f"[FAIL] {e}")

    print(f"Done. Succeeded: {ok} | failed: {fail} | skipped: {skipped} | total: {len(tifs)}")


if __name__ == "__main__":
    main()
