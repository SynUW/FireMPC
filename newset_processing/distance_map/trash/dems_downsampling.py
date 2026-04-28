#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
10x downsampling for DEM and distance maps (auto-detected from filename):
- Names containing "distance" (case-insensitive) -> use Resampling.min
- Names containing "dem"      (case-insensitive) -> use average / median / bilinear (default: average)
- Other files are skipped
- Single-band only (specify with --band); recursion optional

Notes:
- Uses "factor-scaled grid": new pixel size = original pixel size * factor (default 10)
- Preserves the original image's top-left origin and extent alignment (the last row / column may be partially covered; no extra padding)
- For target pixels whose window has a valid-pixel ratio below the threshold (e.g. 0.2), write NoData rather than a forced value
"""

import argparse
from pathlib import Path
import math
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject

def parse_args():
    ap = argparse.ArgumentParser("10× downsample for DEM & Distance rasters")
    ap.add_argument("--input-folder", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip/distance_map_interpolated_margin_nodata', help="Input directory")
    ap.add_argument("--output-folder", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x/distance_map_interpolated_margin_nodata_new', help="Output directory")
    ap.add_argument("--recursive", action="store_true", help="Recursively traverse subdirectories")
    ap.add_argument("--pattern", default="*.tif", help="Match pattern (default *.tif)")
    ap.add_argument("--band", type=int, default=1, help="Band to process (default 1)")

    ap.add_argument("--factor", type=int, default=2, help="Downsampling factor (default 10)")

    # DEM options
    ap.add_argument("--dem-method", choices=["average", "median", "bilinear"],
                    default="average", help="DEM downsampling method (default average)")

    # NoData and valid ratio
    ap.add_argument("--src-nodata", type=float, default=-32768, help="Source NoData (e.g. -32768)")
    ap.add_argument("--dst-nodata", type=float, default=-32768, help="Target NoData (e.g. -32768)")
    ap.add_argument("--valid-fraction", type=float, default=0.2,
                    help="Valid-pixel ratio threshold inside the window (default 0.2)")

    # Output write-out
    ap.add_argument("--compress", default="ZSTD", choices=["NONE", "LZW", "DEFLATE", "ZSTD"],
                    help="Compression (default ZSTD)")
    ap.add_argument("--blocksize", type=int, default=512, help="Tile block size (default 512)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite if it already exists")
    return ap.parse_args()

def pick_mode_from_name(name_lower: str):
    if "distance" in name_lower:
        return "distance"
    if "dem" in name_lower:
        return "dem"
    return None

def resampling_from_method(method: str) -> Resampling:
    if method == "average":  return Resampling.average
    if method == "median":   return Resampling.med
    if method == "bilinear": return Resampling.bilinear
    raise ValueError(method)

def build_x10_template(src, factor: int):
    """Build a 10x (or other factor) new grid from the original transform, preserving top-left origin and extent alignment."""
    t = src.transform
    # Original pixel size (note that pixel height is typically negative)
    xres = abs(t.a)
    yres = abs(t.e)

    new_xres = xres * factor
    new_yres = yres * factor

    # New size (rounding up covers the full extent; round up here to avoid losing the last rows/columns)
    new_w = math.ceil(src.width  / factor)
    new_h = math.ceil(src.height / factor)

    # Top-left origin unchanged; keep pixel height negative (north up)
    new_transform = Affine(new_xres, 0.0, t.c, 0.0, -new_yres, t.f)
    return new_transform, new_w, new_h, src.crs

def write_tif(path, arr, profile, compress="ZSTD", blocksize=512, nodata=None, overwrite=False):
    meta = profile.copy()
    meta.update(
        dtype=str(arr.dtype),
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        tiled=True,
        blockxsize=blocksize,
        blockysize=blocksize
    )
    if compress != "NONE":
        meta.update(compress=compress)
        if compress == "ZSTD":
            meta.update(zstd_level=9)
    if nodata is not None:
        meta.update(nodata=nodata)

    p = Path(path)
    if p.exists():
        if overwrite:
            p.unlink()
        else:
            raise FileExistsError(f"Output already exists (--overwrite not specified): {p}")

    with rasterio.open(p, "w", **meta) as dst:
        dst.write(arr, 1)

def downsample_by_factor(src, src_arr, src_nodata, tmpl_transform, tmpl_w, tmpl_h, mode, dem_method, valid_fraction, dst_nodata):
    """Reproject + aggregate: distance -> min; DEM -> average / median / bilinear; filter by valid-ratio threshold."""
    dst_shape = (tmpl_h, tmpl_w)

    # Valid-ratio map: average the valid mask (1/0) onto the target grid to get the valid ratio per coarse pixel
    valid_src = (src_arr != src_nodata) & (~np.isnan(src_arr))
    valid_src = valid_src.astype(np.uint8)
    valid_ratio = np.zeros(dst_shape, dtype=np.float32)

    reproject(
        source=valid_src,
        destination=valid_ratio,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=tmpl_transform,
        dst_crs=src.crs,
        resampling=Resampling.average,
        src_nodata=0,
        dst_nodata=0
    )

    # Main aggregation
    out = np.full(dst_shape, dst_nodata, dtype=np.float32)
    if mode == "distance":
        rs = Resampling.min
    else:
        rs = resampling_from_method(dem_method)

    reproject(
        source=src_arr,
        destination=out,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=tmpl_transform,
        dst_crs=src.crs,
        resampling=rs,
        src_nodata=src_nodata,
        dst_nodata=dst_nodata
    )

    # Valid-ratio filtering
    out[valid_ratio < float(valid_fraction)] = dst_nodata
    return out

def main():
    args = parse_args()
    in_dir = Path(args.input_folder)
    out_dir = Path(args.output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.rglob(args.pattern) if args.recursive else in_dir.glob(args.pattern))
    if not files:
        print("No matching TIFFs found."); return

    total = ok = skip = 0
    for f in files:
        mode = pick_mode_from_name(f.name.lower())
        if mode is None:
            skip += 1
            continue
        total += 1

        try:
            with rasterio.open(f) as src:
                arr = src.read(args.band).astype(np.float32, copy=False)

                # Unify source NoData
                if src.nodata is not None:
                    arr[np.isclose(arr, src.nodata, equal_nan=True)] = args.src_nodata
                arr[np.isnan(arr)] = args.src_nodata
                src_nodata = args.src_nodata

                # Build the 10x grid template
                tform, w, h, crs = build_x10_template(src, args.factor)

                out_arr = downsample_by_factor(
                    src=src,
                    src_arr=arr,
                    src_nodata=src_nodata,
                    tmpl_transform=tform,
                    tmpl_w=w,
                    tmpl_h=h,
                    mode=mode,
                    dem_method=args.dem_method,
                    valid_fraction=args.valid_fraction,
                    dst_nodata=args.dst_nodata
                )

                # Output path (preserves relative hierarchy)
                rel = f.relative_to(in_dir) if f.is_relative_to(in_dir) else Path(f.name)
                out_path = out_dir / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)

                profile = src.profile.copy()
                profile.update(
                    height=h, width=w, transform=tform, crs=crs,
                    nodata=args.dst_nodata, dtype="float32", count=1
                )
                write_tif(
                    out_path, out_arr, profile,
                    compress=args.compress, blocksize=args.blocksize,
                    nodata=args.dst_nodata, overwrite=args.overwrite
                )
                ok += 1
                print(f"[OK] {mode:<8} ×{args.factor} -> {out_path}")
        except Exception as e:
            print(f"[FAIL] {f} | {e}")

    print(f"Done. Processed: {total} | succeeded: {ok} | skipped (not dem / distance): {skip} | total scanned: {len(files)}")

if __name__ == "__main__":
    main()
