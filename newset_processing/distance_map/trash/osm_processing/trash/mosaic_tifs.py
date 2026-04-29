"""Mosaic the two clipped distance-raster pieces"""

import argparse
import os
import sys
from glob import glob
from typing import List, Tuple, Dict

import numpy as np
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling


def list_cropped(path: str, suffix: str) -> Dict[str, str]:
    files = glob(os.path.join(path, f"*{suffix}"))
    return {os.path.basename(fp): fp for fp in files}


def choose_nodata(datasets: List[rasterio.io.DatasetReader], dtype: str):
    # Prefer A's nodata, then B's; if neither exists, use a dtype-specific default
    for ds in datasets:
        if ds.nodata is not None:
            return ds.nodata
    if np.issubdtype(np.dtype(dtype), np.floating):
        return -9999.0
    return 0


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def merge_pair(a_path: str, b_path: str, out_path: str):
    with rasterio.open(a_path) as a_ds, rasterio.open(b_path) as b_ds:
        # Use A's resolution (pixel size) as the baseline so the output stays on A's grid
        a_res = (abs(a_ds.transform.a), abs(a_ds.transform.e))
        dtype = a_ds.dtypes[0]
        nodata_value = choose_nodata([a_ds, b_ds], dtype)

        # If B's CRS differs from A's, reproject B to A's CRS via WarpedVRT first
        b_src = b_ds
        if b_ds.crs != a_ds.crs:
            b_src = WarpedVRT(
                b_ds,
                crs=a_ds.crs,
                resampling=Resampling.nearest,
                src_nodata=b_ds.nodata,
                dst_nodata=nodata_value,
            )

        # Data order [A, B] with method='last' => B overrides A in the overlap
        mosaic, out_transform = rio_merge(
            [a_ds, b_src],
            method="last",
            nodata=nodata_value,
            res=a_res,
        )

        profile = a_ds.profile.copy()
        profile.update(
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=out_transform,
            nodata=nodata_value,
            compress="lzw",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )
        # Add a predictor for float data
        if np.issubdtype(np.dtype(dtype), np.floating):
            profile["predictor"] = 3

        ensure_dir(os.path.dirname(out_path))
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mosaic)


def find_pairs(dir_a: str, dir_b: str, suffix: str) -> List[Tuple[str, str, str]]:
    a_map = list_cropped(dir_a, suffix)
    b_map = list_cropped(dir_b, suffix)
    names = sorted(set(a_map.keys()) & set(b_map.keys()))
    return [(name, a_map[name], b_map[name]) for name in names]


def parse_args():
    ap = argparse.ArgumentParser(
        description="Mosaic same-named *cropped.tif files from two folders by geographic position; B overrides A in overlaps"
    )
    ap.add_argument("--dir-a", default='/mnt/raid/user/wildfire_dataset/self_built_materials/osm_map/canada-distance', help="Folder A path")
    ap.add_argument("--dir-b", default='/mnt/raid/user/wildfire_dataset/self_built_materials/osm_map/us-west-distance', help="Folder B path")
    ap.add_argument("--out-dir", default='/mnt/raid/user/wildfire_dataset/self_built_materials/osm_map/distance_mosaic', help="Output folder")
    ap.add_argument("--suffix", default="cropped.tif", help="Filename suffix (default: cropped.tif)")
    return ap.parse_args()


def main():
    args = parse_args()
    pairs = find_pairs(args.dir_a, args.dir_b, args.suffix)

    total = len(pairs)
    if total == 0:
        print("No same-named image pairs ending with the specified suffix were found.")
        return

    for i, (name, a_path, b_path) in enumerate(pairs, 1):
        out_path = os.path.join(args.out_dir, name)
        # Single-line dynamic progress: i/n current file
        sys.stdout.write(f"\r[{i}/{total}] mosaicking {name} ...")
        sys.stdout.flush()
        merge_pair(a_path, b_path, out_path)

    sys.stdout.write("\n")
    print(f"Done. Output directory: {args.out_dir}")


if __name__ == "__main__":
    main()