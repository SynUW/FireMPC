#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import argparse
import warnings
from pathlib import Path
from multiprocessing import Pool

warnings.filterwarnings("ignore", category=UserWarning)

# GDAL / RasterIO thread tuning
os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.shutil import copy as rio_copy
from tqdm import tqdm


def list_tifs(folder: Path):
    exts = (".tif", ".tiff")
    return {p.stem: p for p in sorted(folder.rglob("*")) if p.suffix.lower() in exts and p.is_file()}


def grids_compatible(A: rasterio.DatasetReader, B: rasterio.DatasetReader) -> bool:
    try:
        return (A.crs == B.crs) and (A.transform == B.transform) and (A.width == B.width) and (A.height == B.height)
    except Exception:
        return False


def choose_resample(dtype, force=None, auto=True):
    if force is not None:
        m = {
            "nearest": Resampling.nearest,
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
            "average": Resampling.average,
            "mode": Resampling.mode,
        }
        return m.get(force, Resampling.nearest)
    if not auto:
        return Resampling.nearest
    return Resampling.nearest if np.issubdtype(dtype, np.integer) else Resampling.bilinear


def band_valid_mask(ds: rasterio.DatasetReader, band_idx: int, arr: np.ndarray, window=None) -> np.ndarray:
    """
    True = valid.
    Build a mask matching arr's window and shape:
      - Initial mask = read_masks(window) > 0 (all True on failure)
      - If ds.nodata is not None, intersect with (arr != nodata)
    """
    try:
        mask = ds.read_masks(band_idx, window=window) > 0
        if mask.shape != arr.shape:
            # Shapes inconsistent (e.g. the driver returned the full mask) -> safe fallback: do not replace
            return np.ones_like(arr, dtype=bool)
    except Exception:
        mask = np.ones_like(arr, dtype=bool)

    nd = ds.nodata
    if nd is not None:
        try:
            valid_nd = (arr != nd)
            if valid_nd.shape != arr.shape:
                return np.ones_like(arr, dtype=bool)
            mask = mask & valid_nd
        except Exception:
            # Any exception -> do not replace
            return np.ones_like(arr, dtype=bool)
    return mask


def reproject_to_template(src: rasterio.DatasetReader, tmpl: rasterio.DatasetReader,
                          resampling: Resampling):
    """Reproject src to the tmpl grid; return a (count, H, W) array."""
    H, W = tmpl.height, tmpl.width
    out = np.zeros((src.count, H, W), dtype=np.dtype(src.dtypes[0]))
    dst_nd = tmpl.nodata if tmpl.nodata is not None else src.nodata
    for b in range(1, src.count + 1):
        src_arr = src.read(b)
        out_band = out[b - 1]
        if dst_nd is not None:
            out_band.fill(dst_nd)
        reproject(
            source=src_arr,
            destination=out_band,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=tmpl.transform,
            dst_crs=tmpl.crs,
            resampling=resampling,
            src_nodata=src.nodata,
            dst_nodata=dst_nd,
        )
    return out


def fuse_pair(b_path: Path,
              a_path: Path,
              out_path: Path,
              suffix: str = "",
              force_resample: bool = False,
              resample_policy: str | None = None,
              resample_auto: bool = True,
              copy_when_b_missing: bool = True) -> tuple[str, bool, str]:
    """
    Fill B's invalid pixels with A (per band). On exception or mismatch, fall back to "smaller is invalid -> just copy the larger one" without replacement.
    Returns: (basename, success, message)
    """
    name = (b_path or a_path).stem

    if b_path is None:
        if not copy_when_b_missing or a_path is None:
            return name, False, "B missing and A missing or copy disabled"
        out_path = out_path.with_name(out_path.stem + suffix + out_path.suffix)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rio_copy(a_path, out_path, driver="GTiff", copy_src_overviews=True)
        return name, True, "copied A (B missing)"

    with rasterio.open(b_path) as B:
        profile = B.profile.copy()
        profile.update(
            driver="GTiff",
            tiled=True,
            compress="LZW",
            BIGTIFF="IF_SAFER",
            num_threads="ALL_CPUS",
        )
        # predictor
        if np.issubdtype(np.dtype(B.dtypes[0]), np.floating):
            profile.update(predictor=3)
        else:
            profile.update(predictor=2)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        dst_fn = out_path.with_name(out_path.stem + (suffix or "") + ".tif")

        if a_path is None:
            rio_copy(b_path, dst_fn, driver="GTiff", copy_src_overviews=True)
            return name, True, "copied B (A missing)"

        with rasterio.open(a_path) as A:
            need_reproj = force_resample or (not grids_compatible(A, B))
            if need_reproj:
                resampling = choose_resample(np.dtype(A.dtypes[0]), force=resample_policy, auto=resample_auto)
                A_on_B = reproject_to_template(A, B, resampling=resampling)  # (count,H,W)
                # nodata used for the validity check
                a_nd = B.nodata if B.nodata is not None else A.nodata

            with rasterio.open(dst_fn, "w", **profile) as dst:
                if B.nodata is not None:
                    dst.nodata = B.nodata

                # Per band
                for b in range(1, B.count + 1):
                    # Read B
                    out_band = np.zeros((B.height, B.width), dtype=np.dtype(B.dtypes[b - 1]))

                    # Window-by-window, avoiding loading the whole image at once
                    for ji, win in B.block_windows(b):
                        try:
                            b_arr = B.read(b, window=win)
                            b_valid = band_valid_mask(B, b, b_arr, window=win)
                        except Exception:
                            # Any exception: treat B's window as valid and write back directly
                            out_band[win.row_off:win.row_off+win.height,
                                     win.col_off:win.col_off+win.width] = B.read(b, window=win)
                            continue

                        # Prepare A's same-window data and valid mask
                        if need_reproj:
                            try:
                                a_arr = A_on_B[b - 1][win.row_off:win.row_off+win.height,
                                                      win.col_off:win.col_off+win.width]
                                if a_arr.shape != b_arr.shape:
                                    # Shape mismatch: do not replace
                                    fill_mask = np.zeros_like(b_arr, dtype=bool)
                                else:
                                    if a_nd is not None:
                                        a_valid = (a_arr != a_nd)
                                    else:
                                        # No nodata: conservative, do not use A to replace
                                        a_valid = np.zeros_like(a_arr, dtype=bool)
                                    fill_mask = (~b_valid) & a_valid
                            except Exception:
                                # Any exception: do not replace
                                fill_mask = np.zeros_like(b_arr, dtype=bool)
                                a_arr = b_arr  # Avoid undefined
                        else:
                            # Grids match: read the same window directly
                            try:
                                a_arr = A.read(b, window=win)
                                if a_arr.shape != b_arr.shape:
                                    # Shape mismatch: do not replace (treat the smaller as invalid)
                                    fill_mask = np.zeros_like(b_arr, dtype=bool)
                                else:
                                    a_valid = band_valid_mask(A, b, a_arr, window=win)
                                    # Do one more defensive shape check
                                    if a_valid.shape != b_arr.shape:
                                        fill_mask = np.zeros_like(b_arr, dtype=bool)
                                    else:
                                        fill_mask = (~b_valid) & a_valid
                            except Exception:
                                fill_mask = np.zeros_like(b_arr, dtype=bool)
                                a_arr = b_arr

                        # Compose: copy B by default; replace with A only on pixels where fill_mask is True
                        out_slice = b_arr.copy()
                        if fill_mask.any():
                            out_slice[fill_mask] = a_arr[fill_mask]

                        out_band[win.row_off:win.row_off+win.height,
                                 win.col_off:win.col_off+win.width] = out_slice

                    dst.write(out_band, b)

        return name, True, "fused"


def _worker(args):
    return fuse_pair(*args)


def main():
    ap = argparse.ArgumentParser(description="Fill B (MOD)'s invalid pixels with A (MYD) (per band, robust-fallback version)")
    ap.add_argument("--b_dir", default='/mnt/raid/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic/MOD11A2_mosaic', help="B base directory (MOD)")
    ap.add_argument("--a_dir", default='/mnt/raid/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic/MYD11A2_mosaic', help="A fill-source directory (MYD)")
    ap.add_argument("--out_dir", default='/mnt/raid/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic/MCD11A2_mosaic', help="Output directory")
    ap.add_argument("--suffix", default="", help="Output filename suffix (e.g. _fused)")
    ap.add_argument("--workers", type=int, default=4, help="Number of parallel processes")
    ap.add_argument("--resample_auto", action="store_true", help="Auto-select resampling (integer = nearest; float = bilinear)")
    ap.add_argument("--resample_policy", default=None,
                    choices=[None, "nearest", "bilinear", "cubic", "average", "mode"],
                    help="Force a resampling strategy")
    ap.add_argument("--force_resample", action="store_true", help="Resample A -> B even when grids match")
    ap.add_argument("--copy_when_b_missing", action="store_true", help="When B is missing and A exists, copy A to the output")
    args = ap.parse_args()

    b_dir = Path(args.b_dir)
    a_dir = Path(args.a_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    b_map = list_tifs(b_dir)
    a_map = list_tifs(a_dir)
    keys = sorted(set(b_map.keys()) | set(a_map.keys()))

    tasks = []
    for k in keys:
        b_path = b_map.get(k)
        a_path = a_map.get(k)
        out_path = out_dir / f"{k}.tif"
        tasks.append((
            b_path,
            a_path,
            out_path,
            args.suffix,
            args.force_resample,
            args.resample_policy,
            args.resample_auto,
            args.copy_when_b_missing
        ))

    with Pool(processes=max(1, args.workers)) as pool:
        results = []
        for r in tqdm(pool.imap_unordered(_worker, tasks), total=len(tasks), desc="Fusing"):
            results.append(r)

    ok = sum(1 for _, s, _ in results if s)
    fail = len(results) - ok
    print(f"\nDone. Success: {ok}, Failed: {fail}")
    for name, s, msg in results:
        if not s:
            print(f"[FAIL] {name}: {msg}")


if __name__ == "__main__":
    main()
