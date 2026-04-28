#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iterate over all GeoTIFFs in a directory and compute per-band statistics:
- Global min/max (ignoring invalid values)
- 0.1% and 99.9% percentiles (ignoring invalid values)

Implementation details:
- Two passes:
  1) Compute per-band global min/max
  2) Build a fixed-bins histogram over [min, max], accumulate counts, then derive percentiles from the cumulative counts
- Block-wise reads (rasterio.block_windows) to avoid loading the entire image at once
- Invalid values: prefer the file's band.nodata; --default-nodata can specify default invalid values (multiple allowed); NaN is always ignored
"""

import sys
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from osgeo import gdal
from tqdm import tqdm
import multiprocessing as mp


def list_tifs(root: Path, recursive: bool = True) -> List[Path]:
    if recursive:
        return sorted(list(root.rglob('*.tif')) + list(root.rglob('*.TIF')))
    return sorted(list(root.glob('*.tif')) + list(root.glob('*.TIF')))


def is_valid(data: np.ndarray, nodata: Optional[List[float]]) -> np.ndarray:
    """
    Returns True for valid pixels.
    Optimizations:
    - When there is only one common nodata value, use a direct != comparison to avoid the overhead of np.isin.
    - NaN/Inf are always excluded (np.isfinite).
    """
    data = np.asarray(data)
    mask = np.isfinite(data)
    if not nodata:
        return mask

    # Clean the nodata list: drop NaN (already handled by isfinite)
    nd = [v for v in nodata if not (isinstance(v, float) and math.isnan(v))]
    if not nd:
        return mask
    if len(nd) == 1:
        v = nd[0]
        # For integer rasters, nodata is usually an integer; direct comparison also works for float rasters
        mask &= (data != v)
        return mask
    mask &= ~np.isin(data, np.asarray(nd, dtype=data.dtype))
    return mask


def _gdal_pass1_one(tif_path: Path, default_nodata: Optional[List[float]]) -> Tuple[int, List[float], List[float]]:
    band_count: Optional[int] = None
    global_min: List[float] = []
    global_max: List[float] = []
    ds = gdal.Open(str(tif_path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Cannot open file: {tif_path}")
    band_count = ds.RasterCount
    global_min = [math.inf] * band_count
    global_max = [-math.inf] * band_count
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    for bidx in range(1, band_count + 1):
        band = ds.GetRasterBand(bidx)
        nodata = band.GetNoDataValue()
        if nodata is None:
            nodata_list = default_nodata
        else:
            nodata_list = [float(nodata)]
        bx, by = band.GetBlockSize()
        if bx is None or bx <= 0:
            bx = 256
        if by is None or by <= 0:
            by = 256
        for yoff in range(0, ysize, by):
            ywin = min(by, ysize - yoff)
            for xoff in range(0, xsize, bx):
                xwin = min(bx, xsize - xoff)
                arr = band.ReadAsArray(xoff, yoff, xwin, ywin)
                if arr is None:
                    continue
                arr = np.asarray(arr)
                valid = is_valid(arr, nodata_list)
                if not valid.any():
                    continue
                # Avoid the large copy from arr[valid]: use np.where directly
                vmin = float(np.amin(arr, where=valid, initial=np.inf))
                vmax = float(np.amax(arr, where=valid, initial=-np.inf))
                if vmin < global_min[bidx - 1]:
                    global_min[bidx - 1] = vmin
                if vmax > global_max[bidx - 1]:
                    global_max[bidx - 1] = vmax
    ds = None
    return band_count, global_min, global_max


def _gdal_pass1_one_star(args: Tuple[Path, Optional[List[float]]]):
    """Adapter for imap arguments to avoid pickle errors caused by lambdas"""
    return _gdal_pass1_one(*args)


def pass1_min_max(tif_paths: List[Path], default_nodata: Optional[List[float]], workers: int) -> Tuple[int, List[float], List[float]]:
    """Returns: (band_count, mins, maxs) -- mins/maxs are lists indexed by band (length = band count)"""
    band_count: Optional[int] = None
    global_min: List[float] = []
    global_max: List[float] = []

    # Global GDAL configuration (threads / cache)
    gdal.UseExceptions()
    gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')
    gdal.SetCacheMax(1024 * 1024 * 1024)

    # Parallelize: each process handles a subset of files and returns local min/max; the main process reduces
    results = []
    with mp.Pool(processes=max(1, workers)) as pool:
        args_iter = ((p, default_nodata) for p in tif_paths)
        for res in tqdm(
            pool.imap_unordered(_gdal_pass1_one_star, args_iter),
            total=len(tif_paths), desc="Pass1 min/max", ncols=100, unit="file"
        ):
            results.append(res)

    # Reduce
    for idx, (bc, mins_part, maxs_part) in enumerate(results):
        if band_count is None:
            band_count = bc
            global_min = [math.inf] * band_count
            global_max = [-math.inf] * band_count
        elif bc != band_count:
            raise RuntimeError(f"Band counts differ: first is {band_count}; one file returned {bc}")
        for i in range(band_count):
            if math.isfinite(mins_part[i]) and mins_part[i] < global_min[i]:
                global_min[i] = mins_part[i]
            if math.isfinite(maxs_part[i]) and maxs_part[i] > global_max[i]:
                global_max[i] = maxs_part[i]

    if band_count is None:
        raise RuntimeError("No TIF files found in the directory")

    # Handle the edge case where everything is invalid
    for i in range(band_count):
        if not math.isfinite(global_min[i]) or not math.isfinite(global_max[i]):
            global_min[i] = math.inf
            global_max[i] = -math.inf

    return band_count, global_min, global_max


def _gdal_pass2_one(
    tif_path: Path,
    band_count: int,
    mins: List[float],
    maxs: List[float],
    bins: int,
    default_nodata: Optional[List[float]],
    scales: List[float],
    offsets: List[float],
) -> Tuple[List[np.ndarray], List[int]]:
    hist_list = [np.zeros(bins, dtype=np.int64) for _ in range(band_count)]  # noqa: F841
    valid_counts = [0] * band_count  # noqa: F841

    ds = gdal.Open(str(tif_path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Cannot open file: {tif_path}")
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    for bidx in range(1, band_count + 1):
        lo, hi = mins[bidx - 1], maxs[bidx - 1]
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
            continue
        band = ds.GetRasterBand(bidx)
        nodata = band.GetNoDataValue()
        if nodata is None:
            nodata_list = default_nodata
        else:
            nodata_list = [float(nodata)]
        bx, by = band.GetBlockSize()
        if bx is None or bx <= 0:
            bx = 256
        if by is None or by <= 0:
            by = 256
        scale = scales[bidx - 1]
        offset = offsets[bidx - 1]
        for yoff in range(0, ysize, by):
            ywin = min(by, ysize - yoff)
            for xoff in range(0, xsize, bx):
                xwin = min(bx, xsize - xoff)
                arr = band.ReadAsArray(xoff, yoff, xwin, ywin)
                if arr is None:
                    continue
                arr = np.asarray(arr)
                valid = is_valid(arr, nodata_list)
                if not valid.any():
                    continue
                # Histogram: avoid the large copy of vals = arr[valid].astype(float64)
                # Cast the whole block to float32/float64 first, then run bincount only on valid indices
                arr_f = arr.astype(np.float32, copy=False)
                idx_f = (arr_f + np.float32(offset)) * np.float32(scale)
                idx = idx_f.astype(np.int64, copy=False)
                idx_valid = idx[valid]
                if idx_valid.size == 0:
                    continue
                max_bin = bins - 1
                # clamp
                idx_valid[idx_valid < 0] = 0
                idx_valid[idx_valid > max_bin] = max_bin
                bincount = np.bincount(idx_valid, minlength=bins)
                hist_list[bidx - 1] += bincount
                valid_counts[bidx - 1] += int(idx_valid.size)
    ds = None
    return hist_list, valid_counts


def _gdal_pass2_one_star(
    args: Tuple[Path, int, List[float], List[float], int, Optional[List[float]], List[float], List[float]]
):
    """Adapter for imap arguments to avoid pickle errors caused by lambdas"""
    return _gdal_pass2_one(*args)


def pass2_histogram(
    tif_paths: List[Path],
    band_count: int,
    mins: List[float],
    maxs: List[float],
    bins: int,
    default_nodata: Optional[List[float]],
    workers: int
) -> Tuple[List[np.ndarray], List[int]]:
    """Returns: (per-band histogram list, per-band valid-pixel-count list)"""
    hist_list = [np.zeros(bins, dtype=np.int64) for _ in range(band_count)]
    valid_counts = [0] * band_count

    # Precompute per-band scaling coefficients (pass to children to avoid recomputing per file)
    scales: List[float] = []
    offsets: List[float] = []
    for i in range(band_count):
        lo, hi = mins[i], maxs[i]
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
            scales.append(0.0)
            offsets.append(0.0)
        else:
            scales.append((bins - 1) / (hi - lo))
            offsets.append(-lo)

    # Global GDAL configuration
    gdal.UseExceptions()
    gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')
    gdal.SetCacheMax(1024 * 1024 * 1024)

    # Parallelize: per-file local histograms and counts; the main process reduces by summation
    total_hist = [np.zeros(bins, dtype=np.int64) for _ in range(band_count)]
    total_counts = [0] * band_count
    args_iter = ((p, band_count, mins, maxs, bins, default_nodata, scales, offsets) for p in tif_paths)
    with mp.Pool(processes=max(1, workers)) as pool:
        for h_part, c_part in tqdm(
                pool.imap_unordered(_gdal_pass2_one_star, args_iter),
                total=len(tif_paths), desc="Pass2 hist", ncols=100, unit="file"):
            for i in range(band_count):
                total_hist[i] += h_part[i]
                total_counts[i] += c_part[i]

    return total_hist, total_counts


def percentile_from_hist(lo: float, hi: float, hist: np.ndarray, q: float) -> float:
    """Estimate a percentile from a histogram (linear interpolation). q in [0, 1]"""
    total = int(hist.sum())
    if total == 0 or not (math.isfinite(lo) and math.isfinite(hi)):
        return float('nan')
    target = q * (total - 1)
    cdf = np.cumsum(hist)
    # Find the first bin with cumulative count >= target + 1
    k = int(np.searchsorted(cdf, target + 1))
    k = max(0, min(k, len(hist) - 1))
    # bin edges
    width = (hi - lo) / len(hist)
    bin_lo = lo + k * width
    # Estimate the position within the bin
    prev_cum = 0 if k == 0 else int(cdf[k - 1])
    bin_count = int(hist[k])
    if bin_count <= 0:
        return bin_lo  # Rare case
    within = (target - prev_cum) / bin_count
    return float(bin_lo + within * width)


def main():
    parser = argparse.ArgumentParser(description="Compute per-band min/max/0.1%/99.9% over all TIFs in a directory (ignoring invalid values); GDAL parallel version")
    parser.add_argument("--input_dir",  
                        default='/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled_5xdownsampled_merged_clipped_unified/ERA5_merged_clipped_unified',
                        help="Input directory path")
    parser.add_argument("--bins", type=int, default=2048, help="Number of histogram bins; larger = finer but slower")
    parser.add_argument("--recursive", action="store_true", help="Search subdirectories recursively")
    parser.add_argument(
        "--default-nodata",
        type=float,
        nargs="+",
        default=[-32768, 32768, 255],
        help="Default invalid-value list (used when the file has no nodata set); multiple values allowed, e.g. --default-nodata -32768 32768 255",
    )
    parser.add_argument("--output", default=None, help="Path to save results as JSON (optional)")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2), help="Number of parallel processes (file-level parallelism)")

    args = parser.parse_args()
    root = Path(args.input_dir)
    if not root.exists() or not root.is_dir():
        print(f"Directory does not exist: {root}")
        sys.exit(1)

    tif_paths = list_tifs(root, recursive=args.recursive)
    if not tif_paths:
        print("No TIF files found")
        sys.exit(1)

    # On NFS / object storage (e.g. /s3), too many workers causes jitter and slowdown; empirically 2-4 is more stable.
    # Apply a conservative default cap here (you can still pass a larger workers value manually).
    input_str = str(root.as_posix()).lower()
    if ('/s3/' in input_str or 'nfs' in input_str) and args.workers > 4:
        print(f"Possible network-storage path detected ({root}); reducing workers from {args.workers} to 4 (more stable).")
        args.workers = 4

    # Pass 1: global min/max (parallel)
    band_count, mins, maxs = pass1_min_max(tif_paths, args.default_nodata, args.workers)

    # Pass 2: histogram (parallel)
    hists, counts = pass2_histogram(tif_paths, band_count, mins, maxs, args.bins, args.default_nodata, args.workers)

    # Aggregate results
    results: Dict[str, Dict[str, float]] = {}
    for i in range(band_count):
        lo, hi = mins[i], maxs[i]
        total = int(counts[i])
        p001 = percentile_from_hist(lo, hi, hists[i], 0.001)
        p999 = percentile_from_hist(lo, hi, hists[i], 0.999)
        results[f"band_{i+1}"] = {
            "count": total,
            "min": None if not math.isfinite(lo) else lo,
            "max": None if not math.isfinite(hi) else hi,
            "p0_1": None if not np.isfinite(p001) else p001,
            "p99_9": None if not np.isfinite(p999) else p999,
        }

    # Print
    print("Statistics (ignoring invalid values):")
    for k in sorted(results.keys(), key=lambda x: int(x.split('_')[1])):
        r = results[k]
        print(f"- {k}: count={r['count']}, min={r['min']}, p0.1%={r['p0_1']}, p99.9%={r['p99_9']}, max={r['max']}")

    # Save
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Result saved -> {out_path}")


if __name__ == "__main__":
    main()


