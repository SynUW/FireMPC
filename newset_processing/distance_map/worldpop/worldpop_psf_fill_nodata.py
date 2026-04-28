#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fill NoData in a population raster using a point-spread-function (PSF) approach: instead of EDT distances, perform a Gaussian-weighted diffusion of valid pixels,
and estimate missing pixels via a normalized convolution over surrounding valid values.

For a "human accessibility / heavy-tailed gravity" continuous field (without Gaussian), use the sibling worldpop_accessibility_idw.py (KD-tree power-law IDW) instead.

Mathematically equivalent to: apply the same Gaussian smoothing to both the value field v and the valid mask m, and estimate missing pixels as
  v_hat = G * (v . m) / max(G * m, eps),
while valid pixels keep their original values; pixels still without neighborhood support keep NaN.

For each input TIF, write one single-band float32 file (keeping the input filename stem):
  {stem}_psf_fill.tif

Performance and progress:
  - By default, two gaussian_filter passes on float32 reduce memory bandwidth.
  - --jobs N: multi-file parallelism (N > 1 limits BLAS/OpenMP threads in children to avoid contention).
  - --tile-size K: per-file streaming K x K tiles (with halo); saves memory on large images and shows per-tile progress; 0 reads the whole image (fastest single-file).
  - When tqdm is installed, a progress bar is shown; otherwise brief progress is printed.

Dependencies: rasterio, numpy, scipy; tqdm optional
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter

from worldpop_norm import list_tifs, write_output

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]


def parse_args():
    ap = argparse.ArgumentParser(
        description="Population raster: Gaussian-PSF normalized-convolution NoData fill (not a distance transform)"
    )
    ap.add_argument(
        "--input-dir",
        type=Path,
        default="/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/worldpop_v2_clipped/clipped",
        help="Population GeoTIFF directory",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default="/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/worldpop_v2_clipped/psf_fill",
        help="Output directory (will be created)",
    )
    ap.add_argument("--pattern", default="*.tif", help="glob, relative to input-dir")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--band", type=int, default=1)
    ap.add_argument(
        "--sigma-pixels",
        type=float,
        default=1000,
        help="Gaussian PSF standard deviation (pixels); if --sigma-m is given, physical length is used instead",
    )
    ap.add_argument(
        "--sigma-m",
        type=float,
        default=None,
        help="Gaussian PSF standard deviation (in CRS-consistent length units, e.g. meters); overrides --sigma-pixels when set",
    )
    ap.add_argument(
        "--truncate",
        type=float,
        default=4.0,
        help="Gaussian-kernel truncation multiplier (scipy gaussian_filter truncate, default 4)",
    )
    ap.add_argument(
        "--weight-eps",
        type=float,
        default=1e-12,
        help="If the denominator G*m is below this threshold, the pixel is treated as still unfillable and kept as NaN",
    )
    ap.add_argument(
        "--tile-size",
        type=int,
        default=0,
        help="Single-file tile side length (pixels); 0 reads the whole image. Set 2048-8192 on large images to cap memory and show per-tile progress",
    )
    ap.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="Number of files to process in parallel; 1 means sequential",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress bar / brief progress output",
    )
    ap.add_argument(
        "--compress",
        default="ZSTD",
        choices=["NONE", "LZW", "DEFLATE", "ZSTD"],
    )
    ap.add_argument("--blocksize", type=int, default=512)
    ap.add_argument("--meta-name", default="worldpop_psf_fill_meta.json")
    args = ap.parse_args()
    if args.sigma_pixels <= 0:
        ap.error("--sigma-pixels must be positive")
    if args.sigma_m is not None and args.sigma_m <= 0:
        ap.error("--sigma-m must be positive")
    if args.truncate <= 0:
        ap.error("--truncate must be positive")
    if args.tile_size < 0:
        ap.error("--tile-size cannot be negative")
    if args.jobs < 1:
        ap.error("--jobs must be >= 1")
    return args


def sigma_meters_to_pixels(transform, sigma_m: float) -> float:
    """Convert a physical sigma into a pixel sigma using the RMS of pixel sizes (isotropic approximation)."""
    px = abs(float(transform.a))
    py = abs(float(transform.e))
    mean_size = 0.5 * (px + py)
    if mean_size < 1e-30:
        raise ValueError("Pixel size is too small to convert --sigma-m")
    return float(sigma_m) / mean_size


def kernel_radius_pixels(sigma_pixels: float, truncate: float) -> int:
    return max(1, int(np.ceil(float(sigma_pixels) * float(truncate))))


def _limit_blas_threads() -> None:
    """In multiprocess mode, avoid each child opening many BLAS threads."""
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, "1")


def psf_fill_nodata(
    pop: np.ndarray,
    nodata: float | None,
    sigma_pixels: float,
    truncate: float,
    weight_eps: float,
) -> tuple[np.ndarray, dict]:
    """
    pop: raw band; valid pixel = finite and not nodata.
    Keep original values at valid pixels; fill invalid pixels via G * (v . m) / G * m (Gaussian PSF).
    Internally uses float32 for filtering to reduce memory and bandwidth.
    """
    x = pop.astype(np.float32, copy=True)
    if nodata is not None:
        x[np.isclose(x, np.float32(nodata), equal_nan=True)] = np.nan
    x[~np.isfinite(x)] = np.nan

    valid = np.isfinite(x)
    meta = {
        "sigma_pixels": float(sigma_pixels),
        "truncate": float(truncate),
        "weight_eps": float(weight_eps),
        "valid_pixels": int(np.sum(valid)),
        "invalid_pixels": int(np.size(x) - np.sum(valid)),
    }

    if not np.any(valid):
        meta["note"] = "all_invalid"
        return np.full(x.shape, np.nan, dtype=np.float32), meta

    vm = np.where(valid, x, np.float32(0.0))
    m = valid.astype(np.float32)
    num = gaussian_filter(
        vm, sigma=sigma_pixels, truncate=truncate, mode="constant", cval=0.0
    )
    den = gaussian_filter(
        m, sigma=sigma_pixels, truncate=truncate, mode="constant", cval=0.0
    )

    out = x.copy()
    hole = ~valid
    # Compare the denominator threshold in float64 to avoid float32 vs tiny-eps comparison artifacts
    den64 = den.astype(np.float64)
    reachable = hole & (den64 > weight_eps)
    if np.any(reachable):
        out[reachable] = (
            num.astype(np.float64)[reachable] / den64[reachable]
        ).astype(np.float32)

    meta["filled_pixels"] = int(np.sum(reachable))
    meta["still_nan_after_psf"] = int(np.sum(hole & ~reachable))
    return out, meta


def _write_profile_from_src(prof: dict, compress: str, blocksize: int) -> dict:
    p = prof.copy()
    p.update(
        dtype="float32",
        count=1,
        nodata=None,
        tiled=True,
        blockxsize=blocksize,
        blockysize=blocksize,
    )
    if compress != "NONE":
        p.update(compress=compress)
        if compress == "ZSTD":
            p.update(zstd_level=9)
    return p


def psf_fill_one_geotiff_tiled(
    fp: Path,
    dst: Path,
    band: int,
    sigma_pixels: float,
    truncate: float,
    weight_eps: float,
    tile_size: int,
    compress: str,
    blocksize: int,
    show_tile_progress: bool,
) -> dict:
    """Block-wise read/write of a single file; tile_size is the square tile side length (pixels)."""
    r = kernel_radius_pixels(sigma_pixels, truncate)
    fmeta: dict = {"tiled_io": True, "tile_size": tile_size, "halo_pixels": r}

    with rasterio.open(fp) as src:
        if band < 1 or band > src.count:
            raise ValueError(f"band {band} out-of-range (count={src.count})")
        prof = src.profile
        h, w = int(src.height), int(src.width)
        nod = src.nodata
        wprof = _write_profile_from_src(prof, compress, blocksize)
        dst.parent.mkdir(parents=True, exist_ok=True)

        n_row = (h + tile_size - 1) // tile_size
        n_col = (w + tile_size - 1) // tile_size
        total_windows = n_row * n_col
        fmeta["grid_windows"] = total_windows

        pbar = None
        if show_tile_progress and total_windows > 1 and tqdm is not None:
            pbar = tqdm(total=total_windows, desc=f"tiles {fp.name}", unit="win", leave=False)

        with rasterio.open(dst, "w", **wprof) as dst_ds:
            wi = 0
            for row in range(0, h, tile_size):
                th = min(tile_size, h - row)
                for col in range(0, w, tile_size):
                    tw = min(tile_size, w - col)
                    row0 = max(0, row - r)
                    col0 = max(0, col - r)
                    row1 = min(h, row + th + r)
                    col1 = min(w, col + tw + r)
                    win = Window(col0, row0, col1 - col0, row1 - row0)
                    pop = src.read(band, window=win)
                    filled, _ = psf_fill_nodata(
                        pop, nod, sigma_pixels, truncate, weight_eps
                    )
                    cr = row - row0
                    cc = col - col0
                    core = filled[cr : cr + th, cc : cc + tw]
                    out_win = Window(col, row, tw, th)
                    dst_ds.write(core.astype(np.float32, copy=False), 1, window=out_win)
                    wi += 1
                    if pbar is not None:
                        pbar.update(1)
                    elif show_tile_progress and total_windows > 1 and (wi % max(1, total_windows // 20) == 0 or wi == total_windows):
                        print(f"    [{wi}/{total_windows}] tile", flush=True)

            if pbar is not None:
                pbar.close()

    # Counting would require a second pass and is expensive; in tiled mode, do not write the whole-image filled_pixels count
    fmeta["note_stats_skipped_tiled"] = True
    return fmeta


def process_one_file(task: dict) -> tuple[str, dict]:
    """
    Child-process entry point: task must be pickleable pure data.
    Returns (rel_posix, meta_entry).
    """
    if task.get("_worker_blas", False):
        _limit_blas_threads()

    fp = Path(task["fp"])
    in_dir = Path(task["in_dir"]).resolve()
    out_dir = Path(task["out_dir"]).resolve()
    band = int(task["band"])
    sigma_m = task.get("sigma_m")
    sigma_pixels_arg = float(task["sigma_pixels"])
    truncate = float(task["truncate"])
    weight_eps = float(task["weight_eps"])
    tile_size = int(task["tile_size"])
    compress = str(task["compress"])
    blocksize = int(task["blocksize"])
    show_tile_progress = bool(task.get("show_tile_progress", False))

    rel = fp.relative_to(in_dir) if fp.is_relative_to(in_dir) else Path(fp.name)
    stem = fp.stem
    dst = out_dir / rel.parent / f"{stem}_psf_fill.tif"
    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        with rasterio.open(fp) as src:
            if band < 1 or band > src.count:
                return str(rel.as_posix()), {"error": f"band {band} out-of-range (count={src.count})"}
            prof = src.profile
            if sigma_m is not None:
                sigma_px = sigma_meters_to_pixels(src.transform, float(sigma_m))
            else:
                sigma_px = sigma_pixels_arg
            h, w = int(src.height), int(src.width)
            nod = src.nodata
            use_tiles = tile_size > 0 and max(h, w) > tile_size
            pop = None if use_tiles else src.read(band)

        if use_tiles:
            fmeta_extra = psf_fill_one_geotiff_tiled(
                fp,
                dst,
                band,
                sigma_px,
                truncate,
                weight_eps,
                tile_size,
                compress,
                blocksize,
                show_tile_progress=show_tile_progress,
            )
            entry = {
                "psf_fill": {
                    "sigma_pixels_resolved": sigma_px,
                    **fmeta_extra,
                },
                "output_path": str(dst),
            }
            return str(rel.as_posix()), entry

        filled, fmeta = psf_fill_nodata(
            pop, nod, sigma_pixels=sigma_px, truncate=truncate, weight_eps=weight_eps
        )
        fmeta["sigma_pixels_resolved"] = sigma_px

        write_output(dst, filled.astype(np.float32, copy=False), prof, compress, blocksize)
        return str(rel.as_posix()), {"psf_fill": fmeta, "output_path": str(dst)}
    except Exception as e:
        return str(rel.as_posix()), {"error": str(e)}


def _task_dict(fp: Path, args, in_dir: Path, out_dir: Path, show_tile_progress: bool) -> dict:
    return {
        "fp": str(fp.resolve()),
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "band": args.band,
        "sigma_m": args.sigma_m,
        "sigma_pixels": float(args.sigma_pixels),
        "truncate": float(args.truncate),
        "weight_eps": float(args.weight_eps),
        "tile_size": int(args.tile_size),
        "compress": args.compress,
        "blocksize": int(args.blocksize),
        "show_tile_progress": show_tile_progress,
        "_worker_blas": args.jobs > 1,
    }


def main():
    args = parse_args()
    in_dir = args.input_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_tifs(in_dir, args.pattern, args.recursive)
    if not files:
        print(f"No match found: {in_dir} / {args.pattern}")
        return

    n = len(files)
    show_progress = not args.no_progress
    # tqdm in children easily interleaves under multiprocessing; only enable per-tile progress in single-process mode
    tile_prog = show_progress and args.jobs == 1

    tasks = [_task_dict(fp, args, in_dir, out_dir, tile_prog) for fp in files]
    all_meta: dict[str, dict] = {}

    if args.jobs == 1:
        if show_progress and tqdm is not None:
            task_iter = tqdm(tasks, desc="PSF infill", unit="file")
        else:
            task_iter = tasks
            if show_progress:
                print(f"Total {n} files; starting to process...", flush=True)
        for i, t in enumerate(task_iter, 1):
            if show_progress and tqdm is None:
                print(f"[{i}/{n}] {Path(t['fp']).name} …", flush=True)
            key, entry = process_one_file(t)
            all_meta[key] = entry
            if "error" not in entry and show_progress and tqdm is None:
                print(f"    -> OK {entry.get('output_path', '')}", flush=True)
            elif "error" in entry and show_progress:
                print(f"    -> FAIL {entry['error']}", flush=True)
    else:
        _limit_blas_threads()
        if show_progress:
            print(f"Total {n} files, {args.jobs} processes in parallel...", flush=True)
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futures = {ex.submit(process_one_file, t): t for t in tasks}
            completed_iter = as_completed(futures)
            if show_progress and tqdm is not None:
                completed_iter = tqdm(completed_iter, total=n, desc="PSF infill", unit="file")
            for fut in completed_iter:
                key, entry = fut.result()
                all_meta[key] = entry

    meta_path = out_dir / args.meta_name
    meta_path.write_text(json.dumps(all_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done, metadata: {meta_path}")


if __name__ == "__main__":
    main()
