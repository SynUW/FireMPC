#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WorldPop (or similar non-negative raster) batch normalization (multiprocess version):

For each GeoTIFF in the input folder (default single band):
1. NoData -> NaN; original 0 and NoData remain 0 in the output.
2. Take raw values only at pop > 0 (no log), map them to approximately standard-normal via sklearn QuantileTransformer,
   then linearly scale by the batch result's min/max to [0.01, 1.0].
3. Write float32; also write worldpop_norm_params.json.

Dependencies: pip install scikit-learn
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
from sklearn.preprocessing import QuantileTransformer


def parse_args():
    ap = argparse.ArgumentParser(
        description="WorldPop: QuantileTransformer -> normal, then min-max to [0.01, 1] (pop > 0 only)"
    )
    ap.add_argument("--input-dir", default='/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/worldpop_v2_clipped/access_idw', type=Path, help="Directory containing multiple TIFs")
    ap.add_argument("--output-dir", default='/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/worldpop_v2_clipped/access_idw_norm', type=Path, help="Output directory (will be created)")
    ap.add_argument("--pattern", default="*.tif", help="Glob relative to input-dir, default *.tif")
    ap.add_argument("--recursive", action="store_true", help="Recursive rglob(pattern)")
    ap.add_argument("--band", type=int, default=1, help="Band index to read (1-based)")
    ap.add_argument(
        "--compress",
        default="ZSTD",
        choices=["NONE", "LZW", "DEFLATE", "ZSTD"],
        help="Output compression",
    )
    ap.add_argument("--blocksize", type=int, default=512, help="Block size")
    ap.add_argument("--meta-name", default="worldpop_norm_params.json", help="Parameter JSON filename")
    ap.add_argument(
        "--n-quantiles",
        type=int,
        default=1000,
        help="QuantileTransformer.n_quantiles (capped at the number of positive pixels; auto-truncated)",
    )
    ap.add_argument("--random-state", type=int, default=42, help="QuantileTransformer random seed")
    ap.add_argument(
        "--target-min",
        type=float,
        default=0.01,
        help="Normalization lower bound for positive pixels (default 0.01)",
    )
    ap.add_argument(
        "--target-max",
        type=float,
        default=1.0,
        help="Normalization upper bound for positive pixels (default 1.0)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel processes (default = CPU core count)",
    )
    ap.add_argument(
        "--no-log1p",
        action="store_true",
        help="Disable the log1p pre-transform (on by default; recommended to keep for strongly right-skewed fields like IDW)",
    )
    args = ap.parse_args()
    if args.target_min >= args.target_max:
        ap.error("--target-min must be less than --target-max")
    if args.workers is None:
        args.workers = os.cpu_count() or 4
    return args


def list_tifs(input_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    if recursive:
        files = sorted(input_dir.rglob(pattern))
    else:
        files = sorted(input_dir.glob(pattern))
    return [f for f in files if f.is_file()]


def quantile_normalize_positive_mask(
    x: np.ndarray,
    pos: np.ndarray,
    *,
    n_quantiles: int = 1000,
    random_state: int = 42,
    target_min: float = 0.01,
    target_max: float = 1.0,
    log1p_first: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Only on pixels with pos == True: apply QuantileTransformer -> normal, then linearly map to [target_min, target_max];
    other pixels output 0. x and pos have the same shape; typically pos implies finite & (x > 0).

    log1p_first=True (default): apply log1p before QuantileTransformer to suppress the right tail.
    Strongly recommended for strongly right-skewed fields like population IDW (e.g. all of Canada); makes the final distribution more symmetric.
    """
    meta: dict = {
        "norm_positive_pixels": int(np.sum(pos)),
        "output_distribution": "normal",
        "target_range": [target_min, target_max],
        "log1p_first": log1p_first,
    }
    if not np.any(pos):
        meta.update({"note": "no_positive_for_norm", "transformed_min": None, "transformed_max": None})
        return np.zeros(x.shape, dtype=np.float32), meta

    pos_values = x[pos].astype(np.float64)
    if log1p_first:
        # IDW values are always >= 0, so log1p is safe; suppressing the right tail first then applying the quantile transform works much better than transforming directly
        pos_values = np.log1p(pos_values)
    pos_values = pos_values.reshape(-1, 1)

    n_pos = int(pos_values.shape[0])
    nq = max(1, min(n_quantiles, n_pos))
    meta["n_quantiles_used"] = nq
    meta["random_state"] = random_state

    qt = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=nq,
        random_state=random_state,
    )
    transformed_pos = qt.fit_transform(pos_values).ravel()

    t_min = float(np.min(transformed_pos))
    t_max = float(np.max(transformed_pos))
    meta["transformed_min"] = t_min
    meta["transformed_max"] = t_max

    span = t_max - t_min
    lo, hi = float(target_min), float(target_max)
    if span < 1e-15:
        norm_pos = np.full(n_pos, (lo + hi) * 0.5, dtype=np.float64)
        meta["note"] = "degenerate_transform_span"
    else:
        norm_pos = (transformed_pos - t_min) / span * (hi - lo) + lo

    result = np.zeros(x.shape, dtype=np.float32)
    result[pos] = norm_pos.astype(np.float32)
    result[~np.isfinite(result)] = 0.0
    return result, meta


def normalize_worldpop_array(
    arr: np.ndarray,
    nodata: float | None,
    *,
    n_quantiles: int = 1000,
    random_state: int = 42,
    target_min: float = 0.01,
    target_max: float = 1.0,
    log1p_first: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    For valid pixels only (finite & x >= 0): optionally apply log1p first, then QuantileTransformer(output_distribution='normal'),
    then linearly map to [target_min, target_max]. NoData and NaN output as 0.

    The IDW accessibility field's range is [0, +inf), always non-negative, so log1p is safe.
    For strongly right-skewed population fields (e.g. Canada-wide), log1p_first=True (default) significantly improves distributional symmetry.
    """
    x = arr.astype(np.float64, copy=True)
    if nodata is not None:
        x[np.isclose(x, nodata, equal_nan=True)] = np.nan
    x[~np.isfinite(x)] = np.nan

    finite = np.isfinite(x)
    # IDW output is always >= 0; use finite & (x >= 0) instead of x > 0 to avoid mistakenly excluding floating-point zeros
    pos = finite & (x >= 0)
    meta: dict = {
        "finite_pixels": int(np.sum(finite)),
        "positive_pixels": int(np.sum(pos)),
        "log1p_first": log1p_first,
    }

    if not np.any(finite):
        meta.update({"note": "all_invalid", "transformed_min": None, "transformed_max": None})
        return np.zeros(arr.shape, dtype=np.float32), meta
    if not np.any(pos):
        meta.update({"note": "no_positive", "transformed_min": None, "transformed_max": None})
        return np.zeros(arr.shape, dtype=np.float32), meta

    out, qmeta = quantile_normalize_positive_mask(
        x,
        pos,
        n_quantiles=n_quantiles,
        random_state=random_state,
        target_min=target_min,
        target_max=target_max,
        log1p_first=log1p_first,
    )
    meta.update(qmeta)
    meta["positive_pixels"] = int(np.sum(pos))
    return out, meta


def write_output(
    path_out: Path,
    data: np.ndarray,
    src_profile: dict,
    compress: str,
    blocksize: int,
) -> None:
    path_out.parent.mkdir(parents=True, exist_ok=True)
    prof = src_profile.copy()
    prof.update(
        dtype="float32",
        count=1,
        nodata=None,
        tiled=True,
        blockxsize=blocksize,
        blockysize=blocksize,
    )
    if compress != "NONE":
        prof.update(compress=compress)
        if compress == "ZSTD":
            prof.update(zstd_level=9)
    with rasterio.open(path_out, "w", **prof) as dst:
        dst.write(data, 1)


def _process_single_file(
    fp: Path,
    in_dir: Path,
    out_dir: Path,
    band: int,
    n_quantiles: int,
    random_state: int,
    target_min: float,
    target_max: float,
    compress: str,
    blocksize: int,
    log1p_first: bool = True,
) -> tuple[str, dict]:
    """Process a single TIF file in a child process and return (rel_key, meta_dict)."""
    rel = fp.relative_to(in_dir) if fp.is_relative_to(in_dir) else Path(fp.name)
    rel_key = str(rel.as_posix())
    dst = out_dir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        with rasterio.open(fp) as src:
            if band < 1 or band > src.count:
                return rel_key, {"skip": f"band {band} out-of-range (count={src.count})"}
            arr = src.read(band)
            nod = src.nodata
            prof = src.profile

        z, meta = normalize_worldpop_array(
            arr,
            nod,
            n_quantiles=n_quantiles,
            random_state=random_state,
            target_min=target_min,
            target_max=target_max,
            log1p_first=log1p_first,
        )
        meta["input"] = str(fp)
        meta["output"] = str(dst)

        write_output(dst, z, prof, compress, blocksize)
        return rel_key, meta
    except Exception as e:
        return rel_key, {"error": str(e)}


def main():
    args = parse_args()
    in_dir = args.input_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_tifs(in_dir, args.pattern, args.recursive)
    if not files:
        print(f"No matching files found: {in_dir} / {args.pattern}")
        return

    log1p_first = not args.no_log1p
    n_workers = min(args.workers, len(files))
    print(f"Total {len(files)} files, using {n_workers} processes processing in parallel; log1p_first={log1p_first}")

    all_meta: dict[str, dict] = {}
    done = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                _process_single_file,
                fp,
                in_dir,
                out_dir,
                args.band,
                args.n_quantiles,
                args.random_state,
                args.target_min,
                args.target_max,
                args.compress,
                args.blocksize,
                log1p_first,
            ): fp
            for fp in files
        }

        for future in as_completed(futures):
            fp = futures[future]
            rel_key, meta = future.result()
            all_meta[rel_key] = meta
            done += 1
            if "error" in meta:
                print(f"[FAIL] ({done}/{len(files)}) {rel_key}: {meta['error']}")
            elif "skip" in meta:
                print(f"[SKIP] ({done}/{len(files)}) {rel_key}: {meta['skip']}")
            else:
                print(f"[OK]   ({done}/{len(files)}) {rel_key}")

    meta_path = out_dir / args.meta_name
    meta_path.write_text(json.dumps(all_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done {len(files)} files; parameters written to {meta_path}")


if __name__ == "__main__":
    main()
