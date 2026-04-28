#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Given a population raster, compute each pixel's Euclidean distance (in physical length, using the transform's pixel size) to the nearest pixel with "population > threshold".

For each input TIF, two single-band float32 files are written (keeping the input filename stem):
  {stem}_distance.tif       raw distance (seed pixels are 0; all NaN if there are no seeds; NoData / invalid-population pixels still get a distance to the nearest seed)
  {stem}_distance_norm.tif  apply QuantileTransformer + [target_min, target_max] to pixels with dist > 0 and finite (also covers originally-invalid pixels)

Dependencies: rasterio, numpy, scipy, scikit-learn
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt

from worldpop_norm import (
    list_tifs,
    quantile_normalize_positive_mask,
    write_output,
)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Population raster -> distance to the nearest pop > threshold pixel (raw + quantile normalization)"
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
        default="/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/worldpop_v2_clipped/distance_to10people",
        help="Output directory (will be created)",
    )
    ap.add_argument("--pattern", default="*.tif", help="glob, relative to input-dir")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--band", type=int, default=1)
    ap.add_argument(
        "--pop-threshold",
        type=float,
        default=10.0,
        help="Population lower bound for treating a pixel as a \"high-density seed\" (default 10)",
    )
    ap.add_argument(
        "--compress",
        default="ZSTD",
        choices=["NONE", "LZW", "DEFLATE", "ZSTD"],
    )
    ap.add_argument("--blocksize", type=int, default=512)
    ap.add_argument("--meta-name", default="distance_to_dense_meta.json")
    ap.add_argument("--n-quantiles", type=int, default=1000)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--target-min", type=float, default=0.01)
    ap.add_argument("--target-max", type=float, default=1.0)
    args = ap.parse_args()
    if args.target_min >= args.target_max:
        ap.error("--target-min must be less than --target-max")
    return args


def edt_sampling_from_transform(transform) -> tuple[float, float]:
    """distance_transform_edt sampling: (row spacing, column spacing) in CRS units (e.g. projected meters)."""
    return (abs(float(transform.e)), abs(float(transform.a)))


def distance_to_pop_threshold(
    pop: np.ndarray,
    nodata: float | None,
    threshold: float,
    sampling: tuple[float, float],
) -> tuple[np.ndarray, dict]:
    """
    pop: raw band; valid pixels are finite; seeds = finite & (pop > threshold).
    Run EDT on the full (h, w) grid: every pixel (including NoData / invalid-population) gets the Euclidean distance to the nearest seed.
    Returns distances (float64; 0 at seeds; all NaN when there are no seeds).
    """
    x = pop.astype(np.float64, copy=True)
    if nodata is not None:
        x[np.isclose(x, nodata, equal_nan=True)] = np.nan
    x[~np.isfinite(x)] = np.nan

    valid = np.isfinite(x)
    seed = valid & (x > threshold)
    meta = {
        "pop_threshold": float(threshold),
        "finite_pixels": int(np.sum(valid)),
        "seed_pixels": int(np.sum(seed)),
        "sampling_row_col": list(sampling),
    }

    h, w = x.shape
    if not np.any(seed):
        meta["note"] = "no_seed_pixels"
        return np.full((h, w), np.nan, dtype=np.float64), meta

    # Boolean seeds: EDT on ~features yields the Euclidean distance to the nearest True (seed) pixel (per scipy's convention for bool inputs)
    features = np.zeros((h, w), dtype=bool)
    features[seed] = True
    dist = distance_transform_edt(~features, sampling=sampling).astype(np.float64)

    dist[np.isinf(dist)] = np.nan
    return dist, meta


def main():
    args = parse_args()
    in_dir = args.input_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_tifs(in_dir, args.pattern, args.recursive)
    if not files:
        print(f"No match found: {in_dir} / {args.pattern}")
        return

    all_meta: dict[str, dict] = {}
    for fp in files:
        rel = fp.relative_to(in_dir) if fp.is_relative_to(in_dir) else Path(fp.name)
        stem = fp.stem
        dst_raw = out_dir / rel.parent / f"{stem}_distance.tif"
        dst_norm = out_dir / rel.parent / f"{stem}_distance_norm.tif"
        dst_raw.parent.mkdir(parents=True, exist_ok=True)

        try:
            with rasterio.open(fp) as src:
                if args.band < 1 or args.band > src.count:
                    print(f"[SKIP] {fp}: band out-of-range")
                    continue
                pop = src.read(args.band)
                nod = src.nodata
                prof = src.profile
                sampling = edt_sampling_from_transform(src.transform)

            dist, dmeta = distance_to_pop_threshold(
                pop, nod, args.pop_threshold, sampling
            )

            # Raw distance: keep NaN, write as float32
            dist_f32 = dist.astype(np.float32)
            write_output(dst_raw, dist_f32, prof, args.compress, args.blocksize)

            # Normalization: dist > 0 and finite (covers original NoData pixels; seed dist == 0 does not participate in fitting)
            pos = np.isfinite(dist) & (dist > 0)
            norm, nmeta = quantile_normalize_positive_mask(
                dist,
                pos,
                n_quantiles=args.n_quantiles,
                random_state=args.random_state,
                target_min=args.target_min,
                target_max=args.target_max,
            )
            write_output(dst_norm, norm, prof, args.compress, args.blocksize)

            entry = {
                "distance": dmeta,
                "distance_norm": nmeta,
                "raw_path": str(dst_raw),
                "norm_path": str(dst_norm),
            }
            all_meta[str(rel.as_posix())] = entry
            print(f"[OK] {rel} -> {dst_raw.name}, {dst_norm.name}")
        except Exception as e:
            print(f"[FAIL] {fp}: {e}")
            all_meta[str(rel.as_posix())] = {"error": str(e)}

    meta_path = out_dir / args.meta_name
    meta_path.write_text(json.dumps(all_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done, metadata: {meta_path}")


if __name__ == "__main__":
    main()
