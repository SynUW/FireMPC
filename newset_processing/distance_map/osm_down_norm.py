#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OSM binary raster batch downsampling + distance transform + normalization.

Input folder layout:
  <osm-dir>/
    roads/      *.tif  (processed independently)
    powerlines/ *.tif  (processed independently)
    buildings/  *.tif  (processed independently)
    rivers/     *.tif  ┐
    lakes/      *.tif  ┘ matched by year and unioned, then processed

Output folder layout (under --output-dir):
  roads/      Roads_{year}.tif
  powerlines/ Powerlines_{year}.tif
  buildings/  Buildings_{year}.tif
  waters/     Waters_{year}.tif   <- union of rivers + lakes

Processing pipeline (per binary TIF):
  1. Read the binary raster (1 = feature present, 0 = absent)
  2. Compute the Euclidean distance transform at native resolution (1 km) -> distance in meters
  3. Downsample to 5 km (mean aggregation)
  4. log1p + min-max normalize to [0, 1] (or exponential decay, optional)
  5. Write float32 GeoTIFF, filenames standardized as {category_name}_{year}.tif

rivers + lakes handling:
  - Extract the year from the filename (first 4-digit run), then union by year.
  - If only one of the two exists for a year, log a warning and process it alone.

Dependencies: pip install rasterio scipy numpy
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy.ndimage import distance_transform_edt


# ---------------------------------------------------------------------------
# Utility: extract the first 4-digit year from a filename
# ---------------------------------------------------------------------------

# Underscore-separated year (e.g. canada_roads_2010_1km_EPSG4326).
# Cannot use \b(\d{4})\b: in Python, _ is a "word character", so _2010_ has no word boundary on either side.
_YEAR_RE = re.compile(r"_(\d{4})(?:_|$)")


def extract_year(stem: str) -> str | None:
    """Extract a year from a file stem: match _YYYY_ or _YYYY at the end of the stem."""
    m = _YEAR_RE.search(stem)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="OSM binary rasters: EDT distance field -> 5 km downsample -> normalize"
    )
    ap.add_argument(
        "--osm-dir",
        type=Path,
        default='/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/osm/yearly_1km_tifs',
        help="OSM input root, with roads/powerlines/buildings/rivers/lakes subdirectories",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default='/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/OSMs_downsampled',
        help="Output root directory (will be created); same layout as input, with rivers+lakes -> waters",
    )
    ap.add_argument(
        "--pixel-size-m",
        type=float,
        default=5000.0,
        help="Input raster pixel side length (meters), default 1000",
    )
    ap.add_argument(
        "--downsample-factor",
        type=int,
        default=5,
        help="Downsampling factor (default 5, i.e. 1 km -> 5 km)",
    )
    ap.add_argument(
        "--norm-method",
        choices=["log_minmax", "exp_decay", "none"],
        default="log_minmax",
        help="Normalization method: log_minmax (recommended) | exp_decay | none (raw meters)",
    )
    ap.add_argument(
        "--exp-scale-m",
        type=float,
        default=50000.0,
        help="Characteristic distance for exp_decay (meters); default 50000 = 50 km",
    )
    ap.add_argument(
        "--compress",
        default="ZSTD",
        choices=["NONE", "LZW", "DEFLATE", "ZSTD"],
        help="Output compression format, default ZSTD",
    )
    ap.add_argument("--blocksize", type=int, default=512, help="Output tile block size")
    ap.add_argument(
        "--meta-name",
        default="osm_norm_params.json",
        help="Output parameter JSON filename (written under --output-dir)",
    )
    ap.add_argument(
        "--pattern", default="*.tif", help="File glob within subdirectories, default *.tif"
    )
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def compute_distance_field(binary: np.ndarray, pixel_size_m: float) -> np.ndarray:
    """
    For a binary array (1 = feature, 0 = background), compute the Euclidean distance to the nearest feature in meters.
    If the entire image is features, the distance field is all zeros.
    """
    dist_px = distance_transform_edt(binary == 0)
    return dist_px.astype(np.float32) * pixel_size_m


def block_mean_downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample arr by mean aggregation with (factor x factor) blocks.
    If rows/columns are not divisible by factor, trailing extra rows/columns are discarded (no zero padding).
    """
    H, W = arr.shape
    Hn = (H // factor) * factor
    Wn = (W // factor) * factor
    clipped = arr[:Hn, :Wn]
    nh = Hn // factor
    nw = Wn // factor
    return clipped.reshape(nh, factor, nw, factor).mean(axis=(1, 3))


def normalize(arr: np.ndarray, method: str, exp_scale_m: float) -> np.ndarray:
    if method == "log_minmax":
        d = np.log1p(arr)
        lo, hi = float(d.min()), float(d.max())
        span = hi - lo
        if span < 1e-9:
            return np.zeros_like(arr, dtype=np.float32)
        return ((d - lo) / span).astype(np.float32)
    elif method == "exp_decay":
        return np.exp(-arr / exp_scale_m).astype(np.float32)
    else:
        return arr.astype(np.float32)


def process_single(
    binary: np.ndarray,
    src_profile: dict,
    *,
    pixel_size_m: float,
    downsample_factor: int,
    norm_method: str,
    exp_scale_m: float,
) -> tuple[np.ndarray, dict, Affine]:
    """
    Full pipeline: EDT -> downsample -> normalize.
    Returns (normalized array, statistics meta, new transform).
    """
    dist_1km = compute_distance_field(binary, pixel_size_m)
    dist_5km = block_mean_downsample(dist_1km, downsample_factor)
    normed = normalize(dist_5km, norm_method, exp_scale_m)

    old_transform: Affine = src_profile["transform"]
    new_transform = old_transform * old_transform.scale(
        downsample_factor, downsample_factor
    )

    meta = {
        "input_shape": list(binary.shape),
        "output_shape": list(normed.shape),
        "pixel_size_m": pixel_size_m,
        "downsample_factor": downsample_factor,
        "norm_method": norm_method,
        "dist_1km_max": float(dist_1km.max()),
        "dist_5km_mean": float(dist_5km.mean()),
        "normed_min": float(normed.min()),
        "normed_max": float(normed.max()),
    }
    if norm_method == "exp_decay":
        meta["exp_scale_m"] = exp_scale_m

    return normed, meta, new_transform


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def read_binary(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1)
        profile = src.profile
    return arr.astype(np.uint8), profile


def write_output(
    path_out: Path,
    data: np.ndarray,
    src_profile: dict,
    new_transform: Affine,
    compress: str,
    blocksize: int,
) -> None:
    path_out.parent.mkdir(parents=True, exist_ok=True)
    prof = src_profile.copy()
    prof.update(
        dtype="float32",
        count=1,
        height=data.shape[0],
        width=data.shape[1],
        transform=new_transform,
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


# ---------------------------------------------------------------------------
# Independent batch processing per subdirectory
# ---------------------------------------------------------------------------

def process_category(
    subdir: Path,
    out_subdir: Path,
    category_label: str,
    *,
    pattern: str,
    pixel_size_m: float,
    downsample_factor: int,
    norm_method: str,
    exp_scale_m: float,
    compress: str,
    blocksize: int,
) -> dict:
    """
    Process a single category subdirectory. Output filename: {category_label}_{year}.tif.
    If no year is found in the filename, keep the original name (fallback).
    """
    files = sorted(subdir.glob(pattern))
    if not files:
        print(f"  [WARN] {subdir}: not found {pattern}")
        return {}

    category_meta: dict = {}
    for fp in files:
        year = extract_year(fp.stem)
        out_name = f"{category_label}_{year}.tif" if year else fp.name
        dst = out_subdir / out_name
        try:
            binary, profile = read_binary(fp)
            normed, meta, new_transform = process_single(
                binary,
                profile,
                pixel_size_m=pixel_size_m,
                downsample_factor=downsample_factor,
                norm_method=norm_method,
                exp_scale_m=exp_scale_m,
            )
            meta["input"] = str(fp)
            meta["output"] = str(dst)
            category_meta[out_name] = meta

            write_output(dst, normed, profile, new_transform, compress, blocksize)
            print(f"  [OK] {fp.name} → {out_name}")
        except Exception as e:
            print(f"  [FAIL] {fp.name}: {e}")
            category_meta[out_name] = {"error": str(e)}

    return category_meta


# ---------------------------------------------------------------------------
# rivers + lakes -> waters merge (matched by year)
# ---------------------------------------------------------------------------

def _index_by_year(directory: Path, pattern: str) -> dict[str, Path]:
    """Returns a {year_str: Path} dict; files without a parseable year are skipped with a warning."""
    result: dict[str, Path] = {}
    if not directory.is_dir():
        return result
    for fp in sorted(directory.glob(pattern)):
        year = extract_year(fp.stem)
        if year is None:
            print(f"  [WARN] cannot extract year from filename; skipping: {fp.name}")
            continue
        if year in result:
            print(f"  [WARN] year {year} duplicated; the latter overrides the former: {fp.name}")
        result[year] = fp
    return result


def process_waters(
    rivers_dir: Path,
    lakes_dir: Path,
    out_waters_dir: Path,
    *,
    pattern: str,
    pixel_size_m: float,
    downsample_factor: int,
    norm_method: str,
    exp_scale_m: float,
    compress: str,
    blocksize: int,
) -> dict:
    """
    Match rivers and lakes by year, union pixel-by-pixel, and write Waters_{year}.tif.
    If only one side exists for a given year, log a warning and process that side alone.
    """
    rivers_by_year = _index_by_year(rivers_dir, pattern)
    lakes_by_year  = _index_by_year(lakes_dir,  pattern)

    all_years = sorted(set(rivers_by_year) | set(lakes_by_year))
    if not all_years:
        print(f"  [WARN] no valid year files found under rivers or lakes")
        return {}

    only_rivers = sorted(set(rivers_by_year) - set(lakes_by_year))
    only_lakes  = sorted(set(lakes_by_year)  - set(rivers_by_year))
    if only_rivers:
        print(f"  [WARN] only found these years in rivers (no matching lakes); will process them alone: {only_rivers}")
    if only_lakes:
        print(f"  [WARN] only found these years in lakes (no matching rivers); will process them alone: {only_lakes}")

    waters_meta: dict = {}
    for year in all_years:
        out_name = f"Waters_{year}.tif"
        dst = out_waters_dir / out_name
        r_path = rivers_by_year.get(year)
        l_path = lakes_by_year.get(year)
        try:
            if r_path and l_path:
                r_bin, profile = read_binary(r_path)
                l_bin, _       = read_binary(l_path)
                binary = np.where((r_bin > 0) | (l_bin > 0), 1, 0).astype(np.uint8)
                source_desc = f"{r_path} | {l_path}"
            elif r_path:
                binary, profile = read_binary(r_path)
                source_desc = str(r_path)
            else:
                binary, profile = read_binary(l_path)
                source_desc = str(l_path)

            normed, meta, new_transform = process_single(
                binary,
                profile,
                pixel_size_m=pixel_size_m,
                downsample_factor=downsample_factor,
                norm_method=norm_method,
                exp_scale_m=exp_scale_m,
            )
            meta["input"] = source_desc
            meta["output"] = str(dst)
            waters_meta[out_name] = meta

            write_output(dst, normed, profile, new_transform, compress, blocksize)
            print(f"  [OK] {out_name}")
        except Exception as e:
            print(f"  [FAIL] {out_name}: {e}")
            waters_meta[out_name] = {"error": str(e)}

    return waters_meta


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

# Category subdir name -> output filename prefix
SOLO_CATEGORIES: dict[str, str] = {
    "roads":      "Roads",
    "powerlines": "Powerlines",
    "buildings":  "Buildings",
}


def main():
    args = parse_args()
    osm_dir: Path = args.osm_dir.resolve()
    out_dir: Path  = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_meta: dict = {}

    # Independent categories
    for cat, label in SOLO_CATEGORIES.items():
        subdir = osm_dir / cat
        if not subdir.is_dir():
            print(f"[SKIP] subdirectory does not exist: {subdir}")
            continue
        print(f"[{cat}] starting to process...")
        out_subdir = out_dir / cat
        meta = process_category(
            subdir,
            out_subdir,
            label,
            pattern=args.pattern,
            pixel_size_m=args.pixel_size_m,
            downsample_factor=args.downsample_factor,
            norm_method=args.norm_method,
            exp_scale_m=args.exp_scale_m,
            compress=args.compress,
            blocksize=args.blocksize,
        )
        all_meta[cat] = meta

    # rivers + lakes → waters
    rivers_dir = osm_dir / "rivers"
    lakes_dir  = osm_dir / "lakes"
    has_rivers = rivers_dir.is_dir()
    has_lakes  = lakes_dir.is_dir()
    if has_rivers or has_lakes:
        print("[waters] Starting processing (rivers UNION lakes)...")
        out_waters_dir = out_dir / "waters"
        out_waters_dir.mkdir(parents=True, exist_ok=True)
        meta = process_waters(
            rivers_dir,
            lakes_dir,
            out_waters_dir,
            pattern=args.pattern,
            pixel_size_m=args.pixel_size_m,
            downsample_factor=args.downsample_factor,
            norm_method=args.norm_method,
            exp_scale_m=args.exp_scale_m,
            compress=args.compress,
            blocksize=args.blocksize,
        )
        all_meta["waters"] = meta
    else:
        print("[SKIP] Neither rivers nor lakes subdirectory exists; skipping waters processing")

    meta_path = out_dir / args.meta_name
    meta_path.write_text(
        json.dumps(all_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nAll processing complete; parameters written to {meta_path}")

    # Stack multi-band TIFs by year
    print("\n[stack] Starting per-year multi-band TIF stacking...")
    stack_per_year(out_dir, compress=args.compress, blocksize=args.blocksize)


# ---------------------------------------------------------------------------
# Stack multi-band TIFs by year
# ---------------------------------------------------------------------------

# Band order (subdirectory names matching SOLO_CATEGORIES + waters)
BAND_ORDER: list[tuple[str, str]] = [
    ("roads",      "Roads"),
    ("powerlines", "Powerlines"),
    ("buildings",  "Buildings"),
    ("waters",     "Waters"),
]


def stack_per_year(
    out_dir: Path,
    *,
    compress: str,
    blocksize: int,
) -> None:
    """
    Scan subdirectories under out_dir, collect all bands per year,
    and write OSM_{year}.tif (multi-band) and OSM_band_order.txt.
    """
    # Collect the file path for each year and each band
    # year -> list of (band_name, path), in BAND_ORDER order
    year_bands: dict[str, list[tuple[str, Path]]] = {}

    for subdir_name, band_label in BAND_ORDER:
        subdir = out_dir / subdir_name
        if not subdir.is_dir():
            continue
        for fp in sorted(subdir.glob("*.tif")):
            year = extract_year(fp.stem)
            if year is None:
                continue
            year_bands.setdefault(year, [])
            year_bands[year].append((band_label, fp))

    if not year_bands:
        print("  [WARN] No processed files found; skipping stacking")
        return

    # Determine each year's band order: arrange according to BAND_ORDER
    band_order_names = [label for _, label in BAND_ORDER]

    for year in sorted(year_bands):
        entries = year_bands[year]
        # Sort by BAND_ORDER; skip missing bands (with a warning)
        ordered: list[tuple[str, Path]] = []
        for band_label in band_order_names:
            matched = [(n, p) for n, p in entries if n == band_label]
            if matched:
                ordered.append(matched[0])
            else:
                print(f"  [WARN] {year} missing band {band_label}; this band will not appear in OSM_{year}.tif")

        if not ordered:
            print(f"  [WARN] {year} has no bands; skipping")
            continue

        out_path = out_dir / f"OSM_{year}.tif"
        try:
            _write_stack(ordered, out_path, compress=compress, blocksize=blocksize)
            band_names = [n for n, _ in ordered]
            print(f"  [OK] OSM_{year}.tif  bands: {band_names}")
        except Exception as e:
            print(f"  [FAIL] OSM_{year}.tif: {e}")

    # Write the band-description txt (based on the first year that actually appears)
    first_year = sorted(year_bands)[0]
    actual_bands = [
        n for n in band_order_names
        if any(n == name for name, _ in year_bands[first_year])
    ]
    txt_path = out_dir / "OSM_band_order.txt"
    lines = [
        "OSM multi-band TIF band-order description",
        "=" * 40,
        f"Band count: {len(actual_bands)}",
        "",
        "Band index (1-based)  driver name",
        "-" * 40,
    ]
    for i, name in enumerate(actual_bands, start=1):
        lines.append(f"  Band {i:>2d}              {name}")
    lines += [
        "",
        "Method: Euclidean distance transform (EDT) -> mean downsampling -> log1p min-max normalize to [0, 1]",
        "Note: smaller values mean closer to the feature (after log compression); Waters = Rivers UNION Lakes",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  [OK] band-order description written to {txt_path}")


def _write_stack(
    ordered: list[tuple[str, Path]],
    out_path: Path,
    *,
    compress: str,
    blocksize: int,
) -> None:
    """Read multiple single-band TIFs and write them in order as a multi-band TIF, setting band descriptions."""
    arrays: list[np.ndarray] = []
    profile: dict | None = None

    for band_name, fp in ordered:
        with rasterio.open(fp) as src:
            arrays.append(src.read(1))
            if profile is None:
                profile = src.profile.copy()

    n_bands = len(arrays)
    profile.update(
        count=n_bands,
        dtype="float32",
        nodata=None,
        tiled=True,
        blockxsize=blocksize,
        blockysize=blocksize,
    )
    if compress != "NONE":
        profile.update(compress=compress)
        if compress == "ZSTD":
            profile.update(zstd_level=9)

    with rasterio.open(out_path, "w", **profile) as dst:
        for i, (band_name, arr) in enumerate(zip([n for n, _ in ordered], arrays), start=1):
            dst.write(arr.astype(np.float32), i)
            dst.update_tags(i, name=band_name)
            dst.set_band_description(i, band_name)


if __name__ == "__main__":
    main()
