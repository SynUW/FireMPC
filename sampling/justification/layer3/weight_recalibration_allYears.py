#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 Plan 03 — Task 1: FWI Weight Recalibration (all years 2000-2025).

Recalibrates the w1-w6 weight vector using ALL fire years 2000-2025.
Reads FWI from precomputed fwi_components/ tiles (NOT ERA5).
Reads fire detections from MCD14A1_combined (firemask band 1).

Method: normalized separation score per FWI component
    sep_i = max(0, (mean_fire_i - mean_nonfire_i) / std_nonfire_i)
    w_i   = sep_i / sum(sep)

Sample strategy: 3 anchor days per year (Jul 15, Aug 01, Aug 15)
= 78 sample days total across 2000-2025 (~3 per year * 26 years)

Outputs:
    fwi/outputs/weight_vector_w1w6.json          (OVERWRITES Phase 1 version)
    justification/layer3/separation_scores_allYears.csv
    justification/layer3/separation_scores_allYears.png
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from osgeo import gdal

gdal.UseExceptions()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO = Path("/mnt/raid/user/hackathons/dataset_building_entire_canada/sampling")
MODIS_DIR = Path(
    "/s3/user/wildfire_dataset/self_built_materials/"
    "bc_raw_data_new_fast/drivers/processed/MCD14A1_combined"
)
FWI_DIR = Path(
    "/s3/user/wildfire_dataset/self_built_materials/"
    "bc_raw_data_new_fast/drivers/processed/fwi_components"
)
OUT_JSON = _REPO / "fwi" / "outputs" / "weight_vector_w1w6.json"
OUT_DIR  = _REPO / "justification" / "layer3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BAND_NAMES = ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]
YEARS      = list(range(2000, 2026))          # 2000-2025 inclusive
ANCHOR_MONTHS_DAYS = [(7, 15), (8, 1), (8, 15)]   # Jul 15, Aug 1, Aug 15
MAX_FIRE_PIX    = 2000    # max fire pixels sampled per day
MAX_NONFIRE_PIX = 4000    # max nonfire pixels sampled per day
RNG_SEED = 42

# FWI NoData (use 0 as FWI can be 0 on non-fire days, but we use NaN for missing)
FWI_NODATA = -9999.0

# ---------------------------------------------------------------------------
# Low-level GDAL reads (ReadRaster + frombuffer, never ReadAsArray)
# ---------------------------------------------------------------------------

def _read_band_float32(path: str, band_idx: int = 1) -> tuple[np.ndarray, tuple]:
    """Read a single band as float32. Returns (array, geotransform)."""
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open: {path}")
    gt = ds.GetGeoTransform()
    W, H = ds.RasterXSize, ds.RasterYSize
    raw = ds.GetRasterBand(band_idx).ReadRaster(
        0, 0, W, H, buf_type=gdal.GDT_Float32)
    arr = np.frombuffer(raw, dtype=np.float32).reshape(H, W).copy()
    nd = ds.GetRasterBand(band_idx).GetNoDataValue()
    ds = None
    if nd is not None:
        arr[arr == np.float32(nd)] = np.nan
    return arr, gt


def _read_band_byte(path: str, band_idx: int = 1) -> tuple[np.ndarray, tuple]:
    """Read a single band as uint8. Returns (array, geotransform)."""
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open: {path}")
    gt = ds.GetGeoTransform()
    W, H = ds.RasterXSize, ds.RasterYSize
    raw = ds.GetRasterBand(band_idx).ReadRaster(
        0, 0, W, H, buf_type=gdal.GDT_Byte)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(H, W).copy()
    ds = None
    return arr, gt


# ---------------------------------------------------------------------------
# MCD14A1 mosaic helpers
# ---------------------------------------------------------------------------

def _build_vrt_firemask(day: date) -> np.ndarray | None:
    """Build a VRT from 4 MCD14A1 tiles for `day`, read firemask band.

    Returns a float32 array mosaic, or None if no tiles found.
    Uses gdal.BuildVRT to correctly handle overlapping/non-contiguous tiles.
    """
    dstr = day.strftime("%Y_%m_%d")
    tile_paths = sorted(MODIS_DIR.glob(f"MCD14A1_{dstr}_tile*.tif"))
    if not tile_paths:
        return None, None

    vrt_path = f"/vsimem/modis_{dstr}.vrt"
    vrt_opts = gdal.BuildVRTOptions(resolution="highest", resampleAlg="nearest",
                                    addAlpha=False)
    vrt_ds = gdal.BuildVRT(vrt_path, [str(p) for p in tile_paths], options=vrt_opts)
    if vrt_ds is None:
        return None, None

    gt = vrt_ds.GetGeoTransform()
    W, H = vrt_ds.RasterXSize, vrt_ds.RasterYSize
    raw = vrt_ds.GetRasterBand(1).ReadRaster(0, 0, W, H, buf_type=gdal.GDT_Byte)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(H, W).copy()
    vrt_ds = None
    gdal.Unlink(vrt_path)
    return arr, gt


# ---------------------------------------------------------------------------
# Pixel lookup: (lon, lat) → raster row/col
# ---------------------------------------------------------------------------

def _lonlat_to_rc(lons: np.ndarray, lats: np.ndarray, gt: tuple
                  ) -> tuple[np.ndarray, np.ndarray]:
    cols = ((lons - gt[0]) / gt[1]).astype(np.int32)
    rows = ((lats - gt[3]) / gt[5]).astype(np.int32)
    return rows, cols


def _pixel_lonlat(arr_shape: tuple, gt: tuple,
                   mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return centroid lon/lat for pixels where mask is True."""
    rows, cols = np.where(mask)
    lons = gt[0] + (cols + 0.5) * gt[1]
    lats = gt[3] + (rows + 0.5) * gt[5]
    return lons, lats


# ---------------------------------------------------------------------------
# Per-day sampling
# ---------------------------------------------------------------------------

def sample_day(day: date, rng: np.random.Generator
               ) -> dict[str, np.ndarray] | None:
    """
    Returns dict with keys 'fire' and 'nonfire', each a (N, 6) array of
    [FFMC, DMC, DC, ISI, BUI, FWI] values.  Returns None if data missing.
    """
    # --- 1. Load FWI tile (full Canada mosaic, single file) ---
    fwi_path = FWI_DIR / f"fwis_{day.strftime('%Y_%m_%d')}.tif"
    if not fwi_path.exists():
        return None

    fwi_bands = []
    fwi_gt = None
    fwi_shape = None
    for b in range(1, 7):
        arr, gt = _read_band_float32(str(fwi_path), band_idx=b)
        fwi_bands.append(arr)
        if fwi_gt is None:
            fwi_gt = gt
            fwi_shape = arr.shape
    fwi_stack = np.stack(fwi_bands, axis=0)  # (6, H, W)

    # --- 2. Load MCD14A1 firemask mosaic ---
    fire_arr, fire_gt = _build_vrt_firemask(day)
    if fire_arr is None:
        return None

    # --- 3. Find fire / nonfire pixel centroids ---
    fire_mask    = (fire_arr == 9)
    nonfire_mask = (fire_arr == 5)

    if fire_mask.sum() == 0:
        return None   # no fire pixels this day — skip

    fire_lons,    fire_lats    = _pixel_lonlat(fire_arr.shape, fire_gt,    fire_mask)
    nonfire_lons, nonfire_lats = _pixel_lonlat(fire_arr.shape, fire_gt,    nonfire_mask)

    # Sub-sample to MAX_* limits
    if len(fire_lons) > MAX_FIRE_PIX:
        idx = rng.choice(len(fire_lons), MAX_FIRE_PIX, replace=False)
        fire_lons, fire_lats = fire_lons[idx], fire_lats[idx]
    if len(nonfire_lons) > MAX_NONFIRE_PIX:
        idx = rng.choice(len(nonfire_lons), MAX_NONFIRE_PIX, replace=False)
        nonfire_lons, nonfire_lats = nonfire_lons[idx], nonfire_lats[idx]

    # --- 4. Look up FWI at pixel locations ---
    def lookup_fwi(lons, lats):
        H, W = fwi_shape
        rows = ((lats - fwi_gt[3]) / fwi_gt[5]).astype(np.int32)
        cols = ((lons - fwi_gt[0]) / fwi_gt[1]).astype(np.int32)
        valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
        out = np.full((len(lons), 6), np.nan, dtype=np.float32)
        r, c = rows[valid], cols[valid]
        for i in range(6):
            out[valid, i] = fwi_stack[i, r, c]
        # Drop rows with any NaN
        good = ~np.isnan(out).any(axis=1)
        return out[good]

    fire_vals    = lookup_fwi(fire_lons,    fire_lats)
    nonfire_vals = lookup_fwi(nonfire_lons, nonfire_lats)

    if len(fire_vals) < 10 or len(nonfire_vals) < 10:
        return None

    return {"fire": fire_vals, "nonfire": nonfire_vals}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(RNG_SEED)

    # Collect all sample days
    sample_days = []
    for year in YEARS:
        for month, day_of_month in ANCHOR_MONTHS_DAYS:
            try:
                d = date(year, month, day_of_month)
                sample_days.append(d)
            except ValueError:
                pass
    print(f"Recalibrating weights using {len(sample_days)} anchor days "
          f"across years {YEARS[0]}-{YEARS[-1]}")

    # Accumulate fire and nonfire arrays
    all_fire    = []
    all_nonfire = []
    days_used   = 0

    for i, d in enumerate(sample_days):
        result = sample_day(d, rng)
        sys.stdout.write(
            f"\r  [{i+1:3d}/{len(sample_days)}] {d}  "
            f"days_used={days_used}  "
            f"fire_px={sum(len(x) for x in all_fire) if all_fire else 0}"
        )
        sys.stdout.flush()
        if result is None:
            continue
        all_fire.append(result["fire"])
        all_nonfire.append(result["nonfire"])
        days_used += 1

    print(f"\nDays with data: {days_used} / {len(sample_days)}")

    if days_used == 0:
        print("ERROR: No valid days found. Exiting.")
        sys.exit(1)

    fire_vals    = np.concatenate(all_fire,    axis=0)
    nonfire_vals = np.concatenate(all_nonfire, axis=0)
    print(f"Fire pixels:    {len(fire_vals):,}")
    print(f"Nonfire pixels: {len(nonfire_vals):,}")

    # ---------------------------------------------------------------------------
    # Compute separation scores
    # ---------------------------------------------------------------------------
    records = []
    for i, name in enumerate(BAND_NAMES):
        fv = fire_vals[:, i]
        nv = nonfire_vals[:, i]
        # Drop NaN
        fv = fv[~np.isnan(fv)]
        nv = nv[~np.isnan(nv)]
        if len(fv) == 0 or len(nv) == 0:
            records.append(dict(component=name, fire_mean=np.nan,
                                nonfire_mean=np.nan, std=np.nan, sep_score=0.0))
            continue
        fire_mean    = float(np.mean(fv))
        nonfire_mean = float(np.mean(nv))
        std_nonfire  = float(np.std(nv))
        sep = max(0.0, (fire_mean - nonfire_mean) / (std_nonfire + 1e-9))
        records.append(dict(
            component=name,
            fire_mean=round(fire_mean, 4),
            nonfire_mean=round(nonfire_mean, 4),
            std=round(std_nonfire, 4),
            sep_score=round(sep, 6),
        ))

    df = pd.DataFrame(records)
    total_sep = df["sep_score"].sum()
    df["weight"] = (df["sep_score"] / total_sep).round(6)

    print("\nSeparation scores and weights:")
    print(df.to_string(index=False))

    # ---------------------------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------------------------
    csv_path = OUT_DIR / "separation_scores_allYears.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSeparation scores saved: {csv_path}")

    # ---------------------------------------------------------------------------
    # Compare with Phase 1 (2017/2018/2021) weights
    # ---------------------------------------------------------------------------
    phase1_weights = {
        "FFMC": 0.209029, "DMC": 0.151368, "DC": 0.164503,
        "ISI": 0.150533, "BUI": 0.161320, "FWI": 0.163247
    }

    # ---------------------------------------------------------------------------
    # Plot comparison
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: weights comparison
    ax = axes[0]
    x = np.arange(len(BAND_NAMES))
    w = 0.35
    new_weights  = df["weight"].values
    old_weights  = [phase1_weights[n] for n in BAND_NAMES]
    bars1 = ax.bar(x - w/2, old_weights, w, label="2017/18/21 (Phase 1)",
                   color="#5b9bd5", alpha=0.85)
    bars2 = ax.bar(x + w/2, new_weights, w, label="2000–2025 (All years)",
                   color="#ed7d31", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(BAND_NAMES)
    ax.set_ylabel("Normalized weight")
    ax.set_title("FWI Component Weights: Phase 1 vs. All-Year Recalibration")
    ax.legend()
    ax.set_ylim(0, 0.35)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    # Right: separation scores
    ax2 = axes[1]
    ax2.bar(BAND_NAMES, df["sep_score"].values, color="#70ad47", alpha=0.85)
    ax2.set_ylabel("Separation score  max(0, (μ_fire − μ_nonfire) / σ_nonfire)")
    ax2.set_title("Per-component Separation Scores (2000–2025)")

    plt.tight_layout()
    png_path = OUT_DIR / "separation_scores_allYears.png"
    fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {png_path}")

    # ---------------------------------------------------------------------------
    # Compute p99 normalization thresholds (stored in weight JSON for Layer 3)
    # ---------------------------------------------------------------------------
    p99_thresholds = {}
    for i, name in enumerate(BAND_NAMES):
        fv = fire_vals[:, i]
        fv = fv[~np.isnan(fv)]
        # Use union of fire + nonfire for p99 to cover full range
        nv = nonfire_vals[:, i]
        nv = nv[~np.isnan(nv)]
        all_vals = np.concatenate([fv, nv])
        p99 = float(np.percentile(all_vals, 99))
        p99_thresholds[name] = round(p99, 4)
    print(f"\np99 normalization thresholds: {p99_thresholds}")

    # ---------------------------------------------------------------------------
    # Save updated weight_vector_w1w6.json
    # ---------------------------------------------------------------------------
    weight_dict = {
        name: float(df.loc[df["component"] == name, "weight"].iloc[0])
        for name in BAND_NAMES
    }
    weight_dict["sum_check"] = round(sum(weight_dict[n] for n in BAND_NAMES), 6)
    weight_dict["calibration_years"] = list(range(2000, 2026))
    weight_dict["n_days_used"]       = days_used
    weight_dict["fire_pixels"]       = int(len(fire_vals))
    weight_dict["nonfire_pixels"]    = int(len(nonfire_vals))
    weight_dict["method"] = (
        "normalized_separation_score: max(0, (mean_fire - mean_nonfire) / std_nonfire)"
    )
    weight_dict["p99_normalization_thresholds"] = p99_thresholds

    with open(OUT_JSON, "w") as f:
        json.dump(weight_dict, f, indent=2)
    print(f"\nWeight vector saved: {OUT_JSON}")
    print(f"  Weights:")
    for name in BAND_NAMES:
        print(f"    {name}: {weight_dict[name]:.6f}")
    print(f"  Sum: {weight_dict['sum_check']}")
    print(f"  Calibration years: 2000–2025 ({days_used} days used)")


if __name__ == "__main__":
    main()
