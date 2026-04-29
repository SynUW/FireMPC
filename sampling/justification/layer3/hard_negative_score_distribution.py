#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 Plan 03 — Task 2: Hard Negative Score Distribution.

Demonstrates that the FWI-based hard negative score separates three pixel classes:
  - fire:        firemask==9 pixels (positive labels)
  - hard_neg:    nonfire pixels (firemask==5) with high FWI score (top 25%)
  - random_neg:  randomly selected nonfire pixels (firemask==5)

Score formula:
    score(pixel) = Σ wᵢ × clip(FWIᵢ / p99_i, 0, 1)
    where wᵢ, p99_i are loaded from weight_vector_w1w6.json

Outputs:
    justification/layer3/hn_score_distribution.csv
    justification/layer3/hn_score_distribution.png

Prerequisite: weight_recalibration_allYears.py must have run first.
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
WEIGHT_JSON = _REPO / "fwi" / "outputs" / "weight_vector_w1w6.json"
OUT_DIR     = _REPO / "justification" / "layer3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BAND_NAMES = ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]

# Sample representative years — 6 peak years covering different fire regimes
SAMPLE_YEARS = [2015, 2016, 2017, 2018, 2021, 2023]
ANCHOR_MONTHS_DAYS = [(7, 1), (7, 15), (8, 1), (8, 15), (9, 1)]
MAX_FIRE_PIX    = 1000
MAX_NONFIRE_PIX = 3000
RNG_SEED        = 99
HARD_NEG_THRESH = 0.75   # top 25% by score = hard negatives


# ---------------------------------------------------------------------------
# GDAL helpers (ReadRaster + frombuffer)
# ---------------------------------------------------------------------------

def _read_band_float32(path: str, band_idx: int = 1) -> tuple[np.ndarray, tuple]:
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


def _build_vrt_firemask(day: date) -> tuple[np.ndarray | None, tuple | None]:
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


def _pixel_lonlat(arr_shape: tuple, gt: tuple, mask: np.ndarray):
    rows, cols = np.where(mask)
    lons = gt[0] + (cols + 0.5) * gt[1]
    lats = gt[3] + (rows + 0.5) * gt[5]
    return lons, lats


def lookup_fwi(lons, lats, fwi_stack, fwi_gt, fwi_shape):
    H, W = fwi_shape
    rows = ((lats - fwi_gt[3]) / fwi_gt[5]).astype(np.int32)
    cols = ((lons - fwi_gt[0]) / fwi_gt[1]).astype(np.int32)
    valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    out = np.full((len(lons), 6), np.nan, dtype=np.float32)
    r, c = rows[valid], cols[valid]
    for i in range(6):
        out[valid, i] = fwi_stack[i, r, c]
    good = ~np.isnan(out).any(axis=1)
    return out[good]


# ---------------------------------------------------------------------------
# Score function
# ---------------------------------------------------------------------------

def compute_scores(fwi_vals: np.ndarray, weights: np.ndarray,
                   p99: np.ndarray) -> np.ndarray:
    """Compute hard negative score for each pixel.

    score = sum_i w_i * clip(FWI_i / p99_i, 0, 1)
    """
    norm = np.clip(fwi_vals / (p99 + 1e-9), 0.0, 1.0)
    return (norm * weights).sum(axis=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load weights and p99 thresholds
    if not WEIGHT_JSON.exists():
        print(f"ERROR: {WEIGHT_JSON} not found.")
        print("Run weight_recalibration_allYears.py first.")
        sys.exit(1)

    with open(WEIGHT_JSON) as f:
        wdata = json.load(f)

    weights = np.array([wdata[n] for n in BAND_NAMES], dtype=np.float64)
    p99_dict = wdata.get("p99_normalization_thresholds", {})
    if not p99_dict:
        print("WARNING: No p99 thresholds in weight JSON. Using fallback values.")
        # Fallback: reasonable FWI component maxima
        p99_dict = {"FFMC": 95.0, "DMC": 200.0, "DC": 700.0,
                    "ISI": 20.0, "BUI": 200.0, "FWI": 50.0}
    p99 = np.array([p99_dict[n] for n in BAND_NAMES], dtype=np.float64)

    print(f"Loaded weights: { {n: round(float(w), 4) for n, w in zip(BAND_NAMES, weights)} }")
    print(f"p99 thresholds: { {n: v for n, v in zip(BAND_NAMES, p99)} }")

    rng = np.random.default_rng(RNG_SEED)

    fire_scores    = []
    nonfire_scores = []   # all nonfire (for random_neg selection)

    sample_days = [
        date(year, m, d)
        for year in SAMPLE_YEARS
        for m, d in ANCHOR_MONTHS_DAYS
    ]
    total = len(sample_days)
    days_used = 0

    for i, day in enumerate(sample_days):
        fwi_path = FWI_DIR / f"fwis_{day.strftime('%Y_%m_%d')}.tif"
        if not fwi_path.exists():
            continue

        # Load FWI
        fwi_bands = []
        fwi_gt = None
        fwi_shape = None
        for b in range(1, 7):
            arr, gt = _read_band_float32(str(fwi_path), band_idx=b)
            fwi_bands.append(arr)
            if fwi_gt is None:
                fwi_gt = gt
                fwi_shape = arr.shape
        fwi_stack = np.stack(fwi_bands, axis=0)

        # Load firemask
        fire_arr, fire_gt = _build_vrt_firemask(day)
        if fire_arr is None:
            continue

        fire_mask    = (fire_arr == 9)
        nonfire_mask = (fire_arr == 5)
        if fire_mask.sum() == 0:
            continue

        fire_lons, fire_lats       = _pixel_lonlat(fire_arr.shape, fire_gt, fire_mask)
        nonfire_lons, nonfire_lats = _pixel_lonlat(fire_arr.shape, fire_gt, nonfire_mask)

        if len(fire_lons) > MAX_FIRE_PIX:
            idx = rng.choice(len(fire_lons), MAX_FIRE_PIX, replace=False)
            fire_lons, fire_lats = fire_lons[idx], fire_lats[idx]
        if len(nonfire_lons) > MAX_NONFIRE_PIX:
            idx = rng.choice(len(nonfire_lons), MAX_NONFIRE_PIX, replace=False)
            nonfire_lons, nonfire_lats = nonfire_lons[idx], nonfire_lats[idx]

        fv = lookup_fwi(fire_lons, fire_lats, fwi_stack, fwi_gt, fwi_shape)
        nv = lookup_fwi(nonfire_lons, nonfire_lats, fwi_stack, fwi_gt, fwi_shape)

        if len(fv) >= 5:
            fire_scores.append(compute_scores(fv, weights, p99))
        if len(nv) >= 5:
            nonfire_scores.append(compute_scores(nv, weights, p99))

        days_used += 1
        sys.stdout.write(
            f"\r  [{i+1:2d}/{total}] {day}  fire_days={days_used}  "
            f"fire_px={sum(len(x) for x in fire_scores) if fire_scores else 0}"
        )
        sys.stdout.flush()

    print(f"\nDays with data: {days_used}")

    if not fire_scores or not nonfire_scores:
        print("ERROR: insufficient data. Exiting.")
        sys.exit(1)

    fire_all    = np.concatenate(fire_scores)
    nonfire_all = np.concatenate(nonfire_scores)

    # Hard negatives = top HARD_NEG_THRESH percentile of nonfire
    thresh = np.percentile(nonfire_all, HARD_NEG_THRESH * 100)
    hard_neg = nonfire_all[nonfire_all >= thresh]
    random_neg = nonfire_all

    print(f"\nFire pixels:      {len(fire_all):,}  "
          f"mean_score={np.mean(fire_all):.3f}")
    print(f"Hard neg pixels:  {len(hard_neg):,}  "
          f"mean_score={np.mean(hard_neg):.3f}  (top {int((1-HARD_NEG_THRESH)*100)}%)")
    print(f"Random neg pixels:{len(random_neg):,}  "
          f"mean_score={np.mean(random_neg):.3f}")

    # ---------------------------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------------------------
    rows = []
    for s in fire_all:
        rows.append({"pixel_type": "fire", "score": float(s)})
    for s in hard_neg:
        rows.append({"pixel_type": "hard_neg", "score": float(s)})
    # Sample random_neg to same size as hard_neg for CSV balance
    rng2 = np.random.default_rng(42)
    rand_sample = rng2.choice(random_neg, size=min(len(hard_neg), len(random_neg)),
                              replace=False)
    for s in rand_sample:
        rows.append({"pixel_type": "random_neg", "score": float(s)})

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "hn_score_distribution.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nScore distribution CSV saved: {csv_path}  ({len(df):,} rows)")

    # ---------------------------------------------------------------------------
    # Plot KDE
    # ---------------------------------------------------------------------------
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(9, 5))

    def plot_kde(data, color, label, lw=2.0):
        if len(data) < 5:
            return
        kde = gaussian_kde(data, bw_method=0.15)
        xs = np.linspace(0, 1, 500)
        ys = kde(xs)
        ax.plot(xs, ys, color=color, lw=lw, label=label)
        ax.fill_between(xs, 0, ys, color=color, alpha=0.15)

    plot_kde(random_neg, "#5b9bd5", "Random negatives (firemask=5)")
    plot_kde(hard_neg,   "#ed7d31", f"Hard negatives (top {int((1-HARD_NEG_THRESH)*100)}% by score)")
    plot_kde(fire_all,   "#c00000", "Fire pixels (firemask=9)",  lw=2.5)

    ax.axvline(thresh, color="#ed7d31", lw=1.2, ls="--", alpha=0.7,
               label=f"Hard-neg threshold ({thresh:.2f})")
    ax.set_xlabel("Hard negative score  Σ wᵢ × clip(FWIᵢ / p99ᵢ, 0, 1)")
    ax.set_ylabel("Density")
    ax.set_title(
        "Layer 3: FWI Hard Negative Score Distribution\n"
        f"Fire vs. hard negatives vs. random negatives  "
        f"({', '.join(str(y) for y in SAMPLE_YEARS)})"
    )
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)

    # Summary stats box
    stats_text = (
        f"Mean scores:\n"
        f"  Fire:       {np.mean(fire_all):.3f}\n"
        f"  Hard neg:   {np.mean(hard_neg):.3f}\n"
        f"  Random neg: {np.mean(random_neg):.3f}"
    )
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    plt.tight_layout()
    png_path = OUT_DIR / "hn_score_distribution.png"
    fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {png_path}")

    print("\nTask 2 complete.")


if __name__ == "__main__":
    main()
