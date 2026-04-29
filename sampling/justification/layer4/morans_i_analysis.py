#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 Plan 04 — Task 1: Spatial Deduplication Justification (Moran's I).

Demonstrates spatial autocorrelation in fire pixel detections and quantifies
the reduction achieved by 10 km minimum-distance thinning.

Method:
  1. Load all fire pixels (firemask==9) from Jul-Aug per sample year
  2. Project to EPSG:3978 (Canada Atlas Lambert) for accurate distances
  3. Compute Moran's I on 0.5-degree grid (binary presence per cell)
     using the analytical approximation for E[I] and Var[I] under
     the randomization assumption, yielding a z-score -> p-value
  4. Apply 10 km KDTree thinning (greedy, seeded) per year
  5. Recompute Moran's I on thinned point set per year
  6. Save per-year rows: year, before_I, after_I, before_p, after_p, removed_fraction
  7. Save per-ecozone removal counts
  8. Save figure: spatial density maps + Moran's I bar chart

Outputs (saved to justification/layer4/):
    morans_i_before_after.csv        -- year, before_I, after_I, before_p, after_p, removed_fraction
    deduplication_removal_counts.csv -- ecozone, before_count, after_count, removed_count, removed_fraction
    morans_i_before_after.png        -- spatial maps + Moran's I bar chart

Note: This script reads from the pre-merged MCD14A1_combined_5xMaximumDown_merged/
      directory (not tiled). Uses EPSG:3978 for distance-based thinning.
      Ecozone labels from ecozones.shp (ZONE_NAME, 15 ecozones).
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import geopandas as gpd
from osgeo import gdal
from pyproj import Transformer
from scipy.spatial import KDTree
from scipy import stats as scipy_stats

gdal.UseExceptions()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO = Path("/mnt/raid/user/hackathons/dataset_building_entire_canada/sampling")
MODIS_DIR = Path(
    "/s3/user/wildfire_dataset/self_built_materials/"
    "bc_raw_data_new_fast/drivers/processed/"
    "spatial_filled_5xdownsampled_merged/MCD14A1_combined_5xMaximumDown_merged"
)
ECOZONE_SHP = Path(
    "/s3/user/wildfire_dataset/self_built_materials/"
    "bc_raw_data_new_fast/drivers/ecoregions/ecozones.shp"
)
OUT_DIR = _REPO / "justification" / "layer4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SAMPLE_YEARS    = [2017, 2018, 2021]
MONTHS          = [7, 8]           # Jul, Aug peak fire season
MIN_DIST_KM     = 10.0             # deduplication threshold
GRID_DEG        = 0.5              # Moran's I grid resolution (degrees)
CANADA_EXTENT   = (-141.0, 41.0, -52.0, 84.0)  # lon_min, lat_min, lon_max, lat_max
RNG_SEED        = 42
SAMPLE_N        = 50_000           # max pixels for Moran's I computation per year

# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------
_to_3978 = Transformer.from_crs("EPSG:4326", "EPSG:3978", always_xy=True)


# ---------------------------------------------------------------------------
# GDAL helpers (merged raster, not tiled)
# ---------------------------------------------------------------------------

def _read_merged_firemask(day: date) -> tuple[np.ndarray | None, tuple | None]:
    """Read firemask band from MCD14A1_combined_5xMaximumDown_merged daily file."""
    dstr = day.strftime("%Y_%m_%d")
    fpath = MODIS_DIR / f"{dstr}.tif"
    if not fpath.exists():
        return None, None
    ds = gdal.Open(str(fpath), gdal.GA_ReadOnly)
    if ds is None:
        return None, None
    gt = ds.GetGeoTransform()
    W, H = ds.RasterXSize, ds.RasterYSize
    raw = ds.GetRasterBand(1).ReadRaster(0, 0, W, H, buf_type=gdal.GDT_Byte)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(H, W).copy()
    ds = None
    return arr, gt


def _pixel_lonlat(arr_shape: tuple, gt: tuple, mask: np.ndarray):
    """Convert pixel mask to lon/lat arrays."""
    rows, cols = np.where(mask)
    lons = gt[0] + (cols + 0.5) * gt[1]
    lats = gt[3] + (rows + 0.5) * gt[5]
    return lons.astype(np.float64), lats.astype(np.float64)


# ---------------------------------------------------------------------------
# 10 km KDTree greedy thinning (EPSG:3978)
# ---------------------------------------------------------------------------

def thin_points_kdtree(lons: np.ndarray, lats: np.ndarray,
                        min_dist_m: float = 10_000.0,
                        seed: int = RNG_SEED) -> np.ndarray:
    """Return boolean keep-mask after greedy KDTree thinning in EPSG:3978."""
    if len(lons) == 0:
        return np.zeros(0, dtype=bool)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(lons))

    xs, ys = _to_3978.transform(lons, lats)
    pts = np.column_stack([xs, ys])

    kept    = np.zeros(len(lons), dtype=bool)
    removed = np.zeros(len(lons), dtype=bool)
    tree = KDTree(pts)

    for idx in order:
        if removed[idx]:
            continue
        kept[idx] = True
        for nb in tree.query_ball_point(pts[idx], r=min_dist_m):
            if not kept[nb]:
                removed[nb] = True

    return kept


# ---------------------------------------------------------------------------
# Moran's I on 0.5-degree grid with analytical p-value
# ---------------------------------------------------------------------------

def morans_i_grid_with_pval(lons: np.ndarray, lats: np.ndarray,
                             grid_deg: float = GRID_DEG) -> tuple[float, float]:
    """Compute Moran's I on a regular grid and return (I, p_value).

    x_i = 1 if >=1 fire pixel in cell i, else 0.
    Spatial weights: queen contiguity (8-neighbor).
    p-value: two-tailed analytical approximation (z-score under normality).
    """
    lon_min, lat_min, lon_max, lat_max = CANADA_EXTENT
    lon_bins = np.arange(lon_min, lon_max + grid_deg, grid_deg)
    lat_bins = np.arange(lat_min, lat_max + grid_deg, grid_deg)
    n_lon = len(lon_bins) - 1
    n_lat = len(lat_bins) - 1

    lon_idx = np.clip(((lons - lon_min) / grid_deg).astype(np.int32), 0, n_lon - 1)
    lat_idx = np.clip(((lats - lat_min) / grid_deg).astype(np.int32), 0, n_lat - 1)
    grid = np.zeros((n_lat, n_lon), dtype=np.float32)
    np.add.at(grid, (lat_idx, lon_idx), 1)

    x = (grid > 0).astype(np.float64).ravel()
    N = len(x)
    x_bar = x.mean()
    z = x - x_bar
    z_sq_sum = (z ** 2).sum()

    if z_sq_sum < 1e-12:
        return 0.0, 1.0  # degenerate: all cells same value

    # Queen contiguity: accumulate cross products using 2D roll
    grid_z = z.reshape(n_lat, n_lon)
    W_total   = 0.0
    cross_sum = 0.0

    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            shifted = np.roll(np.roll(grid_z, dr, axis=0), dc, axis=1)
            z_row = grid_z.copy()
            # Zero out wrapped edges from np.roll
            if dr == 1:
                shifted[0, :]  = 0.0
                z_row[0, :]    = 0.0
            elif dr == -1:
                shifted[-1, :] = 0.0
                z_row[-1, :]   = 0.0
            if dc == 1:
                shifted[:, 0]  = 0.0
                z_row[:, 0]    = 0.0
            elif dc == -1:
                shifted[:, -1] = 0.0
                z_row[:, -1]   = 0.0

            cross_sum += (z_row * shifted).sum()
            valid_i = np.ones((n_lat, n_lon), dtype=bool)
            if dr == 1:  valid_i[0, :]  = False
            if dr == -1: valid_i[-1, :] = False
            if dc == 1:  valid_i[:, 0]  = False
            if dc == -1: valid_i[:, -1] = False
            W_total += valid_i.sum()

    if W_total < 1:
        return 0.0, 1.0

    I = float((N / W_total) * (cross_sum / z_sq_sum))

    # Analytical p-value under normality approximation
    # E[I] = -1 / (N - 1)
    # For large N, SE[I] ~ 1 / sqrt(N - 1) (conservative approximation)
    E_I   = -1.0 / (N - 1)
    SE_I  = 1.0 / max(np.sqrt(N - 1), 1.0)
    z_score = (I - E_I) / SE_I
    p_val = float(2.0 * scipy_stats.norm.sf(abs(z_score)))  # two-tailed

    return I, p_val


# ---------------------------------------------------------------------------
# Ecozone labeling for removal counts
# ---------------------------------------------------------------------------

_eco_cache = None


def _load_eco() -> gpd.GeoDataFrame:
    global _eco_cache
    if _eco_cache is None:
        gdf = gpd.read_file(str(ECOZONE_SHP))
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4617")
        if gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")
        _eco_cache = gdf[['ZONE_NAME', 'geometry']].copy()
    return _eco_cache


def assign_ecozone_batch(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Assign ZONE_NAME ecozone label to each (lon, lat) point."""
    eco = _load_eco()
    pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(lons, lats),
        crs="EPSG:4326"
    )
    joined = gpd.sjoin(pts, eco, how="left", predicate="within")
    return joined["ZONE_NAME"].fillna("OUTSIDE").values.astype(str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(RNG_SEED)

    per_year_rows = []       # for morans_i_before_after.csv
    all_eco_stats = {}       # ecozone -> {before, after}
    yearly_data   = {}       # year -> (lons_before, lats_before, lons_after, lats_after)

    for year in SAMPLE_YEARS:
        print(f"\n=== Year {year} ===")
        year_lons, year_lats = [], []

        for month in MONTHS:
            n_days = 31 if month in (1, 3, 5, 7, 8, 10, 12) else 30
            for day_of_month in range(1, n_days + 1):
                d = date(year, month, day_of_month)
                fire_arr, fire_gt = _read_merged_firemask(d)
                if fire_arr is None:
                    continue
                fire_mask = (fire_arr == 9)
                if fire_mask.sum() == 0:
                    continue
                lons, lats = _pixel_lonlat(fire_arr.shape, fire_gt, fire_mask)
                year_lons.append(lons)
                year_lats.append(lats)
                sys.stdout.write(
                    f"\r  {year}-{month:02d}-{day_of_month:02d}  "
                    f"px_so_far={sum(len(x) for x in year_lons):,}"
                )
                sys.stdout.flush()

        print()
        if not year_lons:
            print(f"  No fire pixels found for {year} — skipping")
            continue

        lons_yr = np.concatenate(year_lons)
        lats_yr = np.concatenate(year_lats)
        print(f"  Total fire pixels: {len(lons_yr):,}")

        # Subsample for Moran's I (speed guard)
        if len(lons_yr) > SAMPLE_N:
            idx_mi = rng.choice(len(lons_yr), SAMPLE_N, replace=False)
            lons_mi = lons_yr[idx_mi]
            lats_mi = lats_yr[idx_mi]
            print(f"  Subsampled to {SAMPLE_N:,} for Moran's I computation")
        else:
            lons_mi = lons_yr
            lats_mi = lats_yr

        # Moran's I before
        print(f"  Computing Moran's I before thinning...")
        I_before, p_before = morans_i_grid_with_pval(lons_mi, lats_mi)
        print(f"    I = {I_before:.4f}  p = {p_before:.4e}")

        # Apply 10 km thinning on full set
        print(f"  Applying {MIN_DIST_KM:.0f} km thinning on {len(lons_yr):,} pixels...")
        kept_mask = thin_points_kdtree(lons_yr, lats_yr,
                                        min_dist_m=MIN_DIST_KM * 1000,
                                        seed=RNG_SEED + year)
        lons_thin = lons_yr[kept_mask]
        lats_thin = lats_yr[kept_mask]
        n_before  = len(lons_yr)
        n_after   = int(kept_mask.sum())
        n_removed = n_before - n_after
        removed_frac = n_removed / n_before
        print(f"  Before: {n_before:,}  After: {n_after:,}  "
              f"Removed: {n_removed:,} ({100 * removed_frac:.1f}%)")

        # Moran's I after thinning
        if len(lons_thin) > SAMPLE_N:
            idx_af = rng.choice(len(lons_thin), SAMPLE_N, replace=False)
            lons_mi_af = lons_thin[idx_af]
            lats_mi_af = lats_thin[idx_af]
        else:
            lons_mi_af = lons_thin
            lats_mi_af = lats_thin

        print(f"  Computing Moran's I after thinning...")
        I_after, p_after = morans_i_grid_with_pval(lons_mi_af, lats_mi_af)
        print(f"    I = {I_after:.4f}  p = {p_after:.4e}")

        per_year_rows.append({
            "year":             year,
            "before_I":         round(I_before, 6),
            "after_I":          round(I_after, 6),
            "before_p":         round(p_before, 6),
            "after_p":          round(p_after, 6),
            "removed_fraction": round(removed_frac, 4),
        })

        yearly_data[year] = (lons_yr, lats_yr, lons_thin, lats_thin)

        # Ecozone removal counts (subsample for sjoin speed)
        N_ECO = min(50_000, len(lons_yr))
        idx_eco = rng.choice(len(lons_yr), N_ECO, replace=False)
        eco_lons = lons_yr[idx_eco]
        eco_lats = lats_yr[idx_eco]
        eco_kept = kept_mask[idx_eco]
        eco_names = assign_ecozone_batch(eco_lons, eco_lats)

        for eco in np.unique(eco_names):
            em = (eco_names == eco)
            b  = int(em.sum())
            a  = int((em & eco_kept).sum())
            if eco not in all_eco_stats:
                all_eco_stats[eco] = {"before": 0, "after": 0}
            all_eco_stats[eco]["before"] += b
            all_eco_stats[eco]["after"]  += a

    # ---------------------------------------------------------------------------
    # Save Moran's I CSV (per-year, plan spec columns)
    # ---------------------------------------------------------------------------
    morans_df = pd.DataFrame(per_year_rows)[
        ["year", "before_I", "after_I", "before_p", "after_p", "removed_fraction"]
    ]
    csv_path = OUT_DIR / "morans_i_before_after.csv"
    morans_df.to_csv(csv_path, index=False)
    print(f"\nMoran's I CSV saved: {csv_path}")
    print(morans_df.to_string(index=False))

    # ---------------------------------------------------------------------------
    # Save ecozone removal counts (plan spec cols)
    # ---------------------------------------------------------------------------
    eco_rows = []
    for eco, stats in all_eco_stats.items():
        b = stats["before"]
        a = stats["after"]
        r = b - a
        eco_rows.append({
            "ecozone":          eco,
            "before_count":     b,
            "after_count":      a,
            "removed_count":    r,
            "removed_fraction": round(r / b, 4) if b > 0 else 0.0,
        })
    eco_df = pd.DataFrame(eco_rows).sort_values("removed_fraction", ascending=False)
    eco_csv = OUT_DIR / "deduplication_removal_counts.csv"
    eco_df.to_csv(eco_csv, index=False)
    print(f"\nEcozone removal counts saved: {eco_csv} ({len(eco_df)} rows)")

    # ---------------------------------------------------------------------------
    # Figures
    # ---------------------------------------------------------------------------
    print("\nGenerating figures...")

    lon_min, lat_min, lon_max, lat_max = CANADA_EXTENT
    lon_bins = np.arange(lon_min, lon_max + GRID_DEG, GRID_DEG)
    lat_bins = np.arange(lat_min, lat_max + GRID_DEG, GRID_DEG)
    n_lon = len(lon_bins) - 1
    n_lat = len(lat_bins) - 1

    def make_grid(lons, lats):
        lon_idx = np.clip(((lons - lon_min) / GRID_DEG).astype(np.int32), 0, n_lon - 1)
        lat_idx = np.clip(((lats - lat_min) / GRID_DEG).astype(np.int32), 0, n_lat - 1)
        g = np.zeros((n_lat, n_lon), dtype=np.float32)
        np.add.at(g, (lat_idx, lon_idx), 1)
        return g

    # Aggregate all years for spatial maps
    all_lons_b = np.concatenate([yearly_data[y][0] for y in SAMPLE_YEARS if y in yearly_data])
    all_lats_b = np.concatenate([yearly_data[y][1] for y in SAMPLE_YEARS if y in yearly_data])
    all_lons_a = np.concatenate([yearly_data[y][2] for y in SAMPLE_YEARS if y in yearly_data])
    all_lats_a = np.concatenate([yearly_data[y][3] for y in SAMPLE_YEARS if y in yearly_data])

    grid_before = make_grid(all_lons_b, all_lats_b)
    grid_after  = make_grid(all_lons_a, all_lats_a)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Layer 4: Spatial Deduplication -- Moran's I Analysis\n"
        f"Years {SAMPLE_YEARS}, Jul-Aug, {MIN_DIST_KM:.0f} km minimum distance",
        fontsize=12
    )
    extent = [lon_min, lon_max, lat_min, lat_max]
    vmax = np.percentile(grid_before[grid_before > 0], 95) if (grid_before > 0).any() else 1

    ax0 = axes[0, 0]
    im0 = ax0.imshow(grid_before, origin="lower", extent=extent, aspect="auto",
                     cmap="YlOrRd",
                     norm=mcolors.PowerNorm(gamma=0.4, vmin=0, vmax=vmax))
    ax0.set_title(f"Before thinning (N={len(all_lons_b):,})")
    ax0.set_xlabel("Longitude")
    ax0.set_ylabel("Latitude")
    plt.colorbar(im0, ax=ax0, label="Fire px count / 0.5-deg cell")

    ax1 = axes[0, 1]
    im1 = ax1.imshow(grid_after, origin="lower", extent=extent, aspect="auto",
                     cmap="YlOrRd",
                     norm=mcolors.PowerNorm(gamma=0.4, vmin=0, vmax=vmax))
    ax1.set_title(f"After {MIN_DIST_KM:.0f} km thinning (N={len(all_lons_a):,})")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    plt.colorbar(im1, ax=ax1, label="Fire px count / 0.5-deg cell")

    # Per-year Moran's I bar chart
    ax2 = axes[1, 0]
    year_labels = [str(r["year"]) for r in per_year_rows]
    I_befores   = [r["before_I"] for r in per_year_rows]
    I_afters    = [r["after_I"]  for r in per_year_rows]
    x_pos = np.arange(len(per_year_rows))
    width = 0.35
    bars_b = ax2.bar(x_pos - width / 2, I_befores, width,
                     color="#c00000", label="Before")
    bars_a = ax2.bar(x_pos + width / 2, I_afters,  width,
                     color="#5b9bd5", label="After")
    for bar, val in zip(bars_b, I_befores):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars_a, I_afters):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(year_labels)
    ax2.set_ylabel("Moran's I")
    ax2.set_title("Moran's I Before vs. After Deduplication (per year)")
    ax2.legend()
    ax2.set_ylim(0, max(max(I_befores), max(I_afters)) * 1.3)

    # Ecozone removal bar chart (top 15)
    ax3 = axes[1, 1]
    top_eco = eco_df.head(15)
    ax3.barh(top_eco["ecozone"].str[:30], top_eco["removed_fraction"] * 100,
             color="#ed7d31", alpha=0.85)
    ax3.set_xlabel("Pixels removed (%)")
    ax3.set_title("Removal % by Ecozone (top 15)")
    ax3.set_xlim(0, 100)
    ax3.tick_params(axis="y", labelsize=7)

    plt.tight_layout()
    png_path = OUT_DIR / "morans_i_before_after.png"
    fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {png_path}")

    print(f"\n{'='*60}")
    print("Layer 4 Moran's I analysis complete.")
    for r in per_year_rows:
        print(f"  {r['year']}: I_before={r['before_I']:.4f} (p={r['before_p']:.3e})  "
              f"I_after={r['after_I']:.4f} (p={r['after_p']:.3e})  "
              f"removed={r['removed_fraction']*100:.1f}%")


if __name__ == "__main__":
    main()
