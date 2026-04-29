#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 Plan 05 — Task 1: Fire Event Clustering (per-year spatial DBSCAN).

Assigns each high-confidence MODIS fire detection (firemask==9) to a distinct
fire event using per-year spatial DBSCAN on (lat, lon). Events are the unit of
split assignment so that all pixels from the same fire stay in one split.

Data source: MCD14A1_combined_5xMaximumDown_merged  (pre-merged daily GeoTIFFs)
             One file per day: yyyy_mm_dd.tif  (band 1 = firemask)
Research area: ecoregions/research_area_shpfile_merge.shp

DBSCAN parameters (per year, fire season May-Oct):
    eps          = 0.25 degrees
    min_samples  = 5
    algorithm    = ball_tree

One representative day per month (15th) sampled to keep runtime tractable.
Noise pixels (label == -1) treated as singleton events.

Outputs (all to justification/layer5/):
    fire_event_clusters.parquet   — pixel-level:
                                    date, pixel_row, pixel_col, lat, lon,
                                    year, event_id, cluster_label
    fire_event_cluster_stats.csv  — event-level summary:
                                    event_id, year, n_pixels,
                                    lat_centroid, lon_centroid,
                                    lat_min, lat_max, lon_min, lon_max,
                                    is_singleton
    fire_event_year_stats.csv     — per-year aggregate:
                                    year, n_events, n_fire_pixels,
                                    mean_event_size, max_event_size,
                                    singleton_fraction
    fire_event_cluster_stats.png  — scatter + size histogram for 3 rep. years
"""
from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from osgeo import gdal
from sklearn.cluster import DBSCAN

gdal.UseExceptions()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODIS_DIR = Path(
    "/s3/user/wildfire_dataset/self_built_materials/"
    "bc_raw_data_new_fast/drivers/processed/"
    "spatial_filled_5xdownsampled_merged/MCD14A1_combined_5xMaximumDown_merged"
)
RESEARCH_AREA_SHP = Path(
    "/s3/user/wildfire_dataset/self_built_materials/"
    "bc_raw_data_new_fast/drivers/ecoregions/research_area_shpfile_merge.shp"
)
OUT_DIR = Path(
    "/mnt/raid/user/hackathons/dataset_building_entire_canada/"
    "sampling/justification/layer5"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUT_DIR / "fire_event_clustering.log"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_PATH), mode="w"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
YEARS              = list(range(2000, 2026))
FIRE_MONTHS        = list(range(5, 11))   # May-Oct inclusive
REP_DAY            = 15                   # representative day of month
DBSCAN_EPS         = 0.25                 # degrees
DBSCAN_MIN_SAMPLES = 5
FIREMASK_POSITIVE  = 9

# ---------------------------------------------------------------------------
# Research area cache
# ---------------------------------------------------------------------------
_ra_union_cache = None


def _get_research_union():
    global _ra_union_cache
    if _ra_union_cache is not None:
        return _ra_union_cache
    gdf = gpd.read_file(str(RESEARCH_AREA_SHP))
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4617")
    gdf = gdf.to_crs("EPSG:4326")
    _ra_union_cache = gdf.geometry.unary_union
    log.info(f"Research area loaded: {RESEARCH_AREA_SHP.name}, "
             f"bounds={_ra_union_cache.bounds}")
    return _ra_union_cache


# ---------------------------------------------------------------------------
# GDAL reader — merged single-file per day
# ---------------------------------------------------------------------------

def _read_merged_day(day: date):
    """Read firemask band from merged daily TIF.

    Returns (uint8 array, geotransform) or (None, None) if file absent.
    File name format: yyyy_mm_dd.tif  (band 1 = firemask).
    Uses ReadRaster+frombuffer — NOT ReadAsArray (avoids gdal_array ABI issue).
    """
    fname = f"{day.year:04d}_{day.month:02d}_{day.day:02d}.tif"
    fpath = MODIS_DIR / fname
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


def _pixel_lonlat(gt, row_idx, col_idx):
    """Convert pixel row/col indices to lon/lat float32 arrays using GeoTransform."""
    lons = (gt[0] + (col_idx + 0.5) * gt[1]).astype(np.float32)
    lats = (gt[3] + (row_idx + 0.5) * gt[5]).astype(np.float32)
    return lons, lats


# ---------------------------------------------------------------------------
# Per-year clustering
# ---------------------------------------------------------------------------

def cluster_year(year: int) -> tuple[list[dict], list[dict]]:
    """Load one representative day per fire-season month, run DBSCAN.

    Returns:
        pixel_records — list of dicts (one per pixel) for parquet
        event_records — list of dicts (one per event) for cluster_stats CSV
    """
    ra_union = _get_research_union()

    all_rows  = []
    all_cols  = []
    all_lons  = []
    all_lats  = []
    all_dates = []

    for month in FIRE_MONTHS:
        day = date(year, month, REP_DAY)
        arr, gt = _read_merged_day(day)
        if arr is None:
            log.debug(f"  {day}: merged file not found")
            continue

        row_idx, col_idx = np.where(arr == FIREMASK_POSITIVE)
        if len(row_idx) == 0:
            log.debug(f"  {day}: no fire pixels")
            continue

        lons, lats = _pixel_lonlat(gt, row_idx.astype(np.float64),
                                   col_idx.astype(np.float64))

        # Research area filter (vectorised via contains)
        in_ra = np.array([
            ra_union.contains(Point(float(lo), float(la)))
            for lo, la in zip(lons, lats)
        ], dtype=bool)

        n_in = int(in_ra.sum())
        if n_in == 0:
            continue

        all_rows.extend(row_idx[in_ra].tolist())
        all_cols.extend(col_idx[in_ra].tolist())
        all_lons.extend(lons[in_ra].tolist())
        all_lats.extend(lats[in_ra].tolist())
        all_dates.extend([day.isoformat()] * n_in)
        log.debug(f"  {day}: {len(row_idx)} raw fire px → {n_in} in research area")

    if not all_lons:
        log.info(f"  {year}: no fire pixels in research area during fire season")
        return [], []

    lons_arr  = np.array(all_lons,  dtype=np.float32)
    lats_arr  = np.array(all_lats,  dtype=np.float32)
    rows_arr  = np.array(all_rows,  dtype=np.int32)
    cols_arr  = np.array(all_cols,  dtype=np.int32)
    dates_arr = np.array(all_dates)

    log.info(f"  {year}: {len(lons_arr):,} fire pixels → DBSCAN "
             f"(eps={DBSCAN_EPS}°, min_samples={DBSCAN_MIN_SAMPLES})")

    # DBSCAN in (lat, lon) degree space
    X = np.column_stack([lats_arr, lons_arr])
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES,
                algorithm="ball_tree", n_jobs=1)
    labels = db.fit_predict(X)

    # Build event_id strings
    # Clusters (label >= 0): "{year}_{label:05d}"
    # Noise    (label == -1): "{year}_noise_{counter:06d}" — singleton events
    event_ids_list = []
    noise_ctr = 0
    for lbl in labels:
        if lbl >= 0:
            event_ids_list.append(f"{year}_{lbl:05d}")
        else:
            event_ids_list.append(f"{year}_noise_{noise_ctr:06d}")
            noise_ctr += 1
    event_ids = np.array(event_ids_list)

    n_clusters  = int(labels.max() + 1) if labels.max() >= 0 else 0
    n_noise     = int((labels == -1).sum())
    log.info(f"  {year}: {n_clusters} clusters + {n_noise} singletons = "
             f"{n_clusters + n_noise} total events")

    # Pixel-level records (for parquet)
    pixel_records = []
    for i in range(len(lons_arr)):
        pixel_records.append({
            "date":          dates_arr[i],
            "pixel_row":     int(rows_arr[i]),
            "pixel_col":     int(cols_arr[i]),
            "lat":           float(lats_arr[i]),
            "lon":           float(lons_arr[i]),
            "year":          year,
            "event_id":      event_ids[i],
            "cluster_label": int(labels[i]),
        })

    # Event-level records (for cluster_stats CSV)
    event_records = []
    for eid in np.unique(event_ids):
        mask_e = (event_ids == eid)
        ev_lons = lons_arr[mask_e]
        ev_lats = lats_arr[mask_e]
        event_records.append({
            "event_id":     eid,
            "year":         year,
            "n_pixels":     int(mask_e.sum()),
            "lat_centroid": float(np.mean(ev_lats)),
            "lon_centroid": float(np.mean(ev_lons)),
            "lat_min":      float(ev_lats.min()),
            "lat_max":      float(ev_lats.max()),
            "lon_min":      float(ev_lons.min()),
            "lon_max":      float(ev_lons.max()),
            "is_singleton": bool("_noise_" in eid),
        })

    return pixel_records, event_records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 70)
    log.info("Fire Event Clustering — Phase 2 Plan 05 Task 1")
    log.info(f"MODIS dir  : {MODIS_DIR}")
    log.info(f"Res. area  : {RESEARCH_AREA_SHP}")
    log.info(f"DBSCAN     : eps={DBSCAN_EPS}°  min_samples={DBSCAN_MIN_SAMPLES}")
    log.info(f"Rep. day   : {REP_DAY}th of each fire-season month (May-Oct)")
    log.info(f"Years      : {YEARS[0]}–{YEARS[-1]}")
    log.info("=" * 70)

    all_pixel_records = []
    all_event_records = []
    per_year_stats    = []

    for year in YEARS:
        log.info(f"\nProcessing year {year}:")
        pix_recs, evt_recs = cluster_year(year)

        all_pixel_records.extend(pix_recs)
        all_event_records.extend(evt_recs)

        n_ev  = len(evt_recs)
        n_px  = len(pix_recs)
        n_sin = sum(1 for r in evt_recs if r["is_singleton"])
        per_year_stats.append({
            "year":              year,
            "n_events":          n_ev,
            "n_fire_pixels":     n_px,
            "mean_event_size":   float(np.mean([r["n_pixels"] for r in evt_recs]))
                                 if evt_recs else 0.0,
            "max_event_size":    int(max((r["n_pixels"] for r in evt_recs), default=0)),
            "singleton_fraction": float(n_sin / n_ev) if n_ev > 0 else 0.0,
        })

    log.info(f"\nTotal: {len(all_event_records):,} events, "
             f"{len(all_pixel_records):,} pixels")

    # -----------------------------------------------------------------------
    # Save parquet (pixel-level)
    # -----------------------------------------------------------------------
    pix_df = pd.DataFrame(all_pixel_records)
    parquet_path = OUT_DIR / "fire_event_clusters.parquet"
    pix_df.to_parquet(str(parquet_path), index=False)
    log.info(f"Parquet saved : {parquet_path}  ({len(pix_df):,} rows)")

    # -----------------------------------------------------------------------
    # Save event-level cluster stats CSV
    # -----------------------------------------------------------------------
    evt_df = pd.DataFrame(all_event_records)
    evt_csv = OUT_DIR / "fire_event_cluster_stats.csv"
    evt_df.to_csv(str(evt_csv), index=False)
    log.info(f"Cluster stats : {evt_csv}  ({len(evt_df):,} events)")

    # Per-year summary CSV
    yr_df = pd.DataFrame(per_year_stats)
    yr_csv = OUT_DIR / "fire_event_year_stats.csv"
    yr_df.to_csv(str(yr_csv), index=False)
    log.info(f"Year stats    : {yr_csv}")
    log.info("\nPer-year summary:")
    log.info(yr_df.to_string(index=False))

    # -----------------------------------------------------------------------
    # Figure: 3 representative years — scatter + size histogram
    # -----------------------------------------------------------------------
    rep_years = [2004, 2015, 2023]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"Fire Event Clustering — DBSCAN(eps={DBSCAN_EPS}°, "
        f"min_samples={DBSCAN_MIN_SAMPLES})\n"
        f"{len(evt_df):,} total events across 2000–2025 "
        f"(rep day = {REP_DAY}th, fire season May-Oct)",
        fontsize=12
    )

    for ci, yr in enumerate(rep_years):
        yr_events = evt_df[evt_df["year"] == yr].copy()

        # Top: spatial scatter of event centroids
        ax_top = axes[0, ci]
        clusters_ev  = yr_events[~yr_events["is_singleton"]]
        singleton_ev = yr_events[yr_events["is_singleton"]]
        if len(clusters_ev) > 0:
            ax_top.scatter(clusters_ev["lon_centroid"], clusters_ev["lat_centroid"],
                           c="steelblue", s=20, alpha=0.7, label="cluster events",
                           zorder=3)
        if len(singleton_ev) > 0:
            ax_top.scatter(singleton_ev["lon_centroid"], singleton_ev["lat_centroid"],
                           c="lightgray", s=5, alpha=0.5, label="singletons",
                           zorder=2)
        ax_top.set_xlim(-141, -52)
        ax_top.set_ylim(41, 84)
        ax_top.set_xlabel("Longitude")
        ax_top.set_ylabel("Latitude")
        ax_top.set_title(f"{yr}: {len(yr_events):,} events")
        ax_top.legend(fontsize=7, loc="lower right")

        # Bottom: event size histogram
        ax_bot = axes[1, ci]
        if len(yr_events) > 0:
            sizes = yr_events["n_pixels"].clip(upper=500)
            ax_bot.hist(sizes, bins=30, color="#c00000", alpha=0.8)
            ax_bot.set_xlabel("Event size (pixels, clipped 500)")
            ax_bot.set_ylabel("Count")
            med = yr_events["n_pixels"].median()
            mx  = yr_events["n_pixels"].max()
            ax_bot.set_title(f"{yr}: median={med:.0f}px, max={mx}px")

    plt.tight_layout()
    png_path = OUT_DIR / "fire_event_cluster_stats.png"
    fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Figure saved  : {png_path}")
    log.info("\nTask 1 complete.")


if __name__ == "__main__":
    main()
