"""Phase 2 Plan 08 — Multi-Variant Full-Dataset Sampling (Gap Closure).

Produces 5 pretty-printed JSON sample files (samples_variant_{A-D,F}.json), a shared
fixed_test_set.json, and dataset_statistics.json.

Each sample file has a single ``samples`` list (like ``samples.json`` in
``active_fire_selection/fire_sample_selection.py``, but without separate
``positive_samples`` / ``negative_samples`` keys). Records include
``firms_value``: 4 for positives, 0 for negatives (matches dataload_h5.py min_fire_threshold=2).

Design decisions (2026-04-14):
  - Variant F is run first to determine reference counts.
  - Fixed test set (test years 2023-2025): Variant F negatives, shared across A-D,F.
  - Train/val (2000-2022): each variant draws from its own filtered pool,
    subsampled WITHOUT replacement (seed=42) to match Variant F reference counts.
  - Positive pixels are identical across all variants (firemask==9, L1 filter).
  - Temporal split: train ≤ 2019, val 2020-2022, test 2023-2025.
  - Fire season: May-October (months 5-10).

Variant pipeline:
  A: research-area firemask==5 pixels only (no spatial/temporal buffer)
  B: Layer 1 (30km buffer + ±3-day temporal exclusion)
  C: Layers 1 + 2 (ecozone × LC × month stratified quotas)
  D: Layers 1 + 2 + 3 (50% FWI hard negatives + 50% random)
  F: Layers 1 + 2 + 3 + 4 (+ 10km spatial deduplication)  # formerly named E

Usage:
    python generate_samples.py
    python generate_samples.py --dry-run          # 2018-2020 only, smoke test
    python generate_samples.py --years 2000 2010  # restrict year range

Output:
    /mnt/raid/.../sampling/fixed_test_set.json
    /mnt/raid/.../sampling/samples_variant_{A-D,F}.json
    /mnt/raid/.../sampling/dataset_statistics.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import multiprocessing as mp
from scipy.spatial import KDTree

# ---------------------------------------------------------------------------
# Path setup — import sibling modules
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from layer1_filter import (
    Layer1Filter,
    RESEARCH_AREA_SHP,
    _read_firemask,
    _surrounding_dates,
    _pixel_centroids,
    FIREMASK_HIGH_CONF,
    FIREMASK_NONFIRE_LAND,
    FIREMASK_NOM_CONF,
    FIREMASK_LOW_CONF,
    EPSG_WGS84,
    EPSG_CANADA,
)
from layer2_stratification import (
    Layer2Stratification,
    lookup_lc,
    sample_negatives_stratified,
)
from layer3_hard_negative_scorer import Layer3HardNegativeScorer
from layer4_spatial_dedup import Layer4SpatialDedup
from split_constructor import SplitConstructor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(
    "//s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/sampling"
)
TEMP_DIR = OUTPUT_DIR / "_tmp_year_pools"
# Final JSON outputs go alongside existing A-D,F variants
FINAL_OUTPUT_DIR = Path(
    "/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/sampling/entire_can_samples"
)

FIRE_SEASON_MONTHS: frozenset[int] = frozenset(range(5, 11))  # May–Oct
FULL_START_YEAR = 2000
FULL_END_YEAR   = 2025

TRAIN_YEARS: frozenset[int] = frozenset(range(2000, 2020))   # ≤ 2019
VAL_YEARS:   frozenset[int] = frozenset({2020, 2021, 2022})
TEST_YEARS:  frozenset[int] = frozenset({2023, 2024, 2025})

MAX_TEST_POS      = 50_000   # cap on test positive count
NEG_POS_RATIO     = 2.0      # target negatives-to-positives ratio
HARD_NEG_FRACTION = 0.5      # fraction of L2 candidates assigned as hard negatives
SEED              = 42

VARIANTS = list("ABCDFZ")  # F replaces legacy E (identical definition)
ACTIVE_VARIANTS: list[str] = list(VARIANTS)  # narrowed in main() via --only-variants
FIREMASK_CLOUD = 4  # MODIS cloud (land or water)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MODIS firemask LRU cache (per-process; avoids re-opening TIFs for +-3-day window)
# ---------------------------------------------------------------------------
_firemask_cache: dict = {}
_FIREMASK_CACHE_SIZE = 20  # covers 7-date sliding window with overlap


def _cached_read_firemask(date_str: str):
    """LRU wrapper around _read_firemask; eliminates repeated TIF opens."""
    if date_str in _firemask_cache:
        return _firemask_cache[date_str]
    result = _read_firemask(date_str)
    if len(_firemask_cache) >= _FIREMASK_CACHE_SIZE:
        _firemask_cache.pop(next(iter(_firemask_cache)))
    _firemask_cache[date_str] = result
    return result


# Cached pyproj Transformer (one instance per process)
_transformer_to_3978 = None


def _get_transformer():
    global _transformer_to_3978
    if _transformer_to_3978 is None:
        from pyproj import Transformer as _Transformer
        _transformer_to_3978 = _Transformer.from_crs("EPSG:4326", EPSG_CANADA, always_xy=True)
    return _transformer_to_3978


# ---------------------------------------------------------------------------
# Module-level layer instances (initialized once in main)
# ---------------------------------------------------------------------------
_l1: Layer1Filter | None = None           # with spatial/temporal buffer (B-D, F)
_l2: Layer2Stratification | None = None
_l3: Layer3HardNegativeScorer | None = None
_l4: Layer4SpatialDedup | None = None
_modis_gt: tuple | None = None            # GeoTransform for pixel ↔ lon/lat
_modis_W: int = 0
_modis_H: int = 0
_research_mask: np.ndarray | None = None  # (H, W) bool — rasterized research area


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _temporal_split(year: int) -> str:
    if year in TEST_YEARS:
        return "test"
    if year in VAL_YEARS:
        return "val"
    return "train"


def _lonlat_to_rowcol(
    lons: np.ndarray, lats: np.ndarray, gt: tuple
) -> tuple[np.ndarray, np.ndarray]:
    """lon/lat → (pixel_row, pixel_col) in MODIS merged raster space."""
    cols = ((lons - gt[0]) / gt[1]).astype(np.int32)
    rows = ((lats - gt[3]) / gt[5]).astype(np.int32)
    return rows, cols


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _empty_pool() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["date", "pixel_row", "pixel_col", "lulc_class"]
    )


# ---------------------------------------------------------------------------
# Research area rasterization (fast mask, computed once)
# ---------------------------------------------------------------------------

def _build_research_mask(gt: tuple, W: int, H: int, shp_path: str) -> np.ndarray:
    """Rasterize research_area_shpfile_merge.shp at MODIS raster resolution.

    Returns a (H, W) boolean array: True inside the research area.
    Uses GDAL RasterizeLayer — avoids per-pixel shapely within() checks.
    """
    import os
    # Fix PROJ database location so gdal CRS operations work
    os.environ.setdefault(
        "PROJ_LIB",
        "/home/user/.local/lib/python3.10/site-packages/pyproj/proj_dir/share/proj",
    )
    from osgeo import gdal, ogr, osr
    log.info("Rasterizing research area shapefile (once) …")

    # Open shapefile and reproject to MODIS CRS (WGS84)
    shp_ds = ogr.Open(shp_path)
    if shp_ds is None:
        raise FileNotFoundError(f"Cannot open research area: {shp_path}")
    shp_layer = shp_ds.GetLayer()

    # Determine MODIS raster SRS from GeoTransform + a reference MODIS file
    MODIS_DIR = Path(
        "/s3/user/wildfire_dataset/self_built_materials/"
        "bc_raw_data_new_fast/drivers/processed/"
        "spatial_filled_5xdownsampled_merged/"
        "MCD14A1_combined_5xMaximumDown_merged"
    )
    wkt_proj = None
    for fname in ("2018_07_15.tif", "2017_07_15.tif", "2020_07_15.tif"):
        fp = MODIS_DIR / fname
        if fp.exists():
            ref_ds = gdal.Open(str(fp), gdal.GA_ReadOnly)
            if ref_ds is not None:
                wkt_proj = ref_ds.GetProjection()
                ref_ds = None
                break

    # Create in-memory raster
    mem_drv = gdal.GetDriverByName("MEM")
    out_ds = mem_drv.Create("", W, H, 1, gdal.GDT_Byte)
    out_ds.SetGeoTransform(gt)
    if wkt_proj:
        out_ds.SetProjection(wkt_proj)
    band = out_ds.GetRasterBand(1)
    band.Fill(0)
    band.SetNoDataValue(0)

    gdal.RasterizeLayer(out_ds, [1], shp_layer, burn_values=[1])
    out_ds.FlushCache()

    raw = band.ReadRaster(0, 0, W, H, buf_type=gdal.GDT_Byte)
    mask = np.frombuffer(raw, dtype=np.uint8).reshape(H, W).astype(bool)

    shp_ds = None
    out_ds = None
    log.info(
        "Research mask ready: %d / %d pixels inside (%.1f%%)",
        mask.sum(), W * H, 100.0 * mask.sum() / (W * H),
    )
    return mask


def _apply_research_mask(
    lons: np.ndarray, lats: np.ndarray
) -> np.ndarray:
    """Return boolean mask: True for (lon, lat) points inside the research area.

    Uses the pre-rasterized _research_mask array — O(n) array lookup,
    replacing the slow shapely within() check.
    """
    cols = ((lons - _modis_gt[0]) / _modis_gt[1]).astype(np.int32)
    rows = ((lats - _modis_gt[3]) / _modis_gt[5]).astype(np.int32)
    in_bounds = (rows >= 0) & (rows < _modis_H) & (cols >= 0) & (cols < _modis_W)
    result = np.zeros(len(lons), dtype=bool)
    ib = in_bounds
    result[ib] = _research_mask[rows[ib], cols[ib]]
    return result


# ---------------------------------------------------------------------------
# Fast Layer 1 extraction (bypasses shapely within() bottleneck)
# ---------------------------------------------------------------------------

def _fast_get_fire_pixels(date_str: str) -> gpd.GeoDataFrame:
    """Like Layer1Filter.get_fire_pixels but uses raster research mask."""
    result = _cached_read_firemask(date_str)
    if result is None:
        return _empty_gdf_l1(date_str)
    arr, gt = result
    fire_mask = arr == FIREMASK_HIGH_CONF
    if not fire_mask.any():
        return _empty_gdf_l1(date_str)
    lons, lats = _pixel_centroids(arr, gt, fire_mask)
    keep = _apply_research_mask(lons, lats)
    lons, lats = lons[keep], lats[keep]
    if len(lons) == 0:
        return _empty_gdf_l1(date_str)
    n = len(lons)
    return gpd.GeoDataFrame(
        {"date": [date_str] * n,
         "firemask": np.full(n, FIREMASK_HIGH_CONF, dtype=np.uint8)},
        geometry=gpd.points_from_xy(lons, lats),
        crs=EPSG_WGS84,
    )


def _fast_get_neg_candidates(
    date_str: str,
    with_buffer: bool = True,
) -> gpd.GeoDataFrame:
    """Like Layer1Filter.get_negative_candidates but uses raster research mask.

    with_buffer=False → Variant A (no spatial/temporal exclusion).
    with_buffer=True  → Variants B-D, F (30km buffer, ±3-day temporal).
    """
    result = _cached_read_firemask(date_str)
    if result is None:
        return _empty_gdf_l1(date_str)
    arr, gt = result
    cand_mask = arr == FIREMASK_NONFIRE_LAND
    if not cand_mask.any():
        return _empty_gdf_l1(date_str)

    lons_c, lats_c = _pixel_centroids(arr, gt, cand_mask)
    # Fast research area check
    keep_ra = _apply_research_mask(lons_c, lats_c)
    lons_c, lats_c = lons_c[keep_ra], lats_c[keep_ra]
    if len(lons_c) == 0:
        return _empty_gdf_l1(date_str)

    if not with_buffer:
        n = len(lons_c)
        return gpd.GeoDataFrame(
            {"date": [date_str] * n,
             "firemask": np.full(n, FIREMASK_NONFIRE_LAND, dtype=np.uint8)},
            geometry=gpd.points_from_xy(lons_c, lats_c),
            crs=EPSG_WGS84,
        )

    # Build spatial exclusion zone from ±3-day context
    context_dates = _surrounding_dates(date_str, 3)
    fire_lons_all, fire_lats_all = [], []
    for d in context_dates:
        d_result = _cached_read_firemask(d)
        if d_result is None:
            continue
        d_arr, d_gt = d_result
        excl = (d_arr == FIREMASK_HIGH_CONF) | (d_arr == FIREMASK_NOM_CONF) | (d_arr == FIREMASK_LOW_CONF)
        if not excl.any():
            continue
        fl, flt = _pixel_centroids(d_arr, d_gt, excl)
        fire_lons_all.append(fl)
        fire_lats_all.append(flt)

    if not fire_lons_all:
        # No fire context — all candidates are valid
        n = len(lons_c)
        return gpd.GeoDataFrame(
            {"date": [date_str] * n,
             "firemask": np.full(n, FIREMASK_NONFIRE_LAND, dtype=np.uint8)},
            geometry=gpd.points_from_xy(lons_c, lats_c),
            crs=EPSG_WGS84,
        )

    fire_lons_cat = np.concatenate(fire_lons_all)
    fire_lats_cat = np.concatenate(fire_lats_all)

    to_3978 = _get_transformer()
    fx, fy = to_3978.transform(fire_lons_cat, fire_lats_cat)
    cx, cy = to_3978.transform(lons_c, lats_c)

    # KDTree nearest-fire-pixel distance filter
    # Replaces shapely buffer+unary_union+within (was O(n_fire^2 + n_cand*complexity))
    # Now O((n_fire + n_cand) * log n_fire) with scipy KDTree
    fire_tree = KDTree(np.column_stack([fx, fy]))
    min_dists, _ = fire_tree.query(np.column_stack([cx, cy]), k=1, workers=1)
    not_excl = min_dists > 30_000.0

    lons_ok = lons_c[not_excl]
    lats_ok = lats_c[not_excl]
    n = len(lons_ok)
    if n == 0:
        return _empty_gdf_l1(date_str)
    return gpd.GeoDataFrame(
        {"date": [date_str] * n,
         "firemask": np.full(n, FIREMASK_NONFIRE_LAND, dtype=np.uint8)},
        geometry=gpd.points_from_xy(lons_ok, lats_ok),
        crs=EPSG_WGS84,
    )


def _empty_gdf_l1(date_str: str) -> gpd.GeoDataFrame:
    import pandas as _pd
    return gpd.GeoDataFrame(
        {"date": _pd.Series([], dtype=str),
         "firemask": _pd.Series([], dtype=np.uint8)},
        geometry=gpd.GeoSeries([], crs=EPSG_WGS84),
        crs=EPSG_WGS84,
    )



def _fast_get_fire_pixels_z(date_str: str) -> gpd.GeoDataFrame:
    """Variant Z positives: firemask 8 (nominal) + 9 (high confidence)."""
    result = _cached_read_firemask(date_str)
    if result is None:
        return _empty_gdf_l1(date_str)
    arr, gt = result
    fire_mask = (arr == FIREMASK_HIGH_CONF) | (arr == FIREMASK_NOM_CONF)
    if not fire_mask.any():
        return _empty_gdf_l1(date_str)
    lons, lats = _pixel_centroids(arr, gt, fire_mask)
    keep = _apply_research_mask(lons, lats)
    lons, lats = lons[keep], lats[keep]
    if len(lons) == 0:
        return _empty_gdf_l1(date_str)
    n = len(lons)
    return gpd.GeoDataFrame(
        {"date": [date_str] * n,
         "firemask": np.full(n, FIREMASK_HIGH_CONF, dtype=np.uint8)},
        geometry=gpd.points_from_xy(lons, lats),
        crs=EPSG_WGS84,
    )


def _fast_get_neg_candidates_z(date_str: str) -> gpd.GeoDataFrame:
    """Variant Z negatives: cloud (4) + non-fire land (5), no spatial/temporal buffer."""
    result = _cached_read_firemask(date_str)
    if result is None:
        return _empty_gdf_l1(date_str)
    arr, gt = result
    cand_mask = (arr == FIREMASK_NONFIRE_LAND) | (arr == FIREMASK_CLOUD)
    if not cand_mask.any():
        return _empty_gdf_l1(date_str)
    lons_c, lats_c = _pixel_centroids(arr, gt, cand_mask)
    keep_ra = _apply_research_mask(lons_c, lats_c)
    lons_c, lats_c = lons_c[keep_ra], lats_c[keep_ra]
    if len(lons_c) == 0:
        return _empty_gdf_l1(date_str)
    n = len(lons_c)
    return gpd.GeoDataFrame(
        {"date": [date_str] * n,
         "firemask": np.full(n, FIREMASK_NONFIRE_LAND, dtype=np.uint8)},
        geometry=gpd.points_from_xy(lons_c, lats_c),
        crs=EPSG_WGS84,
    )


# ---------------------------------------------------------------------------
# Layer initialisation
# ---------------------------------------------------------------------------

def _init_layers() -> None:
    global _l1, _l2, _l3, _l4, _modis_gt, _modis_W, _modis_H, _research_mask
    log.info("Initialising layer modules …")

    # L1 still used for iter_dates() only; pixel extraction done by _fast_* funcs
    _l1 = Layer1Filter(apply_research_mask=False)
    _l2 = Layer2Stratification()
    _l3 = Layer3HardNegativeScorer()
    _l4 = Layer4SpatialDedup(min_dist_km=10.0, seed=SEED)

    # Load MODIS GeoTransform + dimensions
    from osgeo import gdal
    MODIS_DIR = Path(
        "/s3/user/wildfire_dataset/self_built_materials/"
        "bc_raw_data_new_fast/drivers/processed/"
        "spatial_filled_5xdownsampled_merged/"
        "MCD14A1_combined_5xMaximumDown_merged"
    )
    for fname in ("2018_07_15.tif", "2017_07_15.tif", "2020_07_15.tif"):
        fp = MODIS_DIR / fname
        if fp.exists():
            ds = gdal.Open(str(fp), gdal.GA_ReadOnly)
            if ds is not None:
                _modis_gt = ds.GetGeoTransform()
                _modis_W  = ds.RasterXSize
                _modis_H  = ds.RasterYSize
                ds = None
                break
    if _modis_gt is None:
        raise RuntimeError(f"Cannot load MODIS GeoTransform from {MODIS_DIR}")

    # Build rasterized research area mask
    _research_mask = _build_research_mask(
        _modis_gt, _modis_W, _modis_H, RESEARCH_AREA_SHP
    )

    # Pre-build ecozone + LC rasters at MODIS resolution (fast O(n) lookups)
    from layer2_stratification import init_fast_rasters as _l2_init_fast
    _l2_init_fast(_modis_gt, _modis_W, _modis_H)

    log.info("All layer modules ready.")


# ---------------------------------------------------------------------------
# Per-date pipeline
# ---------------------------------------------------------------------------

def _gdf_to_pool(
    gdf: gpd.GeoDataFrame,
    date_str: str,
    year: int,
    lc_from_col: str | None = None,
) -> pd.DataFrame:
    """Convert a GeoDataFrame of pixels to a pool DataFrame.

    If lc_from_col is set, read lulc_class from that column instead of
    calling lookup_lc again (avoids redundant raster reads).
    """
    if len(gdf) == 0:
        return _empty_pool()

    lons = gdf.geometry.x.values
    lats = gdf.geometry.y.values
    rows, cols = _lonlat_to_rowcol(lons, lats, _modis_gt)

    if lc_from_col and lc_from_col in gdf.columns:
        lc = gdf[lc_from_col].values.astype(np.uint8)
    else:
        lc = lookup_lc(lons, lats, year)

    # Keep only fire-relevant LC (IGBP 1-11)
    keep = (lc >= 1) & (lc <= 11)
    return pd.DataFrame(
        {
            "date":      date_str,
            "pixel_row": rows[keep],
            "pixel_col": cols[keep],
            "lulc_class": lc[keep].astype(int),
        }
    )


def _process_date(date_str: str) -> dict[str, pd.DataFrame]:
    """Run pipeline for one date; return {variant_or_pos → pool DataFrame}.

    Keys: 'pos', 'pos_Z', 'A'-'D', 'F', 'Z'.
    Each DataFrame has columns: date, pixel_row, pixel_col, lulc_class.
    """
    year  = int(date_str[:4])
    quota_df = _l2.get_quota_table()

    result: dict[str, pd.DataFrame] = {k: _empty_pool() for k in ["pos", "pos_Z"] + VARIANTS}

    # ---- Positive pixels (same for A-D, F) ----
    fire_gdf = _fast_get_fire_pixels(date_str)
    result["pos"] = _gdf_to_pool(fire_gdf, date_str, year)

    # ---- Positive pixels for Variant Z (firemask 8+9) ----
    if "Z" in ACTIVE_VARIANTS:
        fire_z_gdf = _fast_get_fire_pixels_z(date_str)
        result["pos_Z"] = _gdf_to_pool(fire_z_gdf, date_str, year)

    # ---- Variant A: research-area firemask==5, no spatial/temporal buffer ----
    neg_a_gdf = _fast_get_neg_candidates(date_str, with_buffer=False)
    result["A"] = _gdf_to_pool(neg_a_gdf, date_str, year)

    # ---- Variants B–D, F share the L1-buffered negative candidates ----
    neg_b_gdf = _fast_get_neg_candidates(date_str, with_buffer=True)
    if len(neg_b_gdf) == 0:
        return result

    # Add pixel_row/pixel_col to GDF so Layer 2 can propagate them
    lons_b = neg_b_gdf.geometry.x.values
    lats_b = neg_b_gdf.geometry.y.values
    rows_b, cols_b = _lonlat_to_rowcol(lons_b, lats_b, _modis_gt)
    neg_b_gdf = neg_b_gdf.copy()
    neg_b_gdf["pixel_row"] = rows_b
    neg_b_gdf["pixel_col"] = cols_b

    # Variant B: Layer 1 candidates (LC filter applied inside _gdf_to_pool)
    result["B"] = _gdf_to_pool(neg_b_gdf, date_str, year)

    # ---- Layer 2: ecozone×LC×month stratified quota sampling ----
    neg_l2_df = sample_negatives_stratified(
        neg_b_gdf, quota_df, total_negatives=None
    )
    if len(neg_l2_df) == 0:
        return result

    # Variant C: after Layer 2 (lc_class column from L2 output)
    result["C"] = pd.DataFrame(
        {
            "date":      date_str,
            "pixel_row": neg_l2_df["pixel_row"].values,
            "pixel_col": neg_l2_df["pixel_col"].values,
            "lulc_class": neg_l2_df["lc_class"].values.astype(int),
        }
    )

    # ---- Layer 3: FWI hard negative scoring ----
    lons_l2 = neg_l2_df["lon"].values
    lats_l2 = neg_l2_df["lat"].values
    scores   = _l3.score_candidates(date_str, lons_l2, lats_l2)
    n_valid  = int((~np.isnan(scores)).sum())

    if n_valid == 0:
        # No FWI data for this date — fall back to C pool for D and F
        log.debug("No FWI tile for %s — D/F fall back to C", date_str)
        result["D"] = result["C"].copy()
        # F still applies L4 dedup below
        l3_df = neg_l2_df.copy()
    else:
        hard_mask = _l3.select_hard_negatives(scores, fraction=HARD_NEG_FRACTION)
        rand_mask = ~hard_mask & ~np.isnan(scores)
        hard_idx  = np.where(hard_mask)[0]
        rand_idx  = np.where(rand_mask)[0]

        # 50% hard + up-to-50% random (matched count)
        rng = np.random.default_rng(SEED)
        n_rand_keep = min(len(hard_idx), len(rand_idx))
        if n_rand_keep > 0:
            rand_sel = rng.choice(rand_idx, size=n_rand_keep, replace=False)
        else:
            rand_sel = rand_idx

        l3_idx = np.sort(np.concatenate([hard_idx, rand_sel]))
        l3_df  = neg_l2_df.iloc[l3_idx].copy().reset_index(drop=True)

        result["D"] = pd.DataFrame(
            {
                "date":      date_str,
                "pixel_row": l3_df["pixel_row"].values,
                "pixel_col": l3_df["pixel_col"].values,
                "lulc_class": l3_df["lc_class"].values.astype(int),
            }
        )

    # ---- Layer 4: spatial deduplication (Variant F only) ----
    if len(l3_df) == 0:
        return result

    l3_gdf = gpd.GeoDataFrame(
        l3_df,
        geometry=gpd.points_from_xy(l3_df["lon"], l3_df["lat"]),
        crs="EPSG:4326",
    )
    thinned_gdf = _l4.thin(l3_gdf)

    result["F"] = pd.DataFrame(
        {
            "date":      date_str,
            "pixel_row": thinned_gdf["pixel_row"].values,
            "pixel_col": thinned_gdf["pixel_col"].values,
            "lulc_class": thinned_gdf["lc_class"].values.astype(int),
        }
    )

    # ---- Variant Z: L0 baseline (firemask 8+9 pos, cloud+nonfire neg) ----
    if "Z" in ACTIVE_VARIANTS:
        neg_z_gdf = _fast_get_neg_candidates_z(date_str)
        result["Z"] = _gdf_to_pool(neg_z_gdf, date_str, year)

    return result


# ---------------------------------------------------------------------------
# Per-year processing with checkpointing
# ---------------------------------------------------------------------------

def _checkpoint_exists(year: int) -> bool:
    """True if all pool parquets for this year are already saved."""
    files = [TEMP_DIR / f"year_{year}_pos.parquet"] + [
        TEMP_DIR / f"year_{year}_neg_{v}.parquet" for v in ACTIVE_VARIANTS
    ]
    if "Z" in ACTIVE_VARIANTS:
        files.append(TEMP_DIR / f"year_{year}_pos_Z.parquet")
    return all(f.exists() for f in files)


def _save_empty_parquet(path: Path) -> None:
    _empty_pool().to_parquet(path, index=False)


def process_year(year: int) -> None:
    """Process all fire-season dates in year; save parquet checkpoints."""
    if _checkpoint_exists(year):
        log.info("Year %d already checkpointed — skipping.", year)
        return

    log.info("Processing year %d …", year)
    dates = [
        d for d in _l1.iter_dates(year, year)
        if int(d[5:7]) in FIRE_SEASON_MONTHS
    ]
    if not dates:
        log.warning("No MODIS tiles found for year %d", year)
        _save_empty_parquet(TEMP_DIR / f"year_{year}_pos.parquet")
        if "Z" in ACTIVE_VARIANTS:
            _save_empty_parquet(TEMP_DIR / f"year_{year}_pos_Z.parquet")
        for v in ACTIVE_VARIANTS:
            _save_empty_parquet(TEMP_DIR / f"year_{year}_neg_{v}.parquet")
        return

    pos_parts: list[pd.DataFrame] = []
    pos_z_parts: list[pd.DataFrame] = []
    neg_parts: dict[str, list[pd.DataFrame]] = {v: [] for v in ACTIVE_VARIANTS}

    for i, date_str in enumerate(dates):
        if i % 20 == 0:
            log.info("  %d / %d  (%s)", i, len(dates), date_str)
        try:
            day_result = _process_date(date_str)
        except Exception as exc:
            log.warning("  Error on %s: %s — skipping date", date_str, exc)
            continue

        if len(day_result["pos"]) > 0:
            pos_parts.append(day_result["pos"])
        if "Z" in ACTIVE_VARIANTS and len(day_result.get("pos_Z", [])) > 0:
            pos_z_parts.append(day_result["pos_Z"])
        for v in ACTIVE_VARIANTS:
            if len(day_result.get(v, [])) > 0:
                neg_parts[v].append(day_result[v])

    # Save positives
    pos_out = TEMP_DIR / f"year_{year}_pos.parquet"
    if pos_parts:
        pd.concat(pos_parts, ignore_index=True).to_parquet(pos_out, index=False)
    else:
        _save_empty_parquet(pos_out)

    if "Z" in ACTIVE_VARIANTS:
        pos_z_out = TEMP_DIR / f"year_{year}_pos_Z.parquet"
        if pos_z_parts:
            pd.concat(pos_z_parts, ignore_index=True).to_parquet(pos_z_out, index=False)
        else:
            _save_empty_parquet(pos_z_out)

    # Save per-variant negatives
    for v in ACTIVE_VARIANTS:
        neg_out = TEMP_DIR / f"year_{year}_neg_{v}.parquet"
        if neg_parts[v]:
            pd.concat(neg_parts[v], ignore_index=True).to_parquet(
                neg_out, index=False
            )
        else:
            _save_empty_parquet(neg_out)

    n_pos = sum(len(p) for p in pos_parts)
    n_z   = sum(len(p) for p in neg_parts.get("Z", []))
    log.info(
        "  Year %d done — %d pos, %d Z-neg across %d dates.",
        year, n_pos, n_z, len(dates),
    )


# ---------------------------------------------------------------------------
# Load all-year pools for a given variant / positives
# ---------------------------------------------------------------------------

def _load_years(years: list[int], key: str) -> pd.DataFrame:
    """Load and concatenate parquet pools for a list of years.

    key: 'pos', 'pos_Z', or variant letter 'A'-'Z'
    """
    parts = []
    for yr in years:
        if key == "pos":
            fname = TEMP_DIR / f"year_{yr}_pos.parquet"
        elif key == "pos_Z":
            fname = TEMP_DIR / f"year_{yr}_pos_Z.parquet"
        else:
            fname = TEMP_DIR / f"year_{yr}_neg_{key}.parquet"
        if fname.exists():
            df = pd.read_parquet(fname)
            if len(df) > 0:
                parts.append(df)
    if not parts:
        return _empty_pool()
    return pd.concat(parts, ignore_index=True)



# Path to existing Variant F JSONL — used as fallback when F parquets are absent
_F_JSONL_FALLBACK = Path(
    "/mnt/raid/user/hackathons/dataset_building_entire_canada/sampling/samples_variant_F.jsonl"
)


def _load_ref_and_test_from_jsonl(jsonl_path: Path) -> tuple[dict, pd.DataFrame]:
    """Read ref_counts and fixed_test from an existing variant JSONL.

    Fallback used when F parquets are not available in TEMP_DIR (e.g. --only-variants Z).
    """
    log.info("Loading ref_counts fallback from %s ...", jsonl_path.name)
    records = []
    with open(jsonl_path) as fh:
        for line in fh:
            records.append(json.loads(line.strip()))
    df = pd.DataFrame(records)

    def _n(split, stype):
        return int(((df["split"] == split) & (df["sample_type"] == stype)).sum())

    ref_counts = dict(
        n_train_pos=_n("train", "positive"),
        n_train_neg=_n("train", "negative"),
        n_val_pos=_n("val",   "positive"),
        n_val_neg=_n("val",   "negative"),
        n_test_pos=_n("test", "positive"),
        n_test_neg=_n("test", "negative"),
    )
    fixed_test = df[df["split"] == "test"].copy().reset_index(drop=True)
    log.info(
        "Fallback ref_counts — train: %d pos / %d neg  |  val: %d pos / %d neg  "
        "|  test: %d pos / %d neg",
        ref_counts["n_train_pos"], ref_counts["n_train_neg"],
        ref_counts["n_val_pos"],   ref_counts["n_val_neg"],
        ref_counts["n_test_pos"],  ref_counts["n_test_neg"],
    )
    return ref_counts, fixed_test


# ---------------------------------------------------------------------------
# Reference count determination and fixed test set
# ---------------------------------------------------------------------------

def build_reference_and_test_set(
    rng: np.random.Generator,
) -> tuple[dict, pd.DataFrame]:
    """Run Step 1 of the plan: compute reference counts from Variant F.

    Returns
    -------
    ref_counts : dict with keys n_train_pos/neg, n_val_pos/neg
    fixed_test : DataFrame (all columns including sample_type, split)
    """
    log.info("Building reference counts from Variant F …")

    # ---- Test split (2023-2025): positives + F negatives ----
    all_test_years = sorted(TEST_YEARS)
    pos_test  = _load_years(all_test_years, "pos")
    neg_e_test = _load_years(all_test_years, "F")

    # Fallback: if F parquets missing for any split, read from existing JSONL
    if len(neg_e_test) == 0 and _F_JSONL_FALLBACK.exists():
        log.warning("F neg parquets not in TEMP_DIR — using F JSONL fallback for ref_counts.")
        return _load_ref_and_test_from_jsonl(_F_JSONL_FALLBACK)

    # Subsample positives to MAX_TEST_POS
    n_pos_test = min(len(pos_test), MAX_TEST_POS)
    if len(pos_test) > n_pos_test:
        pos_test = pos_test.sample(n=n_pos_test, replace=False, random_state=SEED)
    n_neg_test = int(round(n_pos_test * NEG_POS_RATIO))
    if len(neg_e_test) > n_neg_test:
        neg_e_test = neg_e_test.sample(n=n_neg_test, replace=False, random_state=SEED)
    elif len(neg_e_test) < n_neg_test:
        log.warning(
            "Test neg pool (%d) < target (%d) — sampling with replacement",
            len(neg_e_test), n_neg_test,
        )
        neg_e_test = neg_e_test.sample(n=n_neg_test, replace=True, random_state=SEED)

    pos_test  = pos_test.copy();  pos_test["sample_type"]  = "positive"
    neg_e_test = neg_e_test.copy(); neg_e_test["sample_type"] = "negative"
    fixed_test = pd.concat([pos_test, neg_e_test], ignore_index=True)
    fixed_test["split"] = "test"

    # ---- Train split (2000-2019): Variant F ----
    all_train_years = sorted(TRAIN_YEARS)
    pos_train  = _load_years(all_train_years, "pos")
    neg_e_train = _load_years(all_train_years, "F")

    # ---- Val split (2020-2022): Variant F ----
    all_val_years = sorted(VAL_YEARS)
    pos_val  = _load_years(all_val_years, "pos")
    neg_e_val = _load_years(all_val_years, "F")

    # Reference: 2:1 neg/pos ratio, positives are all available F-train/val pixels
    n_train_pos = len(pos_train)
    n_train_neg = min(len(neg_e_train), int(round(n_train_pos * NEG_POS_RATIO)))
    n_val_pos   = len(pos_val)
    n_val_neg   = min(len(neg_e_val),   int(round(n_val_pos   * NEG_POS_RATIO)))

    ref_counts = dict(
        n_train_pos=n_train_pos,
        n_train_neg=n_train_neg,
        n_val_pos=n_val_pos,
        n_val_neg=n_val_neg,
        n_test_pos=int(pos_test["sample_type"].eq("positive").sum()),
        n_test_neg=int(neg_e_test["sample_type"].eq("negative").sum()),
    )

    log.info(
        "Reference counts — train: %d pos / %d neg  |  val: %d pos / %d neg  "
        "|  test: %d pos / %d neg",
        n_train_pos, n_train_neg,
        n_val_pos,   n_val_neg,
        ref_counts["n_test_pos"], ref_counts["n_test_neg"],
    )
    return ref_counts, fixed_test


# ---------------------------------------------------------------------------
# Per-variant train/val assembly
# ---------------------------------------------------------------------------

def build_variant_trainval(
    variant: str,
    ref_counts: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Return train+val DataFrame for one variant, subsampled to ref_counts."""

    log.info("Building train/val pool for Variant %s …", variant)

    def _subsample(pool: pd.DataFrame, n: int, split_name: str) -> pd.DataFrame:
        if len(pool) == 0:
            log.warning("  Variant %s %s pool is EMPTY", variant, split_name)
            return pool
        if len(pool) == n:
            return pool  # exact match — no sampling needed
        if len(pool) < n:
            log.warning(
                "  Variant %s %s pool (%d) < target (%d) — sampling with replacement",
                variant, split_name, len(pool), n,
            )
            return pool.sample(n=n, replace=True, random_state=SEED)
        return pool.sample(n=n, replace=False, random_state=SEED)

    # Positives: Z uses firemask 8+9 pool; A-D, F use firemask 9 (L1)
    all_train_years = sorted(TRAIN_YEARS)
    all_val_years   = sorted(VAL_YEARS)
    pos_key = "pos_Z" if variant == "Z" else "pos"

    pos_train = _load_years(all_train_years, pos_key)
    pos_val   = _load_years(all_val_years,   pos_key)
    neg_train = _load_years(all_train_years, variant)
    neg_val   = _load_years(all_val_years,   variant)

    pos_train_s = _subsample(pos_train, ref_counts["n_train_pos"], "train_pos")
    pos_val_s   = _subsample(pos_val,   ref_counts["n_val_pos"],   "val_pos")
    neg_train_s = _subsample(neg_train, ref_counts["n_train_neg"], "train_neg")
    neg_val_s   = _subsample(neg_val,   ref_counts["n_val_neg"],   "val_neg")

    pos_train_s = pos_train_s.copy(); pos_train_s["sample_type"] = "positive"
    pos_val_s   = pos_val_s.copy();   pos_val_s["sample_type"]   = "positive"
    neg_train_s = neg_train_s.copy(); neg_train_s["sample_type"] = "negative"
    neg_val_s   = neg_val_s.copy();   neg_val_s["sample_type"]   = "negative"

    pos_train_s["split"] = "train"
    pos_val_s["split"]   = "val"
    neg_train_s["split"] = "train"
    neg_val_s["split"]   = "val"

    combined = pd.concat(
        [pos_train_s, neg_train_s, pos_val_s, neg_val_s], ignore_index=True
    )
    log.info(
        "  Variant %s: %d train-pos, %d train-neg, %d val-pos, %d val-neg",
        variant,
        len(pos_train_s), len(neg_train_s),
        len(pos_val_s),   len(neg_val_s),
    )
    return combined


# ---------------------------------------------------------------------------
# JSON writing (single ``samples`` array; same field style as fire_sample_selection)
# ---------------------------------------------------------------------------

def _normalize_date_str(raw: str) -> str:
    """Pool dates use ``YYYY_MM_DD`` from MODIS names; JSON uses ``YYYY-MM-DD``."""
    s = str(raw).strip()
    return s.replace("_", "-")


def _df_rows_to_sample_dicts(df: pd.DataFrame) -> list[dict]:
    """Build records (vectorised - replaces itertuples loop)."""
    cols = ["date", "pixel_row", "pixel_col", "sample_type", "lulc_class", "split"]
    df_out = df[cols].copy()
    df_out["pixel_row"]  = df_out["pixel_row"].astype(int)
    df_out["pixel_col"]  = df_out["pixel_col"].astype(int)
    df_out["lulc_class"] = df_out["lulc_class"].astype(int)
    # Vectorised date parsing (replaces per-row datetime.strptime)
    date_clean           = df_out["date"].str.replace("_", "-", regex=False)
    dt                   = pd.to_datetime(date_clean, format="%Y-%m-%d")
    df_out["year"]       = dt.dt.year.astype(int)
    df_out["day_of_year"] = dt.dt.day_of_year.astype(int)
    df_out["date"]       = dt.dt.strftime("%Y-%m-%d")
    df_out["firms_value"] = (df_out["sample_type"] == "positive").map({True: 4, False: 0})
    out_cols = [
        "year", "day_of_year", "date", "pixel_row", "pixel_col",
        "firms_value", "sample_type", "lulc_class", "split",
    ]
    return df_out[out_cols].to_dict("records")


def _write_samples_json(df: pd.DataFrame, out_path: Path) -> None:
    """Write ``{ "positive_samples": [...], "negative_samples": [...] }`` (matches dataload_h5.py format)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    samples = _df_rows_to_sample_dicts(df)
    pos = [r for r in samples if r["sample_type"] == "positive"]
    neg = [r for r in samples if r["sample_type"] == "negative"]
    payload: dict = {
        "positive_samples": pos,
        "negative_samples": neg,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f)
    log.info("  Wrote %d records → %s", len(samples), out_path.name)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verify_file(
    path: Path,
    ref_counts: dict,
    fixed_test: pd.DataFrame,
) -> dict:
    """Read a variant JSON file, verify counts and test-set identity."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    df = pd.DataFrame(data["positive_samples"] + data["negative_samples"])

    # Count by split × sample_type
    counts: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        n_pos = int((sub["sample_type"] == "positive").sum())
        n_neg = int((sub["sample_type"] == "negative").sum())
        counts[split] = {"n_pos": n_pos, "n_neg": n_neg, "total": n_pos + n_neg}

    # Check ratio
    ratio_ok = True
    for split in ("train", "val", "test"):
        n_pos = counts[split]["n_pos"]
        n_neg = counts[split]["n_neg"]
        if n_pos > 0:
            ratio = n_neg / n_pos
            if not (1.9 <= ratio <= 2.1):
                log.warning(
                    "  neg/pos ratio %s=%s=%.3f outside [1.9, 2.1]",
                    path.name, split, ratio,
                )
                ratio_ok = False

    # Spot-check test records vs fixed_test (first min(1000, n_test) rows)
    test_rows = df[df["split"] == "test"][
        ["date", "pixel_row", "pixel_col", "sample_type"]
    ].copy()
    test_rows["date"] = test_rows["date"].map(_normalize_date_str)
    test_rows = test_rows.sort_values(
        ["date", "pixel_row", "pixel_col"]
    ).reset_index(drop=True)
    ref_rows = fixed_test[
        ["date", "pixel_row", "pixel_col", "sample_type"]
    ].copy()
    ref_rows["date"] = ref_rows["date"].map(_normalize_date_str)
    ref_rows = ref_rows.sort_values(
        ["date", "pixel_row", "pixel_col"]
    ).reset_index(drop=True)

    n_check = min(1000, len(test_rows), len(ref_rows))
    test_identical = test_rows.iloc[:n_check].equals(ref_rows.iloc[:n_check])

    return {
        "counts": counts,
        "ratio_ok": ratio_ok,
        "test_spot_check_n": n_check,
        "test_identical": test_identical,
    }


# ---------------------------------------------------------------------------
# Statistics file
# ---------------------------------------------------------------------------

def write_statistics(
    ref_counts: dict,
    variant_info: dict,
    fixed_test_path: Path,
    verification: dict,
    start_year: int,
    end_year: int,
) -> Path:
    stats = {
        "generated_at":     datetime.now(timezone.utc).isoformat(),
        "reference_variant": "F",
        "random_seed":      SEED,
        "year_range":       f"{start_year}-{end_year}",
        "split_type":       "temporal",
        "test_years":       sorted(TEST_YEARS),
        "val_years":        sorted(VAL_YEARS),
        "sample_counts": {
            "train": {
                "n_pos":  ref_counts["n_train_pos"],
                "n_neg":  ref_counts["n_train_neg"],
                "total":  ref_counts["n_train_pos"] + ref_counts["n_train_neg"],
            },
            "val": {
                "n_pos":  ref_counts["n_val_pos"],
                "n_neg":  ref_counts["n_val_neg"],
                "total":  ref_counts["n_val_pos"] + ref_counts["n_val_neg"],
            },
            "test": {
                "n_pos":  ref_counts["n_test_pos"],
                "n_neg":  ref_counts["n_test_neg"],
                "total":  ref_counts["n_test_pos"] + ref_counts["n_test_neg"],
            },
        },
        "variants":    variant_info,
        "fixed_test_set": {
            "file":   fixed_test_path.name,
            "sha256": _sha256(fixed_test_path),
            "n_pos":  ref_counts["n_test_pos"],
            "n_neg":  ref_counts["n_test_neg"],
        },
        "composition_checks": {
            "counts_match_reference": all(
                v.get("counts_match") for v in verification.values()
            ),
            "neg_pos_ratio_ok": all(
                v.get("ratio_ok") for v in verification.values()
            ),
            "test_sets_identical": all(
                v.get("test_identical") for v in verification.values()
            ),
        },
    }

    out_path = FINAL_OUTPUT_DIR / "dataset_statistics.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    log.info("Wrote dataset_statistics.json")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multi-variant samples.")
    parser.add_argument(
        "--only-variants",
        nargs="+",
        metavar="V",
        default=None,
        help="Restrict to these variant letters, e.g. --only-variants Z",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only 2018-2020 for a quick smoke test.",
    )
    parser.add_argument(
        "--years",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Restrict to year range (inclusive).",
    )
    args = parser.parse_args()

    global ACTIVE_VARIANTS
    if args.only_variants:
        ACTIVE_VARIANTS = [v.upper() for v in args.only_variants]
        log.info("Restricting to variants: %s", ACTIVE_VARIANTS)

    if args.dry_run:
        start_year, end_year = 2018, 2020
        log.info("DRY RUN mode: years 2018-2020 only.")
    elif args.years:
        start_year, end_year = args.years
    else:
        start_year, end_year = FULL_START_YEAR, FULL_END_YEAR

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _init_layers()

    rng = np.random.default_rng(SEED)

    # ------------------------------------------------------------------
    # Step 1: Process all years → per-year parquet checkpoints
    # ------------------------------------------------------------------
    log.info("=== Step 1: Processing years %d–%d ===", start_year, end_year)
    all_years = list(range(start_year, end_year + 1))
    todo      = [y for y in all_years if not _checkpoint_exists(y)]
    done      = [y for y in all_years if _checkpoint_exists(y)]
    for y in done:
        log.info("Year %d already checkpointed — skipping.", y)
    if todo:
        # _init_layers() was called above; workers inherit all globals via fork
        n_workers = min(mp.cpu_count(), len(todo))
        log.info(
            "Dispatching %d years across %d parallel workers ...",
            len(todo), n_workers,
        )
        with mp.Pool(processes=n_workers) as pool:
            pool.map(process_year, todo)
    else:
        log.info("All years already checkpointed.")

    # ------------------------------------------------------------------
    # Step 2: Determine reference counts + build fixed test set
    # ------------------------------------------------------------------
    log.info("=== Step 2: Building reference counts and fixed test set ===")
    ref_counts, fixed_test = build_reference_and_test_set(rng)

    fixed_test_path = FINAL_OUTPUT_DIR / "fixed_test_set.json"
    _write_samples_json(fixed_test, fixed_test_path)

    # ------------------------------------------------------------------
    # Step 3: For each variant, assemble train/val + append fixed test
    # ------------------------------------------------------------------
    log.info("=== Step 3: Assembling variant files ===")
    variant_info: dict = {}
    verification: dict = {}

    for variant in ACTIVE_VARIANTS:
        trainval_df = build_variant_trainval(variant, ref_counts, rng)
        full_df     = pd.concat([trainval_df, fixed_test], ignore_index=True)

        out_path = FINAL_OUTPUT_DIR / f"samples_variant_{variant}.json"
        _write_samples_json(full_df, out_path)

        sha = _sha256(out_path)
        n_neg = int((full_df["sample_type"] == "negative").sum())
        n_pos = int((full_df["sample_type"] == "positive").sum())
        ratio = n_neg / max(n_pos, 1)

        variant_info[variant] = {
            "file":          out_path.name,
            "sha256":        sha,
            "neg_pos_ratio": round(ratio, 4),
        }

        verif = _verify_file(out_path, ref_counts, fixed_test)
        verif["counts_match"] = (
            abs(verif["counts"]["train"]["n_pos"] - ref_counts["n_train_pos"]) <= 1
            and abs(verif["counts"]["train"]["n_neg"] - ref_counts["n_train_neg"]) <= 1
            and abs(verif["counts"]["val"]["n_pos"]   - ref_counts["n_val_pos"])   <= 1
            and abs(verif["counts"]["val"]["n_neg"]   - ref_counts["n_val_neg"])   <= 1
        )
        verification[variant] = verif

        log.info(
            "  Variant %s  total=%d  neg/pos=%.3f  test_identical=%s  ratio_ok=%s",
            variant, len(full_df), ratio,
            verif["test_identical"], verif["ratio_ok"],
        )

    # ------------------------------------------------------------------
    # Step 4: Print summary table
    # ------------------------------------------------------------------
    log.info("=== Summary ===")
    print()
    print(f"{'Variant':<10} {'Train_pos':>10} {'Train_neg':>10} "
          f"{'Val_pos':>8} {'Val_neg':>8} "
          f"{'Test_pos':>9} {'Test_neg':>9} "
          f"{'Ratio':>7}  TestOK")
    for v in ACTIVE_VARIANTS:
        c = verification[v]["counts"]
        ratio = variant_info[v]["neg_pos_ratio"]
        ok = verification[v]["test_identical"]
        print(
            f"{v:<10} "
            f"{c['train']['n_pos']:>10,} {c['train']['n_neg']:>10,} "
            f"{c['val']['n_pos']:>8,} {c['val']['n_neg']:>8,} "
            f"{c['test']['n_pos']:>9,} {c['test']['n_neg']:>9,} "
            f"{ratio:>7.3f}  {ok}"
        )
    print()

    # ------------------------------------------------------------------
    # Step 5: Write dataset_statistics.json
    # ------------------------------------------------------------------
    log.info("=== Step 5: Writing dataset_statistics.json ===")
    write_statistics(
        ref_counts, variant_info, fixed_test_path,
        verification, start_year, end_year,
    )

    log.info("Done. All outputs in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
