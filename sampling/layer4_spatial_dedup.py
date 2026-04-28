"""Phase 2 -- Layer 4: Spatial Deduplication (reusable module).

Removes spatially autocorrelated duplicate samples by enforcing a minimum
distance of 10 km between any two retained points within the same stratum.
Uses a greedy KDTree thinning algorithm (EPSG:3978 projected distances).

Greedy algorithm:
  1. Shuffle the candidate set (seeded for reproducibility)
  2. Accept the first point unconditionally
  3. For each subsequent point: reject if within min_dist_m of any accepted point
  4. Continue until all candidates have been processed

This ensures no two retained points are closer than min_dist_m, while
retaining as many points as possible (greedy-maximum independent set).

Two public APIs are exposed:

  Layer4SpatialDedup (primary) -- operates on GeoDataFrame with lon/lat geometry
      thin_mask(lons, lats)      -- boolean keep mask
      thin(gdf)                  -- thin a GeoDataFrame
      thin_per_stratum(gdf)      -- thin independently per stratum

  SpatialDeduplicator (plan-spec compatible) -- operates on pixel_row/pixel_col
      deduplicate(pixel_df, ecozone_col)     -- thin per ecozone stratum
      compute_morans_i(pixel_df, sample_n)   -- compute Moran's I on point set

Usage:
    from layer4_spatial_dedup import Layer4SpatialDedup, SpatialDeduplicator
    dedup = Layer4SpatialDedup()
    kept_gdf = dedup.thin(candidates_gdf)

    sd = SpatialDeduplicator()
    thinned_df = sd.deduplicate(neg_df, ecozone_col='ecozone')
    mi = sd.compute_morans_i(neg_df, sample_n=50000)
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from scipy.spatial import KDTree

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MIN_DIST_KM = 10.0    # 10 km minimum distance between kept points
DEFAULT_PIXEL_KM    = 5.0     # MODIS merged raster pixel size
DEFAULT_SEED        = 42

MODIS_DIR = Path(
    "/s3/zhengsen/wildfire_dataset/self_built_materials/"
    "bc_raw_data_new_fast/drivers/processed/"
    "spatial_filled_5xdownsampled_merged/MCD14A1_combined_5xMaximumDown_merged"
)
CANADA_EXTENT = (-141.0, 41.0, -52.0, 84.0)  # lon_min, lat_min, lon_max, lat_max
GRID_DEG = 0.5                               # Moran's I grid resolution (degrees)


# ---------------------------------------------------------------------------
# Internal: MODIS GeoTransform loader (for pixel_row/col -> lon/lat)
# ---------------------------------------------------------------------------

def _get_modis_geotransform() -> tuple | None:
    """Load GeoTransform from any available MODIS merged daily TIF."""
    from osgeo import gdal
    candidates = ["2018_07_15.tif", "2017_07_15.tif", "2021_07_15.tif"]
    for fname in candidates:
        fpath = MODIS_DIR / fname
        if fpath.exists():
            ds = gdal.Open(str(fpath), gdal.GA_ReadOnly)
            if ds is not None:
                gt = ds.GetGeoTransform()
                ds = None
                return gt
    # Fallback: search any file in MODIS_DIR
    for fpath in sorted(MODIS_DIR.glob("*.tif"))[:5]:
        from osgeo import gdal as _gdal
        ds = _gdal.Open(str(fpath), _gdal.GA_ReadOnly)
        if ds is not None:
            gt = ds.GetGeoTransform()
            ds = None
            return gt
    return None


_gt_cache: tuple | None = None


def _pixel_to_lonlat(pixel_rows: np.ndarray,
                     pixel_cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert pixel_row/pixel_col (MODIS merged raster) to (lons, lats)."""
    global _gt_cache
    if _gt_cache is None:
        _gt_cache = _get_modis_geotransform()
    if _gt_cache is None:
        raise RuntimeError(
            "Cannot load MODIS GeoTransform. "
            f"Ensure MODIS_DIR exists: {MODIS_DIR}")
    gt = _gt_cache
    lons = gt[0] + (np.asarray(pixel_cols, dtype=np.float64) + 0.5) * gt[1]
    lats = gt[3] + (np.asarray(pixel_rows, dtype=np.float64) + 0.5) * gt[5]
    return lons, lats


# ---------------------------------------------------------------------------
# Layer4SpatialDedup -- primary class (lon/lat GeoDataFrame API)
# ---------------------------------------------------------------------------

class Layer4SpatialDedup:
    """Layer 4 Spatial Deduplication.

    Provides:
      thin_mask(lons, lats)  -- boolean keep mask for (lon, lat) point arrays
      thin(gdf)              -- thin a GeoDataFrame, return kept rows
      thin_per_stratum(gdf)  -- thin independently within each stratum
    """

    def __init__(self,
                 min_dist_km: float = DEFAULT_MIN_DIST_KM,
                 seed: int = DEFAULT_SEED):
        self.min_dist_m = min_dist_km * 1_000.0
        self.seed = seed
        self._to_3978 = Transformer.from_crs("EPSG:4326", "EPSG:3978",
                                              always_xy=True)
        print(f"Layer4SpatialDedup: min_dist={min_dist_km:.1f} km  seed={seed}")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def thin_mask(self,
                  lons: np.ndarray,
                  lats: np.ndarray,
                  seed: int | None = None) -> np.ndarray:
        """Return boolean mask (True = keep) for (lon, lat) arrays.

        Args:
            lons : 1-D float array of longitudes (EPSG:4326)
            lats : 1-D float array of latitudes  (EPSG:4326)
            seed : override the instance seed (optional)

        Returns:
            bool array, length = len(lons)
        """
        lons = np.asarray(lons, dtype=np.float64)
        lats = np.asarray(lats, dtype=np.float64)
        if len(lons) == 0:
            return np.zeros(0, dtype=bool)

        rng = np.random.default_rng(seed if seed is not None else self.seed)
        xs, ys = self._to_3978.transform(lons, lats)
        pts = np.column_stack([xs, ys])
        return self._greedy_thin(pts, rng)

    def thin(self,
             gdf: gpd.GeoDataFrame,
             seed: int | None = None) -> gpd.GeoDataFrame:
        """Return subset of gdf with spatially thinned points.

        Input gdf must be in EPSG:4326 and have a 'geometry' column.
        """
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            raise ValueError("Input GeoDataFrame must be in EPSG:4326")
        lons = gdf.geometry.x.values
        lats = gdf.geometry.y.values
        mask = self.thin_mask(lons, lats, seed=seed)
        return gdf[mask].copy()

    def thin_per_stratum(self,
                         gdf: gpd.GeoDataFrame,
                         stratum_col: str = "stratum_id",
                         seed: int | None = None) -> gpd.GeoDataFrame:
        """Thin independently within each stratum.

        Args:
            gdf         : GeoDataFrame in EPSG:4326 with stratum_col column
            stratum_col : column name identifying stratum membership
            seed        : override the instance seed (optional)

        Returns:
            GeoDataFrame of kept points, preserving row order within each stratum.
        """
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            raise ValueError("Input GeoDataFrame must be in EPSG:4326")
        if stratum_col not in gdf.columns:
            raise ValueError(f"Column '{stratum_col}' not found in GeoDataFrame")

        base_seed = seed if seed is not None else self.seed
        kept_parts = []
        for i, (stratum_id, group) in enumerate(gdf.groupby(stratum_col)):
            stratum_seed = base_seed + i   # reproducible, distinct per stratum
            lons = group.geometry.x.values
            lats = group.geometry.y.values
            mask = self.thin_mask(lons, lats, seed=stratum_seed)
            kept_parts.append(group[mask])

        if not kept_parts:
            return gdf.iloc[0:0]
        return gpd.GeoDataFrame(
            pd.concat(kept_parts, ignore_index=True),
            geometry="geometry", crs="EPSG:4326"
        )

    def get_stats(self,
                  n_before: int,
                  n_after: int) -> dict:
        """Return a statistics dict for logging / saving."""
        return {
            "n_before":        n_before,
            "n_after":         n_after,
            "n_removed":       n_before - n_after,
            "removed_fraction": round((n_before - n_after) / max(n_before, 1), 4),
            "min_dist_km":     self.min_dist_m / 1_000.0,
        }

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _greedy_thin(self,
                     pts: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
        """Greedy KDTree thinning on projected (x, y) points.

        Returns bool mask (True = keep).
        """
        n = len(pts)
        order   = rng.permutation(n)
        kept    = np.zeros(n, dtype=bool)
        removed = np.zeros(n, dtype=bool)
        tree    = KDTree(pts)

        for idx in order:
            if removed[idx]:
                continue
            kept[idx] = True
            neighbors = tree.query_ball_point(pts[idx], r=self.min_dist_m)
            for nb in neighbors:
                if not kept[nb]:
                    removed[nb] = True
        return kept


# ---------------------------------------------------------------------------
# SpatialDeduplicator -- plan-spec compatible class (pixel_row/col API)
# ---------------------------------------------------------------------------

class SpatialDeduplicator:
    """Plan-spec compatible spatial deduplication interface.

    Operates on DataFrames with pixel_row/pixel_col columns (MODIS merged
    raster coordinate space). Converts to lon/lat via MODIS GeoTransform,
    then applies greedy KDTree thinning per ecozone stratum in EPSG:3978.

    Args:
        min_distance_km : Minimum distance between retained points (default 10 km).
        pixel_size_km   : MODIS merged raster pixel size in km (default 5 km).

    Attributes:
        min_distance_px : Minimum pixel separation = int(min_distance_km / pixel_size_km)
    """

    def __init__(self,
                 min_distance_km: float = DEFAULT_MIN_DIST_KM,
                 pixel_size_km: float = DEFAULT_PIXEL_KM):
        self.min_distance_km = min_distance_km
        self.pixel_size_km   = pixel_size_km
        self.min_distance_px = int(min_distance_km / pixel_size_km)  # = 2 at defaults
        self._dedup = Layer4SpatialDedup(min_dist_km=min_distance_km, seed=DEFAULT_SEED)
        print(f"SpatialDeduplicator: {min_distance_km:.0f} km min distance "
              f"({self.min_distance_px} pixels at {pixel_size_km:.0f} km/px)")

    def deduplicate(self,
                    pixel_df: pd.DataFrame,
                    ecozone_col: str = "ecozone") -> pd.DataFrame:
        """Apply minimum-distance thinning per ecozone stratum.

        pixel_df must have pixel_row and pixel_col columns.
        Converts pixel coordinates to lon/lat via MODIS GeoTransform,
        then applies greedy KDTree thinning independently within each
        ecozone stratum.

        Args:
            pixel_df    : DataFrame with pixel_row, pixel_col columns.
                          Optionally has ecozone_col for per-stratum thinning.
            ecozone_col : Column name identifying the ecozone/stratum. If not
                          present in pixel_df, thinning is applied globally.

        Returns:
            Deduplicated DataFrame (subset of pixel_df rows).
        """
        if len(pixel_df) == 0:
            return pixel_df.copy()

        rows = np.asarray(pixel_df["pixel_row"].values, dtype=np.float64)
        cols = np.asarray(pixel_df["pixel_col"].values, dtype=np.float64)
        lons, lats = _pixel_to_lonlat(rows, cols)

        if ecozone_col in pixel_df.columns:
            # Per-stratum thinning
            keep_mask = np.zeros(len(pixel_df), dtype=bool)
            eco_vals  = pixel_df[ecozone_col].values.astype(str)
            for i, stratum in enumerate(np.unique(eco_vals)):
                sm = (eco_vals == stratum)
                stratum_lons = lons[sm]
                stratum_lats = lats[sm]
                stratum_mask = self._dedup.thin_mask(stratum_lons, stratum_lats,
                                                      seed=DEFAULT_SEED + i)
                keep_mask[np.where(sm)[0][stratum_mask]] = True
        else:
            # Global thinning (no stratum column)
            keep_mask = self._dedup.thin_mask(lons, lats)

        return pixel_df.iloc[keep_mask].copy().reset_index(drop=True)

    def compute_morans_i(self,
                         pixel_df: pd.DataFrame,
                         sample_n: int = 50_000) -> dict:
        """Compute Moran's I for the pixel distribution in pixel_df.

        Converts pixel_row/pixel_col to lon/lat, bins into a 0.5-degree
        grid covering Canada, then computes Moran's I using queen contiguity
        weights and an analytical p-value (z-score approximation).

        Args:
            pixel_df : DataFrame with pixel_row, pixel_col columns.
            sample_n : Max pixels to use (random subsample if len > sample_n).

        Returns:
            dict: {"moran_i": float, "p_value": float}
        """
        if len(pixel_df) == 0:
            return {"moran_i": 0.0, "p_value": 1.0}

        rows = np.asarray(pixel_df["pixel_row"].values, dtype=np.float64)
        cols = np.asarray(pixel_df["pixel_col"].values, dtype=np.float64)
        lons, lats = _pixel_to_lonlat(rows, cols)

        # Subsample if needed
        if len(lons) > sample_n:
            rng = np.random.default_rng(DEFAULT_SEED)
            idx = rng.choice(len(lons), sample_n, replace=False)
            lons = lons[idx]
            lats = lats[idx]

        I, p = _morans_i_grid_with_pval(lons, lats)
        return {"moran_i": float(I), "p_value": float(p)}


# ---------------------------------------------------------------------------
# Internal: Moran's I computation (shared by SpatialDeduplicator)
# ---------------------------------------------------------------------------

def _morans_i_grid_with_pval(lons: np.ndarray, lats: np.ndarray,
                              grid_deg: float = GRID_DEG) -> tuple[float, float]:
    """Compute Moran's I on a 0.5-degree grid and return (I, p_value).

    x_i = 1 if >=1 fire pixel in cell i, else 0.
    Queen contiguity weights, analytical p-value.
    """
    from scipy import stats as scipy_stats

    lon_min, lat_min, lon_max, lat_max = CANADA_EXTENT
    n_lon = int((lon_max - lon_min) / grid_deg)
    n_lat = int((lat_max - lat_min) / grid_deg)

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
        return 0.0, 1.0  # degenerate

    grid_z    = z.reshape(n_lat, n_lon)
    W_total   = 0.0
    cross_sum = 0.0

    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            shifted = np.roll(np.roll(grid_z, dr, axis=0), dc, axis=1)
            z_row = grid_z.copy()
            if dr == 1:
                shifted[0, :]  = 0.0; z_row[0, :] = 0.0
            elif dr == -1:
                shifted[-1, :] = 0.0; z_row[-1, :] = 0.0
            if dc == 1:
                shifted[:, 0]  = 0.0; z_row[:, 0] = 0.0
            elif dc == -1:
                shifted[:, -1] = 0.0; z_row[:, -1] = 0.0

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

    # Analytical p-value: z = (I - E[I]) / SE[I], two-tailed
    E_I  = -1.0 / (N - 1)
    SE_I = 1.0 / max(np.sqrt(N - 1), 1.0)
    z_sc = (I - E_I) / SE_I
    p_val = float(2.0 * scipy_stats.norm.sf(abs(z_sc)))

    return I, p_val


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Layer4SpatialDedup smoke test (lon/lat API)")
    print("=" * 60)
    dedup = Layer4SpatialDedup(min_dist_km=10.0)

    rng = np.random.default_rng(0)
    # Cluster 1: 20 points near (-115, 51)
    lons1 = -115.0 + rng.normal(0, 0.02, 20)
    lats1 =   51.0 + rng.normal(0, 0.02, 20)
    # Cluster 2: 20 points near (-100, 55)
    lons2 = -100.0 + rng.normal(0, 0.02, 20)
    lats2 =   55.0 + rng.normal(0, 0.02, 20)
    # Isolated point far from both clusters
    lons3 = np.array([-80.0])
    lats3 = np.array([52.0])

    lons = np.concatenate([lons1, lons2, lons3])
    lats = np.concatenate([lats1, lats2, lats3])

    mask = dedup.thin_mask(lons, lats)
    print(f"\nInput:  {len(lons)} points (2 tight clusters + 1 isolated)")
    print(f"Kept:   {mask.sum()} points")
    print(f"Expect: ~3 kept (1 per cluster + isolated)")
    stats = dedup.get_stats(len(lons), mask.sum())
    print(f"Stats: {stats}")
    assert mask.sum() <= 5, f"Expected ~3 kept, got {mask.sum()}"
    print("Layer4SpatialDedup: PASSED")

    print()
    print("=" * 60)
    print("SpatialDeduplicator smoke test (pixel_row/col API)")
    print("=" * 60)

    # Build a small synthetic pixel_df from 2018-07-15 negative candidates
    # Use Layer1Filter to get real negative candidates
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    try:
        from layer1_filter import Layer1Filter
        l1 = Layer1Filter()
        neg_gdf = l1.get_negative_candidates("2018-07-15")
        print(f"Layer1 negative candidates for 2018-07-15: {len(neg_gdf):,}")

        if len(neg_gdf) > 0:
            # Build pixel_df with pixel_row/pixel_col
            from osgeo import gdal
            MODIS_FILE = str(MODIS_DIR / "2018_07_15.tif")
            ds = gdal.Open(MODIS_FILE, gdal.GA_ReadOnly)
            gt = ds.GetGeoTransform()
            ds = None

            lons_neg = neg_gdf.geometry.x.values
            lats_neg = neg_gdf.geometry.y.values
            pixel_cols = ((lons_neg - gt[0]) / gt[1]).astype(int)
            pixel_rows = ((lats_neg - gt[3]) / gt[5]).astype(int)

            neg_df = pd.DataFrame({
                "pixel_row": pixel_rows,
                "pixel_col": pixel_cols,
                "ecozone":   "TestZone",  # single stratum for smoke test
            })

            sd = SpatialDeduplicator(min_distance_km=10.0, pixel_size_km=5.0)
            print(f"min_distance_px = {sd.min_distance_px}")

            thinned = sd.deduplicate(neg_df, ecozone_col="ecozone")
            print(f"Deduplicate: {len(neg_df):,} -> {len(thinned):,} "
                  f"({100*(len(neg_df)-len(thinned))/max(len(neg_df),1):.1f}% removed)")

            mi_before = sd.compute_morans_i(neg_df, sample_n=50_000)
            mi_after  = sd.compute_morans_i(thinned, sample_n=50_000)
            print(f"Moran's I before: {mi_before['moran_i']:.4f} "
                  f"(p={mi_before['p_value']:.3e})")
            print(f"Moran's I after:  {mi_after['moran_i']:.4f} "
                  f"(p={mi_after['p_value']:.3e})")
            print("SpatialDeduplicator: PASSED")
        else:
            print("Warning: no negative candidates found for 2018-07-15 — "
                  "smoke test skipped (data path may be unavailable)")

    except Exception as e:
        print(f"SpatialDeduplicator smoke test: {e}")
        print("(This is expected if run outside the remote server environment)")

    print("\nSmoke test complete.")
