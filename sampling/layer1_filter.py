"""Phase 2 — Layer 1: MODIS Reliability Filter (reusable module).

Implements the Layer 1 reliability filter for the CanWildfire sampling pipeline.
This module is imported by Layer 2, 3, 4 and the split construction (Plans 02-05).

Filter logic (from Phase 1 calibration):
  Positive pixels:   firemask == 9 (high confidence only)
  Exclusion buffer:  spatial 30 km + temporal +/-3 days around any fire event;
                     also excludes firemask in {7, 8} pixels
  Cloud exclusion:   firemask == 4 (never a negative candidate)
  Negative candidate: firemask == 5 AND outside all spatial/temporal buffers
                      AND within research_area_shpfile_merge.shp

Data source: MCD14A1_combined_5xMaximumDown_merged — daily merged GeoTIFF
             (single file per day, firemask band 1)
File naming: yyyy_mm_dd.tif  (e.g. 2018_07_15.tif)

GDAL note: ALL raster reads use ReadRaster()+frombuffer — NOT ReadAsArray.
           (ReadAsArray triggers gdal_array NumPy ABI error in non-patches envs.)

Research area: all candidate pixels (positive and negative) are masked to
               research_area_shpfile_merge.shp before returning.

Usage:
    from layer1_filter import Layer1Filter
    f = Layer1Filter()
    # Get all fire pixel locations for a given date:
    fire_df = f.get_fire_pixels("2018-07-15")
    # Get negative candidates for a date (with spatial+temporal buffer applied):
    neg_df = f.get_negative_candidates("2018-07-15")
"""
import os
import glob as glob_module
import datetime
from typing import Optional
import numpy as np
import geopandas as gpd
import pandas as pd
from osgeo import gdal, osr

# --------------------------------------------------------------------------- #
# Paths                                                                         #
# --------------------------------------------------------------------------- #
MODIS_DIR = (
    "/s3/user/wildfire_dataset/self_built_materials/"
    "bc_raw_data_new_fast/drivers/processed/"
    "spatial_filled_5xdownsampled_merged/MCD14A1_combined_5xMaximumDown_merged"
)
RESEARCH_AREA_SHP = (
    "/s3/user/wildfire_dataset/self_built_materials/"
    "bc_raw_data_new_fast/drivers/ecoregions/research_area_shpfile_merge.shp"
)

# --------------------------------------------------------------------------- #
# Constants from Phase 1 calibration                                            #
# --------------------------------------------------------------------------- #
BUFFER_RADIUS_KM     = 30         # SAMP-01: calibrated buffer radius
TEMPORAL_BUFFER_DAYS = 3          # +/- days around any fire event to exclude
FIREMASK_HIGH_CONF   = 9          # positive label
FIREMASK_NOM_CONF    = 8          # exclusion buffer (not positive)
FIREMASK_LOW_CONF    = 7          # exclusion buffer (not positive)
FIREMASK_CLOUD       = 4          # cloud — always excluded
FIREMASK_NONFIRE_LAND = 5         # valid negative candidate (before buffer)
FIREMASK_WATER       = 3          # water — always excluded
EPSG_CANADA          = "EPSG:3978"    # Canada Atlas Lambert (buffer computations)
EPSG_WGS84           = "EPSG:4326"


# --------------------------------------------------------------------------- #
# Research area (cached)                                                        #
# --------------------------------------------------------------------------- #
_research_union_cache = None


def _get_research_union():
    """Return shapely geometry union of the research area (EPSG:4326)."""
    global _research_union_cache
    if _research_union_cache is not None:
        return _research_union_cache
    gdf = gpd.read_file(RESEARCH_AREA_SHP)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4617")
    gdf = gdf.to_crs(EPSG_WGS84)
    _research_union_cache = gdf.geometry.unary_union
    return _research_union_cache


# --------------------------------------------------------------------------- #
# Low-level GDAL reader                                                          #
# --------------------------------------------------------------------------- #
def _read_firemask(date_str: str) -> tuple[np.ndarray, tuple] | None:
    """Read firemask band from a MCD14A1 merged daily file.

    date_str: 'YYYY-MM-DD'
    Returns (firemask_array uint8, geotransform tuple) or None if not found.
    Uses ReadRaster()+frombuffer to avoid gdal_array NumPy ABI issues.
    """
    parts = date_str.split("-")
    fname = f"{parts[0]}_{parts[1]}_{parts[2]}.tif"
    fpath = os.path.join(MODIS_DIR, fname)
    if not os.path.exists(fpath):
        return None
    ds = gdal.Open(fpath, gdal.GA_ReadOnly)
    if ds is None:
        return None
    gt = ds.GetGeoTransform()
    W, H = ds.RasterXSize, ds.RasterYSize
    raw = ds.GetRasterBand(1).ReadRaster(0, 0, W, H, buf_type=gdal.GDT_Byte)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(H, W)
    ds = None
    return arr, gt


def _pixel_centroids(arr: np.ndarray, gt: tuple,
                     mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (lon_array, lat_array) in EPSG:4326 for pixels where mask is True."""
    H, W = arr.shape
    cols  = np.arange(W, dtype=np.float64)
    rows  = np.arange(H, dtype=np.float64)
    xc = gt[0] + (cols + 0.5) * gt[1]   # longitudes
    yc = gt[3] + (rows + 0.5) * gt[5]   # latitudes (gt[5] < 0)
    xx, yy = np.meshgrid(xc, yc)
    return xx[mask], yy[mask]


# --------------------------------------------------------------------------- #
# Public API                                                                    #
# --------------------------------------------------------------------------- #
class Layer1Filter:
    """Layer 1 MODIS Reliability Filter.

    Provides:
      get_fire_pixels(date_str)       — GeoDataFrame of positive (firemask==9) pixels
      get_negative_candidates(date_str, context_dates) — GeoDataFrame of valid
                                          negative candidates after buffer exclusion
    Both methods mask results to research_area_shpfile_merge.shp.
    """

    def __init__(self,
                 buffer_radius_km: float = BUFFER_RADIUS_KM,
                 temporal_buffer_days: int = TEMPORAL_BUFFER_DAYS,
                 apply_research_mask: bool = True):
        self.buffer_radius_km      = buffer_radius_km
        self.temporal_buffer_days  = temporal_buffer_days
        self.apply_research_mask   = apply_research_mask

    # ------------------------------------------------------------------ #
    # Fire pixels for one date                                             #
    # ------------------------------------------------------------------ #
    def get_fire_pixels(self, date_str: str) -> gpd.GeoDataFrame:
        """Return GeoDataFrame of high-confidence fire pixels for date_str.

        Args:
            date_str: 'YYYY-MM-DD'

        Returns:
            GeoDataFrame in EPSG:4326, columns ['date', 'firemask', 'geometry'].
            Masked to research area. Empty if no file or no fire pixels found.
        """
        result = _read_firemask(date_str)
        if result is None:
            return _empty_gdf(date_str)

        arr, gt = result
        fire_mask = arr == FIREMASK_HIGH_CONF
        if fire_mask.sum() == 0:
            return _empty_gdf(date_str)

        lons, lats = _pixel_centroids(arr, gt, fire_mask)
        n = len(lons)
        gdf = gpd.GeoDataFrame(
            {'date': [date_str] * n,
             'firemask': np.full(n, FIREMASK_HIGH_CONF, dtype=np.uint8)},
            geometry=gpd.points_from_xy(lons, lats),
            crs=EPSG_WGS84
        )

        if self.apply_research_mask:
            ra_union = _get_research_union()
            gdf = gdf[gdf.geometry.within(ra_union)].copy()

        return gdf

    # ------------------------------------------------------------------ #
    # Negative candidates for one date                                     #
    # ------------------------------------------------------------------ #
    def get_negative_candidates(
        self,
        date_str: str,
        context_dates: Optional[list] = None,
    ) -> gpd.GeoDataFrame:
        """Return negative candidate pixels after Layer 1 filtering.

        Applies:
          1. Cloud exclusion  (firemask==4 → excluded)
          2. Spatial buffer   (30 km around ALL fire pixels in context_dates)
          3. Temporal buffer  (context_dates = +/-TEMPORAL_BUFFER_DAYS around date_str)
          4. Research area mask

        Args:
            date_str:      Target date 'YYYY-MM-DD' to extract negative candidates.
            context_dates: List of date strings to use as fire pixel sources for
                           spatial buffer construction. If None, auto-generates
                           +/-TEMPORAL_BUFFER_DAYS dates around date_str.

        Returns:
            GeoDataFrame in EPSG:4326, columns ['date', 'firemask', 'geometry'].
            Only firemask==5 pixels not within any spatial exclusion buffer,
            within the research area.
        """
        if context_dates is None:
            context_dates = _surrounding_dates(date_str, self.temporal_buffer_days)

        # Step 1: extract negative candidate pixels (firemask==5) for date_str
        result = _read_firemask(date_str)
        if result is None:
            return _empty_gdf(date_str)

        arr, gt = result
        cand_mask = arr == FIREMASK_NONFIRE_LAND
        if cand_mask.sum() == 0:
            return _empty_gdf(date_str)

        cand_lons, cand_lats = _pixel_centroids(arr, gt, cand_mask)
        n_cand = len(cand_lons)

        cand_gdf = gpd.GeoDataFrame(
            {'date': [date_str] * n_cand,
             'firemask': np.full(n_cand, FIREMASK_NONFIRE_LAND, dtype=np.uint8)},
            geometry=gpd.points_from_xy(cand_lons, cand_lats),
            crs=EPSG_WGS84
        )

        # Step 2: research area mask (apply before buffer check for efficiency)
        if self.apply_research_mask:
            ra_union = _get_research_union()
            cand_gdf = cand_gdf[cand_gdf.geometry.within(ra_union)].copy()
            if len(cand_gdf) == 0:
                return _empty_gdf(date_str)

        # Step 3: collect fire/exclusion pixels from context_dates for spatial buffer
        fire_lons, fire_lats = [], []
        for d in context_dates:
            d_result = _read_firemask(d)
            if d_result is None:
                continue
            d_arr, d_gt = d_result
            # Exclude firemask in {7, 8, 9} from negative pool spatially
            excl_mask = (d_arr == FIREMASK_HIGH_CONF) | \
                        (d_arr == FIREMASK_NOM_CONF)  | \
                        (d_arr == FIREMASK_LOW_CONF)
            if excl_mask.sum() == 0:
                continue
            lons, lats = _pixel_centroids(d_arr, d_gt, excl_mask)
            fire_lons.append(lons)
            fire_lats.append(lats)

        # Step 4: apply spatial buffer in EPSG:3978
        if not fire_lons:
            # No fire context — all remaining candidates are valid
            return cand_gdf

        fire_gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(
                np.concatenate(fire_lons),
                np.concatenate(fire_lats)
            ),
            crs=EPSG_WGS84
        ).to_crs(EPSG_CANADA)

        cand_3978 = cand_gdf.to_crs(EPSG_CANADA)
        radius_m  = self.buffer_radius_km * 1000.0

        from shapely.ops import unary_union
        exclusion_zone = unary_union(fire_gdf.geometry.buffer(radius_m))
        not_excluded   = ~cand_3978.geometry.within(exclusion_zone)

        result_gdf = cand_gdf[not_excluded.values].copy()
        return result_gdf

    # ------------------------------------------------------------------ #
    # Convenience: all dates in a date range                               #
    # ------------------------------------------------------------------ #
    def iter_dates(self, start_year: int = 2000,
                   end_year: int = 2025) -> list:
        """Return list of date strings 'YYYY-MM-DD' for all available MODIS tiles."""
        files = sorted(glob_module.glob(os.path.join(MODIS_DIR, "*.tif")))
        dates = []
        seen = set()
        for fpath in files:
            fname = os.path.basename(fpath)  # e.g. 2018_07_15.tif
            parts = fname.replace('.tif', '').split('_')
            if len(parts) != 3:
                continue
            try:
                y = int(parts[0])
                m = int(parts[1])
                d = int(parts[2])
            except ValueError:
                continue
            if not (start_year <= y <= end_year):
                continue
            ds = f"{y:04d}-{m:02d}-{d:02d}"
            if ds not in seen:
                seen.add(ds)
                dates.append(ds)
        return sorted(dates)


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #
def _surrounding_dates(date_str: str, n_days: int) -> list:
    """Return list of dates within +/-n_days of date_str (inclusive of date_str)."""
    dt = datetime.date.fromisoformat(date_str)
    result = []
    for delta in range(-n_days, n_days + 1):
        result.append((dt + datetime.timedelta(days=delta)).isoformat())
    return result


def _empty_gdf(date_str: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {'date': pd.Series([], dtype=str),
         'firemask': pd.Series([], dtype=np.uint8)},
        geometry=gpd.GeoSeries([], crs=EPSG_WGS84),
        crs=EPSG_WGS84
    )


# --------------------------------------------------------------------------- #
# Quick smoke test                                                              #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    f = Layer1Filter()
    print("Layer1Filter initialized.")
    print(f"Buffer: {f.buffer_radius_km} km spatial, +/-{f.temporal_buffer_days} days temporal")
    print(f"Research area masking: {f.apply_research_mask}")

    test_date = "2018-07-15"
    print(f"\nTesting with {test_date}...")

    fire_df = f.get_fire_pixels(test_date)
    print(f"  Fire pixels (firemask==9): {len(fire_df):,}")

    neg_df = f.get_negative_candidates(test_date)
    print(f"  Negative candidates after buffer: {len(neg_df):,}")

    all_dates = f.iter_dates(2018, 2018)
    print(f"  Available dates in 2018: {len(all_dates)}")
    if all_dates:
        print(f"  First date: {all_dates[0]}, last date: {all_dates[-1]}")
