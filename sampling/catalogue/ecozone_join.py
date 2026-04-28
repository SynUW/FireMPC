"""Ecozone spatial join: assign Canadian terrestrial ecozone labels to (lon, lat) coordinates.

Uses geopandas.sjoin for efficient spatial indexing (NOT a custom point-in-polygon loop).
All geometries are in EPSG:4326 (WGS84) for input; shapefile CRS is handled automatically.

Reference: Ecoregions of Canada shapefile from /s3/.../drivers/ecoregions/
The 15 terrestrial ecozones are the standard Environment Canada ecozone classification.
Shapefile columns include ECOZONE (int, 1-15) and REGION_NAM (ecoregion name string).
"""
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import os

# Path to ecoregion shapefile directory on remote server
ECOREGION_DIR = ("/s3/zhengsen/wildfire_dataset/self_built_materials/"
                 "bc_raw_data_new_fast/drivers/ecoregions")

# Module-level cache to avoid re-loading the shapefile on every call
_ecozone_gdf_cache = None


def _load_ecozone_shapefile() -> gpd.GeoDataFrame:
    """Load ecozone shapefile once and cache. Returns GeoDataFrame in EPSG:4326."""
    global _ecozone_gdf_cache
    if _ecozone_gdf_cache is not None:
        return _ecozone_gdf_cache

    shp_files = [f for f in os.listdir(ECOREGION_DIR) if f.endswith('.shp')]
    if not shp_files:
        raise FileNotFoundError(f"No .shp file found in {ECOREGION_DIR}")
    shp_path = os.path.join(ECOREGION_DIR, shp_files[0])

    gdf = gpd.read_file(shp_path)
    # Reproject to WGS84 (EPSG:4326) for consistent sjoin with input points
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    _ecozone_gdf_cache = gdf
    return gdf


def get_ecozone_name_column(gdf: gpd.GeoDataFrame) -> str:
    """Auto-detect the ecozone name column from common column names.

    Prefers string columns that represent ecozone names; falls back to numeric
    ECOZONE ID. Returns the column name to use for labelling.
    """
    # Preferred string columns in priority order
    candidate_cols = ['ZONE_NAME', 'ECO_NAME', 'ECOZONE_NM', 'ECOZ_NAME',
                      'REGION_NAM', 'ecozone', 'zone_name', 'eco_name', 'NAME']
    for col in candidate_cols:
        if col in gdf.columns and gdf[col].dtype == object:
            return col
    # Accept numeric ECOZONE column -- values will be converted to strings
    if 'ECOZONE' in gdf.columns:
        return 'ECOZONE'
    # Last resort: first object column that is not geometry
    for col in gdf.columns:
        if col != 'geometry' and gdf[col].dtype == object:
            return col
    raise ValueError(f"Cannot detect ecozone name column. Available: {gdf.columns.tolist()}")


def assign_ecozone_labels(lons, lats) -> list:
    """Assign Canadian terrestrial ecozone labels to coordinate arrays.

    Args:
        lons: array-like of longitudes in degrees (WGS84, negative for west)
        lats: array-like of latitudes in degrees (WGS84)

    Returns:
        list of ecozone label strings. 'OUTSIDE_CANADA' for points outside all ecozones.

    Note: Uses geopandas.sjoin for spatial indexing -- NOT a pixel-by-pixel loop.
    For large inputs (>100K points), pass numpy arrays directly.
    """
    lons = np.asarray(lons, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)

    points_gdf = gpd.GeoDataFrame(
        {'lon': lons, 'lat': lats},
        geometry=[Point(x, y) for x, y in zip(lons, lats)],
        crs="EPSG:4326"
    )

    ecozone_gdf = _load_ecozone_shapefile()
    name_col = get_ecozone_name_column(ecozone_gdf)

    joined = gpd.sjoin(points_gdf, ecozone_gdf[[name_col, 'geometry']],
                       how='left', predicate='within')

    # Convert to string (handles numeric ECOZONE column) and fill NaN
    raw_labels = joined[name_col]
    labels = []
    for val in raw_labels:
        if val is None:
            labels.append('OUTSIDE_CANADA')
        else:
            try:
                import math
                if math.isnan(float(val)):
                    labels.append('OUTSIDE_CANADA')
                    continue
            except (TypeError, ValueError):
                pass
            s = str(val).strip()
            labels.append(s if s and s != 'nan' else 'OUTSIDE_CANADA')

    return labels
