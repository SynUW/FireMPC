"""Phase 2 Layer 2 — Raw positive distribution by ecozone x LC x month.

Counts high-confidence fire pixels (firemask==9) per stratum (ecozone, LC_type, month)
across all available fire years (2000-2025). Quantifies the geographic and seasonal skew
in raw positive samples before stratification.

Data sources (all updated to 2026-04-13 merged data):
  MODIS fire: MCD14A1_combined_5xMaximumDown_merged/yyyy_mm_dd.tif (single merged file/day)
  Land cover: MCD12Q1/MCD12Q1_YYYY_01_01_tileNN.tif (4 tiles/year, IGBP band 1)
  Ecozones:   ecozones.shp (15 unique ecozones, ZONE_NAME column, 25 polygons)
  Research:   research_area_shpfile_merge.shp

Method:
  For each year 2000-2025:
    - Sample up to DAYS_PER_YEAR fire days from peak season (Jun-Sep)
    - Read merged MODIS file, extract fire pixels (firemask==9)
    - Apply research area mask
    - Look up land cover from MCD12Q1 (year-matched; clamp to 2001-2024)
    - Assign ecozone via spatial join (ZONE_NAME column)
    - Record (ecozone, lc_class, month, count)

Output:
  raw_positive_distribution_by_ecoregion.csv  (cols: ecoregion, lc_class, month,
                                                      positive_count, fraction)
  raw_positive_distribution_by_ecoregion.png  (bar chart + Lorenz curve)
  positive_distribution.log

CRITICAL: GDAL reads use ReadRaster()+frombuffer — NOT ReadAsArray.
Note: 'ecoregion' column name preserved for compatibility with stratum_quotas.py;
      values are ecozone names (ZONE_NAME) from ecozones.shp.
"""
import os
import glob as glob_module
import logging
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from osgeo import gdal

# --------------------------------------------------------------------------- #
# Paths                                                                         #
# --------------------------------------------------------------------------- #
MODIS_DIR = (
    "/s3/user/wildfire_dataset/self_built_materials/"
    "bc_raw_data_new_fast/drivers/processed/"
    "spatial_filled_5xdownsampled_merged/MCD14A1_combined_5xMaximumDown_merged"
)
LC_DIR = (
    "/s3/user/wildfire_dataset/self_built_materials/"
    "bc_raw_data_new_fast/drivers/processed/MCD12Q1"
)
ECOZONE_SHP = (
    "/s3/user/wildfire_dataset/self_built_materials/"
    "bc_raw_data_new_fast/drivers/ecoregions/ecozones.shp"
)
RESEARCH_AREA_SHP = (
    "/s3/user/wildfire_dataset/self_built_materials/"
    "bc_raw_data_new_fast/drivers/ecoregions/research_area_shpfile_merge.shp"
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = _THIS_DIR

# --------------------------------------------------------------------------- #
# Constants                                                                     #
# --------------------------------------------------------------------------- #
FIREMASK_HIGH_CONF = 9
LC_FIRE_CLASSES    = set(range(1, 12))   # IGBP classes 1-11 (fire-relevant only)
LC_MIN_YEAR        = 2001
LC_MAX_YEAR        = 2024
DAYS_PER_YEAR      = 15     # sample up to N fire days per year from Jun-Sep
PEAK_MONTHS        = {6, 7, 8, 9}

# --------------------------------------------------------------------------- #
# Logging                                                                       #
# --------------------------------------------------------------------------- #
log_path = os.path.join(OUTPUT_DIR, "positive_distribution.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(log_path, mode='w'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# GDAL readers                                                                  #
# --------------------------------------------------------------------------- #
def _read_band1(path: str) -> tuple | None:
    """Read band 1 via ReadRaster()+frombuffer. Returns (array_uint8, gt) or None."""
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        return None
    gt = ds.GetGeoTransform()
    W, H = ds.RasterXSize, ds.RasterYSize
    raw = ds.GetRasterBand(1).ReadRaster(0, 0, W, H, buf_type=gdal.GDT_Byte)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(H, W)
    ds = None
    return arr, gt


def _pixel_lonlat(rows_f: np.ndarray, cols_f: np.ndarray,
                  gt: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Convert pixel row/col indices to (lon, lat) via GeoTransform."""
    lons = gt[0] + (cols_f + 0.5) * gt[1]
    lats = gt[3] + (rows_f + 0.5) * gt[5]   # gt[5] < 0
    return lons, lats


# --------------------------------------------------------------------------- #
# Land cover lookup (cached per year)                                           #
# --------------------------------------------------------------------------- #
_lc_cache: dict[int, list] = {}


def _get_lc_tiles(year: int) -> list:
    lc_year = max(LC_MIN_YEAR, min(LC_MAX_YEAR, year))
    if lc_year in _lc_cache:
        return _lc_cache[lc_year]
    pattern = os.path.join(LC_DIR, f"MCD12Q1_{lc_year}_01_01_tile*.tif")
    tiles = sorted(glob_module.glob(pattern))
    result = []
    for tp in tiles:
        r = _read_band1(tp)
        if r is not None:
            result.append(r)
    if not result:
        log.warning(f"  No LC tiles found for year {lc_year}: {pattern}")
    _lc_cache[lc_year] = result
    return result


def lookup_lc(lons: np.ndarray, lats: np.ndarray, year: int) -> np.ndarray:
    """Return IGBP LC_Type1 value for each (lon, lat). 0 = outside all tiles."""
    tiles = _get_lc_tiles(year)
    result = np.zeros(len(lons), dtype=np.uint8)
    for arr, gt in tiles:
        H, W = arr.shape
        cols_f = (lons - gt[0]) / gt[1]
        rows_f = (lats - gt[3]) / gt[5]
        in_tile = (cols_f >= 0) & (cols_f < W) & (rows_f >= 0) & (rows_f < H)
        if not in_tile.any():
            continue
        ci = cols_f[in_tile].astype(np.int32).clip(0, W - 1)
        ri = rows_f[in_tile].astype(np.int32).clip(0, H - 1)
        result[in_tile] = arr[ri, ci]
    return result


# --------------------------------------------------------------------------- #
# Ecozone and research area (cached)                                            #
# --------------------------------------------------------------------------- #
_eco_gdf_cache  = None
_research_union = None


def _load_eco() -> gpd.GeoDataFrame:
    global _eco_gdf_cache
    if _eco_gdf_cache is not None:
        return _eco_gdf_cache
    gdf = gpd.read_file(ECOZONE_SHP)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4617")
    gdf = gdf.to_crs("EPSG:4326")
    _eco_gdf_cache = gdf[['ZONE_NAME', 'geometry']].copy()
    log.info(f"Loaded {len(_eco_gdf_cache)} ecozone polygons ({gdf['ZONE_NAME'].nunique()} unique zones)")
    return _eco_gdf_cache


def _load_research_union():
    global _research_union
    if _research_union is not None:
        return _research_union
    ra = gpd.read_file(RESEARCH_AREA_SHP)
    if ra.crs is None:
        ra = ra.set_crs("EPSG:4617")
    ra = ra.to_crs("EPSG:4326")
    _research_union = ra.geometry.unary_union
    return _research_union


def assign_ecozone(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Return ZONE_NAME for each (lon, lat). 'OUTSIDE' if not in any ecozone."""
    eco = _load_eco()
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lons, lats), crs="EPSG:4326")
    joined = gpd.sjoin(pts, eco, how='left', predicate='within')
    return joined['ZONE_NAME'].fillna('OUTSIDE').values.astype(str)


_research_prep_cache = None


def _load_research_prepared():
    """Return a shapely PreparedGeometry for the research area union (EPSG:4326).
    Uses prepared geometry for fast repeated point-in-polygon tests.
    """
    global _research_prep_cache
    if _research_prep_cache is not None:
        return _research_prep_cache
    from shapely import prepare
    ra = gpd.read_file(RESEARCH_AREA_SHP)
    if ra.crs is None:
        ra = ra.set_crs("EPSG:4617")
    ra4326 = ra.to_crs("EPSG:4326")
    union_geom = ra4326.geometry.union_all()
    prepare(union_geom)
    _research_prep_cache = union_geom
    return _research_prep_cache


def filter_research_area(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Boolean mask: True if point within research_area_shpfile_merge.
    Uses prepared shapely geometry for fast vectorized point-in-polygon.
    """
    from shapely import contains_xy
    geom = _load_research_prepared()
    return contains_xy(geom, lons, lats)


# --------------------------------------------------------------------------- #
# Main analysis                                                                 #
# --------------------------------------------------------------------------- #
def collect_positive_distribution() -> pd.DataFrame:
    """Iterate all years, sample fire days, accumulate per-stratum counts."""
    # Discover available dates grouped by year
    all_files = sorted(glob_module.glob(os.path.join(MODIS_DIR, "*.tif")))
    if not all_files:
        raise RuntimeError(f"No MODIS files found in: {MODIS_DIR}")

    year_days: dict[int, list] = {}
    for fp in all_files:
        stem = os.path.basename(fp).replace('.tif', '')   # e.g. 2018_07_15
        parts = stem.split('_')
        if len(parts) != 3:
            continue
        try:
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        if m not in PEAK_MONTHS:
            continue
        year_days.setdefault(y, []).append((m, d, fp))

    records = []

    for year in sorted(year_days.keys()):
        candidates = year_days[year]
        rng = np.random.default_rng(seed=year)
        if len(candidates) > DAYS_PER_YEAR:
            idxs = rng.choice(len(candidates), size=DAYS_PER_YEAR, replace=False)
            sampled = [candidates[i] for i in sorted(idxs)]
        else:
            sampled = candidates

        log.info(f"Year {year}...")
        log.info(f"  Sampling {len(sampled)} fire days...")

        for month, day, fpath in sampled:
            stem = f"MCD14A1_{year}_{month:02d}_{day:02d}"
            result = _read_band1(fpath)
            if result is None:
                log.warning(f"    {stem}: cannot open")
                continue
            arr, gt = result

            fire_mask = arr == FIREMASK_HIGH_CONF
            n_fire = int(fire_mask.sum())
            if n_fire == 0:
                continue

            rows_f, cols_f = np.where(fire_mask)
            lons, lats = _pixel_lonlat(rows_f.astype(float), cols_f.astype(float), gt)

            # Research area filter
            ra_mask = filter_research_area(lons, lats)
            lons, lats = lons[ra_mask], lats[ra_mask]
            if len(lons) == 0:
                log.info(f"    {stem}: {n_fire:,} fire px, 0 within research area")
                continue

            # LC lookup
            lc_vals = lookup_lc(lons, lats, year)

            # Keep fire-relevant LC only
            fr_mask = np.array([v in LC_FIRE_CLASSES for v in lc_vals])
            lons, lats = lons[fr_mask], lats[fr_mask]
            lc_vals = lc_vals[fr_mask]

            if len(lons) == 0:
                log.info(f"    {stem}: {n_fire:,} fire px, 0 fire-relevant LC")
                continue

            log.info(f"    {stem}: {n_fire:,} fire px, {len(lons):,} fire-relevant LC")

            # Ecozone assignment
            eco_vals = assign_ecozone(lons, lats)

            for eco, lc in zip(eco_vals, lc_vals):
                if eco == 'OUTSIDE':
                    continue
                records.append({'ecoregion': eco, 'lc_class': int(lc), 'month': month})

    if not records:
        log.warning("No fire pixels collected — check paths.")
        return pd.DataFrame(
            columns=['ecoregion', 'lc_class', 'month', 'positive_count', 'fraction'])

    df = pd.DataFrame(records)
    agg = (df.groupby(['ecoregion', 'lc_class', 'month'])
             .size()
             .reset_index(name='positive_count')
             .sort_values('positive_count', ascending=False)
             .reset_index(drop=True))
    total = agg['positive_count'].sum()
    agg['fraction'] = agg['positive_count'] / total
    return agg


# --------------------------------------------------------------------------- #
# Figure                                                                        #
# --------------------------------------------------------------------------- #
def save_figure(df: pd.DataFrame, out_path: str) -> None:
    eco_totals = (df.groupby('ecoregion')['positive_count']
                    .sum()
                    .sort_values(ascending=False))

    n = len(eco_totals)
    sorted_vals = np.sort(eco_totals.values)
    cumvals = np.cumsum(sorted_vals)
    gini = float(1 - 2 * np.sum(cumvals) / (n * cumvals[-1]) + 1 / n)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: horizontal bar chart
    ax = axes[0]
    colors = ['salmon' if v / eco_totals.iloc[0] > 0.05 else 'steelblue'
              for v in eco_totals.values]
    ax.barh(range(n), eco_totals.values, color=colors)
    ax.set_yticks(range(n))
    ax.set_yticklabels([e[:40] for e in eco_totals.index], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Raw positive sample count')
    ax.set_title('Raw Positive Distribution by Ecozone\n(before stratification)')
    ax.text(0.98, 0.02, f'Gini = {gini:.3f}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    # Right: Lorenz curve
    ax2 = axes[1]
    lorenz_x = np.linspace(0, 1, n)
    lorenz_y = cumvals / cumvals[-1]
    ax2.plot(lorenz_x, lorenz_y, color='steelblue', lw=2, label='Actual')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect equality')
    ax2.fill_between(lorenz_x, lorenz_y, lorenz_x, alpha=0.2, color='steelblue')
    ax2.set_xlabel('Cumulative ecozone fraction')
    ax2.set_ylabel('Cumulative positive sample fraction')
    ax2.set_title(f'Lorenz Curve (Gini = {gini:.3f})\nTarget: reduce via stratification')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved: {out_path}")


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    log.info("=== Phase 2 Layer 2: positive_distribution.py ===")
    log.info(f"MODIS dir:     {MODIS_DIR}")
    log.info(f"LC dir:        {LC_DIR}")
    log.info(f"Ecozone shp:   {ECOZONE_SHP}")
    log.info(f"Research area: {RESEARCH_AREA_SHP}")
    log.info("")

    dist_df = collect_positive_distribution()

    csv_path = os.path.join(OUTPUT_DIR, "raw_positive_distribution_by_ecoregion.csv")
    dist_df.to_csv(csv_path, index=False)
    log.info(f"\nSaved: {csv_path}  ({len(dist_df)} strata)")

    png_path = os.path.join(OUTPUT_DIR, "raw_positive_distribution_by_ecoregion.png")
    save_figure(dist_df, png_path)

    log.info("\n=== Top 10 strata by count ===")
    log.info(dist_df.head(10).to_string(index=False))
