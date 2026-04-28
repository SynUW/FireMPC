"""Phase 2 Layer 1 — Cloud contamination analysis by ecozone x season.

Quantifies the fraction of MODIS observations that are cloud-masked (firemask==4)
per ecozone and meteorological season. This justifies why cloud-masked pixels
must be excluded from negative candidate selection.

Method:
  - For each year 2000-2025, for each of the 12 calendar months,
    sample DAYS_PER_MONTH days evenly across the month.
  - For each sampled day: read merged daily TIF, mask to research area,
    assign ecozone labels, compute cloud fraction per ecozone.
  - Aggregate to (ecozone, season): mean +/- std cloud fraction across all
    sampled year-month observations.

Output:
  - cloud_contamination_by_ecoregion_season.csv  (kept for naming compatibility)
  - cloud_contamination_by_ecoregion_season.png  (heatmap)

Seasons:
  DJF = Dec/Jan/Feb  |  MAM = Mar/Apr/May
  JJA = Jun/Jul/Aug  |  SON = Sep/Oct/Nov

CRITICAL: GDAL reads use ReadRaster()+frombuffer — NOT ReadAsArray (NumPy ABI issue).
CRITICAL: Files are pre-merged per day. Format: yyyy_mm_dd.tif (no tiles).
CRITICAL: Mask all pixels to research_area_shpfile_merge.shp before counting.
"""
import csv
import os
import glob as glob_module
import random
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
    "/s3/zhengsen/wildfire_dataset/self_built_materials/"
    "bc_raw_data_new_fast/drivers/processed/"
    "spatial_filled_5xdownsampled_merged/MCD14A1_combined_5xMaximumDown_merged"
)
ECOZONE_SHP = (
    "/s3/zhengsen/wildfire_dataset/self_built_materials/"
    "bc_raw_data_new_fast/drivers/ecoregions/ecozones.shp"
)
RESEARCH_AREA_SHP = (
    "/s3/zhengsen/wildfire_dataset/self_built_materials/"
    "bc_raw_data_new_fast/drivers/ecoregions/research_area_shpfile_merge.shp"
)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
# Constants                                                                     #
# --------------------------------------------------------------------------- #
YEARS         = list(range(2000, 2026))
MONTHS        = list(range(1, 13))
DAYS_PER_MONTH = 2          # sample this many days per month per year
MAX_PIXELS    = 60_000      # land pixel subsample per daily file
RANDOM_SEED   = 42

SEASON_MAP = {
    12: 'DJF', 1: 'DJF', 2: 'DJF',
    3: 'MAM', 4: 'MAM', 5: 'MAM',
    6: 'JJA', 7: 'JJA', 8: 'JJA',
    9: 'SON', 10: 'SON', 11: 'SON',
}
FIREMASK_CLOUD        = 4
FIREMASK_NONFIRE_LAND = 5


# --------------------------------------------------------------------------- #
# Ecozone + research area loader (cached)                                       #
# --------------------------------------------------------------------------- #
_eco_gdf_cache = None
_research_area_cache = None
_research_union_cache = None


def _load_ecozone_gdf() -> gpd.GeoDataFrame:
    global _eco_gdf_cache
    if _eco_gdf_cache is not None:
        return _eco_gdf_cache
    gdf = gpd.read_file(ECOZONE_SHP)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4617")
    gdf = gdf.to_crs("EPSG:4326")
    _eco_gdf_cache = gdf
    print(f"  Ecozone shapefile loaded: {len(gdf)} polygons")
    return gdf


def _load_research_union():
    """Return shapely geometry union of the research area."""
    global _research_union_cache
    if _research_union_cache is not None:
        return _research_union_cache
    gdf = gpd.read_file(RESEARCH_AREA_SHP)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4617")
    gdf = gdf.to_crs("EPSG:4326")
    _research_union_cache = gdf.geometry.unary_union
    print(f"  Research area loaded.")
    return _research_union_cache


# --------------------------------------------------------------------------- #
# GDAL daily-file reader                                                        #
# --------------------------------------------------------------------------- #
def _read_firemask(date_str: str) -> tuple[np.ndarray, tuple] | None:
    """Return (firemask_array, geotransform) for one merged daily file.

    date_str format: 'YYYY-MM-DD'
    File naming: yyyy_mm_dd.tif
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


def _pixel_centroids_flat(arr: np.ndarray, gt: tuple,
                           mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    H, W = arr.shape
    cols = np.arange(W, dtype=np.float64)
    rows = np.arange(H, dtype=np.float64)
    xcoords = gt[0] + (cols + 0.5) * gt[1]
    ycoords = gt[3] + (rows + 0.5) * gt[5]
    xx, yy = np.meshgrid(xcoords, ycoords)
    return xx[mask], yy[mask]


# --------------------------------------------------------------------------- #
# Process one day: get per-ecozone cloud fraction                               #
# --------------------------------------------------------------------------- #
def process_day(year: int, month: int, date_str: str,
                rng: np.random.Generator) -> list[dict]:
    """Return list of {ecozone, is_cloud, is_land} dicts for sampled pixels."""
    result = _read_firemask(date_str)
    if result is None:
        return []

    arr, gt = result

    # Processed pixels only (fm > 0, not water=3, not unknown=6)
    land_mask = (arr > 0) & (arr != 3) & (arr != 6)
    lons, lats = _pixel_centroids_flat(arr, gt, land_mask)
    fm_vals = arr[land_mask]

    if len(lons) == 0:
        return []

    # Subsample
    n = len(lons)
    if n > MAX_PIXELS:
        idx = rng.choice(n, size=MAX_PIXELS, replace=False)
        lons = lons[idx]
        lats = lats[idx]
        fm_vals = fm_vals[idx]

    # Build GDF, mask to research area
    gdf = gpd.GeoDataFrame(
        {'fm': fm_vals},
        geometry=gpd.points_from_xy(lons, lats),
        crs="EPSG:4326"
    )
    ra_union = _load_research_union()
    gdf = gdf[gdf.geometry.within(ra_union)].copy()

    if len(gdf) == 0:
        return []

    # Assign ecozone via sjoin
    eco_gdf = _load_ecozone_gdf()
    joined  = gpd.sjoin(gdf, eco_gdf[['ZONE_NAME', 'geometry']],
                        how='left', predicate='within')
    ecozones = joined['ZONE_NAME'].fillna('OUTSIDE').values
    fm_arr = gdf['fm'].values

    rows = []
    for fm, eco in zip(fm_arr, ecozones):
        if eco == 'OUTSIDE':
            continue
        rows.append({
            'year':     year,
            'month':    month,
            'season':   SEASON_MAP[month],
            'ecozone':  eco,
            'is_cloud': int(fm == FIREMASK_CLOUD),
            'is_land':  1,
        })
    return rows


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #
def run_analysis() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    all_pixel_rows = []

    for year in YEARS:
        print(f"\nYear {year}...")
        for month in MONTHS:
            month_str = f"{month:02d}"
            # Find all daily files for this year-month
            files_in_month = sorted(glob_module.glob(
                os.path.join(MODIS_DIR, f"{year}_{month_str}_*.tif")))
            if not files_in_month:
                continue

            # Sample DAYS_PER_MONTH days
            step = max(1, len(files_in_month) // DAYS_PER_MONTH)
            sample_files = files_in_month[::step][:DAYS_PER_MONTH]

            for fpath in sample_files:
                # Reconstruct date_str from filename: yyyy_mm_dd.tif
                fname = os.path.basename(fpath)  # e.g. 2018_07_15.tif
                parts = fname.replace('.tif', '').split('_')
                date_str = f"{parts[0]}-{parts[1]}-{parts[2]}"

                rows = process_day(year, month, date_str, rng)
                all_pixel_rows.extend(rows)
                n = len(rows)
                n_cloud = sum(r['is_cloud'] for r in rows)
                if n > 0:
                    print(f"  {date_str}: {n:,} pixels, {n_cloud:,} cloud "
                          f"({n_cloud/n*100:.1f}%)")
                else:
                    print(f"  {date_str}: 0 pixels")

    if not all_pixel_rows:
        raise RuntimeError("No pixel data collected — check MODIS paths.")

    df = pd.DataFrame(all_pixel_rows)
    print(f"\nTotal pixel-observations: {len(df):,}")

    # Aggregate: per (ecozone, season) — mean cloud fraction
    agg = (df.groupby(['ecozone', 'season'])
             .agg(n_obs=('is_land', 'sum'),
                  n_cloud=('is_cloud', 'sum'))
             .reset_index())
    agg['cloud_fraction'] = agg['n_cloud'] / agg['n_obs']

    # Also compute std across year-month samples
    per_ym = (df.groupby(['ecozone', 'season', 'year', 'month'])
                .agg(n_obs=('is_land', 'sum'),
                     n_cloud=('is_cloud', 'sum'))
                .reset_index())
    per_ym['cf'] = per_ym['n_cloud'] / per_ym['n_obs']
    std_df = (per_ym.groupby(['ecozone', 'season'])['cf']
                    .std()
                    .reset_index()
                    .rename(columns={'cf': 'cloud_fraction_std'}))

    result = agg.merge(std_df, on=['ecozone', 'season'], how='left')
    result['cloud_fraction_std'] = result['cloud_fraction_std'].fillna(0.0)
    result = result.sort_values(['ecozone', 'season'])
    return result


def save_outputs(df: pd.DataFrame) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # CSV — keep filename for downstream compatibility
    csv_path = os.path.join(OUTPUT_DIR,
                            "cloud_contamination_by_ecoregion_season.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Heatmap PNG
    png_path = os.path.join(OUTPUT_DIR,
                            "cloud_contamination_by_ecoregion_season.png")
    seasons_order = ['DJF', 'MAM', 'JJA', 'SON']

    # Pivot: rows = ecozone, cols = season
    pivot = df.pivot_table(index='ecozone', columns='season',
                           values='cloud_fraction', aggfunc='mean')
    # Reorder columns
    pivot = pivot.reindex(columns=[s for s in seasons_order
                                   if s in pivot.columns])

    # All ecozones (max 25 — manageable)
    fig, ax = plt.subplots(figsize=(8, max(6, len(pivot) * 0.4 + 2)))
    im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd',
                   vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Cloud fraction')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([e[:40] for e in pivot.index], fontsize=8)
    ax.set_title('MODIS Cloud Contamination by Ecozone x Season\n'
                 '(fraction of observations with firemask==4; research area)',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path}")

    # Summary stats
    print("\n=== Cloud Contamination Summary ===")
    season_means = df.groupby('season')['cloud_fraction'].mean()
    for s in seasons_order:
        if s in season_means:
            print(f"  {s}: mean cloud fraction = {season_means[s]:.3f}")
    top5 = df.groupby('ecozone')['cloud_fraction'].mean().nlargest(5)
    print("\n  Top 5 most cloud-affected ecozones:")
    for eco, cf in top5.items():
        print(f"    {eco[:45]}: {cf:.3f}")


if __name__ == "__main__":
    df = run_analysis()
    save_outputs(df)
