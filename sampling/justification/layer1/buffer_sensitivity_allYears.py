"""Phase 2 Layer 1 — Buffer sensitivity analysis across ALL years 2000-2025.

Uses the 5km unified merged MODIS data (single file per day, no tiles).
Uses ecozones (ZONE_NAME, 25 polygons) instead of the old REGION_NAM column.
Masks all candidates to research_area_shpfile_merge.shp before analysis.

For each year 2000-2025:
  - Collects high-confidence fire pixel centroids from Jul-Aug peak season
    (samples up to MAX_FIRE_DAYS days evenly across season)
  - Computes spatial exclusion fraction at multiple radii (EPSG:3978) per ecozone
  - Records available negative fraction

Output: buffer_sensitivity_allYears.csv + buffer_sensitivity_allYears.png
Both saved to the same directory as this script.

CRITICAL: All buffer computations use EPSG:3978 (Canada Atlas Lambert).
CRITICAL: GDAL reads use ReadRaster()+frombuffer — NOT ReadAsArray (NumPy ABI issue).
CRITICAL: Files are pre-merged per day (single TIF, no tiles). Format: yyyy_mm_dd.tif
"""
import csv
import os
import glob as glob_module
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.ops import unary_union
from osgeo import gdal

# --------------------------------------------------------------------------- #
# Paths                                                                         #
# --------------------------------------------------------------------------- #
MODIS_DIR = (
    "/s3/user/wildfire_dataset/self_built_materials/"
    "bc_raw_data_new_fast/drivers/processed/"
    "spatial_filled_5xdownsampled_merged/MCD14A1_combined_5xMaximumDown_merged"
)
ECOZONE_SHP = (
    "/s3/user/wildfire_dataset/self_built_materials/"
    "bc_raw_data_new_fast/drivers/ecoregions/ecozones.shp"
)
RESEARCH_AREA_SHP = (
    "/s3/user/wildfire_dataset/self_built_materials/"
    "bc_raw_data_new_fast/drivers/ecoregions/research_area_shpfile_merge.shp"
)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
# Constants                                                                     #
# --------------------------------------------------------------------------- #
YEARS = list(range(2000, 2026))          # 2000-2025 inclusive
RADII_KM = [15, 20, 25, 30, 40, 50]     # Multiple radii for sensitivity analysis
EPSG_CANADA = "EPSG:3978"
MAX_LAND_PIXELS = 80_000                 # land pixel sample per year (constant grid)
MAX_FIRE_PIXELS = 6_000                  # cap fire pixels per year (for unary_union speed)
MAX_FIRE_DAYS = 7                        # peak-season days sampled per year
FIREMASK_HIGH_CONF = 9
FIREMASK_NONFIRE_LAND = 5
RANDOM_SEED = 42


# --------------------------------------------------------------------------- #
# Ecozone + research area loader (cached)                                       #
# --------------------------------------------------------------------------- #
_eco_gdf_cache = None
_research_area_cache = None


def _load_ecozone_gdf() -> gpd.GeoDataFrame:
    global _eco_gdf_cache
    if _eco_gdf_cache is not None:
        return _eco_gdf_cache
    gdf = gpd.read_file(ECOZONE_SHP)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4617")
    gdf = gdf.to_crs("EPSG:4326")
    _eco_gdf_cache = gdf
    print(f"  Ecozone shapefile loaded: {len(gdf)} polygons, "
          f"columns: {list(gdf.columns)}")
    return gdf


def _load_research_area() -> gpd.GeoDataFrame:
    global _research_area_cache
    if _research_area_cache is not None:
        return _research_area_cache
    gdf = gpd.read_file(RESEARCH_AREA_SHP)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4617")
    gdf = gdf.to_crs("EPSG:4326")
    _research_area_cache = gdf
    print(f"  Research area loaded: {len(gdf)} polygon(s)")
    return gdf


# --------------------------------------------------------------------------- #
# GDAL merged-file reader (safe: ReadRaster + frombuffer, no gdal_array)       #
# --------------------------------------------------------------------------- #
def _read_firemask(date_str: str) -> tuple[np.ndarray, tuple] | None:
    """Return (firemask_array, geotransform) for one merged daily file.

    date_str format: 'YYYY-MM-DD'
    File naming convention: yyyy_mm_dd.tif
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


def _pixel_centroids(arr: np.ndarray, gt: tuple, mask: np.ndarray) -> tuple:
    """Return (lon_array, lat_array) for pixels where mask is True."""
    H, W = arr.shape
    cols = np.arange(W, dtype=np.float64)
    rows = np.arange(H, dtype=np.float64)
    xcoords = gt[0] + (cols + 0.5) * gt[1]   # lon
    ycoords = gt[3] + (rows + 0.5) * gt[5]   # lat (gt[5] < 0)
    xx, yy = np.meshgrid(xcoords, ycoords)
    return xx[mask], yy[mask]


# --------------------------------------------------------------------------- #
# Build stable land-pixel sample grid (computed once using 2018-07-15)        #
# --------------------------------------------------------------------------- #
def build_land_pixel_grid(rng: np.random.Generator) -> gpd.GeoDataFrame:
    """Sample MAX_LAND_PIXELS non-fire land pixels from 2018-07-15.

    Masked to research_area_shpfile_merge.shp.
    Ecozone label assigned via sjoin.
    Returns GeoDataFrame in EPSG:4326 with columns ['firemask', 'ecozone', 'geometry'].
    """
    print("Building land pixel grid from 2018-07-15 (stable across years)...")
    result = _read_firemask("2018-07-15")
    if result is None:
        # Fallback: try any 2018 July file
        candidates = sorted(glob_module.glob(
            os.path.join(MODIS_DIR, "2018_07_*.tif")))
        if not candidates:
            raise FileNotFoundError("No 2018 July merged files found.")
        ds2 = gdal.Open(candidates[0], gdal.GA_ReadOnly)
        gt = ds2.GetGeoTransform()
        W, H = ds2.RasterXSize, ds2.RasterYSize
        raw = ds2.GetRasterBand(1).ReadRaster(0, 0, W, H, buf_type=gdal.GDT_Byte)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(H, W)
        ds2 = None
    else:
        arr, gt = result

    # Keep processed pixels (fm > 0), excluding water (3) and unknowns (6)
    land_mask = (arr > 0) & (arr != 3) & (arr != 6)
    lons, lats = _pixel_centroids(arr, gt, land_mask)
    fm_vals = arr[land_mask]

    # Subsample
    n = len(lons)
    if n > MAX_LAND_PIXELS:
        idx = rng.choice(n, size=MAX_LAND_PIXELS, replace=False)
        lons = lons[idx]
        lats = lats[idx]
        fm_vals = fm_vals[idx]

    gdf = gpd.GeoDataFrame(
        {'firemask': fm_vals},
        geometry=gpd.points_from_xy(lons, lats),
        crs="EPSG:4326"
    )

    # Mask to research area
    print(f"  {len(gdf):,} raw land pixels. Masking to research area...")
    ra_gdf = _load_research_area()
    ra_union = ra_gdf.geometry.unary_union
    gdf = gdf[gdf.geometry.within(ra_union)].copy()
    print(f"  {len(gdf):,} pixels inside research area.")

    # Assign ecozone labels via sjoin
    print(f"  Assigning ecozone labels...")
    eco_gdf = _load_ecozone_gdf()
    joined = gpd.sjoin(gdf, eco_gdf[['ZONE_NAME', 'geometry']], how='left',
                       predicate='within')
    gdf['ecozone'] = joined['ZONE_NAME'].fillna('OUTSIDE').values
    n_inside = (gdf['ecozone'] != 'OUTSIDE').sum()
    print(f"  {n_inside:,} pixels inside ecozone polygons; "
          f"{len(gdf)-n_inside:,} outside.")
    return gdf


# --------------------------------------------------------------------------- #
# Fire pixel collector for one year's peak season                              #
# --------------------------------------------------------------------------- #
def collect_fire_pixels_for_year(year: int, rng: np.random.Generator
                                  ) -> gpd.GeoDataFrame:
    """Return GeoDataFrame (EPSG:3978) of high-conf fire pixels from Jul-Aug.

    Samples MAX_FIRE_DAYS days evenly across Jul-Aug, then subsamples to
    MAX_FIRE_PIXELS for unary_union speed.
    """
    # Find all Jul-Aug daily files for this year
    jul_files = sorted(glob_module.glob(
        os.path.join(MODIS_DIR, f"{year}_07_*.tif")))
    aug_files = sorted(glob_module.glob(
        os.path.join(MODIS_DIR, f"{year}_08_*.tif")))
    all_files = jul_files + aug_files

    if not all_files:
        return gpd.GeoDataFrame(geometry=[], crs=EPSG_CANADA)

    # Sample MAX_FIRE_DAYS days evenly across season
    step = max(1, len(all_files) // MAX_FIRE_DAYS)
    sampled_files = all_files[::step][:MAX_FIRE_DAYS]

    all_lons, all_lats = [], []
    for fpath in sampled_files:
        ds = gdal.Open(fpath, gdal.GA_ReadOnly)
        if ds is None:
            continue
        gt = ds.GetGeoTransform()
        W, H = ds.RasterXSize, ds.RasterYSize
        raw = ds.GetRasterBand(1).ReadRaster(0, 0, W, H, buf_type=gdal.GDT_Byte)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(H, W)
        ds = None
        fire_mask = arr == FIREMASK_HIGH_CONF
        if fire_mask.sum() == 0:
            continue
        lons, lats = _pixel_centroids(arr, gt, fire_mask)
        all_lons.extend(lons.tolist())
        all_lats.extend(lats.tolist())

    if not all_lons:
        return gpd.GeoDataFrame(geometry=[], crs=EPSG_CANADA)

    lon_arr = np.array(all_lons, dtype=np.float64)
    lat_arr = np.array(all_lats, dtype=np.float64)

    # Subsample fire pixels for unary_union speed
    n = len(lon_arr)
    if n > MAX_FIRE_PIXELS:
        idx = rng.choice(n, size=MAX_FIRE_PIXELS, replace=False)
        lon_arr = lon_arr[idx]
        lat_arr = lat_arr[idx]

    n = len(lon_arr)
    gdf = gpd.GeoDataFrame(
        {'year': np.full(n, year, dtype=np.int32)},
        geometry=gpd.points_from_xy(lon_arr, lat_arr),
        crs="EPSG:4326"
    ).to_crs(EPSG_CANADA)
    return gdf


# --------------------------------------------------------------------------- #
# Main analysis                                                                 #
# --------------------------------------------------------------------------- #
def run_analysis() -> list[dict]:
    rng = np.random.default_rng(RANDOM_SEED)

    # Build stable land grid ONCE (masked to research area)
    land_gdf_4326 = build_land_pixel_grid(rng)
    land_gdf_3978 = land_gdf_4326.to_crs(EPSG_CANADA)
    nonfire_mask = land_gdf_3978['firemask'] == FIREMASK_NONFIRE_LAND
    nonfire_gdf  = land_gdf_3978[nonfire_mask].copy()
    print(f"\nNon-fire land pixels for exclusion check: {len(nonfire_gdf):,}\n")

    rows = []
    for year in YEARS:
        print(f"Year {year} — collecting fire pixels...")
        fire_gdf = collect_fire_pixels_for_year(year, rng)
        n_fire = len(fire_gdf)
        print(f"  {n_fire:,} fire pixels collected.")

        if n_fire == 0:
            print(f"  Skipping year {year}: no fire pixels found.")
            continue

        for radius_km in RADII_KM:
            radius_m = radius_km * 1000.0
            print(f"  Radius {radius_km} km — building exclusion zone...")
            fire_buffer    = fire_gdf.geometry.buffer(radius_m)
            exclusion_zone = unary_union(fire_buffer)

            # Vectorized within-check
            excluded = nonfire_gdf.geometry.within(exclusion_zone).values
            nonfire_copy = nonfire_gdf.copy()
            nonfire_copy['excluded'] = excluded

            # Per-ecozone stats
            ecozones = [e for e in nonfire_copy['ecozone'].unique()
                        if e not in ('OUTSIDE', 'OUTSIDE_CANADA')]
            for eco in sorted(ecozones):
                subset = nonfire_copy[nonfire_copy['ecozone'] == eco]
                if len(subset) < 5:
                    continue
                excl_frac = float(subset['excluded'].mean())
                avail_frac = 1.0 - excl_frac
                rows.append({
                    'year':               year,
                    'radius_km':          radius_km,
                    'ecozone':            eco,
                    'n_pixels':           len(subset),
                    'exclusion_fraction': round(excl_frac, 4),
                    'available_fraction': round(avail_frac, 4),
                    'adequate':           'yes' if avail_frac >= 0.10 else 'no',
                })
            overall_excl = float(excluded.mean()) if len(excluded) > 0 else 0.0
            print(f"    Overall exclusion: {overall_excl:.3f}")

    return rows


def save_csv(rows: list[dict]) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "buffer_sensitivity_allYears.csv")
    fieldnames = ['year', 'radius_km', 'ecozone', 'n_pixels',
                  'exclusion_fraction', 'available_fraction', 'adequate']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {csv_path}  ({len(rows)} rows)")
    return csv_path


def save_figure(rows: list[dict]) -> str:
    """Line chart: mean exclusion fraction per radius across years, one line per radius.
    Inset bar chart shows per-ecozone available fraction at 30 km (2018).
    """
    import pandas as pd
    df = pd.DataFrame(rows)
    png_path = os.path.join(OUTPUT_DIR, "buffer_sensitivity_allYears.png")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Left: mean exclusion fraction by year x radius ---
    ax = axes[0]
    colors_map = {15: 'navy', 20: 'steelblue', 25: 'cadetblue',
                  30: 'darkorange', 40: 'crimson', 50: 'forestgreen'}
    for r in RADII_KM:
        subset = df[df['radius_km'] == r]
        yearly = subset.groupby('year')['exclusion_fraction'].mean()
        ax.plot(yearly.index, yearly.values, marker='o', markersize=4,
                label=f'{r} km', color=colors_map.get(r, 'gray'))
    ax.set_xlabel('Year')
    ax.set_ylabel('Mean Exclusion Fraction (across ecozones)')
    ax.set_title('Buffer Exclusion Fraction by Year and Radius\n'
                 '(Canada research area mean, peak fire season Jul-Aug)')
    ax.legend(title='Buffer radius')
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    # --- Right: per-ecozone available fraction at 30 km for representative year ---
    ax2 = axes[1]
    rep_year = 2018
    rep_data = df[(df['radius_km'] == 30) & (df['year'] == rep_year)]
    if not rep_data.empty:
        rep_data = rep_data.sort_values('available_fraction')
        # Show all ecozones (max 25)
        top = rep_data.nlargest(25, 'n_pixels')
        bars = ax2.barh(range(len(top)),
                        top['available_fraction'].values,
                        color=['steelblue' if v >= 0.1 else 'salmon'
                               for v in top['available_fraction'].values])
        ax2.set_yticks(range(len(top)))
        ax2.set_yticklabels([e[:35] for e in top['ecozone']], fontsize=8)
        ax2.axvline(0.1, color='red', linestyle='--', alpha=0.7,
                    label='10% adequacy threshold')
        ax2.set_xlabel('Available Fraction (1 - exclusion_fraction)')
        ax2.set_title(f'Available Fraction per Ecozone at 30 km ({rep_year})\n'
                      f'(red = inadequate < 10% available)')
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, f'No data for {rep_year}', ha='center', va='center',
                 transform=ax2.transAxes)

    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path}")
    return png_path


if __name__ == "__main__":
    rows = run_analysis()
    if rows:
        save_csv(rows)
        save_figure(rows)
        # Summary
        import pandas as pd
        df = pd.DataFrame(rows)
        print("\n=== Summary ===")
        for r in RADII_KM:
            sub = df[df['radius_km'] == r]
            pct_adequate = (sub['adequate'] == 'yes').mean() * 100
            mean_avail   = sub['available_fraction'].mean()
            print(f"  {r} km: {pct_adequate:.1f}% ecozone-years adequate, "
                  f"mean available fraction = {mean_avail:.3f}")
        print("\nRecommended 30 km details:")
        sub30 = df[df['radius_km'] == 30]
        yearly_excl = sub30.groupby('year')['exclusion_fraction'].mean()
        print(f"  Mean exclusion fraction across years: {yearly_excl.mean():.3f}")
        print(f"  Max exclusion fraction (worst year): {yearly_excl.max():.3f} "
              f"({yearly_excl.idxmax()})")
    else:
        print("No data produced — check MODIS merged file paths.")
