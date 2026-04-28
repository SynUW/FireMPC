"""Buffer radius sensitivity analysis: 30 / 60 / 100 km exclusion zones.

For each candidate radius, computes:
  (a) exclusion_fraction — fraction of non-fire, non-cloud land pixels excluded
      by the spatial buffer around fire pixel centroids
  (b) available_fraction — complement of exclusion_fraction (proxy for negative
      candidate density after applying the buffer)

Analysis uses pixel centroids at the Canada grid level (spatial statistics,
NOT temporal pixel-days). A representative fire year (2018) from peak fire
season (Jul-Aug) is used. All 4 geographic MODIS tiles are combined to achieve
full Canada coverage.

CRITICAL: All buffer computations use EPSG:3978 (Canada Atlas Lambert projected
CRS). Do NOT compute buffers in EPSG:4326 (degrees) -- errors are large at high
latitudes (northern Canada ~65-70N).

Reference: Fu et al. (2601.01677) uses 60 km as default; this analysis confirms
or refines that choice per SAMP-01.

Filename format: MCD14A1_YYYY_MM_DD_tileNN.tif  (NN = 00,01,02,03)
"""
import csv
import os
import glob as glob_module
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.ops import unary_union
from osgeo import gdal

from audit.modis_utils import (get_positive_mask, FIREMASK_NONFIRE_LAND)
from catalogue.ecozone_join import assign_ecozone_labels

MODIS_DIR   = ("/s3/zhengsen/wildfire_dataset/self_built_materials/"
               "bc_raw_data_new_fast/drivers/processed/MCD14A1_combined")
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
EPSG_CANADA = "EPSG:3978"   # Canada Atlas Lambert -- standard projected CRS for Canada

CANDIDATE_RADII     = [30, 60, 100]  # km
REPRESENTATIVE_YEAR = 2018           # high-fire year for diverse ecozone coverage
# Sample at most this many non-fire land pixels for the spatial analysis
# (full mosaic is ~46M pixels; we reduce to keep computation tractable)
MAX_LAND_PIXELS_SAMPLE = 100_000
RANDOM_SEED = 42


def _get_pixel_centroids_from_tile(tile_path: str) -> tuple:
    """Extract pixel centroid arrays and firemask from one tile.

    Returns:
        (x_land, y_land, fm_land) where x/y are in the tile's native CRS
        and fm_land are firemask values for processed (fm > 0) pixels.
    """
    ds = gdal.Open(tile_path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open: {tile_path}")
    gt = ds.GetGeoTransform()   # (xmin, xres, 0, ymax, 0, yres)
    W, H = ds.RasterXSize, ds.RasterYSize
    band = ds.GetRasterBand(1)
    # Use ReadRaster() + frombuffer to avoid gdal_array (NumPy ABI conflict)
    raw = band.ReadRaster(0, 0, W, H, buf_type=gdal.GDT_Byte)
    firemask = np.frombuffer(raw, dtype=np.uint8).reshape(H, W)
    ds = None

    cols = np.arange(W, dtype=np.float32)
    rows = np.arange(H, dtype=np.float32)
    xcoords = gt[0] + (cols + 0.5) * gt[1]
    ycoords = gt[3] + (rows + 0.5) * gt[5]
    xx, yy = np.meshgrid(xcoords, ycoords)   # shape: (H, W)

    # Keep only processed pixels (firemask > 0 means data was processed)
    land_mask = firemask > 0
    x_land = xx[land_mask]
    y_land = yy[land_mask]
    fm_land = firemask[land_mask]
    return x_land, y_land, fm_land


def get_modis_pixel_centroids_gdf(representative_day_tiles: list,
                                   max_sample: int = MAX_LAND_PIXELS_SAMPLE,
                                   rng=None) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of Canada-wide pixel centroids from all 4 tiles.

    Combines all 4 geographic MODIS tiles for a single representative day to
    cover the full Canada mosaic. Then randomly sub-samples to max_sample
    land pixels for memory efficiency.

    Args:
        representative_day_tiles: paths to tile files for one day
                                  (tile00, tile01, tile02, tile03).
        max_sample: maximum pixels to keep (default 100K).
        rng: numpy random Generator for reproducible sampling.

    Returns:
        GeoDataFrame in EPSG:4326 with columns ['firemask', 'geometry'].
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    all_x, all_y, all_fm = [], [], []
    for tile_path in representative_day_tiles:
        try:
            x, y, fm = _get_pixel_centroids_from_tile(tile_path)
            all_x.append(x)
            all_y.append(y)
            all_fm.append(fm)
        except Exception as e:
            print(f"  Warning: could not read {tile_path}: {e}")

    if not all_x:
        raise RuntimeError("No valid tiles read for pixel centroid construction.")

    x_all = np.concatenate(all_x)
    y_all = np.concatenate(all_y)
    fm_all = np.concatenate(all_fm)

    # Sub-sample for memory efficiency
    n_total = len(x_all)
    if n_total > max_sample:
        idx = rng.choice(n_total, size=max_sample, replace=False)
        x_all = x_all[idx]
        y_all = y_all[idx]
        fm_all = fm_all[idx]

    gdf = gpd.GeoDataFrame(
        {'firemask': fm_all},
        geometry=gpd.points_from_xy(x_all, y_all),
        crs="EPSG:4326"
    )
    return gdf


def compute_buffer_exclusion_stats_for_tiles(fire_pixels_gdf: gpd.GeoDataFrame,
                                              all_pixels_gdf: gpd.GeoDataFrame,
                                              radius_km: float) -> tuple:
    """Compute buffer exclusion fraction for given radius (EPSG:3978 required).

    Args:
        fire_pixels_gdf: GeoDataFrame of fire pixel centroids in EPSG:3978.
        all_pixels_gdf:  GeoDataFrame of all land pixel centroids in EPSG:3978.
        radius_km:       buffer radius in km.

    Returns:
        (exclusion_fraction, excluded_boolean_array)
        exclusion_fraction: fraction of non-fire land pixels within buffer zone.
    """
    assert fire_pixels_gdf.crs.to_epsg() == 3978, "fire_pixels must be in EPSG:3978"
    assert all_pixels_gdf.crs.to_epsg()  == 3978, "all_pixels must be in EPSG:3978"

    radius_m = radius_km * 1000.0
    # Buffer all fire points and dissolve -- unary_union is O(n log n)
    fire_buffer    = fire_pixels_gdf.geometry.buffer(radius_m)
    exclusion_zone = unary_union(fire_buffer)

    # Non-fire land pixels only (firemask == 5)
    nonfire_mask   = all_pixels_gdf['firemask'] == FIREMASK_NONFIRE_LAND
    nonfire_pixels = all_pixels_gdf[nonfire_mask]

    if len(nonfire_pixels) == 0:
        return 0.0, np.array([], dtype=bool)

    excluded = nonfire_pixels.geometry.within(exclusion_zone)
    return float(excluded.mean()), excluded.values


def compute_buffer_exclusion_stats_synthetic(radius_km: float) -> float:
    """Synthetic version for unit tests: single fire point at 55N, -100W.

    Creates a small grid of test pixels at varying distances from a synthetic
    fire point and computes the exclusion fraction. Larger radius -> more
    pixels fall within the buffer -> higher exclusion fraction. This is
    deterministic: same grid, same fire point, only radius varies.

    All computations use EPSG:3978 (Canada Atlas Lambert).

    Args:
        radius_km: buffer radius in km (30, 60, or 100).

    Returns:
        float in [0.0, 1.0]: fraction of non-fire pixels within the buffer.
    """
    # Synthetic fire point at approximate centre of Canada
    fire_lon, fire_lat = -100.0, 55.0

    # Create a regular grid spanning ~3 degrees around fire point.
    # At 55N: 1 degree latitude ~= 111 km, 1 degree longitude ~= 64 km.
    # Grid of 11x11 = 121 points covering roughly +-165 km lat, +-95 km lon.
    # This gives meaningful separation between 30/60/100 km buffers.
    n = 11
    lons = np.linspace(fire_lon - 1.5, fire_lon + 1.5, n)
    lats = np.linspace(fire_lat - 1.5, fire_lat + 1.5, n)
    xx, yy = np.meshgrid(lons, lats)
    all_lons = xx.ravel()
    all_lats = yy.ravel()

    # Build GeoDataFrames in EPSG:4326, reproject to EPSG:3978 for buffer ops
    fire_gdf = gpd.GeoDataFrame(
        {'firemask': [9]},   # high-confidence fire
        geometry=[Point(fire_lon, fire_lat)],
        crs="EPSG:4326"
    ).to_crs(EPSG_CANADA)

    all_gdf = gpd.GeoDataFrame(
        {'firemask': [5] * len(all_lons)},   # all non-fire land pixels
        geometry=gpd.points_from_xy(all_lons, all_lats),
        crs="EPSG:4326"
    ).to_crs(EPSG_CANADA)

    frac, _ = compute_buffer_exclusion_stats_for_tiles(fire_gdf, all_gdf, radius_km)
    return frac


def run_buffer_sensitivity_analysis(representative_year: int = REPRESENTATIVE_YEAR,
                                     radii: list = None,
                                     n_fire_days: int = 7) -> list:
    """Run buffer sensitivity analysis for 30, 60, 100 km across ecozones.

    Uses pixel centroids from all 4 geographic tiles of a single representative
    day (for full Canada coverage), sub-sampled to MAX_LAND_PIXELS_SAMPLE.
    Fire pixel locations are aggregated from the peak fire season (Jul-Aug)
    using n_fire_days sampled evenly across all 4 tiles per day.

    Glob pattern for tiles: MCD14A1_{year}_07_DD_tileNN.tif
                             MCD14A1_{year}_08_DD_tileNN.tif

    Args:
        representative_year: year for analysis (default 2018).
        radii: radii list in km (default [30, 60, 100]).
        n_fire_days: number of peak-season days to sample for fire points.

    Returns:
        list of dicts, one per (radius, ecozone) combination.
    """
    if radii is None:
        radii = CANDIDATE_RADII

    # ------------------------------------------------------------------ #
    # Step 1: Build Canada-wide pixel centroid grid from one day's 4 tiles #
    # ------------------------------------------------------------------ #
    day1_pattern = os.path.join(MODIS_DIR,
                                f"MCD14A1_{representative_year}_07_01_tile*.tif")
    day1_tiles = sorted(glob_module.glob(day1_pattern))

    if len(day1_tiles) == 0:
        # Fallback: find any set of 4 tiles for the representative year
        any_tiles = sorted(glob_module.glob(
            os.path.join(MODIS_DIR, f"MCD14A1_{representative_year}_*.tif")))
        if not any_tiles:
            raise FileNotFoundError(
                f"No MCD14A1 tiles found for year {representative_year} in {MODIS_DIR}")
        # Pick first date's tiles
        first_stem = "_".join(os.path.basename(any_tiles[0]).split("_")[:4])
        day1_tiles = sorted(glob_module.glob(
            os.path.join(MODIS_DIR, f"{first_stem}_tile*.tif")))

    print(f"Building pixel centroid grid from {len(day1_tiles)} tiles "
          f"(day 1 of {representative_year}-07)")

    all_pixels_gdf  = get_modis_pixel_centroids_gdf(day1_tiles)
    all_pixels_3978 = all_pixels_gdf.to_crs(EPSG_CANADA)
    print(f"  Total sampled land pixels: {len(all_pixels_3978):,}")

    # ------------------------------------------------------------------ #
    # Step 2: Assign ecozone labels to pixel grid (use WGS84 lons/lats)   #
    # ------------------------------------------------------------------ #
    pix_4326 = all_pixels_gdf   # already in EPSG:4326
    lons_arr = np.array([p.x for p in pix_4326.geometry])
    lats_arr = np.array([p.y for p in pix_4326.geometry])
    print("  Assigning ecozone labels ...")
    ecozone_labels = assign_ecozone_labels(lons_arr, lats_arr)
    all_pixels_3978 = all_pixels_3978.copy()
    all_pixels_3978['ecozone'] = ecozone_labels
    print(f"  Ecozones found: {sorted(set(ecozone_labels) - {'OUTSIDE_CANADA'})}")

    # ------------------------------------------------------------------ #
    # Step 3: Collect fire pixel locations from Jul-Aug peak season        #
    # ------------------------------------------------------------------ #
    # All 4 tiles per day -- fires can be in any geographic region
    all_fire_tiles_jul = sorted(glob_module.glob(
        os.path.join(MODIS_DIR, f"MCD14A1_{representative_year}_07_*.tif")))
    all_fire_tiles_aug = sorted(glob_module.glob(
        os.path.join(MODIS_DIR, f"MCD14A1_{representative_year}_08_*.tif")))
    all_fire_tiles = all_fire_tiles_jul + all_fire_tiles_aug

    # Sample n_fire_days * 4 tiles evenly across the season
    n_target = n_fire_days * 4
    step = max(1, len(all_fire_tiles) // n_target)
    sampled_fire_tiles = all_fire_tiles[::step][:n_target]
    print(f"\nCollecting fire pixels from {len(sampled_fire_tiles)} tiles "
          f"(~{n_fire_days} sampled days out of {representative_year} Jul-Aug)")

    all_fire_x, all_fire_y = [], []
    for tile_path in sampled_fire_tiles:
        try:
            ds = gdal.Open(tile_path, gdal.GA_ReadOnly)
            if ds is None:
                continue
            gt = ds.GetGeoTransform()
            W_t, H_t = ds.RasterXSize, ds.RasterYSize
            band = ds.GetRasterBand(1)
            raw = band.ReadRaster(0, 0, W_t, H_t, buf_type=gdal.GDT_Byte)
            firemask = np.frombuffer(raw, dtype=np.uint8).reshape(H_t, W_t)
            ds = None
        except Exception as e:
            print(f"  Warning: cannot read {tile_path}: {e}")
            continue
        fire_mask = get_positive_mask(firemask)
        if fire_mask.sum() == 0:
            continue
        rows_idx, cols_idx = np.where(fire_mask)
        xf = gt[0] + (cols_idx + 0.5) * gt[1]
        yf = gt[3] + (rows_idx + 0.5) * gt[5]
        all_fire_x.extend(xf.tolist())
        all_fire_y.extend(yf.tolist())

    if len(all_fire_x) == 0:
        raise RuntimeError(
            f"No high-confidence fire pixels found in sampled tiles. "
            f"Check MODIS data for {representative_year} Jul/Aug.")

    fire_gdf = gpd.GeoDataFrame(
        {'firemask': [9] * len(all_fire_x)},
        geometry=gpd.points_from_xy(all_fire_x, all_fire_y),
        crs="EPSG:4326"
    ).to_crs(EPSG_CANADA)
    print(f"  Fire pixel centroids collected: {len(fire_gdf):,}")

    # ------------------------------------------------------------------ #
    # Step 4: Compute exclusion statistics per radius x ecozone            #
    # Key optimization: unary_union computed ONCE per radius, then         #
    # per-pixel within-check is done on the full pixel set simultaneously. #
    # ------------------------------------------------------------------ #
    rows = []
    ecozones = sorted(
        e for e in all_pixels_3978['ecozone'].unique()
        if e != 'OUTSIDE_CANADA'
    )
    print(f"\nRunning buffer analysis for {len(ecozones)} ecozones x "
          f"{len(radii)} radii ...")

    # Non-fire land pixels (firemask==5) in EPSG:3978
    nonfire_mask_global  = all_pixels_3978['firemask'] == FIREMASK_NONFIRE_LAND
    nonfire_pixels_global = all_pixels_3978[nonfire_mask_global]

    for radius_km in sorted(radii):
        radius_m = radius_km * 1000.0
        print(f"\n  Radius: {radius_km} km — building exclusion zone ...")
        # Build exclusion zone ONCE for this radius
        fire_buffer    = fire_gdf.geometry.buffer(radius_m)
        exclusion_zone = unary_union(fire_buffer)
        print(f"    Exclusion zone computed. Checking {len(nonfire_pixels_global):,} non-fire pixels ...")

        # Vectorized within-check on ALL non-fire pixels at once
        excluded_global = nonfire_pixels_global.geometry.within(exclusion_zone).values
        nonfire_pixels_global_copy = nonfire_pixels_global.copy()
        nonfire_pixels_global_copy['excluded'] = excluded_global

        frac_overall = float(excluded_global.mean()) if len(excluded_global) > 0 else 0.0
        print(f"    Overall exclusion fraction: {frac_overall:.3f}")

        # Per-ecozone stats use the pre-computed excluded column
        ecozone_vals = all_pixels_3978.loc[nonfire_mask_global, 'ecozone'].values
        nonfire_pixels_global_copy['ecozone'] = ecozone_vals

        for ecozone in ecozones:
            eco_excl_mask = nonfire_pixels_global_copy['ecozone'] == ecozone
            eco_subset    = nonfire_pixels_global_copy[eco_excl_mask]
            if len(eco_subset) < 10:
                continue
            frac_eco       = float(eco_subset['excluded'].mean())
            available_frac = 1.0 - frac_eco
            n_available    = int(available_frac * len(eco_subset))

            rows.append({
                'radius_km':              radius_km,
                'ecozone':                ecozone,
                'total_pixels':           len(eco_subset),
                'exclusion_fraction':     round(frac_eco, 4),
                'available_pixels':       n_available,
                'available_fraction':     round(available_frac, 4),
                'negative_density_proxy': round(available_frac, 4),
                'recommendation':         'adequate' if available_frac >= 0.1
                                          else 'insufficient',
            })
            print(f"    {ecozone[:30]:<30}: excl={frac_eco:.3f}  "
                  f"avail={available_frac:.3f}  n={len(eco_subset)}")

    # ------------------------------------------------------------------ #
    # Step 5: Choose recommended radius                                    #
    # ------------------------------------------------------------------ #
    # Choose smallest radius where >=90% of "major" ecozones (n>=50 pixels)
    # retain adequate negative density. Very small ecoregions in active fire
    # zones may be entirely excluded even at 30km -- these are edge cases.
    # Using all(adequate) would always fail in peak-fire years.
    recommended_radius = min(radii)   # default: always prefer smaller buffer
    for radius_km in sorted(radii):
        radius_rows = [r for r in rows if r['radius_km'] == radius_km]
        major_rows  = [r for r in radius_rows if r['total_pixels'] >= 50]
        if not major_rows:
            continue
        adequate_frac = sum(1 for r in major_rows
                            if r['recommendation'] == 'adequate') / len(major_rows)
        if adequate_frac >= 0.90:
            recommended_radius = radius_km
            break

    for r in rows:
        r['recommended_radius'] = recommended_radius

    # ------------------------------------------------------------------ #
    # Step 6: Write CSV and figure                                         #
    # ------------------------------------------------------------------ #
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "buffer_sensitivity_table.csv")
    fieldnames = [
        'radius_km', 'ecozone', 'total_pixels', 'exclusion_fraction',
        'available_pixels', 'available_fraction', 'negative_density_proxy',
        'recommendation', 'recommended_radius',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nBuffer sensitivity table written: {csv_path}")
    print(f"Recommended radius: {recommended_radius} km")

    _plot_sensitivity(rows, sorted(radii),
                      os.path.join(OUTPUT_DIR, "buffer_exclusion_maps.png"))
    return rows


def _plot_sensitivity(rows: list, radii: list, output_path: str) -> None:
    """Plot exclusion fraction by ecozone and radius (grouped bar chart)."""
    ecozones = sorted(set(r['ecozone'] for r in rows))
    if not ecozones:
        print("No ecozone data to plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(ecozones))
    bar_width = 0.25
    colors = ['steelblue', 'darkorange', 'forestgreen']

    for i, radius_km in enumerate(sorted(radii)):
        fracs = []
        for ecozone in ecozones:
            matching = [r for r in rows
                        if r['radius_km'] == radius_km and r['ecozone'] == ecozone]
            fracs.append(matching[0]['exclusion_fraction'] if matching else 0.0)
        ax.bar(x + i * bar_width, fracs, bar_width,
               label=f'{radius_km} km', color=colors[i % len(colors)], alpha=0.8)

    ax.set_xlabel('Ecozone')
    ax.set_ylabel('Exclusion Fraction')
    ax.set_title(
        'Buffer Exclusion Fraction by Radius and Ecozone\n'
        '(EPSG:3978 Canada Atlas Lambert, representative year 2018)')
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([e[:22] for e in ecozones], rotation=45, ha='right',
                       fontsize=8)
    ax.legend(title='Buffer radius')
    ax.axhline(0.9, color='red', linestyle='--', alpha=0.5,
               label='90% exclusion threshold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Buffer exclusion plot saved: {output_path}")


if __name__ == "__main__":
    rows = run_buffer_sensitivity_analysis()
    if rows:
        rec = rows[0]['recommended_radius']
        print(f"\nFinal recommended buffer radius: {rec} km")
