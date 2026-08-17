"""FWI Distribution EDA: compute fire vs. non-fire FWI statistics per ecozone
and calibrate hard-negative scoring weight vector w1-w6.

Calibration method (from CONTEXT.md OQ3 guidance):
  For each FWI component c in {FFMC, DMC, DC, ISI, BUI, FWI}:
    separation_c = (mean_c_at_fire - mean_c_at_nonfire) / (std_c_at_nonfire + 1e-6)
    weight_c = max(separation_c, 0)   # negative separation -> 0 weight
  Normalized: weights = weights / (sum(weights) + 1e-8)

IMPORTANT - tile format deviation from plan:
  ERA5 tiles: ERA5_YYYY_MM_DD_tileNN.tif (4 spatial tiles per day, NN=00,01,02,03)
  MODIS tiles: MCD14A1_YYYY_MM_DD_tileNN.tif (4 spatial tiles per day, NN=00,01,02,03)
  ERA5 tile layout: 2x2 spatial mosaic covering Canada
    tile00: SW (443x226 px, -141.1 to -96.8 lon, ~41.6 to ~64.2 lat)
    tile01: NW (443x214 px, -141.1 to -96.8 lon, ~62.4 to ~83.8 lat)
    tile02: SE (443x226 px, -96.9 to -52.6 lon, ~41.6 to ~64.2 lat)
    tile03: NE (443x214 px, -96.9 to -52.6 lon, ~62.4 to ~83.8 lat)
  MODIS tile layout: same 2x2 spatial layout but at higher resolution (4920x2361 for one tile)

Reference years: 2017 (BC fires), 2018 (BC/AB fires), 2021 (BC/YT record fires).
Outputs: weight_vector_w1w6.json, fwi_distributions_fire_vs_nonfire.png
"""
import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for server
import matplotlib.pyplot as plt
from datetime import date, timedelta

from osgeo import gdal

from fwi.fwi_module import (
    extract_fwi_inputs_from_bands,
    read_era5_tile,
    ffmc_next, dmc_next, dc_next,
    isi_calc, bui_calc, fwi_calc,
    FFMC_INIT, DMC_INIT, DC_INIT,
)
from audit.modis_utils import get_positive_mask, read_modis_tile
from catalogue.ecozone_join import assign_ecozone_labels


def _nn_zoom(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Nearest-neighbor resampling via NumPy index mapping (no scipy required).

    Replacement for scipy.ndimage.zoom with order=0 (nearest neighbor).
    Uses integer index mapping to avoid the scipy ABI issue with NumPy 2.x.
    """
    src_h, src_w = arr.shape
    tgt_h, tgt_w = target_shape
    row_idx = (np.arange(tgt_h) * src_h / tgt_h).astype(np.int32)
    col_idx = (np.arange(tgt_w) * src_w / tgt_w).astype(np.int32)
    return arr[np.ix_(row_idx, col_idx)]


def _numpy_kde(values: np.ndarray, x: np.ndarray,
               bandwidth: float = None) -> np.ndarray:
    """Simple Gaussian KDE implemented with NumPy (no scipy required).

    Uses Scott's rule for bandwidth if not provided.
    Vectorized over x evaluation points.
    """
    n = len(values)
    if bandwidth is None:
        # Scott's rule: h = n^(-1/5) * std
        bw = n ** (-0.2) * float(np.std(values))
    else:
        bw = bandwidth
    if bw < 1e-12:
        bw = 1e-6
    # KDE: (1/n) * sum_i K((x - x_i) / h) / h  where K is std normal
    diff = (x[:, np.newaxis] - values[np.newaxis, :]) / bw  # (len_x, n)
    kernel = np.exp(-0.5 * diff ** 2) / (np.sqrt(2 * np.pi) * bw)
    return kernel.mean(axis=1)

ERA5_DIR   = ("/s3/zhengsen/wildfire_dataset/self_built_materials/"
              "bc_raw_data_new_fast/drivers/ERA5")
MODIS_DIR  = ("/s3/zhengsen/wildfire_dataset/self_built_materials/"
              "bc_raw_data_new_fast/drivers/processed/MCD14A1_combined")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

FWI_COMPONENTS = ['ffmc', 'dmc', 'dc', 'isi', 'bui', 'fwi']
FWI_LABELS     = {'ffmc': 'FFMC', 'dmc': 'DMC', 'dc': 'DC',
                  'isi': 'ISI', 'bui': 'BUI', 'fwi': 'FWI'}
N_TILES        = 4   # 4 spatial tiles per day


# ---------------------------------------------------------------------------
# Tile path helpers
# ---------------------------------------------------------------------------

def _era5_tile_path(date_str: str, tile_idx: int) -> str:
    """Construct ERA5 tile path. date_str format: YYYY-MM-DD"""
    yyyy, mm, dd = date_str.split('-')
    return os.path.join(ERA5_DIR,
                        f"ERA5_{yyyy}_{mm}_{dd}_tile{tile_idx:02d}.tif")


def _modis_tile_path(date_str: str, tile_idx: int) -> str:
    """Construct MODIS tile path. date_str format: YYYY-MM-DD"""
    yyyy, mm, dd = date_str.split('-')
    return os.path.join(MODIS_DIR,
                        f"MCD14A1_{yyyy}_{mm}_{dd}_tile{tile_idx:02d}.tif")


def _era5_tiles_exist(date_str: str) -> bool:
    """Return True if at least 2 of 4 ERA5 tiles exist for the given date."""
    count = sum(
        1 for i in range(N_TILES)
        if os.path.exists(_era5_tile_path(date_str, i))
    )
    return count >= 2


def _modis_tiles_exist(date_str: str) -> bool:
    """Return True if at least 2 of 4 MODIS tiles exist for the given date."""
    count = sum(
        1 for i in range(N_TILES)
        if os.path.exists(_modis_tile_path(date_str, i))
    )
    return count >= 2


# ---------------------------------------------------------------------------
# Per-tile FWI season computation
# ---------------------------------------------------------------------------

def _init_moisture_codes(grid_shape: tuple) -> tuple:
    """Return (ffmc, dmc, dc) arrays initialized to CWFIS spring startup values."""
    return (
        np.full(grid_shape, FFMC_INIT, dtype=np.float32),
        np.full(grid_shape, DMC_INIT,  dtype=np.float32),
        np.full(grid_shape, DC_INIT,   dtype=np.float32),
    )


def _get_tile_grid_shape(tile_path: str) -> tuple:
    """Read (H, W) from a GeoTIFF without loading full data."""
    ds = gdal.Open(tile_path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open tile: {tile_path}")
    h, w = ds.RasterYSize, ds.RasterXSize
    ds = None
    return (h, w)


def compute_fwi_season_per_tile(year: int, tile_idx: int) -> dict:
    """Compute FWI components for one fire season for a single spatial tile.

    This replaces compute_fwi_season from fwi_module.py because the actual ERA5
    tiles use the naming format ERA5_YYYY_MM_DD_tileNN.tif, not YYYYMMDD.tif.

    Args:
        year:      fire season year (April 1 - October 31).
        tile_idx:  spatial tile index (0-3).

    Returns:
        dict with keys: 'ffmc', 'dmc', 'dc', 'isi', 'bui', 'fwi', 'dates', 'missing_days'
        Each component is a list of 2D arrays (one per season day).
    """
    # Determine grid shape from first available tile for this year
    grid_shape = None
    for month in range(4, 11):
        sample_date = date(year, month, 1).strftime('%Y-%m-%d')
        p = _era5_tile_path(sample_date, tile_idx)
        if os.path.exists(p):
            grid_shape = _get_tile_grid_shape(p)
            break

    if grid_shape is None:
        print(f"  Tile {tile_idx}: no ERA5 data found for {year} - skipping")
        return None

    ffmc, dmc, dc = _init_moisture_codes(grid_shape)

    outputs = {k: [] for k in FWI_COMPONENTS}
    dates        = []
    missing_days = []

    season_start = date(year, 4, 1)
    season_end   = date(year, 10, 31)
    current      = season_start

    while current <= season_end:
        date_str  = current.strftime('%Y-%m-%d')
        tile_path = _era5_tile_path(date_str, tile_idx)

        if not os.path.exists(tile_path):
            missing_days.append(date_str)
            for k in FWI_COMPONENTS:
                outputs[k].append(np.full(grid_shape, np.nan, dtype=np.float32))
        else:
            bands  = read_era5_tile(tile_path, bands=[2, 3, 5, 7, 13])
            # Mask out-of-domain pixels: ERA5 encodes ocean/outside as 0.0 with no nodata.
            # Band 13 (temperature_2m_max) in Kelvin: 0 K is physically impossible.
            # Band 7 (dewpoint) in Kelvin: same issue.
            # Set these to NaN so FWI equations don't operate on spurious zeros.
            outside_mask = (bands[13] == 0.0) | (bands[7] == 0.0)
            for b_idx in [2, 3, 5, 7, 13]:
                bands[b_idx] = np.where(outside_mask, np.nan, bands[b_idx])
            # Clamp Band 5 (precipitation in metres) to >= 0 to prevent assertion failure
            # from ERA5 floating-point rounding artefacts (tiny negatives ~-1e-7 m).
            bands[5] = np.maximum(bands[5], 0.0)
            T_c, RH, wind, rain = extract_fwi_inputs_from_bands(bands)
            month  = current.month

            ffmc = ffmc_next(ffmc, T_c, RH, wind, rain)
            dmc  = dmc_next(dmc, T_c, RH, rain, month)
            dc   = dc_next(dc, T_c, rain, month)
            isi  = isi_calc(ffmc, wind)
            bui  = bui_calc(dmc, dc)
            fwi_val = fwi_calc(isi, bui)

            outputs['ffmc'].append(ffmc.copy())
            outputs['dmc'].append(dmc.copy())
            outputs['dc'].append(dc.copy())
            outputs['isi'].append(isi.copy())
            outputs['bui'].append(bui.copy())
            outputs['fwi'].append(fwi_val.copy())

        dates.append(date_str)
        current += timedelta(days=1)

    outputs['dates']        = dates
    outputs['missing_days'] = missing_days
    return outputs


# ---------------------------------------------------------------------------
# MODIS alignment
# ---------------------------------------------------------------------------

def _align_mask_to_grid(mask: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resample a boolean mask to target_shape using nearest-neighbor resampling."""
    if mask.shape == target_shape:
        return mask
    return _nn_zoom(mask.astype(np.uint8), target_shape) > 0


def _load_modis_masks_for_date(date_str: str, era5_shape: tuple) -> tuple:
    """Load and merge MODIS fire/non-fire masks from all 4 tiles for one date.

    Reads all available MODIS tile files for the date, extracts fire and
    non-fire pixel masks, and resamples them to match the ERA5 grid shape.

    Returns:
        (fire_mask, nonfire_mask): both bool arrays of shape era5_shape.
        Returns (None, None) if no MODIS tiles available.
    """
    # Aggregate across all MODIS tiles individually (each aligned to ERA5 tile shape)
    # Since ERA5 and MODIS tiles are co-registered spatially (same tile index),
    # we align each MODIS tile to the corresponding ERA5 tile shape and
    # return combined masks (same shape as one ERA5 tile).
    # In practice for statistics we just need fire/non-fire pixel counts per tile.
    # This function is called per tile_idx externally.
    raise NotImplementedError("Use load_modis_masks_for_tile instead")


def load_modis_masks_for_tile(date_str: str, tile_idx: int,
                               era5_shape: tuple) -> tuple:
    """Load MODIS fire/non-fire masks for a single tile index and date.

    Args:
        date_str:   YYYY-MM-DD
        tile_idx:   0-3
        era5_shape: (H, W) of the ERA5 tile to align to

    Returns:
        (fire_mask, nonfire_mask): bool arrays of shape era5_shape.
        Returns (None, None) if MODIS tile not available.
    """
    modis_path = _modis_tile_path(date_str, tile_idx)
    if not os.path.exists(modis_path):
        return None, None

    try:
        firemask, _, _ = read_modis_tile(modis_path)
    except Exception as e:
        print(f"  Warning: cannot read MODIS tile {modis_path}: {e}")
        return None, None

    fire_mask    = _align_mask_to_grid(get_positive_mask(firemask), era5_shape)
    nonfire_mask = _align_mask_to_grid(firemask == 5, era5_shape)
    return fire_mask, nonfire_mask


# ---------------------------------------------------------------------------
# Sample collection
# ---------------------------------------------------------------------------

def collect_fwi_samples_for_year(year: int) -> dict:
    """Compute FWI for one fire season across all 4 tiles and collect per-pixel values.

    For each day and tile: if there are fire pixels in the MODIS mask, extract
    FWI values at fire and non-fire locations, sub-sampling non-fire to 5x fire count.

    Returns:
        dict with keys 'fire' and 'nonfire', each containing
        {component_name: list of float values}.
    """
    fire_samples    = {c: [] for c in FWI_COMPONENTS}
    nonfire_samples = {c: [] for c in FWI_COMPONENTS}

    for tile_idx in range(N_TILES):
        print(f"  Year {year}, tile {tile_idx}...")
        season = compute_fwi_season_per_tile(year, tile_idx)
        if season is None:
            continue

        for i, date_str in enumerate(season['dates']):
            # Skip if FWI arrays are all NaN (missing ERA5 tile)
            if np.all(np.isnan(season['ffmc'][i])):
                continue

            era5_shape = season['ffmc'][i].shape
            fire_mask, nonfire_mask = load_modis_masks_for_tile(
                date_str, tile_idx, era5_shape
            )
            if fire_mask is None or fire_mask.sum() == 0:
                continue

            for c in FWI_COMPONENTS:
                fwi_arr      = season[c][i]
                fire_vals    = fwi_arr[fire_mask]
                nonfire_vals = fwi_arr[nonfire_mask] if nonfire_mask is not None else np.array([])

                fire_vals    = fire_vals[~np.isnan(fire_vals)]
                nonfire_vals = nonfire_vals[~np.isnan(nonfire_vals)]

                if len(fire_vals) == 0:
                    continue

                # Sub-sample non-fire to avoid class-imbalance bias in stats
                if len(nonfire_vals) > len(fire_vals) * 5:
                    rng = np.random.default_rng(seed=year * 1000 + tile_idx * 200 + i)
                    nonfire_vals = rng.choice(nonfire_vals,
                                              size=len(fire_vals) * 5,
                                              replace=False)

                fire_samples[c].extend(fire_vals.tolist())
                nonfire_samples[c].extend(nonfire_vals.tolist())

    return {
        'fire':    {c: np.array(fire_samples[c])    for c in FWI_COMPONENTS},
        'nonfire': {c: np.array(nonfire_samples[c]) for c in FWI_COMPONENTS},
    }


# ---------------------------------------------------------------------------
# Weight calibration
# ---------------------------------------------------------------------------

def compute_weight_vector(fire_samples: dict, nonfire_samples: dict) -> dict:
    """Compute normalized FWI component weight vector using separation scores.

    Method (CONTEXT.md OQ3 guidance):
      separation_c = max(0, (mean_fire_c - mean_nonfire_c) / (std_nonfire_c + 1e-6))
      weights = separation / sum(separation)

    Args:
        fire_samples:    dict {component: 1D array}
        nonfire_samples: dict {component: 1D array}

    Returns:
        dict mapping component name (uppercase) to normalized weight (float).
        Includes 'sum_check', 'method', 'calibration_years' keys.
    """
    separations = {}
    for c in FWI_COMPONENTS:
        f = fire_samples[c]
        n = nonfire_samples[c]
        if len(f) == 0 or len(n) == 0:
            print(f"  Warning: no samples for {c} - assigning 0 separation")
            separations[c] = 0.0
            continue
        fire_mean    = float(np.nanmean(f))
        nonfire_mean = float(np.nanmean(n))
        nonfire_std  = float(np.nanstd(n))
        sep = (fire_mean - nonfire_mean) / (nonfire_std + 1e-6)
        separations[c] = max(sep, 0.0)
        print(f"  {FWI_LABELS[c]:4s}: fire_mean={fire_mean:.2f}  nonfire_mean={nonfire_mean:.2f}"
              f"  std={nonfire_std:.2f}  sep={sep:.4f}")

    total = sum(separations.values()) + 1e-8
    weights_lower = {c: separations[c] / total for c in FWI_COMPONENTS}
    weights_upper = {FWI_LABELS[c]: round(weights_lower[c], 6) for c in FWI_COMPONENTS}
    weights_upper['sum_check'] = round(sum(weights_lower.values()), 6)
    return weights_upper


# ---------------------------------------------------------------------------
# Distribution plot
# ---------------------------------------------------------------------------

def plot_fwi_distributions(fire_samples: dict, nonfire_samples: dict,
                            output_path: str) -> None:
    """Plot fire vs. non-fire FWI component distributions (6-panel KDE figure).

    Uses a pure-NumPy Gaussian KDE (_numpy_kde) to avoid scipy dependency
    (scipy has a NumPy 2.x ABI conflict in the geo_env).
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('FWI Component Distributions: Fire vs. Non-Fire Pixels\n'
                 '(Canada, calibration years 2017/2018/2021)', fontsize=13)

    for idx, c in enumerate(FWI_COMPONENTS):
        ax = axes[idx // 3][idx % 3]
        label_name = FWI_LABELS[c]

        fire_vals    = fire_samples[c]
        nonfire_vals = nonfire_samples[c]

        if len(fire_vals) < 10 or len(nonfire_vals) < 10:
            ax.text(0.5, 0.5, f'{label_name}\nInsufficient data\n'
                    f'fire n={len(fire_vals)}, nonfire n={len(nonfire_vals)}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label_name)
            continue

        # Clip extreme outliers for cleaner KDE
        all_vals = np.concatenate([fire_vals, nonfire_vals])
        vmin = float(np.percentile(all_vals, 1))
        vmax = float(np.percentile(all_vals, 99))
        x    = np.linspace(vmin, vmax, 300)

        fire_clipped    = fire_vals[   (fire_vals    >= vmin) & (fire_vals    <= vmax)]
        nonfire_clipped = nonfire_vals[(nonfire_vals >= vmin) & (nonfire_vals <= vmax)]

        # Subsample for KDE evaluation if too many points (speed)
        rng = np.random.default_rng(seed=42)
        MAX_KDE_PTS = 50_000
        fire_kde_in    = (fire_clipped    if len(fire_clipped)    <= MAX_KDE_PTS
                          else rng.choice(fire_clipped,    MAX_KDE_PTS, replace=False))
        nonfire_kde_in = (nonfire_clipped if len(nonfire_clipped) <= MAX_KDE_PTS
                          else rng.choice(nonfire_clipped, MAX_KDE_PTS, replace=False))

        try:
            kde_fire_y    = _numpy_kde(fire_kde_in,    x)
            kde_nonfire_y = _numpy_kde(nonfire_kde_in, x)
            ax.plot(x, kde_fire_y,    color='red',  lw=2, label=f'Fire (n={len(fire_vals):,})')
            ax.plot(x, kde_nonfire_y, color='blue', lw=2, label=f'Non-fire (n={len(nonfire_vals):,})')
            ax.fill_between(x, kde_fire_y,    alpha=0.15, color='red')
            ax.fill_between(x, kde_nonfire_y, alpha=0.15, color='blue')
        except Exception as e:
            print(f"  KDE fallback to histogram for {label_name}: {e}")
            ax.hist(fire_vals,    bins=50, color='red',  alpha=0.5, density=True,
                    label=f'Fire (n={len(fire_vals):,})')
            ax.hist(nonfire_vals, bins=50, color='blue', alpha=0.5, density=True,
                    label=f'Non-fire (n={len(nonfire_vals):,})')

        fire_mean    = float(np.nanmean(fire_vals))
        nonfire_mean = float(np.nanmean(nonfire_vals))
        nonfire_std  = float(np.nanstd(nonfire_vals))
        sep = (fire_mean - nonfire_mean) / (nonfire_std + 1e-6)

        ax.axvline(fire_mean,    color='red',    linestyle='--', alpha=0.8, lw=1.5)
        ax.axvline(nonfire_mean, color='blue',   linestyle='--', alpha=0.8, lw=1.5)
        ax.set_xlabel(label_name, fontsize=10)
        ax.set_ylabel('Density',  fontsize=9)
        ax.set_title(f'{label_name}  (sep={sep:.3f})', fontsize=11)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Distribution plot saved: {output_path}")
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  File size: {size_kb:.1f} KB")


# ---------------------------------------------------------------------------
# Main EDA runner
# ---------------------------------------------------------------------------

def run_fwi_eda(calibration_years: list = None) -> dict:
    """Run full FWI EDA: compute seasons, calibrate weights, produce plots.

    Args:
        calibration_years: list of years to use (default: [2017, 2018, 2021]).

    Returns:
        dict with 'weights', 'fire_samples', 'nonfire_samples'.
    """
    if calibration_years is None:
        calibration_years = [2017, 2018, 2021]

    # Check ERA5 availability (fire season = Apr-Oct, ~214 days * 4 tiles = 856 tiles)
    available_years = []
    for year in calibration_years:
        n_tiles = len(glob.glob(os.path.join(ERA5_DIR, f"ERA5_{year}_*.tif")))
        if n_tiles >= 300:  # at least 75 days * 4 tiles
            available_years.append(year)
            print(f"ERA5 {year}: {n_tiles} tiles available - INCLUDED")
        else:
            print(f"Warning: ERA5 {year} has only {n_tiles} tiles - SKIPPING")

    if not available_years:
        raise RuntimeError(
            f"No ERA5 data available for any of {calibration_years}. "
            "Check ERA5 download status."
        )

    print(f"\nCalibrating FWI weights using years: {available_years}")

    # Collect samples across all years
    all_fire    = {c: [] for c in FWI_COMPONENTS}
    all_nonfire = {c: [] for c in FWI_COMPONENTS}

    for year in available_years:
        print(f"\nProcessing {year}...")
        samples = collect_fwi_samples_for_year(year)
        for c in FWI_COMPONENTS:
            all_fire[c].extend(samples['fire'][c].tolist())
            all_nonfire[c].extend(samples['nonfire'][c].tolist())
        # Print interim sample counts
        print(f"  {year} fire samples: {len(all_fire['ffmc'])}, "
              f"nonfire samples: {len(all_nonfire['ffmc'])}")

    fire_arrays    = {c: np.array(all_fire[c])    for c in FWI_COMPONENTS}
    nonfire_arrays = {c: np.array(all_nonfire[c]) for c in FWI_COMPONENTS}

    print(f"\nTotal fire samples: {len(fire_arrays['ffmc']):,}")
    print(f"Total nonfire samples: {len(nonfire_arrays['ffmc']):,}")

    # Compute weight vector
    print("\nComputing separation scores and weight vector...")
    weights = compute_weight_vector(fire_arrays, nonfire_arrays)
    weights['calibration_years'] = available_years
    weights['method'] = "normalized_separation_score: max(0, (mean_fire - mean_nonfire) / std_nonfire)"

    # Save weight vector
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    weight_path = os.path.join(OUTPUT_DIR, "weight_vector_w1w6.json")
    with open(weight_path, 'w') as f:
        json.dump(weights, f, indent=2)
    print(f"\nWeight vector saved: {weight_path}")
    print(f"  Weights:")
    for k in ['FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']:
        print(f"    {k}: {weights[k]:.6f}")
    print(f"  Sum: {weights['sum_check']}")

    # Produce distribution plot
    plot_path = os.path.join(OUTPUT_DIR, "fwi_distributions_fire_vs_nonfire.png")
    print(f"\nGenerating distribution plot...")
    plot_fwi_distributions(fire_arrays, nonfire_arrays, plot_path)

    return {
        'weights':        weights,
        'fire_samples':   fire_arrays,
        'nonfire_samples': nonfire_arrays,
    }


if __name__ == "__main__":
    import sys
    years = None
    if "--years" in sys.argv:
        idx = sys.argv.index("--years")
        years = [int(y) for y in sys.argv[idx + 1].split(',')]

    result = run_fwi_eda(calibration_years=years)
    print("\n" + "=" * 60)
    print("FWI EDA complete.")
    print("Weight vector:")
    for k in ['FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']:
        print(f"  {k}: {result['weights'][k]:.6f}")
    print(f"  sum_check: {result['weights']['sum_check']}")
    print(f"  calibration_years: {result['weights']['calibration_years']}")
