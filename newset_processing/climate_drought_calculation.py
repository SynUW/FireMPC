# -*- coding: utf-8 -*-
"""
Compute climate-drought indicators: VPD and SPEI-3M (modelling-oriented).
- VPD: Vapor Pressure Deficit
- SPEI-3M: daily resolution; time scale ~3 months = 90-day rolling accumulation D = P - PET;
  per pixel, stratified by day of year (DOY), fitted with log-logistic (scipy.stats.fisk) and mapped to a standard normal;
  Winsor-clip the fitting samples + hard-clip the output SPEI to suppress daily-scale noise and fitting instability at the extremes.
  Units match ECMWF/ERA5_LAND/DAILY_AGGR (m, K).

Parallel (default): submit two independent tasks (VPD, SPEI) per tile and run with ProcessPoolExecutor;
with 4 tiles, up to 8 tasks; use --workers to raise/lower concurrency. Spatial outlier infill still uses scipy.griddata.
Serial: python climate_drought_calculation.py --serial
Overwrite recomputation: python climate_drought_calculation.py --overwrite (clears VPD/SPEI outputs and SPEI scratch for this tile batch; no corruption detection)
Default: at startup, check whether output GeoTIFFs can be fully opened by rasterio and read a sample; files that cannot be read or are too small are treated as corrupt and deleted, then only the missing days are recomputed.
Run a single category: --compute vpd or --compute spei (default 'all' runs both).
SPEI: roll90.dat and similar files represent only the first-pass memmap; the SPEI_3M directory is not written until the third pass, after the second-pass DOY fitting completes -- it may stay empty until then.
VPD acceleration: --vpd-no-spatial-interp (skip griddata); --vpd-inner-workers N (multi-process per day inside a tile).
SPEI second pass: vectorized (method-of-moments log-logistic, sort instead of nanpercentile, all NumPy/ndtri);
parallelized per DOY (--spei-doy-workers; total processes = workers x number of tiles).
Speed comparison: original per-pixel fisk.fit MLE ~530 h/tile -> vectorized ~0.5 min/tile (~60000x faster).
"""

import argparse
import os
import shutil
from typing import Optional
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from scipy.stats import fisk, norm
import glob
import warnings
import logging
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ===== Configuration parameters =====
# Input directory (ERA5 data)
ERA5_DIR = r'/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/ERA5'

# Output directory
OUTPUT_DIR = r'/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/climate_drought'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'VPD'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'SPEI_3M'), exist_ok=True)

# Date range
DATE_START = "2000-01-01"
DATE_END = "2025-12-31"

# Extreme-value thresholds (used for filtering)
TEMP_MIN, TEMP_MAX = -100, 70  # Celsius
DEWPOINT_MIN, DEWPOINT_MAX = -100, 60  # Celsius
PRECIP_MIN, PRECIP_MAX = 0, 500  # mm/day
PET_MIN, PET_MAX = -50, 50  # mm/day
VPD_MIN, VPD_MAX = 0, 10  # kPa

# SPEI-3M: daily resolution + 90-day rolling scale + DOY standardization
SPEI_ROLLING_DAYS = 90
SPEI_WINDOW_MIN_VALID_RATIO = 0.85  # Minimum valid-day ratio inside the 90-day window (per pixel)
SPEI_DOY_MIN_SAMPLES = 15  # Minimum number of historical-year samples on the same DOY required to fit
SPEI_WINSOR_LO_PCT = 1.0  # Lower-quantile Winsor (%) applied to historical samples before fitting
SPEI_WINSOR_HI_PCT = 99.0
SPEI_PROB_CLIP = 1e-6  # CDF clip to avoid overflow in norm.ppf
SPEI_VALUE_CLIP = 4.0  # Clip output SPEI to [-CLIP, CLIP] to suppress extreme instability
SPEI_SCRATCH_SUBDIR = "_spei_scratch"  # memmap subdirectory; root defaults to OUTPUT_DIR, can be redirected to a local disk via the SPEI_TMPDIR environment variable (mmap can be slow on object storage)
SPEI_DELETE_SCRATCH = True  # Remove the temp directory after computation to save space

# Spatial interpolation parameters (uses scipy.interpolate.griddata throughout, preserving linear/cubic interpolation semantics)
INTERPOLATION_METHOD = 'linear'  # 'linear', 'nearest', 'cubic'
MAX_INTERPOLATION_DISTANCE = 3  # Maximum interpolation distance (pixels)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# A GeoTIFF smaller than this byte size is almost certainly an incomplete write or empty stub
MIN_GEOTIFF_BYTES = 512


def is_readable_geotiff(path: str) -> bool:
    """
    Check whether the output raster is fully readable (avoids treating a half-written leftover, after a killed write, as already existing and skipping it).
    Tries to open and read a 1x1 window in the top-left of band 1.
    """
    try:
        if not os.path.isfile(path):
            return False
        if os.path.getsize(path) < MIN_GEOTIFF_BYTES:
            return False
        with rasterio.open(path) as src:
            if src.count < 1 or src.width < 1 or src.height < 1:
                return False
            _ = src.read(1, window=Window(0, 0, 1, 1))
        return True
    except Exception:
        return False


def spei_scratch_dir(tile_idx: int) -> str:
    root = os.environ.get("SPEI_TMPDIR") or OUTPUT_DIR
    return os.path.join(root, SPEI_SCRATCH_SUBDIR, f"tile{tile_idx:02d}")


def purge_tile_outputs_for_overwrite(tile_indices, compute: str = "all") -> None:
    """
    --overwrite: per `compute`, delete each tile's outputs and (if spei is included) the SPEI scratch.
    compute: 'all' | 'vpd' | 'spei'
    """
    do_vpd = compute in ("all", "vpd")
    do_spei = compute in ("all", "spei")
    for ti in tile_indices:
        if do_vpd:
            for p in glob.glob(os.path.join(OUTPUT_DIR, "VPD", f"VPD_*_tile{ti:02d}.tif")):
                try:
                    os.remove(p)
                except OSError:
                    pass
        if do_spei:
            for p in glob.glob(
                os.path.join(OUTPUT_DIR, "SPEI_3M", f"SPEI_3M_*_tile{ti:02d}.tif")
            ):
                try:
                    os.remove(p)
                except OSError:
                    pass
            shutil.rmtree(spei_scratch_dir(ti), ignore_errors=True)


def repair_corrupt_output_geotiffs(tile_indices, compute: str = "all") -> int:
    """
    Non-overwrite path: per `compute`, scan tifs of this tile batch in the corresponding directory and delete those that cannot be fully read.
    Returns the number deleted.
    """
    pairs = []
    if compute in ("all", "vpd"):
        pairs.append((os.path.join(OUTPUT_DIR, "VPD"), "VPD_"))
    if compute in ("all", "spei"):
        pairs.append((os.path.join(OUTPUT_DIR, "SPEI_3M"), "SPEI_3M_"))

    removed = 0
    for ti in tile_indices:
        for folder, prefix in pairs:
            pat = os.path.join(folder, f"{prefix}*_tile{ti:02d}.tif")
            for path in glob.glob(pat):
                if is_readable_geotiff(path):
                    continue
                try:
                    os.remove(path)
                    removed += 1
                    logger.warning(f"Deleted corrupt or unreadable output: {path}")
                except OSError as e:
                    logger.warning(f"Cannot delete {path}: {e}")
    return removed


# ================== Utility functions ==================

def kelvin_to_celsius(temp_k):
    """Kelvin to Celsius"""
    return temp_k - 273.15

def meter_to_mm(value_m):
    """Meters to millimeters"""
    return value_m * 1000.0

def calc_saturation_vapor_pressure(temp_c):
    """
    Compute saturation vapor pressure (kPa).
    Uses Tetens' formula: es = 0.611 * exp(17.27*T / (T + 237.3))
    
    Args:
        temp_c: temperature (Celsius)
    Returns:
        saturation vapor pressure (kPa)
    """
    return 0.611 * np.exp(17.27 * temp_c / (temp_c + 237.3))

def calc_vpd(temp_c, dewpoint_c):
    """
    Compute VPD (vapor pressure deficit).
    VPD = es(T) - ea(Td)
    
    Args:
        temp_c: air temperature (Celsius)
        dewpoint_c: dewpoint temperature (Celsius)
    Returns:
        VPD (kPa)
    """
    es = calc_saturation_vapor_pressure(temp_c)  # Saturation vapor pressure
    ea = calc_saturation_vapor_pressure(dewpoint_c)  # Actual vapor pressure
    vpd = es - ea
    return np.maximum(vpd, 0)  # VPD should not be negative

def detect_outliers(data, min_val, max_val):
    """
    Detect extreme values.
    
    Args:
        data: data array
        min_val: minimum valid value
        max_val: maximum valid value
    Returns:
        mask: True for valid values, False for extreme values
    """
    mask = np.ones_like(data, dtype=bool)
    mask[np.isnan(data)] = False
    mask[data < min_val] = False
    mask[data > max_val] = False
    return mask

def spatial_interpolation_2d(data, valid_mask, method='linear', max_distance=3):
    """
    Spatially interpolate missing values (using surrounding pixels).
    
    Args:
        data: input data (2D array)
        valid_mask: valid-value mask (True = valid, False = needs interpolation)
        method: interpolation method ('linear', 'nearest', 'cubic')
        max_distance: maximum interpolation distance (pixels)
    Returns:
        interpolated data
    """
    if valid_mask.sum() == 0:
        return data  # No valid values; cannot interpolate

    result = data.copy()
    rows, cols = np.where(valid_mask)
    if len(rows) < 4:  # In 2D, griddata's linear/cubic methods need a minimum number of non-degenerate control points
        return result

    values = data[rows, cols]
    invalid_rows, invalid_cols = np.where(~valid_mask)
    if len(invalid_rows) == 0:
        return result

    try:
        from scipy.interpolate import griddata
        from scipy.spatial import cKDTree

        points = np.column_stack([rows, cols])
        xi = np.column_stack([invalid_rows, invalid_cols])

        if max_distance is not None and max_distance > 0:
            tree = cKDTree(points)
            try:
                distances, _ = tree.query(xi, k=1, workers=-1)
            except TypeError:
                distances, _ = tree.query(xi, k=1)
            close_enough = distances <= max_distance
            if close_enough.sum() > 0:
                xi_close = xi[close_enough]
                interp_values = griddata(
                    points, values, xi_close, method=method, fill_value=np.nan
                )
                ir = invalid_rows[close_enough]
                ic = invalid_cols[close_enough]
                m = np.isfinite(interp_values)
                result[ir[m], ic[m]] = interp_values[m]
        else:
            interp_values = griddata(points, values, xi, method=method, fill_value=np.nan)
            m = np.isfinite(interp_values)
            result[invalid_rows[m], invalid_cols[m]] = interp_values[m]

    except Exception as e:
        logger.warning(f"Spatial interpolation failed: {e}; using raw data")

    return result

def mask_outliers_to_nan(data: np.ndarray, min_val, max_val) -> np.ndarray:
    """
    Mark pixels outside [min, max] and original NaNs as NaN; do not perform spatial interpolation (VPD fast path, avoiding griddata).
    """
    out = np.asarray(data, dtype=np.float32, copy=True)
    valid = detect_outliers(out, min_val, max_val)
    out[~valid] = np.nan
    return out


def filter_and_interpolate(data, min_val, max_val, method='linear', max_distance=3):
    """
    Filter extreme values and interpolate.
    
    Args:
        data: input data
        min_val: minimum valid value
        max_val: maximum valid value
        method: interpolation method
        max_distance: maximum interpolation distance
    Returns:
        processed data
    """
    if data.ndim == 2:
        valid_mask = detect_outliers(data, min_val, max_val)
        return spatial_interpolation_2d(data, valid_mask, method, max_distance)
    elif data.ndim == 3:
        # Process layer by layer
        result = np.zeros_like(data)
        for i in range(data.shape[0]):
            valid_mask = detect_outliers(data[i], min_val, max_val)
            result[i] = spatial_interpolation_2d(data[i], valid_mask, method, max_distance)
        return result
    else:
        return data

# ================== ERA5 data loading ==================

def get_era5_file_path(date_str, tile_idx=0):
    """
    Build the ERA5 file path.
    Format: ERA5_YYYY_MM_DD_tile00.tif
    """
    date_formatted = date_str.replace('-', '_')
    filename = f"ERA5_{date_formatted}_tile{tile_idx:02d}.tif"
    return os.path.join(ERA5_DIR, filename)

def read_era5_band(file_path, band_index):
    """
    Read the specified band of an ERA5 GeoTIFF (band order matches GEE ECMWF/ERA5_LAND/DAILY_AGGR export):

    0 temperature_2m, 1 u_component_of_wind_10m, 2 v_component_of_wind_10m, 3 snow_cover,
    4 total_precipitation_sum, 5 surface_latent_heat_flux_sum, 6 dewpoint_temperature_2m,
    7 surface_pressure, 8-11 volumetric_soil_water_layer_1-4, 12 temperature_2m_max,
    13 skin_temperature_max, 14 potential_evaporation_sum, 15 total_evaporation_sum,
    16 skin_reservoir_content, 17 surface_net_solar_radiation_sum
    """
    try:
        with rasterio.open(file_path) as src:
            data = src.read(band_index + 1)  # rasterio bands are 1-based
            profile = src.profile
            return data, profile
    except Exception as e:
        logger.error(f"Failed to read file {file_path}, band {band_index}: {e}")
        return None, None


def read_era5_data(date_str, tile_idx=0, subset_keys=None):
    """
    Read the required ERA5 bands for a single day (open once, read once or in batches; avoid reopening the same file).

    subset_keys:
        None means read all four of temperature_2m, dewpoint, precipitation, PET;
        e.g. ['temperature_2m', 'dewpoint_temperature_2m'] reads only the two bands needed for VPD.

    Returns:
        dict: 2D array per key, plus 'profile'
    """
    file_path = get_era5_file_path(date_str, tile_idx)
    if not os.path.exists(file_path):
        logger.warning(f"File does not exist: {file_path}")
        return None

    band_indices = {
        "temperature_2m": 0,
        "dewpoint_temperature_2m": 6,
        "total_precipitation_sum": 4,
        "potential_evaporation_sum": 14,
    }
    if subset_keys is None:
        keys = list(band_indices.keys())
    else:
        keys = [k for k in subset_keys if k in band_indices]
        if not keys:
            logger.error("read_era5_data: subset_keys is empty or invalid")
            return None

    try:
        with rasterio.open(file_path) as src:
            profile = src.profile
            # rasterio bands are 1-based; reading multiple bands in one call reduces decoding/IO
            band_nos = [band_indices[k] + 1 for k in keys]
            stacked = src.read(band_nos)
            data = {k: stacked[i] for i, k in enumerate(keys)}
            data["profile"] = profile
            return data
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        return None

# ================== VPD computation ==================

def calculate_vpd_for_date(
    date_str,
    tile_idx=0,
    use_spatial_interp: bool = True,
):
    """
    Compute single-day VPD (Magnus/Tetens approximation; 2 m air temperature and 2 m dewpoint are both in K, matching Daily AGGR).
    When use_spatial_interp=False, griddata is skipped and extreme values simply become NaN -- significantly faster (ERA5 is typically already smooth).
    """
    # Load data (only two bands, single open)
    data = read_era5_data(
        date_str,
        tile_idx,
        subset_keys=["temperature_2m", "dewpoint_temperature_2m"],
    )
    if data is None:
        return None, None

    temp_c = kelvin_to_celsius(data["temperature_2m"])
    dewpoint_c = kelvin_to_celsius(data["dewpoint_temperature_2m"])

    if use_spatial_interp:
        temp_c = filter_and_interpolate(
            temp_c, TEMP_MIN, TEMP_MAX, INTERPOLATION_METHOD, MAX_INTERPOLATION_DISTANCE
        )
        dewpoint_c = filter_and_interpolate(
            dewpoint_c,
            DEWPOINT_MIN,
            DEWPOINT_MAX,
            INTERPOLATION_METHOD,
            MAX_INTERPOLATION_DISTANCE,
        )
    else:
        temp_c = mask_outliers_to_nan(temp_c, TEMP_MIN, TEMP_MAX)
        dewpoint_c = mask_outliers_to_nan(dewpoint_c, DEWPOINT_MIN, DEWPOINT_MAX)

    vpd = calc_vpd(temp_c, dewpoint_c)

    if use_spatial_interp:
        vpd = filter_and_interpolate(
            vpd, VPD_MIN, VPD_MAX, INTERPOLATION_METHOD, MAX_INTERPOLATION_DISTANCE
        )
    else:
        vpd = mask_outliers_to_nan(vpd, VPD_MIN, VPD_MAX)

    return vpd, data["profile"]

def save_geotiff(data, profile, output_path):
    """Save as GeoTIFF"""
    profile_out = profile.copy()
    profile_out.update({
        'count': 1,
        'dtype': 'float32',
        'compress': 'lzw',
        'nodata': np.nan
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with rasterio.open(output_path, 'w', **profile_out) as dst:
        dst.write(data.astype('float32'), 1)


def _vpd_one_day_worker(task: tuple) -> tuple:
    """Single-day VPD in a child process: (date_str, tile_idx, overwrite, use_spatial_interp) -> (date_str, status)."""
    _worker_limit_blas_threads()
    date_str, tile_idx, overwrite, use_spatial = task
    output_path = os.path.join(
        OUTPUT_DIR,
        "VPD",
        f"VPD_{date_str.replace('-', '_')}_tile{tile_idx:02d}.tif",
    )
    if os.path.exists(output_path):
        if overwrite:
            try:
                os.remove(output_path)
            except OSError:
                pass
        elif is_readable_geotiff(output_path):
            return (date_str, "skip")
        else:
            try:
                os.remove(output_path)
            except OSError:
                pass
    vpd, profile = calculate_vpd_for_date(
        date_str, tile_idx, use_spatial_interp=use_spatial
    )
    if vpd is None:
        return (date_str, "missing")
    save_geotiff(vpd, profile, output_path)
    return (date_str, "ok")


def process_vpd_batch(
    dates,
    tile_idx=0,
    show_progress=True,
    overwrite=False,
    inner_workers: int = 1,
    use_spatial_interp: bool = True,
):
    """
    Batch VPD computation.

    inner_workers: when > 1, parallelize per day with multiple processes **within this tile** (composes with the outer per-tile parallelism).
    use_spatial_interp: when False, skip griddata; extreme values just become NaN -- much faster.
    """
    logger.info(f"Starting VPD computation, total {len(dates)} days, tile {tile_idx:02d}")
    inner_workers = max(1, int(inner_workers))

    def vpd_day_needs_work(date_str: str) -> bool:
        output_path = os.path.join(
            OUTPUT_DIR,
            "VPD",
            f"VPD_{date_str.replace('-', '_')}_tile{tile_idx:02d}.tif",
        )
        if not os.path.exists(output_path):
            return True
        if overwrite:
            try:
                os.remove(output_path)
            except OSError:
                pass
            return True
        if is_readable_geotiff(output_path):
            return False
        try:
            os.remove(output_path)
            logger.warning(f"Deleting corrupt VPD output and recomputing: {output_path}")
        except OSError:
            pass
        return True

    if inner_workers <= 1:
        it = (
            tqdm(dates, desc=f"VPD computation (tile{tile_idx:02d})")
            if show_progress
            else dates
        )
        for date_str in it:
            if not vpd_day_needs_work(date_str):
                continue
            vpd, profile = calculate_vpd_for_date(
                date_str, tile_idx, use_spatial_interp=use_spatial_interp
            )
            if vpd is None:
                logger.warning(f"Skip date {date_str} (data unavailable)")
                continue
            output_path = os.path.join(
                OUTPUT_DIR,
                "VPD",
                f"VPD_{date_str.replace('-', '_')}_tile{tile_idx:02d}.tif",
            )
            save_geotiff(vpd, profile, output_path)
        return

    pending = [d for d in dates if vpd_day_needs_work(d)]
    logger.info(
        f"tile {tile_idx:02d} VPD intra-day parallelism inner_workers={inner_workers}，"
        f"Pending {len(pending)}/{len(dates)} days"
    )
    if not pending:
        return
    with ProcessPoolExecutor(max_workers=inner_workers) as ex:
        futures = {
            ex.submit(
                _vpd_one_day_worker,
                (d, tile_idx, overwrite, use_spatial_interp),
            ): d
            for d in pending
        }
        done_iter = as_completed(futures)
        if show_progress:
            done_iter = tqdm(
                done_iter,
                total=len(pending),
                desc=f"VPD parallel tile{tile_idx:02d}",
            )
        for fut in done_iter:
            date_str, status = fut.result()
            if status == "missing":
                logger.warning(f"Skip date {date_str} (data unavailable)")

# ================== SPEI-3M computation ==================

def calculate_water_balance(precip_m, pet_m):
    """
    Compute the water balance D = P - PET.
    
    Args:
        precip_m: precipitation (m/day); ERA5-Land Daily AGGR is daily accumulated water depth (m)
        pet_m: potential evapotranspiration (m/day); same as above
    Returns:
        D (mm/day)
    """
    precip_mm = meter_to_mm(precip_m)
    pet_mm = meter_to_mm(pet_m)

    # GEE/ECMWF docs note that precipitation totals may have tiny negative values (GRIB packing); for water-balance use, treat as non-negative physical quantities
    precip_mm = np.maximum(precip_mm, 0.0)
    
    # Filter extreme values
    precip_mm = filter_and_interpolate(precip_mm, PRECIP_MIN, PRECIP_MAX,
                                        INTERPOLATION_METHOD, MAX_INTERPOLATION_DISTANCE)
    pet_mm = filter_and_interpolate(pet_mm, PET_MIN, PET_MAX,
                                     INTERPOLATION_METHOD, MAX_INTERPOLATION_DISTANCE)

    if precip_mm.shape != pet_mm.shape:
        raise ValueError(
            f"Precipitation and PET shape mismatch: precip {precip_mm.shape} vs pet {pet_mm.shape}"
        )

    D = precip_mm - pet_mm
    return D


def _crop_pad_array_to_shape(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    """Crop or top-left pad with NaN to (height, width)."""
    out = np.full((height, width), np.nan, dtype=np.float64)
    hh = min(height, arr.shape[0])
    ww = min(width, arr.shape[1])
    out[:hh, :ww] = np.asarray(arr, dtype=np.float64)[:hh, :ww]
    return out


def _resample_D_to_reference_grid(
    D: np.ndarray,
    src_profile,
    dst_profile,
    dst_shape: tuple,
):
    """
    Align the 2D water balance D to the reference grid (H, W).
    Prefer rasterio.reproject (bilinear); if CRS/transform is unavailable or fails, fall back to scipy.ndimage.zoom (bilinear).
    On success returns float64 (H, W); on failure returns None.
    """
    H, W = int(dst_shape[0]), int(dst_shape[1])
    if D.shape == (H, W):
        return np.asarray(D, dtype=np.float64, copy=False)

    src = np.ascontiguousarray(np.asarray(D, dtype=np.float32))
    dest = np.full((H, W), np.nan, dtype=np.float32)

    st = src_profile.get("transform")
    dt = dst_profile.get("transform")
    sc = src_profile.get("crs")
    dc = dst_profile.get("crs")

    if st is not None and dt is not None and sc is not None and dc is not None:
        try:
            reproject(
                source=src,
                destination=dest,
                src_transform=st,
                src_crs=sc,
                dst_transform=dt,
                dst_crs=dc,
                resampling=Resampling.bilinear,
                src_nodata=np.nan,
                dst_nodata=np.nan,
            )
            return dest.astype(np.float64, copy=False)
        except Exception:
            pass

    try:
        from scipy.ndimage import zoom

        sh, sw = src.shape
        if sh < 1 or sw < 1:
            return None
        zf = (H / sh, W / sw)
        out = zoom(np.asarray(D, dtype=np.float64), zf, order=1)
        out = _crop_pad_array_to_shape(out, H, W)
        if out.shape != (H, W):
            return None
        return out
    except Exception:
        return None


def _get_first_valid_era5_shape(dates, tile_idx):
    """Scan the date list and return (profile, shape) for the first readable ERA5 raster whose precipitation/PET share the same shape."""
    for date_str in dates:
        data = read_era5_data(date_str, tile_idx)
        if data is None:
            continue
        shp_p = data["total_precipitation_sum"].shape
        shp_e = data["potential_evaporation_sum"].shape
        if shp_p != shp_e:
            logger.warning(
                f"tile {tile_idx:02d} {date_str}: precipitation shape {shp_p} and PET {shp_e} mismatch; skipping this day as reference"
            )
            continue
        return data["profile"], shp_p
    return None, None


def _winsorize_sample(x, lo_pct, hi_pct):
    """
    Return the Winsor-clipped samples used for fitting and the (lo, hi) bounds; lo/hi may be None when samples are insufficient or degenerate.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return x, None, None
    lo = float(np.percentile(x, lo_pct))
    hi = float(np.percentile(x, hi_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo = float(np.min(x))
        hi = float(np.max(x))
    return np.clip(x, lo, hi), lo, hi


def _spei_fit_doy_vectorized(
    block: np.ndarray,
    min_samples: int = SPEI_DOY_MIN_SAMPLES,
    winsor_lo: float = SPEI_WINSOR_LO_PCT,
    winsor_hi: float = SPEI_WINSOR_HI_PCT,
    prob_clip: float = SPEI_PROB_CLIP,
    val_clip: float = SPEI_VALUE_CLIP,
) -> np.ndarray:
    """
    Vectorized: fit log-logistic for all pixels of a single DOY simultaneously and compute SPEI.

    Parameters
    ----------
    block : (n_times, H, W) float32  -- 90-day rolling D values across all historical years for this DOY

    Returns
    -------
    spei : (n_times, H, W) float32   -- corresponding SPEI values; pixels that cannot be fit are filled with NaN

    Algorithm
    ---------
    Estimate log-logistic parameters with the method of moments, fully vectorized:
      1. Winsor clip (using sort+index instead of nanpercentile, ~200x faster)
      2. Shift to positives: loc = min(D) - eps
      3. Compute mean/std of log(D - loc) -> c = pi / (std * sqrt(3)), scale = exp(mean)
      4. Vectorized fisk CDF -> ndtri(p) -> SPEI
    Accuracy is comparable to MLE (with 26 samples, L-moments is even more stable), and ~3 orders of magnitude faster.
    """
    from scipy.special import ndtri

    n, H, W = block.shape
    b64 = block.astype(np.float64, copy=False)

    # ---- Number of valid samples ----
    nvalid = np.sum(np.isfinite(b64), axis=0)  # (H, W)
    good = nvalid >= min_samples               # (H, W) bool

    # ---- Winsor (sort+index, ~200x faster than nanpercentile) ----
    b_sorted = np.sort(b64, axis=0)  # Sort NaNs to the tail
    lo_idx = max(0, int(np.floor(n * winsor_lo / 100.0)))
    hi_idx = min(n - 1, int(np.ceil(n * winsor_hi / 100.0)))
    lo = b_sorted[lo_idx]  # (H, W)
    hi = b_sorted[hi_idx]  # (H, W)

    b_w = np.clip(b64, lo[np.newaxis], hi[np.newaxis])

    # ---- Shift to positive values; loc = global minimum - eps ----
    loc = b_sorted[0] - 1e-3  # (H, W)
    shifted = b_w - loc[np.newaxis]
    shifted = np.where(np.isfinite(shifted) & (shifted > 1e-9), shifted, np.nan)

    log_s = np.where(shifted > 0, np.log(shifted), np.nan)

    # ---- Method-of-moments estimation ----
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu_L = np.nanmean(log_s, axis=0)    # (H, W)
        n_fin = np.sum(np.isfinite(log_s), axis=0)
        diff2 = (log_s - mu_L[np.newaxis]) ** 2
        std_L = np.sqrt(np.nansum(diff2, axis=0) / np.maximum(n_fin - 1, 1))

    c = np.where(
        good & (std_L > 0),
        np.pi / (std_L * np.sqrt(3) + 1e-12),
        np.nan,
    )
    scale = np.where(good, np.exp(mu_L), np.nan)
    loc_f = np.where(good, loc, np.nan)

    # ---- Vectorized CDF: fisk(c, loc, scale) ----
    x = np.clip(b64, lo[np.newaxis], hi[np.newaxis])
    x_sh = np.maximum(x - loc_f[np.newaxis], 1e-9)

    # log((x-loc)/scale); avoid log(0)
    log_ratio = np.log(x_sh / np.where(np.isfinite(scale), scale, 1.0)[np.newaxis] + 1e-300)
    val = c[np.newaxis] * log_ratio  # c·log((x-loc)/scale)

    # Numerically stable sigmoid
    p = np.where(
        val >= 0,
        1.0 / (1.0 + np.exp(-val)),
        np.exp(val) / (1.0 + np.exp(val)),
    )
    p = np.clip(p, prob_clip, 1.0 - prob_clip)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        spei = ndtri(p)

    spei = np.clip(spei, -val_clip, val_clip)

    # Invalid positions (invalid parameters or original D is NaN) -> NaN
    invalid = ~np.isfinite(c[np.newaxis]) | ~np.isfinite(b64)
    spei = np.where(invalid, np.nan, spei)

    return spei.astype(np.float32)


def _worker_limit_blas_threads() -> None:
    """Set BLAS/OpenMP to single-thread inside multiprocessing children to avoid oversubscription."""
    for k in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(k, "1")


# Read by SPEI second-pass ProcessPool children (injected via initializer)
_SPEI_DOY_ARRAY: Optional[np.ndarray] = None


def _spei_doy_pool_init(doy_arr: np.ndarray) -> None:
    global _SPEI_DOY_ARRAY
    _SPEI_DOY_ARRAY = doy_arr


def _spei_fit_one_doy_core(
    d: int,
    doy: np.ndarray,
    roll: np.ndarray,
    spei_mm: np.ndarray,
    H: int,
    W: int,
) -> None:
    """Single DOY: vectorized fit across all pixels and write into spei_mm[t,:,:]."""
    idx = np.where(doy == d)[0]
    if idx.size < SPEI_DOY_MIN_SAMPLES:
        return
    block = np.asarray(roll[idx], dtype=np.float32)  # (n_times, H, W)
    spei_vals = _spei_fit_doy_vectorized(block)       # (n_times, H, W)
    for k, t in enumerate(idx):
        spei_mm[t] = spei_vals[k]


def _spei_fit_one_doy_worker(entry: tuple) -> int:
    """Child process: open the memmap and process a single DOY d."""
    _worker_limit_blas_threads()
    d, path_roll, path_spei, N, H, W = entry
    doy = _SPEI_DOY_ARRAY
    roll = np.memmap(path_roll, dtype=np.float32, mode="r", shape=(N, H, W))
    spei_mm = np.memmap(path_spei, dtype=np.float32, mode="r+", shape=(N, H, W))
    try:
        _spei_fit_one_doy_core(d, doy, roll, spei_mm, H, W)
    finally:
        spei_mm.flush()
        del spei_mm
        del roll
    return d


def calculate_spei_3m(
    dates,
    tile_idx=0,
    show_progress=True,
    overwrite=False,
    spei_doy_workers: int = 1,
):
    """
    Daily-resolution SPEI-3M (modelling): 90-day rolling D = P - PET -> stratified by DOY, log-logistic standardized per pixel.
    Extreme-value control: Winsor-clip historical samples before fitting; clip output SPEI to +/- SPEI_VALUE_CLIP.
    Output: one GeoTIFF per day after the window fills; the first SPEI_ROLLING_DAYS-1 days have no full 90-day window, so no file is generated.

    show_progress: recommended False in multiprocess children to avoid tqdm interleaving.
    overwrite: when True, clear the SPEI scratch and SPEI outputs for this tile before computing (matches the top-level --overwrite).
    spei_doy_workers: number of processes used to parallelize DOYs in the second pass; > 1 is safe because each DOY writes a disjoint time slice t. 1 means serial.
    """
    import pandas as pd

    logger.info(
        f"Starting SPEI-3M computation (daily resolution / {SPEI_ROLLING_DAYS}d rolling / DOY), "
        f"Total {len(dates)} days, tile {tile_idx:02d}"
    )

    profile, shape = _get_first_valid_era5_shape(dates, tile_idx)
    if profile is None or shape is None:
        logger.error("No usable ERA5 raster; cannot compute SPEI-3M")
        return

    N = len(dates)
    H, W = shape
    dt = pd.to_datetime(dates)
    doy = dt.dayofyear.to_numpy(dtype=np.int32)

    scratch_dir = spei_scratch_dir(tile_idx)
    path_roll = os.path.join(scratch_dir, "roll90.dat")
    path_spei = os.path.join(scratch_dir, "spei_daily.dat")
    expected_memmap_bytes = N * H * W * np.dtype(np.float32).itemsize

    if overwrite:
        shutil.rmtree(scratch_dir, ignore_errors=True)
        for p in glob.glob(
            os.path.join(OUTPUT_DIR, "SPEI_3M", f"SPEI_3M_*_tile{tile_idx:02d}.tif")
        ):
            try:
                os.remove(p)
            except OSError:
                pass
    else:
        bad_scratch = False
        if os.path.isfile(path_roll) and os.path.getsize(path_roll) != expected_memmap_bytes:
            bad_scratch = True
        if os.path.isfile(path_spei) and os.path.getsize(path_spei) != expected_memmap_bytes:
            bad_scratch = True
        if bad_scratch:
            logger.warning(
                f"tile {tile_idx:02d} SPEI scratch file size differs from expected {expected_memmap_bytes} bytes; "
                f"treated as incomplete write; temp directory deleted"
            )
            shutil.rmtree(scratch_dir, ignore_errors=True)

    os.makedirs(scratch_dir, exist_ok=True)

    min_valid = max(1, int(np.ceil(SPEI_ROLLING_DAYS * SPEI_WINDOW_MIN_VALID_RATIO)))

    roll = np.memmap(path_roll, dtype=np.float32, mode="w+", shape=(N, H, W))
    roll[:] = np.nan

    logger.info("Pass 1: daily D and 90-day rolling accumulation (memmap)...")
    buf = deque(maxlen=SPEI_ROLLING_DAYS)
    roll_it = (
        tqdm(dates, desc=f"Rolling-accumulated D (tile{tile_idx:02d})")
        if show_progress
        else dates
    )
    n_dim_mismatch = 0
    n_water_balance_err = 0
    spei_skip_write_ti = set()
    for ti, date_str in enumerate(roll_it):
        data = read_era5_data(date_str, tile_idx)
        if data is None:
            D = np.full(shape, np.nan, dtype=np.float64)
        else:
            try:
                D = calculate_water_balance(
                    data["total_precipitation_sum"], data["potential_evaporation_sum"]
                )
            except ValueError as e:
                n_water_balance_err += 1
                if n_water_balance_err <= 5:
                    logger.warning(
                        f"tile {tile_idx:02d} {date_str}: water-balance skipped ({e})"
                    )
                D = np.full(shape, np.nan, dtype=np.float64)
        if D.shape != shape:
            n_dim_mismatch += 1
            logger.warning(
                f"tile {tile_idx:02d} {date_str} (ti={ti}): D shape {D.shape} and reference shape={shape} mismatch; "
                f"trying resampling alignment"
            )
            D_fixed = None
            if data is not None:
                D_fixed = _resample_D_to_reference_grid(
                    D, data["profile"], profile, shape
                )
            if D_fixed is not None and D_fixed.shape == shape:
                D = D_fixed
                logger.info(
                    f"tile {tile_idx:02d} {date_str}: resampling alignment succeeded -> {D.shape}"
                )
            else:
                logger.warning(
                    f"tile {tile_idx:02d} {date_str}: resampling failed; today's rolling is NaN and the SPEI GeoTIFF for the day is **not** written"
                )
                spei_skip_write_ti.add(ti)
                D = np.full(shape, np.nan, dtype=np.float64)
        buf.append(D.astype(np.float64, copy=False))
        if len(buf) < SPEI_ROLLING_DAYS:
            continue
        stack = np.stack(list(buf), axis=0)
        valid = np.isfinite(stack)
        cnt = np.sum(valid, axis=0)
        ssum = np.nansum(stack, axis=0)
        roll[ti] = np.where(cnt >= min_valid, ssum, np.nan).astype(np.float32)
    if n_dim_mismatch or n_water_balance_err or spei_skip_write_ti:
        logger.warning(
            f"tile {tile_idx:02d} rolling-accumulation summary: shape was inconsistent {n_dim_mismatch} days"
            f" (no SPEI tif rewrite {len(spei_skip_write_ti)} days), "
            f"water-balance anomaly {n_water_balance_err} days; reference grid {shape}"
        )
    roll.flush()
    logger.info(
        f"tile {tile_idx:02d} SPEI: first pass ({SPEI_ROLLING_DAYS}-day rolling memmap) complete."
        "The second-pass DOY x pixel fitting is usually the most time-consuming; **no** GeoTIFFs are written to SPEI_3M during this stage."
        "SPEI_3M files are only created during the third-pass export, so it is normal for scratch to have data while SPEI_3M is still empty."
    )

    spei_mm = np.memmap(path_spei, dtype=np.float32, mode="w+", shape=(N, H, W))
    spei_mm[:] = np.nan

    logger.info("Pass 2: fit log-logistic per DOY x pixel and write SPEI...")
    doy_max = int(np.max(doy))
    spei_doy_workers = max(1, int(spei_doy_workers))

    if spei_doy_workers <= 1:
        doy_it = (
            tqdm(range(1, doy_max + 1), desc=f"Standardize by DOY (tile{tile_idx:02d})")
            if show_progress
            else range(1, doy_max + 1)
        )
        for d in doy_it:
            _spei_fit_one_doy_core(d, doy, roll, spei_mm, H, W)
        spei_mm.flush()
    else:
        roll.flush()
        spei_mm.flush()
        del roll
        del spei_mm
        logger.info(
            f"tile {tile_idx:02d} SPEI second pass: {spei_doy_workers} processes parallelized by DOY"
            "(Each DOY writes a different time index t, so there are no write conflicts.)"
        )
        doy_np = np.ascontiguousarray(doy)
        with ProcessPoolExecutor(
            max_workers=spei_doy_workers,
            initializer=_spei_doy_pool_init,
            initargs=(doy_np,),
        ) as ex:
            futures = {
                ex.submit(
                    _spei_fit_one_doy_worker,
                    (d, path_roll, path_spei, N, H, W),
                ): d
                for d in range(1, doy_max + 1)
            }
            done = as_completed(futures)
            if show_progress:
                done = tqdm(
                    done,
                    total=len(futures),
                    desc=f"DOY-parallel spei tile{tile_idx:02d}",
                )
            for fut in done:
                fut.result()
        spei_mm = np.memmap(
            path_spei, dtype=np.float32, mode="r+", shape=(N, H, W)
        )
        spei_mm.flush()

    logger.info("Writing daily-scale SPEI GeoTIFFs...")
    warmup_skip = SPEI_ROLLING_DAYS - 1
    write_it = (
        tqdm(dates, desc=f"Write SPEI (tile{tile_idx:02d})")
        if show_progress
        else dates
    )
    for ti, date_str in enumerate(write_it):
        if ti < warmup_skip:
            continue
        output_path = os.path.join(
            OUTPUT_DIR,
            "SPEI_3M",
            f"SPEI_3M_{date_str.replace('-', '_')}_tile{tile_idx:02d}.tif",
        )
        if ti in spei_skip_write_ti:
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            logger.warning(
                f"tile {tile_idx:02d} {date_str}: skip writing SPEI GeoTIFF (raster alignment failed for this day)"
            )
            continue
        if os.path.exists(output_path):
            if overwrite:
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            elif is_readable_geotiff(output_path):
                continue
            else:
                try:
                    os.remove(output_path)
                    logger.warning(f"Delete corrupt SPEI output and rewrite: {output_path}")
                except OSError:
                    pass
        try:
            save_geotiff(
                np.asarray(spei_mm[ti], dtype=np.float32), profile, output_path
            )
        except Exception as e:
            logger.warning(
                f"tile {tile_idx:02d} {date_str}: failed to write SPEI ({e}); delete the partial output (if any)"
            )
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
            except OSError:
                pass

    if SPEI_DELETE_SCRATCH:
        try:
            del roll
            del spei_mm
            shutil.rmtree(scratch_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Failed to clean the SPEI temp directory: {e}")

    logger.info("SPEI-3M (daily resolution) computation complete")


def _climate_worker_task(payload: tuple) -> tuple:
    """
    Child-process task:
      - VPD: (kind, tile_idx, dates, overwrite, vpd_inner_workers, vpd_no_spatial)
      - SPEI: (kind, tile_idx, dates, overwrite, spei_doy_workers)
    Returns (kind, tile_idx, err_type_name_or_None, err_msg_or_None).
    """
    _worker_limit_blas_threads()
    kind = payload[0]
    tile_idx = payload[1]
    dates = payload[2]
    overwrite = payload[3]
    try:
        if kind == "vpd":
            inner_w = payload[4]
            no_spatial = payload[5]
            process_vpd_batch(
                dates,
                tile_idx,
                show_progress=False,
                overwrite=overwrite,
                inner_workers=inner_w,
                use_spatial_interp=not no_spatial,
            )
        elif kind == "spei":
            spei_dw = payload[4]
            calculate_spei_3m(
                dates,
                tile_idx,
                show_progress=False,
                overwrite=overwrite,
                spei_doy_workers=spei_dw,
            )
        else:
            raise ValueError(f"unknown job kind: {kind!r}")
        return (kind, tile_idx, None, None)
    except Exception as e:
        return (kind, tile_idx, type(e).__name__, str(e))


# ================== Main ==================

def main():
    parser = argparse.ArgumentParser(
        description="VPD + SPEI-3M: by default run VPD and SPEI for each tile in parallel processes (interpolation still uses griddata)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Upper bound on parallel processes; default: min(2 x number of tiles, logical CPU cores)",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Serial: per tile, first run full VPD then full SPEI (legacy behavior, useful for debugging)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="After clearing (per --compute scope), recompute everything; no corrupt-file detection",
    )
    parser.add_argument(
        "--compute",
        choices=("all", "vpd", "spei"),
        default="spei",
        help="Compute VPD only, SPEI-3M only, or both (default: all). Affects self-check/clearing and parallel tasks",
    )
    parser.add_argument(
        "--vpd-inner-workers",
        type=int,
        default=1,
        help="Number of processes used to parallelize VPD per day inside each tile (default: 1 = serial per day)."
        "Multiplies with the outer per-tile parallelism; mind the total process count and S3/disk IO; 4-8 is recommended",
    )
    parser.add_argument(
        "--vpd-no-spatial-interp",
        action="store_true",
        help="VPD: skip scipy.griddata; mark out-of-range values as NaN (much faster, suitable for already-smoothed ERA5)",
    )
    parser.add_argument(
        "--spei-doy-workers",
        type=int,
        default=None,
        help="SPEI second pass: number of processes used to parallelize DOYs inside each tile (default: auto = CPU / number of tiles, capped at 32);"
        "1 = serial. Multiplies with the outer multi-tile parallelism; total processes ~= tiles x this value",
    )
    args = parser.parse_args()

    import pandas as pd

    dates = pd.date_range(start=DATE_START, end=DATE_END).strftime("%Y-%m-%d").tolist()
    logger.info(f"Processing date range: {DATE_START} to {DATE_END}, total {len(dates)} days")

    sample_files = glob.glob(os.path.join(ERA5_DIR, "ERA5_*_tile*.tif"))
    if not sample_files:
        logger.error(f"ERA5 data file not found: {ERA5_DIR}")
        return

    tile_indices_set = set()
    for f in sample_files:
        basename = os.path.basename(f)
        parts = basename.split("_")
        if len(parts) >= 5:
            tile_part = parts[-1].replace(".tif", "")
            tile_idx = int(tile_part.replace("tile", ""))
            tile_indices_set.add(tile_idx)

    tile_indices = sorted(tile_indices_set)
    n_tiles = len(tile_indices)
    logger.info(f"Found {n_tiles} tiles: {tile_indices}")
    logger.info(f"Compute scope: --compute {args.compute}")
    if args.spei_doy_workers is None:
        spei_dw = max(1, min(32, (mp.cpu_count() or 8) // max(1, n_tiles)))
    else:
        spei_dw = max(1, int(args.spei_doy_workers))
    if args.compute in ("all", "vpd"):
        logger.info(
            f"VPD：inner_workers={args.vpd_inner_workers}，"
            f"Spatial interpolation={'off' if args.vpd_no_spatial_interp else 'on (griddata, slow)'}"
        )
    if args.compute in ("all", "spei"):
        logger.info(
            f"SPEI: per-tile DOY parallelism spei_doy_workers={spei_dw} "
            f" (running multiple tiles concurrently, total processes ~{n_tiles * spei_dw}; lower --spei-doy-workers or the outer --workers as needed)"
        )

    if args.overwrite:
        purge_tile_outputs_for_overwrite(tile_indices, args.compute)
        if args.compute == "all":
            msg = "VPD, SPEI_3M GeoTIFFs, and each tile's SPEI scratch"
        elif args.compute == "vpd":
            msg = "VPD GeoTIFFs only"
        else:
            msg = "SPEI_3M GeoTIFFs and each tile's SPEI scratch only"
        logger.info(f"--overwrite: deleted within range {msg}")
    else:
        n_bad = repair_corrupt_output_geotiffs(tile_indices, args.compute)
        logger.info(
            f"Output self-check ({args.compute}): deleted GeoTIFFs that cannot be fully read, total {n_bad} files; "
            f"afterwards only recompute missing or deleted dates"
        )

    if args.serial:
        for tile_idx in tile_indices:
            logger.info(f"\n{'='*60}\nProcessing tile {tile_idx:02d}\n{'='*60}")
            if args.compute in ("all", "vpd"):
                try:
                    process_vpd_batch(
                        dates,
                        tile_idx,
                        show_progress=True,
                        overwrite=args.overwrite,
                        inner_workers=max(1, args.vpd_inner_workers),
                        use_spatial_interp=not args.vpd_no_spatial_interp,
                    )
                    logger.info(f"✓ tile {tile_idx:02d} VPD computation complete")
                except Exception as e:
                    logger.error(f"✗ tile {tile_idx:02d} VPD computation failed: {e}")
            if args.compute in ("all", "spei"):
                try:
                    calculate_spei_3m(
                        dates,
                        tile_idx,
                        show_progress=True,
                        overwrite=args.overwrite,
                        spei_doy_workers=spei_dw,
                    )
                    logger.info(f"✓ tile {tile_idx:02d} SPEI-3M computation complete")
                except Exception as e:
                    logger.error(f"✗ tile {tile_idx:02d} SPEI-3M computation failed: {e}")
    else:
        cpu_n = mp.cpu_count() or 1
        n_kinds = 2 if args.compute == "all" else 1
        default_w = min(max(1, n_kinds * n_tiles), cpu_n)
        n_workers = max(1, args.workers if args.workers is not None else default_w)
        jobs: list = []
        if args.compute in ("all", "vpd"):
            for t in tile_indices:
                jobs.append(
                    (
                        "vpd",
                        t,
                        dates,
                        args.overwrite,
                        max(1, args.vpd_inner_workers),
                        args.vpd_no_spatial_interp,
                    )
                )
        if args.compute in ("all", "spei"):
            for t in tile_indices:
                jobs.append(("spei", t, dates, args.overwrite, spei_dw))
        if not jobs:
            logger.error("No tasks to run (check --compute)")
            return
        logger.info(
            f"Parallel scheduling {len(jobs)} tasks (--compute {args.compute}），"
            f"process count={n_workers}; child-process BLAS single-thread"
        )
        failures: list = []
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            future_map = {ex.submit(_climate_worker_task, job): job for job in jobs}
            for fut in as_completed(future_map):
                kind, tile_idx, err_t, err_m = fut.result()
                if err_t is not None:
                    logger.error(f"✗ [{kind}] tile {tile_idx:02d} failed: {err_t}: {err_m}")
                    failures.append((kind, tile_idx, err_t, err_m))
                else:
                    logger.info(f"✓ [{kind}] tile {tile_idx:02d} complete")
        if failures:
            logger.error(f"Total {len(failures)} tasks failed; scroll up for logs")

    logger.info(f"\n{'='*60}")
    logger.info("All computations complete!")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()

