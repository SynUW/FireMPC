"""Vectorized Van Wagner (1987) Canadian Forest Fire Weather Index (FWI) equations.

All functions operate on 2D numpy arrays covering the full Canada ERA5 grid (443x226).
Unit conventions (CRITICAL - ERA5 values must be converted before calling FWI functions):
  Temperature: degrees Celsius (ERA5 Band 13 is Kelvin - subtract 273.15)
  Relative humidity: percent 0-100 (derived from Magnus/Tetens formula)
  Wind speed: km/h (ERA5 Bands 2,3 are m/s - multiply sqrt(u2+v2) by 3.6)
  Precipitation: mm (ERA5 Band 5 is metres - multiply by 1000)

References:
  Van Wagner, C.E. (1987). Development and Structure of the Canadian Forest Fire
  Weather Index System. Canadian Forestry Service Technical Report 35.

Startup convention (CWFIS standard):
  April 1 each year: FFMC=85.0, DMC=6.0, DC=15.0
  Fire season: April 1 - October 31 (months 4-10 inclusive)
  Pre-season values: set to NaN (not computed)
"""
import numpy as np
from osgeo import gdal
import os
from datetime import date, timedelta

# CWFIS canonical spring startup values (Van Wagner 1987 / CWFIS methodology)
FFMC_INIT: float = 85.0
DMC_INIT:  float = 6.0
DC_INIT:   float = 15.0


# ---------------------------------------------------------------------------
# ERA5 unit conversion and input extraction
# ---------------------------------------------------------------------------

def extract_fwi_inputs_from_bands(bands: dict) -> tuple:
    """Convert ERA5 band arrays to FWI-ready units.

    Args:
        bands: dict mapping 1-based band index (int) to 2D numpy array.
               Required keys: 2, 3, 5, 7, 13.

    Returns:
        (T_c, RH, wind, rain) - all 2D float32 arrays, same shape as input bands.
        T_c:   temperature in degrees Celsius
        RH:    relative humidity in percent [0, 100]
        wind:  wind speed in km/h
        rain:  24h precipitation in mm
    """
    T_max_k = bands[13].astype(np.float32)   # temperature_2m_max, Kelvin
    Td_k    = bands[7].astype(np.float32)    # dewpoint_temperature_2m, Kelvin
    u10     = bands[2].astype(np.float32)    # u_component_of_wind_10m, m/s
    v10     = bands[3].astype(np.float32)    # v_component_of_wind_10m, m/s
    precip  = bands[5].astype(np.float32)    # total_precipitation_sum, metres

    T_c  = T_max_k - 273.15                       # Kelvin -> Celsius
    Td_c = Td_k - 273.15                          # Kelvin -> Celsius

    # Magnus/Tetens RH formula
    RH = 100.0 * (
        np.exp(17.625 * Td_c / (243.04 + Td_c)) /
        np.exp(17.625 * T_c  / (243.04 + T_c))
    )
    RH = np.clip(RH, 0.0, 100.0).astype(np.float32)

    wind = np.sqrt(u10**2 + v10**2) * 3.6           # m/s -> km/h
    rain = precip * 1000.0                           # metres -> mm

    # Sanity assertions (catch unit errors early)
    assert float(np.nanmin(T_c)) > -80 and float(np.nanmax(T_c)) < 60, \
        f"T_c out of plausible range: min={np.nanmin(T_c):.1f}, max={np.nanmax(T_c):.1f}. Check Kelvin conversion."
    assert float(np.nanmin(rain)) >= 0, \
        f"Negative precipitation: min={np.nanmin(rain):.4f}. Check metre->mm conversion."

    return T_c, RH, wind, rain


def read_era5_tile(tile_path: str, bands: list = None) -> dict:
    """Read ERA5 GeoTIFF tile and return dict of band arrays.

    Args:
        tile_path: path to 18-band ERA5 daily GeoTIFF tile.
        bands: list of 1-based band indices to read. None = read all 18.

    Returns:
        dict mapping band index (int) to 2D float32 array with NaN for nodata.
    """
    ds = gdal.Open(tile_path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open ERA5 tile: {tile_path}")
    n_bands = ds.RasterCount
    band_list = bands if bands is not None else list(range(1, n_bands + 1))
    result = {}
    for b in band_list:
        assert b >= 1, f"Band index must be >= 1 (GDAL convention), got {b}"
        band_obj = ds.GetRasterBand(b)
        arr = band_obj.ReadAsArray().astype(np.float32)
        nodata = band_obj.GetNoDataValue()
        if nodata is not None:
            arr[arr == nodata] = np.nan
        result[b] = arr
    ds = None
    return result


# ---------------------------------------------------------------------------
# FWI moisture code equations (Van Wagner 1987)
# ---------------------------------------------------------------------------

def ffmc_next(ffmc_prev: np.ndarray, temp: np.ndarray, rh: np.ndarray,
              wind: np.ndarray, rain: np.ndarray) -> np.ndarray:
    """Fine Fuel Moisture Code (FFMC) - one time step.

    Args:
        ffmc_prev: previous day FFMC (2D array). April 1 = FFMC_INIT (85.0).
        temp:  temperature in degrees C (2D array)
        rh:    relative humidity in % (2D array, 0-100)
        wind:  wind speed in km/h (2D array)
        rain:  24h precipitation in mm (2D array)

    Returns:
        ffmc_new: FFMC clipped to [0, 101] (2D float32 array)
    """
    ffmc_prev = ffmc_prev.astype(np.float32)
    # Previous moisture content
    mo = 147.2 * (101.0 - ffmc_prev) / (59.5 + ffmc_prev)

    # Effect of rain on fine fuel moisture
    rf = np.where(
        rain > 0.5,
        mo + 42.5 * rain * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rain)),
        mo
    )
    rf = np.clip(rf, 0.0, 250.0)

    # Equilibrium moisture contents
    ed = (0.942 * rh**0.679
          + 11.0 * np.exp((rh - 100.0) / 10.0)
          + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh)))
    ew = (0.618 * rh**0.753
          + 10.0 * np.exp((rh - 100.0) / 10.0)
          + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh)))

    # Drying and wetting rates
    ko = (0.424 * (1.0 - (rh / 100.0)**1.7)
          + 0.0694 * np.sqrt(wind) * (1.0 - (rh / 100.0)**8))
    kd = ko * 0.581 * np.exp(0.0365 * temp)

    kl = (0.424 * (1.0 - ((100.0 - rh) / 100.0)**1.7)
          + 0.0694 * np.sqrt(wind) * (1.0 - ((100.0 - rh) / 100.0)**8))
    kw = kl * 0.581 * np.exp(0.0365 * temp)

    m = np.where(rf > ed,
                 ed + (rf - ed) * np.exp(-kd),
                 np.where(rf < ew,
                          ew - (ew - rf) * np.exp(-kw),
                          rf))

    ffmc_new = 59.5 * (250.0 - m) / (147.2 + m)
    return np.clip(ffmc_new, 0.0, 101.0).astype(np.float32)


def dmc_next(dmc_prev: np.ndarray, temp: np.ndarray, rh: np.ndarray,
             rain: np.ndarray, month: int) -> np.ndarray:
    """Duff Moisture Code (DMC) - one time step.

    Args:
        dmc_prev: previous day DMC (2D array). April 1 = DMC_INIT (6.0).
        temp:  temperature in degrees C
        rh:    relative humidity in %
        rain:  24h precipitation in mm
        month: calendar month (1-12)

    Returns:
        dmc_new: DMC >= 0 (2D float32 array)
    """
    dmc_prev = np.maximum(dmc_prev.astype(np.float32), 0.0)

    # Day-length factor Le by month (Van Wagner 1987, Table 1)
    Le_table = {4: 0.9, 5: 12.5, 6: 13.9, 7: 13.9, 8: 12.4, 9: 10.9, 10: 9.4}
    Le = Le_table.get(month, 9.0)

    # Rain effect
    re = np.where(rain > 1.5, 0.92 * rain - 1.27, 0.0)
    mo = 20.0 + np.exp(5.6348 - dmc_prev / 43.43)
    b = np.where(
        dmc_prev <= 33,
        100.0 / (0.5 + 0.3 * dmc_prev),
        np.where(dmc_prev <= 65,
                 14.0 - 1.3 * np.log(dmc_prev),
                 6.2 * np.log(dmc_prev) - 17.2)
    )
    mr = mo + 1000 * re / (48.77 + b * re)
    pr = 244.72 - 43.43 * np.log(mr - 20.0)
    pr = np.where(rain > 1.5, np.maximum(pr, 0.0), dmc_prev)

    # Drying
    k = np.where(
        temp > -1.1,
        1.894 * (temp + 1.1) * (100.0 - rh) * Le * 1e-6,
        0.0
    )
    dmc_new = pr + 100.0 * k
    return np.maximum(dmc_new, 0.0).astype(np.float32)


def dc_next(dc_prev: np.ndarray, temp: np.ndarray, rain: np.ndarray,
            month: int) -> np.ndarray:
    """Drought Code (DC) - one time step.

    Args:
        dc_prev: previous day DC (2D array). April 1 = DC_INIT (15.0).
        temp:  temperature in degrees C
        rain:  24h precipitation in mm
        month: calendar month (1-12)

    Returns:
        dc_new: DC >= 0 (2D float32 array)
    """
    dc_prev = np.maximum(dc_prev.astype(np.float32), 0.0)

    # Day-length adjustment Lf by month (Van Wagner 1987, Table 2)
    Lf_table = {4: -1.6, 5: 1.2, 6: 4.0, 7: 4.0, 8: 3.0, 9: 0.4, 10: -2.0}
    Lf = Lf_table.get(month, -1.6)

    # Rain effect
    rd = np.where(rain > 2.8, 0.83 * rain - 1.27, 0.0)
    qo = 800.0 * np.exp(-dc_prev / 400.0)
    qr = qo + 3.937 * rd
    dr = np.where(rain > 2.8,
                  400.0 * np.log(800.0 / qr),
                  dc_prev)
    dr = np.maximum(dr, 0.0)

    # Potential evapotranspiration
    v = np.where(temp > -2.8,
                 0.36 * (temp + 2.8) + Lf,
                 0.0)
    v = np.maximum(v, 0.0)
    dc_new = dr + 0.5 * v
    return np.maximum(dc_new, 0.0).astype(np.float32)


def isi_calc(ffmc: np.ndarray, wind: np.ndarray) -> np.ndarray:
    """Initial Spread Index (ISI) from FFMC and wind speed."""
    fm = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    fw = np.exp(0.05039 * wind)
    ff = 91.9 * np.exp(-0.1386 * fm) * (1.0 + fm**5.31 / 4.93e7)
    return (0.208 * fw * ff).astype(np.float32)


def bui_calc(dmc: np.ndarray, dc: np.ndarray) -> np.ndarray:
    """Buildup Index (BUI) from DMC and DC."""
    den = dmc + 0.4 * dc
    bui = np.where(
        den < 1e-6,
        0.0,
        np.where(
            dmc <= 0.4 * dc,
            0.8 * dmc * dc / den,
            dmc - (1.0 - 0.8 * dc / den) * (0.92 + (0.0114 * dmc) ** 1.7),
        ),
    )
    return np.maximum(bui, 0.0).astype(np.float32)


def fwi_calc(isi: np.ndarray, bui: np.ndarray) -> np.ndarray:
    """Fire Weather Index (FWI) from ISI and BUI."""
    bb = 0.1 * isi * (0.626 * bui**0.809 + 2.0)
    fwi = np.where(
        bb > 1.0,
        np.exp(2.72 * (0.434 * np.log(np.maximum(bb, 1e-6)))**0.647),
        bb
    )
    return np.maximum(fwi, 0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Full fire season computation
# ---------------------------------------------------------------------------

def compute_fwi_season(era5_tile_dir: str, year: int,
                       grid_shape: tuple = (443, 226)) -> dict:
    """Compute FWI components for one fire season (April-October).

    Processes daily ERA5 GeoTIFF tiles sequentially. Missing tiles are handled
    by carrying forward the previous day's moisture codes (nearest-neighbor
    imputation) and recording the missing date.

    Args:
        era5_tile_dir: directory containing daily ERA5 GeoTIFF tiles named
                       YYYYMMDD.tif (e.g., 20180601.tif).
        year:          calendar year (e.g., 2018).
        grid_shape:    (H, W) of output arrays. Default: full Canada grid.

    Returns:
        dict with keys: 'ffmc', 'dmc', 'dc', 'isi', 'bui', 'fwi', 'dates', 'missing_days'
        Each component value is a list of 2D arrays (one per season day, ~183 days Apr-Oct).
        'dates': list of date strings (YYYY-MM-DD) for each array position.
        'missing_days': list of date strings for days with missing ERA5 tiles.
    """
    ffmc = np.full(grid_shape, FFMC_INIT, dtype=np.float32)
    dmc  = np.full(grid_shape, DMC_INIT,  dtype=np.float32)
    dc   = np.full(grid_shape, DC_INIT,   dtype=np.float32)

    outputs = {k: [] for k in ['ffmc', 'dmc', 'dc', 'isi', 'bui', 'fwi']}
    dates = []
    missing_days = []

    season_start = date(year, 4, 1)
    season_end   = date(year, 10, 31)
    current_day  = season_start

    while current_day <= season_end:
        tile_path = os.path.join(era5_tile_dir, f"{current_day.strftime('%Y%m%d')}.tif")
        date_str  = current_day.strftime('%Y-%m-%d')
        dates.append(date_str)

        if not os.path.exists(tile_path):
            # Missing tile: carry forward previous values (NaN for first day if missing)
            missing_days.append(date_str)
            for k in ['ffmc', 'dmc', 'dc', 'isi', 'bui', 'fwi']:
                outputs[k].append(np.full(grid_shape, np.nan, dtype=np.float32))
            current_day += timedelta(days=1)
            continue

        bands  = read_era5_tile(tile_path, bands=[2, 3, 5, 7, 13])
        T_c, RH, wind, rain = extract_fwi_inputs_from_bands(bands)
        month  = current_day.month

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

        current_day += timedelta(days=1)

    outputs['dates']        = dates
    outputs['missing_days'] = missing_days
    return outputs
