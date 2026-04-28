#!/usr/bin/env python3
"""
Compute terrain factors from a DEM directory and stack them with the original DEM into a multi-band output:
- Input: single-band DEM (any GTiff), batch-processed by directory.
- Compute: Slope (degrees), Aspect sin/cos (derived from GDAL's aspect-degrees, in [-1, 1]), Hillshade (0-255).
- Optional TPI / TWI (consistent with trash/dems_cal): TPI = Z - neighborhood mean (scipy); TWI = ln(SCA / tan β), where SCA is pyflwdir upstream area (m^2) and β is Horn slope (radians).
- Output: 5- or 7-band GeoTIFF. Default 7 bands: Bands 1-5 as above; Band 6 = TPI; Band 7 = TWI.
  Pass both --no-tpi and --no-twi to output only 5 bands.
- Flat regions (GDAL's -zero_for_flat = -1) and invalid-aspect locations have sin/cos set to NoData.
- External mask (optional): enabled when --mask-dir or --mask-file is specified; aligned strictly by absolute pixel position and must match DEM size. If neither is specified, no masking is applied.
- Projection / resolution / extent preserved; the same NoData is set on every band (the DEM's NoData or --default-nodata).
- Computed via GDAL DEMProcessing, with LZW compression, tiling, BIGTIFF support; parallel processing.
- Dependencies: TPI requires scipy; TWI requires pyflwdir (and affine, usually installed alongside pyflwdir).
"""

import os
import glob
import logging
import argparse
from typing import List, Tuple, Optional, Any, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from affine import Affine
from osgeo import gdal, osr

try:
    from scipy.ndimage import uniform_filter
except ImportError:
    uniform_filter = None

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('dem_to_terrain')

def list_tifs(folder: str, pattern: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, pattern)))


def read_info(ds) -> Tuple[int, int, int, str, tuple, int]:
    w = ds.RasterXSize
    h = ds.RasterYSize
    b = ds.RasterCount
    prj = ds.GetProjection()
    gt = ds.GetGeoTransform()
    dtype = ds.GetRasterBand(1).DataType
    return w, h, b, prj, gt, dtype


def demprocess_to_array(src_ds, mode: str, options: List[str]) -> np.ndarray:
    vsipath = f"/vsimem/demproc_{mode}_{id(src_ds)}.tif"
    gdal.DEMProcessing(vsipath, src_ds, mode, options=options)
    out_ds = gdal.Open(vsipath, gdal.GA_ReadOnly)
    if out_ds is None:
        raise RuntimeError(f"DEMProcessing failed: {mode}")
    arr = out_ds.GetRasterBand(1).ReadAsArray()
    out_ds = None
    gdal.Unlink(vsipath)
    return arr


def pick_mask_source(mask_dir: Optional[str], mask_pattern: str, mask_file: Optional[str]) -> Optional[str]:
    if mask_file:
        if os.path.isfile(mask_file):
            return mask_file
        else:
            raise FileNotFoundError(f"Specified mask file does not exist: {mask_file}")
    if mask_dir:
        cands = list_tifs(mask_dir, mask_pattern)
        if cands:
            logger.info(f"Using mask reference: {cands[0]}")
            return cands[0]
        else:
            raise FileNotFoundError(f"No matching TIFs in the mask directory: {mask_dir}/{mask_pattern}")
    return None


def read_mask_array_strict(mask_src_path: str, dem_width: int, dem_height: int) -> np.ndarray:
    """Read the mask array, strictly requiring the same size as the DEM; no interpolation or resampling."""
    mds = gdal.Open(mask_src_path, gdal.GA_ReadOnly)
    if mds is None:
        raise FileNotFoundError(f"Cannot open mask source: {mask_src_path}")
    mw, mh = mds.RasterXSize, mds.RasterYSize
    if (mw, mh) != (dem_width, dem_height):
        mds = None
        raise ValueError(f"Mask size ({mw}x{mh}) and DEM size ({dem_width}x{dem_height}) mismatch; resampling / interpolation is disabled.")
    marr = mds.GetRasterBand(1).ReadAsArray()
    mnd = mds.GetRasterBand(1).GetNoDataValue()
    mds = None
    return marr, mnd


def is_geographic_wkt(wkt: str) -> bool:
    try:
        srs = osr.SpatialReference()
        srs.ImportFromWkt(wkt)
        return bool(srs.IsGeographic())
    except Exception:
        return False


def compute_center_lat_from_gt(gt: tuple, width: int, height: int) -> float:
    center_x_pix = width / 2.0
    center_y_pix = height / 2.0
    center_y = gt[3] + center_x_pix * gt[4] + center_y_pix * gt[5]
    return float(center_y)


def compute_auto_z_factor(projection: str, gt: tuple, width: int, height: int) -> float:
    if not is_geographic_wkt(projection):
        return 1.0
    lat = compute_center_lat_from_gt(gt, width, height)
    # Rough along-meridian m/deg (latitude-dependent) approximation: 111320 * cos(lat)
    meters_per_degree = 111320.0 * np.cos(np.deg2rad(lat))
    # Safety net: avoid zero in edge cases
    if meters_per_degree <= 0 or not np.isfinite(meters_per_degree):
        meters_per_degree = 111320.0
    return float(meters_per_degree)


def clip_finite_to_range(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clamp: clip finite values to [lo, hi]; leave nan unchanged; +inf -> hi, -inf -> lo."""
    out = np.asarray(arr, dtype=np.float32).copy()
    out[np.isposinf(out)] = hi
    out[np.isneginf(out)] = lo
    m = np.isfinite(out)
    out[m] = np.clip(out[m], lo, hi)
    return out


def meters_per_degree_lat(lat_rad: np.ndarray) -> np.ndarray:
    return 111132.954 - 559.822 * np.cos(2 * lat_rad) + 1.175 * np.cos(4 * lat_rad)


def meters_per_degree_lon(lat_rad: np.ndarray) -> np.ndarray:
    return 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad) + 0.118 * np.cos(5 * lat_rad)


def horn_gradients_unitsafe(
    dem: np.ndarray,
    transform: Affine,
    nodata_val: float,
    crs_is_geographic: bool,
    z_unit_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    h, w = dem.shape
    pad = 1
    dem_pad = np.pad(dem, pad, mode="edge")

    def shifted(arr, dy, dx):
        return arr[pad + dy : pad + dy + h, pad + dx : pad + dx + w]

    n00 = shifted(dem_pad, -1, -1)
    n01 = shifted(dem_pad, -1, 0)
    n02 = shifted(dem_pad, -1, 1)
    n10 = shifted(dem_pad, 0, -1)
    n11 = shifted(dem_pad, 0, 0)
    n12 = shifted(dem_pad, 0, 1)
    n20 = shifted(dem_pad, 1, -1)
    n21 = shifted(dem_pad, 1, 0)
    n22 = shifted(dem_pad, 1, 1)
    valid = (dem != nodata_val) & (~np.isnan(dem))
    valid_pad = np.pad(valid, pad, mode="edge")
    v00 = shifted(valid_pad, -1, -1)
    v01 = shifted(valid_pad, -1, 0)
    v02 = shifted(valid_pad, -1, 1)
    v10 = shifted(valid_pad, 0, -1)
    v11 = shifted(valid_pad, 0, 0)
    v12 = shifted(valid_pad, 0, 1)
    v20 = shifted(valid_pad, 1, -1)
    v21 = shifted(valid_pad, 1, 0)
    v22 = shifted(valid_pad, 1, 1)
    window_valid = v00 & v01 & v02 & v10 & v11 & v12 & v20 & v21 & v22
    gx_num = (
        n00 * kx[0, 0]
        + n01 * kx[0, 1]
        + n02 * kx[0, 2]
        + n10 * kx[1, 0]
        + n11 * kx[1, 1]
        + n12 * kx[1, 2]
        + n20 * kx[2, 0]
        + n21 * kx[2, 1]
        + n22 * kx[2, 2]
    )
    gy_num = (
        n00 * ky[0, 0]
        + n01 * ky[0, 1]
        + n02 * ky[0, 2]
        + n10 * ky[1, 0]
        + n11 * ky[1, 1]
        + n12 * ky[1, 2]
        + n20 * ky[2, 0]
        + n21 * ky[2, 1]
        + n22 * ky[2, 2]
    )
    if crs_is_geographic:
        rows = np.arange(h, dtype=np.float64)
        lat_deg_center = transform.f + (rows + 0.5) * transform.e
        lat_rad_center = np.deg2rad(lat_deg_center)
        xres_deg = abs(transform.a)
        yres_deg = abs(transform.e)
        xres_m_row = xres_deg * meters_per_degree_lon(lat_rad_center)
        yres_m_row = yres_deg * meters_per_degree_lat(lat_rad_center)
    else:
        xres_m_row = np.full(h, abs(transform.a), dtype=np.float64)
        yres_m_row = np.full(h, abs(transform.e), dtype=np.float64)
    denom_x = 8.0 * xres_m_row * (1.0 / z_unit_scale)
    denom_y = 8.0 * yres_m_row * (1.0 / z_unit_scale)
    gx = gx_num / denom_x[:, None]
    gy = gy_num / denom_y[:, None]
    gx = np.where(window_valid, gx, np.nan)
    gy = np.where(window_valid, gy, np.nan)
    return gx, gy


def compute_tpi(dem: np.ndarray, nodata_val: float, radius_px: int) -> np.ndarray:
    if uniform_filter is None:
        raise ImportError("TPI requires scipy: pip install scipy")
    if radius_px < 1:
        raise ValueError("tpi-radius must be >= 1")
    k = 2 * radius_px + 1
    dem64 = dem.astype(np.float64, copy=False)
    valid = (dem64 != nodata_val) & np.isfinite(dem64)
    dem_z = np.where(valid, dem64, 0.0)
    sum_z = uniform_filter(dem_z, size=k, mode="nearest")
    frac_valid = uniform_filter(valid.astype(np.float64), size=k, mode="nearest")
    mean_z = np.divide(
        sum_z,
        frac_valid,
        out=np.full_like(sum_z, np.nan),
        where=frac_valid > 0,
    )
    tpi = dem64 - mean_z
    tpi[~valid] = np.nan
    tpi[frac_valid < 0.25] = np.nan
    return tpi.astype(np.float32)


def compute_twi_pyflwdir(
    dem: np.ndarray,
    nodata_val: float,
    slope_rad: np.ndarray,
    transform: Affine,
    crs_is_geographic: bool,
    min_slope_deg: float,
) -> np.ndarray:
    import pyflwdir

    d = np.ascontiguousarray(dem.astype(np.float64))
    flw = pyflwdir.from_dem(
        d,
        nodata=nodata_val,
        transform=transform,
        latlon=crs_is_geographic,
    )
    sca = flw.upstream_area(unit="m2")
    min_r = np.deg2rad(min_slope_deg)
    tanb = np.tan(np.maximum(slope_rad, min_r))
    tanb = np.maximum(tanb, 1e-8)
    twi = np.log(np.maximum(sca / tanb, 1e-30))
    # pyflwdir sets uparea to -9999 outside the mask; combined with nan tanb this produces nan, so we mark them all as invalid
    bad = (
        ~np.isfinite(d)
        | (d == nodata_val)
        | ~np.isfinite(slope_rad)
        | ~np.isfinite(sca)
        | (sca <= 0)
    )
    twi[bad] = np.nan
    return twi.astype(np.float32)


def process_one(
    src_path: str,
    dst_path: str,
    azimuth: float,
    altitude: float,
    z_factor: float,
    default_nodata: float,
    mask_src: Optional[str],
    mask_nodata_arg: Optional[float],
    mask_zero_invalid: bool,
    topo_extra: Dict[str, Any],
) -> Tuple[str, bool]:
    try:
        ds = gdal.Open(src_path, gdal.GA_ReadOnly)
        if ds is None:
            logger.error(f"Cannot open file: {src_path}")
            return src_path, False
        width, height, bands, projection, geotransform, dtype = read_info(ds)
        if bands < 1:
            logger.error(f"Insufficient band count: {src_path}")
            return src_path, False
        band1 = ds.GetRasterBand(1)
        dem_nodata = band1.GetNoDataValue()
        if dem_nodata is None:
            dem_nodata = default_nodata

        # Auto z-factor: for geographic coords (degrees), approximate-correct the horizontal unit by BC central latitude
        z_auto = compute_auto_z_factor(projection, geotransform, width, height)
        z_use = z_factor if (z_factor and z_factor > 0) else z_auto

        slope = demprocess_to_array(
            ds,
            'slope',
            ['-compute_edges', '-s', str(z_use), '-of', 'GTiff', '-p']
        ).astype(np.float32)
        aspect = demprocess_to_array(
            ds,
            'aspect',
            ['-compute_edges', '-s', str(z_use), '-zero_for_flat', '-of', 'GTiff']
        ).astype(np.float32)
        hillshade = demprocess_to_array(
            ds,
            'hillshade',
            ['-compute_edges', '-s', str(z_use), '-az', str(azimuth), '-alt', str(altitude), '-of', 'GTiff']
        ).astype(np.float32)

        dem = band1.ReadAsArray().astype(np.float32)

        if dem_nodata is not None:
            dem_valid = (dem != dem_nodata)
        else:
            dem_valid = np.ones(dem.shape, dtype=bool)

        if mask_src:
            mask_arr, mask_nd_from_file = read_mask_array_strict(mask_src, width, height)
            mask_nd = mask_nodata_arg if (mask_nodata_arg is not None) else mask_nd_from_file
            if mask_nd is not None:
                valid_mask = (mask_arr != mask_nd)
            else:
                valid_mask = np.ones(mask_arr.shape, dtype=bool)
            if mask_zero_invalid:
                valid_mask &= (mask_arr != 0)
            dem_valid &= valid_mask

        dem[~dem_valid] = dem_nodata
        slope[~dem_valid] = dem_nodata
        hillshade[~dem_valid] = dem_nodata

        # Aspect (degrees) -> sin/cos; output NoData for flat (-zero_for_flat = -1) and non-finite locations
        aspect_deg = aspect.astype(np.float64, copy=False)
        valid_aspect = (
            dem_valid
            & np.isfinite(aspect_deg)
            & (aspect_deg >= 0.0)
            & (aspect_deg <= 360.0)
        )
        theta = np.deg2rad(aspect_deg)
        aspect_sin = np.full(aspect.shape, dem_nodata, dtype=np.float32)
        aspect_cos = np.full(aspect.shape, dem_nodata, dtype=np.float32)
        aspect_sin[valid_aspect] = np.sin(theta[valid_aspect]).astype(np.float32)
        aspect_cos[valid_aspect] = np.cos(theta[valid_aspect]).astype(np.float32)

        write_extra = bool(topo_extra.get("write_extra", False))
        n_bands = 7 if write_extra else 5
        tpi_out: Optional[np.ndarray] = None
        twi_out: Optional[np.ndarray] = None
        if write_extra:
            crs_geo = is_geographic_wkt(projection)
            aff = Affine.from_gdal(*geotransform)
            nd_f = float(dem_nodata)
            dem64 = dem.astype(np.float64, copy=False)
            gx, gy = horn_gradients_unitsafe(
                dem64, aff, nd_f, crs_geo, float(topo_extra["z_unit_scale"])
            )
            slope_rad = np.arctan(np.hypot(gx, gy))
            slope_rad[~dem_valid] = np.nan

            if topo_extra["enable_tpi"]:
                try:
                    tpi_out = compute_tpi(dem64, nd_f, int(topo_extra["tpi_radius"]))
                    if not topo_extra["no_tpi_range_clip"]:
                        tpi_out = clip_finite_to_range(
                            tpi_out,
                            float(topo_extra["tpi_clip_min"]),
                            float(topo_extra["tpi_clip_max"]),
                        )
                    tpi_out = tpi_out.copy()
                    tpi_out[~np.isfinite(tpi_out)] = dem_nodata
                    tpi_out[~dem_valid] = dem_nodata
                except Exception as e:
                    logger.warning("TPI skipped %s: %s", src_path, e)
                    tpi_out = np.full(dem.shape, dem_nodata, dtype=np.float32)
            else:
                tpi_out = np.full(dem.shape, dem_nodata, dtype=np.float32)

            if topo_extra["enable_twi"]:
                try:
                    twi_out = compute_twi_pyflwdir(
                        dem64,
                        nd_f,
                        slope_rad,
                        aff,
                        crs_geo,
                        float(topo_extra["twi_min_slope_deg"]),
                    )
                    if not topo_extra["no_twi_range_clip"]:
                        twi_out = clip_finite_to_range(
                            twi_out,
                            float(topo_extra["twi_clip_min"]),
                            float(topo_extra["twi_clip_max"]),
                        )
                    twi_out = twi_out.copy()
                    twi_out[~np.isfinite(twi_out)] = dem_nodata
                    twi_out[~dem_valid] = dem_nodata
                except ImportError:
                    logger.warning("TWI skipped (pyflwdir not installed): %s", src_path)
                    twi_out = np.full(dem.shape, dem_nodata, dtype=np.float32)
                except Exception as e:
                    logger.warning("TWI skipped %s: %s", src_path, e)
                    twi_out = np.full(dem.shape, dem_nodata, dtype=np.float32)
            else:
                twi_out = np.full(dem.shape, dem_nodata, dtype=np.float32)

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(
            dst_path, width, height, n_bands, gdal.GDT_Float32,
            options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256']
        )
        out_ds.SetProjection(projection)
        out_ds.SetGeoTransform(geotransform)
        for b in range(1, n_bands + 1):
            out_ds.GetRasterBand(b).SetNoDataValue(dem_nodata)

        out_ds.GetRasterBand(1).WriteArray(dem)
        out_ds.GetRasterBand(2).WriteArray(slope)
        out_ds.GetRasterBand(3).WriteArray(aspect_sin)
        out_ds.GetRasterBand(4).WriteArray(aspect_cos)
        out_ds.GetRasterBand(5).WriteArray(hillshade)
        out_ds.GetRasterBand(1).SetDescription("DEM")
        out_ds.GetRasterBand(2).SetDescription("Slope_deg")
        out_ds.GetRasterBand(3).SetDescription("Aspect_sin")
        out_ds.GetRasterBand(4).SetDescription("Aspect_cos")
        out_ds.GetRasterBand(5).SetDescription("Hillshade")
        if write_extra and tpi_out is not None and twi_out is not None:
            out_ds.GetRasterBand(6).WriteArray(tpi_out)
            out_ds.GetRasterBand(7).WriteArray(twi_out)
            out_ds.GetRasterBand(6).SetDescription("TPI")
            out_ds.GetRasterBand(7).SetDescription("TWI")
        out_ds.FlushCache()

        ds = None
        out_ds = None
        return src_path, True
    except Exception as e:
        logger.error(f"Processing failed {src_path}: {e}")
        return src_path, False


def main():
    ap = argparse.ArgumentParser(
        description='Compute DEM Slope, Aspect (sin/cos), Hillshade, optionally TPI/TWI; write a 5- or 7-band GeoTIFF (supports external mask, aligned by absolute pixel position with no interpolation)'
    )
    ap.add_argument('--input-dir', default='/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/merit_dem', help='Input DEM directory')
    ap.add_argument('--output-dir', default='/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/merit_dem/dem_slope_aspect_hillshade', help='Output directory (will be created)')
    ap.add_argument('--pattern', default='*.tif', help='Input match pattern, default *.tif')
    ap.add_argument('--azimuth', type=float, default=315.0, help='Hillshade azimuth (degrees)')
    ap.add_argument('--altitude', type=float, default=35.0, help='Hillshade altitude (degrees); for BC mid-high latitudes a slightly lower value is recommended (e.g. 35 deg)')
    ap.add_argument('--z-factor', type=float, default=0.0, help='Vertical scale factor (Z scale). <= 0 auto-computes m/deg from the central latitude for geographic CRS')
    ap.add_argument('--default-nodata', type=float, default=-9999.0, help='NoData used when the DEM has none')
    ap.add_argument(
        '--mask-dir',
        default=None,
        help='Mask reference directory (any TIF inside it will be picked). Not specified by default; no masking if neither this nor --mask-file is given',
    )
    ap.add_argument('--mask-pattern', default='*.tif', help='Mask directory match pattern')
    ap.add_argument('--mask-file', default=None, help='Explicit mask reference TIF (takes precedence over the directory)')
    ap.add_argument('--mask-nodata', type=float, default=None, help='NoData value of the mask reference (specifiable when the mask file does not define one)')
    ap.add_argument('--mask-zero-invalid', action='store_true', help='When the mask has no NoData or it is not specified, treat pixel value 0 as invalid')
    ap.add_argument('--max-workers', type=int, default=16, help='Number of parallel processes')
    ap.add_argument(
        '--z-unit-scale',
        type=float,
        default=1.0,
        help='DEM vertical-unit conversion to meters (Horn slope used for TWI; default 1.0)',
    )
    ap.add_argument('--tpi-radius', type=int, default=5, help='TPI neighborhood half-width (pixels); window is (2r+1)^2')
    ap.add_argument('--no-tpi', action='store_true', help='Do not compute TPI (in 7-band mode, Band 6 is filled with NoData)')
    ap.add_argument('--no-twi', action='store_true', help='Do not compute TWI (in 7-band mode, Band 7 is filled with NoData)')
    ap.add_argument('--twi-min-slope-deg', type=float, default=0.05, help='TWI slope lower bound (degrees)')
    ap.add_argument('--no-tpi-range-clip', action='store_true', help='Do not range-clamp TPI')
    ap.add_argument('--tpi-clip-min', type=float, default=-5000.0)
    ap.add_argument('--tpi-clip-max', type=float, default=5000.0)
    ap.add_argument('--no-twi-range-clip', action='store_true', help='Do not range-clamp TWI')
    ap.add_argument('--twi-clip-min', type=float, default=-12.0)
    ap.add_argument('--twi-clip-max', type=float, default=42.0)

    args = ap.parse_args()
    if args.tpi_clip_min >= args.tpi_clip_max:
        ap.error('--tpi-clip-min must be less than --tpi-clip-max')
    if args.twi_clip_min >= args.twi_clip_max:
        ap.error('--twi-clip-min must be less than --twi-clip-max')

    tifs = list_tifs(args.input_dir, args.pattern)
    if not tifs:
        logger.error(f"No TIFs in input directory: {args.input_dir}")
        return

    mask_src = pick_mask_source(args.mask_dir, args.mask_pattern, args.mask_file) if (args.mask_dir or args.mask_file) else None

    topo_extra: Dict[str, Any] = {
        "write_extra": not (args.no_tpi and args.no_twi),
        "enable_tpi": not args.no_tpi,
        "enable_twi": not args.no_twi,
        "tpi_radius": args.tpi_radius,
        "z_unit_scale": args.z_unit_scale,
        "twi_min_slope_deg": args.twi_min_slope_deg,
        "no_tpi_range_clip": args.no_tpi_range_clip,
        "tpi_clip_min": args.tpi_clip_min,
        "tpi_clip_max": args.tpi_clip_max,
        "no_twi_range_clip": args.no_twi_range_clip,
        "twi_clip_min": args.twi_clip_min,
        "twi_clip_max": args.twi_clip_max,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    if topo_extra["write_extra"]:
        logger.info(
            "Will output 7 bands (Band 6 = TPI, Band 7 = TWI). For only 5 bands, pass --no-tpi --no-twi"
        )

    total = len(tifs)
    done = 0
    ok_cnt = 0

    def progress():
        print(f"\rProgress {done}/{total} success: {ok_cnt}", end='', flush=True)

    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = []
        for src in tifs:
            dst = os.path.join(args.output_dir, os.path.basename(src))
            futs.append(
                ex.submit(
                    process_one,
                    src,
                    dst,
                    args.azimuth,
                    args.altitude,
                    args.z_factor,
                    args.default_nodata,
                    mask_src,
                    args.mask_nodata,
                    args.mask_zero_invalid,
                    topo_extra,
                )
            )
        for f in as_completed(futs):
            _, ok = f.result()
            done += 1
            if ok:
                ok_cnt += 1
            progress()

    print()
    logger.info(f"Done: {ok_cnt}/{total} files succeeded")


if __name__ == '__main__':
    main()