#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute slope, aspect, hillshade, and (optionally) TPI and TWI.

- TPI (Topographic Position Index): Z - neighborhood mean over a (2*radius+1) pixel window; requires scipy.
- TWI (Topographic Wetness Index): ln(SCA / tan(beta)), where SCA is upstream contributing area (m^2) and beta is slope (radians).
  Flow routing uses pyflwdir: Wang & Liu pit-filling + D8; upstream area via FlwdirRaster.upstream_area('m2'); needs pip install pyflwdir.

**Value ranges (conceptual, not strict bounds)**

- **TPI**: same units as elevation (typically meters). Represents elevation above/below the neighborhood mean, with no fixed global bounds; commonly within +/- a few hundred meters,
  but can be larger on very steep terrain or at coarse resolutions. By default, finite values are clipped to [--tpi-clip-min, --tpi-clip-max] (default +/-5000) before writing,
  to suppress outliers and ease downstream normalization.
- **TWI**: dimensionless. Literature and applications most often see ~5-15; low-lying terrain, large contributing areas, or coarse rasters can produce noticeably higher values (e.g. 20+),
  while steep slopes near divides can be lower or even slightly negative. Finite values are clipped to [--twi-clip-min, --twi-clip-max] by default (-12 to 42),
  with +/-inf handled. Use --no-tpi-range-clip / --no-twi-range-clip to disable clipping.
"""

import argparse
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import Affine

try:
    from scipy.ndimage import uniform_filter
except ImportError:
    uniform_filter = None

def parse_args():
    ap = argparse.ArgumentParser("Units-safe slope/aspect/hillshade from DEM")
    ap.add_argument("--input-folder", default='/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/merit_dem', help="Input directory")
    ap.add_argument("--output-folder", default='/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/merit_dem/dem_slope_aspect_hillshade', help="Output directory")
    ap.add_argument("--pattern", default="Canada_Full_DEM_1km_MERIT.tif")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--band", type=int, default=1)
    ap.add_argument("--src-nodata", type=float, default=-32768)
    ap.add_argument("--dst-nodata", type=float, default=-32768)
    # If the DEM's vertical unit is not meters, use this to convert to meters (e.g. cm -> m: 0.01; ft -> m: 0.3048)
    ap.add_argument("--z-unit-scale", type=float, default=1.0,
                    help="Scale the DEM's vertical unit to meters (default 1.0 means already in meters)")
    ap.add_argument("--slope-unit", choices=["degree","percent"], default="degree")
    ap.add_argument("--azimuth", type=float, default=315.0)
    ap.add_argument("--altitude", type=float, default=45.0)
    ap.add_argument("--compress", choices=["NONE","LZW","DEFLATE","ZSTD"], default="ZSTD")
    ap.add_argument("--blocksize", type=int, default=512)
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing GeoTIFFs in the output directory (default: error and abort when an existing file is encountered)",
    )
    # TPI: neighborhood radius (pixels); window side = 2*radius + 1
    ap.add_argument("--tpi-radius", type=int, default=5,
                    help="TPI neighborhood half-width (pixels); window is (2r+1) x (2r+1); default 5 -> 11 x 11")
    ap.add_argument("--no-tpi", action="store_true", help="Do not output TPI")
    # TWI: requires pyflwdir (numba + numpy, supports Python 3.12)
    ap.add_argument("--no-twi", action="store_true", help="Do not output TWI")
    ap.add_argument(
        "--twi-min-slope-deg",
        type=float,
        default=0.05,
        help="TWI slope lower bound (degrees) to prevent tan(beta) -> 0 blowups; default 0.05 deg",
    )
    ap.add_argument(
        "--no-tpi-range-clip",
        action="store_true",
        help="Do not apply numeric range clipping to TPI (clipping is on by default)",
    )
    ap.add_argument(
        "--tpi-clip-min",
        type=float,
        default=-5000.0,
        help="TPI clip lower bound (meters, same as elevation units); default -5000",
    )
    ap.add_argument(
        "--tpi-clip-max",
        type=float,
        default=5000.0,
        help="TPI clip upper bound (meters); default 5000",
    )
    ap.add_argument(
        "--no-twi-range-clip",
        action="store_true",
        help="Do not apply numeric range clipping to TWI (clipping is on by default)",
    )
    ap.add_argument(
        "--twi-clip-min",
        type=float,
        default=-12.0,
        help="TWI clip lower bound (ln scale); default -12",
    )
    ap.add_argument(
        "--twi-clip-max",
        type=float,
        default=42.0,
        help="TWI clip upper bound; default 42 (raise as needed for large basins / coarse resolutions)",
    )
    args = ap.parse_args()
    if args.tpi_clip_min >= args.tpi_clip_max:
        ap.error(f"--tpi-clip-min must be less than --tpi-clip-max; current {args.tpi_clip_min} >= {args.tpi_clip_max}")
    if args.twi_clip_min >= args.twi_clip_max:
        ap.error(f"--twi-clip-min must be less than --twi-clip-max; current {args.twi_clip_min} >= {args.twi_clip_max}")
    return args


def clip_finite_to_range(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    Clamp: among finite values, replace those below lo with lo and those above hi with hi; leave nan unchanged; +inf -> hi, -inf -> lo.
    """
    out = np.asarray(arr, dtype=np.float32).copy()
    out[np.isposinf(out)] = hi
    out[np.isneginf(out)] = lo
    m = np.isfinite(out)
    out[m] = np.clip(out[m], lo, hi)
    return out


def is_dem(name: str) -> bool:
    n = name.lower()
    return ("dem" in n) and ("distance" not in n)

# Approximate "meters per degree" (varies with latitude)
def meters_per_degree_lat(lat_rad):
    # Common spherical approximation, sufficient for slope-scale work
    return (111132.954 - 559.822*np.cos(2*lat_rad) + 1.175*np.cos(4*lat_rad))

def meters_per_degree_lon(lat_rad):
    return (111412.84*np.cos(lat_rad) - 93.5*np.cos(3*lat_rad) + 0.118*np.cos(5*lat_rad))

def horn_gradients_unitsafe(dem, transform: Affine, nodata_val: float,
                            crs_is_geographic: bool, z_unit_scale: float):
    """
    Compute unit-safe dz/dx and dz/dy:
    - If the CRS is geographic (degrees), estimate per-row "deg -> m" pixel sizes (x and y separately) so the denominator is in meters, not degrees.
    - If the CRS is projected (meters), use the pixel size from transform directly (in meters).
    - The vertical unit is unified to meters via z_unit_scale (e.g. cm -> m: 0.01).
    - If any value in the 3 x 3 neighborhood is invalid, the output is NaN (later converted to NoData).
    """
    # Horn kernels
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float64)
    ky = np.array([[ 1,2,1],[ 0,0,0],[-1,-2,-1]], dtype=np.float64)

    h, w = dem.shape
    pad = 1
    dem_pad = np.pad(dem, pad, mode="edge")

    def shifted(arr, dy, dx):
        return arr[pad+dy:pad+dy+h, pad+dx:pad+dx+w]

    n00 = shifted(dem_pad,-1,-1); n01 = shifted(dem_pad,-1,0); n02 = shifted(dem_pad,-1,1)
    n10 = shifted(dem_pad, 0,-1); n11 = shifted(dem_pad, 0,0); n12 = shifted(dem_pad, 0,1)
    n20 = shifted(dem_pad, 1,-1); n21 = shifted(dem_pad, 1,0); n22 = shifted(dem_pad, 1,1)

    valid = (dem != nodata_val) & (~np.isnan(dem))
    valid_pad = np.pad(valid, pad, mode="edge")
    v00 = shifted(valid_pad,-1,-1); v01 = shifted(valid_pad,-1,0); v02 = shifted(valid_pad,-1,1)
    v10 = shifted(valid_pad, 0,-1); v11 = shifted(valid_pad, 0,0); v12 = shifted(valid_pad, 0,1)
    v20 = shifted(valid_pad, 1,-1); v21 = shifted(valid_pad, 1,0); v22 = shifted(valid_pad, 1,1)
    window_valid = v00 & v01 & v02 & v10 & v11 & v12 & v20 & v21 & v22

    gx_num = (n00*kx[0,0] + n01*kx[0,1] + n02*kx[0,2] +
              n10*kx[1,0] + n11*kx[1,1] + n12*kx[1,2] +
              n20*kx[2,0] + n21*kx[2,1] + n22*kx[2,2])

    gy_num = (n00*ky[0,0] + n01*ky[0,1] + n02*ky[0,2] +
              n10*ky[1,0] + n11*ky[1,1] + n12*ky[1,2] +
              n20*ky[2,0] + n21*ky[2,1] + n22*ky[2,2])

    # Horizontal pixel size (meters)
    if crs_is_geographic:
        # Compute the central latitude of each row from the affine (degrees -> radians)
        # Row y center latitude (degrees) = y0 + (row + 0.5) * pixelHeight; note pixelHeight is typically negative
        rows = np.arange(h, dtype=np.float64)
        # Image-to-geographic: lat = f + (row + 0.5) * e; e is typically negative; we take the center latitude
        lat_deg_center = transform.f + (rows + 0.5) * transform.e
        lat_rad_center = np.deg2rad(lat_deg_center)

        m_per_deg_x = meters_per_degree_lon(lat_rad_center)      # shape (h,)
        m_per_deg_y = meters_per_degree_lat(lat_rad_center)      # shape (h,)

        xres_deg = abs(transform.a)
        yres_deg = abs(transform.e)

        xres_m_row = xres_deg * m_per_deg_x                      # shape (h,)
        yres_m_row = yres_deg * m_per_deg_y                      # shape (h,)
    else:
        xres_m_row = np.full(h, abs(transform.a), dtype=np.float64)
        yres_m_row = np.full(h, abs(transform.e), dtype=np.float64)

    # Horn denominator: 8 * cellsize (meters); also multiplied by z_unit_scale (to convert the vertical unit to meters)
    denom_x = (8.0 * xres_m_row * (1.0 / z_unit_scale))  # Equivalent to: convert dz (in vertical units) to meters, then divide by meters
    denom_y = (8.0 * yres_m_row * (1.0 / z_unit_scale))

    # Row-wise broadcasted division
    gx = gx_num / denom_x[:, None]
    gy = gy_num / denom_y[:, None]

    gx = np.where(window_valid, gx, np.nan)
    gy = np.where(window_valid, gy, np.nan)
    return gx, gy


def compute_tpi(
    dem: np.ndarray,
    nodata_val: float,
    radius_px: int,
) -> np.ndarray:
    """
    TPI = elevation - rectangular neighborhood mean; invalid pixels (nodata or too few valid neighbors) output as nan.
    """
    if uniform_filter is None:
        raise ImportError("TPI requires scipy: pip install scipy")
    if radius_px < 1:
        raise ValueError("--tpi-radius must be >= 1")
    k = 2 * radius_px + 1
    dem64 = dem.astype(np.float64, copy=False)
    valid = (dem64 != nodata_val) & np.isfinite(dem64)
    dem_z = np.where(valid, dem64, 0.0)
    sum_z = uniform_filter(dem_z, size=k, mode="nearest")
    # uniform_filter on a 0/1 mask gives the average ratio of valid neighbors (in [0, 1]), not a pixel count
    frac_valid = uniform_filter(valid.astype(np.float64), size=k, mode="nearest")
    mean_z = np.divide(
        sum_z,
        frac_valid,
        out=np.full_like(sum_z, np.nan),
        where=frac_valid > 0,
    )
    tpi = dem64 - mean_z
    tpi[~valid] = np.nan
    # If less than 25% of neighbors are valid, treat as unreliable and mark nan
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
    """
    TWI = ln(SCA / tan(beta)), where SCA is upstream contributing area (m^2).
    DEM is Wang & Liu pit-filled, then D8 flow direction is derived, and pyflwdir computes upstream_area(unit='m2').
    """
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
    bad = ~np.isfinite(dem) | (d == nodata_val) | ~np.isfinite(slope_rad)
    twi[bad] = np.nan
    return twi.astype(np.float32)


def slope_aspect_hillshade_unitsafe(dem, transform, nodata_val, crs_is_geographic,
                                    z_unit_scale, slope_unit, azimuth_deg, altitude_deg):
    gx, gy = horn_gradients_unitsafe(dem, transform, nodata_val, crs_is_geographic, z_unit_scale)

    slope_rad = np.arctan(np.hypot(gx, gy))
    if slope_unit == "degree":
        slope_out = np.degrees(slope_rad).astype(np.float32)
    else:
        slope_out = (np.tan(slope_rad) * 100.0).astype(np.float32)

    aspect_rad = np.arctan2(gy, -gx)
    aspect_rad = np.where(aspect_rad < 0, aspect_rad + 2*np.pi, aspect_rad)
    aspect_out = np.degrees(aspect_rad).astype(np.float32)

    az = np.deg2rad(azimuth_deg)
    alt = np.deg2rad(altitude_deg)
    zen = np.pi/2.0 - alt
    hill = 255.0 * (np.cos(zen)*np.cos(slope_rad) + np.sin(zen)*np.sin(slope_rad)*np.cos(az - aspect_rad))
    hill = np.clip(hill, 0.0, 255.0).astype(np.float32)

    nan_mask = ~np.isfinite(slope_out) | ~np.isfinite(aspect_out) | ~np.isfinite(hill)
    slope_out[nan_mask]  = np.float32(nodata_val)
    aspect_out[nan_mask] = np.float32(nodata_val)
    hill[nan_mask]       = np.float32(nodata_val)
    # slope_rad feeds TWI; invalid locations are set to nan to prevent misuse in ln(SCA/tan beta)
    slope_rad_out = slope_rad.astype(np.float64)
    slope_rad_out[nan_mask] = np.nan
    return slope_out, aspect_out, hill, slope_rad_out

def write_gtiff(path: Path, arr, tpl_profile, compress="ZSTD", blocksize=512, nodata=None, overwrite=False):
    prof = tpl_profile.copy()
    prof.update(dtype="float32", count=1, nodata=nodata, tiled=True,
                blockxsize=blocksize, blockysize=blocksize)
    if compress != "NONE":
        prof.update(compress=compress)
        if compress == "ZSTD":
            prof.update(zstd_level=9)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise FileExistsError(
                f"Output already exists: {path}  (use --overwrite to overwrite)"
            )
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr, 1)

def main():
    args = parse_args()
    in_dir = Path(args.input_folder)
    out_dir = Path(args.output_folder); out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.rglob(args.pattern) if args.recursive else in_dir.glob(args.pattern))
    if not files:
        print("No matching images found"); return

    total=ok=skip=0
    for fp in files:
        if not is_dem(fp.name):
            skip += 1; continue
        total += 1
        try:
            with rasterio.open(fp) as src:
                dem = src.read(args.band).astype(np.float64, copy=False)
                if src.nodata is not None:
                    dem[np.isclose(dem, src.nodata, equal_nan=True)] = args.src_nodata
                dem[np.isnan(dem)] = args.src_nodata

                crs_is_geo = (src.crs is not None and src.crs.is_geographic)
                slope, aspect, hill, slope_rad = slope_aspect_hillshade_unitsafe(
                    dem=dem,
                    transform=src.transform,
                    nodata_val=args.src_nodata,
                    crs_is_geographic=crs_is_geo,
                    z_unit_scale=args.z_unit_scale,
                    slope_unit=args.slope_unit,
                    azimuth_deg=args.azimuth,
                    altitude_deg=args.altitude
                )

                # Replace the internal nodata marker with the target nodata
                slope[slope == args.src_nodata]   = args.dst_nodata
                aspect[aspect == args.src_nodata] = args.dst_nodata
                hill[hill == args.src_nodata]     = args.dst_nodata

                rel = fp.relative_to(in_dir) if fp.is_relative_to(in_dir) else Path(fp.name)
                base = fp.stem
                prof = src.profile.copy(); prof.update(nodata=args.dst_nodata, dtype="float32")

                out_slope  = out_dir / rel.parent / f"{base}_slope_{args.slope_unit}.tif"
                out_aspect = out_dir / rel.parent / f"{base}_aspect.tif"
                out_hill   = out_dir / rel.parent / f"{base}_hillshade.tif"
                out_slope.parent.mkdir(parents=True, exist_ok=True)

                write_gtiff(out_slope,  slope,  prof, args.compress, args.blocksize, args.dst_nodata, args.overwrite)
                write_gtiff(out_aspect, aspect, prof, args.compress, args.blocksize, args.dst_nodata, args.overwrite)
                write_gtiff(out_hill,   hill,   prof, args.compress, args.blocksize, args.dst_nodata, args.overwrite)

                extras = []
                if not args.no_tpi:
                    try:
                        tpi = compute_tpi(dem, args.src_nodata, args.tpi_radius)
                        if not args.no_tpi_range_clip:
                            tpi = clip_finite_to_range(
                                tpi, args.tpi_clip_min, args.tpi_clip_max
                            )
                        tpi[~np.isfinite(tpi)] = args.dst_nodata
                        out_tpi = out_dir / rel.parent / f"{base}_tpi_r{args.tpi_radius}.tif"
                        write_gtiff(out_tpi, tpi, prof, args.compress, args.blocksize, args.dst_nodata, args.overwrite)
                        extras.append("tpi")
                    except Exception as e:
                        print(f"[TPI skipped] {fp} | {e}")

                if not args.no_twi:
                    try:
                        twi = compute_twi_pyflwdir(
                            dem,
                            args.src_nodata,
                            slope_rad,
                            src.transform,
                            crs_is_geo,
                            args.twi_min_slope_deg,
                        )
                        if not args.no_twi_range_clip:
                            twi = clip_finite_to_range(
                                twi, args.twi_clip_min, args.twi_clip_max
                            )
                        twi[~np.isfinite(twi)] = args.dst_nodata
                        out_twi = out_dir / rel.parent / f"{base}_twi.tif"
                        write_gtiff(out_twi, twi, prof, args.compress, args.blocksize, args.dst_nodata, args.overwrite)
                        extras.append("twi")
                    except ImportError:
                        print(
                            f"[TWI skipped] pyflwdir not installed: pip install pyflwdir  ({fp.name}）",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"[TWI skipped] {fp} | {e}")

                ok += 1
                extra_s = ("+" + "/".join(extras)) if extras else ""
                print(f"[OK] {fp} -> slope/aspect/hillshade{extra_s}")
        except FileExistsError as e:
            print(f"[FAIL] {fp}\n       {e}")
        except Exception as e:
            print(f"[FAIL] {fp} | {e}")

    print(f"Done. Processed: {ok} | skipped: {skip} | total: {total}")

if __name__ == "__main__":
    main()
