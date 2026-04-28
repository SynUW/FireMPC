#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TIF file masking tool.
Use a template file's spatial valid range to mask TIFs in the target directory.
The fill value in the template's invalid regions and the per-band NoData metadata in the output
are both derived from each input TIF's GetNoDataValue; the template's NoData is no longer used as the fill.

--warp-to-template is on by default: when the input grid/projection differs from the template (e.g. global WorldPop against a local MCD template),
first use GDAL Warp to resample to the template's resolution and extent, then apply the mask pixel-wise. Use --no-warp-to-template to require an exactly matching grid.
"""

import os
import sys
import glob
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from osgeo import gdal
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _resample_alg_from_name(name: str):
    m = {
        "nearest": gdal.GRA_NearestNeighbour,
        "bilinear": gdal.GRA_Bilinear,
        "cubic": gdal.GRA_Cubic,
        "cubicspline": gdal.GRA_CubicSpline,
        "average": gdal.GRA_Average,
        "mode": gdal.GRA_Mode,
    }
    key = (name or "bilinear").lower().replace("-", "")
    if key not in m:
        raise ValueError(f"Unknown resampling: {name}; valid options {sorted(m)}")
    return m[key]


def warp_dataset_to_template(src_path: str, template_path: str, resample_alg) -> tuple:
    """
    Reproject / resample src to match the template's pixel size, affine, and projection.
    Returns (data, geotransform, projection, nodata) consistent with read_tiff_with_metadata; (None, ...) on failure.
    """
    tpl = gdal.Open(template_path, gdal.GA_ReadOnly)
    if tpl is None:
        logger.error(f"Cannot open template: {template_path}")
        return None, None, None, None
    tw, th = tpl.RasterXSize, tpl.RasterYSize
    tgt = tpl.GetGeoTransform()
    tproj = tpl.GetProjection()
    minx = tgt[0]
    maxy = tgt[3]
    maxx = tgt[0] + tw * tgt[1]
    miny = tgt[3] + th * tgt[5]

    src_ds = gdal.Open(src_path, gdal.GA_ReadOnly)
    if src_ds is None:
        logger.error(f"Cannot open input: {src_path}")
        tpl = None
        return None, None, None, None

    n_b = src_ds.RasterCount
    src_nodata = [src_ds.GetRasterBand(i + 1).GetNoDataValue() for i in range(n_b)]
    # GDAL Warp: keep dstNodata aligned with src so that downstream masking still uses the input semantics for invalid values
    kwargs = dict(
        format="MEM",
        width=tw,
        height=th,
        outputBounds=(minx, miny, maxx, maxy),
        resampleAlg=resample_alg,
    )
    if tproj:
        kwargs["dstSRS"] = tproj
    if any(v is not None for v in src_nodata):
        kwargs["srcNodata"] = src_nodata[0] if n_b == 1 else src_nodata
        kwargs["dstNodata"] = src_nodata[0] if n_b == 1 else src_nodata

    try:
        wo = gdal.WarpOptions(**kwargs)
        dst = gdal.Warp("", src_ds, options=wo)
    finally:
        src_ds = None
        tpl = None

    if dst is None:
        logger.error(f"Warp failed: {src_path}")
        return None, None, None, None

    try:
        nb = dst.RasterCount
        if nb == 1:
            data = dst.GetRasterBand(1).ReadAsArray()
            nodata = dst.GetRasterBand(1).GetNoDataValue()
        else:
            data = np.stack(
                [dst.GetRasterBand(i + 1).ReadAsArray() for i in range(nb)], axis=0
            )
            nodata = [dst.GetRasterBand(i + 1).GetNoDataValue() for i in range(nb)]
        gt = dst.GetGeoTransform()
        proj = dst.GetProjection()
        return data, gt, proj, nodata
    finally:
        dst = None


def read_tiff_with_metadata(file_path):
    """
    Read a TIF file and return data, geotransform, projection, and NoData value.
    """
    try:
        ds = gdal.Open(file_path, gdal.GA_ReadOnly)
        if ds is None:
            logger.error(f"Cannot open file: {file_path}")
            return None, None, None, None
        
        # Get basic info
        width = ds.RasterXSize
        height = ds.RasterYSize
        bands = ds.RasterCount
        
        # Get geotransform and projection
        geotransform = ds.GetGeoTransform()
        projection = ds.GetProjection()
        
        # Read data
        if bands == 1:
            data = ds.GetRasterBand(1).ReadAsArray()
            nodata = ds.GetRasterBand(1).GetNoDataValue()
        else:
            data = np.array([ds.GetRasterBand(i+1).ReadAsArray() for i in range(bands)])
            nodata = [ds.GetRasterBand(i+1).GetNoDataValue() for i in range(bands)]
        
        ds = None
        return data, geotransform, projection, nodata
        
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        return None, None, None, None

def get_template_mask_and_nodata(template_file):
    """
    Retrieve NoData value and spatial mask from a template file.
    """
    logger.info(f"Get NoData value and spatial mask from template: {template_file}")
    
    data, _, _, nodata = read_tiff_with_metadata(template_file)
    if data is None:
        logger.error("Cannot read template file")
        return None, None, None
    
    # Handle multi-band data
    if data.ndim == 3:
        # Multi-band: use the first band to determine the mask
        band_data = data[0]
        band_nodata = nodata[0] if isinstance(nodata, list) else nodata
    else:
        # Single band
        band_data = data
        band_nodata = nodata
    
    # Build the valid mask (non-NoData and finite values)
    valid_mask = np.isfinite(band_data)
    if band_nodata is not None:
        valid_mask &= (band_data != band_nodata)
    
    # Verbose debug info
    logger.info(f"Template file data shape: {band_data.shape}")
    logger.info(f"Template file dtype: {band_data.dtype}")
    logger.info(f"Template file NoData value: {band_nodata}")
    logger.info(f"Template file data range: {np.nanmin(band_data):.3f} ~ {np.nanmax(band_data):.3f}")
    logger.info(f"Template file finite value count: {np.sum(np.isfinite(band_data))}")
    if band_nodata is not None:
        logger.info(f"Template file NoData value count: {np.sum(band_data == band_nodata)}")
    logger.info(f"Template file valid pixel count: {np.sum(valid_mask)} / {valid_mask.size} ({100*np.sum(valid_mask)/valid_mask.size:.1f}%)")
    
    if not np.any(valid_mask):
        logger.error("Template file has no valid data")
        return None, None, None
    
    return valid_mask, band_nodata, data


def _per_band_nodata(nodata, band_index: int):
    """Pick a single band's NoData from read_tiff_with_metadata's nodata (scalar or list)."""
    if nodata is None:
        return None
    if isinstance(nodata, (list, tuple)):
        if band_index < len(nodata):
            return nodata[band_index]
        return None
    return nodata


def _fill_masked_pixels(masked_data, template_mask, nodata_per_input):
    """
    Set pixels in the template's invalid regions to each band's NoData read from the input TIF (not a template-specified value).
    nodata_per_input: scalar or per-band, consistent with read_tiff_with_metadata.
    """
    if masked_data.ndim == 3:
        n_bands = masked_data.shape[0]
        for i in range(n_bands):
            nd = _per_band_nodata(nodata_per_input, i)
            if nd is None:
                if np.issubdtype(masked_data.dtype, np.floating):
                    masked_data[i][~template_mask] = np.nan
                else:
                    raise ValueError(
                        f"Band {i + 1} has no NoData set in the GeoTIFF; cannot fill invalid regions with a scalar (integer types do not support default NaN)"
                    )
            else:
                masked_data[i][~template_mask] = nd
    else:
        nd = _per_band_nodata(nodata_per_input, 0)
        if nd is None:
            if np.issubdtype(masked_data.dtype, np.floating):
                masked_data[~template_mask] = np.nan
            else:
                raise ValueError(
                    "Single-band GeoTIFF has no NoData set; cannot fill invalid regions with a scalar (integer types do not support default NaN)"
                )
        else:
            masked_data[~template_mask] = nd


def apply_template_mask_to_tiff(
    input_file,
    output_file,
    template_mask,
    template_gt,
    template_proj,
    *,
    warp_to_template: bool,
    template_path: str | None,
    resample_alg,
):
    """
    Mask a TIF using the template file's spatial mask.
    With warp_to_template=True, first warp the input onto the template grid, then apply the mask.
    """
    try:
        if warp_to_template and template_path:
            data, geotransform, projection, nodata = warp_dataset_to_template(
                input_file, template_path, resample_alg
            )
            if data is None:
                return False
            # Output strictly uses the template's affine and projection, to avoid post-Warp float / mask pixel misalignment
            geotransform = template_gt
            projection = template_proj
        else:
            data, geotransform, projection, nodata = read_tiff_with_metadata(input_file)
            if data is None:
                return False

            if not np.allclose(geotransform, template_gt, rtol=1e-6, atol=1e-9):
                logger.warning(
                    f"Geotransform mismatch (default alignment is available: drop --no-warp-to-template): {input_file}"
                )
                return False

            if projection != template_proj:
                logger.warning(f"Projection mismatch: {input_file}")
                return False

        if data.ndim == 3:
            if data.shape[1:] != template_mask.shape:
                logger.warning(
                    f"Data size mismatch: {input_file} {data.shape[1:]} vs {template_mask.shape}"
                )
                return False
        else:
            if data.shape != template_mask.shape:
                logger.warning(
                    f"Data size mismatch: {input_file} {data.shape} vs {template_mask.shape}"
                )
                return False

        # Invalid-region fill value and output metadata use each input band's own NoData (GetNoDataValue), not the template's nodata
        masked_data = data.copy()
        try:
            _fill_masked_pixels(masked_data, template_mask, nodata)
        except ValueError as e:
            logger.error(f"{input_file}: {e}")
            return False

        # Write the output file with per-band NoData matching the input
        write_tiff_with_metadata(output_file, masked_data, geotransform, projection, nodata)
        return True
        
    except Exception as e:
        logger.error(f"Failed to process file {input_file}: {e}")
        return False

def write_tiff_with_metadata(output_file, data, geotransform, projection, nodata):
    """
    Write data to a TIF file.
    """
    try:
        # Create output directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Determine data type
        if data.dtype == np.float32 or data.dtype == np.float64:
            dtype = gdal.GDT_Float32
        elif data.dtype == np.uint8:
            dtype = gdal.GDT_Byte
        elif data.dtype == np.uint16:
            dtype = gdal.GDT_UInt16
        elif data.dtype == np.int16:
            dtype = gdal.GDT_Int16
        else:
            dtype = gdal.GDT_Float32
        
        # Create dataset
        if data.ndim == 3:
            bands, height, width = data.shape
        else:
            bands = 1
            height, width = data.shape
            data = data[np.newaxis, ...]
        
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(output_file, width, height, bands, dtype)
        
        if ds is None:
            logger.error(f"Cannot create output file: {output_file}")
            return False
        
        # Set georeferencing
        ds.SetGeoTransform(geotransform)
        ds.SetProjection(projection)
        
        # Write data
        for i in range(bands):
            band = ds.GetRasterBand(i + 1)
            band.WriteArray(data[i])
            
            # Set NoData value
            if isinstance(nodata, list) and i < len(nodata):
                if nodata[i] is not None:
                    band.SetNoDataValue(nodata[i])
            elif not isinstance(nodata, list) and nodata is not None:
                band.SetNoDataValue(nodata)
        
        ds = None
        return True
        
    except Exception as e:
        logger.error(f"Failed to write file {output_file}: {e}")
        return False

def process_single_file(args_tuple):
    """
    Function to process a single file (used by multithreading).
    """
    (
        input_file,
        output_file,
        template_mask,
        template_gt,
        template_proj,
        template_path,
        warp_to_template,
        resample_alg,
    ) = args_tuple

    success = apply_template_mask_to_tiff(
        input_file,
        output_file,
        template_mask,
        template_gt,
        template_proj,
        warp_to_template=warp_to_template,
        template_path=template_path,
        resample_alg=resample_alg,
    )
    
    if success:
        return input_file, True, None
    else:
        return input_file, False, f"Processing failed: {input_file}"

def main():
    parser = argparse.ArgumentParser(description='Use the template file\'s valid-value range to mask TIF files')
    parser.add_argument('--template', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/MCD12Q1_mosaic/2019_01_01.tif', help='Template TIF file path')
    parser.add_argument('--input_dir', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/worldpop/v2', help='Input TIF file directory')
    parser.add_argument('--output_dir', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/worldpop_v2_clipped', help='Output directory for masked files')
    parser.add_argument('--pattern', default='*.tif', help='File match pattern (default: *.tif)')
    parser.add_argument('--recursive', action='store_true', help='Search subdirectories recursively')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads (default: 4)')
    parser.add_argument(
        '--no-warp-to-template',
        action='store_true',
        help='Disable Warp: require pixel-aligned input with the template projection (by default, resample onto the template grid)',
    )
    parser.add_argument(
        '--resample',
        type=str,
        default='bilinear',
        choices=['nearest', 'bilinear', 'cubic', 'cubicspline', 'average', 'mode'],
        help='Resampling when warping to the template (bilinear is recommended for continuous rasters like population; default bilinear)',
    )

    args = parser.parse_args()
    warp_to_template = not args.no_warp_to_template
    try:
        resample_alg = _resample_alg_from_name(args.resample)
    except ValueError as e:
        logger.error(e)
        return
    
    # Check template file
    if not os.path.exists(args.template):
        logger.error(f"Template file does not exist: {args.template}")
        return
    
    # Check input directory
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get the template file's mask and NoData value
    template_mask, _, _ = get_template_mask_and_nodata(args.template)
    if template_mask is None:
        logger.error("Cannot retrieve mask info from the template file")
        return
    
    # Get the template file's georeferencing
    _, template_gt, template_proj, _ = read_tiff_with_metadata(args.template)
    if template_gt is None:
        logger.error("Cannot retrieve georeferencing from the template file")
        return
    
    # Find input files
    if args.recursive:
        pattern = os.path.join(args.input_dir, '**', args.pattern)
        input_files = glob.glob(pattern, recursive=True)
    else:
        pattern = os.path.join(args.input_dir, args.pattern)
        input_files = glob.glob(pattern)
    
    if not input_files:
        logger.error(f"In {args.input_dir}: no match found {args.pattern} files")
        return
    
    logger.info(f"Found {len(input_files)} input files")
    logger.info(f"Using {args.threads} threads for processing")
    if warp_to_template:
        logger.info("Warp -> template grid is enabled (compatible with non-aligned inputs like WorldPop)")
    else:
        logger.info("Warp is disabled: input must match the template's GeoTransform / projection / size")
    
    # Prepare multithreading parameters
    file_args = []
    for input_file in input_files:
        # Build the output file path
        rel_path = os.path.relpath(input_file, args.input_dir)
        output_file = os.path.join(args.output_dir, rel_path)
        
        # Create the output file's directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        file_args.append(
            (
                input_file,
                output_file,
                template_mask,
                template_gt,
                template_proj,
                args.template,
                warp_to_template,
                resample_alg,
            )
        )

    success_count = 0
    failed_files = []
    executor = ThreadPoolExecutor(max_workers=args.threads)
    interrupted = False
    try:
        future_to_file = {
            executor.submit(process_single_file, file_arg): file_arg[0]
            for file_arg in file_args
        }
        with tqdm(total=len(input_files), desc="Process files") as pbar:
            for future in as_completed(future_to_file):
                input_file, success, error_msg = future.result()

                if success:
                    success_count += 1
                else:
                    failed_files.append(input_file)
                    logger.warning(error_msg)

                pbar.update(1)
                pbar.set_postfix({
                    'success': success_count,
                    'failed': len(failed_files),
                    'progress': f"{success_count + len(failed_files)}/{len(input_files)}",
                })
    except KeyboardInterrupt:
        interrupted = True
        logger.warning("User interrupted; cancelling unfinished tasks...")
    finally:
        try:
            executor.shutdown(wait=not interrupted, cancel_futures=interrupted)
        except TypeError:
            executor.shutdown(wait=not interrupted)
        if interrupted:
            sys.exit(130)
    
    logger.info(f"Processing complete: {success_count}/{len(input_files)} files succeeded")
    if failed_files:
        logger.warning(f"Failed files: {failed_files}")

if __name__ == "__main__":
    main()