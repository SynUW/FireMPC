#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TIF file masking tool.
Use a template file's valid-value range to mask every TIF image in a target folder.
"""

import os
import glob
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from osgeo import gdal, osr
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def apply_template_mask_to_tiff(input_file, output_file, template_mask, template_nodata, template_gt, template_proj):
    """
    Mask a TIF file using the template file's spatial mask and NoData value.
    """
    try:
        # Read input file
        data, geotransform, projection, nodata = read_tiff_with_metadata(input_file)
        if data is None:
            return False
        
        # Check whether georeferencing matches
        if not np.allclose(geotransform, template_gt, rtol=1e-6):
            logger.warning(f"Geotransform mismatch: {input_file}")
            return False
        
        if projection != template_proj:
            logger.warning(f"Projection mismatch: {input_file}")
            return False
        
        # Check whether data sizes match
        if data.ndim == 3:
            if data.shape[1:] != template_mask.shape:
                logger.warning(f"Data size mismatch: {input_file} {data.shape[1:]} vs {template_mask.shape}")
                return False
        else:
            if data.shape != template_mask.shape:
                logger.warning(f"Data size mismatch: {input_file} {data.shape} vs {template_mask.shape}")
                return False
        
        # Ensure the NoData value's dtype is compatible with the data
        if data.dtype != np.float32 and data.dtype != np.float64:
            # For integer dtypes, ensure the NoData value is within range
            if data.dtype == np.uint8:
                if template_nodata < 0 or template_nodata > 255:
                    template_nodata = 255
            elif data.dtype == np.uint16:
                if template_nodata < 0 or template_nodata > 65535:
                    template_nodata = 65535
        
        # Apply template mask
        if data.ndim == 3:
            # Multi-band data
            masked_data = data.copy()
            for i in range(data.shape[0]):
                # Use template mask
                masked_data[i][~template_mask] = template_nodata
        else:
            # Single-band data
            masked_data = data.copy()
            # Use template mask
            masked_data[~template_mask] = template_nodata
        
        
        
        # Write the output file using the template's NoData value
        write_tiff_with_metadata(output_file, masked_data, geotransform, projection, template_nodata)
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
    input_file, output_file, template_mask, template_nodata, template_gt, template_proj = args_tuple
    
    # Apply template mask
    success = apply_template_mask_to_tiff(input_file, output_file, template_mask, template_nodata, template_gt, template_proj)
    
    if success:
        return input_file, True, None
    else:
        return input_file, False, f"Processing failed: {input_file}"

def main():
    parser = argparse.ArgumentParser(description='Use the template file\'s valid-value range to mask TIF files')
    parser.add_argument('--template', default='/mnt/raid/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x_norm/ERA5_consistent_mosaic_withnodata_downsampled/2024_12_31.tif', help='Template TIF file path')
    parser.add_argument('--input_dir', default='/mnt/raid/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x_norm/MCD14A1_mosaic_binary_downsampled', help='Input TIF file directory')
    parser.add_argument('--output_dir', default='/mnt/raid/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x_norm/MCD14A1_mosaic_binary_downsampled_masked', help='Output directory for masked files')
    parser.add_argument('--pattern', default='*.tif', help='File match pattern (default: *.tif)')
    parser.add_argument('--recursive', action='store_true', help='Search subdirectories recursively')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads (default: 4)')
    
    args = parser.parse_args()
    
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
    template_mask, template_nodata, template_data = get_template_mask_and_nodata(args.template)
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
    
    # Prepare multithreading parameters
    file_args = []
    for input_file in input_files:
        # Build the output file path
        rel_path = os.path.relpath(input_file, args.input_dir)
        output_file = os.path.join(args.output_dir, rel_path)
        
        # Create the output file's directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        file_args.append((input_file, output_file, template_mask, template_nodata, template_gt, template_proj))
    
    # Process files with multithreading
    success_count = 0
    failed_files = []
    
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_single_file, file_arg): file_arg[0] for file_arg in file_args}
        
        # Use tqdm to display progress
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
                    'progress': f"{success_count + len(failed_files)}/{len(input_files)}"
                })
    
    logger.info(f"Processing complete: {success_count}/{len(input_files)} files succeeded")
    if failed_files:
        logger.warning(f"Failed files: {failed_files}")

if __name__ == "__main__":
    main()