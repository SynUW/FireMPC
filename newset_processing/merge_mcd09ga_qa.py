# -*- coding: utf-8 -*-
"""
Merge TIFFs from two folders (MOD/MYD09GA and MOD/MYD09GA_QA_state1km) by stacking bands.

Features:
- Takes two input folders, each containing tif files.
- Filename format: *_YYYY_MM_DD_tilexx.tif (xx is 00, 01, 02, 03, ...)
- Match files with the same date and tile and merge them.
- Merge: bands from the second folder are appended after the bands of the matching first-folder file.
- Resolution handling: the second image's resolution is used as the reference; the first image is nearest-neighbor downsampled to it.

Usage:
    python merge_two_folders_bands.py --input-dir1 /path/to/folder1 --input-dir2 /path/to/folder2 --output-dir /path/to/output
"""

import os
import sys
import argparse
import logging
import time
import re
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from osgeo import gdal

# ================== Defaults ==================
# Parallel-processing configuration
DEFAULT_MAX_WORKERS = max(1, os.cpu_count() // 2)  # Use half the CPU cores by default

# GDAL optimization settings
GDAL_CACHE_SIZE    = 102400       # GDAL cache size (MB)
GDAL_NUM_THREADS   = "ALL_CPUS"   # GDAL thread count
COMPRESS           = "LZW"        # Output compression
TILED              = True          # Output block-tiling flag
BIGTIFF            = "IF_SAFER"    # Auto-switch to BigTIFF
BLOCK_SIZE         = 256           # Block size

# Thread-safe counter
progress_lock = Lock()

# ============ Logging configuration ============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("merge_bands")

# Initialize GDAL
gdal.UseExceptions()
gdal.SetCacheMax(GDAL_CACHE_SIZE * 1024 * 1024)
gdal.SetConfigOption('GDAL_NUM_THREADS', GDAL_NUM_THREADS)
gdal.SetConfigOption('GDAL_CACHEMAX', str(GDAL_CACHE_SIZE))

# Filename match pattern: *_YYYY_MM_DD_tilexx.tif
FILENAME_PATTERN = re.compile(r'(.+)_(\d{4})[_-](\d{2})[_-](\d{2})_tile(\d+)\.tif$', re.IGNORECASE)


def format_time(seconds):
    """Format time display"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def parse_filename(filename):
    """
    Parse the filename and extract the date and tile number.
    
    Args:
        filename: filename (no path)
    
    Returns:
        tuple: (date_str, tile_num) or None
        - date_str: date string in YYYY_MM_DD format
        - tile_num: tile-number string (e.g. "00", "01")
    """
    match = FILENAME_PATTERN.match(filename)
    if match:
        prefix, year, month, day, tile_num = match.groups()
        date_str = f"{year}_{month}_{day}"
        return date_str, tile_num
    return None


def group_files_by_date_tile(directory):
    """
    Group files by date and tile.
    
    Args:
        directory: input directory path
    
    Returns:
        dict: {(date_str, tile_num): file_path}
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    
    file_dict = {}
    patterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
    
    for pattern in patterns:
        for file_path in dir_path.glob(pattern):
            parsed = parse_filename(file_path.name)
            if parsed:
                date_str, tile_num = parsed
                key = (date_str, tile_num)
                if key in file_dict:
                    logger.warning(f"Duplicate date+tile combination found: {file_path.name}; using the first file")
                else:
                    file_dict[key] = file_path
    
    return file_dict


def get_file_info(ds):
    """Get basic info from a GDAL dataset"""
    return {
        'width': ds.RasterXSize,
        'height': ds.RasterYSize,
        'bands': ds.RasterCount,
        'geotransform': ds.GetGeoTransform(),
        'projection': ds.GetProjection(),
        'datatype': ds.GetRasterBand(1).DataType,
    }


def resample_to_target(ds_source, ds_target, output_path=None):
    """
    Resample the source dataset to the target dataset's resolution and geographic extent.
    
    Args:
        ds_source: source GDAL dataset (file from the first folder)
        ds_target: target GDAL dataset (file from the second folder, used as reference)
        output_path: optional temp output path; if provided, save to file
    
    Returns:
        GDAL dataset object, or None on failure
    """
    try:
        # Get target dataset info
        target_info = get_file_info(ds_target)
        target_gt = target_info['geotransform']
        target_proj = target_info['projection']
        target_width = target_info['width']
        target_height = target_info['height']
        
        # Compute target geographic extent
        pixel_size_x = abs(target_gt[1])
        pixel_size_y = abs(target_gt[5])
        target_min_x = target_gt[0]
        target_max_y = target_gt[3]
        target_max_x = target_min_x + target_width * pixel_size_x
        target_min_y = target_max_y - target_height * pixel_size_y
        
        # Use gdal.Warp for resampling
        # Important: specify output width and height explicitly to ensure exact size match
        warp_options = gdal.WarpOptions(
            format='GTiff' if output_path else 'MEM',
            width=target_width,  # Explicit output width
            height=target_height,  # Explicit output height
            outputBounds=(target_min_x, target_min_y, target_max_x, target_max_y),  # Target geographic extent
            resampleAlg=gdal.GRA_NearestNeighbour,  # Nearest-neighbor resampling
            dstSRS=target_proj,  # Target projection
            outputType=ds_source.GetRasterBand(1).DataType,  # Preserve original dtype
            creationOptions=[
                f'COMPRESS={COMPRESS}',
                'TILED=YES' if TILED else 'TILED=NO',
                f'BLOCKXSIZE={BLOCK_SIZE}',
                f'BLOCKYSIZE={BLOCK_SIZE}',
                f'BIGTIFF={BIGTIFF}',
                'NUM_THREADS=ALL_CPUS'
            ] if output_path else None,
            multithread=True,
            warpOptions=['NUM_THREADS=ALL_CPUS']
        )
        
        # Execute resampling
        warp_dest = output_path if output_path else '/vsimem/temp_resampled'
        resampled_ds = gdal.Warp(
            warp_dest,
            ds_source,
            options=warp_options
        )
        
        if resampled_ds is None:
            gdal_error = gdal.GetLastErrorMsg()
            error_msg = "Resampling failed"
            if gdal_error:
                error_msg += f": {gdal_error}"
            logger.error(error_msg)
            return None
        
        # Validate output size (should match exactly because width and height were explicit)
        if resampled_ds.RasterXSize != target_width or resampled_ds.RasterYSize != target_height:
            logger.warning(f"Size mismatch after resampling: {resampled_ds.RasterXSize}x{resampled_ds.RasterYSize} vs {target_width}x{target_height}")
            logger.warning("This should not happen because width and height were explicit")
            # If still mismatched, force-adjust
            resampled_ds = None
            return resample_to_target_force(ds_source, ds_target, output_path)
        
        # Ensure the geotransform fully matches (all 6 parameters)
        resampled_gt = resampled_ds.GetGeoTransform()
        if resampled_gt != target_gt:
            logger.debug("Adjusting the geotransform to match the target")
            resampled_ds.SetGeoTransform(target_gt)
            # Re-fetch to verify
            resampled_gt = resampled_ds.GetGeoTransform()
        
        # Verify pixel sizes are identical
        resampled_pixel_x = abs(resampled_gt[1])
        resampled_pixel_y = abs(resampled_gt[5])
        target_pixel_x = abs(target_gt[1])
        target_pixel_y = abs(target_gt[5])
        
        if abs(resampled_pixel_x - target_pixel_x) > 1e-10 or abs(resampled_pixel_y - target_pixel_y) > 1e-10:
            logger.warning(f"Pixel sizes do not match exactly: {resampled_pixel_x:.10f}x{resampled_pixel_y:.10f} vs {target_pixel_x:.10f}x{target_pixel_y:.10f}")
            # Force-set the geo-transform
            resampled_ds.SetGeoTransform(target_gt)
        
        # Verify spatial extents
        resampled_min_x = resampled_gt[0]
        resampled_max_y = resampled_gt[3]
        target_min_x = target_gt[0]
        target_max_y = target_gt[3]
        
        if abs(resampled_min_x - target_min_x) > 1e-6 or abs(resampled_max_y - target_max_y) > 1e-6:
            logger.warning(f"Spatial coverage does not match exactly: top-left ({resampled_min_x:.6f}, {resampled_max_y:.6f}) vs ({target_min_x:.6f}, {target_max_y:.6f})")
            # Force-set the geo-transform
            resampled_ds.SetGeoTransform(target_gt)
        
        return resampled_ds
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error during resampling: {e}")
        logger.error(f"Detailed error:\n{error_details}")
        gdal_error = gdal.GetLastErrorMsg()
        if gdal_error:
            logger.error(f"GDAL error: {gdal_error}")
        # If the above methods fail, try the forced method
        try:
            return resample_to_target_force(ds_source, ds_target, output_path)
        except Exception as e2:
            import traceback
            error_details2 = traceback.format_exc()
            logger.error(f"Force-resampling also failed: {e2}")
            logger.error(f"Force-resampling detailed error:\n{error_details2}")
            return None


def resample_to_target_force(ds_source, ds_target, output_path=None):
    """
    Forced resampling method: directly create the output dataset with the requested size and copy data manually.
    Used to handle size-mismatch cases.
    """
    try:
        target_info = get_file_info(ds_target)
        target_gt = target_info['geotransform']
        target_proj = target_info['projection']
        target_width = target_info['width']
        target_height = target_info['height']
        target_datatype = ds_source.GetRasterBand(1).DataType
        
        # First use Warp for resampling (without specifying width/height, let GDAL compute them)
        pixel_size_x = abs(target_gt[1])
        pixel_size_y = abs(target_gt[5])
        target_min_x = target_gt[0]
        target_max_y = target_gt[3]
        target_max_x = target_min_x + target_width * pixel_size_x
        target_min_y = target_max_y - target_height * pixel_size_y
        
        warp_options = gdal.WarpOptions(
            format='MEM',
            outputBounds=(target_min_x, target_min_y, target_max_x, target_max_y),
            xRes=pixel_size_x,
            yRes=pixel_size_y,
            resampleAlg=gdal.GRA_NearestNeighbour,
            dstSRS=target_proj,
            outputType=target_datatype,
            multithread=True
        )
        
        temp_ds = gdal.Warp('/vsimem/temp_warp', ds_source, options=warp_options)
        if temp_ds is None:
            return None
        
        # Create the output dataset with the correct size
        driver = gdal.GetDriverByName('GTiff' if output_path else 'MEM')
        creation_options = [
            f'COMPRESS={COMPRESS}',
            'TILED=YES' if TILED else 'TILED=NO',
            f'BLOCKXSIZE={BLOCK_SIZE}',
            f'BLOCKYSIZE={BLOCK_SIZE}',
            f'BIGTIFF={BIGTIFF}',
            'NUM_THREADS=ALL_CPUS'
        ] if output_path else None
        
        out_ds = driver.Create(
            output_path if output_path else '/vsimem/temp_resampled',
            target_width,
            target_height,
            temp_ds.RasterCount,
            target_datatype,
            options=creation_options
        )
        
        if out_ds is None:
            temp_ds = None
            return None
        
        out_ds.SetGeoTransform(target_gt)
        out_ds.SetProjection(target_proj)
        
        # Compute the region to copy (from temp_ds's center or top-left)
        src_width = temp_ds.RasterXSize
        src_height = temp_ds.RasterYSize
        
        # If the source is larger than the target, crop from the top-left
        # If the source is smaller than the target, pad starting from the top-left
        copy_width = min(target_width, src_width)
        copy_height = min(target_height, src_height)
        src_x = 0
        src_y = 0
        dst_x = 0
        dst_y = 0
        
        # Copy data
        for band_idx in range(1, temp_ds.RasterCount + 1):
            src_band = temp_ds.GetRasterBand(band_idx)
            dst_band = out_ds.GetRasterBand(band_idx)
            
            # Set NoData value
            nodata = src_band.GetNoDataValue()
            if nodata is not None:
                dst_band.SetNoDataValue(nodata)
                # Fill with NoData values first
                dst_band.Fill(nodata)
            
            # Read source data and write to the target
            if copy_width > 0 and copy_height > 0:
                src_data = src_band.ReadAsArray(src_x, src_y, copy_width, copy_height)
                dst_band.WriteArray(src_data, dst_x, dst_y)
            
            dst_band.FlushCache()
        
        temp_ds = None
        return out_ds
                
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Force-resampling failed: {e}")
        logger.error(f"Detailed error:\n{error_details}")
        gdal_error = gdal.GetLastErrorMsg()
        if gdal_error:
            logger.error(f"GDAL error: {gdal_error}")
        return None


def merge_bands(ds1, ds2, output_path, overwrite=False):
    """
    Merge bands from two datasets.
    
    Args:
        ds1: first dataset (will be resampled)
        ds2: second dataset (used as resolution reference)
        output_path: output file path
        overwrite: whether to overwrite an existing file
    
    Returns:
        bool: True on success, False on failure
    """
    if not overwrite and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return True
    
    try:
        # Get the second dataset's info (used as reference)
        target_info = get_file_info(ds2)
        target_width = target_info['width']
        target_height = target_info['height']
        target_gt = target_info['geotransform']
        target_proj = target_info['projection']
        target_datatype = target_info['datatype']
        
        # Get band count
        bands1 = ds1.RasterCount
        bands2 = ds2.RasterCount
        total_bands = bands1 + bands2
        
        logger.debug(f"Merge bands: {bands1} + {bands2} = {total_bands}")
        logger.debug(f"Output size: {target_width} x {target_height}")
        
        # Resample the first dataset to the second dataset's resolution
        logger.debug("Starting to resample the first dataset...")
        logger.debug(f"Source dataset info: {ds1.RasterXSize}x{ds1.RasterYSize}, {ds1.RasterCount}Band")
        logger.debug(f"Target dataset info: {target_width}x{target_height}, {ds2.RasterCount}Band")
        resampled_ds1 = resample_to_target(ds1, ds2)
        if resampled_ds1 is None:
            logger.error("Failed to resample the first dataset")
            logger.error(f"Source file: {ds1.GetDescription() if hasattr(ds1, 'GetDescription') else 'N/A'}")
            logger.error(f"Target file: {ds2.GetDescription() if hasattr(ds2, 'GetDescription') else 'N/A'}")
            return False
        
        # Validate the post-resampling size
        if resampled_ds1.RasterXSize != target_width or resampled_ds1.RasterYSize != target_height:
            logger.error(f"Size mismatch after resampling: {resampled_ds1.RasterXSize}x{resampled_ds1.RasterYSize} vs {target_width}x{target_height}")
            resampled_ds1 = None
            return False
        
        # Create output dataset
        driver = gdal.GetDriverByName('GTiff')
        creation_options = [
            f'COMPRESS={COMPRESS}',
            'TILED=YES' if TILED else 'TILED=NO',
            f'BLOCKXSIZE={BLOCK_SIZE}',
            f'BLOCKYSIZE={BLOCK_SIZE}',
            f'BIGTIFF={BIGTIFF}',
            'NUM_THREADS=ALL_CPUS'
        ]
        
        out_ds = driver.Create(
            output_path,
            target_width,
            target_height,
            total_bands,
            target_datatype,
            options=creation_options
        )
        
        if out_ds is None:
            gdal_error = gdal.GetLastErrorMsg()
            error_msg = f"Cannot create output file: {output_path}"
            if gdal_error:
                error_msg += f"\nGDAL error: {gdal_error}"
            logger.error(error_msg)
            resampled_ds1 = None
            return False
        
        # Set georeferencing
        out_ds.SetGeoTransform(target_gt)
        out_ds.SetProjection(target_proj)
        
        # Unify the NoData value to -32768
        UNIFIED_NODATA = -32768
        
        # Mapping from GDAL data types to numpy data types
        dtype_map = {
            gdal.GDT_Byte: np.uint8,
            gdal.GDT_UInt16: np.uint16,
            gdal.GDT_Int16: np.int16,
            gdal.GDT_UInt32: np.uint32,
            gdal.GDT_Int32: np.int32,
            gdal.GDT_Float32: np.float32,
            gdal.GDT_Float64: np.float64,
        }
        target_np_dtype = dtype_map.get(target_datatype, np.int16)  # Default int16
        
        # Use block-wise processing to improve performance (especially for large images)
        # Block size: 256x256 or larger, but not exceeding the image size
        block_size = min(512, target_width, target_height)  # 512 x 512 pixel blocks
        
        # Copy the first dataset's bands (block-wise)
        logger.debug(f"Copying the first dataset's {bands1} bands (block-wise processing)...")
        for band_idx in range(1, bands1 + 1):
            src_band = resampled_ds1.GetRasterBand(band_idx)
            dst_band = out_ds.GetRasterBand(band_idx)
            
            # Get source NoData value
            src_nodata = src_band.GetNoDataValue()
            need_nodata_replace = (src_nodata is not None and src_nodata != UNIFIED_NODATA)
            
            # Set the unified NoData value
            dst_band.SetNoDataValue(UNIFIED_NODATA)
            
            # Block-wise read and write
            for y in range(0, target_height, block_size):
                y_size = min(block_size, target_height - y)
                for x in range(0, target_width, block_size):
                    x_size = min(block_size, target_width - x)
                    
                    # Read block data
                    data = src_band.ReadAsArray(x, y, x_size, y_size)
                    
                    # Handle NoData values
                    if need_nodata_replace:
                        data = np.where(data == src_nodata, UNIFIED_NODATA, data)
                    
                    # Ensure dtype match
                    if data.dtype != target_np_dtype:
                        data = data.astype(target_np_dtype)
                    
                    # Write block data
                    dst_band.WriteArray(data, x, y)
        
        # Copy the second dataset's bands (block-wise)
        logger.debug(f"Copying the second dataset's {bands2} bands (block-wise processing)...")
        for band_idx in range(1, bands2 + 1):
            src_band = ds2.GetRasterBand(band_idx)
            dst_band = out_ds.GetRasterBand(bands1 + band_idx)
            
            # Get source NoData value
            src_nodata = src_band.GetNoDataValue()
            need_nodata_replace = (src_nodata is not None and src_nodata != UNIFIED_NODATA)
            
            # Set the unified NoData value
            dst_band.SetNoDataValue(UNIFIED_NODATA)
            
            # Block-wise read and write
            for y in range(0, target_height, block_size):
                y_size = min(block_size, target_height - y)
                for x in range(0, target_width, block_size):
                    x_size = min(block_size, target_width - x)
                    
                    # Read block data
                    data = src_band.ReadAsArray(x, y, x_size, y_size)
                    
                    # Handle NoData values
                    if need_nodata_replace:
                        data = np.where(data == src_nodata, UNIFIED_NODATA, data)
                    
                    # Ensure dtype match
                    if data.dtype != target_np_dtype:
                        data = data.astype(target_np_dtype)
                    
                    # Write block data
                    dst_band.WriteArray(data, x, y)
        
        # Flush the cache once at the end (rather than per band)
        out_ds.FlushCache()
        
        # Clean up resources
        out_ds = None
        resampled_ds1 = None
        
        logger.debug(f"Successfully merged and saved: {output_path}")
        return True
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error merging bands: {e}")
        logger.error(f"Detailed error:\n{error_details}")
        logger.error(f"Output path: {output_path}")
        # Try to get GDAL error info
        gdal_error = gdal.GetLastErrorMsg()
        if gdal_error:
            logger.error(f"GDAL error: {gdal_error}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Merge TIFF files from two folders by stacking bands')
    
    parser.add_argument('--input-dir1', default='/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MYD09GA_b1237',
                       help='First input folder path')
    parser.add_argument('--input-dir2', default='/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MYD09GA_QA_state1km',
                       help='Second input folder path (1 km QA, used as resolution reference)')
    parser.add_argument('--output-dir', default='/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MYD09GA_b1237_withQA',
                       help='Output folder path')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing output files')
    parser.add_argument('--verbose', action='store_true',
                       help='Show verbose info')
    parser.add_argument('--workers', type=int, default=DEFAULT_MAX_WORKERS//4,
                       help=f'Parallel worker threads (default: {DEFAULT_MAX_WORKERS})')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input directory
    for dir_path, name in [(args.input_dir1, 'First input directory'), 
                           (args.input_dir2, 'Second input directory')]:
        if not os.path.isdir(dir_path):
            logger.error(f"{name}does not exist: {dir_path}")
            sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group files
    logger.info("Scanning the first folder...")
    files1 = group_files_by_date_tile(args.input_dir1)
    logger.info(f"First folder found {len(files1)} files")
    
    logger.info("Scanning the second folder...")
    files2 = group_files_by_date_tile(args.input_dir2)
    logger.info(f"Second folder found {len(files2)} files")
    
    # Find matching file pairs
    matched_pairs = []
    for key in files1:
        if key in files2:
            matched_pairs.append((key, files1[key], files2[key]))
    
    if not matched_pairs:
        logger.warning("No matching file pairs (same date and tile) found")
        sys.exit(0)
    
    logger.info(f"Found {len(matched_pairs)} matching pairs")
    logger.info(f"Using {args.workers} parallel worker threads")
    
    # Processing function (used for parallel processing)
    def process_file_pair(pair_data):
        """Function to process a single file pair"""
        idx, key, file1_path, file2_path, output_dir, overwrite = pair_data
        date_str, tile_num = key
        # Use the filename from the first folder as the output filename
        output_filename = file1_path.name
        output_path = output_dir / output_filename
        
        try:
            # Open files
            ds1 = gdal.Open(str(file1_path), gdal.GA_ReadOnly)
            if ds1 is None:
                gdal_error = gdal.GetLastErrorMsg()
                error_msg = f"Cannot open file 1: {file1_path.name}"
                if gdal_error:
                    error_msg += f" (GDAL error: {gdal_error})"
                return False, error_msg
            
            ds2 = gdal.Open(str(file2_path), gdal.GA_ReadOnly)
            if ds2 is None:
                gdal_error = gdal.GetLastErrorMsg()
                error_msg = f"Cannot open file 2: {file2_path.name}"
                if gdal_error:
                    error_msg += f" (GDAL error: {gdal_error})"
                ds1 = None
                return False, error_msg
            
            # Merge bands
            success = merge_bands(ds1, ds2, str(output_path), overwrite)
            
            # Clean up resources
            ds1 = None
            ds2 = None
            
            if success:
                return True, None
            else:
                # merge_bands already logs detailed error info; return only a brief summary here
                return False, f"Merge failed: {date_str} tile{tile_num} (file 1: {file1_path.name}, file 2: {file2_path.name}, output: {output_filename})"
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            error_msg = f"Error while processing file pair {date_str} tile{tile_num}: {e}\nDetailed error:\n{error_details}"
            return False, error_msg
    
    # Prepare task data
    tasks = [
        (idx, key, file1_path, file2_path, output_dir, args.overwrite)
        for idx, (key, file1_path, file2_path) in enumerate(matched_pairs, 1)
    ]
    
    # Parallel processing
    total_success = 0
    total_failed = 0
    start_time = time.time()
    completed_count = 0
    failed_files = []  # Collect failed-file information
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(process_file_pair, task): task for task in tasks}
        
        # Process completed tasks
        for future in as_completed(future_to_task):
            try:
                success, error_msg = future.result()
                task = future_to_task[future]
                _, key, file1_path, file2_path, _, _ = task
                output_filename = file1_path.name
                
                with progress_lock:
                    completed_count += 1
                    if success:
                        total_success += 1
                    else:
                        total_failed += 1
                        # Log failed-file information
                        failed_files.append((output_filename, error_msg or "Unknown error"))
                        if error_msg:
                            logger.error(error_msg)
                
                # Show progress
                if completed_count % 10 == 0 or completed_count == len(matched_pairs):
                    elapsed = time.time() - start_time
                    progress = completed_count / len(matched_pairs) * 100
                    avg_time = elapsed / completed_count if completed_count > 0 else 0
                    eta = avg_time * (len(matched_pairs) - completed_count)
                    print(f"\rProgress: [{completed_count}/{len(matched_pairs)}] ({progress:.1f}%) | "
                          f"Succeeded: {total_success} | failed: {total_failed} | "
                          f"Elapsed: {format_time(elapsed)} | ETA: {format_time(eta)}", 
                          end='', flush=True)
            except Exception as e:
                task = future_to_task.get(future)
                output_filename = "Unknown file"
                if task:
                    _, _, file1_path, _, _, _ = task
                    output_filename = file1_path.name
                
                with progress_lock:
                    completed_count += 1
                    total_failed += 1
                    error_msg = f"Task execution exception: {e}"
                    failed_files.append((output_filename, error_msg))
                    logger.error(error_msg)
    
    print()  # Newline
    
    elapsed_time = time.time() - start_time
    
    # Print statistics
    logger.info("=" * 60)
    logger.info("Processing complete!")
    logger.info(f"Total elapsed: {format_time(elapsed_time)}")
    logger.info(f"Succeeded: {total_success}/{len(matched_pairs)}")
    if total_failed > 0:
        logger.warning(f"Failed: {total_failed}/{len(matched_pairs)}")
    logger.info("=" * 60)
    
    # Print the list of failed files
    if failed_files:
        logger.info(f"\nFailed file list ({len(failed_files)}):")
        for filename, error in failed_files:
            # Extract the first line of the error message (the main error)
            error_first_line = error.split('\n')[0] if error else "Unknown error"
            logger.info(f"  - {filename}: {error_first_line}")


if __name__ == "__main__":
    main()

