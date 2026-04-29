#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global-valid-position mask checker.
Use a global mask file to check whether any tif image has invalid values at valid positions.
"""

import os
import argparse
import logging
from datetime import datetime
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

import numpy as np
from osgeo import gdal

# Optimize GDAL performance configuration
gdal.UseExceptions()
os.environ['GDAL_CACHEMAX'] = '102400'  # 100 GB cache

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    # If tqdm is unavailable, create a simple substitute
    HAS_TQDM = False
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_mask_file(mask_path: str) -> np.ndarray:
    """
    Load the global valid-position mask file.
    
    Args:
        mask_path: mask file path
        
    Returns:
        mask array; 1 at valid positions, 0 at invalid positions
    """
    try:
        ds = gdal.Open(mask_path, gdal.GA_ReadOnly)
        if ds is None:
            raise RuntimeError(f"Cannot open mask file: {mask_path}")
        
        # Read mask data
        band = ds.GetRasterBand(1)
        mask = band.ReadAsArray().astype(np.uint8)
        
        height, width = ds.RasterYSize, ds.RasterXSize
        geotransform = ds.GetGeoTransform()
        projection = ds.GetProjection()
        
        ds = None
        
        logger.info(f"Mask file loaded successfully: {width}x{height}, valid pixel count: {np.sum(mask == 1)}")
        
        return mask, (width, height), geotransform, projection
        
    except Exception as e:
        logger.error(f"Failed to load mask file: {e}")
        raise


def check_tif_file_optimized(tif_path: str, mask: np.ndarray, mask_info: Dict, 
                             chunk_size: int = 2048) -> Dict:
    """
    Optimized: check whether a single tif file has invalid values at valid positions.
    Uses block-wise reading to improve performance and avoid loading huge files entirely into memory.
    
    Args:
        tif_path: tif file path
        mask: global mask array
        mask_info: mask metadata (width, height, geotransform, projection)
        chunk_size: block size (default 2048 pixels)
        
    Returns:
        result dictionary
    """
    mask_width, mask_height = mask_info['size']
    
    result = {
        'file': tif_path,
        'status': 'unknown',
        'errors': [],
        'valid_pixels_checked': 0,
        'invalid_pixels_found': 0
    }
    
    try:
        ds = gdal.Open(tif_path, gdal.GA_ReadOnly)
        if ds is None:
            result['status'] = 'failed'
            result['errors'].append('Cannot open file')
            return result
        
        # Quickly check whether spatial sizes match
        tif_width, tif_height = ds.RasterXSize, ds.RasterYSize
        if tif_width != mask_width or tif_height != mask_height:
            result['status'] = 'failed'
            result['errors'].append(f'Size mismatch: {tif_width}x{tif_height} vs {mask_width}x{mask_height}')
            ds = None
            return result
        
        # Get the valid-position mask (computed once)
        valid_mask = (mask == 1)
        num_valid_pixels = np.sum(valid_mask)
        result['valid_pixels_checked'] = num_valid_pixels
        
        # If there are no valid pixels, return quickly
        if num_valid_pixels == 0:
            ds = None
            result['status'] = 'passed'
            return result
        
        # Check each band
        bands = ds.RasterCount
        found_invalid = False  # Only record whether invalid values exist, no count
        
        for band_idx in range(1, bands + 1):
            if found_invalid:
                break  # If invalid values are already found, break out of the band loop
            
            band = ds.GetRasterBand(band_idx)
            nodata = band.GetNoDataValue()
            
            # Use block-wise reading to optimize performance
            for y in range(0, tif_height, chunk_size):
                y_end = min(y + chunk_size, tif_height)
                height = y_end - y
                
                # Read one block
                data = band.ReadAsArray(0, y, tif_width, height)
                
                # Get the mask for this block
                mask_chunk = valid_mask[y:y_end, :]
                
                # If this block has no valid pixels, skip it
                if not np.any(mask_chunk):
                    continue
                
                # Check invalid values at valid positions
                valid_data = data[mask_chunk]
                
                # Fast vectorized invalid-value check
                invalid_mask = ~np.isfinite(valid_data)  # NaN or Inf
                
                if nodata is not None:
                    invalid_mask |= (valid_data == float(nodata))
                
                # As soon as any invalid value is found, mark and break out
                if np.any(invalid_mask):
                    found_invalid = True
                    result['errors'].append(f'Band{band_idx}: invalid value found at valid positions (nodata={nodata})')
                    break  # Break out of the block loop
        
        ds = None
        
        result['invalid_pixels_found'] = 1 if found_invalid else 0  # Only record presence/absence
        
        if found_invalid:
            result['status'] = 'failed'
        else:
            result['status'] = 'passed'
        
        return result
        
    except Exception as e:
        result['status'] = 'failed'
        result['errors'].append(f'Error during check: {str(e)}')
        return result


def find_all_tif_files(root_dir: str) -> List[str]:
    """
    Recursively find every tif file under a root directory.
    Supports two cases:
    1. The root directory itself is a tif file.
    2. The root directory contains subfolders with tif files inside.
    
    Args:
        root_dir: root directory path
        
    Returns:
        list of tif file paths
    """
    tif_files = []
    
    # First, check whether the root directory itself is a tif file
    if os.path.isfile(root_dir):
        if root_dir.lower().endswith('.tif') or root_dir.lower().endswith('.tiff'):
            return [root_dir]
        else:
            logger.warning(f"Specified path is a file but not a tif: {root_dir}")
            return []
    
    # Recursively find every tif file under the root directory
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.tif') or file.lower().endswith('.tiff'):
                tif_files.append(os.path.join(root, file))
    
    return sorted(tif_files)


def process_files_parallel(tif_files: List[str], mask: np.ndarray, mask_info: Dict, 
                          max_workers: int = 8, log_path: str = None) -> List[Dict]:
    """
    Process multiple tif files in parallel (optimized; logs written dynamically).
    
    Args:
        tif_files: list of tif file paths
        mask: mask array
        mask_info: mask metadata
        max_workers: maximum number of parallel threads
        log_path: log file path (for dynamic writing)
        
    Returns:
        list of check results
    """
    results = []
    failed_count = 0
    
    # Filter out the mask file itself
    mask_file_path = mask_info.get('mask_file', '')
    filtered_files = [f for f in tif_files if os.path.abspath(f) != os.path.abspath(mask_file_path)]
    
    logger.info(f"Starting check of {len(filtered_files)} tif files (using {max_workers} threads)")
    
    # Initialize log file
    if log_path:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Global valid-position mask check results (dynamic update)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total files: {len(filtered_files)}\n\n")
            f.write("Failed file list:\n")
            f.write("-" * 80 + "\n")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(check_tif_file_optimized, tif_file, mask, mask_info): tif_file 
            for tif_file in filtered_files
        }
        
        # Use tqdm to display progress
        for future in tqdm(as_completed(futures), total=len(futures), 
                          desc="Check progress", ncols=100):
            result = future.result()
            results.append(result)
            
            # Write failed files to the log dynamically
            if result['status'] == 'failed' and log_path:
                failed_count += 1
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(f"\n#{failed_count} file: {result['file']}\n")
                    f.write(f"  status: {result['status']}\n")
                    f.write(f"  checked valid pixel count: {result['valid_pixels_checked']}\n")
                    if result['errors']:
                        f.write("  Error message:\n")
                        for error in result['errors']:
                            f.write(f"    - {error}\n")
            
            # Dynamic progress indicator (reduced output frequency)
            completed = len(results)
            total = len(filtered_files)
            if completed % 100 == 0 or completed == total:  # Print every 100 items or on the last one
                passed = sum(1 for r in results if r['status'] == 'passed')
                failed = sum(1 for r in results if r['status'] == 'failed')
                if HAS_TQDM:
                    tqdm.write(f"Progress: {completed}/{total}, passed: {passed}, failed: {failed}")
                else:
                    print(f"Progress: {completed}/{total}, passed: {passed}, failed: {failed}")
                
                # Also update log statistics
                if log_path and completed == total:
                    with open(log_path, 'a', encoding='utf-8') as f:
                        f.write("\n" + "=" * 80 + "\n")
                        f.write("Check complete\n")
                        f.write(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Check passed: {passed}\n")
                        f.write(f"Check failed: {failed}\n")
                        f.write("=" * 80 + "\n")
    
    return sorted(results, key=lambda x: x['file'])


def write_log(results: List[Dict], log_path: str):
    """
    Write check results to a log file.
    
    Args:
        results: list of check results
        log_path: log file path
    """
    total_files = len(results)
    passed_files = sum(1 for r in results if r['status'] == 'passed')
    failed_files = sum(1 for r in results if r['status'] == 'failed')
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Global valid-position mask check results\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Check time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total files: {total_files}\n")
        f.write(f"Check passed: {passed_files}\n")
        f.write(f"Check failed: {failed_files}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Write grouped by status
        failed_results = [r for r in results if r['status'] == 'failed']
        passed_results = [r for r in results if r['status'] == 'passed']
        
        if failed_files > 0:
            f.write(f"\nFailed file list ({failed_files} items):\n")
            f.write("-" * 80 + "\n")
            for result in failed_results:
                f.write(f"\nFile: {result['file']}\n")
                f.write(f"Status: {result['status']}\n")
                f.write(f"Checked valid pixel count: {result['valid_pixels_checked']}\n")
                if result['errors']:
                    f.write("Error message:\n")
                    for error in result['errors']:
                        f.write(f"  - {error}\n")
                f.write("\n")
        
        if passed_files > 0:
            f.write(f"\nPassed file list ({passed_files} items):\n")
            f.write("-" * 80 + "\n")
            for result in passed_results[:100]:  # Only show the first 100 passed entries
                f.write(f"{result['file']}\n")
            
            if passed_files > 100:
                f.write(f"... (and {passed_files - 100} more files passed)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Check complete\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Log saved to: {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Use a global mask to check every tif file for invalid values at valid positions"
    )
    parser.add_argument(
        "--mask",
        type=str,
        default='/mnt/raid/user/wildfire_dataset/self_built_materials/h5_dataset/global_valid_mask_v5.tif',
        help="Global valid-position mask file path"
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default='/mnt/raid/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x_norm/distance_maps',
        help="Root directory containing tif files, or a single tif file path"
    )
    parser.add_argument(
        "--output-log",
        type=str,
        default='/mnt/raid/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x_norm/mask_check_log.txt',
        help="Output log file path (default: <root_dir>/mask_check_log.txt)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=f"Maximum parallel threads (default: CPU cores {multiprocessing.cpu_count()})"
    )
    
    args = parser.parse_args()
    
    # Set default thread count
    if args.max_workers is None:
        args.max_workers = multiprocessing.cpu_count()
    
    # Check whether the mask file exists
    if not os.path.exists(args.mask):
        logger.error(f"Mask file does not exist: {args.mask}")
        return
    
    # Check whether the root directory or file exists
    if not os.path.exists(args.root_dir):
        logger.error(f"Specified path does not exist: {args.root_dir}")
        return
    
    # Set default log path
    if args.output_log is None:
        # If root_dir is a file, use its parent directory; if it is a directory, use it directly
        if os.path.isfile(args.root_dir):
            root_dir_for_log = os.path.dirname(args.root_dir)
        else:
            root_dir_for_log = args.root_dir
        args.output_log = os.path.join(root_dir_for_log, 'mask_check_log.txt')
    
    logger.info("=" * 80)
    logger.info("Starting global valid-position mask check")
    logger.info("=" * 80)
    logger.info(f"Mask file: {args.mask}")
    logger.info(f"Check path: {args.root_dir}")
    logger.info(f"Output log: {args.output_log}")
    logger.info(f"Parallel thread count: {args.max_workers}")
    logger.info("")
    
    # Load the mask file
    logger.info("Step 1/4: loading global mask file...")
    mask, size, geotransform, projection = load_mask_file(args.mask)
    mask_info = {
        'size': size,
        'geotransform': geotransform,
        'projection': projection,
        'mask_file': args.mask
    }
    
    # Find every tif file
    logger.info("Step 2/3: finding all tif files...")
    tif_files = find_all_tif_files(args.root_dir)
    logger.info(f"Found {len(tif_files)} tif files")
    
    # Check files (write log dynamically)
    logger.info("Step 3/3: checking tif files (log file is being updated in real time)...")
    results = process_files_parallel(tif_files, mask, mask_info, args.max_workers, args.output_log)
    
    # Print statistics
    total_files = len(results)
    passed_files = sum(1 for r in results if r['status'] == 'passed')
    failed_files = sum(1 for r in results if r['status'] == 'failed')
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("Check complete")
    logger.info("=" * 80)
    logger.info(f"Total files: {total_files}")
    logger.info(f"Check passed: {passed_files}")
    logger.info(f"Check failed: {failed_files}")
    logger.info(f"Detailed results written in real time: {args.output_log}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
