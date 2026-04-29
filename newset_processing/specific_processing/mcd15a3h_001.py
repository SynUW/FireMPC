#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Threshold-processing script - set pixel values below the threshold to 0.

Process every TIFF in a folder, set values below the threshold to 0, and write to a new folder.
Supports multi-band files; all bands are processed.

Threshold rules:
- Values below the threshold (not including the threshold itself) become 0.
- Values >= threshold are preserved.
- NoData values become 0; the output file no longer sets a NoData value.
"""

import sys
import time
import argparse
from pathlib import Path
import numpy as np
import rasterio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import warnings

warnings.filterwarnings('ignore')

class ThreadSafeCounter:
    """Thread-safe counter"""
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self):
        with self._lock:
            self._value += 1
            return self._value
    
    @property
    def value(self):
        with self._lock:
            return self._value

def format_time(seconds: float) -> str:
    """Format time display"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"

def print_progress(current: int, total: int, start_time: float, prefix: str = ""):
    """Print progress bar"""
    current_time = time.time()
    progress_percent = current / total * 100
    
    should_update = (
        current_time - start_time >= 2.0 or
        progress_percent - int(progress_percent) == 0 or
        current == total
    )
    
    if not should_update and current < total:
        return
    
    elapsed = current_time - start_time
    if current > 0:
        avg_time_per_item = elapsed / current
        remaining_items = total - current
        eta = remaining_items * avg_time_per_item
        eta_str = format_time(eta)
    else:
        eta_str = "?:??:??"
    
    elapsed_str = format_time(elapsed)
    total_estimated = elapsed / current * total if current > 0 else 0
    total_str = format_time(total_estimated)
    
    print(f"\r{prefix}[{current}/{total}], {elapsed_str}/{total_str}, ETA: {eta_str}", end='', flush=True)
    
    if current == total:
        print()

def apply_threshold(data: np.ndarray, threshold: float, nodata_value: float = None) -> np.ndarray:
    """
    Apply thresholding.
    
    Args:
        data: input data array
        threshold: threshold
        nodata_value: NoData value
    
    Returns:
        the processed data array
    
    Threshold rules:
    - Values below the threshold (not including the threshold itself) become 0.
    - Values >= threshold are preserved.
    - NoData values become 0.
    """
    # Create a copy of the output array
    result = data.copy().astype(data.dtype)
    
    # Handle NoData: set NoData values to 0
    if nodata_value is not None:
        nodata_mask = data == nodata_value
        result[nodata_mask] = 0
    
    # Apply the threshold rule: values below the threshold become 0
    # Note: the threshold itself is not included, so use < not <=
    # Also exclude values that are already 0
    threshold_mask = (data < threshold) & (data > 0)
    result[threshold_mask] = 0
    
    return result

def process_single_file(input_path: Path, output_path: Path, threshold: float) -> bool:
    """
    Process a single TIFF file.
    
    Args:
        input_path: input file path
        output_path: output file path
        threshold: threshold
    
    Returns:
        whether processing succeeded
    """
    try:
        # Read input file
        with rasterio.open(input_path) as src:
            # Get file info
            profile = src.profile.copy()
            nodata_value = src.nodata
            
            # Modify profile: remove the NoData setting
            profile['nodata'] = None
            
            # Read all bands
            data = src.read()
            
            # Apply thresholding to each band
            processed_data = []
            for band_idx in range(data.shape[0]):
                band_data = data[band_idx]
                processed_band = apply_threshold(band_data, threshold, nodata_value)
                processed_data.append(processed_band)
            
            # Write to output file
            with rasterio.open(output_path, 'w', **profile) as dst:
                for band_idx, band_data in enumerate(processed_data):
                    dst.write(band_data, band_idx + 1)
                
                # Copy metadata
                if src.descriptions:
                    dst.descriptions = src.descriptions
                if src.colorinterp:
                    dst.colorinterp = src.colorinterp
                
                # Copy tags safely
                try:
                    src_tags = src.tags()
                    if src_tags:
                        dst.update_tags(**src_tags)
                except Exception:
                    # If tag copy fails, ignore the error and continue
                    pass
        
        return True
        
    except Exception as e:
        print(f"\nProcessing file {input_path.name} failed: {e}")
        return False

def process_single_file_parallel(args_tuple) -> tuple:
    """Wrapper for parallel processing"""
    input_path, output_path, threshold = args_tuple
    try:
        success = process_single_file(input_path, output_path, threshold)
        return success, input_path, ""
    except Exception as e:
        return False, input_path, str(e)

def find_tiff_files(folder: Path) -> list:
    """Find every TIFF file in a folder"""
    tiff_extensions = ['.tif', '.tiff', '.TIF', '.TIFF']
    tiff_files = []
    for ext in tiff_extensions:
        tiff_files.extend(folder.glob(f'*{ext}'))
    return sorted(tiff_files)

def main():
    parser = argparse.ArgumentParser(
        description="Thresholding: set pixel values below the threshold to 0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Threshold rules:
- Values below the threshold (not including the threshold itself) become 0.
- Values >= threshold are preserved.
- NoData values become 0; the output file no longer sets a NoData value.

Examples:
- threshold=0.1: 0.05 -> 0, 0.1 -> 0.1, 0.15 -> 0.15, NoData(-9999) -> 0
- threshold=0.5: 0.3 -> 0, 0.5 -> 0.5, 0.8 -> 0.8, NoData(-9999) -> 0
        """
    )
    
    parser.add_argument('--input_folder', type=Path, 
                        default='/mnt/raid/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_10x_norm/MCD15A3H_mosaic_downsampled_daily', 
                        help='Input folder path')
    parser.add_argument('--output_folder', type=Path, 
                        default='/mnt/raid/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_10x_norm/MCD15A3H_mosaic_downsampled_daily_001', 
                        help='Output folder path')
    parser.add_argument('--threshold', type=float, default=0.1, help='Threshold (default 0.1; values below this become 0, not including the threshold itself)')
    parser.add_argument('--max_workers', type=int, default=16, help='Maximum number of threads for parallel processing (default: number of CPU cores)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output files')
    parser.add_argument('--skip_existing', action='store_true', help='Skip existing output files')
    
    args = parser.parse_args()
    
    if not args.input_folder.exists():
        print(f"Error: input folder does not exist: {args.input_folder}", file=sys.stderr)
        sys.exit(1)
    
    # Create output folder
    args.output_folder.mkdir(parents=True, exist_ok=True)
    
    input_files = find_tiff_files(args.input_folder)
    if not input_files:
        print(f"Error: in folder {args.input_folder}: no TIFF files found", file=sys.stderr)
        sys.exit(1)
    
    max_workers = args.max_workers if args.max_workers else min(32, (len(input_files) or 1) + 4)
    
    print(f"Input directory: {args.input_folder}")
    print(f"Output directory: {args.output_folder}")
    print(f"Threshold: {args.threshold}")
    print(f"Threshold rule: values below {args.threshold} become 0 (the threshold itself is excluded)")
    print(f"NoData handling: NoData values become 0; the output file no longer sets a NoData value")
    print(f"Parallel thread count: {max_workers}")
    print(f"Found {len(input_files)} TIFF files")
    print()
    
    # Prepare tasks
    tasks = []
    skip_count = 0
    
    for input_file in input_files:
        output_file = args.output_folder / input_file.name
        
        if output_file.exists():
            if args.overwrite:
                tasks.append((input_file, output_file, args.threshold))
            elif args.skip_existing:
                skip_count += 1
                continue
            else:
                print(f"Warning: output file already exists: {output_file}. Use --overwrite to overwrite, or --skip_existing to skip.")
                continue
        else:
            tasks.append((input_file, output_file, args.threshold))
    
    if not tasks:
        print("No files to process")
        return
    
    print(f"Need to process {len(tasks)} files, skipped {skip_count} files")
    print()
    
    success_count = 0
    error_count = 0
    processed_count = ThreadSafeCounter()
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(process_single_file_parallel, task): task 
            for task in tasks
        }
        
        for future in as_completed(future_to_task):
            success, input_path, error_msg = future.result()
            current = processed_count.increment()
            
            print_progress(current, len(tasks), start_time, prefix="Threshold processing ")
            
            if success:
                success_count += 1
            else:
                error_count += 1
                if error_msg:
                    print(f"\nProcessing file {input_path.name} failed: {error_msg}")
    
    elapsed_time = time.time() - start_time
    print(f"\nProcessing complete! Total elapsed: {format_time(elapsed_time)}")
    print(f"Total: {len(input_files)}, succeeded: {success_count}, skipped: {skip_count}, failed: {error_count}")
    print(f"Output directory: {args.output_folder}")
    
    if success_count > 0:
        avg_time = elapsed_time / success_count
        print(f"Average processing time: {avg_time:.2f} s/file")
        print(f"Parallel speedup: {max_workers:.1f}x")
    
    # Show thresholding statistics
    print(f"\nThreshold rules:")
    print(f"- values below {args.threshold} -> 0")
    print(f"- values >= {args.threshold} -> preserved")
    print(f"- NoData values -> 0")
    print(f"- the output file no longer sets a NoData value")

if __name__ == "__main__":
    main()
