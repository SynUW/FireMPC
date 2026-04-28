#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIRMS data binarization script (simplified version).

Processes only the first band of FIRMS data:
- Read only the first band
- Threshold 8: >= 8 -> 1, < 8 -> 0
- Save as a new TIFF file

Usage:
python firms_binary_simple.py --input_folder /path/to/input --output_folder /path/to/output
"""

import argparse
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import rasterio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Set log level to show only warnings and errors
warnings.filterwarnings('ignore')

# Thread-safe counter
class ThreadSafeCounter:
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
    
    # Throttle update frequency
    should_update = (
        current_time - start_time >= 2.0 or  # At least 2 seconds interval
        progress_percent - int(progress_percent) == 0 or  # Every 1% progress
        current == total  # When complete
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
        print()  # Newline after completion

def process_single_file_parallel(args_tuple) -> tuple:
    """
    Wrapper for processing a single file in parallel.
    
    Args:
        args_tuple: (input_path, output_path, threshold) tuple
    
    Returns:
        (success: bool, input_path: Path, error_msg: str)
    """
    input_path, output_path, threshold = args_tuple
    try:
        success = process_single_file(input_path, output_path, threshold)
        return success, input_path, ""
    except Exception as e:
        return False, input_path, str(e)

def process_single_file(input_path: Path, output_path: Path, threshold: int = 8) -> bool:
    """
    Process the first band of a single TIFF file.
    
    Args:
        input_path: input file path
        output_path: output file path
        threshold: binarization threshold
    
    Returns:
        whether processing succeeded
    """
    try:
        with rasterio.open(input_path) as src:
            # Read only the first band
            band_data = src.read(1)
            
            # Get source file info
            profile = src.profile.copy()
            
            # Check whether the NoData value is compatible with uint8
            nodata_value = src.nodata
            if nodata_value is not None:
                if nodata_value < 0 or nodata_value > 255:
                    # If the NoData value is outside the uint8 range, use 255 as the NoData value
                    output_nodata = 255
                else:
                    output_nodata = int(nodata_value)
            else:
                output_nodata = None
            
            # Binarize
            binary_data = np.zeros_like(band_data, dtype=np.uint8)
            
            # Mark NoData positions
            if nodata_value is not None:
                nodata_mask = (band_data == nodata_value)
            else:
                nodata_mask = np.zeros_like(band_data, dtype=bool)
            
            # Binarize: >= threshold -> 1, < threshold -> 0
            binary_data[band_data >= threshold] = 1
            binary_data[band_data < threshold] = 0
            
            # Keep NoData values unchanged (use 255 as the uint8 NoData value)
            if nodata_value is not None:
                binary_data[nodata_mask] = 255
            
            # Update profile for output
            profile.update({
                'dtype': 'uint8',
                'count': 1,  # Only one band
                'compress': 'LZW',
                'tiled': True,
                'blockxsize': 512,
                'blockysize': 512,
                'BIGTIFF': 'IF_SAFER',
                'nodata': output_nodata
            })
            
            # Write to output file
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(binary_data, 1)
                
                # Copy metadata
                dst.descriptions = src.descriptions[:1] if src.descriptions else None
                dst.colorinterp = src.colorinterp[:1] if src.colorinterp else None
        
        return True
        
    except Exception as e:
        print(f"\nProcessing file {input_path.name} failed: {e}")
        return False

def find_tiff_files(folder: Path) -> list:
    """Find all TIFF files in a folder"""
    tiff_extensions = ['.tif', '.tiff', '.TIF', '.TIFF']
    tiff_files = []
    
    for ext in tiff_extensions:
        tiff_files.extend(folder.glob(f'*{ext}'))
    
    return sorted(tiff_files)

def main():
    parser = argparse.ArgumentParser(
        description="FIRMS first-band binarization script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python firms_binary_simple.py --input_folder /path/to/input --output_folder /path/to/output
  python firms_binary_simple.py --input_folder /path/to/input --output_folder /path/to/output --threshold 8
  python firms_binary_simple.py --input_folder /path/to/input --output_folder /path/to/output --max_workers 16
        """
    )
    
    parser.add_argument('--input_folder', type=Path, 
                        default='/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled_5xdownsampled_merged_unifiedEntireCan_scaledInt/MCD14A1_combined_5xMaximumDown_merged',
                       help='Input folder path')
    parser.add_argument('--output_folder', type=Path, 
                        default='/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled_5xdownsampled_merged_unifiedEntireCan_scaledInt/MCD14A1_combined_5xMaximumDown_merged_binary',
                       help='Output folder path')
    parser.add_argument('--threshold', type=int, default=9,
                       help='Binarization threshold (default: 8)')
    parser.add_argument('--max_workers', type=int, default=None,
                       help='Maximum number of threads for parallel processing (default: number of CPU cores)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing output files')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip existing output files')
    
    args = parser.parse_args()
    
    # Check input folder
    if not args.input_folder.exists():
        print(f"Error: input folder does not exist: {args.input_folder}")
        sys.exit(1)
    
    # Auto-create output folder
    args.output_folder.mkdir(parents=True, exist_ok=True)
    
    # Find input files
    input_files = find_tiff_files(args.input_folder)
    if not input_files:
        print(f"Error: in folder {args.input_folder}: no TIFF files found")
        sys.exit(1)
    
    # Determine the number of parallel threads
    max_workers = args.max_workers if args.max_workers else min(32, (len(input_files) or 1) + 4)
    
    print(f"Input directory: {args.input_folder}")
    print(f"Output directory: {args.output_folder}")
    print(f"Binarization threshold: {args.threshold}")
    print(f"Parallel thread count: {max_workers}")
    print(f"Found {len(input_files)} TIFF files")
    print("Processing mode: first band only")
    print()
    
    # Preprocessing: check whether output files exist
    tasks = []
    skip_count = 0
    
    for input_file in input_files:
        output_file = args.output_folder / input_file.name
        
        # Check if output file already exists
        if output_file.exists():
            if args.overwrite:
                # Overwrite mode: add to task list
                tasks.append((input_file, output_file, args.threshold))
            elif args.skip_existing:
                # Skip existing files
                skip_count += 1
                continue
            else:
                # Default mode: error out
                print(f"Warning: output file already exists: {output_file}")
                print("Use --overwrite to overwrite, or --skip_existing to skip")
                continue
        else:
            # File does not exist: add to the task list
            tasks.append((input_file, output_file, args.threshold))
    
    if not tasks:
        print("No files to process")
        return
    
    print(f"Need to process {len(tasks)} files, skipped {skip_count} files")
    print()
    
    # Process files in parallel
    success_count = 0
    error_count = 0
    processed_count = ThreadSafeCounter()
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(process_single_file_parallel, task): task 
            for task in tasks
        }
        
        # Process completed tasks
        for future in as_completed(future_to_task):
            success, input_path, error_msg = future.result()
            current = processed_count.increment()
            
            # Update progress bar
            print_progress(current, len(tasks), start_time, prefix="Binarize ")
            
            if success:
                success_count += 1
            else:
                error_count += 1
                if error_msg:
                    print(f"\nProcessing file {input_path.name} failed: {error_msg}")
    
    # Print statistics
    elapsed_time = time.time() - start_time
    print(f"\nProcessing complete! Total elapsed: {format_time(elapsed_time)}")
    print(f"Total: {len(input_files)}, succeeded: {success_count}, skipped: {skip_count}, failed: {error_count}")
    print(f"Output directory: {args.output_folder}")
    
    if success_count > 0:
        avg_time = elapsed_time / success_count
        print(f"Average processing time: {avg_time:.2f} s/file")
        print(f"Parallel speedup: {max_workers:.1f}x")

if __name__ == "__main__":
    main()
