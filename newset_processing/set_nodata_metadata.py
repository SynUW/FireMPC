#!/usr/bin/env python3
"""
TIFF NoData removal script.
Features:
1. Removes the NoData setting directly from TIFFs.
2. Does not change any pixel values.
3. In-place: modifies files directly.
4. Supports multithreaded parallelism.
5. Supports recursive subfolder processing.
"""

import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from osgeo import gdal
import numpy as np

# Suppress GDAL warning messages
gdal.PushErrorHandler('CPLQuietErrorHandler')
gdal.SetConfigOption('CPL_LOG', '/dev/null')

gdal.UseExceptions()

def _fmt_time(s: float) -> str:
    """Format time display"""
    s = int(s)
    return f"{s//3600}:{(s%3600)//60:02d}:{s%60:02d}"

def process_single_file(args):
    """In-place processing of a single TIFF file: remove the NoData setting without changing pixel values"""
    input_path, overwrite = args
    
    try:
        # Open the file for update
        src_ds = gdal.Open(str(input_path), gdal.GA_Update)
        if src_ds is None:
            return ('error', f"Cannot open: {input_path.name}")
        
        nbands = src_ds.RasterCount
        processed_bands = 0
        removed_nodata_bands = 0
        
        # Process each band
        for band_idx in range(1, nbands + 1):
            band = src_ds.GetRasterBand(band_idx)
            
            # Get the current NoData value
            current_nodata = band.GetNoDataValue()
            
            if current_nodata is not None:
                # Remove the NoData setting
                band.DeleteNoDataValue()
                removed_nodata_bands += 1
            
            processed_bands += 1
            band.FlushCache()
        
        # Close file
        src_ds.FlushCache()
        src_ds = None
        
        return ('success', f"{input_path.name} (total bands:{nbands}, processed:{processed_bands}, NoData removed:{removed_nodata_bands})")
        
    except Exception as e:
        return ('error', f"{input_path.name}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(
        description='Modify TIFF files in place: remove the NoData setting without changing pixel values'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input folder path')
    parser.add_argument('--pattern', type=str, default='*.tif',
                        help='File match pattern (default: *.tif)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel threads (default: 8)')
    parser.add_argument('--recursive', action='store_true',
                        help='Process subfolders recursively')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    
    if not input_dir.exists():
        print(f"Input folder does not exist: {input_dir}")
        return
    
    # Find all TIFF files
    print("Scanning TIFF files...")
    if args.recursive:
        all_tifs = list(input_dir.rglob(args.pattern))
    else:
        all_tifs = list(input_dir.glob(args.pattern))
    
    total = len(all_tifs)
    
    if total == 0:
        print(f"No matching files found: {args.pattern}")
        return
    
    print(f"Found {total} files")
    print(f"Processing directory: {input_dir}")
    print(f"Using {args.workers} threads")
    print("=" * 60)
    
    # Prepare tasks (in-place, no output path needed)
    tasks = []
    for tif_path in all_tifs:
        tasks.append((tif_path, True))  # overwrite parameter is kept but not used
    
    # Process files
    t0 = time.time()
    done = 0
    success = 0
    error = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_file, task): task for task in tasks}
        
        for future in as_completed(futures):
            status, msg = future.result()
            done += 1
            
            if status == 'success':
                success += 1
            elif status == 'error':
                error += 1
                print(f"\nError: {msg}")
            
            # Update progress
            elapsed = time.time() - t0
            eta = elapsed * (total - done) / max(1, done)
            avg_time = elapsed / done
            
            print(f"\rProgress [{done}/{total}] | "
                  f"Succeeded: {success} errors: {error} | "
                  f"Avg: {avg_time:.3f} s/file | "
                  f"Elapsed: {_fmt_time(elapsed)} "
                  f"ETA: {_fmt_time(eta)}", end='', flush=True)
    
    print(f"\n\n{'='*60}")
    print(f"Processing complete!")
    print(f"   - Total files: {total}")
    print(f"   - Succeeded: {success}")
    print(f"   - Errors: {error}")
    print(f"\nTotal elapsed: {_fmt_time(time.time() - t0)}")
    print(f"Average speed: {total / (time.time() - t0):.1f} files/sec")
    print("=" * 60)
    
    print(f"\nProcessing notes:")
    print(f"   1. In-place: modify the original files directly")
    print(f"   2. Remove all NoData settings")
    print(f"   3. Do not change any pixel values")
    print(f"   4. Preserve the original dtype and compression")

if __name__ == '__main__':
    main()
