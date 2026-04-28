#!/usr/bin/env python3
"""
MCD15A3H data-processing script.
Features:
1. Map pixel value 249 to 0.
2. Map pixel value 0 to -32768.
3. Set NoData = -32768.
4. Cast dtype to Int16 (supports negatives).
5. Output to a new folder.
"""

import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from osgeo import gdal

# Suppress GDAL warning messages
gdal.PushErrorHandler('CPLQuietErrorHandler')
gdal.SetConfigOption('CPL_LOG', '/dev/null')

gdal.UseExceptions()

def _fmt_time(s: float) -> str:
    """Format time display"""
    s = int(s)
    return f"{s//3600}:{(s%3600)//60:02d}:{s%60:02d}"

def process_single_file(args):
    """Process a single TIFF file"""
    input_path, output_path, overwrite = args
    
    try:
        # Check if output file already exists
        if output_path.exists() and not overwrite:
            return ('skipped', f"Already exists: {output_path.name}")
        
        # Open the source file
        src_ds = gdal.Open(str(input_path), gdal.GA_ReadOnly)
        if src_ds is None:
            return ('error', f"Cannot open: {input_path.name}")
        
        # Get metadata
        width = src_ds.RasterXSize
        height = src_ds.RasterYSize
        nbands = src_ds.RasterCount
        geotransform = src_ds.GetGeoTransform()
        projection = src_ds.GetProjection()
        
        # Create the output file (Int16 to support negatives)
        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(
            str(output_path),
            width, height, nbands,
            gdal.GDT_Int16,  # Use Int16 to support -32768
            options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER']
        )
        
        if dst_ds is None:
            src_ds = None
            return ('error', f"Cannot create output: {output_path.name}")
        
        # Set georeferencing
        dst_ds.SetGeoTransform(geotransform)
        dst_ds.SetProjection(projection)
        
        # Process each band
        for band_idx in range(1, nbands + 1):
            src_band = src_ds.GetRasterBand(band_idx)
            dst_band = dst_ds.GetRasterBand(band_idx)
            
            # Set NoData to -32768
            dst_band.SetNoDataValue(-32768)
            
            # Read data
            data = src_band.ReadAsArray()
            if data is None:
                continue
            
            # Cast to Int16 to support negatives
            data = data.astype(np.int16)
            
            # Process pixel values:
            # 1. Map 0 to -32768 (NoData) first
            # 2. Then map 249 to 0
            mask_zero = (data == 0)
            mask_249 = (data == 249)
            
            data[mask_zero] = -32768
            data[mask_249] = 0
            
            # Write data
            dst_band.WriteArray(data)
            dst_band.FlushCache()
            
            # Copy band description
            desc = src_band.GetDescription()
            if desc:
                dst_band.SetDescription(desc)
        
        # Close file
        dst_ds.FlushCache()
        src_ds = None
        dst_ds = None
        
        return ('success', output_path.name)
        
    except Exception as e:
        return ('error', f"{input_path.name}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(
        description='MCD15A3H data processing: 249 -> 0, 0 -> -32768 (NoData)'
    )
    parser.add_argument('--input', type=str, default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic/MCD15A3H_mosaic',
                        help='Input folder path')
    parser.add_argument('--output', type=str, default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic/MCD15A3H_mosaic_249',
                        help='Output folder path')
    parser.add_argument('--pattern', type=str, default='*.tif',
                        help='File match pattern (default: *.tif)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel threads (default: 8)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing files')
    parser.add_argument('--recursive', action='store_true',
                        help='Process subfolders recursively')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"Input folder does not exist: {input_dir}")
        return
    
    # Create output folder
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Using {args.workers} threads")
    print("=" * 60)
    
    # Prepare tasks
    tasks = []
    for tif_path in all_tifs:
        # Compute relative path
        if args.recursive:
            rel_path = tif_path.relative_to(input_dir)
            out_path = output_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path = output_dir / tif_path.name
        
        tasks.append((tif_path, out_path, args.overwrite))
    
    # Process files
    t0 = time.time()
    done = 0
    success = 0
    skipped = 0
    error = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_file, task): task for task in tasks}
        
        for future in as_completed(futures):
            status, msg = future.result()
            done += 1
            
            if status == 'success':
                success += 1
            elif status == 'skipped':
                skipped += 1
            elif status == 'error':
                error += 1
                print(f"\nError: {msg}")
            
            # Update progress
            elapsed = time.time() - t0
            eta = elapsed * (total - done) / max(1, done)
            avg_time = elapsed / done
            
            print(f"\rProgress [{done}/{total}] | "
                  f"Succeeded: {success} skipped: {skipped} errors: {error} | "
                  f"Avg: {avg_time:.3f} s/file | "
                  f"Elapsed: {_fmt_time(elapsed)} "
                  f"ETA: {_fmt_time(eta)}", end='', flush=True)
    
    print(f"\n\n{'='*60}")
    print(f"Processing complete!")
    print(f"   - Total files: {total}")
    print(f"   - Succeeded: {success}")
    print(f"   - skipped: {skipped}")
    print(f"   - Errors: {error}")
    print(f"\nTotal elapsed: {_fmt_time(time.time() - t0)}")
    print(f"Average speed: {total / (time.time() - t0):.1f} files/sec")
    print("=" * 60)
    
    print(f"\nProcessing notes:")
    print(f"   1. pixel value 249 -> 0")
    print(f"   2. pixel value 0 -> -32768")
    print(f"   3. NoData set to: -32768")
    print(f"   4. Dtype: Int16")
    print(f"   5. Compression: LZW")

if __name__ == '__main__':
    main()

