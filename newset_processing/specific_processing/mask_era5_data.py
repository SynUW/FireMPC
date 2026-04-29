#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ERA5 data masking tool.
- Randomly select one TIFF file's first band and build a mask (non-zero values mark the valid region).
- Apply the mask to every TIFF file.
- Set out-of-mask regions to -32768 (NoData value).
- If the raw data contains -32768, first replace it with -32767 to avoid confusion with the mask NoData value.
  (Inside-mask original -32768 -> -32767; outside-mask regions -> -32768.)
"""

import argparse
import random
import time
from pathlib import Path
from typing import List, Tuple
import numpy as np
from osgeo import gdal
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure GDAL
gdal.UseExceptions()
gdal.PushErrorHandler('CPLQuietErrorHandler')

class ERA5MaskProcessor:
    def __init__(self, input_path: Path, output_path: Path, workers: int = 8, replace_existing: bool = False):
        self.input_path = input_path
        self.output_path = output_path
        self.workers = workers
        self.replace_existing = replace_existing
        self.mask = None
        self.mask_info = None
        
    def find_tiff_files(self) -> List[Path]:
        """Find all TIFF files"""
        tiff_files = list(self.input_path.glob("*.tif")) + list(self.input_path.glob("*.TIF"))
        return sorted(tiff_files)
    
    def create_mask_from_sample(self, sample_file: Path) -> Tuple[np.ndarray, dict]:
        """Build a mask from a sample file's first band"""
        print(f"Building mask from sample file: {sample_file.name}")
        
        # Open the sample file
        ds = gdal.Open(str(sample_file))
        if ds is None:
            raise RuntimeError(f"Cannot open sample file: {sample_file}")
        
        # Get the first band
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray()
        
        # Get georeferencing
        geotransform = ds.GetGeoTransform()
        projection = ds.GetProjection()
        width = ds.RasterXSize
        height = ds.RasterYSize
        
        # Build the mask: non-zero regions are True
        mask = (data != 0)
        
        # Ensure the mask has the correct shape (height, width)
        if mask.shape != (height, width):
            print(f"Mask shape needs adjustment: {mask.shape} -> ({height}, {width})")
            # If the shape is (width, height), transpose
            if mask.shape == (width, height):
                mask = mask.T
                print(f"Mask transposed: {mask.shape}")
        
        mask_info = {
            'geotransform': geotransform,
            'projection': projection,
            'width': width,
            'height': height
        }
        
        ds = None
        
        # Compute mask statistics
        total_pixels = mask.size
        valid_pixels = np.sum(mask)
        valid_ratio = valid_pixels / total_pixels * 100
        
        print(f"   Mask info:")
        print(f"   Size: {width} x {height}")
        print(f"   Total pixels: {total_pixels:,}")
        print(f"   Valid pixels: {valid_pixels:,}")
        print(f"   Valid ratio: {valid_ratio:.2f}%")
        
        return mask, mask_info
    
    def process_single_file(self, input_file: Path) -> bool:
        """Process a single TIFF file"""
        try:
            # Build the output file path
            output_file = self.output_path / input_file.name
            
            # If the output file exists and replacement is not needed, skip
            if output_file.exists() and not self.replace_existing:
                return True
            
            # Open input file
            src_ds = gdal.Open(str(input_file))
            if src_ds is None:
                print(f"\nCannot open file: {input_file.name}")
                return False
            
            # Get file info
            width = src_ds.RasterXSize
            height = src_ds.RasterYSize
            bands = src_ds.RasterCount
            geotransform = src_ds.GetGeoTransform()
            projection = src_ds.GetProjection()
            dtype = src_ds.GetRasterBand(1).DataType
            
            # Check whether the mask size matches
            if self.mask is not None:
                expected_mask_shape = (height, width)
                if self.mask.shape != expected_mask_shape:
                    print(f"\nMask size mismatch: {input_file.name}")
                    print(f"   File size: {width}x{height}")
                    print(f"   Expected mask size: {expected_mask_shape}")
                    print(f"   Actual mask size: {self.mask.shape}")
                    
                    # Try transposing the mask (working on a copy, not the original)
                    if self.mask.shape == (width, height):
                        print(f"   Trying to transpose the mask...")
                        mask_to_use = self.mask.T
                        print(f"   Mask size after transpose: {mask_to_use.shape}")
                    else:
                        print(f"   Cannot fix size mismatch; skipping file")
                        src_ds = None
                        return False
                else:
                    mask_to_use = self.mask
            else:
                mask_to_use = None
            
            # Create output file
            driver = gdal.GetDriverByName('GTiff')
            dst_ds = driver.Create(
                str(output_file),
                width, height, bands,
                dtype,
                options=['TILED=YES', 'COMPRESS=LZW', 'BIGTIFF=IF_SAFER']
            )
            
            if dst_ds is None:
                print(f"\nCannot create output file: {output_file}")
                src_ds = None
                return False
            
            # Copy georeferencing
            dst_ds.SetGeoTransform(geotransform)
            dst_ds.SetProjection(projection)
            
            # Process each band
            for b in range(bands):
                try:
                    src_band = src_ds.GetRasterBand(b + 1)
                    dst_band = dst_ds.GetRasterBand(b + 1)
                    
                    # Read data
                    data = src_band.ReadAsArray()
                    
                    # Apply mask: regions outside the mask are set to -32768
                    if mask_to_use is not None:
                        # Check whether the raw data contains -32768
                        has_32768 = np.any(data == -32768)
                        
                        # If -32768 is present in the raw data, replace it first to avoid confusion with the mask NoData value
                        if has_32768:
                            # Pixels inside the mask with value -32768 are replaced with -32767
                            # This way, -32768 outside the mask is distinguishable from raw -32768 inside the mask
                            data = np.where((mask_to_use) & (data == -32768), -32767, data)
                        
                        # Apply mask: regions outside the mask are set to -32768
                        data = np.where(mask_to_use, data, -32768)
                    
                    # Write data
                    dst_band.WriteArray(data)
                    dst_band.SetNoDataValue(-32768)
                    
                    # Copy band description
                    description = src_band.GetDescription()
                    if description:
                        dst_band.SetDescription(description)
                        
                except Exception as band_error:
                    print(f"\nFailed to process band{b+1}failed {input_file.name}: {band_error}")
                    dst_ds = None
                    src_ds = None
                    return False
            
            # Close file
            dst_ds.FlushCache()
            dst_ds = None
            src_ds = None
            
            return True
            
        except Exception as e:
            print(f"\nFailed to process file {input_file.name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_all_files(self, tiff_files: List[Path]):
        """Process all TIFF files"""
        print(f"\nStarting to process {len(tiff_files)} TIFF files...")
        
        # Prepare the output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Record start time
        start_time = time.time()
        success_count = 0
        
        # Process files with a thread pool
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_single_file, file_path): file_path 
                for file_path in tiff_files
            }
            
            # Process completed tasks
            processed_count = 0
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                processed_count += 1
                
                try:
                    success = future.result(timeout=300)  # 5-minute timeout
                    if success:
                        success_count += 1
                    
                    # Update progress -- fixed calculation logic
                    elapsed_time = time.time() - start_time
                    avg_time = elapsed_time / processed_count if processed_count > 0 else 0
                    remaining = (len(tiff_files) - processed_count) * avg_time
                    
                    eta_str = f"{remaining/60:.0f}:{remaining%60:02.0f}" if remaining > 0 else "0:00"
                    elapsed_str = f"{elapsed_time/60:.0f}:{elapsed_time%60:02.0f}"
                    
                    print(f"\r[{processed_count}/{len(tiff_files)}] | succeeded: {success_count} failed: {processed_count-success_count} | avg: {avg_time:.1f} s/file | elapsed: {elapsed_str} ETA: {eta_str}", 
                          end='', flush=True)
                    
                    # Show detailed status every 100 files
                    # if processed_count % 100 == 0:
                    #     print(f"\nStatus update: processed {processed_count}, succeeded {success_count}, failed {processed_count-success_count}")
                    
                except Exception as e:
                    print(f"\nException while processing file {file_path.name}: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
        
        print()  # Newline
        
        # Show final statistics
        total_time = time.time() - start_time
        print(f"\nProcessing complete!")
        print(f"   Succeeded: {success_count}/{len(tiff_files)}")
        print(f"   Failed: {len(tiff_files) - success_count}")
        print(f"   Total elapsed: {total_time/60:.1f} minutes")
        print(f"   Average speed: {total_time/len(tiff_files):.1f} s/file")
    
    def run(self):
        """Run the main processing pipeline"""
        print("="*70)
        print("ERA5 data masking tool")
        print("="*70)
        
        # Find TIFF files
        tiff_files = self.find_tiff_files()
        if not tiff_files:
            print(f"No TIFF files found: {self.input_path}")
            return
        
        print(f"Input directory: {self.input_path}")
        print(f"Output directory: {self.output_path}")
        print(f"Found {len(tiff_files)} TIFF files")
        print(f"Worker threads: {self.workers}")
        print(f"Replace mode: {'enabled' if self.replace_existing else 'disabled (skip existing files)'}")
        
        # Randomly select a sample file
        sample_file = random.choice(tiff_files)
        print(f"Randomly selected sample file: {sample_file.name}")
        
        # Build the mask
        self.mask, self.mask_info = self.create_mask_from_sample(sample_file)
        
        # Process all files
        self.process_all_files(tiff_files)


def main():
    parser = argparse.ArgumentParser(description="ERA5 data masking tool")
    parser.add_argument("input_path", type=Path, nargs='?',
                        default = '/mnt/raid/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic/ERA5_consistent_mosaic',
                        help="Input TIFF directory")
    parser.add_argument("output_path", type=Path, nargs='?',
                        default = '/mnt/raid/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic/ERA5_consistent_mosaic_withnodata',
                        help="Output directory")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads (default 8)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    parser.add_argument("--replace", action="store_true", help="Replace existing output files (default: skip)")
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Check input directory
    if not args.input_path.exists():
        print(f"Input directory does not exist: {args.input_path}")
        return
    
    if not args.input_path.is_dir():
        print(f"Input path is not a directory: {args.input_path}")
        return
    
    # Create the processor and run it
    processor = ERA5MaskProcessor(
        input_path=args.input_path,
        output_path=args.output_path,
        workers=args.workers,
        replace_existing=args.replace
    )
    
    processor.run()


if __name__ == "__main__":
    main()
