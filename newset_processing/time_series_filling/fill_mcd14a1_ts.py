#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MCD14A1 time-series filling tool.
Fills missing time points using forward fill, backward fill, or mean fill.
"""

import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from osgeo import gdal
import time

# Configure GDAL
gdal.UseExceptions()
gdal.PushErrorHandler('CPLQuietErrorHandler')

class MCD14A1TimeSeriesFiller:
    def __init__(self, input_folder: Path):
        self.input_folder = input_folder
        
    def parse_date_from_filename(self, filename: str) -> Optional[datetime]:
        """Parse the date from the filename"""
        try:
            # Format: yyyy_mm_dd.tif
            date_str = filename.replace('.tif', '').replace('.TIF', '')
            parts = date_str.split('_')
            if len(parts) == 3:
                year, month, day = map(int, parts)
                return datetime(year, month, day)
        except:
            pass
        return None
    
    def get_existing_files(self) -> dict:
        """Get existing files and their dates"""
        existing_files = {}
        
        for file_path in self.input_folder.glob("*.tif"):
            date = self.parse_date_from_filename(file_path.name)
            if date:
                existing_files[date] = file_path
        
        for file_path in self.input_folder.glob("*.TIF"):
            date = self.parse_date_from_filename(file_path.name)
            if date:
                existing_files[date] = file_path
                
        return existing_files
    
    def create_zero_tiff(self, template_file: Path, output_file: Path) -> bool:
        """Create a TIFF file with all-zero pixels"""
        try:
            # Open the template file to get structural info
            src_ds = gdal.Open(str(template_file))
            if src_ds is None:
                return False
            
            # Get file info
            width = src_ds.RasterXSize
            height = src_ds.RasterYSize
            bands = src_ds.RasterCount
            geotransform = src_ds.GetGeoTransform()
            projection = src_ds.GetProjection()
            dtype = src_ds.GetRasterBand(1).DataType
            
            # Create output file
            driver = gdal.GetDriverByName('GTiff')
            dst_ds = driver.Create(
                str(output_file),
                width, height, bands,
                dtype,
                options=['TILED=YES', 'COMPRESS=LZW']
            )
            
            if dst_ds is None:
                src_ds = None
                return False
            
            # Copy georeferencing
            dst_ds.SetGeoTransform(geotransform)
            dst_ds.SetProjection(projection)
            
            # Write all bands
            for b in range(bands):
                src_band = src_ds.GetRasterBand(b + 1)
                dst_band = dst_ds.GetRasterBand(b + 1)
                
                # Get the original nodata value
                original_nodata = src_band.GetNoDataValue()
                
                # Create data: read raw data, then set non-nodata pixels to 0
                if original_nodata is not None:
                    # Read raw data
                    original_data = src_band.ReadAsArray()
                    # Build new data: keep nodata pixels as nodata, set the rest to 0
                    zero_data = np.where(original_data == original_nodata, original_nodata, 0)
                else:
                    # If there is no nodata value, simply create an all-zero array
                    if dtype == gdal.GDT_Byte:
                        zero_data = np.zeros((height, width), dtype=np.uint8)
                    elif dtype == gdal.GDT_Int16:
                        zero_data = np.zeros((height, width), dtype=np.int16)
                    elif dtype == gdal.GDT_UInt16:
                        zero_data = np.zeros((height, width), dtype=np.uint16)
                    elif dtype == gdal.GDT_Int32:
                        zero_data = np.zeros((height, width), dtype=np.int32)
                    elif dtype == gdal.GDT_UInt32:
                        zero_data = np.zeros((height, width), dtype=np.uint32)
                    elif dtype == gdal.GDT_Float32:
                        zero_data = np.zeros((height, width), dtype=np.float32)
                    elif dtype == gdal.GDT_Float64:
                        zero_data = np.zeros((height, width), dtype=np.float64)
                    else:
                        # Default to float32
                        zero_data = np.zeros((height, width), dtype=np.float32)
                
                dst_band.WriteArray(zero_data)
                
                # Keep the original nodata value
                if original_nodata is not None:
                    dst_band.SetNoDataValue(original_nodata)
                
                # Copy band description
                description = src_band.GetDescription()
                if description:
                    dst_band.SetDescription(description)
            
            # Close file
            dst_ds.FlushCache()
            dst_ds = None
            src_ds = None
            
            return True
            
        except Exception as e:
            print(f"Failed to create zero-valued file: {e}")
            return False
    
    def create_mean_tiff(self, file1: Path, file2: Path, output_file: Path) -> bool:
        """Create a pixel-wise mean TIFF from two files"""
        try:
            # Open both input files
            ds1 = gdal.Open(str(file1))
            ds2 = gdal.Open(str(file2))
            
            if ds1 is None or ds2 is None:
                return False
            
            # Check whether file sizes match
            if (ds1.RasterXSize != ds2.RasterXSize or 
                ds1.RasterYSize != ds2.RasterYSize or
                ds1.RasterCount != ds2.RasterCount):
                print(f"File size mismatch: {file1.name} vs {file2.name}")
                ds1 = None
                ds2 = None
                return False
            
            # Get file info
            width = ds1.RasterXSize
            height = ds1.RasterYSize
            bands = ds1.RasterCount
            geotransform = ds1.GetGeoTransform()
            projection = ds1.GetProjection()
            dtype = ds1.GetRasterBand(1).DataType
            
            # Create output file
            driver = gdal.GetDriverByName('GTiff')
            dst_ds = driver.Create(
                str(output_file),
                width, height, bands,
                dtype,
                options=['TILED=YES', 'COMPRESS=LZW']
            )
            
            if dst_ds is None:
                ds1 = None
                ds2 = None
                return False
            
            # Copy georeferencing
            dst_ds.SetGeoTransform(geotransform)
            dst_ds.SetProjection(projection)
            
            # Process each band
            for b in range(bands):
                band1 = ds1.GetRasterBand(b + 1)
                band2 = ds2.GetRasterBand(b + 1)
                
                # Get the original nodata value
                nodata1 = band1.GetNoDataValue()
                nodata2 = band2.GetNoDataValue()
                
                # Read data
                data1 = band1.ReadAsArray().astype(np.float64)
                data2 = band2.ReadAsArray().astype(np.float64)
                
                # Handle nodata
                if nodata1 is not None:
                    data1 = np.where(data1 == nodata1, np.nan, data1)
                if nodata2 is not None:
                    data2 = np.where(data2 == nodata2, np.nan, data2)
                
                # Compute the mean and handle NaN
                with np.errstate(invalid='ignore'):
                    mean_data = np.round((data1 + data2) / 2.0)
                
                # If either of the two pixels is nodata, the result is also nodata
                if nodata1 is not None and nodata2 is not None:
                    mask = np.isnan(data1) | np.isnan(data2)
                    mean_data = np.where(mask, nodata1, mean_data)
                elif nodata1 is not None:
                    mask = np.isnan(data1)
                    mean_data = np.where(mask, nodata1, mean_data)
                elif nodata2 is not None:
                    mask = np.isnan(data2)
                    mean_data = np.where(mask, nodata2, mean_data)
                
                # Cast based on the original dtype
                if dtype == gdal.GDT_Byte:
                    mean_data = mean_data.astype(np.uint8)
                elif dtype == gdal.GDT_Int16:
                    mean_data = mean_data.astype(np.int16)
                elif dtype == gdal.GDT_UInt16:
                    mean_data = mean_data.astype(np.uint16)
                elif dtype == gdal.GDT_Int32:
                    mean_data = mean_data.astype(np.int32)
                elif dtype == gdal.GDT_UInt32:
                    mean_data = mean_data.astype(np.uint32)
                elif dtype == gdal.GDT_Float32:
                    mean_data = mean_data.astype(np.float32)
                elif dtype == gdal.GDT_Float64:
                    mean_data = mean_data.astype(np.float64)
                else:
                    # Default to float32
                    mean_data = mean_data.astype(np.float32)
                
                # Write data
                out_band = dst_ds.GetRasterBand(b + 1)
                out_band.WriteArray(mean_data)
                
                # Keep the original nodata value
                if nodata1 is not None:
                    out_band.SetNoDataValue(nodata1)
                elif nodata2 is not None:
                    out_band.SetNoDataValue(nodata2)
                
                # Copy band description
                description = band1.GetDescription()
                if description:
                    out_band.SetDescription(description)
            
            # Close file
            dst_ds.FlushCache()
            dst_ds = None
            ds1 = None
            ds2 = None
            
            return True
            
        except Exception as e:
            print(f"Failed to create mean file: {e}")
            return False
    
    def fill_time_series(self, start_date: str, end_date: str) -> bool:
        """Fill in the time series"""
        try:
            # Parse date
            start_dt = datetime.strptime(start_date, "%Y%m%d")
            end_dt = datetime.strptime(end_date, "%Y%m%d")
            
            print(f"Time range: {start_date} - {end_date}")
            print(f"Working folder: {self.input_folder}")
            
            # Get existing files
            existing_files = self.get_existing_files()
            print(f"Found {len(existing_files)} existing files")
            
            # Generate the full date sequence
            current_date = start_dt
            all_dates = []
            while current_date <= end_dt:
                all_dates.append(current_date)
                current_date += timedelta(days=1)
            
            print(f"Need to fill {len(all_dates)-len(existing_files)} dates")
            
            # Find a suitable template file (prefer newer files)
            if not existing_files:
                print("No existing file found to use as a template")
                return False
            
            # Prefer newer files as the template (more likely to have correct nodata values)
            sorted_dates = sorted(existing_files.keys(), reverse=True)
            template_file = None
            
            # Try to find a file whose nodata value is non-zero as the template
            for date in sorted_dates[:10]:  # Check the 10 newest files
                test_file = existing_files[date]
                try:
                    ds = gdal.Open(str(test_file))
                    if ds:
                        band = ds.GetRasterBand(1)
                        nodata = band.GetNoDataValue()
                        ds = None
                        if nodata is not None and nodata != 0:
                            template_file = test_file
                            print(f"Using template file: {template_file.name} (nodata={nodata})")
                            break
                except:
                    continue
            
            # If no suitable template is found, use the newest file
            if template_file is None:
                template_file = existing_files[sorted_dates[0]]
                print(f"Using template file: {template_file.name} (default)")
            
            # Begin filling
            start_time = time.time()
            filled_count = 0
            zero_filled_count = 0
            mean_filled_count = 0
            
            for i, target_date in enumerate(all_dates):
                output_filename = target_date.strftime("%Y_%m_%d.tif")
                output_file = self.input_folder / output_filename
                
                # Skip if the file already exists
                if output_file.exists():
                    continue
                
                # Skip if a file already exists for this date (no copy needed)
                if target_date in existing_files:
                    continue
                
                # Cases that need filling
                first_existing_date = min(existing_files.keys())
                if target_date < first_existing_date:
                    # Early gap: fill with zeros
                    if self.create_zero_tiff(template_file, output_file):
                        zero_filled_count += 1
                        filled_count += 1
                else:
                    # Find the nearest preceding and following files
                    prev_file = None
                    next_file = None
                    
                    # Find the previous file
                    for date in sorted(existing_files.keys(), reverse=True):
                        if date < target_date:
                            prev_file = existing_files[date]
                            break
                    
                    # Find the next file
                    for date in sorted(existing_files.keys()):
                        if date > target_date:
                            next_file = existing_files[date]
                            break
                    
                    if prev_file and next_file:
                        # Use the mean of the previous and next files
                        if self.create_mean_tiff(prev_file, next_file, output_file):
                            mean_filled_count += 1
                            filled_count += 1
                    elif prev_file:
                        # Only the previous file exists; copy it
                        import shutil
                        shutil.copy2(prev_file, output_file)
                        filled_count += 1
                    elif next_file:
                        # Only the next file exists; copy it
                        import shutil
                        shutil.copy2(next_file, output_file)
                        filled_count += 1
                
                # Update progress
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (i + 1)
                    remaining = (len(all_dates) - i - 1) * avg_time
                    eta_str = f"{remaining/60:.0f}:{remaining%60:02.0f}" if remaining > 0 else "0:00"
                    elapsed_str = f"{elapsed/60:.0f}:{elapsed%60:02.0f}"
                    
                    print(f"\r[{i+1}/{len(all_dates)}] | filled: {filled_count} | zero-valued: {zero_filled_count} mean: {mean_filled_count} | elapsed: {elapsed_str} ETA: {eta_str}", 
                          end='', flush=True)
            
            # Final statistics
            total_time = time.time() - start_time
            print(f"\nFill complete!")
            print(f"Statistics:")
            print(f"   Total processed: {len(all_dates)} dates")
            print(f"   Newly filled: {filled_count} files")
            print(f"   Zero fill: {zero_filled_count} files")
            print(f"   Mean fill: {mean_filled_count} files")
            print(f"   Total elapsed: {total_time/60:.1f} minutes")
            
            return True
            
        except Exception as e:
            print(f"Fill failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    parser = argparse.ArgumentParser(description="MCD14A1 time-series filling tool")
    parser.add_argument("--start_date", default='20000101', help="Start date (format: YYYYMMDD)")
    parser.add_argument("--end_date", default='20251231', help="End date (format: YYYYMMDD)")
    parser.add_argument("--folder", type=Path, default='/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled_5xdownsampled_merged_unifiedEntireCan_scaledInt/MCD14A1_combined_5xMaximumDown_merged_binary', help="Working folder path (filling happens in-place inside this folder)")
    
    args = parser.parse_args()
    
    # Create the filler
    filler = MCD14A1TimeSeriesFiller(args.folder)
    
    # Begin filling
    success = filler.fill_time_series(args.start_date, args.end_date)
    
    if success:
        print("Time series filling complete!")
    else:
        print("Time series filling failed!")
        exit(1)

if __name__ == "__main__":
    main()
