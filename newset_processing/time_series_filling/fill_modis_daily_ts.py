#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MODIS time-series forward-fill tool.
Fills missing time points using the forward-fill strategy:
1. For gaps in the series, fill with data from the nearest preceding date.
2. For dates missing before the series begins, fill with data from the nearest future date.

Use this fill method for any daily data, including MCD09GA, NDVI-EVI, MCD11A1, MCD09CMG.
"""

import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from osgeo import gdal
import time
import shutil
import re

# Configure GDAL
gdal.UseExceptions()
gdal.PushErrorHandler('CPLQuietErrorHandler')

class MODISTimeSeriesFiller:
    def __init__(self, input_folder: Path):
        self.input_folder = input_folder
        
    def parse_date_from_filename(self, filename: str) -> Optional[datetime]:
        """Parse the date from the filename"""
        stem = Path(filename).stem
        # Supports common formats:
        # - YYYY_MM_DD / YYYY-MM-DD
        # - YYYYMMDD
        # - With prefix / suffix: xxx_YYYY_MM_DD_xxx.tif
        patterns = [
            r"(?P<y>\d{4})[_-](?P<m>\d{2})[_-](?P<d>\d{2})",
            r"(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})",
        ]
        for pat in patterns:
            m = re.search(pat, stem)
            if not m:
                continue
            try:
                y = int(m.group("y"))
                mo = int(m.group("m"))
                d = int(m.group("d"))
                return datetime(y, mo, d)
            except Exception:
                continue
        return None
    
    def get_existing_files(self) -> Tuple[dict, set]:
        """Get existing files and their dates + an existing-filename set (avoiding per-day exists / stat)"""
        existing_files = {}
        existing_names = set()

        patterns = ("*.tif", "*.TIF", "*.tiff", "*.TIFF")
        for pat in patterns:
            for file_path in self.input_folder.glob(pat):
                existing_names.add(file_path.name)
                date = self.parse_date_from_filename(file_path.name)
                if date:
                    existing_files[date] = file_path

        return existing_files, existing_names
    
    def copy_tiff_with_metadata(self, source_file: Path, target_file: Path) -> bool:
        """Copy a TIFF file, preserving all metadata"""
        try:
            shutil.copy2(source_file, target_file)
            return True
        except Exception as e:
            print(f"Failed to copy file {source_file.name} -> {target_file.name}: {e}")
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
            existing_files, existing_names = self.get_existing_files()
            print(f"Found {len(existing_files)} existing files")
            
            # Generate the full date sequence
            current_date = start_dt
            all_dates = []
            while current_date <= end_dt:
                all_dates.append(current_date)
                current_date += timedelta(days=1)
            
            missing_count_est = len(all_dates) - len(existing_files)
            print(f"Need to fill {missing_count_est} dates")
            
            # Check whether existing files are present
            if not existing_files:
                print("No existing files found")
                return False

            # If coverage is already complete, exit quickly (avoid lots of exists / stat on slow storage)
            if missing_count_est <= 0:
                print("No missing dates found; nothing to fill; exited quickly.")
                return True
            
            # Get the earliest and latest dates of existing files
            existing_dates = sorted(existing_files.keys())
            earliest_date = existing_dates[0]
            latest_date = existing_dates[-1]
            
            print(f"Existing data range: {earliest_date.strftime('%Y_%m_%d')} - {latest_date.strftime('%Y_%m_%d')}")
            existing_date_set = set(existing_dates)
            
            # Begin filling
            start_time = time.time()
            filled_count = 0
            forward_filled_count = 0  # Fill early gaps with future data
            gap_filled_count = 0      # Fill gaps in the time series

            # Process only missing dates to avoid output_file.exists() on every date (slow on /s3)
            missing_dates = [d for d in all_dates if d not in existing_date_set]
            missing_dates.sort()

            # Linear pointer: quickly find the nearest existing date <= target_date
            idx = 0
            last_leq_date = None

            for i, target_date in enumerate(missing_dates):
                output_filename = target_date.strftime("%Y_%m_%d.tif")

                # Do not overwrite an existing same-named file (protective even if the date cannot be parsed)
                if output_filename in existing_names:
                    continue

                output_file = self.input_folder / output_filename
                
                # Cases that need filling
                source_file = None
                
                if target_date < earliest_date:
                    # Early gap: use the earliest existing file (future data)
                    source_file = existing_files[earliest_date]
                    forward_filled_count += 1
                else:
                    while idx < len(existing_dates) and existing_dates[idx] <= target_date:
                        last_leq_date = existing_dates[idx]
                        idx += 1

                    if last_leq_date is None:
                        source_file = existing_files[earliest_date]
                    else:
                        source_file = existing_files[last_leq_date]
                    gap_filled_count += 1
                
                # Copy file
                if source_file and self.copy_tiff_with_metadata(source_file, output_file):
                    filled_count += 1
                    existing_names.add(output_filename)
                
                # Update progress
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (i + 1)
                    remaining = (len(missing_dates) - i - 1) * avg_time
                    eta_str = f"{remaining/60:.0f}:{remaining%60:02.0f}" if remaining > 0 else "0:00"
                    elapsed_str = f"{elapsed/60:.0f}:{elapsed%60:02.0f}"
                    
                    print(f"\r[{i+1}/{len(missing_dates)}] | filled: {filled_count} | forward fill: {forward_filled_count} gap fill: {gap_filled_count} | elapsed: {elapsed_str} ETA: {eta_str}", 
                          end='', flush=True)
            
            # Final statistics
            total_time = time.time() - start_time
            print(f"\nFill complete!")
            print(f"Statistics:")
            print(f"   Total processed: {len(all_dates)} dates")
            print(f"   Newly filled: {filled_count} files")
            print(f"   Forward fill: {forward_filled_count} files (using future data to fill early gaps)")
            print(f"   Gap fill: {gap_filled_count} files (using prior data to fill gaps)")
            print(f"   Total elapsed: {total_time/60:.1f} minutes")
            
            return True
            
        except Exception as e:
            print(f"Fill failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    parser = argparse.ArgumentParser(description="MODIS time-series forward-fill tool")
    parser.add_argument("--start_date", default='20000101', help="Start date (format: YYYYMMDD)")
    parser.add_argument("--end_date", default='20251231', help="End date (format: YYYYMMDD)")
    parser.add_argument("--folder", type=Path, 
                       default='/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled_5xdownsampled_merged_unifiedEntireCan_scaledInt/MCD09CMG_QAapplied_spatialFilled_merged', 
                       help="Working folder path (filling happens in-place inside this folder)")
    
    args = parser.parse_args()
    
    # Create the filler
    filler = MODISTimeSeriesFiller(args.folder)
    
    # Begin filling
    success = filler.fill_time_series(args.start_date, args.end_date)
    
    if success:
        print("Time series filling complete!")
    else:
        print("Time series filling failed!")
        exit(1)

if __name__ == "__main__":
    main()
