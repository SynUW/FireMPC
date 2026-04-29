#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Use a template TIFF to clip and fill every TIFF in a folder.

Features:
1. Clip the images in the folder to the template tif's extent.
2. Resize each output to the template's dimensions (same width, height, CRS, and transform as the template).
3. All invalid values (nodata / NaN) inside the template's spatial extent are filled with 0.

Notes:
- Outputs share the same size and spatial reference as the template.
- The template's nodata is ignored: any invalid value inside the template's extent in the input is filled with 0.
- Outputs are written to the output folder with the original filenames.
"""

import os
import argparse
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from tqdm import tqdm
import warnings
import re
from datetime import datetime, timedelta
import calendar

warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)


def get_nodata_mask(raster_array, nodata_value):
    """Get the nodata mask"""
    if nodata_value is None:
        # If there is no nodata value, check NaN
        if np.issubdtype(raster_array.dtype, np.floating):
            return np.isnan(raster_array)
        else:
            # Integer type with no nodata: return all False (everything valid)
            return np.zeros_like(raster_array, dtype=bool)
    else:
        return raster_array == nodata_value


def extract_year_from_filename(filename: str) -> int:
    """
    Extract the year from a filename.
    Filename format: ppp_yyyy_1km_Aggregated.tif
    """
    match = re.search(r'ppp_(\d{4})_1km_Aggregated', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def interpolate_year_to_daily(
    input_tif: str,
    year: int,
    output_folder: str,
    template_tif: str = None,
    resampling: Resampling = Resampling.bilinear,
    preprocessed_data: tuple = None
):
    """
    Interpolate annual data to daily (one file per day).
    
    Args:
        input_tif: input annual TIFF file path
        year: year
        output_folder: output folder path
        template_tif: optional template file (if provided, clip and fill first)
        resampling: resampling method
        preprocessed_data: optional, a (data, profile, nodata) tuple of already-preprocessed data to avoid redundant work
    """
    # If preprocessed data is provided, use it directly
    if preprocessed_data is not None:
        data, profile, nodata = preprocessed_data
    else:
        # If a template file is provided, clip and fill first
        if template_tif:
            # Create a temp file for clipping
            import tempfile
            temp_fd, temp_path = tempfile.mkstemp(suffix='.tif')
            os.close(temp_fd)
            
            try:
                # First clip and fill
                clip_and_fill_with_template(
                    input_tif=input_tif,
                    template_tif=template_tif,
                    output_tif=temp_path,
                    resampling=resampling
                )
                
                # Read the clipped data
                with rasterio.open(temp_path) as temp_src:
                    data = temp_src.read(1)
                    profile = temp_src.profile.copy()
                    nodata = temp_src.nodata
            finally:
                    # Delete temporary file
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
        else:
            # Read the input file directly
            with rasterio.open(input_tif) as src:
                data = src.read(1)  # Read the first band
                profile = src.profile.copy()
                nodata = src.nodata
    
    # Check whether it is a leap year
    is_leap = calendar.isleap(year)
    days_in_year = 366 if is_leap else 365
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate one file per day
    for day_of_year in range(1, days_in_year + 1):
        # Compute date
        date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
        month = date.month
        day = date.day
        
        # Build output filename: ppp_yyyy_mm_dd_1km_Aggregated.tif
        output_filename = f"ppp_{year}_{month:02d}_{day:02d}_1km_Aggregated.tif"
        output_path = os.path.join(output_folder, output_filename)
        
        # Write the daily file (data is identical because annual data is being interpolated)
        output_profile = profile.copy()
        output_profile.update(
            compress='lzw',
            tiled=False,
        )
        
        with rasterio.open(output_path, 'w', **output_profile) as dst:
            dst.write(data, 1)


def process_yearly_to_daily(
    input_folder: str,
    output_folder: str,
    start_year: int = 2000,
    end_year: int = 2025,
    template_tif: str = None,
    resampling: str = 'bilinear'
):
    """
    Interpolate annual data to daily, handling missing years.
    
    Args:
        input_folder: input folder path (contains annual TIFF files)
        output_folder: output folder path
        start_year: start year
        end_year: end year (inclusive)
        template_tif: optional template file (if provided, clip and fill first)
        resampling: resampling method
    """
    import glob
    
    # Parse resampling method
    resampling_map = {
        'nearest': Resampling.nearest,
        'bilinear': Resampling.bilinear,
        'cubic': Resampling.cubic,
        'average': Resampling.average,
        'mode': Resampling.mode,
        'max': Resampling.max,
        'min': Resampling.min,
        'med': Resampling.med,
        'q1': Resampling.q1,
        'q3': Resampling.q3,
    }
    resampling_method = resampling_map.get(resampling.lower(), Resampling.bilinear)
    
    # Find all annual files
    pattern = os.path.join(input_folder, 'ppp_*_1km_Aggregated.tif')
    yearly_files = glob.glob(pattern)
    yearly_files.extend(glob.glob(os.path.join(input_folder, 'ppp_*_1km_Aggregated.TIF')))
    
    # Parse the year-to-file mapping
    year_to_file = {}
    for filepath in yearly_files:
        year = extract_year_from_filename(os.path.basename(filepath))
        if year is not None:
            year_to_file[year] = filepath
    
    if not year_to_file:
        print(f"Warning: in {input_folder}: no annual files found")
        return
    
    available_years = sorted(year_to_file.keys())
    print(f"Found {len(available_years)} annual files: {available_years}")
    print(f"Target year range: {start_year} - {end_year}")
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each target year
    total_days = 0
    success_years = 0
    
    for target_year in range(start_year, end_year + 1):
        # Find the data file used for this year
        if target_year in year_to_file:
            # Data is available; use directly
            source_file = year_to_file[target_year]
            source_year = target_year
        else:
            # No data; use data from the closest year
            # Find the closest year
            if target_year < available_years[0]:
                # Target year is earlier than all available years; use the earliest
                source_year = available_years[0]
            elif target_year > available_years[-1]:
                # Target year is later than all available years; use the latest
                source_year = available_years[-1]
            else:
                # Within available years; find the nearest
                # Prefer the earlier year (more conservative)
                closest_year = available_years[0]
                for avail_year in available_years:
                    if avail_year <= target_year:
                        closest_year = avail_year
                    else:
                        break
                source_year = closest_year
            
            source_file = year_to_file[source_year]
            print(f"Year {target_year} has no data; using {source_year} year's data")
        
        try:
            # Compute the number of days in this year
            is_leap = calendar.isleap(target_year)
            days_in_year = 366 if is_leap else 365
            total_days += days_in_year
            
            # If a template file is provided, preprocess data once
            preprocessed_data = None
            if template_tif:
                import tempfile
                temp_fd, temp_path = tempfile.mkstemp(suffix='.tif')
                os.close(temp_fd)
                
                try:
                    # First clip and fill
                    clip_and_fill_with_template(
                        input_tif=source_file,
                        template_tif=template_tif,
                        output_tif=temp_path,
                        resampling=resampling_method
                    )
                    
                    # Read preprocessed data
                    with rasterio.open(temp_path) as temp_src:
                        data = temp_src.read(1)
                        profile = temp_src.profile.copy()
                        preprocessed_data = (data, profile, temp_src.nodata)
                finally:
                    # Delete temporary file
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
            
            # Interpolate to daily (reuse preprocessed data)
            interpolate_year_to_daily(
                input_tif=source_file,
                year=target_year,
                output_folder=output_folder,
                template_tif=None,  # Already preprocessed
                resampling=resampling_method,
                preprocessed_data=preprocessed_data
            )
            
            success_years += 1
            print(f"Year complete {target_year} ({days_in_year} days)")
            
        except Exception as e:
            print(f"Error: processing year {target_year} failed: {str(e)}")
            continue
    
    print("\nProcessing complete!")
    print(f"Successfully processed: {success_years} years")
    print(f"Generated file: {total_days} daily files")


def clip_and_fill_with_template(
    input_tif: str,
    template_tif: str,
    output_tif: str,
    resampling: Resampling = Resampling.bilinear
):
    """
    Use a template TIFF to clip and fill the target TIFF.
    The output shares the template's size, CRS, and transform.
    All invalid values (nodata / NaN) inside the template's spatial extent are filled with 0.
    
    Args:
        input_tif: input TIFF file path
        template_tif: template TIFF file path
        output_tif: output TIFF file path
        resampling: resampling method
    """
    # Open the template file to get size, CRS, and transform
    with rasterio.open(template_tif) as template_src:
        template_crs = template_src.crs
        template_transform = template_src.transform
        template_width = template_src.width
        template_height = template_src.height
    
    # Open input file
    with rasterio.open(input_tif) as input_src:
        input_crs = input_src.crs
        input_nodata = input_src.nodata
        input_count = input_src.count
        input_dtype = input_src.dtypes[0]
        
        # Build the output profile (using the template's size, CRS, and transform)
        # Temporarily disable tiling to avoid the "zero number of tiles" error
        output_profile = {
            'driver': 'GTiff',
            'width': template_width,
            'height': template_height,
            'count': input_count,
            'dtype': input_dtype,
            'crs': template_crs,
            'transform': template_transform,
            'nodata': input_nodata,
            'compress': 'lzw',
            'tiled': False,  # Disable tiling to avoid errors
        }
        
        # If input is float, add a predictor
        if np.issubdtype(np.dtype(input_dtype), np.floating):
            output_profile['predictor'] = 3
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_tif)), exist_ok=True)
        
        # Write to output file
        with rasterio.open(output_tif, 'w', **output_profile) as dst:
            # Process each band
            for band_idx in range(1, input_count + 1):
                # Create the destination array (using the template's size)
                output_array = np.empty((template_height, template_width), dtype=input_dtype)
                
                # Reproject / resample input data to the template's size and CRS
                reproject(
                    source=rasterio.band(input_src, band_idx),
                    destination=output_array,
                    src_transform=input_src.transform,
                    src_crs=input_crs,
                    dst_transform=template_transform,
                    dst_crs=template_crs,
                    resampling=resampling,
                    src_nodata=input_nodata,
                    dst_nodata=input_nodata,
                )
                
                # Get the invalid-value mask (nodata or NaN)
                invalid_mask = get_nodata_mask(output_array, input_nodata)
                
                # Fill all invalid values inside the template's spatial extent with 0
                # Since the output array already has the template's size and extent, every position is inside the template extent
                output_array[invalid_mask] = 0
                
                # Write band
                dst.write(output_array, band_idx)


def process_folder(
    input_folder: str,
    template_tif: str,
    output_folder: str,
    resampling: str = 'bilinear',
    pattern: str = '*.tif'
):
    """
    Batch-process all TIFF files in a folder.
    
    Args:
        input_folder: input folder path
        template_tif: template TIFF file path
        output_folder: output folder path
        resampling: resampling method name ('nearest', 'bilinear', 'cubic', 'average', etc.)
        pattern: file match pattern (default '*.tif')
    """
    import glob
    
    # Parse resampling method
    resampling_map = {
        'nearest': Resampling.nearest,
        'bilinear': Resampling.bilinear,
        'cubic': Resampling.cubic,
        'average': Resampling.average,
        'mode': Resampling.mode,
        'max': Resampling.max,
        'min': Resampling.min,
        'med': Resampling.med,
        'q1': Resampling.q1,
        'q3': Resampling.q3,
    }
    resampling_method = resampling_map.get(resampling.lower(), Resampling.bilinear)
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all TIFF files
    tif_files = []
    for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
        tif_files.extend(glob.glob(os.path.join(input_folder, ext)))
    
    if not tif_files:
        print(f"Warning: in {input_folder}: no TIFF files found")
        return
    
    print(f"Found {len(tif_files)} TIFF files")
    print(f"Template file: {template_tif}")
    print(f"Output directory: {output_folder}")
    print(f"Resampling method: {resampling}\n")
    
    # Process each file
    success_count = 0
    fail_count = 0
    
    for input_tif in tqdm(tif_files, desc="Process files"):
        try:
            # Build the output filename (keep the original)
            filename = os.path.basename(input_tif)
            output_tif = os.path.join(output_folder, filename)
            
            # Perform clipping and filling
            clip_and_fill_with_template(
                input_tif=input_tif,
                template_tif=template_tif,
                output_tif=output_tif,
                resampling=resampling_method
            )
            success_count += 1
            
        except Exception as e:
            print(f"\nError: processing {os.path.basename(input_tif)} failed: {str(e)}")
            fail_count += 1
            continue
    
    print("\nProcessing complete!")
    print(f"Succeeded: {success_count} files")
    print(f"Failed: {fail_count} files")


def main():
    parser = argparse.ArgumentParser(
        description="Use a template TIFF to clip and fill every TIFF in a folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python worldpop_clip_fill.py \\
    --input_folder /path/to/input \\
    --template /path/to/template.tif \\
    --output_folder /path/to/output
  
  # Use nearest-neighbor resampling (suitable for categorical data)
  python worldpop_clip_fill.py \\
    --input_folder /path/to/input \\
    --template /path/to/template.tif \\
    --output_folder /path/to/output \\
    --resampling nearest
        """
    )
    
    parser.add_argument(
        '--input_folder',
        type=str,
        default='/mnt/raid/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/worldpop',
        help='Input folder path (containing TIFFs to be processed)'
    )
    
    parser.add_argument(
        '--template',
        type=str,
        default='/mnt/raid/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MYD09GA_QA_state1km_2002_07_10_merge.tif',
        help='Template TIFF file path'
    )
    
    parser.add_argument(
        '--output_folder',
        type=str,
        default='/mnt/raid/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/worldpop_clip',
        help='Output folder path'
    )
    
    parser.add_argument(
        '--resampling',
        type=str,
        default='bilinear',
        choices=['nearest', 'bilinear', 'cubic', 'average', 'mode', 'max', 'min', 'med', 'q1', 'q3'],
        help='Resampling method (default: bilinear)'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.tif',
        help='File match pattern (default: *.tif)'
    )
    
    parser.add_argument(
        '--interpolate_to_daily',
        action='store_true',
        help='Interpolate annual data to daily (one file per day)'
    )
    
    parser.add_argument(
        '--start_year',
        type=int,
        default=2000,
        help='Interpolation start year (default: 2000)'
    )
    
    parser.add_argument(
        '--end_year',
        type=int,
        default=2025,
        help='Interpolation end year (default: 2025, inclusive)'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.isdir(args.input_folder):
        raise ValueError(f"Input folder does not exist: {args.input_folder}")
    
    if args.interpolate_to_daily:
        # Annual-to-daily interpolation mode
        if args.template and not os.path.isfile(args.template):
            print(f"Warning: template file does not exist: {args.template}; clipping without template")
            args.template = None
        
        process_yearly_to_daily(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            start_year=args.start_year,
            end_year=args.end_year,
            template_tif=args.template,
            resampling=args.resampling
        )
    else:
        # Plain clipping mode
        if not os.path.isfile(args.template):
            raise ValueError(f"Template file does not exist: {args.template}")
        
        process_folder(
            input_folder=args.input_folder,
            template_tif=args.template,
            output_folder=args.output_folder,
            resampling=args.resampling,
            pattern=args.pattern
        )


if __name__ == '__main__':
    main()

