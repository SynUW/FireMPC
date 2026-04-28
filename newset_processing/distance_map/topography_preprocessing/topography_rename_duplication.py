"""
Copy and rename the clipped topography file into per-day files
"""
import os
import shutil
from datetime import datetime, timedelta
import glob
import re


def parse_filename(filename):
    """Parse the date from the filename"""
    # Remove extension
    basename = os.path.splitext(filename)[0]

    # Use a regex to match the date portion of the filename
    match = re.match(r'(\d{4})_(\d+)_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})', basename)
    if not match:
        raise ValueError(f"Filename format is invalid: {filename}")

    year = match.group(1)
    fire_code = match.group(2)
    start_date_str = match.group(3)
    end_date_str = match.group(4)

    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    except ValueError as e:
        raise ValueError(f"Date parse error: {str(e)}")

    return f"{year}_{fire_code}", start_date, end_date


def generate_date_range(start_date, end_date, buffer_days=6):
    """Generate a date range (extended by buffer_days days on each side)"""
    extended_start = start_date - timedelta(days=buffer_days)
    extended_end = end_date + timedelta(days=buffer_days)

    # Generate every day in the date range
    current_date = extended_start
    while current_date <= extended_end:
        yield current_date
        current_date += timedelta(days=1)


def copy_and_rename_tif_files(input_dir, output_dir):
    """Copy and rename TIFF files"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all TIFF files
    tif_files = glob.glob(os.path.join(input_dir, '*.tif'))

    total_files = len(tif_files)
    processed_files = 0

    for tif_file in tif_files:
        try:
            filename = os.path.basename(tif_file)
            fire_id, start_date, end_date = parse_filename(filename)

            # Generate every day in the date range
            date_range = list(generate_date_range(start_date, end_date))

            print(f"Processing file: {filename} (total {len(date_range)} days)")

            for current_date in date_range:
                # Build new filename
                date_str = current_date.strftime("%Y-%m-%d")
                new_filename = f"{fire_id}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}_{date_str}_Topography.tif"
                output_path = os.path.join(output_dir, new_filename)

                # Copy file
                shutil.copy2(tif_file, output_path)
                print(f"  Creating copy: {new_filename}")

            processed_files += 1
            print(f"Processing complete: {filename} ({processed_files}/{total_files})\n")

        except Exception as e:
            print(f"Processing file {filename} error: {str(e)}")
            continue

    print(f"\nProcessing complete! Total processed {processed_files}/{total_files} files")

if __name__ == '__main__':
    # Configure paths (modify according to actual setup)
    input_directory = r"D:\wildfire_dataset\self_built\drivers\Topography_clip"  # Input TIFF file directory
    output_directory = r"D:\wildfire_dataset\self_built\drivers\Topography_clip_everyday"  # Output directory

    # Execute copy and rename
    copy_and_rename_tif_files(input_directory, output_directory)