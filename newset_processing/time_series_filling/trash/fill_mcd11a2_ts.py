#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-series TIFF data reconstruction tool
======================

Features:
1. Scan an input folder for date-named TIFFs (yyyy_mm_dd.tif).
2. Within the specified date range, generate TIFFs for missing dates.
3. Fill strategy:
   - Forward fill first: use data from the nearest past date.
   - If there is no past data, backward fill: use data from the nearest future date.
4. Preserve TIFF metadata (georeferencing, projection, NoData, etc.).

Usage:
  python rebuild_timeseries.py \
    --input-dir /path/to/sparse_tiffs \
    --output-dir /path/to/dense_tiffs \
    --start-date 2000-01-01 \
    --end-date 2024-12-31

  # Overwrite existing files
  python rebuild_timeseries.py \
    --input-dir /path/to/sparse_tiffs \
    --output-dir /path/to/dense_tiffs \
    --start-date 2000-01-01 \
    --end-date 2024-12-31 \
    --overwrite
"""

import re
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional
import shutil

from osgeo import gdal
from tqdm import tqdm

# Disable GDAL error output
gdal.PushErrorHandler('CPLQuietErrorHandler')


def parse_date_from_filename(filename: str) -> Optional[datetime]:
    """
    Parse the date from a filename (yyyy_mm_dd.tif or yyyy-mm-dd.tif).
    """
    patterns = [
        r"(\d{4})[_-](\d{2})[_-](\d{2})\.tif",  # yyyy_mm_dd.tif or yyyy-mm-dd.tif
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            year, month, day = match.groups()
            try:
                return datetime(int(year), int(month), int(day))
            except ValueError:
                continue
    return None


def scan_existing_files(input_dir: Path) -> Dict[datetime, Path]:
    """
    Scan the input directory and build a date -> file-path mapping.
    """
    date_file_map = {}
    
    tiff_files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
    
    for tiff_file in tiff_files:
        date = parse_date_from_filename(tiff_file.name)
        if date:
            date_file_map[date] = tiff_file
    
    return date_file_map


def generate_date_range(start_date: datetime, end_date: datetime):
    """
    Generate every date in the date range.
    """
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def find_fill_source(target_date: datetime, 
                     existing_dates: Dict[datetime, Path],
                     prefer_forward: bool = True) -> Optional[Path]:
    """
    Find a fill-data source for a target date.
    
    Strategy:
    1. Forward fill first: find the nearest past date.
    2. If no past data, backward fill: find the nearest future date.
    
    Args:
        target_date: target date
        existing_dates: dict of dates with existing data
        prefer_forward: prefer forward fill
    
    Returns:
        source file path, or None if no source is found
    """
    if not existing_dates:
        return None
    
    sorted_dates = sorted(existing_dates.keys())
    
    # Forward fill: find the nearest past date
    past_dates = [d for d in sorted_dates if d < target_date]
    if past_dates:
        nearest_past = max(past_dates)  # Nearest past date
        return existing_dates[nearest_past]
    
    # Backward fill: find the nearest future date
    future_dates = [d for d in sorted_dates if d > target_date]
    if future_dates:
        nearest_future = min(future_dates)  # Nearest future date
        return existing_dates[nearest_future]
    
    return None


def copy_tiff_file(src_path: Path, dst_path: Path):
    """
    Copy a TIFF file (preserving all metadata).
    """
    shutil.copy2(src_path, dst_path)


def format_date_filename(date: datetime) -> str:
    """
    Format a date as a filename (yyyy_mm_dd.tif).
    """
    return f"{date.year:04d}_{date.month:02d}_{date.day:02d}.tif"


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct time-series TIFF data and fill missing dates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--input-dir", type=str, required=True,
                       help="Input TIFF folder path")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output TIFF folder path")
    parser.add_argument("--start-date", type=str, default='2000-01-01',
                       help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default='2024-12-31',
                       help="End date (YYYY-MM-DD)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing output files")
    parser.add_argument("--dry-run", action="store_true",
                       help="Dry run; do not actually copy files")
    
    args = parser.parse_args()
    
    # Parse arguments
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse date range
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Date format error (expected YYYY-MM-DD): {e}")
    
    if start_date > end_date:
        raise ValueError("Start date must not be later than end date")
    
    print("=" * 70)
    print("Time-series data reconstruction tool")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Time range: {start_date.date()} to {end_date.date()}")
    print(f"Overwrite mode: {'yes' if args.overwrite else 'no'}")
    print(f"Dry run: {'yes' if args.dry_run else 'no'}")
    print("=" * 70)
    
    # Scan existing files
    print("\n[1/3] Scanning input files...")
    existing_dates = scan_existing_files(input_dir)
    
    if not existing_dates:
        raise FileNotFoundError(f"No valid TIFF files found in the input directory")
    
    sorted_dates = sorted(existing_dates.keys())
    print(f"Found {len(existing_dates)} valid files")
    print(f"  Earliest date: {min(sorted_dates).date()}")
    print(f"  Latest date: {max(sorted_dates).date()}")
    
    # Generate the complete date range
    print("\n[2/3] Analyzing missing dates...")
    total_days = (end_date - start_date).days + 1
    all_dates = list(generate_date_range(start_date, end_date))
    
    missing_dates = [d for d in all_dates if d not in existing_dates]
    
    print(f"Total days: {total_days}")
    print(f"  Existing data: {len(existing_dates)} days")
    print(f"  Missing data: {len(missing_dates)} days")
    print(f"  Coverage: {len(existing_dates)/total_days*100:.1f}%")
    
    # Fill missing dates
    print("\n[3/3] Filling missing dates...")
    
    stats = {
        'copied': 0,      # Files copied from existing inputs
        'forward': 0,     # Forward fill
        'backward': 0,    # Backward fill
        'skipped': 0,     # Already exists; skip
        'failed': 0       # Failed
    }
    
    with tqdm(total=total_days, desc="Processing progress", ncols=100) as pbar:
        for date in all_dates:
            output_file = output_dir / format_date_filename(date)
            
            # Check if output file already exists
            if output_file.exists() and not args.overwrite:
                stats['skipped'] += 1
                pbar.update(1)
                continue
            
            # If existing data, copy directly
            if date in existing_dates:
                src_file = existing_dates[date]
                
                if not args.dry_run:
                    try:
                        copy_tiff_file(src_file, output_file)
                        stats['copied'] += 1
                    except Exception as e:
                        print(f"\nCopy failed {date.date()}: {e}")
                        stats['failed'] += 1
                else:
                    stats['copied'] += 1
            
            # If missing, find a fill source
            else:
                src_file = find_fill_source(date, existing_dates)
                
                if src_file is None:
                    print(f"\nCannot find a fill source: {date.date()}")
                    stats['failed'] += 1
                    pbar.update(1)
                    continue
                
                # Decide forward or backward fill
                src_date = parse_date_from_filename(src_file.name)
                is_forward = src_date < date
                
                if not args.dry_run:
                    try:
                        copy_tiff_file(src_file, output_file)
                        if is_forward:
                            stats['forward'] += 1
                        else:
                            stats['backward'] += 1
                    except Exception as e:
                        print(f"\nFill failed {date.date()}: {e}")
                        stats['failed'] += 1
                else:
                    if is_forward:
                        stats['forward'] += 1
                    else:
                        stats['backward'] += 1
            
            pbar.update(1)
    
    # Print statistics
    print("\n" + "=" * 70)
    print("Processing complete!")
    print("=" * 70)
    print(f"Total files: {total_days}")
    print(f"  Direct copy (existing data): {stats['copied']}")
    print(f"  Forward fill (past -> future): {stats['forward']}")
    print(f"  Backward fill (future -> past): {stats['backward']}")
    print(f"  Skipped (already exists): {stats['skipped']}")
    print(f"  Failed: {stats['failed']}")
    print("=" * 70)
    
    if args.dry_run:
        print("\nThis was a dry run; no files were actually modified")
    else:
        print(f"\nOutput directory: {output_dir}")
        print(f"Time series fully reconstructed ({start_date.date()} to {end_date.date()})")


if __name__ == "__main__":
    main()

