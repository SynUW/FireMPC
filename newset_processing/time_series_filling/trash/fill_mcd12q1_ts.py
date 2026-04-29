#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LULC type
MCD12Q1 annual -> daily "interpolation" (forward-copy to fill the time series; inclusive range)
========================================================

The input directory has one TIF per year, named like yyyy_mm_dd.tif (usually yyyy_01_01.tif).
Within the inclusive range [--start, --end], the script generates one output TIF per day:
- Within the year a given day belongs to, use that year's image (forward-copy from that year's date through every date before the next year's file appears).
- If the series starts earlier than the earliest year (e.g. no 2000 file), fill with the earliest year's image (e.g. 2001).
- If the series ends later than the latest year, fill with the latest year's image.
- Output uses hard links by default to save space; copy or symlink are also options.

MCD09GA_b1237_mosaic_withQA_QAapplied
MCD09GA_b1237_mosaic_withQA_QAapplied

Example
----
python mcd12q1_to_daily.py \
  --input_dir /path/to/yearly \
  --output_dir /path/to/daily \
  --start 2000-01-01 \
  --end   2005-12-31 \
  --mode hardlink --overwrite

Arguments
----
--input_dir   : directory containing yyyy_mm_dd.tif
--output_dir  : per-day output directory (auto-created)
--start       : start date (inclusive), format YYYY-MM-DD
--end         : end date (inclusive), format YYYY-MM-DD
--mode        : output mode: hardlink | copy | symlink (default: hardlink)
--overwrite   : overwrite if the target exists
"""

import argparse
from datetime import datetime, timedelta, date
from pathlib import Path
import re
import os
import shutil
from typing import Dict, List, Tuple
from tqdm import tqdm

DATE_RE = re.compile(r"^(\d{4})_(\d{2})_(\d{2})\.tif$", re.IGNORECASE)

def parse_args():
    p = argparse.ArgumentParser(description="Convert MCD12Q1 annual images to daily (forward copy)")
    p.add_argument("--input_dir", default='/mnt/raid/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x/MCD12Q1_mosaic_downsampled', type=Path, help="Input directory (contains yyyy_mm_dd.tif)")
    p.add_argument("--output_dir", default='/mnt/raid/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x_norm/MCD12Q1_mosaic_downsampled', type=Path, help="Output directory (per-day TIFs)")
    p.add_argument("--start", default='2000-01-01', type=str, help="Start date (inclusive), YYYY-MM-DD")
    p.add_argument("--end", default='2024-12-31', type=str, help="End date (inclusive), YYYY-MM-DD")
    p.add_argument("--mode", default="copy", choices=["hardlink", "copy", "symlink"],
                   help="Output mode (hardlink is recommended to save space)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite if it exists")
    return p.parse_args()

def find_yearly_files(input_dir: Path) -> Dict[int, Path]:
    """
    Scan a directory for files like yyyy_mm_dd.tif; if multiple files appear in the same year, take the earliest date.
    Returns a year -> filepath mapping.
    """
    year_to_file: Dict[int, Tuple[date, Path]] = {}
    for p in input_dir.iterdir():
        if not p.is_file():
            continue
        m = DATE_RE.match(p.name)
        if not m:
            continue
        y, mm, dd = map(int, m.groups())
        d = date(y, mm, dd)
        if y not in year_to_file or d < year_to_file[y][0]:
            year_to_file[y] = (d, p)

    # Keep only the path
    return {y: tup[1] for y, tup in year_to_file.items()}

def daterange_closed(d0: date, d1: date):
    """Per-day generator for the inclusive range [d0, d1]."""
    cur = d0
    one = timedelta(days=1)
    while cur <= d1:
        yield cur
        cur += one

def select_source_for_day(d: date, year_keys_sorted: List[int], year_to_file: Dict[int, Path]) -> Path:
    """
    For a given day d, pick the "year image" to use:
      - If a file exists for y = d.year, use that year's file.
      - Otherwise, use the largest known year <= d.year.
      - If d is earlier than the earliest year, use the earliest year's file.
      - If d is later than the latest year, use the latest year's file.
    """
    # Exact-match this year
    if d.year in year_to_file:
        return year_to_file[d.year]

    # Find the largest year <= d.year
    candidate = None
    for y in year_keys_sorted:
        if y <= d.year:
            candidate = y
        else:
            break

    if candidate is not None:
        return year_to_file[candidate]

    # Earlier than the earliest year
    return year_to_file[year_keys_sorted[0]]

def ensure_output(p: Path, mode: str, src: Path, overwrite: bool):
    """
    Generate a file at p from the source src using the specified mode.
    mode: hardlink | copy | symlink
    """
    if p.exists():
        if overwrite:
            p.unlink()
        else:
            return  # Skip

    p.parent.mkdir(parents=True, exist_ok=True)

    if mode == "hardlink":
        try:
            os.link(src, p)
        except OSError:
            # Fall back to copying when cross-filesystem hardlinks fail
            shutil.copy2(src, p)
    elif mode == "symlink":
        try:
            os.symlink(src, p)
        except OSError:
            # Some environments forbid symlinks; fall back to copying
            shutil.copy2(src, p)
    else:  # copy
        shutil.copy2(src, p)

def main():
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end   = datetime.strptime(args.end, "%Y-%m-%d").date()
    assert start <= end, "start must be <= end (inclusive range)"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    year_to_file = find_yearly_files(input_dir)
    if not year_to_file:
        raise RuntimeError(f"No files matching yyyy_mm_dd.tif found in the input directory: {input_dir}")

    # Sort years
    years_sorted = sorted(year_to_file.keys())

    # Print key information
    print(f"Found year files: {len(years_sorted)} items; year range: {years_sorted[0]}—{years_sorted[-1]}")
    for y in years_sorted:
        print(f"  {y} -> {year_to_file[y].name}")

    # Iterate the inclusive range and emit per-day output
    total_days = (end - start).days + 1
    for d in tqdm(daterange_closed(start, end), total=total_days, desc="Generating per-day"):
        src = select_source_for_day(d, years_sorted, year_to_file)
        out_name = f"{d.year:04d}_{d.month:02d}_{d.day:02d}.tif"
        out_path = output_dir / out_name
        ensure_output(out_path, args.mode, src, args.overwrite)

    print("Done!")

if __name__ == "__main__":
    main()
