#!/usr/bin/env python3
"""
Generate a batch of date-named ASTER image files in the output directory for the specified date range:
- Copy a single template TIF to one file per date: ASTER_yyyy_mm_dd.tif
- Only renames / copies; does not modify file contents or projection metadata
- Supports parallel copies and skipping existing files

Example:
python3 interpolate_files.py \
  --src-file /path/to/template.tif \
  --output-dir /path/to/out \
  --start-date 2000-01-01 --end-date 2000-12-31 \
  --max-workers 16 --overwrite
"""

import os
import shutil
import logging
import argparse
from datetime import datetime, timedelta
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('interpolate_files')


def build_dates(start: str, end: str) -> List[datetime]:
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    if e < s:
        raise ValueError("end-date is earlier than start-date")
    days = (e - s).days + 1
    return [s + timedelta(days=i) for i in range(days)]


def out_name_for_date(d: datetime, prefix: str = "ASTER") -> str:
    return f"{prefix}_{d.year:04d}_{d.month:02d}_{d.day:02d}.tif"


def copy_one(template: str, out_path: str, overwrite: bool) -> bool:
    if os.path.exists(out_path) and not overwrite:
        return True  # Treat as success and skip
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    shutil.copy2(template, out_path)
    return True


def main():
    ap = argparse.ArgumentParser(description='Copy a template TIF over a date range as ASTER_yyyy_mm_dd.tif')
    ap.add_argument('--src-file', default='/mnt/raid/user/wildfire_dataset/self_built_materials/all_data_masked/DEM_and_distance_map/Topo_and_distance_map_stack_nodata_homogenized_downsampled10x_log.tif', help='Template TIF file path (will be copied to per-date files)')
    ap.add_argument('--output-dir', default='/mnt/raid/user/wildfire_dataset/self_built_materials/all_data_masked/DEM_and_distance_map_interpolated_10x', help='Output directory (will be created)')
    ap.add_argument('--start-date', default='2000-01-01', help='Start date YYYY-MM-DD (inclusive)')
    ap.add_argument('--end-date', default='2024-12-31', help='End date YYYY-MM-DD (inclusive)')
    ap.add_argument('--prefix', default='TopoDistance', help='Output filename prefix, default ASTER')
    ap.add_argument('--max-workers', type=int, default=16, help='Number of parallel processes')
    ap.add_argument('--overwrite', action='store_true', help='Overwrite if the target file already exists')

    args = ap.parse_args()

    template = args.src_file
    if not os.path.isfile(template):
        logger.error(f"Template file does not exist: {template}")
        return

    dates = build_dates(args.start_date, args.end_date)

    os.makedirs(args.output_dir, exist_ok=True)

    total = len(dates)
    done = 0
    ok = 0

    def progress():
        print(f"\rProgress {done}/{total} success: {ok}", end='', flush=True)

    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = []
        for d in dates:
            name = out_name_for_date(d, args.prefix)
            dst = os.path.join(args.output_dir, name)
            futs.append(ex.submit(copy_one, template, dst, args.overwrite))
        for f in as_completed(futs):
            res = f.result()
            done += 1
            if res:
                ok += 1
            progress()

    print()
    logger.info(f"Done: {ok}/{total} files generated -> {args.output_dir}")


if __name__ == '__main__':
    main()
