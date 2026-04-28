#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCD15A3H time-series "forward-fill" completion script (per day).
- Operates in place in the source directory; no output directory needed.
- GeoTIFFs in the input directory follow yyyy_mm_dd.tif (e.g. 2023_02_03.tif).
- Within the given range [start_date, end_date], generate GeoTIFFs day by day:
  * If the current day's file exists: skip.
  * If missing: fill with the "nearest past date" image (forward fill).
  * If before the earliest available date: keep filling with the earliest date's image.
  * If after the latest available date: keep extrapolating with the latest date's image.

Example:
python fill_mcd15a3h_ts.py \
  --folder /path/to/mcd15a3h \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --workers 8
"""

import argparse
import re
import sys
import shutil
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm is not installed: pip install tqdm", file=sys.stderr)
    tqdm = None

DATE_RE = re.compile(r"^(\d{4})_(\d{2})_(\d{2})\.tif$", re.IGNORECASE)

def parse_args():
    ap = argparse.ArgumentParser(description="MCD15A3H forward-fill completion (per day)")
    ap.add_argument("--folder", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x_norm/MCD15A3H_mosaic_249_filled_downsampled', help="Working directory (fill in place in the source directory)")
    ap.add_argument("--start-date", default='2000-01-01', help="Start date, YYYY-MM-DD")
    ap.add_argument("--end-date", default='2024-12-31', help="End date, YYYY-MM-DD (inclusive)")
    ap.add_argument("--workers", type=int, default=8, help="Number of concurrent threads (I/O copy parallelism)")
    return ap.parse_args()

def to_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def scan_input(input_dir: Path) -> Dict[date, Path]:
    """Scan the input directory and parse yyyy_mm_dd.tif files; return {date: path}"""
    mapping: Dict[date, Path] = {}
    for p in input_dir.glob("*.tif"):
        m = DATE_RE.match(p.name)
        if not m:
            continue
        y, mth, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            dt = date(y, mth, d)
        except ValueError:
            continue
        # If multiple files share the same day, keep the first; could be replaced with a more sophisticated strategy
        mapping.setdefault(dt, p)
    return mapping

def daterange(d0: date, d1: date):
    """Per-day iterator that includes both endpoints"""
    cur = d0
    one = timedelta(days=1)
    while cur <= d1:
        yield cur
        cur += one

def build_fill_plan(available: List[date], start: date, end: date) -> Dict[date, date]:
    """
    For each day in [start, end], determine the "source date" (which day's image to fill with).
    Rules:
      - If the day exists: use it.
      - If missing: use the nearest past date.
      - If before the earliest available date: use the earliest date.
      - If after the latest available date: use the latest date.
    """
    if not available:
        raise ValueError("No files matching yyyy_mm_dd.tif found in the input directory.")

    available_sorted = sorted(available)
    earliest = available_sorted[0]
    latest = available_sorted[-1]

    plan: Dict[date, date] = {}
    prev_seen: Optional[date] = None

    for d in daterange(start, end):
        if d in available:
            prev_seen = d
            plan[d] = d
        else:
            if prev_seen is not None and prev_seen <= d:
                # Use the nearest past date
                plan[d] = prev_seen
            else:
                # No available date encountered yet (at the very start of the series)
                if d < earliest:
                    plan[d] = earliest  # Back-fill with the earliest date
                elif d > latest:
                    plan[d] = latest    # Extrapolate with the latest date
                else:
                    # Should not reach here in theory; do one leftward search for safety
                    # (Extreme case where prev_seen is empty but d is within the available range and data exists to the right)
                    # Implement a linear backtrack: from d-1 backwards, find the first day with available data
                    back = d - timedelta(days=1)
                    while back >= earliest and back not in available:
                        back -= timedelta(days=1)
                    plan[d] = back if back >= earliest else earliest
    return plan


def materialize(src: Path, dst: Path):
    """Copy a file to the destination"""
    if dst.exists():
        return  # Skip existing files
    
    # Copy the file directly
    shutil.copy2(src, dst)  # Preserve timestamps

def main():
    args = parse_args()
    folder = Path(args.folder)

    start = to_date(args.start_date)
    end = to_date(args.end_date)
    if end < start:
        raise SystemExit("end-date is earlier than start-date.")

    mapping = scan_input(folder)
    if not mapping:
        raise SystemExit("No files matching yyyy_mm_dd.tif found in the working directory.")

    plan = build_fill_plan(sorted(mapping.keys()), start, end)

    tasks = []
    for d, src_d in plan.items():
        src_path = mapping.get(src_d)
        if src_path is None:
            # May be None in rare edge cases (concurrency / external deletion)
            raise SystemExit(f"Source file not found: {src_d}")
        
        # Skip if the target file already exists
        dst_name = f"{d:%Y_%m_%d}.tif"
        dst_path = folder / dst_name
        if dst_path.exists():
            continue
            
        tasks.append((src_path, dst_path))

    total = len(tasks)
    print(f"Time range: {start} to {end}")
    print(f"Working directory: {folder}")
    print(f"Found {len(mapping)} existing files")
    print(f"Need to fill {total} missing files")

    def _worker(item):
        src, dst = item
        materialize(src, dst)
        return dst.name

    if tqdm is None:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            completed = 0
            for _ in as_completed(ex.submit(_worker, t) for t in tasks):
                completed += 1
                print(f"\r[{completed}/{total}] fill progress", end='', flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for _ in tqdm(as_completed(ex.submit(_worker, t) for t in tasks),
                          total=total, desc="Fill missing files"):
                pass

    print(f"\n[Done] in {folder} filled {total} missing files")
    print(f"Time range: {start} to {end}")

if __name__ == "__main__":
    main()
