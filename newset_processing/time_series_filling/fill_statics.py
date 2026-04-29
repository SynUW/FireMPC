#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Static-variable time filling (annual expanded to daily).
================================

Supports two input storage layouts (in the same directory):
1) Per-day files: YYYY_MM_DD.tif
2) Annual files: {driver}_YYYY.tif (driver names may contain underscores)

Rules:
- In either layout, use only the year (yyyy) information and expand that year's image to every day of the year (generating YYYY_MM_DD.tif).
- If a year is missing, back-fill with the nearest preceding year's image.
  - If the very start is missing and no preceding year is available, fall back to the earliest available year.

Notes:
- By default, fill in / generate per-day files (YYYY_MM_DD.tif) in the same directory.
- To speed up slow storage like /s3: do not call Path.exists() per day; instead, scan the directory once to obtain an existing_names set.
"""

from __future__ import annotations

import argparse
import time
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


DAILY_RE = re.compile(r"^(?P<y>\d{4})_(?P<m>\d{2})_(?P<d>\d{2})\.(?P<ext>tif|tiff)$", re.IGNORECASE)
YEARLY_RE = re.compile(r"^(?P<prefix>.+)_(?P<y>\d{4})\.(?P<ext>tif|tiff)$", re.IGNORECASE)


@dataclass(frozen=True)
class SourcePick:
    year: int
    path: Path


def _daterange(d0: date, d1: date) -> Iterable[date]:
    cur = d0
    one = timedelta(days=1)
    while cur <= d1:
        yield cur
        cur += one


def _year_bounds(y: int) -> Tuple[date, date]:
    start = date(y, 1, 1)
    end = date(y, 12, 31)
    return start, end


def _scan_folder(folder: Path) -> Tuple[Dict[int, SourcePick], Set[str]]:
    """
    Returns:
    - year_sources: {year: SourcePick}; one representative image per year.
    - existing_names: set of filenames already present in the directory (used to skip outputs).
    """
    existing_names: Set[str] = set()
    daily_candidates: Dict[int, Path] = {}
    yearly_candidates: Dict[int, Path] = {}

    patterns = ("*.tif", "*.TIF", "*.tiff", "*.TIFF")
    for pat in patterns:
        for p in folder.glob(pat):
            existing_names.add(p.name)

            m = DAILY_RE.match(p.name)
            if m:
                y = int(m.group("y"))
                # A year may have many days; we only need one representative file -- keep the first one encountered
                daily_candidates.setdefault(y, p)
                continue

            m = YEARLY_RE.match(p.name)
            if m:
                y = int(m.group("y"))
                # We assume one annual file per year by default; if there are multiple, keep the first
                yearly_candidates.setdefault(y, p)
                continue

    year_sources: Dict[int, SourcePick] = {}
    # Prefer the annual file (more in line with the "use yyyy only" semantics); otherwise use any per-day file as representative
    for y, p in yearly_candidates.items():
        year_sources[y] = SourcePick(year=y, path=p)
    for y, p in daily_candidates.items():
        year_sources.setdefault(y, SourcePick(year=y, path=p))

    return year_sources, existing_names


def _resolve_fill_years(years_needed: List[int], available_years: List[int]) -> Dict[int, int]:
    """
    For each target year, determine the source year (use the nearest past year if missing; fall back to the earliest available year if there is no past year).
    Returns {target_year: source_year}.
    """
    if not available_years:
        raise ValueError("No tif / tiff files with parseable years found in the directory.")

    available_years_sorted = sorted(set(available_years))
    earliest = available_years_sorted[0]
    mapping: Dict[int, int] = {}

    last_available: Optional[int] = None
    avail_set = set(available_years_sorted)

    for y in sorted(years_needed):
        if y in avail_set:
            mapping[y] = y
            last_available = y
        else:
            if last_available is not None and last_available < y:
                mapping[y] = last_available
            else:
                mapping[y] = earliest
    return mapping


def _copy2(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description="Expand static variables from annual to daily (with nearest-past backfilling for missing years)")
    ap.add_argument(
        "--input-dir",
        type=Path,
        default='/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled_5xdownsampled_merged_unifiedEntireCan_scaledInt/DEMs_downsampled_unified',
        help="Directory containing the static-variable tif / tiff files (input directory)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default='/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled_5xdownsampled_merged_unifiedEntireCan_scaledInt/DEMs_downsampled_unified_tsfilled',
        help="Output directory (per-day YYYY_MM_DD.tif files are written here; may point to an /s3 path)",
    )
    ap.add_argument(
        "--staging-dir",
        type=Path,
        default='/seal/statics_filling',
        help="Optional: local staging root. If set, a unique subdirectory is created beneath it; outputs are written there first, then synced to output-dir in one shot when finished.",
    )
    ap.add_argument("--start-year", type=int, default=2000, help="Start year (inclusive). Defaults to the smallest available year in the directory")
    ap.add_argument("--end-year", type=int, default=2025, help="End year (inclusive). Defaults to the largest available year in the directory")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing YYYY_MM_DD.tif output files")
    ap.add_argument("--dry-run", action="store_true", help="Print statistics only; do not actually write files")
    ap.add_argument("--workers", type=int, default=8, help="Number of concurrent threads (I/O copy parallelism)")
    ap.add_argument("--keep-staging", action="store_true", help="When using staging-dir, keep the staging-directory contents after the sync completes")
    ap.add_argument(
        "--rsync-jobs",
        type=int,
        default=100,
        help="Number of staging -> output rsync shards (> 1 enables parallel rsync). Many shards are allowed, but actually-running rsync processes are capped by --rsync-max-parallel, to avoid hangs from too many processes against slow backends like /s3.",
    )
    ap.add_argument(
        "--rsync-max-parallel",
        type=int,
        default=32,
        help="Maximum number of parallel rsync processes (default 32). Decoupled from --rsync-jobs: jobs only determines how many filename buckets to use, while concurrent rsyncs never exceed this value.",
    )
    args = ap.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    # Actual write directory: when staging-dir is specified, create a unique subdirectory beneath it to avoid mixed / leftover collisions
    staging_root: Optional[Path] = args.staging_dir
    if staging_root is not None:
        try:
            staging_root.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            if getattr(e, "errno", None) == 28:
                raise SystemExit(
                    f"Disk space exhausted (No space left on device). Cannot create staging root directory: {staging_root}\n"
                    f"   Suggest changing `--staging-dir` to a larger local disk, or setting `--staging-dir None` to write directly to the output directory."
                )
            raise
        write_dir = Path(tempfile.mkdtemp(prefix="statics_staging_", dir=staging_root.as_posix()))
    else:
        write_dir = output_dir

    try:
        write_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        if getattr(e, "errno", None) == 28:
            raise SystemExit(
                f"Disk space exhausted (No space left on device). Cannot create directory: {write_dir}\n"
                f"   Suggest changing `--staging-dir` to a larger local disk, or skipping staging and writing directly to `--output-dir`."
            )
        raise

    year_sources, _ = _scan_folder(input_dir)
    if not year_sources:
        raise SystemExit(f"No year-parseable tif / tiff found in the input directory: {input_dir}")

    # Use the output directory's existing filenames to drive skipping (avoiding per-day exists / stat)
    _, out_existing_names = _scan_folder(output_dir)

    available_years = sorted(year_sources.keys())
    start_year = args.start_year if args.start_year is not None else min(available_years)
    end_year = args.end_year if args.end_year is not None else max(available_years)
    if end_year < start_year:
        raise SystemExit("end-year is earlier than start-year")

    years_needed = list(range(start_year, end_year + 1))
    fill_year_map = _resolve_fill_years(years_needed, available_years)

    # Generate tasks (only for missing day files)
    tasks: List[Tuple[Path, Path]] = []
    total_days = 0
    skipped_existing = 0

    for y in years_needed:
        src_year = fill_year_map[y]
        src_path = year_sources[src_year].path
        y0, y1 = _year_bounds(y)
        for d in _daterange(y0, y1):
            total_days += 1
            out_name = f"{d:%Y_%m_%d}.tif"
            out_path = write_dir / out_name

            if not args.overwrite:
                # Use directory-scan results to skip; avoid per-day exists / stat
                if out_name in out_existing_names:
                    skipped_existing += 1
                    continue

            tasks.append((src_path, out_path))

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    if staging_root is not None:
        print(f"Staging root directory: {staging_root}")
        print(f"Current staging subdirectory: {write_dir} (write to staging first, then sync in one batch)")
    print(f"Available years: {available_years[0]} - {available_years[-1]} (total {len(available_years)} years)")
    print(f"Target year range: {start_year} - {end_year} (total {len(years_needed)} years)")
    print(f"Total target days: {total_days}")
    if not args.overwrite:
        print(f"Existing day files skipped: {skipped_existing}")
    print(f"Need to write: {len(tasks)}")
    if args.dry_run:
        print("dry-run mode: no files will be written")
        return

    if not tasks:
        print("Nothing to write; done.")
        return

    # Concurrent copy
    written = 0
    failed = 0

    def _worker(item: Tuple[Path, Path]) -> str:
        src, dst = item
        if dst.exists() and not args.overwrite:
            return dst.name
        _copy2(src, dst)
        return dst.name

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [ex.submit(_worker, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), start=1):
            try:
                _ = fut.result()
                written += 1
            except Exception:
                failed += 1
            if i % 500 == 0 or i == len(futures):
                print(f"\r[{i}/{len(futures)}] written: {written} failed: {failed}", end="", flush=True)

    print()
    if failed:
        raise SystemExit(f"Completed with failures: written={written}, failed={failed}")
    print(f"Write complete: written={written}")

    # staging -> output_dir sync
    if staging_root is not None:
        print("Syncing staging directory to output directory...")
        rsync = shutil.which("rsync")
        if rsync:
            jobs = max(1, int(args.rsync_jobs))
            src_root = f"{write_dir.as_posix().rstrip('/')}/"
            dst_root = f"{output_dir.as_posix().rstrip('/')}/"

            if jobs == 1:
                # -a: preserve attributes; --no-inc-recursive: more stable with many small files
                cmd = [
                    rsync,
                    "-a",
                    "--no-inc-recursive",
                    "--info=progress2",
                    src_root,
                    dst_root,
                ]
                p = subprocess.run(cmd)
                if p.returncode != 0:
                    raise SystemExit(f"rsync sync failed; exit code={p.returncode}")
            else:
                # Parallel rsync: bucket by filename and transfer concurrently across processes
                names = sorted([p.name for p in write_dir.glob("*.tif")])
                total = len(names)
                if total == 0:
                    print("Staging directory is empty; nothing to sync.")
                else:
                    buckets: List[List[str]] = [[] for _ in range(jobs)]
                    for i, n in enumerate(names):
                        buckets[i % jobs].append(n)

                    list_dir = Path(tempfile.mkdtemp(prefix="rsync_lists_"))
                    filelists: List[Tuple[Path, int]] = []
                    for bi, bn in enumerate(buckets):
                        if not bn:
                            continue
                        lp = list_dir / f"files_{bi:03d}.txt"
                        lp.write_text("\n".join(bn) + "\n")
                        filelists.append((lp, len(bn)))

                    t0 = time.time()
                    done = 0

                    def _rsync_one(filelist: Path) -> int:
                        cmd = [
                            rsync,
                            "-a",
                            "--no-inc-recursive",
                            "--files-from",
                            filelist.as_posix(),
                            src_root,
                            dst_root,
                        ]
                        return subprocess.run(cmd).returncode

                    # Without a concurrency cap, large jobs can spawn thousands of rsync processes at once, easily exhausting fds or freezing the mount point on s3fs / NFS, appearing as a long stretch of no output ("stuck").
                    cap = max(1, int(args.rsync_max_parallel))
                    pool_workers = min(jobs, len(filelists), cap)
                    if jobs > pool_workers:
                        print(
                            f"rsync shard count={jobs}, non-empty shards={len(filelists)}，"
                            f"concurrent rsync={pool_workers}（--rsync-max-parallel={cap}）"
                        )
                    with ThreadPoolExecutor(max_workers=pool_workers) as ex2:
                        fut_to_n = {ex2.submit(_rsync_one, lp): n for lp, n in filelists}
                        for fut in as_completed(fut_to_n):
                            rc = fut.result()
                            if rc != 0:
                                shutil.rmtree(list_dir, ignore_errors=True)
                                raise SystemExit(f"Parallel rsync subtask failed; exit code={rc}")
                            done += fut_to_n[fut]
                            elapsed = time.time() - t0
                            rate = done / elapsed if elapsed > 0 else 0.0
                            remaining = (total - done) / rate if rate > 0 else 0.0
                            eta = f"{int(remaining//3600):02d}:{int((remaining%3600)//60):02d}:{int(remaining%60):02d}"
                            print(f"\r[{done}/{total}] ETA: {eta}", end="", flush=True)
                        print()

                    shutil.rmtree(list_dir, ignore_errors=True)
        else:
            # Fallback: per-file move (degrades to copy + delete across filesystems and may be slower)
            files = sorted(write_dir.glob("*.tif"))
            total = len(files)
            t0 = time.time()
            done = 0
            for p in files:
                dst = output_dir / p.name
                if dst.exists() and not args.overwrite:
                    done += 1
                    continue
                shutil.move(p, dst)
                done += 1
                if done % 200 == 0 or done == total:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0.0
                    remaining = (total - done) / rate if rate > 0 else 0.0
                    eta = f"{int(remaining//3600):02d}:{int((remaining%3600)//60):02d}:{int(remaining%60):02d}"
                    print(f"\r[{done}/{total}] ETA: {eta}", end="", flush=True)
            if total > 0:
                print()

        if not args.keep_staging:
            shutil.rmtree(write_dir, ignore_errors=True)
        print("Sync complete")


if __name__ == "__main__":
    main()

