# -*- coding: utf-8 -*-
"""
Check whether TIFF files can be read by GDAL; delete them if not.

Usage:
    python check_and_remove_invalid_files.py --input-dir /path/to/root
    python check_and_remove_invalid_files.py --input-dir /path/to/root --dry-run  # check only, do not delete
    python check_and_remove_invalid_files.py --input-dir /path/to/root -j 16  # multithreaded acceleration (default 8)
    python check_and_remove_invalid_files.py --input-dir /path/to/root --full-check  # also read projection/geotransform (slower)
    python check_and_remove_invalid_files.py --input-dir /path/to/root --exclude-subdir tmp --exclude-subdir cache
"""

import os
import sys
import argparse
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from osgeo import gdal

# ============ Logging configuration ============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("check_files")

# Initialize GDAL
gdal.UseExceptions()


def format_time(seconds):
    """Format time display"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def check_file_readable(file_path, full_check: bool = False):
    """
    Check whether GDAL can open the raster and read at least one pixel.

    Default (full_check=False): does not read GeoTransform/Projection (not needed in most "can it open?" scenarios);
    uses ReadRaster(1x1) instead of ReadAsArray to avoid NumPy allocation.

    full_check=True: additionally reads the affine transform and projection (closer to the old version, slightly slower).

    Returns:
        (is_valid, error_message)
    """
    path = str(file_path)
    ds = None
    try:
        ds = gdal.Open(path, gdal.GA_ReadOnly)
        if ds is None:
            error_msg = gdal.GetLastErrorMsg()
            if not error_msg:
                error_msg = "Cannot open file (GDAL returned None)"
            return False, error_msg

        width = ds.RasterXSize
        height = ds.RasterYSize
        bands = ds.RasterCount
        if width <= 0 or height <= 0 or bands <= 0:
            return False, f"Invalid raster size: {width}x{height}, band count: {bands}"

        band = ds.GetRasterBand(1)
        if band is None:
            return False, "Cannot read the first band"

        if full_check:
            ds.GetGeoTransform()
            ds.GetProjection()

        buf = band.ReadRaster(0, 0, 1, 1, buf_xsize=1, buf_ysize=1)
        if buf is None:
            return False, "Cannot read pixel data (ReadRaster returned None)"

        return True, None

    except RuntimeError as e:
        error_msg = str(e)
        if "not recognized as a supported file format" in error_msg:
            return False, "Unsupported file format"
        return False, f"RuntimeError: {error_msg}"
    except Exception as e:
        return False, f"Error reading file info: {str(e)}"
    finally:
        ds = None


def remove_file(file_path, dry_run=False):
    """Delete file"""
    if dry_run:
        logger.info(f"[DRY-RUN] would delete: {file_path}")
        return True
    
    try:
        os.remove(str(file_path))
        return True
    except Exception as e:
        logger.error(f"Failed to delete file {file_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Check whether TIFF files can be read by GDAL; delete those that cannot')
    
    parser.add_argument('--input-dir', 
                       default=r'/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/MCD09GA_b1237_withQA',
                       help='Input root directory containing multiple subfolders (TIFFs are searched recursively in all subfolders)')
    parser.add_argument(
        '--exclude-subdir',
        action='append',
        default=['MCD13Q1_QAapplied_spatialFilled'],
        help='Subfolder names or relative paths to skip (can be specified multiple times). Example: --exclude-subdir tmp --exclude-subdir MCD09GA_QAapplied',
    )
    parser.add_argument('--dry-run', action='store_true',
                       help='Check only, do not actually delete files')
    parser.add_argument(
        '--full-check',
        action='store_true',
        help='Also read GeoTransform and Projection (slower; the default readability check already includes a 1x1 pixel read)',
    )
    parser.add_argument(
        '-j', '--jobs',
        type=int,
        default=8,
        help='Number of parallel check threads (default 8; for network filesystems try 16-32; set to 1 for sequential execution)',
    )
    parser.add_argument('--verbose', action='store_true',
                       help='Show verbose info')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    if not input_path.is_dir():
        logger.error(f"Input path is not a directory: {args.input_dir}")
        sys.exit(1)

    # Normalize exclusion rules: support both "name" and "relative/path"
    excludes_raw = [str(x).strip().strip("/").strip("\\") for x in (args.exclude_subdir or []) if str(x).strip()]
    exclude_names = {Path(x).name for x in excludes_raw if x}
    exclude_relpaths = {Path(x).as_posix() for x in excludes_raw if x}
    if excludes_raw:
        logger.info(f"Subfolder exclusion rules: {', '.join(excludes_raw)}")
    
    # Find all TIFF files (recursive; with optional subdirectory exclusion)
    patterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
    all_files = []
    # Iterate only the subdirectories under root (small numbers of tifs in root itself are also allowed)
    search_roots = [input_path] + [p for p in sorted(input_path.iterdir()) if p.is_dir()]
    for root in search_roots:
        # Exclusion is not applied to root itself; only to its subdirectories
        if root != input_path:
            rel = root.relative_to(input_path).as_posix()
            if root.name in exclude_names or rel in exclude_relpaths:
                logger.info(f"Skip subfolder: {rel}")
                continue
        for pattern in patterns:
            all_files.extend(root.rglob(pattern))
    
    if not all_files:
        logger.warning(f"In directory {args.input_dir}: no TIFF files found")
        sys.exit(0)
    
    total_files = len(all_files)
    logger.info(f"Found {total_files} TIFF files")
    
    if args.dry_run:
        logger.info("=== DRY-RUN mode: check only; no files will be deleted ===")
    
    # Statistics
    valid_files = 0
    invalid_files = 0
    deleted_files = 0
    failed_deletions = 0
    
    start_time = time.time()
    sorted_files = sorted(all_files)
    stat_lock = threading.Lock()

    def handle_one(file_path, is_valid, error_msg):
        nonlocal valid_files, invalid_files, deleted_files, failed_deletions
        if is_valid:
            with stat_lock:
                valid_files += 1
            if args.verbose:
                logger.debug(f"OK {file_path.name}")
            return
        with stat_lock:
            invalid_files += 1
        logger.warning(f"INVALID {file_path.name}: {error_msg}")
        if remove_file(file_path, dry_run=args.dry_run):
            with stat_lock:
                deleted_files += 1
        else:
            with stat_lock:
                failed_deletions += 1

    if args.jobs <= 1:
        for idx, file_path in enumerate(sorted_files, 1):
            if idx % 100 == 0 or idx == total_files:
                elapsed = time.time() - start_time
                progress = idx / total_files * 100
                print(
                    f"\rProgress: [{idx}/{total_files}] ({progress:.1f}%) | "
                    f"Valid: {valid_files} | Invalid: {invalid_files} | "
                    f"Deleted: {deleted_files} | Elapsed: {format_time(elapsed)}",
                    end="",
                    flush=True,
                )
            is_valid, error_msg = check_file_readable(file_path, full_check=args.full_check)
            handle_one(file_path, is_valid, error_msg)
    else:
        completed = 0
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            future_to_path = {
                ex.submit(check_file_readable, fp, args.full_check): fp for fp in sorted_files
            }
            for fut in as_completed(future_to_path):
                fp = future_to_path[fut]
                try:
                    is_valid, error_msg = fut.result()
                except Exception as e:
                    is_valid, error_msg = False, f"Check exception: {e}"
                handle_one(fp, is_valid, error_msg)
                completed += 1
                if completed % 100 == 0 or completed == total_files:
                    elapsed = time.time() - start_time
                    with stat_lock:
                        vf, inv, delc = valid_files, invalid_files, deleted_files
                    progress = completed / total_files * 100
                    print(
                        f"\rProgress: [{completed}/{total_files}] ({progress:.1f}%) | "
                        f"Valid: {vf} | Invalid: {inv} | "
                        f"Deleted: {delc} | Elapsed: {format_time(elapsed)}",
                        end="",
                        flush=True,
                    )

    print()  # Newline
    
    elapsed_time = time.time() - start_time
    
    # Print statistics
    logger.info("=" * 60)
    logger.info("Check complete!")
    logger.info(f"Total elapsed: {format_time(elapsed_time)}")
    logger.info(f"Total files: {total_files}")
    logger.info(f"Valid files: {valid_files} ({valid_files/total_files*100:.1f}%)")
    logger.info(f"Invalid files: {invalid_files} ({invalid_files/total_files*100:.1f}%)")
    
    if args.dry_run:
        logger.info(f"[DRY-RUN] number of files to delete: {invalid_files}")
    else:
        logger.info(f"Successfully deleted: {deleted_files}")
        if failed_deletions > 0:
            logger.warning(f"Delete failed: {failed_deletions}")
    
    logger.info("=" * 60)
    
    if invalid_files > 0 and not args.dry_run:
        logger.info(f"Cleaned up {deleted_files} invalid files")
    elif invalid_files > 0 and args.dry_run:
        logger.info(f"Found {invalid_files} invalid files (use --dry-run for details)")


if __name__ == "__main__":
    main()

