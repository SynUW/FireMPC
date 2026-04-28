import argparse
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import sys
import time

DATE_FMT_IN = "%Y-%m-%d"       # Format of the input start/end dates
DATE_FMT_OUT = "%Y_%m_%d"      # Format of the date in the output filename

def format_time(seconds: float) -> str:
    """Format time display"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"

def print_progress(current: int, total: int, start_time: float, prefix: str = ""):
    """Print progress bar"""
    current_time = time.time()
    progress_percent = current / total * 100
    
    # Throttle update frequency
    should_update = (
        current_time - start_time >= 2.0 or  # At least 2 seconds interval
        progress_percent - int(progress_percent) == 0 or  # Every 1% progress
        current == total  # When complete
    )
    
    if not should_update and current < total:
        return
    
    elapsed = current_time - start_time
    if current > 0:
        avg_time_per_item = elapsed / current
        remaining_items = total - current
        eta = remaining_items * avg_time_per_item
        eta_str = format_time(eta)
    else:
        eta_str = "?:??:??"
    
    elapsed_str = format_time(elapsed)
    total_estimated = elapsed / current * total if current > 0 else 0
    total_str = format_time(total_estimated)
    
    print(f"\r{prefix}[{current}/{total}], {elapsed_str}/{total_str}, ETA: {eta_str}", end='', flush=True)
    
    if current == total:
        print()  # Newline after completion

def parse_args():
    p = argparse.ArgumentParser(
        description="Copy a .tif template per day into yyyy_mm_dd.tif files, dates inclusive of both ends."
    )
    p.add_argument("--src-dir", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x/distance_maps_materials', help="Source directory containing the template .tif")
    p.add_argument("--template", default=None, help="Template filename (optional). If omitted, the source directory must contain exactly one .tif.")
    p.add_argument("--dst-dir", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x_norm/distance_maps', help="Output directory (auto-created if missing)")
    p.add_argument("--start", default='2000-01-01', help="Start date (inclusive), format YYYY-MM-DD")
    p.add_argument("--end", default='2024-12-31', help="End date (inclusive), format YYYY-MM-DD")
    p.add_argument("--overwrite", action="store_true", help="Overwrite if the target exists (default: skip)")
    return p.parse_args()

def find_template(src_dir: Path, template_name: str | None) -> Path:
    if template_name:
        tpl = src_dir / template_name
        if not tpl.exists():
            sys.exit(f"Specified template does not exist: {tpl}")
        if tpl.suffix.lower() != ".tif" and tpl.suffix.lower() != ".tiff":
            sys.exit(f"Specified template is not .tif/.tiff: {tpl.name}")
        return tpl

    # If no filename is given, find the unique .tif/.tiff in the directory
    tifs = [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in (".tif", ".tiff")]
    if len(tifs) == 0:
        sys.exit("No .tif/.tiff files found in the source directory. Check --src-dir or use --template to specify a filename.")
    if len(tifs) > 1:
        names = ", ".join(p.name for p in tifs[:5])
        more = "" if len(tifs) <= 5 else f" etc. {len(tifs)} items"
        sys.exit(f"Source directory contains multiple .tif files: {names}{more}.\nPlease use --template to specify the template filename.")
    return tifs[0]

def daterange_inclusive(start_date: datetime, end_date: datetime):
    if end_date < start_date:
        sys.exit("End date is earlier than start date; please check --start / --end.")
    delta = (end_date - start_date).days
    for i in range(delta + 1):
        yield start_date + timedelta(days=i)

def main():
    args = parse_args()
    src_dir = Path(args.src_dir).expanduser().resolve()
    dst_dir = Path(args.dst_dir).expanduser().resolve()

    if not src_dir.exists():
        sys.exit(f"Source directory does not exist: {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    try:
        start_dt = datetime.strptime(args.start, DATE_FMT_IN)
        end_dt = datetime.strptime(args.end, DATE_FMT_IN)
    except ValueError as e:
        sys.exit(f"Date parsing failed: {e}\nPlease use the YYYY-MM-DD format.")

    template_path = find_template(src_dir, args.template)
    
    # Compute total number of dates
    total_dates = (end_dt - start_dt).days + 1
    print(f"Template file: {template_path.name}")
    print(f"Date range: {start_dt.strftime(DATE_FMT_IN)} to {end_dt.strftime(DATE_FMT_IN)}, total {total_dates} dates")
    print(f"Output directory: {dst_dir}")
    print()

    created, skipped, overwritten = 0, 0, 0
    start_time = time.time()
    
    for i, d in enumerate(daterange_inclusive(start_dt, end_dt), 1):
        print_progress(i, total_dates, start_time, prefix="Copy ")
        
        out_name = f"{d.strftime(DATE_FMT_OUT)}.tif"
        out_path = dst_dir / out_name

        if out_path.exists():
            if args.overwrite:
                shutil.copy2(template_path, out_path)
                overwritten += 1
            else:
                skipped += 1
        else:
            shutil.copy2(template_path, out_path)
            created += 1

    elapsed_time = time.time() - start_time
    print(f"\nProcessing complete! Total elapsed: {format_time(elapsed_time)}")
    print(f"Total: {total_dates}, created: {created}, skipped: {skipped}, overwritten: {overwritten}")
    print(f"Output directory: {dst_dir}")

if __name__ == "__main__":
    main()
