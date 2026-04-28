"""Dataset scope audit: confirm exact temporal span of MOD14A1 data and count
high-confidence fire detections.

Resolves Open Question OQ4: exact temporal span and positive sample count.

Tile naming convention: MCD14A1_YYYY_MM_DD_tileNN.tif (4 tiles per day, tile00-tile03)

Output: audit/outputs/scope_audit_report.json with keys:
  modis_start_year:         int  -- first year with data files on disk
  modis_end_year:           int  -- last year with data files on disk
  modis_start_date:         str  -- ISO date of first tile found (YYYY-MM-DD)
  modis_end_date:           str  -- ISO date of last tile found (YYYY-MM-DD)
  total_tile_count:         int  -- total number of daily tiles found
  expected_tile_count:      int  -- expected tiles based on date span * 4 tiles/day
  missing_modis_days:       list -- ISO date strings of missing days (no tiles at all)
  high_conf_positive_count: int  -- total pixel-days with firemask==9
  total_fire_days:          int  -- number of days with at least one firemask==9 pixel
  nominal_conf_count:       int  -- total pixel-days with firemask==8
  low_conf_count:           int  -- total pixel-days with firemask==7
  cloud_pixel_days:         int  -- total pixel-days with firemask==4
  nonfire_land_days:        int  -- total pixel-days with firemask==5
"""
import json
import os
import glob
import re
import numpy as np
from datetime import date, timedelta
from audit.modis_utils import (read_modis_tile, get_positive_mask,
                                 FIREMASK_NOM_CONF, FIREMASK_LOW_CONF,
                                 FIREMASK_CLOUD, FIREMASK_NONFIRE_LAND)

MODIS_DATA_DIR = ("/s3/zhengsen/wildfire_dataset/self_built_materials/"
                  "bc_raw_data_new_fast/drivers/processed/MCD14A1_combined")

# Pattern: MCD14A1_YYYY_MM_DD_tileNN.tif
_TILE_RE = re.compile(r"MCD14A1_(\d{4})_(\d{2})_(\d{2})_tile\d+\.tif$")


def _tile_to_date(path: str):
    """Extract date from tile filename like MCD14A1_YYYY_MM_DD_tileNN.tif."""
    basename = os.path.basename(path)
    m = _TILE_RE.match(basename)
    if m is None:
        raise ValueError(f"Unexpected MODIS tile filename format: {basename}")
    return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))


def run_scope_audit(output_path: str = None,
                    modis_dir: str = MODIS_DATA_DIR,
                    dry_run: bool = False) -> dict:
    """Scan all MODIS tiles and produce scope audit report.

    Args:
        output_path: path to write JSON report. If None, writes to default location.
        modis_dir:   directory containing daily MODIS GeoTIFF tiles.
        dry_run:     if True, read only tiles from the first and last day (fast schema
                     test). Returns a report with correct keys but counts from 2 days only.

    Returns:
        report dict (also written to output_path as JSON).
    """
    if output_path is None:
        default_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "audit", "outputs"
        )
        os.makedirs(default_dir, exist_ok=True)
        output_path = os.path.join(default_dir, "scope_audit_report.json")

    # Discover all tiles (sorted so first/last by name = sorted by date)
    tile_pattern = os.path.join(modis_dir, "MCD14A1_*.tif")
    all_tiles = sorted(glob.glob(tile_pattern))

    if len(all_tiles) == 0:
        raise FileNotFoundError(
            f"No MODIS tiles found in {modis_dir}. "
            "Check data path and server mount."
        )

    # Build a dict: date -> list of tile paths
    from collections import defaultdict
    date_to_tiles = defaultdict(list)
    for tp in all_tiles:
        d = _tile_to_date(tp)
        date_to_tiles[d].append(tp)

    sorted_dates = sorted(date_to_tiles.keys())
    first_date = sorted_dates[0]
    last_date  = sorted_dates[-1]

    # Expected days: every calendar day from first to last
    expected_days = (last_date - first_date).days + 1
    # Expected tiles = days * 4 tiles/day
    expected_tile_count = expected_days * 4

    # Find missing days (days with no tiles at all)
    missing_days = []
    d = first_date
    while d <= last_date:
        if d not in date_to_tiles:
            missing_days.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)

    # Count fire detections
    high_conf_positive_count = 0
    total_fire_days           = 0
    nominal_conf_count        = 0
    low_conf_count            = 0
    cloud_pixel_days          = 0
    nonfire_land_days         = 0

    if dry_run:
        days_to_scan = [sorted_dates[0], sorted_dates[-1]]
    else:
        days_to_scan = sorted_dates

    tiles_scanned = 0
    for scan_date in days_to_scan:
        day_high_conf = 0
        for tile_path in date_to_tiles[scan_date]:
            try:
                firemask, _, _ = read_modis_tile(tile_path)
            except Exception as e:
                print(f"Warning: failed to read {tile_path}: {e}")
                continue

            pos_mask = get_positive_mask(firemask)
            n_high_conf = int(pos_mask.sum())
            day_high_conf += n_high_conf

            high_conf_positive_count += n_high_conf
            nominal_conf_count += int((firemask == FIREMASK_NOM_CONF).sum())
            low_conf_count     += int((firemask == FIREMASK_LOW_CONF).sum())
            cloud_pixel_days   += int((firemask == FIREMASK_CLOUD).sum())
            nonfire_land_days  += int((firemask == FIREMASK_NONFIRE_LAND).sum())
            tiles_scanned += 1

        if day_high_conf > 0:
            total_fire_days += 1

    report = {
        "modis_start_year":         first_date.year,
        "modis_end_year":           last_date.year,
        "modis_start_date":         first_date.strftime("%Y-%m-%d"),
        "modis_end_date":           last_date.strftime("%Y-%m-%d"),
        "total_tile_count":         len(all_tiles),
        "expected_tile_count":      expected_tile_count,
        "missing_modis_days":       missing_days,
        "high_conf_positive_count": high_conf_positive_count,
        "total_fire_days":          total_fire_days,
        "nominal_conf_count":       nominal_conf_count,
        "low_conf_count":           low_conf_count,
        "cloud_pixel_days":         cloud_pixel_days,
        "nonfire_land_days":        nonfire_land_days,
        "dry_run":                  dry_run,
        "tiles_scanned":            tiles_scanned,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Scope audit complete. Report written to: {output_path}")
    print(f"  MODIS span: {report['modis_start_date']} -> {report['modis_end_date']}")
    print(f"  Total tiles: {report['total_tile_count']} / {report['expected_tile_count']} expected")
    print(f"  Missing days: {len(missing_days)}")
    if not dry_run:
        print(f"  High-conf positives: {high_conf_positive_count:,}")
        print(f"  Total fire days: {total_fire_days:,}")

    return report


if __name__ == "__main__":
    import sys
    dry = "--dry-run" in sys.argv
    report = run_scope_audit(dry_run=dry)
    print(json.dumps(report, indent=2))
