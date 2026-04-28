"""Step 1 of hard-positive evaluation — extract firemask=7 and firemask=8
positive candidate pools for val years, fire-season only.

Purpose
-------
The existing test/val set uses only firemask==9 positives, which are the very
fires FWI/CFFDRS is designed to predict. We need evaluation positives at
firemask=7 (low-confidence) and firemask=8 (nominal-confidence) to probe
FWI's behaviour on harder, smaller, and earlier fires.

Design choices
--------------
- Val years restricted to 2020-2022 (matches current val split era).
- Fire season only: May–October. Off-season firemask=7/8 detections are mostly
  noise (snow/cloud edge artefacts).
- Each candidate pixel is masked to research_area_shpfile_merge.shp (same
  rule as Layer1Filter).
- No spatial/temporal buffer is applied — these are positives, not negatives.
- We cap each firemask pool at TARGET_PER_FM = 50k samples (random draw);
  downstream probe only needs 5k per firemask.

Output
------
  sampling/probe_hardpos/pool_fm7_val.json
  sampling/probe_hardpos/pool_fm8_val.json

Each record matches the existing samples_variant_*.json schema:
  {year, day_of_year, date, pixel_row, pixel_col, firms_value,
   sample_type, firemask_label, split}

'firms_value' is kept as 4 (downstream code treats it as an opaque positive
marker); 'firemask_label' records the tier (7 or 8) for analysis.
"""
from __future__ import annotations
import json
import logging
import os
import random
import sys
import glob as glob_module
from datetime import date as dtdate
from pathlib import Path

import geopandas as gpd
import numpy as np
from osgeo import gdal

# Make sibling sampling/ modules importable
_SAMPLING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SAMPLING_DIR))

from layer1_filter import (  # noqa: E402
    MODIS_DIR,
    EPSG_WGS84,
    _read_firemask,
    _pixel_centroids,
    _get_research_union,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("extract_hardpos_pool")
gdal.UseExceptions()

# ---------- config ----------
OUT_DIR = Path(__file__).resolve().parent
VAL_YEAR_START = 2020
VAL_YEAR_END   = 2022
FIRE_SEASON_MONTHS = {5, 6, 7, 8, 9, 10}
TARGET_PER_FM  = 50_000
SEED           = 42


def iter_fire_season_dates(y_start: int, y_end: int) -> list[str]:
    files = sorted(glob_module.glob(os.path.join(MODIS_DIR, "*.tif")))
    out = []
    seen = set()
    for f in files:
        stem = os.path.basename(f).replace(".tif", "")
        parts = stem.split("_")
        if len(parts) != 3:
            continue
        try:
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        if not (y_start <= y <= y_end):
            continue
        if m not in FIRE_SEASON_MONTHS:
            continue
        ds = f"{y:04d}-{m:02d}-{d:02d}"
        if ds not in seen:
            seen.add(ds)
            out.append(ds)
    return sorted(out)


def extract_pool(fm_value: int, dates: list[str], cap: int) -> list[dict]:
    ra_union = _get_research_union()
    samples: list[dict] = []
    n_dates = len(dates)
    n_kept_pixels_total = 0
    for i, ds in enumerate(dates, start=1):
        res = _read_firemask(ds)
        if res is None:
            continue
        arr, gt = res
        mask = arr == fm_value
        hits = int(mask.sum())
        if hits == 0:
            continue
        rows, cols = np.where(mask)
        lons, lats = _pixel_centroids(arr, gt, mask)

        # Research-area filter (uses sindex internally)
        pts = gpd.GeoSeries(gpd.points_from_xy(lons, lats), crs=EPSG_WGS84)
        keep = pts.within(ra_union).values
        if keep.sum() == 0:
            continue
        rows, cols = rows[keep], cols[keep]
        n_kept_pixels_total += int(keep.sum())

        y, m, d = ds.split("-")
        year = int(y)
        doy = dtdate(year, int(m), int(d)).timetuple().tm_yday
        for r, c in zip(rows, cols):
            samples.append({
                "year": year,
                "day_of_year": doy,
                "date": ds,
                "pixel_row": int(r),
                "pixel_col": int(c),
                "firms_value": 4,
                "sample_type": "positive",
                "firemask_label": int(fm_value),
                "split": "val",
            })
        if i % 50 == 0:
            log.info("  [%d/%d] FM=%d  kept_total=%d",
                     i, n_dates, fm_value, n_kept_pixels_total)

    if len(samples) > cap:
        rng = random.Random(SEED + fm_value)
        rng.shuffle(samples)
        samples = samples[:cap]
    return samples


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dates = iter_fire_season_dates(VAL_YEAR_START, VAL_YEAR_END)
    log.info("Fire-season dates %d-%d: %d",
             VAL_YEAR_START, VAL_YEAR_END, len(dates))

    for fm in (7, 8):
        log.info("=== Extracting firemask=%d positives ===", fm)
        pool = extract_pool(fm, dates, TARGET_PER_FM)
        out = OUT_DIR / f"pool_fm{fm}_val.json"
        with open(out, "w") as f:
            json.dump(pool, f)
        log.info("  Saved %d samples -> %s", len(pool), out)


if __name__ == "__main__":
    main()
