"""
Extract OSM features from yearly PBF snapshots using osmium + ogr2ogr (C++ tools).
Much faster than pyrosm. Skips already-completed files so it is safe to re-run.

Pipeline per (year, feature):
  1. osmium tags-filter  →  small filtered .osm.pbf
  2. osmium export       →  .geojsonseq
  3. ogr2ogr             →  .shp  (inside yearly_shapefiles/<feature>/)

Requirements:
    sudo apt install osmium-tool gdal-bin

Usage:
    python extract_osm_features.py
"""

import os
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, wait
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR      = "/mnt/raid/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/osm"
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "yearly_snapshots")
SHP_DIR       = os.path.join(BASE_DIR, "yearly_shapefiles")
TMP_DIR       = os.path.join(BASE_DIR, "tmp")

YEARS = list(range(2010, 2026))

# osmium tags-filter expressions  (w=way, r=relation, n=node)
FEATURES = {
    "rivers":     "w/waterway=river,stream,canal r/waterway=river,stream,canal",
    "lakes":      "w/natural=water r/natural=water",
    "roads":      "w/highway r/highway",
    "buildings":  "w/building r/building",
    "powerlines": "w/power=line,minor_line r/power=line,minor_line",
}

# 5 workers = one feature per year runs in parallel.
# More than this causes RAM exhaustion from concurrent osmium PBF loads.
MAX_WORKERS = 5

os.makedirs(TMP_DIR, exist_ok=True)
for feat in FEATURES:
    os.makedirs(os.path.join(SHP_DIR, feat), exist_ok=True)


# ── Core processing function ──────────────────────────────────────────────────
def process(year: int, feature: str, osm_filter: str) -> str:
    out_shp = os.path.join(SHP_DIR, feature, f"canada_{feature}_{year}.shp")
    if os.path.exists(out_shp):
        return f"[SKIP] {feature}/{year}"

    pbf = os.path.join(SNAPSHOTS_DIR, f"canada_{year}.osm.pbf")
    if not os.path.exists(pbf):
        return f"[MISSING] {feature}/{year} — {pbf} not found"

    # Unique temp file names to avoid collisions between parallel workers
    uid          = f"{feature}_{year}"
    filtered_pbf = os.path.join(TMP_DIR, f"{uid}_filtered.osm.pbf")
    geojson_seq  = os.path.join(TMP_DIR, f"{uid}.geojsonseq")

    try:
        # Step 1: filter to just the tags we want
        subprocess.run(
            ["osmium", "tags-filter", pbf]
            + osm_filter.split()
            + ["--overwrite", "-o", filtered_pbf],
            check=True, capture_output=True,
        )

        # Step 2: export to GeoJSONSeq (streaming, no RAM spike)
        subprocess.run(
            ["osmium", "export", filtered_pbf,
             "--geometry-types=point,linestring,polygon",
             "--overwrite", "-f", "geojsonseq", "-o", geojson_seq],
            check=True, capture_output=True,
        )

        # Step 3: convert to Shapefile with ogr2ogr
        subprocess.run(
            ["ogr2ogr",
             "-f", "ESRI Shapefile",
             "-nlt", "PROMOTE_TO_MULTI",
             "-lco", "ENCODING=UTF-8",
             out_shp, geojson_seq],
            check=True, capture_output=True,
        )

        size_mb = os.path.getsize(out_shp) / 1e6
        return f"OK {feature}/{year}  ({size_mb:.1f} MB)"

    except subprocess.CalledProcessError as e:
        return f"ERROR {feature}/{year}: {e.stderr.decode()[:200]}"

    finally:
        for f in [filtered_pbf, geojson_seq]:
            if os.path.exists(f):
                os.remove(f)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Workers     : {MAX_WORKERS} (5 features per year in parallel)")
    print(f"Output      : {SHP_DIR}\n")

    t0 = time.time()
    total = len(YEARS) * len(FEATURES)
    done  = 0

    # Process one year at a time: submit all 5 features, wait for all to finish,
    # then move to next year. Keeps peak memory to ~1 year's PBF at a time.
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for year in YEARS:
            futures = {
                pool.submit(process, year, feat, filt): feat
                for feat, filt in FEATURES.items()
            }
            wait(futures)
            for fut, feat in futures.items():
                done += 1
                try:
                    print(f"  [{done}/{total}] {fut.result()}")
                except Exception as e:
                    print(f"  [{done}/{total}] ERROR {feat}/{year}: {e}")

    elapsed = (time.time() - t0) / 60
    print(f"\nDone in {elapsed:.1f} min.")
    print(f"Shapefiles: {SHP_DIR}")
