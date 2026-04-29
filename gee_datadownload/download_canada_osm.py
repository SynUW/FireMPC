"""
Download 16 years (2010-2025) of Canada OSM historical data.

Pipeline:
  1. Download OSM full-history planet (~140 GB, one-time)
  2. Extract Canada bbox from history → canada-history.osh.pbf (~15 GB)
  3. For each year 2000-2025: slice Dec-31 snapshot with osmium time-filter
  4. Extract features (rivers/lakes/roads/buildings/powerlines) in parallel
  5. Save as GeoPackage per feature per year, then optionally convert to SHP

NOTE: OSM was founded in 2004; data before 2007 is very sparse.

Requirements:
    osmium-tool:  sudo apt install osmium-tool
    sudo apt install gdal-bin
    pip install tqdm requests
    
nohup python download_canada_osm.py > osm_download.log 2>&1 &tail -f osm_download.log
"""

import os
import subprocess
import time
import requests
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR      = "/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/osm"
HISTORY_DIR     = os.path.join(OUTPUT_DIR, "history")
SNAPSHOTS_DIR   = os.path.join(OUTPUT_DIR, "yearly_snapshots")
FEATURES_DIR    = os.path.join(OUTPUT_DIR, "yearly_features")  # kept for backwards compatibility
RASTER_DIR      = os.path.join(OUTPUT_DIR, "yearly_1km_tifs")

HISTORY_URL     = "https://planet.openstreetmap.org/pbf/full-history/history-latest.osm.pbf"
HISTORY_PBF     = os.path.join(HISTORY_DIR, "history-latest.osm.pbf")
CANADA_HIST_PBF = os.path.join(HISTORY_DIR, "canada-history.osh.pbf")

# Canada bounding box
CANADA_BBOX = "-141.0,41.6,-52.6,83.1"   # W,S,E,N  (osmium format)

YEARS = list(range(2010, 2026))   # 2010 – 2025 inclusive

for d in [HISTORY_DIR, SNAPSHOTS_DIR, FEATURES_DIR, RASTER_DIR]:
    os.makedirs(d, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def run(cmd: list[str], desc: str = ""):
    """Run a shell command, streaming output."""
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True, text=True)
    return result


def download_with_progress(url: str, dest: str):
    if os.path.exists(dest):
        size_gb = os.path.getsize(dest) / 1e9
        print(f"[SKIP] {os.path.basename(dest)} already exists ({size_gb:.1f} GB)")
        return
    print(f"Downloading {os.path.basename(dest)} …  (this may take several hours)")
    with requests.get(url, stream=True, timeout=7200) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=os.path.basename(dest)
        ) as bar:
            for chunk in r.iter_content(chunk_size=4 << 20):   # 4 MB chunks
                f.write(chunk)
                bar.update(len(chunk))
    print(f"  Saved → {dest}")


# ── Step 1: Download full-history planet ─────────────────────────────────────
def step1_download_history():
    print("\n" + "="*60)
    print("STEP 1: Download OSM full-history planet (~140 GB)")
    print("="*60)
    download_with_progress(HISTORY_URL, HISTORY_PBF)


# ── Step 2: Extract Canada from history ──────────────────────────────────────
def step2_extract_canada():
    print("\n" + "="*60)
    print("STEP 2: Extract Canada from history planet")
    print("="*60)
    if os.path.exists(CANADA_HIST_PBF):
        size_gb = os.path.getsize(CANADA_HIST_PBF) / 1e9
        print(f"[SKIP] {os.path.basename(CANADA_HIST_PBF)} already exists ({size_gb:.1f} GB)")
        return
    run([
        "osmium", "extract",
        "--bbox", CANADA_BBOX,
        "--with-history",
        "--strategy", "complete_ways",
        HISTORY_PBF,
        "-o", CANADA_HIST_PBF,
        "--overwrite",
    ])
    size_gb = os.path.getsize(CANADA_HIST_PBF) / 1e9
    print(f"  Saved → {CANADA_HIST_PBF}  ({size_gb:.1f} GB)")


# ── Step 3: Create yearly Dec-31 snapshots ───────────────────────────────────
def step3_yearly_snapshots():
    print("\n" + "="*60)
    print("STEP 3: Create yearly snapshots (Dec-31 of each year)")
    print("="*60)
    for year in YEARS:
        out = os.path.join(SNAPSHOTS_DIR, f"canada_{year}.osm.pbf")
        if os.path.exists(out):
            print(f"  [SKIP] {year}")
            continue
        timestamp = f"{year}-12-31T23:59:59Z"
        run([
            "osmium", "time-filter",
            CANADA_HIST_PBF,
            timestamp,
            "-o", out,
            "--overwrite",
        ])
        size_mb = os.path.getsize(out) / 1e6
        print(f"  {year} → {out}  ({size_mb:.0f} MB)")


# ── Step 4: Filter + rasterize to 1km binary GeoTIFF ─────────────────────────
# We avoid pyrosm/geopandas (huge RAM blow-up). This pipeline stays disk/stream based:
# yearly PBF -> osmium tags-filter -> (optional) reproject -> gdal_rasterize to 1km tif.

# Internal processing CRS (meters) so 1000m == 1km.
INTERNAL_EPSG_M = "EPSG:3978"  # Canada Atlas Lambert (meters)
# Final output CRS requested by user.
OUTPUT_EPSG = "EPSG:4326"      # WGS84 (degrees)

# Keep bbox here for making a consistent national grid (in EPSG:4326).
CANADA_BBOX_W, CANADA_BBOX_S, CANADA_BBOX_E, CANADA_BBOX_N = -141.0, 41.6, -52.6, 83.1

FEATURE_OSMIUM_FILTERS: dict[str, list[str]] = {
    # ways
    "roads": ["w/highway"],
    "buildings": ["w/building"],
    "lakes": ["w/natural=water"],
    # rivers: include stream/canal too
    "rivers": ["w/waterway=river", "w/waterway=stream", "w/waterway=canal"],
    "powerlines": ["w/power=line", "w/power=minor_line"],
}

FEATURE_LAYER_PREFERENCE: dict[str, list[str]] = {
    # OGR OSM driver typical layer names
    "roads": ["lines", "multilinestrings"],
    "rivers": ["lines", "multilinestrings"],
    "powerlines": ["lines", "multilinestrings"],
    "buildings": ["multipolygons"],
    "lakes": ["multipolygons"],
}


def _run_capture(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.stdout


def _bbox_to_target_srs_te(n_edge_samples: int = 200) -> tuple[float, float, float, float]:
    """
    Transform CANADA_BBOX from EPSG:4326 to INTERNAL_EPSG_M using gdaltransform.
    Returns (minx, miny, maxx, maxy) in INTERNAL_EPSG_M coordinates.

    IMPORTANT: Only transforming the 4 corners is WRONG for conic projections like
    EPSG:3978 (Canada Atlas Lambert).  In a Lambert Conformal Conic, geographic parallels
    (constant-latitude lines) project as curved arcs that bow toward the pole.  The true
    minimum Y of the southern boundary occurs near the central meridian (~-96 deg lon),
    NOT at the corners.  Using only corners yields a too-small Y extent; when the result is
    then warped back to EPSG:4326 the missing southern strip appears as an arc-shaped NoData gap.

    Fix: densify all 4 edges before transforming so the true projected envelope is captured.
    """
    import numpy as np

    lons = np.linspace(CANADA_BBOX_W, CANADA_BBOX_E, n_edge_samples)
    lats = np.linspace(CANADA_BBOX_S, CANADA_BBOX_N, n_edge_samples)

    pts = []
    for lon in lons:          # bottom edge (lat = S)
        pts.append((lon, CANADA_BBOX_S))
    for lon in lons:          # top edge    (lat = N)
        pts.append((lon, CANADA_BBOX_N))
    for lat in lats:          # left edge   (lon = W)
        pts.append((CANADA_BBOX_W, lat))
    for lat in lats:          # right edge  (lon = E)
        pts.append((CANADA_BBOX_E, lat))

    pts_str = "\n".join(f"{x} {y}" for x, y in pts) + "\n"
    out = subprocess.run(
        ["gdaltransform", "-s_srs", "EPSG:4326", "-t_srs", INTERNAL_EPSG_M],
        input=pts_str,
        text=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip().splitlines()
    xs, ys = [], []
    for line in out:
        parts = line.split()
        if len(parts) >= 2:
            xs.append(float(parts[0]))
            ys.append(float(parts[1]))
    return (min(xs), min(ys), max(xs), max(ys))


def rasterize_year_feature_binary(year: int, feature: str) -> str:
    """
    Create a 1km binary GeoTIFF (0/1) for a feature in a given year.
    """
    if feature not in FEATURE_OSMIUM_FILTERS:
        return f"[UNKNOWN] {feature}/{year}"

    pbf = os.path.join(SNAPSHOTS_DIR, f"canada_{year}.osm.pbf")
    if not os.path.exists(pbf):
        return f"[MISSING PBF] {feature}/{year}"

    out_dir = os.path.join(RASTER_DIR, feature)
    os.makedirs(out_dir, exist_ok=True)
    out_tif = os.path.join(out_dir, f"canada_{feature}_{year}_1km_{OUTPUT_EPSG.replace(':','')}.tif")
    if os.path.exists(out_tif):
        return f"[SKIP] {feature}/{year}"

    # Temporary filtered pbf (small-ish, stream generated)
    tmp_dir = os.path.join(OUTPUT_DIR, "_tmp_osm_rasterize")
    os.makedirs(tmp_dir, exist_ok=True)
    filtered_pbf = os.path.join(tmp_dir, f"canada_{feature}_{year}.osm.pbf")
    reprojected_gpkg = os.path.join(tmp_dir, f"canada_{feature}_{year}_{INTERNAL_EPSG_M.replace(':','')}.gpkg")
    internal_tif = os.path.join(tmp_dir, f"canada_{feature}_{year}_1km_{INTERNAL_EPSG_M.replace(':','')}.tif")

    # 1) Filter
    cmd = ["osmium", "tags-filter", "-O", "-o", filtered_pbf, pbf] + FEATURE_OSMIUM_FILTERS[feature]
    run(cmd)

    # 2) Reproject to TARGET_EPSG on disk (GDAL does NOT reproject geometries during rasterize;
    #    setting -a_srs only sets output metadata). This keeps memory bounded.
    if os.path.exists(reprojected_gpkg):
        try:
            os.remove(reprojected_gpkg)
        except OSError:
            pass

    layers = FEATURE_LAYER_PREFERENCE.get(feature, ["lines", "multipolygons", "multilinestrings"])
    # Pick the first layer that exists in this filtered extract.
    available = _run_capture(["ogrinfo", "-ro", filtered_pbf]).splitlines()
    available_layers = []
    for line in available:
        line = line.strip()
        # format: "1: lines (Line String)"
        if ":" in line and "(" in line:
            name = line.split(":", 1)[1].split("(", 1)[0].strip()
            if name:
                available_layers.append(name)
    chosen_layer = None
    for cand in layers:
        if cand in available_layers:
            chosen_layer = cand
            break
    if chosen_layer is None:
        return f"EMPTY {feature}/{year} (no layers found in filtered extract)"

    # Convert only the chosen layer to GeoPackage and reproject.
    run([
        "ogr2ogr",
        "--config", "OGR_INTERLEAVED_READING", "YES",
        "-f", "GPKG",
        "-t_srs", INTERNAL_EPSG_M,
        reprojected_gpkg,
        filtered_pbf,
        chosen_layer,
    ])

    # 3) Rasterize (binary presence)
    minx, miny, maxx, maxy = _bbox_to_target_srs_te()
    run([
        "gdal_rasterize",
        "-init", "0",
        "-burn", "1",
        "-ot", "Byte",
        # IMPORTANT: don't use 0 as NoData for a 0/1 mask; QGIS will often render
        # zeros as transparent (NoData), making the layer look empty. Use 255.
        "-a_nodata", "255",
        "-of", "GTiff",
        "-co", "COMPRESS=LZW",
        "-co", "TILED=YES",
        "-a_srs", INTERNAL_EPSG_M,
        "-te", str(minx), str(miny), str(maxx), str(maxy),
        "-tr", "1000", "1000",
        "-tap",
        "-l", chosen_layer,
        reprojected_gpkg,
        internal_tif,
    ])

    # 4) Warp to EPSG:4326 output.
    # Use -r max (not near): nearest can erase thin 1-pixel "hits" when reprojecting
    # from EPSG:3978 to geographic, which looks like "southern Canada missing" on the map.
    # max preserves binary presence (0/1) as long as any sub-pixel contribution is 1.
    # Approx degree size for ~1km in latitude. In EPSG:4326 this is only an approximation
    # (true ground size varies with latitude, especially in longitude).
    approx_deg_1km = "0.008983152841"  # ~ 1km / 111.32km-per-degree

    run([
        "gdalwarp",
        "-t_srs", OUTPUT_EPSG,
        "-r", "max",
        "-ot", "Byte",
        "-srcnodata", "255",
        "-dstnodata", "255",
        # Clip to Canada bbox in EPSG:4326 to avoid world-wrapping extents
        "-te_srs", OUTPUT_EPSG,
        "-te",
        str(CANADA_BBOX_W), str(CANADA_BBOX_S), str(CANADA_BBOX_E), str(CANADA_BBOX_N),
        # Use an approximate ~1km output resolution in degrees for visualization/compatibility
        "-tr", approx_deg_1km, approx_deg_1km,
        "-tap",
        "-co", "COMPRESS=LZW",
        "-co", "TILED=YES",
        internal_tif,
        out_tif,
    ])

    # Cleanup temps to save space
    for tmp in (filtered_pbf, reprojected_gpkg, internal_tif):
        try:
            os.remove(tmp)
        except OSError:
            pass

    return f"OK {feature}/{year}: {os.path.basename(out_tif)}"


def step4_rasterize_features_to_1km_tifs():
    print("\n" + "="*60)
    print("STEP 4: Rasterize features to 1km binary GeoTIFFs (0/1)")
    print("="*60)
    tasks = [(year, feature) for feature in FEATURE_OSMIUM_FILTERS.keys() for year in YEARS]
    print(f"  {len(tasks)} tasks across {len(YEARS)} years × {len(FEATURE_OSMIUM_FILTERS)} features")
    print(f"  Output: {RASTER_DIR}")
    print(f"  Grid: {INTERNAL_EPSG_M} @ 1000m, warped to {OUTPUT_EPSG}")

    for yr, feat in tqdm(tasks, total=len(tasks), desc="rasterizing"):
        msg = rasterize_year_feature_binary(yr, feat)
        print(f"  {msg}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()
    step1_download_history()
    step2_extract_canada()
    step3_yearly_snapshots()
    step4_rasterize_features_to_1km_tifs()
    print(f"\nAll done in {(time.time() - t0) / 3600:.1f} hours.")
    print(f"1km tifs: {RASTER_DIR}")
