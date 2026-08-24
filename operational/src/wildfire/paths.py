"""Filesystem locations and the GEE grid contract.

Everything this application writes goes under ``WILDFIRE_DATA_ROOT``. Nothing
is written to the repository or to the user's home directory.
"""

import os

# ---------------------------------------------------------------------------
# Ours (writable)
# ---------------------------------------------------------------------------
DATA_ROOT = os.environ.get("WILDFIRE_DATA_ROOT", "/var/lib/wildfire")

RAW = os.path.join(DATA_ROOT, "raw")          # earthaccess HDF granules
DRIVERS = os.path.join(DATA_ROOT, "drivers")  # GEE-format per-tile GeoTIFFs
REPORTS = os.path.join(DATA_ROOT, "reports")  # availability JSONL, parity CSVs
WORK = os.path.join(DATA_ROOT, "work")        # harness scratch: symlink farms, stage outputs

# Operational MODIS preprocessing and its publishable handoff.
PROCESSED_MODIS = os.path.join(DATA_ROOT, "processed", "modis")
MODIS_PACKAGES = os.path.join(DATA_ROOT, "packages", "modis")
MODIS_PROCESSING_REPORTS = os.path.join(REPORTS, "modis_processing")
MODIS_STAGE_DIRS = {
    name: os.path.join(PROCESSED_MODIS, name)
    for name in (
        "qa_combined", "spatial_filled", "vegetation_indices",
        "downsampled", "merged", "unified", "daily_filled", "scaled",
    )
}

# IFS weather and its derived VPD / FWI products.  `fwi_state` holds the
# per-tile FFMC/DMC/DC checkpoints the recursion resumes from; it is state, not
# a stage output, so it sits beside the stage directories rather than in them.
PROCESSED_WEATHER = os.path.join(DATA_ROOT, "processed", "weather")
WEATHER_PACKAGES = os.path.join(DATA_ROOT, "packages", "weather")
WEATHER_PROCESSING_REPORTS = os.path.join(REPORTS, "weather_processing")
WEATHER_STAGE_DIRS = {
    name: os.path.join(PROCESSED_WEATHER, name)
    for name in ("derived", "merged", "unified", "scaled")
}
FWI_STATE = os.path.join(PROCESSED_WEATHER, "fwi_state")

# Corrected CanFireCast model-input weather products.  These are deliberately
# separate from the raw IFS diagnostic weather route above.
PROCESSED_MODEL_WEATHER = os.path.join(DATA_ROOT, "processed", "model_weather")
MODEL_WEATHER_STAGE_DIRS = {
    name: os.path.join(PROCESSED_MODEL_WEATHER, name)
    for name in ("base_corrected", "cffdrs_corrected", "encoded")
}
CANFIRECAST_PACKAGES = os.path.join(DATA_ROOT, "packages", "canfirecast")
CANFIRECAST_ASSETS = os.path.join(DATA_ROOT, "assets", "canfirecast")
CANFIRECAST_PREDICTIONS = os.path.join(DATA_ROOT, "predictions", "canfirecast")
CANFIRECAST_PROCESSING_REPORTS = os.path.join(REPORTS, "canfirecast_processing")

# Read-only national reference raster used as the alignment grid.
GRID_TEMPLATE = os.environ.get("WILDFIRE_GRID_TEMPLATE", "")

# Region used to cut the 2x2 tile grid (gee_download_fast_copy_parallel.py:490).
CANADA_GRIDS_SHP = os.environ.get("WILDFIRE_CANADA_GRIDS_SHP", "")

# ---------------------------------------------------------------------------
# The GEE grid contract
# ---------------------------------------------------------------------------
# Earth Engine converts a metre `scale` to degrees with the WGS84 equatorial
# metres-per-degree constant, then builds the projection with the transform
# [s, 0, 0, 0, -s, 0] -- origin at lon 0 / lat 0. Every tile of every product on
# every date therefore lands on one global lattice.
#
# Verified against the reference tiles: pixel size matches res/METRES_PER_DEGREE
# to ~1e-18, and every origin is an exact integer multiple of the pixel size.
METRES_PER_DEGREE = 6378137.0 * 3.141592653589793 / 180.0  # 111319.49079327358

# Export scale per product, from data_sources in gee_download_fast_copy_parallel.py.
NATIVE_RES_M = {
    "MOD09GA_b1237": 500, "MYD09GA_b1237": 500,
    "MOD09GA_QA_state1km": 1000, "MYD09GA_QA_state1km": 1000,
    "MOD11A1": 1000, "MYD11A1": 1000,
    "MOD14A1": 1000, "MYD14A1": 1000,
    "MCD15A3H": 500,
    "MOD09CMG": 5600, "MYD09CMG": 5600,
    "MOD09A1": 500, "MYD09A1": 500,
    "MOD11A2": 1000, "MYD11A2": 1000,
    "MOD13Q1": 250, "MYD13Q1": 250,
    "MCD12Q1": 500,
    "ERA5": 11132, "IFS": 11132,
}


def register_native_resolutions(mapping):
    """Fold a domain's export resolutions into NATIVE_RES_M.

    Export resolutions are part of the canonical parity contract, and the MODIS
    registry is authoritative for its own products.  This module used to import
    that registry directly, which made the shared filesystem contract depend on
    a domain package.  The dependency now runs the other way:
    `wildfire.modis.registry` calls this at import time.

    A key that arrives with a value different from the one already recorded is
    a contract violation rather than an update -- the literals above and the
    registry are two statements of the same thing, and they disagreeing means
    one of them is wrong.  Say so instead of letting the later writer win.
    """
    for name, res_m in mapping.items():
        existing = NATIVE_RES_M.get(name)
        if existing is not None and existing != res_m:
            raise ValueError(
                f"conflicting native resolution for {name}: "
                f"{existing} m already recorded, {res_m} m registered"
            )
        NATIVE_RES_M[name] = res_m


def pixel_size(res_m):
    """Degrees per pixel for a GEE export at `res_m` metres."""
    return res_m / METRES_PER_DEGREE


def is_on_lattice(coord, px, tol=1e-9):
    """Is this coordinate an integer multiple of the pixel size?"""
    n = coord / px
    return abs(n - round(n)) < tol
