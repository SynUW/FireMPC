"""Variable catalogue builder: enumerate all 40+ driving factor variables.

Produces a CSV with columns:
  variable_name:       human-readable variable name
  source:              data source (ERA5-Land, MODIS, MERIT DEM, etc.)
  resolution:          spatial resolution (e.g., "0.1 degree (~9 km)")
  temporal_frequency:  sampling frequency (e.g., "daily", "monthly", "static")
  preprocessing_notes: unit conversions, band index, known issues
  coverage_gap_flag:   True if >1% of expected daily files are missing
  directory_path:      full path on server (for verification)
  file_count:          actual number of .tif files found
  expected_count:      expected number of files based on 2000-2024 span
  missing_pct:         percentage of expected files missing

The 18 ERA5-Land bands are enumerated from the known band structure in CONTEXT.md.
The 7 MERIT DEM bands are enumerated from the Canada_Full_DEM_1km_MERIT_norm_stack.tif.
Additional variable directories are discovered by scanning /s3/.../drivers/.
"""
import csv
import glob
import os

DRIVERS_BASE = ("/s3/zhengsen/wildfire_dataset/self_built_materials/"
                "bc_raw_data_new_fast/drivers")
ERA5_DIR     = os.path.join(DRIVERS_BASE, "ERA5")
MODIS_DIR    = os.path.join(DRIVERS_BASE, "processed", "MCD14A1_combined")
DEM_FILE     = os.path.join(DRIVERS_BASE, "merit_dem",
                             "dem_slope_aspect_hillshade",
                             "Canada_Full_DEM_1km_MERIT_norm_stack.tif")

# ERA5-Land 18 bands (known from CONTEXT.md -- hardcoded for correctness)
ERA5_BANDS = [
    {"variable_name": "temperature_2m",               "band": 1,  "unit": "K (daily mean)",
     "preprocessing_notes": "subtract 273.15 for deg C; daily mean"},
    {"variable_name": "u_component_of_wind_10m",      "band": 2,  "unit": "m/s",
     "preprocessing_notes": "use with Band 3 for wind speed: sqrt(u2+v2)*3.6 km/h"},
    {"variable_name": "v_component_of_wind_10m",      "band": 3,  "unit": "m/s",
     "preprocessing_notes": "use with Band 2 for wind speed"},
    {"variable_name": "snow_cover",                   "band": 4,  "unit": "%",
     "preprocessing_notes": "no conversion required"},
    {"variable_name": "total_precipitation_sum",      "band": 5,  "unit": "m (daily sum)",
     "preprocessing_notes": "multiply by 1000 for mm; FWI uses mm"},
    {"variable_name": "surface_latent_heat_flux_sum", "band": 6,  "unit": "J/m2",
     "preprocessing_notes": "no conversion; used as-is"},
    {"variable_name": "dewpoint_temperature_2m",      "band": 7,  "unit": "K",
     "preprocessing_notes": "subtract 273.15 for deg C; used for RH via Magnus/Tetens with Band 13"},
    {"variable_name": "surface_pressure",             "band": 8,  "unit": "Pa",
     "preprocessing_notes": "divide by 100 for hPa if needed"},
    {"variable_name": "volumetric_soil_water_layer_1","band": 9,  "unit": "m3/m3",
     "preprocessing_notes": "layer 1 = top 7 cm"},
    {"variable_name": "volumetric_soil_water_layer_2","band": 10, "unit": "m3/m3",
     "preprocessing_notes": "layer 2 = 7-28 cm"},
    {"variable_name": "volumetric_soil_water_layer_3","band": 11, "unit": "m3/m3",
     "preprocessing_notes": "layer 3 = 28-100 cm"},
    {"variable_name": "volumetric_soil_water_layer_4","band": 12, "unit": "m3/m3",
     "preprocessing_notes": "layer 4 = 100-289 cm"},
    {"variable_name": "temperature_2m_max",           "band": 13, "unit": "K (daily max)",
     "preprocessing_notes": "noon T proxy for FWI; subtract 273.15 for deg C; D-02 decision"},
    {"variable_name": "skin_temperature_max",         "band": 14, "unit": "K",
     "preprocessing_notes": "subtract 273.15 for deg C"},
    {"variable_name": "potential_evaporation_sum",    "band": 15, "unit": "m",
     "preprocessing_notes": "multiply by 1000 for mm"},
    {"variable_name": "total_evaporation_sum",        "band": 16, "unit": "m",
     "preprocessing_notes": "multiply by 1000 for mm"},
    {"variable_name": "skin_reservoir_content",       "band": 17, "unit": "m water equiv",
     "preprocessing_notes": "no conversion required"},
    {"variable_name": "surface_net_solar_radiation_sum", "band": 18, "unit": "J/m2",
     "preprocessing_notes": "no conversion required"},
]

# MERIT DEM 7 bands (static normalized stack)
DEM_BANDS = [
    {"variable_name": "dem (normalized)",          "band": 1,
     "notes": "Digital Elevation Model; normalized 0-1"},
    {"variable_name": "slope (normalized)",        "band": 2,
     "notes": "Terrain slope in degrees; normalized 0-1"},
    {"variable_name": "aspect_sine (normalized)",  "band": 3,
     "notes": "sin(aspect); normalized 0-1; captures N-S orientation"},
    {"variable_name": "aspect_cosine (normalized)","band": 4,
     "notes": "cos(aspect); normalized 0-1; captures E-W orientation"},
    {"variable_name": "hillshade (normalized)",    "band": 5,
     "notes": "Hillshade illumination; normalized 0-1"},
    {"variable_name": "tpi (normalized)",          "band": 6,
     "notes": "Topographic Position Index; normalized 0-1"},
    {"variable_name": "twi (normalized)",          "band": 7,
     "notes": "Topographic Wetness Index; normalized 0-1"},
]

# Expected daily tile count for ERA5 (2000-01-01 to 2025-12-31 = approx 9497 days)
ERA5_EXPECTED_DAYS = 9497

# Expected daily count for MODIS (2000-01-01 to 2024-12-31 = 9131 days)
MODIS_EXPECTED_DAYS = 9131


def count_tif_files(directory: str) -> int:
    """Count .tif files in directory (non-recursive)."""
    if not os.path.isdir(directory):
        return 0
    return len(glob.glob(os.path.join(directory, "*.tif")))


def build_variable_catalogue(output_path: str = None) -> list:
    """Enumerate all driving variables and write catalogue CSV.

    Args:
        output_path: path to write CSV. If None, writes to catalogue/outputs/variable_catalogue.csv.

    Returns:
        list of dicts, one per variable.
    """
    if output_path is None:
        default_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "catalogue", "outputs"
        )
        os.makedirs(default_dir, exist_ok=True)
        output_path = os.path.join(default_dir, "variable_catalogue.csv")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    era5_file_count = count_tif_files(ERA5_DIR)
    era5_missing_pct = max(0.0,
                           (ERA5_EXPECTED_DAYS - era5_file_count) / ERA5_EXPECTED_DAYS * 100)
    era5_gap = era5_missing_pct > 1.0

    rows = []

    # 1. ERA5-Land bands (18 variables)
    for band_info in ERA5_BANDS:
        rows.append({
            "variable_name":       band_info["variable_name"],
            "source":              "ERA5-Land daily aggregated (ECMWF via Google Earth Engine)",
            "resolution":          "0.1 degree (~9 km) regridded to Canada domain",
            "temporal_frequency":  "daily",
            "preprocessing_notes": (f"Band {band_info['band']}; unit: {band_info['unit']}; "
                                     f"{band_info['preprocessing_notes']}"),
            "coverage_gap_flag":   str(era5_gap),
            "directory_path":      ERA5_DIR,
            "file_count":          era5_file_count,
            "expected_count":      ERA5_EXPECTED_DAYS,
            "missing_pct":         f"{era5_missing_pct:.2f}",
        })

    # 2. FWI computed components (6 derived variables -- computed from ERA5, not stored separately)
    fwi_components = [
        ("FFMC", "Fine Fuel Moisture Code",  "dimensionless [0-101]",
         "Computed from ERA5 Bands 2,3,5,7,13 via Van Wagner (1987); startup FFMC=85 April 1"),
        ("DMC",  "Duff Moisture Code",       "dimensionless",
         "Computed from ERA5 Bands 2,3,5,7,13 via Van Wagner (1987); startup DMC=6 April 1"),
        ("DC",   "Drought Code",             "dimensionless",
         "Computed from ERA5 Bands 2,3,5,13 via Van Wagner (1987); startup DC=15 April 1"),
        ("ISI",  "Initial Spread Index",     "dimensionless",
         "Derived from FFMC + wind speed; Van Wagner (1987)"),
        ("BUI",  "Buildup Index",            "dimensionless",
         "Derived from DMC + DC; Van Wagner (1987)"),
        ("FWI",  "Fire Weather Index",       "dimensionless",
         "Derived from ISI + BUI; Van Wagner (1987); primary hard-negative scoring component"),
    ]
    for abbr, name, unit, notes in fwi_components:
        rows.append({
            "variable_name":       f"FWI_{abbr}",
            "source":              "Computed: ERA5-Land + Van Wagner (1987) CFFDRS equations",
            "resolution":          "0.1 degree (~9 km), same as ERA5",
            "temporal_frequency":  "daily (fire season April-October only)",
            "preprocessing_notes": f"{name} ({unit}); {notes}",
            "coverage_gap_flag":   str(era5_gap),
            "directory_path":      "computed -- not stored separately",
            "file_count":          0,
            "expected_count":      0,
            "missing_pct":         "N/A",
        })

    # 3. MODIS fire detection (ground truth)
    modis_count = count_tif_files(MODIS_DIR)
    modis_missing_pct = max(0.0,
                            (MODIS_EXPECTED_DAYS - modis_count) / MODIS_EXPECTED_DAYS * 100)
    rows.append({
        "variable_name":       "MODIS_MCD14A1_firemask",
        "source":              ("MODIS Terra+Aqua Combined Active Fire Product "
                                "(MCD14A1, Collection 6.1)"),
        "resolution":          "1 km, regridded to Canada domain",
        "temporal_frequency":  "daily",
        "preprocessing_notes": ("3 bands: Band1=firemask(0-9), Band2=maxFRP, Band3=QA; "
                                 "positive label = firemask==9 (high confidence); "
                                 "exclusion buffer = firemask in {7,8}; cloud = firemask==4"),
        "coverage_gap_flag":   str(modis_missing_pct > 1.0),
        "directory_path":      MODIS_DIR,
        "file_count":          modis_count,
        "expected_count":      MODIS_EXPECTED_DAYS,
        "missing_pct":         f"{modis_missing_pct:.2f}",
    })

    # 4. MERIT DEM 7 topographic variables (static normalized stack)
    dem_dir = os.path.dirname(DEM_FILE)
    dem_file_exists = os.path.isfile(DEM_FILE)
    dem_file_count = 1 if dem_file_exists else 0
    for dem_info in DEM_BANDS:
        rows.append({
            "variable_name":       dem_info["variable_name"],
            "source":              ("MERIT DEM (Multi-Error-Removed Improved-Terrain DEM); "
                                    "Canada_Full_DEM_1km_MERIT_norm_stack.tif"),
            "resolution":          "1 km",
            "temporal_frequency":  "static",
            "preprocessing_notes": (f"Band {dem_info['band']} of "
                                     f"Canada_Full_DEM_1km_MERIT_norm_stack.tif; "
                                     f"{dem_info['notes']}"),
            "coverage_gap_flag":   str(not dem_file_exists),
            "directory_path":      dem_dir,
            "file_count":          dem_file_count,
            "expected_count":      1,
            "missing_pct":         "0.00" if dem_file_exists else "100.00",
        })

    # 5. Additional driver variables -- discover from drivers/ subdirectories
    known_dirs = {'ERA5', 'ecoregions', 'processed', 'cwfis', 'merit_dem'}
    try:
        subdirs = [d for d in os.listdir(DRIVERS_BASE)
                   if os.path.isdir(os.path.join(DRIVERS_BASE, d))
                   and d not in known_dirs]
    except FileNotFoundError:
        subdirs = []

    for subdir in sorted(subdirs):
        dir_path = os.path.join(DRIVERS_BASE, subdir)
        file_count = count_tif_files(dir_path)
        # Use MODIS_EXPECTED_DAYS as conservative expected count for daily variables
        expected_daily = MODIS_EXPECTED_DAYS
        if file_count > 0:
            missing_pct = max(0.0, (expected_daily - file_count) / expected_daily * 100)
        else:
            missing_pct = 100.0
        gap_flag = missing_pct > 1.0

        rows.append({
            "variable_name":       subdir,
            "source":              f"Auto-discovered -- inspect {dir_path}",
            "resolution":          "TBD -- inspect tile metadata",
            "temporal_frequency":  "TBD -- infer from file_count",
            "preprocessing_notes": (f"Auto-discovered from {dir_path}; "
                                     "verify source and units"),
            "coverage_gap_flag":   str(gap_flag),
            "directory_path":      dir_path,
            "file_count":          file_count,
            "expected_count":      expected_daily,
            "missing_pct":         f"{missing_pct:.2f}",
        })

    # Write CSV
    fieldnames = ["variable_name", "source", "resolution", "temporal_frequency",
                  "preprocessing_notes", "coverage_gap_flag",
                  "directory_path", "file_count", "expected_count", "missing_pct"]
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_gaps = sum(1 for r in rows if r["coverage_gap_flag"] == "True")
    print(f"Variable catalogue written: {output_path}")
    print(f"  Total variables: {len(rows)}")
    print(f"  Variables with coverage gaps: {n_gaps}")
    return rows


if __name__ == "__main__":
    rows = build_variable_catalogue()
    for r in rows:
        flag = "GAP" if r["coverage_gap_flag"] == "True" else "   "
        print(f"  [{flag}] {r['variable_name']}: {r['file_count']} files")
