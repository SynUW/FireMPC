import time
from pathlib import Path
import cdsapi
import sys

DATASET = "derived-fire-fuel-biomass"
VARIABLES = [
    "dead_fuel_moisture_content_group",
    "live_fuel_moisture_content_group",
    "fuel_group",
]
VERSION = "1"

YEARS = [str(y) for y in range(2003, 2022)]
MONTHS = [f"{m:02d}" for m in range(1, 13)]

# BC bounding box: [North, West, South, East]
AREA = [60, -140, 48, -114]

OUT_DIR = Path("/mnt/raid/user/wildfire_dataset/self_built_materials/derived_fire_fuel_biomass")
OUT_DIR.mkdir(parents=True, exist_ok=True)

client = cdsapi.Client()

# Compute total task count and statistics
total_tasks = len(VARIABLES) * len(YEARS) * len(MONTHS)
completed_tasks = 0
skipped_tasks = 0
failed_tasks = 0

print(f"Starting download task, total {total_tasks} files...")
print("=" * 50)

for var in VARIABLES:
    for y in YEARS:
        for m in MONTHS:
            completed_tasks += 1
            target = OUT_DIR / f"{DATASET}_{var}_{y}-{m}.zip"
            
            # Dynamic progress display
            progress = f"[{completed_tasks}/{total_tasks}]"
            sys.stdout.write(f"\r{progress} processing {var} {y}-{m}...")
            sys.stdout.flush()
            
            if target.exists():
                skipped_tasks += 1
                continue

            req = {
                "variable": var,
                "version": VERSION,
                "year": y,
                "month": m,
                "area": AREA
            }

            try:
                print(f"\n{progress} downloading {var} {y}-{m}...")
                client.retrieve(DATASET, req).download(str(target))
                time.sleep(1)
            except Exception as e:
                failed_tasks += 1
                print(f"\n{progress} download failed {var} {y}-{m}: {e}")
                # If download fails, delete the potentially incomplete file
                if target.exists():
                    target.unlink()

print("\n" + "=" * 50)
print("Download complete!")
print(f"Total tasks: {total_tasks}")
print(f"Successfully downloaded: {total_tasks - skipped_tasks - failed_tasks}")
print(f"Skipped files: {skipped_tasks}")
print(f"Failed tasks: {failed_tasks}")
