# FireMPC operational CanFireCast store

This application does one thing: **given a range of dates, it fills those dates
into the monthly CanFireCast store.**

```text
$WILDFIRE_DATA_ROOT/packages/canfirecast/YYYY-MM.h5
```

```bash
python operational/cli.py fill --start 2026-06-01 --end 2026-06-30
```

Nothing here runs a model. Inference, maps, dashboards and evaluation live in
other repositories; they read the store through `canfirecast store publish`
(below) and are not this repository's concern.

"Operational" means the inputs are taken from sources that can be fetched as
close to the date as possible: **IFS** forecast weather from Earth Engine
rather than the reanalysis CanFireCast was trained on, and **MODIS** granules
through **earthaccess** rather than a curated archive. The frozen adapters that
make IFS stand in for ERA5-Land are what make that legitimate.

## How a date is built

A prediction target `D` reads the ten-day window `D-10 .. D-1`. So:

* IFS is downloaded for `D-10 .. D-1`, its 13 shared direct meteorological
  fields are mapped into the ERA5-Land training domain with frozen parameters,
  and the frozen residual correction is applied.
* MODIS is resolved through the frozen 17-channel processing with
  latest-available filling, seeing only observations that had landed by `D`.
* Static layers (OSM, terrain, population), the grid template, and both
  correction artifacts come from an immutable, hash-verified release.
* The result is folded into the month files, **one date stored once**.

Two consequences shape everything else, and neither is negotiable:

**A date has exactly one correct writer.** MODIS is frozen at first write, so
whichever target first touches a date fixes how much hindsight that date
carries. Date `X`'s writer is target `X+1`. `fill` therefore takes a range of
*dates*, runs targets `start+1 .. end+1`, and steps one day at a time. A wider
step is a different experiment, not a faster one.

**Channels 19–25 are target-dependent, not date-dependent.** They come from a
CFFDRS recursion reseeded per target from that target's own certified `D-11`
state. The store holds pre-recursion float32 weather and derives them at read
time. A dense multi-date file cannot represent them — measured, the same date
read through two different targets differs by up to 154/255 on channel 22.

## Environment

FireMPC's existing `gee_down` conda environment is the only environment
contract. Do not create a nested one.

```bash
cd ~/FireMPC
conda env update -f environment.yml --prune
conda activate gee_down
python operational/cli.py --help
```

Required once, per machine:

- Earthdata Login (`~/.netrc`) and Earth Engine access (`~/.config/earthengine`);
- GDAL command-line tools plus the dependencies in `environment.yml`;
- a mounted writable `WILDFIRE_DATA_ROOT`;
- a national grid template and the Canada grid shapefile;
- pinned OSM, WorldPop, and terrain rasters;
- an authoritative ERA5-derived FFMC/DMC/DC reference state to seed the ledger.

Copy `config/operational.env.example` to a machine-local file and replace every
placeholder. Credentials, static assets, state, downloads, packages and logs
stay outside Git.

## One-time bootstrap

Provision copies and hashes every input into an immutable release:

```bash
python operational/cli.py canfirecast provision create \
  --release operational-v1 \
  --grid-template /path/to/national_grid.tif \
  --osm /path/to/OSM_2025.tif \
  --population /path/to/worldpop_2025.tif \
  --terrain /path/to/terrain.tif
python operational/cli.py canfirecast provision verify --release operational-v1
```

Then seed the certified state ledger from a trusted national FWI raster:

```bash
python operational/cli.py canfirecast state-ledger import-reference-fwi \
  --source /path/to/authoritative/FWI_merged/2025_12_31.tif \
  --date 2025-12-31 --evidence 'authoritative ERA5-Land reference state'
```

`fill` advances the ledger from there itself. The ledger is an ERA5-Land
recursion that cannot skip a day, and the pipeline rejects cold starts,
uncertified state, skipped days, wrong state dates, grid mismatches, and
altered release files.

## Filling dates

```bash
# a past month
python operational/cli.py fill --start 2026-06-01 --end 2026-06-30

# catch up to now; ERA5-Land latency is well inside the D-11 the ledger needs
python operational/cli.py fill --start 2026-08-28 --end yesterday
```

`fill` does, in order:

1. `state-ledger advance` through the last target's `D-11` — skipped if the
   ledger is already there, and it downloads ERA5-Land to get there.
2. MODIS download and preprocess **once** for the whole range
   (`first-45 .. last-1`). Pass `--no-modis-ingest` when
   `processed/modis/scaled` already covers it.
3. Per target, in date order: IFS download → align/encode → physical correction
   → `modis package --as-of D` → `store ingest`, then delete the redundant
   `packages/modis/<D>/` handoff and that target's scratch.

It is resumable: a date already frozen in the store is skipped, so re-running a
range costs only what is missing. Reckon on ~85 s per target once MODIS is
preprocessed. A run manifest lands in
`$WILDFIRE_DATA_ROOT/reports/canfirecast_processing/fill/<run-id>/`.

Useful flags: `--overwrite` rebuilds frozen dates, `--keep-work` keeps the
per-target scratch, `--min-free-gb` aborts before a target rather than filling
the disk mid-write.

## Checking and reading the store

```bash
python operational/cli.py canfirecast store status
```

To hand one target's window to inference code elsewhere, publish it. This is
the store's read API and the only correct way to get channels 19–25:

```bash
python operational/cli.py canfirecast store publish \
  --target 2026-08-20 \
  --static $WILDFIRE_DATA_ROOT/assets/canfirecast/operational-v1/statics.h5 \
  --correction-artifact $WILDFIRE_DATA_ROOT/assets/canfirecast/operational-v1/cffdrs_aware_correction.json \
  --output /path/for/inference/window.h5
```

The result is a schema-version-1 `uint8 (10,55,H,W)` package, written by atomic
replace, validated before it lands. A target is publishable once the store
holds `D-10 .. D-1` and the ledger holds `D-11`, so the earliest publishable
target is ten days after a store's first date. Reckon on ~1 minute and ~300 MB
per window.

Before model execution, `CanFireCastIFSCommon` selects the 49 common channels
`1,2,3,5,7..14,18..54`; 55-entry normalization statistics are projected after
selection. Ten is the input-window length — the "15day" experiment name means
fifteen forecast-horizon checkpoints, not fifteen input days.

## Individual stages

`fill` drives them; each is separately runnable for debugging, and
`python operational/cli.py --help` is the index:

- `modis` — granule download, the frozen 17-channel processing, packaging.
- `weather` — IFS/ERA5 download, VPD, the recursive FWI codes.
- `canfirecast pipeline` — one target end to end, including the MODIS lookback
  that `fill` hoists out of the loop. Fine for a single date, wasteful for a
  range.
- `canfirecast state-ledger` — seed, advance and verify the certified state.
- `canfirecast store` — `ingest`, `read`, `publish`, `status`.

See `NOTES.md` for what building the 2026 store taught, and `vendor/README.md`
plus `IMPORT_PROVENANCE.md` for the frozen dependencies.
