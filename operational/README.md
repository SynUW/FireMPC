# FireMPC operational CanFireCast package creation

This directory is the near-real-time application inside FireMPC. The historical
FireMPC scripts at repository top level remain the training-dataset workflow;
this application downloads recent IFS and MODIS inputs, invokes the required
legacy preprocessing stages, and creates the operational interchange package.

CanFireCast was trained with ERA5-Land weather. Operations use IFS for dates
`D-10..D-1`, map its 13 shared direct meteorological fields into the ERA5-Land
training domain with frozen parameters, apply the frozen residual correction,
and recursively derive VPD and CFFDRS from certified `D-11` state. MODIS uses
the frozen 17-channel processing and latest-available filling. Static layers
and correction parameters come from an immutable release.

The default output is a deduplicated, append-only monthly store:

```text
$WILDFIRE_DATA_ROOT/packages/canfirecast/YYYY-MM.h5
```

Each date is stored once. A model window is reconstructed as schema-version-1
`uint8 (10,55,H,W)` data, with target-dependent CFFDRS channels derived at read
time from the certified D-11 state. After a successful monthly ingestion, the
redundant `packages/modis/<target>/` handoff is removed automatically. Use
`--keep-modis` on the CanFireCast pipeline only when retaining that intermediate
package for debugging.

The model does not read the store. The last pipeline stage, `publish`, reads
target `D`'s ten-day window back out of it and writes that window as an
ordinary package to one fixed path:

```text
$WILDFIRE_DATA_ROOT/packages/canfirecast/daily.h5
```

That file is the model's input. It is replaced -- atomically, by rename -- at
the end of every run, so a consumer is pointed at it once and never has to
work out today's date, and a reader always sees either yesterday's package or
today's, never a partial one. History lives in the month files; only the
current window is materialized. Republishing any past target is a read away:

```bash
python operational/cli.py canfirecast store publish \
  --target 2026-08-20 --static /path/to/statics.h5 --output /tmp/window.h5
```

## Historical testing

A past month needs no rebuild once its dates are in the store: a target costs
one window read plus the CFFDRS recursion, seeded from that target's own D-11
ledger state, and nothing is downloaded or reprocessed. Replay a range into
per-target files, which keeps the sweep away from the daily package the model
is running on:

```bash
python operational/cli.py canfirecast store replay \
  --start 2026-07-02 --end 2026-08-01 \
  --static  $WILDFIRE_DATA_ROOT/assets/canfirecast/operational-v1/statics.h5 \
  --correction-artifact $WILDFIRE_DATA_ROOT/assets/canfirecast/operational-v1/cffdrs_aware_correction.json
# -> packages/canfirecast/hindcast/<D>.h5
```

The sweep pre-checks every target -- its ten stored dates and its certified
D-11 state -- and refuses up front rather than failing partway through; pass
`--skip-incomplete` to replay only what the store covers. A target is
replayable once the store holds `D-10..D-1`, so the earliest replayable target
is nine days after a month file's first date. Reckon on a few minutes and
roughly 300 MB per target; if the model runs inline, `store publish` one target
at a time keeps a single file on disk instead.

A hindcast is only as honest as what the store froze. MODIS is frozen at first
write, so a replay can never be shown a granule that had not yet landed -- but
the freeze also records the fill policy of the run that wrote it, and a range
whose dates were ingested under different policies is not a single experiment.
`modis_fill_policy_version` in the run manifests says which.

The legacy per-target materialization remains available with
`canfirecast pipeline --storage daily`, which additionally keeps
`packages/canfirecast/<D>/canfirecast_inputs.h5`. Ten is the input-window
length. The experiment name “15day” means fifteen separate forecast-horizon
checkpoints, not fifteen input days. Before model execution,
`CanFireCastIFSCommon` selects the exact 49 common channels
`1,2,3,5,7..14,18..54`; checkpoint-compatible 55-entry normalization statistics
are projected after selection. Model code, normalization files, checkpoints,
inference, maps, and dashboard hosting are separate from package creation.

## Environment

FireMPC uses its existing `gee_down` conda environment as the only environment
contract:

```bash
cd ~/FireMPC
conda env update -f environment.yml --prune
conda activate gee_down
python operational/cli.py --help
```

Do not create an operational uv project or nested environment. Python
subprocesses use `FIREMPC_PYTHON`, defaulting to the active interpreter.

Required accounts and deployment inputs are:

- Earthdata Login and Earth Engine access/project;
- GDAL command-line tools plus the dependencies in `environment.yml`;
- a mounted writable `WILDFIRE_DATA_ROOT`;
- a national grid template and Canada grid shapefile;
- pinned OSM, WorldPop, and terrain rasters;
- an authoritative ERA5-derived FFMC/DMC/DC reference state.

Copy `config/operational.env.example` to a machine-local configuration or a
systemd environment file and replace every placeholder. Credentials, static
assets, state, downloads, packages, and logs stay outside Git.

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

Import a trusted national state, including evidence, then advance without gaps
through the target's `D-11` date:

```bash
python operational/cli.py canfirecast state-ledger import-reference-fwi \
  --source /path/to/authoritative/FWI_merged/2025_12_31.tif \
  --date 2025-12-31 --evidence 'authoritative ERA5-Land reference state'
python operational/cli.py canfirecast state-ledger advance --through 2026-08-09
```

The pipeline rejects cold starts, uncertified state, skipped recursive days,
wrong state dates, grid mismatches, and altered release files.

## Daily package creation

```bash
python operational/cli.py canfirecast pipeline \
  --target 2026-08-20 --release operational-v1
# Optional legacy per-target materialization:
python operational/cli.py canfirecast pipeline \
  --target 2026-08-20 --release operational-v1 --storage daily
python operational/daily.py --target today
```

The pipeline validates reconstructed-window dimensions, channel order, dtype,
dates, state lineage, geospatial metadata, validity data, and frozen dependency
hashes, and revalidates the published daily package as a file after writing
it.
MODIS source dates and fill ages are retained in the H5 and manifests. A run
uses release copies of both fitted correction JSON files and never refits them.

See `deploy/README.md` for systemd installation. Full acceptance additionally
requires an end-to-end cached rehearsal and byte-identical comparison with the
pre-migration fixture. An optional external one-patch model smoke test can then
verify the 49-channel model interface.
