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
time from the certified D-11 state. The legacy materialized package remains
available with `canfirecast pipeline --storage daily` and is written to
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
hashes.
MODIS source dates and fill ages are retained in the H5 and manifests. A run
uses release copies of both fitted correction JSON files and never refits them.

See `deploy/README.md` for systemd installation. Full acceptance additionally
requires an end-to-end cached rehearsal and byte-identical comparison with the
pre-migration fixture. An optional external one-patch model smoke test can then
verify the 49-channel model interface.
