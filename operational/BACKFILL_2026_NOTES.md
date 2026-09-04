# 2026 backfill + prediction sweep — operator notes

Written 2026-09-02. Covers building the full-2026 CanFireCast store and running
next-day prediction maps over it. Everything here was learned by doing; the
parts that cost the most time are the traps in §3.

## 1. What exists now

**Store** — `$WILDFIRE_DATA_ROOT/packages/canfirecast/YYYY-MM.h5`

* 2026-01-01 … 2026-08-27, all 239 dates frozen, no gaps (~8.9 GB).
* 230 targets replayable: **2026-01-11 … 2026-08-28**.
* Bounds are structural, not arbitrary: the first target needs a full D-10…D-1
  window (so 01-11) *and* certified D-11 state (ledger starts 2025-12-31, so
  again 01-11). The last needs D-1 ≤ 08-27 (drivers stop there), so 08-28.

**Predictions** — day01 horizon, `YYYY_MM_DD.tif` (probability, float32,
nodata -9999) + `_classified.tif` (int32, 4 classes), 928×1968 EPSG:4326.

| month | dir | targets |
|---|---|---|
| Jan–Mar, May, Jul, Aug | `/s3/quinn/prediction_vis/canfirecast_<month>2026/` | 21/28/31/31/31/28 |
| April | same pattern | 30 |
| June | `/caribou/quinn/wildfire/predictions/canfirecast_test_june/` | 30 |

June is the odd one out — it was the pilot, hence the `_test_` name and the
different filesystem. Worth consolidating onto `/s3`.

**Multi-horizon predictions** — 2026-09-03, all 15 leads for one week.

* `/s3/quinn/prediction_vis/canfirecast_2026_multi/`
* Targets 2026-08-01…08-07, leads day01…day15 → 105 rasters + 105 classified
  (473 MB), 21 valid dates, 9 of them (08-07…08-15) at the full 7-lead depth
  a 7-target run allows. See §9.

**Scripts** (added by this work)

* `operational/scripts/ingest_sweep.sh` — store backfill, one target per day.
* `operational/scripts/predict_sweep.sh` — publish → model → delete, per target.
* `operational/scripts/predict_multi_sweep.sh` — all leads from one published
  window, one directory per target.
* `operational/scripts/index_predictions.py` — the by-valid-date view over it.
* Logs: `/caribou/quinn/wildfire/logs/predict/`. The `run_*.sh` wrappers that
  drove each sweep live there too — they are the exact invocations used.

## 2. The two contracts that govern correctness

**as-of.** MODIS is frozen at first write, so whichever target first touches a
date fixes its provenance forever. Date X's *correct* writer is target **X+1**
(`--as-of X+1`), which sees only granules landed by then. Any wider gap hands
that date hindsight it would not have had operationally.

*Cold-start seam:* the first target of any sweep freezes all ten window dates at
once, so they carry up to 10 days of hindsight. **2026-01-01…01-10 are
permanently affected** — no earlier target can exist. A second seam at
06-22…06-30 (from the original Aug-26 sweep) was repaired 2026-09-02; the fix
is in §4.

**Channels 19–25 are target-dependent, not date-dependent.** They come from a
CFFDRS recursion reseeded per target from its own D-11 state. Measured: the same
date read through two different targets differs by up to **154/255** on ch22,
with 45–79% of pixels changing. Consequences:

* A dense multi-date H5 **cannot** represent them — one value per (date,channel)
  is the wrong shape. Do not build one.
* `cffdrs_derive.py:46` hard-rejects anything that isn't `(10,55,H,W)` anyway.
* Every target needs its own published package. `predict_sweep.sh` does
  publish → infer → delete so peak disk stays at one 300 MB file.

## 3. Traps

1. **`canfirecast pipeline` always re-downloads.** `modis_pipeline_command`
   (`canfirecast/pipeline.py:69`) never passes `--from-stage`, so every target
   runs MODIS download+preprocess over a 45-day lookback. For a backfill where
   `processed/modis/scaled` is already complete that is pure waste — it turns a
   14 h sweep into ~4 days. Use the three-step decomposition instead:
   `canfirecast pipeline --to-stage cffdrs-derive` → `modis pipeline --as-of D
   --from-stage package` → `canfirecast store ingest`. (~85 s/target.)
2. **…but that decomposition leaks.** `store ingest` called directly skips
   `remove_ingested_modis_handoff`, which lives in the `assemble` stage. Each
   target then leaves ~123 MB in `packages/modis/`. 178 dirs / 22 GB accumulated
   before this was caught. `ingest_sweep.sh` now cleans per target.
3. **Never `--to-stage publish` in a backfill.** `publish` overwrites the
   operational `daily.h5` with a historical window. Stop at `assemble`.
   (`validate` also re-reads the national window, ~4 min/target, doubling the
   sweep; every date gets validated later at publish time anyway.)
4. **Plain `--force` corrupts neighbours.** `ingest_window --force` rewrites the
   *entire* 10-day window. To fix specific dates, mark them incomplete and
   re-run without force — see §4.
5. **`--model-filter` is an exact `model_name` match** (`CanFireCastIFSCommon`),
   not a filename. Horizon selection is `--forecast-task N`. Getting this wrong
   produces **exit 0 and zero output** — `visualization_gen` logs "No model
   files found" as a warning. `predict_sweep.sh` now fails loudly on empty output.
6. **Torch claims the whole box.** `visualization_gen` sets no thread limits,
   so torch sizes both pools from `os.cpu_count()`: 32 intra-op + 32 inter-op =
   64 spinning threads, 4820% CPU, 75% of all runnable threads on a shared
   64-core machine — while the actual compute sits on the GPU at 99%. Cap with
   `OMP_NUM_THREADS`/`MKL_NUM_THREADS` **plus** `torch.set_num_interop_threads`,
   which has no env var and is set by `vis_overlay/sitecustomize.py` (that
   directory is already on `PYTHONPATH` for the model step). Measured cost:
   inference 8m12s → 8m17s per lead, box load 85 → 14. Put the caps on the
   *model* invocation only — the publish step's CFFDRS recursion is genuinely
   thread-parallel and doubles (1m43s → 3m33s) when capped.
7. **`set -u` breaks conda.** Conda's own activate/deactivate hooks reference
   unset variables (`geotiff-deactivate.sh:5`). Use `set -e` only in wrappers
   that activate environments.
8. **Root filesystem fills.** `/` (1.7 TB NVMe) hit 100% mid-run and killed a
   sweep with ENOSPC. Keep logs and `TMPDIR` on `/caribou` (separate 7 TB
   device, `sdc1`). `/data`, `/beluga`, `/seal` are identical siblings.

## 4. Repairing a cold-start seam

`--force` is wrong (trap 4). Instead:

```python
# 1. mark the affected dates incomplete
with h5py.File(store.month_path(root, day), "a") as h:
    h["complete"][store.day_index(day)] = False
```

```bash
# 2. re-run targets X+1 for each affected date X, WITHOUT --force.
#    ingest skips still-frozen neighbours, so each target writes only its own D-1.
START=<first X+1> END=<last X+1> bash operational/scripts/ingest_sweep.sh
```

Then delete the `.tif`s for every target whose window overlaps the repaired
dates (D ∈ [firstX+1, lastX+10]) and re-run `predict_sweep.sh`; its resume
check rebuilds only what is missing.

Verify afterwards that each date's `written_for_target == date + 1`.

## 5. Environment (fragile — worth fixing properly)

Three pieces, because the original is unreachable:

* **`gee_down`** — operational CLI. Has no torch.
* **`/opt/anaconda3/envs/gsilab_shared`** — model inference. Satisfies every
  import including the lazy ones (`geopandas`, `pyproj`, `shapely`, `wandb`).
  Missing `mamba_ssm`/`spikingjelly`, needed only by other architectures.
* **`/caribou/quinn/wildfire/work/vis_overlay`** on `PYTHONPATH` — `gsilab_shared`
  pairs geopandas 0.14.3 with fiona 1.10.1, which dropped `fiona.path`;
  reading the boundary shapefile raises `AttributeError`. The overlay is
  geopandas 1.1.4 + pyogrio installed `--no-deps`, so only geopandas is shadowed.

The runs that produced July on 2026-08-27 used
`/home/zhengsen/.conda/envs/mamba_env`, which is mode 0750 and unreadable.
Equivalence was **verified**: re-running 2026-07-15 under the stack above gave a
**bit-identical** raster. That also proves the pipeline is deterministic despite
MC dropout. Getting a portable env spec would remove this whole section.

## 6. Facts worth not re-deriving

* **Per target:** publish 54 s, model setup ~60 s (boundary rasterisation),
  inference ~9–11 min → **~12.6 min/target** at one horizon. All 15 horizons
  ≈ 1.9 GPU-h/target, so 230 targets × 15 ≈ **500 GPU-hours**.
* **Boundary mask** keeps 874,442 of 1,826,304 cells (47.9%). Do not pass
  `--no-boundary-mask` — it makes runs slower *and* non-comparable.
* **MODIS latency is seasonal, not broken.** MCD09GA domain completion: January
  p50 1.0 d (96.8% ≤1 d) vs July 2.0 d (0% ≤1 d) — fewer daylight tiles in
  winter complete the domain sooner. So January legitimately shows ~80% same-day
  channel-days where July shows 6%. This is not contamination.
* **Aqua outage 2026-03-17 … 05-10** — `MYD09GA` has no granules. Handled
  correctly: a single complete platform group suffices, so those dates are
  genuine same-day Terra-only observations, not stale fill. April windows are
  100% inside it; May's first ten days partly so.
* **Fill in July:** 94.3% of channel-days resolve to an earlier raster, typical
  age 1–3 d (MCD15A3H ~7.5 d, MCD12Q1 ~927 d by design). Expected under `--as-of`.
* **Contamination has no measurable effect on prediction level.** July targets
  07-02…07-10 (contaminated windows) vs 07-11+ (clean) were indistinguishable
  once the late-July seasonal decline was accounted for.

## 7. Open items

* **June output** lives on `/caribou` as `canfirecast_test_june`; move to `/s3`
  and rename for consistency.
* **No 2026 fire labels.** `drivers/MOD14A1`/`MYD14A1` hold 4 files each, all
  2025 — `OPERATIONAL_EXPORTS` deliberately omits fire labels. Everything so far
  is visual inspection, not measured skill. Downloading them is independent work.
* **2026-01-01…01-10** are not predictable (no D-11 state before 2025-12-31).
* **Live/forward operation** needs: MODIS+IFS caught up (drivers stop 08-27,
  ledger 08-23, i.e. ~5 days behind), `state-ledger advance`, then the pipeline
  run through `publish` to write `daily.h5` (which does not yet exist). See
  `deploy/README.md`.
* **Other horizons** — done for 2026-08-01…08-07 (§9). Extending to a
  fully-stacked 15-lead day costs only the new targets; the sweep skips any
  target already at 15/15.

## 8. Checking on a running sweep

All the 2026 day01 sweeps finished 2026-09-03 06:58 UTC (230/230) and the
multi-horizon week at 22:38 UTC. The recipes below are for the next one.

```bash
# alive?  Bracket the first character so the grep does not match itself.
pgrep -af '[r]un_multi_august2|[p]redict_multi_sweep|[v]isualization_gen'

# how many targets done -- match the raster name exactly.  A bare
# `grep -vc classified` also counts .aux.xml sidecars (gdalinfo -stats leaves
# one) and silently inflates the count.
for m in january february march april may july august; do   # june is elsewhere
  printf "%-10s %s\n" $m $(ls /s3/quinn/prediction_vis/canfirecast_${m}2026 \
    2>/dev/null | grep -c '^2026_[0-9_]*\.tif$')
done
printf "%-10s %s\n" june $(ls /caribou/quinn/wildfire/predictions/\
canfirecast_test_june | grep -c '^2026_[0-9_]*\.tif$')   # 200 + 30 = 230

# multi-horizon: leads per target
for d in /s3/quinn/prediction_vis/canfirecast_2026_multi/by_target/*/; do
  printf "%s %s/15\n" "$(basename $d)" \
    $(ls $d | grep -c '^2026_[0-9_]*\.tif$')
done

# logs -- visualization_gen writes tqdm bars with carriage returns, so a plain
# `tail` returns a ~200 KB blob of batch counters.  Split on \r first.
L=/caribou/quinn/wildfire/logs/predict/multi_august2.log
tr '\r' '\n' < $L | grep -aE '^[0-9]{2}:[0-9]{2}:[0-9]{2} 2026-|^=== ' | tail
tr '\r' '\n' < $L | grep -aE 'Test progress|Model: '        | tail   # per-lead
```

Sweeps are `nohup`'d and survive terminal close, `/exit`, and session clear.
Everything is resumable: `predict_sweep.sh` skips a target that has its `.tif`,
`predict_multi_sweep.sh` resumes at *lead* granularity, so a kill costs only the
lead in flight. Two cautions when stopping one:

* **Never `pkill -f` on the script name** — it matches the shell running the
  `pkill` too, killing your own session mid-sequence and orphaning the model
  process. Kill by PID, parent first.
* A `kill -9` during a raster write leaves a truncated `.tif`, and the resume
  check only tests *existence*. Verify with `gdalinfo` on every raster of the
  interrupted target before trusting the resume.

## 9. Multi-horizon (day01…day15)

`--forecast-task` selects **checkpoints only** — data loading is always
internal `forecast_task=1` (`visualization_gen.py:3549`). Every lead therefore
reads the *same* window D-10…D-1: no extra ingest, no new provenance surface,
and the `--as-of` contract of §2 is untouched.

**Output is named by the valid date, not the anchor.**
`_prediction_output_date` (`visualization_gen.py:5441`) adds `lead - 1` days in
dense-operational mode, so target D at lead N writes `{D+N-1}.tif`. Within one
target the 15 leads are distinct (D…D+14); **across** targets they overlap
completely — which is the point, and also the hazard: `visualization_gen` skips
any lead whose valid-date file already exists (`:4997`), silently, exit 0. A
flat output directory would drop most of a sweep. Hence one directory per
target.

**One publish per target, not per lead.** `_apply_boundary_pixel_filter` runs in
`__init__` (`:2789`) and the data load / normalizer / stats sit outside the
per-model loop (`:3586`), so publish (55 s), boundary rasterisation and the
0.95 GiB load are paid once however many leads follow. Measured: **8m12s per
lead**, ~2h07m per target for 15, so 7 targets ≈ 15 h. All 230 targets × 15
would be ≈ 440 GPU-hours.

**Layout.**

```
canfirecast_2026_multi/
├── by_target/2026-08-01/       # the write path; collision-free, resumable
│   ├── 2026_08_01.tif …        #   lead01 … lead15, valid dates D…D+14
│   └── manifest.json           #   window, D-11 state, lead→valid, ckpt sha256
├── by_valid/2026-08-15/        # derived symlinks, no duplicated rasters
│   └── lead01.tif … lead15.tif
└── index.csv                   # valid_date,lead,target,window_start/end,path
```

`by_valid/` answers "every map we have for this day"; `index.csv` joins a
prediction to the window that produced it without parsing filenames.

```bash
START=2026-08-01 END=2026-08-07 DEVICE=cuda:0 \
  bash operational/scripts/predict_multi_sweep.sh
python operational/scripts/index_predictions.py
```

A valid date reaches all 15 leads only once 15 consecutive targets exist, so the
first fully-stacked day is START+14. The 08-01…08-07 run peaks at 7.

Only `day01`…`day15` checkpoints exist in `/s3/zhengsen/pths/firempc_15days/`;
there is no day16.
