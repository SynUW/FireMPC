# Operator notes

What building the 2026 store taught. Everything here was learned by doing; the
things that cost the most time are in §3.

## 1. The two contracts that govern correctness

**as-of.** MODIS is frozen at first write, so whichever target first touches a
date fixes its provenance forever. Date `X`'s correct writer is target `X+1`
(`--as-of X+1`), which sees only granules landed by then. Any wider gap hands
that date hindsight it would not have had operationally. `fill` enforces this
by construction — the range you give is dates, and it runs `start+1 .. end+1`
one day at a time.

*Cold-start seam.* The first target of any fill freezes all ten of its window
dates at once, so they carry up to 10 days of hindsight. In the 2026 store,
**2026-01-01 … 01-10 are permanently affected** — no earlier target can exist.
Repairing a seam is §4.

**Channels 19–25 are target-dependent, not date-dependent.** They come from a
CFFDRS recursion reseeded per target from that target's own `D-11` state.
Measured: the same date read through two different targets differs by up to
**154/255** on channel 22, with 45–79% of pixels changing. Consequences:

* A dense multi-date H5 **cannot** represent them. Do not build one.
* `cffdrs_derive.py:46` hard-rejects anything that isn't `(10,55,H,W)` anyway.
* Every target needs its own published window. Publish → use → delete, so peak
  disk stays at one ~300 MB file rather than one per target.

## 2. Bounds of a store

They are structural, not arbitrary. A target is buildable once the ledger
covers its `D-11`, and readable once the store holds `D-10 .. D-1`. For the
2026 store — ledger seeded 2025-12-31, drivers ending 08-27 — that made
**2026-01-11 … 08-28** the publishable targets over 239 stored dates (~8.9 GB).

## 3. Traps

1. **`canfirecast pipeline` re-downloads MODIS per target.**
   `modis_pipeline_command` (`canfirecast/pipeline.py`) never passes
   `--from-stage`, so every target runs download+preprocess over a 45-day
   lookback. Over a range that turns a 14 h job into ~4 days. This is why
   `fill` hoists the MODIS ingest out of the per-target loop and each target
   then runs only `modis package --as-of D` (~85 s/target).
2. **The handoff leaks if you drive the stages by hand.** `store ingest` called
   directly skips `remove_ingested_modis_handoff`, which lives in the pipeline's
   `assemble` stage. Each target then leaves ~123 MB in `packages/modis/`; 178
   dirs / 22 GB accumulated before this was caught. `fill` cleans per target —
   before the target too, since a stale handoff makes `modis package` refuse
   rather than rebuild.
3. **Don't `validate` inside a range fill.** `validate` re-reads the whole
   national window (~4 min/target), doubling the sweep. `fill` stops at
   `assemble`; every date gets validated later anyway, when it is published.
4. **Plain `--force` corrupts neighbours.** `ingest_window --force` rewrites the
   *entire* 10-day window, not just the date you meant. To fix specific dates,
   mark them incomplete and re-run without force — §4.
5. **Root filesystem fills.** `/` (1.7 TB NVMe) hit 100% mid-run once and killed
   a sweep with ENOSPC. Keep logs and `TMPDIR` on `/caribou` (separate 7 TB
   device). `fill --min-free-gb` guards the data root, not `/`.
6. **`set -u` breaks conda.** Conda's own activate/deactivate hooks reference
   unset variables. Use `set -e` only in wrappers that activate environments.

## 4. Repairing a seam

`--force` is wrong (trap 4). Instead:

```python
# 1. mark the affected dates incomplete
with h5py.File(store.month_path(root, day), "a") as h:
    h["complete"][store.day_index(day)] = False
```

```bash
# 2. re-fill those dates. Ingest skips still-frozen neighbours, so each target
#    writes only its own D-1.
python operational/cli.py fill --start <first bad date> --end <last bad date>
```

Verify afterwards that each date's `written_for_target == date + 1`. Anything
downstream that consumed a window overlapping the repaired dates
(`D ∈ [first+1, last+10]`) has to be recomputed.

## 5. Facts worth not re-deriving

* **MODIS latency is seasonal, not broken.** MCD09GA domain completion: January
  p50 1.0 d (96.8% ≤1 d) vs July 2.0 d (0% ≤1 d) — fewer daylight tiles in
  winter complete the domain sooner. So January legitimately shows ~80% same-day
  channel-days where July shows 6%. This is not contamination.
* **Aqua outage 2026-03-17 … 05-10** — `MYD09GA` has no granules. Handled
  correctly: a single complete platform group suffices, so those dates are
  genuine same-day Terra-only observations, not stale fill. April windows are
  entirely inside it; May's first ten days partly so.
* **Fill in July:** 94.3% of channel-days resolve to an earlier raster, typical
  age 1–3 d (MCD15A3H ~7.5 d, MCD12Q1 ~927 d by design). Expected under
  `--as-of`.
* **Cold-start contamination had no measurable effect on prediction level.**
  Targets with contaminated windows were indistinguishable from clean ones once
  seasonal trend was accounted for.
* **The pipeline is deterministic.** Re-running a target under a different (but
  equivalent) environment reproduced a bit-identical downstream raster.
* **No 2026 fire labels.** `drivers/MOD14A1`/`MYD14A1` hold 2025 files only —
  `OPERATIONAL_EXPORTS` deliberately omits fire labels, so nothing here supports
  measured skill. Downloading them is independent work.

## 6. Watching a long fill

```bash
# alive?  Bracket the first character so the grep does not match itself.
pgrep -af '[c]li.py fill'

# what the store holds
python operational/cli.py canfirecast store status

# per-target logs
ls -t $WILDFIRE_DATA_ROOT/reports/canfirecast_processing/fill/ | head -1
```

Run long fills under `nohup`; they survive terminal close. Resume is free — a
frozen date is skipped — so a kill costs at most the target in flight. When
stopping one, never `pkill -f` on the command line: it matches the shell running
the `pkill` too. Kill by PID, parent first.
