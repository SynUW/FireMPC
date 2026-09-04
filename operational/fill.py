"""Fill a range of dates into the monthly CanFireCast store.

This is what the application is for.  Everything else here is a stage it
drives.  Given a range of *store dates* it makes each of them present in
`packages/canfirecast/YYYY-MM.h5`, using the operational sources -- IFS from
Earth Engine and MODIS through earthaccess -- so a date is built from what
could actually have been fetched at the time.

Three things about the shape of the run are contracts, not choices:

* **A date is written by the target after it.**  A target `D` reads the window
  `D-10..D-1`, and MODIS is frozen at first write, so the run that first
  touches a date fixes how much hindsight it carries.  Date `X` therefore has
  exactly one correct writer -- target `X+1`, resolved `--as-of X+1` -- and the
  sweep steps one day at a time.  A wider step would hand a window's older
  dates extra granule arrivals and make them systematically fresher than the
  rest of the store: a different experiment, not a faster one.  Hence the
  range given here is dates, and the targets run are `start+1 .. end+1`.

* **The certified state ledger comes first.**  Every target's preflight
  demands certified FFMC/DMC/DC state at `D-11`, and that state is a recursion
  over ERA5-Land that cannot skip a day.  The ledger is advanced once, up
  front, through the last target's `D-11`; `state-ledger advance` is a no-op
  when it is already there.

* **MODIS download and preprocess are done once for the whole range.**
  `canfirecast pipeline` drives `modis pipeline` without `--from-stage`, so
  per target it re-runs a 45-day download and preprocess lookback.  Over a
  range that is nearly all of the cost -- ~85 s of real work per target behind
  hours of repetition.  Here the ingest covers `first-45 .. last-1` once, and
  each target then runs only `modis package --as-of D` off the preprocessed
  stages.
"""

from __future__ import annotations

import os
import shutil
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

import typer

import ops
import paths
from canfirecast import state_ledger
from canfirecast import store as monthly_store

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

# The window a target reads is D-10..D-1, so target D newly freezes D-1.
TARGET_OFFSET_DAYS = 1
# The pipeline reads certified state for D-11; see canfirecast.input_registry.
STATE_LAG_DAYS = 11
# `modis package` resolves a channel-day to the newest observation on or before
# it within this lookback, so the MODIS stages must cover it for every target.
MODIS_LOOKBACK_DAYS = 45


def resolve_day(value: str) -> date:
    """`today` and `yesterday` are what an operational invocation actually wants."""
    text = value.strip().lower()
    today = datetime.now(timezone.utc).date()
    if text == "today":
        return today
    if text == "yesterday":
        return today - timedelta(days=1)
    return date.fromisoformat(value)


def run(command: list[str], log: Path) -> None:
    """Run one stage, appending to `log`: a target's stages share one file."""
    ops.run(command, log, append=True)


def free_gb(path: str) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / 1e9


def drop_modis_handoff(target: date) -> int:
    """Remove `packages/modis/<D>/`, and report what that reclaimed.

    Two reasons, both learned the hard way.  Before the target runs, a handoff
    left by some earlier run makes `modis package` refuse rather than rebuild,
    so a stale one would silently substitute a package built under different
    resolution rules.  After `store ingest`, the handoff is redundant with the
    store -- and the cleanup that `canfirecast pipeline`'s assemble stage would
    have done is not reached on this path, so at ~123 MB per target it is the
    difference between a range fill costing nothing and costing tens of GB.
    """
    package_dir = Path(paths.MODIS_PACKAGES) / target.isoformat()
    if not package_dir.is_dir():
        return 0
    reclaimed = sum(path.stat().st_size for path in package_dir.rglob("*") if path.is_file())
    shutil.rmtree(package_dir)
    return reclaimed


@app.command()
def main(
    start: Annotated[str, typer.Option("--start", help="First store date to fill (YYYY-MM-DD).")],
    end: Annotated[str, typer.Option("--end", help="Last store date, inclusive; 'today'/'yesterday' allowed.")],
    release: Annotated[str, typer.Option("--release", help="Immutable asset release to build against.")]
        = os.environ.get("WILDFIRE_ASSET_RELEASE", "operational-v1"),
    modis_ingest: Annotated[bool, typer.Option(
        "--modis-ingest/--no-modis-ingest",
        help="Download and preprocess MODIS for the range first. Skip only when "
             "processed/modis/scaled already covers it.")] = True,
    advance_ledger: Annotated[bool, typer.Option(
        "--advance-ledger/--no-advance-ledger",
        help="Advance the certified D-11 state ledger through the range first.")] = True,
    overwrite: Annotated[bool, typer.Option(
        "--overwrite", help="Rebuild dates that are already frozen in the store.")] = False,
    keep_work: Annotated[bool, typer.Option(
        "--keep-work", help="Keep each target's processed/model_weather scratch.")] = False,
    min_free_gb: Annotated[float, typer.Option(
        "--min-free-gb", help="Abort before a target if the data root is below this.")] = 40.0,
) -> None:
    """Fill every date in [--start, --end] into the monthly store."""
    first, last = resolve_day(start), resolve_day(end)
    if last < first:
        raise typer.BadParameter("--end precedes --start")
    targets = [first + timedelta(days=n + TARGET_OFFSET_DAYS)
               for n in range((last - first).days + 1)]
    store_root = Path(paths.CANFIRECAST_PACKAGES)
    ledger_through = targets[-1] - timedelta(days=STATE_LAG_DAYS)

    run_id = ops.run_id()
    log_dir = Path(paths.CANFIRECAST_PROCESSING_REPORTS) / "fill" / run_id
    with ops.record(log_dir, {
        "schema_version": 1, "run_id": run_id, "release": release,
        "dates": {"start": first.isoformat(), "end": last.isoformat()},
        "targets": {"start": targets[0].isoformat(), "end": targets[-1].isoformat()},
        "ledger_through": ledger_through.isoformat(),
        "started_utc": ops.utc_now(),
        "filled": [], "reused": [], "reclaimed_bytes": 0,
    }, name="fill.json") as record:
        if advance_ledger:
            typer.echo(f"advancing the state ledger through {ledger_through}")
            if state_ledger.latest_day(state_ledger.DEFAULT_LEDGER) < ledger_through:
                run(ops.cli("canfirecast", "state-ledger", "advance",
                            "--through", ledger_through.isoformat()),
                    log_dir / "state-ledger.log")

        if modis_ingest:
            # One pass over the union of every target's lookback, expressed as
            # the last target's lookback since `modis pipeline` takes a single
            # target and counts backwards from it.
            lookback = (targets[-1] - (targets[0] - timedelta(days=MODIS_LOOKBACK_DAYS))).days
            typer.echo(f"MODIS ingest {targets[0] - timedelta(days=MODIS_LOOKBACK_DAYS)}"
                       f" .. {targets[-1] - timedelta(days=1)}")
            run(ops.cli("modis", "pipeline", "--target", targets[-1].isoformat(),
                        "--from-stage", "download", "--to-stage", "preprocess",
                        "--modis-policy", "latest-available",
                        "--lookback-days", str(lookback)),
                log_dir / "modis-ingest.log")

        for target in targets:
            day = target - timedelta(days=TARGET_OFFSET_DAYS)
            if monthly_store.is_written(store_root, day) and not overwrite:
                typer.echo(f"{day} already frozen, skipping")
                record["reused"].append(day.isoformat())
                continue
            available = free_gb(paths.DATA_ROOT)
            if available < min_free_gb:
                raise RuntimeError(
                    f"only {available:.0f} GB free at {paths.DATA_ROOT} "
                    f"(floor {min_free_gb:.0f} GB)")
            typer.echo(f"{day} filling via target {target} ({available:.0f} GB free)")
            target_log = log_dir / f"{target.isoformat()}.log"

            # Weather half: IFS download, alignment, and the physical
            # correction. The store holds pre-recursion float32 weather, so the
            # CFFDRS derivation is deferred to read time and this stops before
            # `modis-latest`, which would re-run the range ingest above.
            run(ops.cli("canfirecast", "pipeline", "--target", target.isoformat(),
                        "--release", release, "--to-stage", "cffdrs-derive")
                + (["--overwrite"] if overwrite else []), target_log)

            # MODIS half: package this target's window out of the preprocessed
            # stages, seeing only what had landed by the target date.
            drop_modis_handoff(target)
            run(ops.cli("modis", "pipeline", "--target", target.isoformat(),
                        "--as-of", target.isoformat(),
                        "--modis-policy", "latest-available",
                        "--from-stage", "package"), target_log)

            run(ops.cli("canfirecast", "store", "ingest", "--target", target.isoformat())
                + (["--force"] if overwrite else []), target_log)
            record["reclaimed_bytes"] += drop_modis_handoff(target)
            if not keep_work:
                shutil.rmtree(Path(paths.PROCESSED_MODEL_WEATHER) / target.isoformat(),
                              ignore_errors=True)
            record["filled"].append(day.isoformat())

    typer.echo(f"{len(record['filled'])} filled, {len(record['reused'])} already present; "
               f"log dir {log_dir}")
    for path in sorted({monthly_store.month_path(store_root, first + timedelta(days=n))
                        for n in range((last - first).days + 1)}):
        typer.echo(path)


if __name__ == "__main__":
    app()
