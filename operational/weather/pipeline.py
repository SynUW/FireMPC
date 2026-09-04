"""Top-level operational weather orchestration: download -> VPD -> FWI.

Each child stage remains directly runnable; this command only connects their
validated handoffs for one target date.

    python operational/cli.py weather pipeline --target 2026-08-17
    python operational/cli.py weather pipeline --target 2026-08-17 --from-stage vpd

THE WINDOW IS DECIDED BY THE FWI CHECKPOINT, NOT BY THE TARGET
  FWI is recursive: a skipped day is not a hole in one output, it is a
  permanent offset in every later one.  So the run covers the checkpoint's next
  day through the target, not the target alone, and a catch-up longer than
  ``--max-catch-up`` stops the run instead of quietly grinding through months of
  backfill inside what was meant to be a daily job.  That is the gap being
  surfaced rather than swallowed -- `weather_preprocess` would refuse it anyway,
  but by then the download would already have run.

  A checkpoint ahead of the target is also an error here: it means the target
  has already been folded into the recursion and re-running it would
  double-count the day.

The 25-channel weather package is not built: 18 of its channels are the ERA5
model slot, which waits on the five-channel policy (see
`weather_registry.MISSING_ERA5_BANDS`).  This pipeline ends at `scaled`.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

import typer

# Run directly as `python operational/weather/pipeline.py`: put the application root on
# sys.path so the shared modules import by bare name, as elsewhere in FireMPC.
OPERATIONAL = Path(__file__).resolve().parents[1]
if str(OPERATIONAL) not in sys.path:
    sys.path.insert(0, str(OPERATIONAL))

import ops, paths
from weather import derive as derive
from weather import registry as registry

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
STAGES = ("download", "vpd", "fwi")


def checkpoint_day(state_root: Path) -> date | None:
    path = state_root / "fwi_state.npz"
    return derive.FWIState.load(path).day if path.is_file() else None


def window_for(target: date, checkpoint: date | None, max_catch_up: int) -> tuple[date, date]:
    """The date range this run must cover to keep the recursion unbroken."""
    if checkpoint is None:
        return target, target
    if checkpoint >= target:
        raise RuntimeError(
            f"FWI checkpoint already stands at {checkpoint}, on or after the target {target}; "
            "the recursion only moves forward"
        )
    start = checkpoint + timedelta(days=1)
    behind = (target - start).days + 1
    if behind > max_catch_up:
        raise RuntimeError(
            f"FWI checkpoint stands at {checkpoint}, {behind} days behind {target}. "
            f"That is more than --max-catch-up ({max_catch_up}): run the backfill "
            f"deliberately with weather_preprocess.py rather than inside a daily job"
        )
    return start, target


@app.command()
def main(
    target: Annotated[str, typer.Option("--target", help="Prediction date D (YYYY-MM-DD).")],
    source: Annotated[str, typer.Option("--source")] = "auto",
    from_stage: Annotated[str, typer.Option("--from-stage")] = "download",
    to_stage: Annotated[str, typer.Option("--to-stage")] = "fwi",
    state_root: Annotated[Path, typer.Option("--state-root")] = Path(paths.FWI_STATE),
    max_catch_up: Annotated[int, typer.Option(
        "--max-catch-up", help="Refuse a daily run that would backfill more days than this.",
    )] = 14,
    resume: Annotated[bool, typer.Option("--resume")] = True,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    target_day = date.fromisoformat(target)
    selected = ops.select_stages(STAGES, from_stage, to_stage)

    checkpoint = checkpoint_day(state_root)
    start, end = window_for(target_day, checkpoint, max_catch_up)

    run_id = ops.run_id()
    report_dir = Path(paths.WEATHER_PROCESSING_REPORTS) / "pipeline" / run_id
    with ops.record(report_dir, {
        "schema_version": 1, "run_id": run_id, "target_date": target,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "fwi_checkpoint_before": checkpoint.isoformat() if checkpoint else None,
        "source": source, "registry_version": registry.REGISTRY_VERSION,
        "stages": {},
    }) as record:
        if "download" in selected:
            ops.run(ops.cli(
                "weather", "download",
                "--source", source, "--start", start.isoformat(), "--end", end.isoformat(),
            ) + (["--overwrite"] if overwrite else []), report_dir / "download.log")
            record["stages"]["download"] = "complete"

        for stage, product in (("vpd", "VPD"), ("fwi", "FWI")):
            if stage not in selected:
                continue
            command = ops.cli(
                "weather", "preprocess",
                "--product", product, "--start", start.isoformat(), "--end", end.isoformat(),
                "--stage", "all", "--source", source,
            )
            if resume and not overwrite:
                command.append("--resume")
            if overwrite:
                command.append("--overwrite")
            ops.run(command, report_dir / f"{stage}.log")
            record["stages"][stage] = "complete"

        record["fwi_checkpoint_after"] = (
            day.isoformat() if (day := checkpoint_day(state_root)) else None)

    for product in ("VPD", "FWI"):
        typer.echo(Path(paths.PROCESSED_WEATHER) / "scaled" / registry.ROUTES[product].output_dir)


if __name__ == "__main__":
    app()
