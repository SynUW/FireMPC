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

import json
import os
import subprocess
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

import typer

# Run directly as `python operational/weather/pipeline.py`: put the application root on
# sys.path so the shared modules import by bare name, as elsewhere in FireMPC.
import sys
from pathlib import Path
OPERATIONAL = Path(__file__).resolve().parents[1]
if str(OPERATIONAL) not in sys.path:
    sys.path.insert(0, str(OPERATIONAL))

import paths, vendorpaths
from weather import derive as derive
from weather import registry as registry

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
STAGES = ("download", "vpd", "fwi")


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def run_command(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(command) + "\n\n")
        handle.flush()
        completed = subprocess.run(command, cwd=vendorpaths.REPO_ROOT, stdout=handle,
                                   stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"stage exited {completed.returncode}; see {log}\n"
            + log.read_text(encoding="utf-8", errors="replace")[-2000:]
        )


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
    if from_stage not in STAGES or to_stage not in STAGES:
        raise typer.BadParameter(f"stages must be one of {STAGES}")
    lo, hi = STAGES.index(from_stage), STAGES.index(to_stage)
    if hi < lo:
        raise typer.BadParameter("--to-stage precedes --from-stage")
    target_day = date.fromisoformat(target)
    selected = STAGES[lo:hi + 1]

    checkpoint = checkpoint_day(state_root)
    start, end = window_for(target_day, checkpoint, max_catch_up)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = Path(paths.WEATHER_PROCESSING_REPORTS) / "pipeline" / run_id
    record = {
        "schema_version": 1, "run_id": run_id, "target_date": target,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "fwi_checkpoint_before": checkpoint.isoformat() if checkpoint else None,
        "source": source, "registry_version": registry.REGISTRY_VERSION,
        "stages": {},
    }
    try:
        if "download" in selected:
            run_command([
                sys.executable, str(vendorpaths.CLI), "weather", "download",
                "--source", source, "--start", start.isoformat(), "--end", end.isoformat(),
            ] + (["--overwrite"] if overwrite else []), report_dir / "download.log")
            record["stages"]["download"] = "complete"

        for stage, product in (("vpd", "VPD"), ("fwi", "FWI")):
            if stage not in selected:
                continue
            command = [
                sys.executable, str(vendorpaths.CLI), "weather", "preprocess",
                "--product", product, "--start", start.isoformat(), "--end", end.isoformat(),
                "--stage", "all", "--source", source,
            ]
            if resume and not overwrite:
                command.append("--resume")
            if overwrite:
                command.append("--overwrite")
            run_command(command, report_dir / f"{stage}.log")
            record["stages"][stage] = "complete"

        record["fwi_checkpoint_after"] = (
            day.isoformat() if (day := checkpoint_day(state_root)) else None)
        record["status"] = "complete"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
        raise
    finally:
        record["ended_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        _atomic_json(report_dir / "manifest.json", record)

    for product in ("VPD", "FWI"):
        typer.echo(Path(paths.PROCESSED_WEATHER) / "scaled" / registry.ROUTES[product].output_dir)


if __name__ == "__main__":
    app()
