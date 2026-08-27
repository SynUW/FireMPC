"""Top-level operational MODIS orchestration.

Each child stage remains directly runnable; this command only connects their
validated handoffs for one target date.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Literal

import typer

# Run directly as `python operational/modis/pipeline.py`: put the application root on
# sys.path so the shared modules import by bare name, as elsewhere in FireMPC.
import sys
from pathlib import Path
OPERATIONAL = Path(__file__).resolve().parents[1]
if str(OPERATIONAL) not in sys.path:
    sys.path.insert(0, str(OPERATIONAL))

import paths, vendorpaths
from modis.processing_registry import require_operational_lineage
from modis.registry import OPERATIONAL_EXPORTS, PRODUCTS

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
STAGES = ("download", "preprocess", "package")


def operational_collections() -> tuple[str, ...]:
    return tuple(dict.fromkeys(PRODUCTS[name]["earthaccess"]["short_name"] for name in OPERATIONAL_EXPORTS))


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


def run_command(command: list[str], log: Path, *, tolerate: tuple[int, ...] = ()) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(command) + "\n\n")
        handle.flush()
        completed = subprocess.run(command, cwd=vendorpaths.REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if completed.returncode and completed.returncode not in tolerate:
        raise RuntimeError(f"stage exited {completed.returncode}; see {log}")
    return completed.returncode


@app.command()
def main(
    target: Annotated[str, typer.Option("--target", help="Prediction date D (YYYY-MM-DD).")],
    backend: Annotated[Literal["ea", "gee"], typer.Option("--backend")] = "ea",
    stream: Annotated[Literal["standard", "nrt"], typer.Option("--stream")] = "standard",
    from_stage: Annotated[str, typer.Option("--from-stage")] = "download",
    to_stage: Annotated[str, typer.Option("--to-stage")] = "package",
    resume: Annotated[bool, typer.Option("--resume")] = True,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    tiling_manifest: Annotated[Path | None, typer.Option("--tiling-manifest")] = None,
    modis_policy: Annotated[Literal["strict", "latest-available"], typer.Option("--modis-policy")] = "strict",
    lookback_days: Annotated[int, typer.Option("--lookback-days", min=10)] = 45,
    fill_policy: Annotated[Path | None, typer.Option("--fill-policy")] = None,
) -> None:
    require_operational_lineage()
    if from_stage not in STAGES or to_stage not in STAGES:
        raise typer.BadParameter(f"stages must be one of {STAGES}")
    lo, hi = STAGES.index(from_stage), STAGES.index(to_stage)
    if hi < lo:
        raise typer.BadParameter("--to-stage precedes --from-stage")
    target_day = date.fromisoformat(target)
    if fill_policy is None:
        fill_policy = vendorpaths.CONFIG / (
            "modis_latest_available_policy.json"
            if modis_policy == "latest-available" else "modis_fill_policy.json"
        )
    start = target_day - timedelta(days=lookback_days if modis_policy == "latest-available" else 10)
    end = target_day - timedelta(days=1)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = Path(paths.MODIS_PROCESSING_REPORTS) / "pipeline" / run_id
    record = {
        "schema_version": 1, "run_id": run_id, "target_date": target,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "backend": backend, "stream": stream, "modis_policy": modis_policy,
        "lookback_days": (lookback_days if modis_policy == "latest-available" else 10),
        "stages": {},
    }
    try:
        if "download" in STAGES[lo:hi + 1]:
            command = [
                sys.executable, str(vendorpaths.CLI), "modis", "download", backend,
                "--start", start.isoformat(), "--end", end.isoformat(),
                "--stream", stream, "--products", ",".join(operational_collections()),
            ]
            if overwrite:
                # The downloader is already idempotent and exposes no global
                # overwrite switch; tiling/preprocessing own replacement.
                pass
            # The downloader separates "the manifest records errors" (1) from
            # "everything landed, but a product is later than its measured p95"
            # (2). Only the first is a broken window; lateness is what the
            # latest-available fill policy exists to absorb, so it must not stop
            # the run -- it is recorded instead.
            code = run_command(command, report_dir / "download.log", tolerate=(2,))
            record["stages"]["download"] = "complete with latency alerts" if code == 2 else "complete"
            tiling_manifest = Path(paths.REPORTS) / "tiling" / "latest.json"
        if "preprocess" in STAGES[lo:hi + 1]:
            if tiling_manifest is None:
                tiling_manifest = Path(paths.REPORTS) / "tiling" / "latest.json"
            command = [
                sys.executable, str(vendorpaths.CLI), "modis", "preprocess",
                "--product", "all", "--start", start.isoformat(), "--end", end.isoformat(),
                "--stage", "all", "--tiling-manifest", str(tiling_manifest),
            ]
            if resume and not overwrite:
                command.append("--resume")
            if overwrite:
                command.append("--overwrite")
            run_command(command, report_dir / "preprocess.log")
            record["stages"]["preprocess"] = "complete"
        if "package" in STAGES[lo:hi + 1]:
            command = [
                sys.executable, str(vendorpaths.CLI), "modis", "package",
                "--target", target, "--fill-policy", str(fill_policy),
            ]
            if overwrite:
                command.append("--overwrite")
            run_command(command, report_dir / "package.log")
            record["stages"]["package"] = "complete"
        record["status"] = "complete"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
        raise
    finally:
        record["ended_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        _atomic_json(report_dir / "manifest.json", record)
    typer.echo(Path(paths.MODIS_PACKAGES) / target)


if __name__ == "__main__":
    app()
