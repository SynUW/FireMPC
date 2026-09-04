"""End-to-end operational CanFireCast data-preparation runner."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Literal

import h5py
import numpy as np
import typer

# Run directly as `python operational/canfirecast/pipeline.py`: put the application root on
# sys.path so the shared modules import by bare name, as elsewhere in FireMPC.
import sys
from pathlib import Path
OPERATIONAL = Path(__file__).resolve().parents[1]
if str(OPERATIONAL) not in sys.path:
    sys.path.insert(0, str(OPERATIONAL))

from canfirecast import input_registry as registry
from canfirecast import store as monthly_store
import paths, vendorpaths
from canfirecast.state_ledger import DEFAULT_LEDGER, state_path, validate_state
from canfirecast.provision import verify_release
from modis.processing_registry import CHANNELS as MODIS_CHANNELS

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
STAGES = ("preflight", "weather-download", "weather-align-encode", "physical-correction", "cffdrs-derive", "modis-latest", "assemble", "validate", "publish")
DEFAULT_STORAGE = "monthly"


def resolve_target(value: str) -> date:
    return datetime.now(timezone.utc).date() if value.lower() == "today" else date.fromisoformat(value)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run(command: list[str], log: Path, *, cwd: Path = vendorpaths.OPERATIONAL_ROOT) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        handle.write(" ".join(command) + "\n\n"); handle.flush()
        result = subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f"stage exited {result.returncode}; see {log}")


def modis_pipeline_command(
    target: date, *, backend: str, stream: str, modis_policy: str,
    as_of: str | None, overwrite: list[str],
) -> list[str]:
    """The `modis pipeline` invocation for one target.

    A backfill and an operational run differ here and nowhere else: `--as-of`
    holds MODIS resolution to the observations that had actually landed by that
    date, so a rebuilt window cannot be shown granules published after it.
    Omitted, the run resolves against whatever is on disk, which is the right
    answer only when the run really is happening on the target date.
    """
    command = [
        sys.executable, str(vendorpaths.CLI), "modis", "pipeline",
        "--target", target.isoformat(), "--backend", backend,
        "--stream", stream, "--modis-policy", modis_policy,
    ]
    if as_of is not None:
        command += ["--as-of", as_of]
    return command + overwrite


def remove_ingested_modis_handoff(h5_path: Path, target: date, store_root: Path) -> int:
    """Remove a target package only after its whole window is in the month store."""
    expected = registry.target_window_dates(target)
    missing = [day.isoformat() for day in expected if not monthly_store.is_written(store_root, day)]
    if missing:
        raise RuntimeError(
            "refusing to remove MODIS handoff; monthly store is missing: "
            + ", ".join(missing)
        )
    package_dir = h5_path.parent
    if h5_path.name != "modis_inputs.h5" or package_dir.name != target.isoformat():
        raise RuntimeError(f"refusing to remove unexpected MODIS handoff path: {h5_path}")
    reclaimed = sum(path.stat().st_size for path in package_dir.rglob("*") if path.is_file())
    shutil.rmtree(package_dir)
    return reclaimed


@app.command()
def main(
    target: Annotated[str, typer.Option("--target")] = "today",
    release: Annotated[str, typer.Option("--release")] = "operational-v1",
    modis_policy: Annotated[Literal["latest-available", "strict"], typer.Option("--modis-policy")] = "latest-available",
    modis_backend: Annotated[Literal["ea", "gee"], typer.Option("--modis-backend")] = "ea",
    modis_stream: Annotated[Literal["standard", "nrt"], typer.Option("--modis-stream")] = "standard",
    storage: Annotated[Literal["monthly", "daily"], typer.Option(
        "--storage", help="Deduplicated monthly store (default) or legacy per-target package.",
    )] = DEFAULT_STORAGE,
    from_stage: Annotated[str, typer.Option("--from-stage")] = "preflight",
    to_stage: Annotated[str | None, typer.Option("--to-stage")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    cleanup_modis: Annotated[bool, typer.Option(
        "--cleanup-modis/--keep-modis",
        help="After monthly ingestion, remove the redundant per-target MODIS handoff.",
    )] = True,
    state_root: Annotated[Path, typer.Option("--state-root")] = DEFAULT_LEDGER,
    as_of: Annotated[str | None, typer.Option(
        "--as-of",
        help="Resolve MODIS only against observations that had landed by this "
             "date (YYYY-MM-DD). A hindcast target D wants --as-of D; an "
             "operational run wants the default, which is what is on disk.",
    )] = None,
) -> None:
    target_day = resolve_target(target)
    if as_of is not None:
        try:
            date.fromisoformat(as_of)
        except ValueError:
            raise typer.BadParameter("--as-of must be YYYY-MM-DD") from None
    end_stage = to_stage or STAGES[-1]
    if from_stage not in STAGES or end_stage not in STAGES: raise typer.BadParameter(f"stages must be one of {STAGES}")
    lo, hi = STAGES.index(from_stage), STAGES.index(end_stage)
    if hi < lo: raise typer.BadParameter("--to-stage precedes --from-stage")
    selected = STAGES[lo:hi + 1]
    release_dir = Path(paths.CANFIRECAST_ASSETS) / release
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = Path(paths.CANFIRECAST_PROCESSING_REPORTS) / "pipeline" / run_id
    work = Path(paths.PROCESSED_MODEL_WEATHER) / target_day.isoformat()
    package_dir = Path(paths.CANFIRECAST_PACKAGES) / target_day.isoformat()
    raw_window = work / "encoded_raw55.h5"; base = work / "base_corrected55.h5"; corrected = work / "corrected55.h5"
    package = package_dir / "canfirecast_inputs.h5"
    store_root = Path(paths.CANFIRECAST_PACKAGES)
    daily_package = Path(paths.CANFIRECAST_DAILY_PACKAGE)
    month_files = sorted({
        monthly_store.month_path(store_root, day)
        for day in registry.target_window_dates(target_day)
    })
    state = state_path(state_root, registry.state_date(target_day))
    record = {
        "schema_version": 1,
        "storage": storage,
        "package_schema_version": 1,
        "run_id": run_id,
        "target_date": target_day.isoformat(),
        "release": release,
        "modis_policy": modis_policy,
        "modis_as_of": as_of,
        "channel_registry_version": registry.REGISTRY_VERSION,
        "stages": {},
        "status": "running",
        "git": vendorpaths.git_provenance(),
        "imported_source": vendorpaths.imported_source_provenance(),
    }
    try:
        manifest = None
        if "preflight" in selected:
            manifest = verify_release(release_dir)
            grid = release_dir / manifest["artifacts"]["grid"]["path"]
            validate_state(state, expected_date=registry.state_date(target_day), template_path=grid)
            registry.verify_vendored_code()
            record["stages"]["preflight"] = "complete"
        else:
            manifest = verify_release(release_dir); grid = release_dir / manifest["artifacts"]["grid"]["path"]
        record["release_artifacts"] = {
            name: item["sha256"] for name, item in manifest["artifacts"].items()
        }
        provenance = registry.provenance_files()
        record["correction_dependencies"] = {
            relative: provenance[relative]["sha256"]
            for relative in (
                *(f"canfirecast/frozen_repo/{name}" for name in registry.VENDORED_CODE),
                *(f"canfirecast/artifacts/{name}" for name in registry.ARTIFACTS),
            )
        }
        adapter_artifact = release_dir / manifest["artifacts"]["ifs_to_era5_physical_adapter.json"]["path"]
        correction_artifact = release_dir / manifest["artifacts"]["cffdrs_aware_correction.json"]["path"]
        registry.verify_file_hash(adapter_artifact, registry.ARTIFACTS["ifs_to_era5_physical_adapter.json"]["sha256"], label="release adapter artifact")
        registry.verify_file_hash(correction_artifact, registry.ARTIFACTS["cffdrs_aware_correction.json"]["sha256"], label="release CFFDRS artifact")
        record["correction_artifact_paths"] = {
            "ifs_to_era5_physical_adapter.json": str(adapter_artifact),
            "cffdrs_aware_correction.json": str(correction_artifact),
        }
        record["d11_state_sha256"] = registry.sha256(state)
        record["d11_state_date"] = registry.state_date(target_day).isoformat()
        with h5py.File(state, "r") as state_h5:
            record["d11_state_lineage"] = {
                "source": str(state_h5.attrs.get("source", state_h5.attrs.get("evidence", "unknown"))),
                "certified": bool(state_h5.attrs.get("certified", False)),
            }
        common_overwrite = ["--overwrite"] if overwrite else []
        if "weather-download" in selected:
            weather_ready = all(
                (Path(paths.DRIVERS) / "IFS" / f"IFS_{day:%Y_%m_%d}_tile{tile:02d}.tif").is_file()
                for day in registry.target_window_dates(target_day) for tile in range(4)
            )
            if resume and weather_ready and not overwrite:
                record["stages"]["weather-download"] = "reused"
            else:
                run([sys.executable, str(vendorpaths.CLI), "weather", "download", "--source", "ifs", "--start", (target_day-timedelta(days=10)).isoformat(), "--end", (target_day-timedelta(days=1)).isoformat()] + common_overwrite, report_dir / "weather-download.log")
                record["stages"]["weather-download"] = "complete"
        if "weather-align-encode" in selected:
            if resume and raw_window.is_file() and not overwrite:
                record["stages"]["weather-align-encode"] = "reused"
            else:
                run([sys.executable, str(vendorpaths.CLI), "canfirecast", "weather-window", "--target", target_day.isoformat(), "--output", str(raw_window), "--grid-template", str(grid)] + common_overwrite, report_dir / "weather-align-encode.log")
                record["stages"]["weather-align-encode"] = "complete"
        if "physical-correction" in selected:
            if resume and base.is_file() and not overwrite:
                record["stages"]["physical-correction"] = "reused"
            else:
                run([sys.executable, str(vendorpaths.CLI), "canfirecast", "model", "correct", str(raw_window), str(base), "--target-date", target_day.isoformat(), "--adapter-artifact", str(adapter_artifact)] + common_overwrite, report_dir / "physical-correction.log")
                record["stages"]["physical-correction"] = "complete"
        if "cffdrs-derive" in selected:
            if storage == "monthly":
                # Channels 19..25 depend on target D's certified D-11 state. They
                # are derived when a window is read, not duplicated in the store.
                record["stages"]["cffdrs-derive"] = "deferred_to_store_read"
            elif resume and corrected.is_file() and not overwrite:
                record["stages"]["cffdrs-derive"] = "reused"
            else:
                run([sys.executable, str(vendorpaths.CLI), "canfirecast", "model", "derive", "--input", str(base), "--state", str(state), "--output", str(corrected), "--target-date", target_day.isoformat(), "--correction-artifact", str(correction_artifact)] + common_overwrite, report_dir / "cffdrs-derive.log")
                record["stages"]["cffdrs-derive"] = "complete"
        modis = Path(paths.MODIS_PACKAGES) / target_day.isoformat() / "modis_inputs.h5"
        if "modis-latest" in selected:
            if resume and modis.is_file() and not overwrite:
                record["stages"]["modis-latest"] = "reused"
            else:
                run(modis_pipeline_command(target_day, backend=modis_backend, stream=modis_stream, modis_policy=modis_policy, as_of=as_of, overwrite=common_overwrite), report_dir / "modis-latest.log")
                record["stages"]["modis-latest"] = "complete"
        statics = release_dir / manifest["artifacts"]["statics"]["path"]
        if "assemble" in selected:
            if storage == "monthly":
                written = monthly_store.ingest_window(
                    store_root, target=target_day, base_h5=base, modis_h5=modis,
                    force=overwrite,
                )
                record["stages"]["assemble"] = "complete"
                record["stored_dates"] = [day.isoformat() for day in written]
            elif resume and package.is_file() and not overwrite:
                record["stages"]["assemble"] = "reused"
            else:
                run([sys.executable, str(vendorpaths.CLI), "canfirecast", "package", "--corrected-weather", str(corrected), "--static", str(statics), "--modis", str(modis), "--output", str(package), "--target-date", target_day.isoformat()] + common_overwrite, report_dir / "assemble.log")
                record["stages"]["assemble"] = "complete"
        # The store keeps every date; the model reads one ten-day window. Read
        # that window once here and let `validate` and `publish` share it --
        # reconstructing a national window twice costs minutes.
        window = None
        if storage == "monthly" and {"validate", "publish"} & set(selected):
            window = monthly_store.read_window(
                store_root, target_day, static_h5=statics, state_h5=state,
                correction_artifact=correction_artifact,
            )
        if "validate" in selected:
            if storage == "monthly":
                monthly_store.validate_window(*window, target_day)
            else:
                from canfirecast.package import validate_package
                validate_package(package)
            record["stages"]["validate"] = "complete"
        if "publish" in selected:
            # One path, rewritten every day by atomic replace: the model is
            # pointed at it once and never has to work out today's date.
            from canfirecast.package import validate_package
            if storage == "monthly":
                monthly_store.write_window_package(daily_package, *window)
            else:
                daily_package.parent.mkdir(parents=True, exist_ok=True)
                temporary = daily_package.with_suffix(daily_package.suffix + ".tmp")
                shutil.copyfile(package, temporary)
                os.replace(temporary, daily_package)
            validate_package(daily_package)
            record["stages"]["publish"] = "complete"
            record["daily_package"] = str(daily_package)
        if corrected.is_file():
            with h5py.File(corrected, "r") as h5:
                record["weather"] = {
                    "source_cycle": str(h5.attrs.get("source_weather_cycle", "unknown")),
                    "dates": [value.decode() if isinstance(value, bytes) else str(value) for value in h5["dates"][()]],
                }
        if modis.is_file():
            with h5py.File(modis, "r") as h5:
                ages = h5["fill_age_days"][()]
                if "source_dates" in h5:
                    record["modis_source_dates"] = [
                        [value.decode() if isinstance(value, bytes) else str(value) for value in row]
                        for row in h5["source_dates"][()]
                    ]
                record["modis_age_days"] = {
                    product: {
                        "maximum": int(np.max(ages[:, [c.index for c in MODIS_CHANNELS if c.product == product]])),
                        "median": float(np.median(ages[:, [c.index for c in MODIS_CHANNELS if c.product == product]])),
                    }
                    for product in sorted({c.product for c in MODIS_CHANNELS})
                }
                record["modis_fill_policy_version"] = h5.attrs.get("fill_policy_version", "unknown")
        if storage == "monthly" and "assemble" in selected and cleanup_modis:
            reclaimed = remove_ingested_modis_handoff(modis, target_day, store_root)
            record["modis_handoff_cleanup"] = {
                "path": str(modis.parent), "reclaimed_bytes": reclaimed,
            }
        artifacts = (
            {"monthly_stores": [str(path) for path in month_files]}
            if storage == "monthly" else {"package": str(package)}
        )
        if "publish" in selected:
            artifacts["daily_package"] = str(daily_package)
        record.update({"status": "complete", "artifacts": artifacts, "d11_state": str(state)})
    except Exception as exc:
        record.update({"status": "failed", "error": str(exc)}); raise
    finally:
        record["ended_utc"] = datetime.now(timezone.utc).isoformat(); atomic_json(report_dir / "manifest.json", record)
    if storage == "monthly":
        for path in month_files:
            typer.echo(path)
    else:
        typer.echo(package)
    if "publish" in selected:
        typer.echo(daily_package)


if __name__ == "__main__": app()
