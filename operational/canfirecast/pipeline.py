"""End-to-end operational CanFireCast data-preparation runner."""

from __future__ import annotations

import json
import os
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
STAGES = ("preflight", "weather-download", "weather-align-encode", "physical-correction", "cffdrs-derive", "modis-latest", "assemble", "validate")
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
    state_root: Annotated[Path, typer.Option("--state-root")] = DEFAULT_LEDGER,
) -> None:
    target_day = resolve_target(target)
    end_stage = to_stage or "validate"
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
                run([sys.executable, str(vendorpaths.CLI), "modis", "pipeline", "--target", target_day.isoformat(), "--backend", modis_backend, "--stream", modis_stream, "--modis-policy", modis_policy] + common_overwrite, report_dir / "modis-latest.log")
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
        if "validate" in selected:
            if storage == "monthly":
                data, validity, store_provenance = monthly_store.read_window(
                    store_root, target_day, static_h5=statics, state_h5=state,
                    correction_artifact=correction_artifact,
                )
                monthly_store.validate_window(
                    data, validity, store_provenance, target_day)
            else:
                from canfirecast.package import validate_package
                validate_package(package)
            record["stages"]["validate"] = "complete"
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
        artifacts = (
            {"monthly_stores": [str(path) for path in month_files]}
            if storage == "monthly" else {"package": str(package)}
        )
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


if __name__ == "__main__": app()
