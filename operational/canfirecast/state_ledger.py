"""Certified encoded FFMC/DMC/DC state ledger for CanFireCast.

The ledger intentionally has no cold-start command. States enter through
``install`` with explicit certification evidence, preventing the legacy
operational FWI checkpoint from being mistaken for the model's D-11 state.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import subprocess
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import rasterio
import typer

# Run directly as `python operational/canfirecast/state_ledger.py`: put the application root on
# sys.path so the shared modules import by bare name, as elsewhere in FireMPC.
import sys
from pathlib import Path
OPERATIONAL = Path(__file__).resolve().parents[1]
if str(OPERATIONAL) not in sys.path:
    sys.path.insert(0, str(OPERATIONAL))

from canfirecast import input_registry as registry
import paths, vendorpaths

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
DEFAULT_LEDGER = Path(paths.PROCESSED_MODEL_WEATHER) / "state_ledger"
vendorpaths.ensure_frozen_repo()
from cffdrs_aware_correction import compute_codes
from full10_ifs_domain_transform import DIRECT, decode, encode


def state_path(root: Path, day: date) -> Path:
    return root / f"{day.isoformat()}.h5"


def validate_state(path: Path, *, expected_date: date | None = None, template_path: Path | None = None) -> None:
    if not path.is_file(): raise RuntimeError(f"missing state: {path}")
    with h5py.File(path, "r") as h5:
        if "state" not in h5 or h5["state"].dtype != np.dtype("uint8") or h5["state"].shape[0] != 3:
            raise RuntimeError(f"{path}: state must be uint8 (3,H,W)")
        if not bool(h5.attrs.get("certified", False)): raise RuntimeError(f"{path}: state is not certified")
        if expected_date and h5.attrs.get("date") != expected_date.isoformat(): raise RuntimeError(f"{path}: wrong state date")
        for key in ("source", "source_sha256", "certification_evidence", "grid_sha256"):
            if not h5.attrs.get(key): raise RuntimeError(f"{path}: missing provenance {key}")
        shape = h5["state"].shape[1:]
    if template_path:
        with rasterio.open(template_path) as template:
            if shape != template.shape: raise RuntimeError(f"{path}: grid shape {shape} != {template.shape}")


def install_state(*, source: Path, day: date, root: Path, template_path: Path, evidence: str, overwrite: bool = False) -> Path:
    if not source.is_file(): raise RuntimeError(f"missing source state: {source}")
    with rasterio.open(template_path) as template:
        grid_hash = registry.sha256(template_path)
        if source.suffix.lower() == ".npz":
            archive = np.load(source)
            key = "state" if "state" in archive else None
            if key is not None: state = archive[key]
            elif all(k in archive for k in ("ffmc", "dmc", "dc")): state = np.stack([archive[k] for k in ("ffmc", "dmc", "dc")])
            else: raise RuntimeError("NPZ must contain state or ffmc/dmc/dc")
            physical_state = np.asarray(archive["physical_state"], dtype="float32") if "physical_state" in archive else None
        else:
            with h5py.File(source, "r") as h5:
                state = h5["state"][()]
                physical_state = h5["physical_state"][()] if "physical_state" in h5 else None
        state = np.asarray(state)
        if state.shape != (3, template.height, template.width): raise RuntimeError(f"state shape {state.shape} does not match national grid")
        if state.dtype != np.uint8: raise RuntimeError("certified state must already use frozen uint8 encoding")
    destination = state_path(root, day)
    if destination.exists() and not overwrite: raise FileExistsError(f"state exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".state-", suffix=".h5", dir=destination.parent); os.close(fd)
    tmp = Path(name)
    try:
        with h5py.File(tmp, "w") as h5:
            h5.create_dataset("state", data=state, compression="gzip", shuffle=True)
            if physical_state is not None:
                if physical_state.shape != state.shape or not np.isfinite(physical_state).all():
                    raise RuntimeError("invalid physical state")
                h5.create_dataset("physical_state", data=physical_state.astype("float32"), compression="gzip", shuffle=True)
            h5.attrs.update({
                "date": day.isoformat(), "certified": True, "source": str(source),
                "source_sha256": registry.sha256(source), "certification_evidence": evidence,
                "grid_sha256": grid_hash, "channel_names": json.dumps(["FFMC", "DMC", "DC"]),
                "encoding_registry": registry.REGISTRY_VERSION,
            })
        os.replace(tmp, destination)
    finally: tmp.unlink(missing_ok=True)
    validate_state(destination, expected_date=day, template_path=template_path)
    return destination


def advance_window(*, weather_h5: Path, root: Path, template_path: Path) -> tuple[Path, ...]:
    """Advance ten ERA5 days from the certified state immediately before them."""
    with h5py.File(weather_h5, "r") as weather:
        if weather.attrs.get("weather_source") != "era5": raise RuntimeError("state ledger may advance only from ERA5-Land weather")
        data = weather["data"][()]
        days = tuple(date.fromisoformat(x) for x in weather["dates"].asstr()[:])
        validity = weather["spatial_validity"][()] if "spatial_validity" in weather else data[:, 1] > 0
    if data.shape[:2] != (10, 55) or len(days) != 10: raise RuntimeError("weather ledger input must be (10,55,H,W)")
    # A final overlapping ten-day window is allowed: skip already-published
    # prefix dates and continue from the first missing descendant.
    start_index = next((i for i, day in enumerate(days) if not state_path(root, day).is_file()), len(days))
    if start_index == len(days): return ()
    previous_day = days[start_index] - __import__("datetime").timedelta(days=1)
    previous_path = state_path(root, previous_day)
    validate_state(previous_path, expected_date=previous_day, template_path=template_path)
    with h5py.File(previous_path, "r") as h5:
        previous_encoded = h5["state"][()]
        previous_physical = h5["physical_state"][()] if "physical_state" in h5 else None
    outputs = []
    for index in range(start_index, len(days)):
        day = days[index]
        valid = validity[index].astype(bool)
        matrix = np.stack([decode(data[index, channel], channel) for channel in DIRECT], axis=-1)
        previous = (
            previous_physical[:, valid].T
            if previous_physical is not None
            else np.column_stack([decode(previous_encoded[channel][valid], 19 + channel) for channel in range(3)])
        )
        codes = compute_codes(previous, matrix[valid], np.full(int(valid.sum()), int(day.strftime("%Y%m%d")), dtype=np.int64))
        state = np.zeros((3,) + valid.shape, dtype="uint8")
        for state_index, channel in enumerate((19, 20, 21)):
            state[state_index, valid] = np.rint(encode(codes[:, state_index], channel)).clip(0, 255).astype("uint8")
        destination = state_path(root, day); destination.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".state-", suffix=".h5", dir=destination.parent); os.close(fd); tmp = Path(name)
        physical_state = np.zeros((3,) + valid.shape, dtype="float32")
        physical_state[:, valid] = codes[:, :3].T
        with h5py.File(tmp, "w") as h5:
            h5.create_dataset("state", data=state, compression="gzip", shuffle=True)
            h5.create_dataset("physical_state", data=physical_state, compression="gzip", shuffle=True)
            h5.attrs.update({"date": day.isoformat(), "certified": True, "source": str(weather_h5), "source_sha256": registry.sha256(weather_h5), "certification_evidence": "frozen equations advanced from certified previous state", "grid_sha256": registry.sha256(template_path), "previous_state_sha256": registry.sha256(previous_path), "encoding_registry": registry.REGISTRY_VERSION})
        os.replace(tmp, destination); outputs.append(destination)
        previous_path, previous_encoded, previous_physical = destination, state, physical_state
    return tuple(outputs)


@app.command("advance-window")
def advance_window_command(weather_h5: Annotated[Path, typer.Option("--weather")], root: Annotated[Path, typer.Option("--root")] = DEFAULT_LEDGER, template: Annotated[Path, typer.Option("--grid-template")] = Path(paths.GRID_TEMPLATE)) -> None:
    for output in advance_window(weather_h5=weather_h5, root=root, template_path=template): typer.echo(output)


@app.command("import-reference-fwi")
def import_reference_fwi(
    source: Annotated[Path, typer.Option("--source")], day: Annotated[str, typer.Option("--date")],
    evidence: Annotated[str, typer.Option("--evidence")], root: Annotated[Path, typer.Option("--root")] = DEFAULT_LEDGER,
    template: Annotated[Path, typer.Option("--grid-template")] = Path(paths.GRID_TEMPLATE),
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Import channels 19-21 from an authoritative national training/reference FWI raster."""
    parsed = date.fromisoformat(day)
    with rasterio.open(template) as grid, rasterio.open(source) as raster:
        if raster.shape != grid.shape or raster.crs != grid.crs or not raster.transform.almost_equals(grid.transform, precision=1e-12):
            raise RuntimeError("reference FWI raster does not match authoritative national grid")
        if raster.count < 3: raise RuntimeError("reference FWI needs FFMC/DMC/DC bands")
        physical = raster.read((1, 2, 3)).astype("float32")
        state = np.zeros(physical.shape, dtype="uint8")
        for index, channel in enumerate((19, 20, 21)):
            valid = np.isfinite(physical[index]) & (physical[index] >= 0)
            state[index, valid] = np.rint(encode(physical[index, valid], channel)).clip(0, 255).astype("uint8")
    seed = root / "seed_sources" / f"{parsed.isoformat()}-reference.npz"
    seed.parent.mkdir(parents=True, exist_ok=True)
    clean_physical = np.where(np.isfinite(physical) & (physical >= 0), physical, 0).astype("float32")
    np.savez(seed, state=state, physical_state=clean_physical, reference_path=str(source), reference_sha256=registry.sha256(source))
    output = install_state(source=seed, day=parsed, root=root, template_path=template, evidence=evidence, overwrite=overwrite)
    typer.echo(output)


@app.command("install")
def install(
    source: Annotated[Path, typer.Option("--source")], day: Annotated[str, typer.Option("--date")],
    evidence: Annotated[str, typer.Option("--evidence")], root: Annotated[Path, typer.Option("--root")] = DEFAULT_LEDGER,
    template: Annotated[Path, typer.Option("--grid-template")] = Path(paths.GRID_TEMPLATE),
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    typer.echo(install_state(source=source, day=date.fromisoformat(day), root=root, template_path=template, evidence=evidence, overwrite=overwrite))


@app.command("verify")
def verify(day: Annotated[str, typer.Option("--date")], root: Annotated[Path, typer.Option("--root")] = DEFAULT_LEDGER, template: Annotated[Path, typer.Option("--grid-template")] = Path(paths.GRID_TEMPLATE)) -> None:
    path = state_path(root, date.fromisoformat(day)); validate_state(path, expected_date=date.fromisoformat(day), template_path=template); typer.echo(path)


# ---------------------------------------------------------------------------
# Advancing the ledger
#
# Was build_canfirecast_state_ledger.py, a separate script whose only command
# was `advance`.  It belongs beside the ledger it advances.
# ---------------------------------------------------------------------------
def latest_day(root: Path) -> date:
    days = []
    for path in root.glob("????-??-??.h5"):
        try: days.append(date.fromisoformat(path.stem))
        except ValueError: pass
    if not days: raise RuntimeError("ledger has no certified seed; import one first")
    return max(days)


def command(args: list[str]) -> None:
    result = subprocess.run(args, cwd=vendorpaths.REPO_ROOT)
    if result.returncode: raise RuntimeError(f"command failed ({result.returncode}): {' '.join(args)}")


@app.command("advance")
def advance(
    through: Annotated[str, typer.Option("--through")],
    root: Annotated[Path, typer.Option("--root")] = DEFAULT_LEDGER,
    template: Annotated[Path, typer.Option("--grid-template")] = Path(paths.GRID_TEMPLATE),
    keep_windows: Annotated[bool, typer.Option("--keep-windows")] = False,
) -> None:
    """Download ERA5-Land and advance the certified state ledger."""
    from canfirecast.weather_window import build_window

    end = date.fromisoformat(through)
    current = latest_day(root)
    validate_state(state_path(root, current), expected_date=current, template_path=template)
    while current < end:
        block_end = min(current + timedelta(days=10), end)
        # --from-hourly: Earth Engine has been seen to skip a day of DAILY_AGGR
        # while the hourly collection behind it is complete, and the recursion
        # below cannot step over a date.  The rebuild verifies itself against a
        # served neighbouring day before writing, so an unexplained change in
        # the aggregation fails the advance rather than entering the ledger.
        command([sys.executable, str(vendorpaths.CLI), "weather", "download", "--source", "era5", "--from-hourly", "--start", (current + timedelta(days=1)).isoformat(), "--end", block_end.isoformat()])
        window = Path(paths.PROCESSED_MODEL_WEATHER) / "ledger_windows" / f"through-{block_end.isoformat()}.h5"
        # Target block_end+1 yields the ten days ending at block_end. The last
        # short block overlaps certified prefix dates; advance_window skips it.
        build_window(target=block_end + timedelta(days=1), drivers=Path(paths.DRIVERS), template_path=template, output=window, overwrite=True, source_key="era5")
        outputs = advance_window(weather_h5=window, root=root, template_path=template)
        if not outputs or outputs[-1] != state_path(root, block_end): raise RuntimeError(f"ledger did not reach {block_end}")
        if not keep_windows: window.unlink(missing_ok=True)
        current = block_end
        typer.echo(current.isoformat())


if __name__ == "__main__": app()
