"""Assemble the native 55-channel CanFireCast package."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import typer

# Run directly as `python operational/canfirecast/package.py`: put the application root on
# sys.path so the shared modules import by bare name, as elsewhere in FireMPC.
import sys
from pathlib import Path
OPERATIONAL = Path(__file__).resolve().parents[1]
if str(OPERATIONAL) not in sys.path:
    sys.path.insert(0, str(OPERATIONAL))

from canfirecast import input_registry as registry
import paths, vendorpaths
from modis.processing_registry import CHANNEL_NAMES as MODIS_PACKAGE_CHANNEL_NAMES

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def code_revision() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=vendorpaths.GIT_ROOT,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty_cmd = ["git", "status", "--porcelain", *vendorpaths.operational_git_pathspec()]
        dirty = subprocess.run(
            dirty_cmd, cwd=vendorpaths.GIT_ROOT,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def _strings(values) -> np.ndarray:
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in values], dtype=object)


def _read_required_names(h5: h5py.File, expected: tuple[str, ...], label: str) -> None:
    if "channel_names" not in h5:
        raise RuntimeError(f"{label}: missing channel_names")
    names = tuple(_strings(h5["channel_names"][()]))
    if names != expected:
        raise RuntimeError(f"{label}: channel order mismatch")


def validate_package(path: Path) -> None:
    with h5py.File(path, "r") as h5:
        if h5.attrs.get("schema_version") != 1:
            raise RuntimeError("unsupported CanFireCast package schema_version")
        if h5.attrs.get("channel_registry_version") != registry.REGISTRY_VERSION:
            raise RuntimeError("unsupported CanFireCast channel registry version")
        if h5["data"].dtype != np.dtype("uint8"):
            raise RuntimeError("data must be uint8")
        if h5["data"].ndim != 4 or h5["data"].shape[0:2] != (10, 55):
            raise RuntimeError(f"data must have shape (10,55,H,W), got {h5['data'].shape}")
        height, width = h5["data"].shape[2:]
        if "spatial_validity" not in h5:
            raise RuntimeError("package is missing spatial_validity")
        validity = h5["spatial_validity"]
        if validity.shape != (10, height, width) or validity.dtype != np.dtype("bool"):
            raise RuntimeError("spatial_validity must be bool (10,H,W)")
        for attribute in ("crs", "transform", "height", "width"):
            if attribute not in h5.attrs:
                raise RuntimeError(f"package is missing geospatial attribute {attribute}")
        if int(h5.attrs["height"]) != height or int(h5.attrs["width"]) != width:
            raise RuntimeError("height/width attributes do not match package data")
        _read_required_names(h5, registry.CHANNEL_NAMES, str(path))
        data = h5["data"]
        if np.any(data[:, registry.FIRE_DETECTION_CHANNEL] != 0):
            raise RuntimeError("Fire Detection channel must be zero in an input package")
        if np.any(data[:, list(registry.ERA5_ONLY_MISSING_CHANNELS)] != 0):
            raise RuntimeError("ERA5-only missing channels must be zero in cache representation")
        dates = tuple(_strings(h5["dates"][()]))
        parsed_dates = tuple(date.fromisoformat(value) for value in dates)
        if len(parsed_dates) != 10 or any(
            later - earlier != timedelta(days=1)
            for earlier, later in zip(parsed_dates, parsed_dates[1:])
        ):
            raise RuntimeError("dates must be ten consecutive ascending ISO dates")
        if "target_date" in h5.attrs:
            expected = tuple(d.isoformat() for d in registry.target_window_dates(date.fromisoformat(h5.attrs["target_date"])))
            if dates != expected:
                raise RuntimeError("dates must be exactly D-10..D-1 for target_date")


def build_package(
    *,
    target: date,
    corrected_weather_h5: Path,
    static_h5: Path,
    modis_h5: Path,
    output_h5: Path,
    overwrite: bool = False,
) -> Path:
    registry.validate_registry()
    registry.verify_vendored_code()
    if output_h5.exists() and not overwrite:
        raise RuntimeError(f"output exists: {output_h5}")
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_h5.with_suffix(output_h5.suffix + ".tmp")
    strings = h5py.string_dtype("utf-8")
    creation = datetime.now(timezone.utc).isoformat()

    with h5py.File(corrected_weather_h5, "r") as weather, h5py.File(static_h5, "r") as static, h5py.File(modis_h5, "r") as modis:
        _read_required_names(weather, registry.CHANNEL_NAMES, "corrected weather")
        _read_required_names(modis, tuple(MODIS_PACKAGE_CHANNEL_NAMES), "MODIS")
        if "channel_names" in static:
            _read_required_names(static, tuple(registry.CHANNEL_NAMES[i] for i in registry.STATIC_CHANNELS), "static")
        w = weather["data"]
        s = static["data"]
        m = modis["data"]
        if w.shape[0:2] != (10, 55):
            raise RuntimeError(f"corrected weather data must be (10,55,H,W), got {w.shape}")
        if s.shape != (12, w.shape[2], w.shape[3]):
            raise RuntimeError(f"static data must be (12,H,W), got {s.shape}")
        if m.shape != (10, 17, w.shape[2], w.shape[3]):
            raise RuntimeError(f"MODIS data must be (10,17,H,W), got {m.shape}")
        dates = tuple(_strings(weather["dates"][()]))
        expected_dates = tuple(d.isoformat() for d in registry.target_window_dates(target))
        if dates != expected_dates:
            raise RuntimeError(f"weather dates mismatch: expected {expected_dates}, got {dates}")

        with h5py.File(tmp, "w") as out:
            data = out.create_dataset(
                "data", shape=w.shape, dtype="u1",
                chunks=(1, 1, min(256, w.shape[2]), min(256, w.shape[3])),
                compression="gzip", compression_opts=1, shuffle=True,
            )
            data[...] = w[...]
            data[:, registry.FIRE_DETECTION_CHANNEL] = 0
            for channel in registry.ERA5_ONLY_MISSING_CHANNELS:
                data[:, channel] = 0
            static_values = s[...]
            for time_index in range(10):
                data[time_index, 26:38] = static_values
            data[:, 38:55] = m[...]
            if "spatial_validity" in weather:
                out.create_dataset("spatial_validity", data=weather["spatial_validity"][()], compression="gzip")
            out.create_dataset("dates", data=np.asarray(dates, dtype=object), dtype=strings)
            out.create_dataset("channel_names", data=np.asarray(registry.CHANNEL_NAMES, dtype=object), dtype=strings)
            for optional in ("source_dates", "fill_age_days"):
                if optional in modis:
                    values = modis[optional][()]
                    if values.dtype.kind in ("O", "S", "U"):
                        out.create_dataset(f"modis_{optional}", data=_strings(values), dtype=strings)
                    else:
                        out.create_dataset(f"modis_{optional}", data=values)
            out.attrs.update({
                "schema_version": 1,
                "target_date": target.isoformat(),
                "channel_registry_version": registry.REGISTRY_VERSION,
                "dtype": registry.PACKAGE_DTYPE,
                "encoded_nodata": "none",
                "creation_timestamp": creation,
                "code_revision": code_revision(),
                "weather_source": str(corrected_weather_h5),
                "static_source": str(static_h5),
                "modis_source": str(modis_h5),
                "correction_artifact_hashes": json.dumps({k: v["sha256"] for k, v in registry.ARTIFACTS.items()}, sort_keys=True),
            })
            for key in ("crs", "transform", "height", "width", "d11_state_date", "d11_state_source", "d11_state_sha256", "source_weather_cycle"):
                if key in weather.attrs:
                    out.attrs[key] = weather.attrs[key]
            out.flush()
    validate_package(tmp)
    os.replace(tmp, output_h5)
    return output_h5


@app.command()
def main(
    corrected_weather_h5: Annotated[Path, typer.Option("--corrected-weather")],
    static_h5: Annotated[Path, typer.Option("--static")],
    modis_h5: Annotated[Path, typer.Option("--modis")],
    output_h5: Annotated[Path, typer.Option("--output")],
    target_date: Annotated[str, typer.Option("--target-date")],
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    build_package(
        target=date.fromisoformat(target_date),
        corrected_weather_h5=corrected_weather_h5,
        static_h5=static_h5,
        modis_h5=modis_h5,
        output_h5=output_h5,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    app()
