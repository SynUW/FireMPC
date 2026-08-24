"""Physical IFS-to-ERA5 weather correction helpers for CanFireCast inputs.

This is a library/CLI entry point for the corrected-weather product.  It keeps
raw IFS diagnostics separate and delegates equations to the vendored frozen
adapter; no fitting happens here.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

from wildfire import vendorpaths
import h5py
import numpy as np
import typer

from wildfire.canfirecast import input_registry as registry
from wildfire.canfirecast.model.cffdrs_derive import from_frozen_layout, to_frozen_layout

vendorpaths.ensure_frozen_repo()

from full10_ifs_domain_transform import DIRECT, decode, encode  # noqa: E402
from ifs_to_era5_physical_adapter import IFSToERA5Adapter  # noqa: E402

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def parse_date(value: str | date) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


def date_int(day: date) -> int:
    return int(day.strftime("%Y%m%d"))


def require_target_dates(target: date, dates: tuple[date, ...]) -> None:
    expected = registry.target_window_dates(target)
    if tuple(dates) != expected:
        raise ValueError(
            "input dates must be D-10..D-1 oldest first; "
            f"expected {[d.isoformat() for d in expected]}, "
            f"got {[d.isoformat() for d in dates]}"
        )


def load_adapter(path: str | Path | None = None) -> IFSToERA5Adapter:
    registry.verify_vendored_code()
    artifact = Path(path or registry.ARTIFACTS["ifs_to_era5_physical_adapter.json"]["path"])
    expected = registry.ARTIFACTS["ifs_to_era5_physical_adapter.json"]["sha256"]
    registry.verify_file_hash(artifact, expected, label="ifs_to_era5_physical_adapter.json")
    return IFSToERA5Adapter.from_json(artifact)


def apply_physical_adapter(
    encoded_window: np.ndarray,
    *,
    target: date,
    center_row: int,
    adapter: IFSToERA5Adapter | None = None,
) -> np.ndarray:
    """Apply the frozen physical IFS->ERA5 adapter to a native window.

    Input and output layout is ``(10,55,H,W)``.  Only the 13 direct common
    meteorology channels are modified; final rounding to uint8 happens here.
    """
    registry.validate_registry()
    adapter = adapter or load_adapter()
    frozen = to_frozen_layout(encoded_window).astype(np.float32, copy=True)
    height, width = frozen.shape[1:3]
    valid = frozen[1, :, :, 0] > 0
    row_grid = int(center_row) + np.arange(height, dtype=np.int64)[:, None] - height // 2
    row_grid = np.broadcast_to(row_grid, (height, width))
    for time_index, day in enumerate(registry.target_window_dates(target)):
        matrix = np.stack(
            [decode(frozen[channel, :, :, time_index], channel) for channel in DIRECT],
            axis=-1,
        )
        mapped = adapter.map_pairs(
            matrix[valid],
            np.full(int(valid.sum()), date_int(day), dtype=np.int64),
            row_grid[valid],
        )
        for direct_index, channel in enumerate(DIRECT):
            encoded = np.zeros((height, width), dtype=np.float32)
            encoded[valid] = encode(mapped[:, direct_index], channel)
            frozen[channel, :, :, time_index] = encoded
    frozen[list(registry.ERA5_ONLY_MISSING_CHANNELS), :, :, :] = 0.0
    # Do not quantize between the physical and residual stages. Frozen cache
    # parity depends on retaining fractional encoded values until recursive
    # derivation has completed.
    out = from_frozen_layout(frozen).astype(np.float32, copy=False)
    out[:, registry.FIRE_DETECTION_CHANNEL] = 0
    out[:, list(registry.ERA5_ONLY_MISSING_CHANNELS)] = 0
    return out


def _read_dates(h5: h5py.File) -> tuple[date, ...]:
    raw = h5["dates"][()]
    return tuple(parse_date(x.decode() if isinstance(x, bytes) else str(x)) for x in raw)


def write_manifest(path: Path, *, target: date, source: Path, output: Path, adapter_artifact: Path | None = None) -> None:
    artifact = Path(adapter_artifact or registry.ARTIFACTS["ifs_to_era5_physical_adapter.json"]["path"])
    manifest = {
        "schema_version": 1,
        "product": "canfirecast_physical_corrected_weather",
        "target_date": target.isoformat(),
        "dates": [d.isoformat() for d in registry.target_window_dates(target)],
        "source": str(source),
        "output": str(output),
        "registry_version": registry.REGISTRY_VERSION,
        "weather_alpha": registry.WEATHER_ALPHA,
        "state_correction_enabled": registry.ENABLE_STATE_CORRECTION,
        "artifacts": {
            "ifs_to_era5_physical_adapter.json": {
                "path": str(artifact),
                "sha256": registry.sha256(artifact),
                "expected_sha256": registry.ARTIFACTS["ifs_to_era5_physical_adapter.json"]["sha256"],
            }
        },
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@app.command()
def main(
    input_h5: Annotated[Path, typer.Argument(help="H5 containing data=(10,55,H,W) and dates")],
    output_h5: Annotated[Path, typer.Argument(help="Corrected output H5")],
    target_date: Annotated[str, typer.Option("--target-date")],
    center_row: Annotated[int | None, typer.Option("--center-row")] = None,
    adapter_artifact: Annotated[Path | None, typer.Option("--adapter-artifact")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Apply physical correction to an aligned encoded 55-channel window H5."""
    target = parse_date(target_date)
    if output_h5.exists() and not overwrite:
        raise typer.BadParameter(f"output exists: {output_h5}")
    with h5py.File(input_h5, "r") as src:
        data = src["data"][()]
        require_target_dates(target, _read_dates(src))
        # National cubes begin at row zero; infer their center so callers do
        # not manually supply an adapter patch coordinate.
        resolved_center_row = data.shape[2] // 2 if center_row is None else center_row
        corrected = apply_physical_adapter(
            data, target=target, center_row=resolved_center_row,
            adapter=load_adapter(adapter_artifact),
        )
        attrs = dict(src.attrs)
        spatial_validity = src["spatial_validity"][()] if "spatial_validity" in src else None
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_h5.with_suffix(output_h5.suffix + ".tmp")
    with h5py.File(tmp, "w") as dst:
        dst.create_dataset("data", data=corrected, compression="gzip", shuffle=True)
        if spatial_validity is not None:
            dst.create_dataset("spatial_validity", data=spatial_validity, compression="gzip")
        strings = h5py.string_dtype("utf-8")
        dst.create_dataset(
            "dates",
            data=np.asarray([d.isoformat() for d in registry.target_window_dates(target)], dtype=object),
            dtype=strings,
        )
        dst.create_dataset("channel_names", data=np.asarray(registry.CHANNEL_NAMES, dtype=object), dtype=strings)
        dst.attrs.update(attrs)
        dst.attrs["target_date"] = target.isoformat()
        dst.attrs["channel_registry_version"] = registry.REGISTRY_VERSION
        dst.attrs["correction_stage"] = "physical_ifs_to_era5"
    tmp.replace(output_h5)
    write_manifest(
        output_h5.with_suffix(output_h5.suffix + ".manifest.json"),
        target=target, source=input_h5, output=output_h5,
        adapter_artifact=adapter_artifact,
    )


if __name__ == "__main__":
    app()
