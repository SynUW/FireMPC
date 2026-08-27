"""Thin wrappers for frozen corrected VPD/CFFDRS derivation.

The acceptance oracle is the vendored frozen implementation.  This module only
normalizes operational tensor shapes and enforces the final configuration:
weather residual correction with alpha=0.625 and no optional state correction.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Annotated

# Run directly as `python operational/canfirecast/model/cffdrs_derive.py`: put the application root on
# sys.path so the shared modules import by bare name, as elsewhere in FireMPC.
import sys
from pathlib import Path
OPERATIONAL = Path(__file__).resolve().parents[2]
if str(OPERATIONAL) not in sys.path:
    sys.path.insert(0, str(OPERATIONAL))

import vendorpaths
import h5py
import numpy as np
import typer

from canfirecast import input_registry as registry

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

vendorpaths.ensure_frozen_repo()

from cffdrs_aware_correction import CFFDRSAwareCorrection, transform_corrected_window  # noqa: E402


def _date_int(day: date) -> int:
    return int(day.strftime("%Y%m%d"))


def to_frozen_layout(window: np.ndarray) -> np.ndarray:
    """Convert native package layout (10,55,H,W) to frozen (55,H,W,10)."""
    arr = np.asarray(window)
    if arr.ndim != 4 or arr.shape[:2] != (10, 55):
        raise ValueError(f"expected (10,55,H,W), got {arr.shape}")
    return np.transpose(arr, (1, 2, 3, 0))


def from_frozen_layout(window: np.ndarray) -> np.ndarray:
    """Convert frozen layout (55,H,W,10) to native package layout (10,55,H,W)."""
    arr = np.asarray(window)
    if arr.ndim != 4 or arr.shape[0] != 55 or arr.shape[-1] != 10:
        raise ValueError(f"expected (55,H,W,10), got {arr.shape}")
    return np.transpose(arr, (3, 0, 1, 2))


def load_correction(path: str | Path | None = None) -> CFFDRSAwareCorrection:
    registry.verify_vendored_code()
    artifact = Path(path or registry.ARTIFACTS["cffdrs_aware_correction.json"]["path"])
    expected = registry.ARTIFACTS["cffdrs_aware_correction.json"]["sha256"]
    registry.verify_file_hash(artifact, expected, label="cffdrs_aware_correction.json")
    correction = CFFDRSAwareCorrection.from_json(artifact)
    if abs(correction.weather_alpha - registry.WEATHER_ALPHA) > 1e-12:
        raise RuntimeError(
            f"unexpected weather_alpha {correction.weather_alpha}; "
            f"expected {registry.WEATHER_ALPHA}"
        )
    return correction


def derive_corrected_weather_channels(
    encoded_window: np.ndarray,
    *,
    target: date,
    center_row: int,
    initial_cffdrs_encoded: np.ndarray,
    correction: CFFDRSAwareCorrection | None = None,
) -> np.ndarray:
    """Apply final residual weather correction and recursively derive 19-25.

    Parameters
    ----------
    encoded_window:
        Native layout ``(10,55,H,W)`` with direct channels already physically
        adapted/encoded and non-weather channels populated or zero-filled.
    target:
        Target date D; input dates are D-10..D-1.
    center_row:
        National-grid center row used by the frozen latitude-stratum logic.
    initial_cffdrs_encoded:
        Encoded D-11 FFMC/DMC/DC state with shape ``(3,H,W)``.
    """
    registry.validate_registry()
    correction = correction or load_correction()
    frozen = to_frozen_layout(encoded_window).astype(np.float32, copy=False)
    initial = np.asarray(initial_cffdrs_encoded, dtype=np.float32)
    if initial.shape != (3, frozen.shape[1], frozen.shape[2]):
        raise ValueError(f"initial state must be (3,H,W), got {initial.shape}")
    result = transform_corrected_window(
        frozen,
        _date_int(target),
        int(center_row),
        initial,
        correction,
        use_state_correction=registry.ENABLE_STATE_CORRECTION,
    )
    out = np.rint(from_frozen_layout(result)).clip(0, 255).astype(np.uint8)
    out[:, registry.FIRE_DETECTION_CHANNEL] = 0
    out[:, list(registry.ERA5_ONLY_MISSING_CHANNELS)] = 0
    if not np.isfinite(out).all():
        raise RuntimeError("corrected window contains non-finite values")
    return out


@app.command()
def main(
    input_h5: Annotated[Path, typer.Option("--input")],
    state_h5: Annotated[Path, typer.Option("--state")],
    output_h5: Annotated[Path, typer.Option("--output")],
    target_date: Annotated[str, typer.Option("--target-date")],
    correction_artifact: Annotated[Path | None, typer.Option("--correction-artifact")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Apply residual correction and recursive derivation using certified D-11 state."""
    target = date.fromisoformat(target_date)
    if output_h5.exists() and not overwrite:
        raise typer.BadParameter(f"output exists: {output_h5}")
    with h5py.File(input_h5, "r") as source, h5py.File(state_h5, "r") as state:
        data = source["data"][()]
        initial = state["state"][()]
        state_day = str(state.attrs.get("date", ""))
        if state_day != registry.state_date(target).isoformat():
            raise RuntimeError(f"state date {state_day!r} is not target D-11")
        if not state.attrs.get("certified", False):
            raise RuntimeError("D-11 state is not marked certified")
        attrs = dict(source.attrs)
        spatial_validity = source["spatial_validity"][()] if "spatial_validity" in source else None
        # A national window starts at row zero; center_row is only an adapter
        # representation of row identity for patch inputs.
        center_row = data.shape[2] // 2
        corrected = derive_corrected_weather_channels(
            data, target=target, center_row=center_row,
            initial_cffdrs_encoded=initial,
            correction=load_correction(correction_artifact),
        )
        state_hash = registry.sha256(state_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_h5.with_suffix(output_h5.suffix + ".tmp")
    strings = h5py.string_dtype("utf-8")
    with h5py.File(tmp, "w") as output:
        output.create_dataset("data", data=corrected, compression="gzip", shuffle=True)
        if spatial_validity is not None:
            output.create_dataset("spatial_validity", data=spatial_validity, compression="gzip")
        output.create_dataset("dates", data=np.asarray([d.isoformat() for d in registry.target_window_dates(target)], dtype=object), dtype=strings)
        output.create_dataset("channel_names", data=np.asarray(registry.CHANNEL_NAMES, dtype=object), dtype=strings)
        output.attrs.update(attrs)
        output.attrs.update({
            "target_date": target.isoformat(), "correction_stage": "physical_plus_residual_plus_recursive",
            "weather_alpha": registry.WEATHER_ALPHA, "state_correction_enabled": False,
            "d11_state_date": state_day, "d11_state_source": str(state_h5),
            "d11_state_sha256": state_hash,
            "cffdrs_correction_artifact": str(correction_artifact or registry.ARTIFACTS["cffdrs_aware_correction.json"]["path"]),
            "cffdrs_correction_sha256": registry.sha256(correction_artifact or registry.ARTIFACTS["cffdrs_aware_correction.json"]["path"]),
        })
    os.replace(tmp, output_h5)
    artifact = Path(correction_artifact or registry.ARTIFACTS["cffdrs_aware_correction.json"]["path"])
    output_h5.with_suffix(output_h5.suffix + ".manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "product": "canfirecast_cffdrs_corrected_weather",
        "target_date": target.isoformat(),
        "source": str(input_h5),
        "output": str(output_h5),
        "artifact": {
            "path": str(artifact),
            "sha256": registry.sha256(artifact),
            "expected_sha256": registry.ARTIFACTS["cffdrs_aware_correction.json"]["sha256"],
        },
        "d11_state_sha256": state_hash,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    app()
