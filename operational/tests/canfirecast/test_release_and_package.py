from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import h5py
import numpy as np
import pytest

from wildfire.canfirecast import input_registry as registry
from wildfire.canfirecast.package import build_package, validate_package
from wildfire.canfirecast.provision import verify_release
from wildfire.modis.processing_registry import CHANNEL_NAMES as MODIS_NAMES


def _strings(h5: h5py.File, name: str, values) -> None:
    h5.create_dataset(name, data=np.asarray(values, dtype=object), dtype=h5py.string_dtype("utf-8"))


def test_release_verification_rejects_tampering(tmp_path: Path):
    release = tmp_path / "release"
    release.mkdir()
    payload = release / "artifact.bin"
    payload.write_bytes(b"frozen")
    manifest = {
        "schema_version": 1,
        "release_id": "test",
        "artifacts": {
            "artifact": {
                "path": payload.name,
                "sha256": registry.sha256(payload),
                "bytes": payload.stat().st_size,
            }
        },
    }
    (release / "release.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_release(release)["release_id"] == "test"
    payload.write_bytes(b"altered")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_release(release)


def test_all_frozen_dependencies_have_complete_provenance():
    files = registry.provenance_files()
    expected = {
        *(f"canfirecast/frozen_repo/{name}" for name in registry.VENDORED_CODE),
        *(f"canfirecast/artifacts/{name}" for name in registry.ARTIFACTS),
    }
    assert expected <= files.keys()
    required = {"source_project", "experiment_id", "source_path", "copy_date", "bytes", "sha256", "verbatim", "role", "redistribution_decision"}
    for name in expected:
        assert required <= files[name].keys()
        assert files[name]["verbatim"] is True
    registry.verify_pinned_artifacts()


def test_exact_model_projection():
    assert registry.COMPACT49_TO_NATIVE55 == (
        1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
        52, 53, 54,
    )


def test_native_package_contract(tmp_path: Path):
    target = date(2026, 8, 20)
    dates = [day.isoformat() for day in registry.target_window_dates(target)]
    weather_path = tmp_path / "weather.h5"
    static_path = tmp_path / "static.h5"
    modis_path = tmp_path / "modis.h5"
    output = tmp_path / "canfirecast_inputs.h5"
    height, width = 2, 3

    with h5py.File(weather_path, "w") as h5:
        h5.create_dataset("data", data=np.ones((10, 55, height, width), dtype=np.uint8))
        h5.create_dataset("spatial_validity", data=np.ones((10, height, width), dtype=bool))
        _strings(h5, "dates", dates)
        _strings(h5, "channel_names", registry.CHANNEL_NAMES)
        h5.attrs.update({
            "crs": "EPSG:4326", "transform": json.dumps([1, 0, 0, 0, -1, 0]),
            "height": height, "width": width, "d11_state_date": "2026-08-09",
        })
    with h5py.File(static_path, "w") as h5:
        h5.create_dataset("data", data=np.full((12, height, width), 2, dtype=np.uint8))
        _strings(h5, "channel_names", registry.CHANNEL_NAMES[26:38])
    with h5py.File(modis_path, "w") as h5:
        h5.create_dataset("data", data=np.full((10, 17, height, width), 3, dtype=np.uint8))
        _strings(h5, "channel_names", MODIS_NAMES)
        h5.create_dataset("fill_age_days", data=np.zeros((10, 17), dtype=np.int16))
        _strings(h5, "source_dates", np.asarray([dates] * 17).T)

    build_package(
        target=target, corrected_weather_h5=weather_path, static_h5=static_path,
        modis_h5=modis_path, output_h5=output,
    )
    validate_package(output)
    with h5py.File(output) as h5:
        assert h5["data"].shape == (10, 55, height, width)
        assert h5["data"].dtype == np.dtype("uint8")
        assert h5.attrs["schema_version"] == 1
        assert np.all(h5["data"][:, registry.FIRE_DETECTION_CHANNEL] == 0)
        assert np.all(h5["data"][:, list(registry.ERA5_ONLY_MISSING_CHANNELS)] == 0)
        assert np.all(h5["data"][:, 26:38] == 2)
        assert np.all(h5["data"][:, 38:55] == 3)
