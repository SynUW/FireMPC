from datetime import date
from pathlib import Path

import h5py
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from canfirecast import input_registry as registry
from canfirecast.weather_window import SOURCE_TO_CHANNEL, aligned_encoded_day
from canfirecast.state_ledger import install_state, state_path, validate_state
from canfirecast.pipeline import (
    DEFAULT_STORAGE, STAGES, modis_pipeline_command,
    remove_ingested_modis_handoff, resolve_target,
)


def _raster(path: Path, count: int, descriptions=()):
    profile = dict(driver="GTiff", width=3, height=2, count=count, dtype="float32", crs="EPSG:4326", transform=from_origin(-3, 2, 1, 1))
    with rasterio.open(path, "w", **profile) as out:
        out.write(np.full((count, 2, 3), 1.0, dtype="float32"))
        for index, name in enumerate(descriptions, 1): out.set_band_description(index, name)


def test_weather_alignment_resolves_named_bands(tmp_path):
    names = tuple(SOURCE_TO_CHANNEL)
    source = tmp_path / "ifs.tif"; template = tmp_path / "template.tif"
    _raster(source, len(names), names); _raster(template, 1)
    with rasterio.open(template) as grid:
        planes, valid = aligned_encoded_day((source, source, source, source), grid)
    assert set(planes) == set(registry.DIRECT_WEATHER_CHANNELS)
    assert valid.all()
    assert all(plane.dtype == np.float32 for plane in planes.values())
    assert any(np.any(plane != np.rint(plane)) for plane in planes.values())


def test_certified_state_install_is_date_addressed(tmp_path):
    template = tmp_path / "grid.tif"; _raster(template, 1)
    source = tmp_path / "state.npz"
    np.savez(source, state=np.zeros((3, 2, 3), dtype="uint8"))
    day = date(2026, 8, 8)
    output = install_state(source=source, day=day, root=tmp_path / "ledger", template_path=template, evidence="golden parity fixture")
    assert output == state_path(tmp_path / "ledger", day)
    validate_state(output, expected_date=day, template_path=template)



def test_explicit_target_resolution():
    assert resolve_target("2026-08-19") == date(2026, 8, 19)


def test_pipeline_stops_after_publishing_the_input_package():
    assert STAGES[-1] == "publish"
    assert STAGES[-2] == "validate"
    assert "predict" not in STAGES


def test_pipeline_defaults_to_the_deduplicated_monthly_store():
    assert DEFAULT_STORAGE == "monthly"


def test_ingested_modis_handoff_is_removed(tmp_path, monkeypatch):
    target = date(2026, 8, 19)
    package = tmp_path / "modis" / target.isoformat()
    package.mkdir(parents=True)
    h5 = package / "modis_inputs.h5"
    h5.write_bytes(b"handoff")
    (package / "manifest.json").write_bytes(b"manifest")
    monkeypatch.setattr("canfirecast.pipeline.monthly_store.is_written", lambda root, day: True)

    reclaimed = remove_ingested_modis_handoff(h5, target, tmp_path / "store")

    assert reclaimed == len(b"handoffmanifest")
    assert not package.exists()


def test_modis_handoff_is_kept_if_monthly_ingestion_is_incomplete(tmp_path, monkeypatch):
    target = date(2026, 8, 19)
    package = tmp_path / target.isoformat()
    package.mkdir()
    h5 = package / "modis_inputs.h5"
    h5.write_bytes(b"handoff")
    missing_day = registry.target_window_dates(target)[-1]
    monkeypatch.setattr(
        "canfirecast.pipeline.monthly_store.is_written",
        lambda root, day: day != missing_day,
    )

    with pytest.raises(RuntimeError, match=str(missing_day)):
        remove_ingested_modis_handoff(h5, target, tmp_path / "store")

    assert h5.exists()


# ---------------------------------------------------------------------------
# Backfill honesty
# ---------------------------------------------------------------------------
def _modis_command(**kwargs):
    defaults = dict(
        backend="ea", stream="standard", modis_policy="latest-available",
        as_of=None, overwrite=[],
    )
    return modis_pipeline_command(date(2026, 1, 15), **{**defaults, **kwargs})


def test_the_daily_run_does_not_restrict_modis_to_a_past_as_of():
    """An operational run happens on its target date; on-disk is the truth."""
    assert "--as-of" not in _modis_command()


def test_a_hindcast_target_pins_modis_to_what_had_landed_by_then():
    command = _modis_command(as_of="2026-01-15")
    assert command[command.index("--as-of") + 1] == "2026-01-15"


def test_overwrite_still_trails_the_forwarded_as_of():
    command = _modis_command(as_of="2026-01-15", overwrite=["--overwrite"])
    assert command[command.index("--as-of") + 1] == "2026-01-15"
    assert command[-1] == "--overwrite"
