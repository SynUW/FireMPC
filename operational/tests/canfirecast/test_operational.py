from datetime import date
from pathlib import Path

import h5py
import numpy as np
import rasterio
from rasterio.transform import from_origin

from canfirecast import input_registry as registry
from canfirecast.weather_window import SOURCE_TO_CHANNEL, aligned_encoded_day
from canfirecast.state_ledger import install_state, state_path, validate_state
from canfirecast.pipeline import DEFAULT_STORAGE, STAGES, resolve_target


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


def test_pipeline_stops_after_validating_the_input_package():
    assert STAGES[-1] == "validate"
    assert "predict" not in STAGES


def test_pipeline_defaults_to_the_deduplicated_monthly_store():
    assert DEFAULT_STORAGE == "monthly"
