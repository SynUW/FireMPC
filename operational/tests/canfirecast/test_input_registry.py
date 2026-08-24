from datetime import date

import numpy as np

from wildfire.canfirecast import input_registry as r
from wildfire.modis.processing_registry import CHANNEL_NAMES as MODIS_NAMES


def test_native_and_compact_mappings():
    assert len(r.CHANNEL_NAMES) == 55
    assert len(r.COMPACT49_TO_NATIVE55) == 49
    assert r.COMPACT49_OMITTED_NATIVE == (0, 4, 6, 15, 16, 17)
    assert set(r.COMPACT49_TO_NATIVE55).isdisjoint(r.COMPACT49_OMITTED_NATIVE)
    assert r.COMPACT49_TO_NATIVE55[r.NATIVE55_TO_COMPACT49[38]] == 38


def test_expand_compact49_channel_first():
    compact = np.ones((10, 49, 2, 3), dtype=np.uint8)
    expanded = r.expand_compact49(compact)
    assert expanded.shape == (10, 55, 2, 3)
    assert np.all(expanded[:, list(r.COMPACT49_OMITTED_NATIVE)] == 0)
    assert np.all(expanded[:, r.COMPACT49_TO_NATIVE55] == 1)


def test_date_arithmetic_leap_year_boundary():
    target = date(2024, 3, 1)
    assert r.state_date(target) == date(2024, 2, 19)
    assert r.target_window_dates(target)[0] == date(2024, 2, 20)
    assert r.target_window_dates(target)[-1] == date(2024, 2, 29)


def test_weather_groups_and_ranges():
    assert r.FIRE_DETECTION_CHANNEL == 0
    assert r.ERA5_ONLY_MISSING_CHANNELS == (4, 6, 15, 16, 17)
    assert set(r.PHYSICAL_RANGES) == set(range(1, 26)) - set(r.ERA5_ONLY_MISSING_CHANNELS)
    assert r.PHYSICAL_RANGES[25] == (0.0, 1.63)


def test_modis_model_order_is_fpar_before_lai():
    assert tuple(r.CHANNEL_NAMES[38:42]) == ("NDVI", "EVI", "FPAR", "LAI")
    assert MODIS_NAMES[:4] == ("NDVI", "EVI", "FPAR", "LAI")


def test_pinned_artifacts_present_and_hashed():
    r.verify_pinned_artifacts()
