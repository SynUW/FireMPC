"""The month store must hold each date once and give the window back unchanged."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import h5py
import numpy as np
import pytest

from canfirecast import input_registry as registry
from canfirecast import store

GRID = (8, 12)
CRS = "EPSG:4326"
TRANSFORM = (0.0449157642, 0.0, -141.00855, 0.0, -0.0449157642, 70.0)


def _write_run(root: Path, target: date, *, seed: int) -> list[date]:
    """Fold a synthetic run's ten dates in, the way ingest_window would."""
    rng = np.random.default_rng(seed)
    written = []
    for day in registry.target_window_dates(target):
        if store.is_written(root, day):
            continue
        weather = np.zeros((len(registry.CHANNEL_NAMES), *GRID), dtype="f4")
        for channel in store.WEATHER_CHANNELS:
            # Fractional on purpose: the store exists partly to keep the
            # pre-rounding values the recursion needs.
            weather[channel] = rng.uniform(0, 255, GRID).astype("f4")
        store.write_day(
            root, day,
            weather=weather,
            modis=rng.integers(0, 255, (len(store.MODIS_CHANNELS), *GRID), dtype=np.uint8),
            validity=np.ones(GRID, dtype=bool),
            modis_fill_age_days=np.arange(len(store.MODIS_CHANNELS), dtype="i2"),
            modis_source_dates=[day.isoformat()] * len(store.MODIS_CHANNELS),
            written_for_target=target, crs=CRS, transform=TRANSFORM,
        )
        written.append(day)
    return written


def test_stored_channels_exclude_statics_zeros_and_the_recursion():
    stored = set(store.WEATHER_CHANNELS) | set(store.MODIS_CHANNELS)
    assert not stored & set(registry.STATIC_CHANNELS)
    assert not stored & set(registry.COMPACT49_OMITTED_NATIVE)
    # 19-25 come out of the CFFDRS recursion, which is seeded per target from
    # the D-11 ledger and so is not a function of the date.
    assert not stored & set(registry.DERIVED_WEATHER_CHANNELS)
    assert len(stored) == 30


def test_second_run_only_writes_its_new_date(tmp_path: Path):
    root = tmp_path / "store"
    assert len(_write_run(root, date(2026, 7, 20), seed=1)) == 10
    assert _write_run(root, date(2026, 7, 21), seed=2) == [date(2026, 7, 20)]


def test_a_frozen_date_is_refused_not_overwritten(tmp_path: Path):
    root = tmp_path / "store"
    _write_run(root, date(2026, 7, 20), seed=1)
    weather = np.zeros((len(registry.CHANNEL_NAMES), *GRID), dtype="f4")
    modis = np.zeros((len(store.MODIS_CHANNELS), *GRID), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="already written"):
        store.write_day(root, date(2026, 7, 15), weather=weather, modis=modis,
                        validity=np.ones(GRID, dtype=bool))
    store.write_day(root, date(2026, 7, 15), weather=weather, modis=modis,
                    validity=np.ones(GRID, dtype=bool), force=True)


def test_weather_survives_the_round_trip_unrounded(tmp_path: Path):
    """Rounding on the way in would break the recursion downstream."""
    root = tmp_path / "store"
    day = date(2026, 7, 15)
    weather = np.zeros((len(registry.CHANNEL_NAMES), *GRID), dtype="f4")
    for channel in store.WEATHER_CHANNELS:
        weather[channel] = np.full(GRID, 12.375, dtype="f4")
    store.write_day(root, day, weather=weather,
                    modis=np.zeros((len(store.MODIS_CHANNELS), *GRID), dtype=np.uint8),
                    validity=np.ones(GRID, dtype=bool), crs=CRS, transform=TRANSFORM)
    with h5py.File(store.month_path(root, day)) as h5:
        assert h5["weather"].dtype == np.dtype("float32")
        assert np.all(h5["weather"][store.day_index(day)] == 12.375)


def test_an_incomplete_date_is_not_readable(tmp_path: Path):
    root = tmp_path / "store"
    _write_run(root, date(2026, 7, 20), seed=1)
    path = store.month_path(root, date(2026, 7, 20))
    with h5py.File(path, "r+") as h5:
        h5["complete"][store.day_index(date(2026, 7, 15))] = False
    assert not store.is_written(root, date(2026, 7, 15))


def test_a_window_spanning_two_months_opens_both(tmp_path: Path):
    root = tmp_path / "store"
    target = date(2026, 8, 5)  # D-10..D-1 is 2026-07-26 .. 2026-08-04
    _write_run(root, target, seed=3)
    assert store.month_path(root, date(2026, 7, 26)).is_file()
    assert store.month_path(root, date(2026, 8, 4)).is_file()
    assert all(store.is_written(root, day) for day in registry.target_window_dates(target))


def test_month_files_are_labelled_by_month_alone(tmp_path: Path):
    root = tmp_path / "store"
    _write_run(root, date(2026, 7, 20), seed=1)
    assert [p.name for p in sorted(root.glob("*.h5"))] == ["2026-07.h5"]
