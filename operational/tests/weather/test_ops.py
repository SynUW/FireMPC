"""Operational hardening: retention, latency accounting, and the daily window."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from wildfire.weather import pipeline as pipeline
from wildfire.weather import latency as latency
from wildfire.weather import purge as purge
from wildfire.weather import registry as registry


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------
def write_raster(path: Path, *, broken: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if broken:
        path.write_bytes(b"not a raster")
        return path
    with rasterio.open(path, "w", driver="GTiff", dtype="uint8", count=1,
                       height=2, width=2, crs="EPSG:4326",
                       transform=from_origin(-141.0, 64.0, 0.1, 0.1)) as dataset:
        dataset.write(np.ones((2, 2), dtype="uint8"), 1)
    return path


def stage_date(root: Path, product: str, stage: str, day: date, **kwargs) -> Path:
    folder = purge.stage_dir(root, stage, product)
    return write_raster(folder / f"{registry.date_token(day)}.tif", **kwargs)


def test_a_date_is_only_purgeable_once_it_is_scaled_and_old(tmp_path):
    old, recent, unscaled = date(2025, 1, 1), date(2026, 8, 10), date(2025, 2, 1)
    for day in (old, recent):
        stage_date(tmp_path, "FWI", "scaled", day)
    for day in (old, recent, unscaled):
        stage_date(tmp_path, "FWI", "unified", day)

    eligible = purge.purgeable_dates(tmp_path, "FWI", before=date(2026, 7, 18))
    assert eligible == {old}


def test_an_unreadable_scaled_raster_keeps_its_intermediates(tmp_path):
    """That is exactly the case where they are needed to diagnose the failure."""
    day = date(2025, 1, 1)
    stage_date(tmp_path, "FWI", "scaled", day, broken=True)
    stage_date(tmp_path, "FWI", "unified", day)
    assert purge.purgeable_dates(tmp_path, "FWI", before=date(2026, 1, 1)) == set()


def test_the_date_is_read_off_the_filename(tmp_path):
    assert purge.date_of(Path("2025_07_01.tif")) == date(2025, 7, 1)
    assert purge.date_of(Path("VPD_2025_07_01_tile02.tif")) == date(2025, 7, 1)
    assert purge.date_of(Path("latest.tif")) is None
    # A filename that parses as digits but not as a date must not become one.
    assert purge.date_of(Path("2025_13_45.tif")) is None


def test_scaled_is_not_a_purge_target(tmp_path):
    """It is the product this job exists to protect."""
    from typer.testing import CliRunner

    result = CliRunner().invoke(purge.app, ["--stages", "unified,scaled"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------
def test_lag_is_measured_from_midnight_utc_on_the_target_date():
    """So the number reads straight into a cron schedule."""
    assert latency.lag_hours(date(2026, 8, 17), "2026-08-17T06:30:00Z") == pytest.approx(6.5)
    assert latency.lag_hours(date(2026, 8, 17), "2026-08-18T00:00:00Z") == pytest.approx(24.0)


def test_first_seen_keeps_the_earliest_complete_observation():
    rows = [
        {"date": "2026-08-17", "complete": False, "observed_at_utc": "2026-08-17T02:00:00Z"},
        {"date": "2026-08-17", "complete": True, "observed_at_utc": "2026-08-17T08:00:00Z"},
        {"date": "2026-08-17", "complete": True, "observed_at_utc": "2026-08-17T14:00:00Z"},
    ]
    seen = latency.first_complete(rows)
    assert seen == {date(2026, 8, 17): "2026-08-17T08:00:00Z"}
    # An incomplete date is absent rather than recorded at some default lag.
    assert latency.first_complete(rows[:1]) == {}


def test_the_store_survives_a_probe_killed_mid_write(tmp_path):
    path = tmp_path / "ifs.jsonl"
    latency.append(path, {"date": "2026-08-17", "complete": True,
                          "observed_at_utc": "2026-08-17T08:00:00Z"})
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"date": "2026-08-18", "comp')
    assert len(latency.load(path)) == 1


def test_percentile_is_exact_on_a_tiny_sample():
    assert latency.percentile([5.0], 0.5) == 5.0
    assert latency.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.0
    assert latency.percentile([1.0, 2.0, 3.0, 4.0], 0.9) == 4.0


# ---------------------------------------------------------------------------
# The daily window
# ---------------------------------------------------------------------------
def test_the_window_runs_from_the_checkpoint_not_the_target():
    """A skipped day is a permanent offset in every later drought code."""
    assert pipeline.window_for(date(2026, 8, 17), date(2026, 8, 14), 14) == (
        date(2026, 8, 15), date(2026, 8, 17))


def test_a_cold_start_runs_the_target_alone():
    assert pipeline.window_for(date(2026, 8, 17), None, 14) == (
        date(2026, 8, 17), date(2026, 8, 17))


def test_a_checkpoint_at_or_past_the_target_is_refused():
    """Re-running a date the recursion has absorbed would double-count it."""
    with pytest.raises(RuntimeError, match="only moves forward"):
        pipeline.window_for(date(2026, 8, 17), date(2026, 8, 17), 14)
    with pytest.raises(RuntimeError, match="only moves forward"):
        pipeline.window_for(date(2026, 8, 17), date(2026, 8, 18), 14)


def test_a_long_backfill_is_refused_inside_a_daily_job():
    with pytest.raises(RuntimeError, match="more than --max-catch-up"):
        pipeline.window_for(date(2026, 8, 17), date(2026, 1, 1), 14)
