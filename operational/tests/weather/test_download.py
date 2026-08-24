"""Retry classification and per-date source selection in the downloader.

The Earth Engine calls themselves are not exercised here -- they need a live
service -- but everything that decides *whether to call again*, and *which
collection to call*, is pure and is tested.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from wildfire.weather import download as download
from wildfire.weather import registry as registry


class Flaky:
    """Fails with `error` for the first `failures` calls, then returns 'ok'."""

    def __init__(self, failures: int, error: BaseException):
        self.failures = failures
        self.error = error
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return "ok"


def test_transient_service_failures_are_retried():
    """The 594-day backfill exited non-zero on exactly one 503. Not again."""
    call = Flaky(2, RuntimeError("503 Service Unavailable"))
    slept: list[float] = []
    assert download.with_retry("tile", call, attempts=5, sleep=slept.append) == "ok"
    assert call.calls == 3
    # Exponential, and never twice at the same instant across four parallel tiles.
    assert len(slept) == 2 and slept[1] > slept[0]


def test_a_request_error_is_not_retried():
    """A 400 or a missing asset fails identically forever; retrying hides it."""
    call = Flaky(1, ValueError("Image.load: asset not found"))
    with pytest.raises(ValueError):
        download.with_retry("tile", call, attempts=5, sleep=lambda _: None)
    assert call.calls == 1


def test_retries_are_finite():
    call = Flaky(99, RuntimeError("502 Bad Gateway"))
    with pytest.raises(RuntimeError, match="502"):
        download.with_retry("tile", call, attempts=3, sleep=lambda _: None)
    assert call.calls == 3


@pytest.mark.parametrize("message", [
    "503 Service Unavailable", "429 Too Many Requests", "Read timed out",
    "Connection reset by peer", "Deadline exceeded",
])
def test_transient_markers(message):
    assert download.is_transient(RuntimeError(message))


@pytest.mark.parametrize("message", [
    "Image.load: asset not found", "Invalid argument: band 'x' not found",
    "User memory limit exceeded",
])
def test_permanent_failures_are_not_transient(message):
    assert not download.is_transient(RuntimeError(message))


def test_auto_follows_the_changeover_and_an_explicit_source_does_not():
    before = registry.SOURCE_SWITCH_DATE.replace(year=2023)
    assert download.resolve_source("auto", before) is registry.ERA5
    assert download.resolve_source("auto", date(2026, 8, 17)) is registry.IFS
    # An explicit source is taken literally, which is what makes a deliberate
    # ERA5-vs-IFS comparison on the same date possible.
    assert download.resolve_source("ifs", before) is registry.IFS


def test_each_source_lands_in_its_own_directory(tmp_path):
    day = date(2026, 8, 17)
    ifs = download.tile_path(tmp_path, registry.IFS, day, 0)
    era5 = download.tile_path(tmp_path, registry.ERA5, day, 0)
    assert ifs == tmp_path / "IFS" / "IFS_2026_08_17_tile00.tif"
    assert era5 == tmp_path / "ERA5" / "ERA5_2026_08_17_tile00.tif"
    assert download.sidecar_path(tmp_path, registry.IFS, day).name == "IFS_2026_08_17.json"


def test_validate_tile_reports_the_band_count_of_the_named_source(tmp_path):
    """An ERA5 tile is a perfectly valid raster and a wrong IFS tile."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    class Grid:
        px, minx, maxy, width, height = 0.1, -141.0, 64.0, 4, 3

    path = tmp_path / "t.tif"
    with rasterio.open(path, "w", driver="GTiff", dtype="float32",
                       count=len(registry.ERA5.bands), height=3, width=4,
                       crs="EPSG:4326",
                       transform=from_origin(-141.0, 64.0, 0.1, 0.1)) as dataset:
        dataset.write(np.zeros((3, 4), dtype="float32"), 1)

    assert download.validate_tile(Path(path), registry.ERA5, Grid()) == []
    assert any("expected 23" in error
               for error in download.validate_tile(Path(path), registry.IFS, Grid()))
