import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from wildfire.modis.preprocess import load_complete_tiles
from wildfire.modis.purge import (
    load_tiling_history,
    main,
    purge_date,
    raw_date_dir,
    validated_dates,
)


def _write_tile(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", width=2, height=2, count=1, dtype="uint8",
        transform=from_origin(0, 90, 1, 1),
    ) as dataset:
        dataset.write(np.zeros((2, 2), dtype="uint8"), 1)
    return path


def _tiling_report(tmp_path, entries, *, name="tiling.json", failures=0):
    """entries: list of (product, date_str, status, [tile paths])."""
    results = []
    for product, date_str, status, tile_paths in entries:
        tiles = {}
        for tile, path in enumerate(tile_paths):
            tiles[str(tile)] = {"status": "written", "path": str(path)}
        results.append({"product": product, "date": date_str, "status": status, "tiles": tiles})
    report = {"summary": {"failures": failures}, "results": results}
    manifest = tmp_path / name
    manifest.write_text(json.dumps(report))
    return manifest


def _four_tiles(tmp_path, name):
    return [_write_tile(tmp_path / f"{name}_tile{t}.tif") for t in range(4)]


def test_validated_dates_requires_every_export_of_the_collection(tmp_path):
    day = date(2026, 1, 1)
    manifest = _tiling_report(tmp_path, [
        ("MOD09GA_b1237", "2026-01-01", "complete", _four_tiles(tmp_path, "b1237")),
        ("MOD09GA_QA_state1km", "2026-01-01", "complete", _four_tiles(tmp_path, "qa")),
    ])
    complete = load_complete_tiles(manifest)
    assert validated_dates(["MOD09GA_b1237", "MOD09GA_QA_state1km"], complete) == {day}
    # Only one of the two exports is complete: not purge-eligible.
    assert validated_dates(
        ["MOD09GA_b1237", "MOD09GA_QA_state1km_missing"], complete,
    ) == set()


def test_validated_dates_excludes_unopenable_tiles(tmp_path):
    broken = tmp_path / "broken_tile0.tif"
    broken.write_bytes(b"not a tiff")
    tiles = [broken, *_four_tiles(tmp_path, "mod11a1")[1:]]
    manifest = _tiling_report(tmp_path, [
        ("MOD11A1", "2026-01-01", "complete", tiles),
    ])
    complete = load_complete_tiles(manifest)
    assert validated_dates(["MOD11A1"], complete) == set()


def test_deep_validation_is_opt_in_and_applies_the_full_raster_contract(tmp_path):
    tiles = _four_tiles(tmp_path, "mod11a1")
    manifest = _tiling_report(tmp_path, [
        ("MOD11A1", "2026-01-01", "complete", tiles),
    ])
    complete = load_complete_tiles(manifest)
    # The default trusts the manifest: these 2x2 rasters open, so they pass.
    assert validated_dates(["MOD11A1"], complete) == {date(2026, 1, 1)}
    # The audit path rejects them -- wrong band count, pixel size, origin.
    assert validated_dates(["MOD11A1"], complete, deep=True) == set()


def test_dates_inside_the_retention_window_are_never_validated(tmp_path):
    """Retention decides first, so kept dates cost no validation work."""
    manifest = _tiling_report(tmp_path, [
        ("MOD11A1", "2026-01-01", "complete", _four_tiles(tmp_path, "old")),
        ("MOD11A1", "2026-06-01", "complete", _four_tiles(tmp_path, "recent")),
    ])
    complete = load_complete_tiles(manifest)
    validated = []

    def _spy(entries, *, deep):
        validated.append(entries[0].day)
        return []

    from wildfire.modis import purge as modis_purge_raw

    original = modis_purge_raw._tile_failures
    modis_purge_raw._tile_failures = _spy
    try:
        eligible = validated_dates(["MOD11A1"], complete, before=date(2026, 3, 1))
    finally:
        modis_purge_raw._tile_failures = original

    assert eligible == {date(2026, 1, 1)}
    assert validated == [date(2026, 1, 1)]


def test_history_unions_dates_across_reports(tmp_path):
    """One report sees only its own run; purging must see every date tiled."""
    first = _tiling_report(tmp_path, [
        ("MOD11A1", "2026-01-01", "complete", _four_tiles(tmp_path, "jan01")),
    ], name="20260101T000000Z.json")
    second = _tiling_report(tmp_path, [
        ("MOD11A1", "2026-02-01", "complete", _four_tiles(tmp_path, "feb01")),
    ], name="20260201T000000Z.json")

    assert set(load_complete_tiles(first)) == {("MOD11A1", date(2026, 1, 1))}
    assert set(load_complete_tiles(second)) == {("MOD11A1", date(2026, 2, 1))}
    assert set(load_tiling_history(tmp_path)) == {
        ("MOD11A1", date(2026, 1, 1)),
        ("MOD11A1", date(2026, 2, 1)),
    }


def test_history_keeps_good_dates_from_a_report_that_had_a_failure(tmp_path):
    """One bad granule in one old run must not veto every date in that file."""
    manifest = _tiling_report(tmp_path, [
        ("MOD11A1", "2026-01-01", "complete", _four_tiles(tmp_path, "good")),
        ("MOD11A1", "2026-01-02", "failed", _four_tiles(tmp_path, "bad")),
    ], failures=1)

    with pytest.raises(RuntimeError, match="contains failures"):
        load_complete_tiles(manifest)
    # The failed entry is excluded on its own status; the good date survives.
    assert set(load_tiling_history(tmp_path)) == {("MOD11A1", date(2026, 1, 1))}


def test_history_skips_an_unreadable_report_without_losing_the_others(tmp_path):
    _tiling_report(tmp_path, [
        ("MOD11A1", "2026-01-01", "complete", _four_tiles(tmp_path, "good")),
    ], name="20260101T000000Z.json")
    (tmp_path / "20260201T000000Z.json").write_text("{ not json")

    assert set(load_tiling_history(tmp_path)) == {("MOD11A1", date(2026, 1, 1))}


def test_history_lets_the_newest_report_win_for_a_retiled_date(tmp_path):
    day = date(2026, 1, 1)
    stale = _four_tiles(tmp_path / "stale", "mod11a1")
    _tiling_report(tmp_path, [
        ("MOD11A1", "2026-01-01", "complete", stale),
    ], name="20260101T000000Z.json")
    fresh = _four_tiles(tmp_path / "fresh", "mod11a1")
    _tiling_report(tmp_path, [
        ("MOD11A1", "2026-01-01", "complete", fresh),
    ], name="20260601T000000Z.json")

    entries = load_tiling_history(tmp_path)[("MOD11A1", day)]
    assert [entry.path for entry in entries] == fresh


def test_purge_date_dry_run_leaves_disk_untouched(tmp_path):
    directory = tmp_path / "raw" / "MOD11A1" / "2026-01-01"
    directory.mkdir(parents=True)
    (directory / "granule1.hdf").write_bytes(b"x" * 100)
    (directory / "granule2.hdf").write_bytes(b"y" * 50)

    count, reclaimed = purge_date(directory, execute=False)
    assert (count, reclaimed) == (2, 150)
    assert sorted(directory.glob("*.hdf"))


def test_purge_date_execute_removes_granules_and_empty_directory(tmp_path):
    directory = tmp_path / "raw" / "MOD11A1" / "2026-01-01"
    directory.mkdir(parents=True)
    (directory / "granule1.hdf").write_bytes(b"x" * 100)

    count, reclaimed = purge_date(directory, execute=True)
    assert (count, reclaimed) == (1, 100)
    assert not directory.exists()


def test_main_end_to_end_respects_retention_window(tmp_path, monkeypatch):
    monkeypatch.setattr("wildfire.modis.purge.paths.WORK", str(tmp_path / "work"))
    old_day = date.today() - timedelta(days=200)
    recent_day = date.today() - timedelta(days=10)
    manifest = _tiling_report(tmp_path, [
        ("MOD11A1", old_day.isoformat(), "complete", _four_tiles(tmp_path, "old")),
        ("MOD11A1", recent_day.isoformat(), "complete", _four_tiles(tmp_path, "recent")),
    ])
    raw_root = tmp_path / "raw"
    old_dir = raw_date_dir(raw_root, "MOD11A1", old_day)
    recent_dir = raw_date_dir(raw_root, "MOD11A1", recent_day)
    for directory in (old_dir, recent_dir):
        directory.mkdir(parents=True)
        (directory / "granule.hdf").write_bytes(b"x" * 10)

    main(
        tiling_manifest=manifest, raw_root=raw_root,
        retention_days=90, deep_validate=False, execute=True,
    )


    assert not old_dir.exists()
    assert recent_dir.exists()
    assert (recent_dir / "granule.hdf").is_file()
