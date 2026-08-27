import json
import os
import signal
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import pytest

from modis.preprocess import (
    Processor,
    Terminated,
    TileEntry,
    available_platform_groups,
    complete_product_dates,
    load_complete_tiles,
    sha256,
    terminate_group,
    terminate_on_signal,
)
from modis.processing_registry import ROUTES


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def test_terminate_group_reaps_the_workers_the_child_spawned(tmp_path):
    """The FireMPC scripts fan out; killing only the child leaves the fan-out."""
    marker = tmp_path / "worker.pid"
    script = (
        "import subprocess, sys, time\n"
        "worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        f"open({str(marker)!r}, 'w').write(str(worker.pid))\n"
        "time.sleep(120)\n"
    )
    process = subprocess.Popen([sys.executable, "-c", script], start_new_session=True)
    try:
        assert _wait_until(marker.exists), "child never started its worker"
        worker_pid = int(marker.read_text())
        assert _alive(worker_pid)

        terminate_group(process)

        assert process.poll() is not None
        assert _wait_until(lambda: not _alive(worker_pid)), (
            f"worker {worker_pid} survived the group kill"
        )
    finally:
        if process.poll() is None:  # pragma: no cover - only on assertion failure
            terminate_group(process)


def test_terminate_on_signal_turns_sigterm_into_an_exception():
    """SIGTERM's default disposition skips `finally`, so the lock outlives the run."""
    with pytest.raises(Terminated, match="SIGTERM"), terminate_on_signal():
        os.kill(os.getpid(), signal.SIGTERM)


def test_terminate_on_signal_restores_the_previous_handlers():
    before = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
    with terminate_on_signal():
        pass
    assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)) == before


def test_partial_tiling_entries_never_become_preprocess_inputs(tmp_path):
    paths = []
    for tile in range(4):
        path = tmp_path / f"tile{tile}.tif"
        path.touch()
        paths.append(path)
    report = {
        "summary": {"failures": 0},
        "results": [
            {
                "product": "MCD15A3H", "date": "2026-01-01", "status": "complete",
                "tiles": {
                    str(tile): {"status": "written", "path": str(path)}
                    for tile, path in enumerate(paths)
                },
            },
            {
                "product": "MCD15A3H", "date": "2026-01-05", "status": "failed",
                "tiles": {},
            },
        ],
    }
    manifest = tmp_path / "tiling.json"
    manifest.write_text(json.dumps(report))
    complete = load_complete_tiles(manifest)
    assert complete_product_dates("MCD15A3H", complete) == (date(2026, 1, 1),)


def _processor(tmp_path: Path) -> Processor:
    return Processor(
        data_root=tmp_path / "processed",
        report_root=tmp_path / "reports",
        template=tmp_path / "template.tif",
        firempc_python=tmp_path / "python",
        resume=True,
        overwrite=False,
    )


def test_spatial_fill_rejects_cold_start(tmp_path):
    processor = _processor(tmp_path)
    masked = processor.stage_dir("qa_combined", "MCD09GA")
    masked.mkdir(parents=True)
    for tile in range(4):
        (masked / f"MCD09GA_2026_08_02_tile{tile:02d}.tif").touch()

    with pytest.raises(RuntimeError, match="bootstrap state is required"):
        processor.stage_spatial_filled("MCD09GA", date(2026, 8, 2), {})


def test_resume_rejects_spatial_manifest_without_predecessor(tmp_path):
    processor = _processor(tmp_path)
    output = tmp_path / "2026_08_02_tile00.tif"
    output.write_bytes(b"invalid cold-start state")
    manifest = {
        "status": "complete",
        "outputs": [{"path": str(output), "sha256": sha256(output)}],
        "predecessor": None,
    }

    assert not processor._resumable_manifest(
        manifest, "MCD09GA", date(2026, 8, 2), "spatial_filled"
    )


def test_resume_reprocesses_daily_filled_once_a_closer_source_lands(tmp_path):
    processor = _processor(tmp_path)
    unified = processor.stage_dir("unified", "MCD09GA")
    unified.mkdir(parents=True)
    (unified / "MCD09GA_2026_08_08.tif").write_bytes(b"stale unified raster")
    day = date(2026, 8, 9)

    inputs, outputs, command = processor.stage_daily_filled("MCD09GA", day, {})
    manifest = {
        "status": "complete",
        "inputs": [{"path": str(p), "sha256": sha256(p)} for p in inputs],
        "outputs": [{"path": str(p), "sha256": sha256(p)} for p in outputs],
        "command": command,
    }
    assert processor._resumable_manifest(manifest, "MCD09GA", day, "daily_filled")

    # The real 08-09 source lands after the manifest was recorded.
    (unified / "MCD09GA_2026_08_09.tif").write_bytes(b"real unified raster")
    assert not processor._resumable_manifest(manifest, "MCD09GA", day, "daily_filled")


def test_resume_reprocesses_scaled_once_its_input_content_changes(tmp_path):
    processor = _processor(tmp_path)
    daily_filled = processor.stage_dir("daily_filled", "MCD09CMG")
    daily_filled.mkdir(parents=True)
    source = daily_filled / "2026_08_09.tif"
    source.write_bytes(b"original daily-filled content")
    scaled = processor.stage_dir("scaled", "MCD09CMG")
    scaled.mkdir(parents=True)
    output = scaled / "2026_08_09.tif"
    output.write_bytes(b"scaled output, unrelated to the resume check")
    manifest = {
        "status": "complete",
        "inputs": [{"path": str(source), "sha256": sha256(source)}],
        "outputs": [{"path": str(output), "sha256": sha256(output)}],
    }
    assert processor._resumable_manifest(
        manifest, "MCD09CMG", date(2026, 8, 9), "scaled"
    )

    # daily_filled is reprocessed and rewrites the same path with new content.
    source.write_bytes(b"reprocessed daily-filled content")
    assert not processor._resumable_manifest(
        manifest, "MCD09CMG", date(2026, 8, 9), "scaled"
    )


def _write_raster(path, values, nodata=-32768.0):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    array = np.asarray(values, dtype="float32")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
        # No CRS: the envelope gate is purely a value check, and resolving an
        # EPSG code needs a PROJ database this test does not otherwise require.
        count=1, dtype="float32", nodata=nodata, transform=from_origin(0, 90, 1, 1),
    ) as dataset:
        dataset.write(array, 1)
    return path


def test_mcd12q1_historical_route_extracts_land_cover_without_qc_filter(
    tmp_path, monkeypatch,
):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    monkeypatch.setattr("modis.preprocess.paths.WORK", str(tmp_path / "work"))
    processor = _processor(tmp_path)
    day = date(2024, 1, 1)
    land = np.array([[17, 10], [0, 1]], dtype="uint8")
    # Every nonzero QC value would have been erased by the incorrect route.
    qc = np.array([[2, 3], [4, 0]], dtype="uint8")
    entries = []
    for tile in range(4):
        path = tmp_path / f"MCD12Q1_2024_01_01_tile{tile:02d}.tif"
        with rasterio.open(
            path, "w", driver="GTiff", height=2, width=2, count=2,
            dtype="uint8", nodata=0, transform=from_origin(0, 2, 1, 1),
        ) as dataset:
            dataset.write(land, 1)
            dataset.write(qc, 2)
        entries.append(TileEntry("MCD12Q1", day, tile, path))

    inputs, outputs, command = processor.stage_masked(
        "MCD12Q1", day, {("MCD12Q1", day): tuple(entries)}
    )

    assert len(inputs) == len(outputs) == 4
    assert command == ["native", "mcd12q1_retain_lc_type1_band_only"]
    for output in outputs:
        with rasterio.open(output) as dataset:
            assert dataset.count == 1
            assert dataset.nodata == 0
            assert np.array_equal(dataset.read(1), land)


def test_geometry_stage_flags_values_invented_outside_the_input_envelope(tmp_path):
    """A resample cannot produce a value its own input never held.

    This is the nodata-blending failure: bilinear alignment mixes the nodata
    fill into partially observed output pixels, so the aligned raster carries
    large negatives that scaling then clips silently back into range.
    """
    processor = _processor(tmp_path)
    processor.range_tolerance = 0.001  # the default is record-only
    source = _write_raster(tmp_path / "in.tif", [[10.0, 20.0], [30.0, -32768.0]])
    blended = _write_raster(tmp_path / "out.tif", [[10.0, 20.0], [30.0, -31538.9]])

    report = processor._range_report("MCD09GA", "unified", [source], [blended])
    assert report[0]["input_envelope"] == [10.0, 30.0]
    assert report[0]["outside_envelope_fraction"] == 0.25

    with pytest.raises(RuntimeError, match="left the input envelope"):
        processor._enforce_range("MCD09GA", date(2026, 8, 2), "unified", report)


def test_clean_resample_passes_the_envelope_gate(tmp_path):
    processor = _processor(tmp_path)
    source = _write_raster(tmp_path / "in.tif", [[10.0, 20.0], [30.0, -32768.0]])
    averaged = _write_raster(tmp_path / "out.tif", [[15.0, 25.0], [20.0, -32768.0]])

    report = processor._range_report("MCD09GA", "unified", [source], [averaged])
    assert report[0]["outside_envelope_fraction"] == 0.0
    processor._enforce_range("MCD09GA", date(2026, 8, 2), "unified", report)


def test_non_geometry_stages_record_ranges_without_enforcing_them(tmp_path):
    """Masking and index maths legitimately leave their input's envelope."""
    processor = _processor(tmp_path)
    source = _write_raster(tmp_path / "in.tif", [[10.0, 20.0]])
    computed = _write_raster(tmp_path / "out.tif", [[0.9, -0.4]])

    report = processor._range_report("NDVIEVI", "vegetation_indices", [source], [computed])
    assert "outside_envelope_fraction" not in report[0]
    assert report[0]["min"] == pytest.approx(-0.4)


def test_bootstrap_allows_a_cold_start_only_when_requested(tmp_path):
    processor = _processor(tmp_path)
    processor.bootstrap = True
    masked = processor.stage_dir("qa_combined", "MCD09GA")
    masked.mkdir(parents=True)
    for tile in range(4):
        (masked / f"MCD09GA_2026_08_02_tile{tile:02d}.tif").touch()

    # The cold start is permitted, so it fails later -- running the fill -- and
    # no longer at the predecessor check.
    with pytest.raises(Exception) as excinfo:
        processor.stage_spatial_filled("MCD09GA", date(2026, 8, 2), {})
    assert "bootstrap state is required" not in str(excinfo.value)


def test_mcd15a3h_daily_fill_uses_latest_four_day_composite(tmp_path):
    """A calendar day without an MCD15 observation uses its preceding anchor."""
    processor = _processor(tmp_path)
    unified = processor.stage_dir("unified", "MCD15A3H")
    unified.mkdir(parents=True)
    anchor = unified / "2025_12_23.tif"
    anchor.write_bytes(b"processed December 23 composite")

    inputs, outputs, command = processor.stage_daily_filled(
        "MCD15A3H", date(2025, 12, 25), {},
    )

    assert inputs == [anchor]
    assert outputs[0].name == "2025_12_25.tif"
    assert outputs[0].read_bytes() == anchor.read_bytes()
    assert command == ["atomic-copy", "2025-12-23", "2025-12-25"]


def _state(processor, product, day, tiles=4):
    folder = processor.stage_dir("spatial_filled", product)
    folder.mkdir(parents=True, exist_ok=True)
    made = []
    for tile in range(tiles):
        path = folder / f"{day.strftime('%Y_%m_%d')}_tile{tile:02d}.tif"
        path.write_bytes(b"state" + str(tile).encode())
        made.append(path)
    return made


def test_bridge_carries_state_across_a_date_that_was_never_published(tmp_path):
    processor = _processor(tmp_path)
    processor.bridge_missing = True
    _state(processor, "MCD09GA", date(2026, 7, 23))
    tiles = sorted(processor.stage_dir("spatial_filled", "MCD09GA").glob("*.tif"))

    predecessor, carried = processor._bridge_missing(
        "MCD09GA", date(2026, 7, 23), tiles, date(2026, 7, 25), {},
    )

    assert predecessor == date(2026, 7, 24)
    assert [path.name for path in carried] == [
        f"2026_07_24_tile{tile:02d}.tif" for tile in range(4)
    ]
    assert carried[0].read_bytes() == tiles[0].read_bytes()

    manifest = json.loads(
        (processor._manifest_path("MCD09GA", date(2026, 7, 24), "spatial_filled")).read_text()
    )
    assert manifest["carried_forward"] is True
    assert manifest["carried_from"] == "2026-07-23"


def test_bridge_refuses_a_date_that_has_complete_source_data(tmp_path):
    """A processable gap is a processing failure, not a publication hole."""
    from modis.preprocess import TileEntry

    processor = _processor(tmp_path)
    processor.bridge_missing = True
    _state(processor, "MCD09GA", date(2026, 7, 23))
    tiles = sorted(processor.stage_dir("spatial_filled", "MCD09GA").glob("*.tif"))
    source_tiles = {
        (export, date(2026, 7, 24)): tuple(
            TileEntry(export, date(2026, 7, 24), tile, tmp_path / f"{export}{tile}.tif")
            for tile in range(4)
        )
        for export in ROUTES["MCD09GA"].exports
    }

    with pytest.raises(RuntimeError, match="process that date instead of bridging"):
        processor._bridge_missing(
            "MCD09GA", date(2026, 7, 23), tiles, date(2026, 7, 25), source_tiles,
        )


def test_gap_still_raises_without_the_bridge_flag(tmp_path):
    processor = _processor(tmp_path)
    masked = processor.stage_dir("qa_combined", "MCD09GA")
    masked.mkdir(parents=True)
    for tile in range(4):
        (masked / f"MCD09GA_2026_07_25_tile{tile:02d}.tif").touch()
    _state(processor, "MCD09GA", date(2026, 7, 23))

    with pytest.raises(RuntimeError, match="refusing to advance"):
        processor.stage_spatial_filled("MCD09GA", date(2026, 7, 25), {})


def _entries(exports, day):
    return {
        (export, day): tuple(
            TileEntry(export, day, tile, Path(f"/nonexistent/{export}_{tile}.tif"))
            for tile in range(4)
        )
        for export in exports
    }


def test_one_complete_platform_is_enough_to_process_a_date():
    """Terra-complete days with an Aqua hole are ordinary in the archive.

    The reference published 879 single-platform days, so requiring every export
    would discard data FireMPC processed.
    """
    day = date(2026, 8, 9)
    terra_only = _entries(("MOD09GA_b1237", "MOD09GA_QA_state1km"), day)

    assert available_platform_groups("MCD09GA", day, terra_only) == (
        ("MOD09GA_b1237", "MOD09GA_QA_state1km"),
    )
    assert complete_product_dates("MCD09GA", terra_only) == (day,)


def test_aqua_only_dates_are_equally_processable():
    day = date(2003, 12, 18)
    aqua_only = _entries(("MYD11A1",), day)

    assert complete_product_dates("MCD11A1", aqua_only) == (day,)


def test_half_a_platform_is_not_enough():
    """Reflectance without its QA cannot be masked, so it must not qualify."""
    day = date(2026, 8, 9)
    no_qa = _entries(("MOD09GA_b1237",), day)

    assert available_platform_groups("MCD09GA", day, no_qa) == ()
    assert complete_product_dates("MCD09GA", no_qa) == ()


def test_single_group_routes_still_require_every_export():
    day = date(2026, 8, 5)
    assert complete_product_dates("MCD15A3H", _entries(("MCD15A3H",), day)) == (day,)
    assert complete_product_dates("MCD15A3H", {}) == ()


def test_masker_refuses_a_date_with_no_complete_platform(tmp_path):
    processor = _processor(tmp_path)
    day = date(2026, 8, 9)

    with pytest.raises(RuntimeError, match="no complete platform"):
        processor.stage_masked("MCD09GA", day, _entries(("MOD09GA_b1237",), day))


def test_nan_in_the_input_does_not_make_the_envelope_gate_vacuous(tmp_path):
    """A NaN-poisoned envelope silently passed everything.

    The two-platform MCD11A1 combiner writes NaN with no nodata tag, so
    min/max came back (nan, nan) and every `outside` comparison was False --
    the gate reported 0.0 on rasters it had not actually checked.
    """
    import numpy as np

    processor = _processor(tmp_path)
    processor.range_tolerance = 0.001
    source = _write_raster(tmp_path / "in.tif", [[10.0, 20.0], [30.0, np.nan]], nodata=None)
    blended = _write_raster(tmp_path / "out.tif", [[10.0, 20.0], [30.0, -31538.9]], nodata=None)

    report = processor._range_report("MCD09GA", "unified", [source], [blended])
    assert report[0]["input_envelope"] == [10.0, 30.0]
    assert report[0]["outside_envelope_fraction"] == 0.25

    with pytest.raises(RuntimeError, match="left the input envelope"):
        processor._enforce_range("MCD09GA", date(2026, 8, 15), "unified", report)


def test_the_routes_own_invalid_sentinel_is_not_an_envelope_violation(tmp_path):
    """Downsampling writes the route's invalid value where its input was invalid.

    The reference tree is 22.6% -32768 on an ordinary MCD11A1 date, dominated
    by permanently invalid ocean, so counting the sentinel would fail every
    date rather than catch anything.
    """
    processor = _processor(tmp_path)
    processor.range_tolerance = 0.001
    source = _write_raster(tmp_path / "in.tif", [[240.0, 250.0], [32771.0, 0.0]], nodata=None)
    downsampled = _write_raster(tmp_path / "out.tif", [[245.0, -32768.0], [-32768.0, -32768.0]], nodata=None)

    report = processor._range_report("MCD11A1", "downsampled", [source], [downsampled])
    assert report[0]["input_envelope"] == [240.0, 250.0]
    assert report[0]["outside_envelope_fraction"] == 0.0
    assert report[0]["invalid_fraction"] == 0.75
    processor._enforce_range("MCD11A1", date(2026, 8, 15), "downsampled", report)


def test_single_platform_dates_are_held_to_the_measured_looser_tolerance(tmp_path):
    """The warp blends far more on a date only one platform observed.

    Measured at 0.0309 on both the reference's Aqua-only 2003-12-18 and our
    Terra-only 2026-08-15, against 0.0005-0.0007 on two-platform dates.
    """
    processor = _processor(tmp_path)
    processor.range_tolerance = 0.001
    day = date(2026, 8, 15)
    report = [{
        "path": "u.tif", "band": 3, "min": -26166.1, "max": 251.9,
        "input_envelope": [237.4, 251.9], "outside_envelope_fraction": 0.030891,
    }]

    # Without the masker's finding it is an ordinary date, and 3.09% fails.
    with pytest.raises(RuntimeError, match="left the input envelope"):
        processor._enforce_range("MCD11A1", day, "unified", report)

    manifest = processor._manifest_path("MCD11A1", day, "masked")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"single_platform": True, "platforms": ["MOD"]}))

    processor._enforce_range("MCD11A1", day, "unified", report)


def test_the_looser_tolerance_still_catches_a_gross_violation(tmp_path):
    processor = _processor(tmp_path)
    processor.range_tolerance = 0.001
    day = date(2026, 8, 15)
    manifest = processor._manifest_path("MCD11A1", day, "masked")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"single_platform": True}))
    report = [{
        "path": "u.tif", "band": 3, "min": -26166.1, "max": 251.9,
        "input_envelope": [237.4, 251.9], "outside_envelope_fraction": 0.40,
    }]

    with pytest.raises(RuntimeError, match="left the input envelope"):
        processor._enforce_range("MCD11A1", day, "unified", report)
