"""The orchestration guarantees: recursion safety and stage completion."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import rasterio

from wildfire.weather import derive as derive
from wildfire.weather import preprocess as wp
from wildfire.weather import registry as registry
from tests.weather.synthetic import write_ifs_tile, write_source_tile


def make_processor(tmp_path: Path, **overrides) -> wp.Processor:
    settings = dict(
        drivers=tmp_path / "drivers",
        data_root=tmp_path / "processed",
        report_root=tmp_path / "reports",
        state_root=tmp_path / "state",
        template=tmp_path / "template.tif",
        firempc_python=Path("/nonexistent/python"),
    )
    settings.update(overrides)
    return wp.Processor(**settings)


def stage_ifs_day(processor: wp.Processor, day: date, values: dict[str, float]) -> None:
    folder = processor.drivers / "IFS"
    folder.mkdir(parents=True, exist_ok=True)
    for tile in registry.TILES:
        write_ifs_tile(folder / f"{registry.ifs_stem(day, tile)}.tif", values,
                       origin=(-141.0 + 2.0 * (tile % 2), 64.0 - 2.0 * (tile // 2)))


DRY_WARM_DAY = {
    "temperature_2m": 295.0,
    "dewpoint_temperature_2m": 283.15,
    "temperature_2m_max": 298.15,
    "u_component_of_wind_10m": 3.0,
    "v_component_of_wind_10m": 4.0,
    "total_precipitation_sum": 0.0,
}


# ---------------------------------------------------------------------------
# VPD
# ---------------------------------------------------------------------------
def test_vpd_derives_one_raster_per_tile(tmp_path):
    processor = make_processor(tmp_path)
    day = date(2026, 7, 1)
    stage_ifs_day(processor, day, DRY_WARM_DAY)

    processor.run_stage("VPD", day, "derived")
    outputs = sorted(processor.stage_dir("derived", "VPD").glob("*.tif"))
    assert [path.name for path in outputs] == [
        f"VPD_2026_07_01_tile{tile:02d}.tif" for tile in registry.TILES
    ]
    with rasterio.open(outputs[0]) as dataset:
        assert dataset.count == 1
        assert dataset.descriptions == registry.VPD_BAND_NAMES
        assert np.all(dataset.read([1])[0] > 0)


def test_vpd_refuses_an_incomplete_day(tmp_path):
    processor = make_processor(tmp_path)
    day = date(2026, 7, 1)
    stage_ifs_day(processor, day, DRY_WARM_DAY)
    (processor.drivers / "IFS" / f"{registry.ifs_stem(day, 2)}.tif").unlink()

    with pytest.raises(RuntimeError, match=r"ifs tiles \[2\] are absent"):
        processor.run_stage("VPD", day, "derived")
    manifest = json.loads((processor.report_root / "VPD" / day.isoformat() / "derived.json").read_text())
    assert manifest["status"] == "failed"


# ---------------------------------------------------------------------------
# FWI recursion safety
# ---------------------------------------------------------------------------
def test_fwi_cold_start_is_recorded_and_advances_the_checkpoint(tmp_path):
    processor = make_processor(tmp_path)
    day = date(2026, 7, 1)
    stage_ifs_day(processor, day, DRY_WARM_DAY)

    processor.run_stage("FWI", day, "derived")
    manifest = json.loads((processor.report_root / "FWI" / day.isoformat() / "derived.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["cold_start"] is True
    assert manifest["cold_start_values"] == {k: float(v) for k, v in registry.FWI_INIT.items()}

    state = derive.FWIState.load(tmp_path / "state" / "fwi_state.npz")
    assert state.day == day
    output = processor.stage_dir("derived", "FWI") / "2026_07_01.tif"
    with rasterio.open(output) as dataset:
        assert dataset.count == 6
        assert dataset.descriptions == registry.FWI_BAND_NAMES


def test_fwi_refuses_to_advance_across_a_gap(tmp_path):
    """Skipping a day silently changes every later drought code."""
    processor = make_processor(tmp_path)
    first = date(2026, 7, 1)
    stage_ifs_day(processor, first, DRY_WARM_DAY)
    processor.run_stage("FWI", first, "derived")

    later = date(2026, 7, 5)
    stage_ifs_day(processor, later, DRY_WARM_DAY)
    with pytest.raises(RuntimeError, match="3-day gap"):
        processor.run_stage("FWI", later, "derived")
    assert derive.FWIState.load(tmp_path / "state" / "fwi_state.npz").day == first


def test_fwi_bridges_a_gap_only_when_told_to_and_records_it(tmp_path):
    processor = make_processor(tmp_path, bridge_missing=True)
    first = date(2026, 7, 1)
    stage_ifs_day(processor, first, DRY_WARM_DAY)
    processor.run_stage("FWI", first, "derived")

    later = date(2026, 7, 5)
    stage_ifs_day(processor, later, DRY_WARM_DAY)
    processor.run_stage("FWI", later, "derived")

    manifest = json.loads((processor.report_root / "FWI" / later.isoformat() / "derived.json").read_text())
    assert manifest["bridged_days"] == 3
    assert manifest["bridged_from"] == first.isoformat()


def test_fwi_refuses_to_reprocess_a_date_already_in_the_state(tmp_path):
    """Re-running a folded-in date would double-count it into the recursion."""
    processor = make_processor(tmp_path)
    day = date(2026, 7, 1)
    stage_ifs_day(processor, day, DRY_WARM_DAY)
    processor.run_stage("FWI", day, "derived")

    with pytest.raises(RuntimeError, match="only moves forward"):
        processor.run_stage("FWI", day, "derived")


def test_fwi_accumulates_drought_across_consecutive_days(tmp_path):
    processor = make_processor(tmp_path)
    codes = []
    for offset in range(3):
        day = date(2026, 7, 1 + offset)
        stage_ifs_day(processor, day, DRY_WARM_DAY)
        processor.run_stage("FWI", day, "derived")
        codes.append(derive.FWIState.load(tmp_path / "state" / "fwi_state.npz").state[0]["dc"].mean())
    assert codes[0] < codes[1] < codes[2]


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------
def test_stage_is_done_for_fwi_follows_the_checkpoint_not_the_file(tmp_path):
    processor = make_processor(tmp_path)
    day = date(2026, 7, 1)
    stage_ifs_day(processor, day, DRY_WARM_DAY)
    assert not processor.stage_is_done("FWI", day, "derived")

    processor.run_stage("FWI", day, "derived")
    assert processor.stage_is_done("FWI", day, "derived")

    # Removing the raster does not make the date unprocessed: the state has
    # already absorbed it, so re-deriving would double-count.
    (processor.stage_dir("derived", "FWI") / "2026_07_01.tif").unlink()
    assert processor.stage_is_done("FWI", day, "derived")


def test_stage_is_done_for_vpd_requires_all_four_tiles(tmp_path):
    processor = make_processor(tmp_path)
    day = date(2026, 7, 1)
    stage_ifs_day(processor, day, DRY_WARM_DAY)
    processor.run_stage("VPD", day, "derived")
    assert processor.stage_is_done("VPD", day, "derived")

    (processor.stage_dir("derived", "VPD") / "VPD_2026_07_01_tile03.tif").unlink()
    assert not processor.stage_is_done("VPD", day, "derived")


def test_merged_stage_rejects_fwi(tmp_path):
    """FWI mosaics inside its derivation; merging it again would double-warp."""
    processor = make_processor(tmp_path)
    with pytest.raises(RuntimeError, match="already national"):
        processor.stage_merged("FWI", date(2026, 7, 1))


def test_manifest_records_band_ranges(tmp_path):
    processor = make_processor(tmp_path)
    day = date(2026, 7, 1)
    stage_ifs_day(processor, day, DRY_WARM_DAY)
    processor.run_stage("VPD", day, "derived")

    manifest = json.loads((processor.report_root / "VPD" / day.isoformat() / "derived.json").read_text())
    assert manifest["registry_version"] == registry.REGISTRY_VERSION
    assert len(manifest["band_ranges"]) == len(registry.TILES)
    assert all(entry["min"] is not None for entry in manifest["band_ranges"])
    assert len(manifest["inputs"]) == len(registry.TILES)


# ---------------------------------------------------------------------------
# Source switching
# ---------------------------------------------------------------------------
def stage_source_day(processor: wp.Processor, source, day: date,
                     values: dict[str, float]) -> None:
    folder = processor.drivers / source.prefix
    folder.mkdir(parents=True, exist_ok=True)
    for tile in registry.TILES:
        write_source_tile(folder / f"{source.stem(day, tile)}.tif", source, values,
                          origin=(-141.0 + 2.0 * (tile % 2), 64.0 - 2.0 * (tile // 2)),
                          name_bands=True)


def test_the_date_chooses_the_source(tmp_path):
    """ERA5-Land for the spin-up leg, IFS from its first cycle on."""
    processor = make_processor(tmp_path)
    historical = date(2023, 6, 15)
    operational = date(2026, 7, 1)
    stage_source_day(processor, registry.ERA5, historical, DRY_WARM_DAY)
    stage_source_day(processor, registry.IFS, operational, DRY_WARM_DAY)

    assert processor.input_tiles(historical, "FWI")[0] is registry.ERA5
    assert processor.input_tiles(operational, "FWI")[0] is registry.IFS


def test_a_wrong_source_on_disk_is_an_error_that_names_the_fix(tmp_path):
    """Silently substituting would put an unrecorded break in the recursion."""
    processor = make_processor(tmp_path)
    day = date(2023, 6, 15)  # ERA5's date...
    stage_source_day(processor, registry.IFS, day, DRY_WARM_DAY)  # ...with IFS tiles

    with pytest.raises(RuntimeError, match="pass --source ifs"):
        processor.input_tiles(day, "FWI")

    forced = make_processor(tmp_path, source="ifs")
    assert forced.input_tiles(day, "FWI")[0] is registry.IFS


def test_the_source_is_recorded_in_every_manifest(tmp_path):
    processor = make_processor(tmp_path)
    day = date(2023, 6, 15)
    stage_source_day(processor, registry.ERA5, day, DRY_WARM_DAY)

    processor.run_stage("VPD", day, "derived")
    processor.run_stage("FWI", day, "derived")
    for product in ("VPD", "FWI"):
        manifest = json.loads(processor._manifest_path(product, day, "derived").read_text())
        assert manifest["weather_source"] == "era5"


# ---------------------------------------------------------------------------
# State-only
# ---------------------------------------------------------------------------
def test_state_only_advances_the_recursion_without_writing_a_raster(tmp_path):
    """The historical leg needs the moisture codes, not ~47 MB a day of output."""
    processor = make_processor(tmp_path, state_only=True)
    day = date(2023, 6, 15)
    stage_source_day(processor, registry.ERA5, day, DRY_WARM_DAY)

    processor.run_stage("FWI", day, "derived")

    assert sorted(processor.stage_dir("derived", "FWI").glob("*.tif")) == []
    state = derive.FWIState.load(processor._fwi_state_path())
    assert state.day == day
    assert np.all(state.state[0]["dc"] > registry.FWI_INIT["dc"])

    manifest = json.loads(processor._manifest_path("FWI", day, "derived").read_text())
    assert manifest["state_only"] is True
    assert manifest["outputs"] == []
    assert manifest["status"] == "complete"


def test_state_only_still_blocks_a_replay_of_the_same_date(tmp_path):
    """No raster to check means the checkpoint is the only guard left."""
    processor = make_processor(tmp_path, state_only=True)
    day = date(2023, 6, 15)
    stage_source_day(processor, registry.ERA5, day, DRY_WARM_DAY)

    processor.run_stage("FWI", day, "derived")
    assert processor.stage_is_done("FWI", day, "derived")
    with pytest.raises(RuntimeError, match="checkpoint already stands"):
        processor.run_stage("FWI", day, "derived")


def test_a_state_only_leg_hands_its_codes_to_a_normal_run(tmp_path):
    """The changeover has to carry state across, not restart at the constants."""
    processor = make_processor(tmp_path, state_only=True)
    first = date(2023, 6, 15)
    stage_source_day(processor, registry.ERA5, first, DRY_WARM_DAY)
    processor.run_stage("FWI", first, "derived")
    carried = derive.FWIState.load(processor._fwi_state_path()).state[0]["dc"].copy()

    second = date(2023, 6, 16)
    writing = make_processor(tmp_path)
    stage_source_day(writing, registry.ERA5, second, DRY_WARM_DAY)
    writing.run_stage("FWI", second, "derived")

    state = derive.FWIState.load(writing._fwi_state_path())
    assert state.day == second
    assert np.all(state.state[0]["dc"] > carried)
    assert len(sorted(writing.stage_dir("derived", "FWI").glob("*.tif"))) == 1


def test_the_frozen_fraction_is_recorded(tmp_path):
    """A masked pixel holds yesterday's code, which no raster shows."""
    processor = make_processor(tmp_path)
    day = date(2026, 7, 1)
    stage_ifs_day(processor, day, DRY_WARM_DAY)
    folder = processor.drivers / "IFS"
    path = folder / f"{registry.ifs_stem(day, 0)}.tif"
    with rasterio.open(path, "r+") as dataset:
        index = registry.IFS_BAND_INDEX["temperature_2m_max"]
        band = dataset.read([index])[0]
        band[0, :] = registry.FWI_OUTSIDE_SENTINEL
        dataset.write(band, index)

    processor.run_stage("FWI", day, "derived")
    manifest = json.loads(processor._manifest_path("FWI", day, "derived").read_text())
    assert manifest["frozen_fraction"]["0"] == pytest.approx(1.0 / 6.0, abs=1e-6)
    assert manifest["frozen_fraction"]["1"] == 0.0
