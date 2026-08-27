"""The weather contract is data, and these are the invariants it must hold."""

from __future__ import annotations

import ast
from datetime import date, timedelta
from pathlib import Path

import pytest

import vendorpaths
from weather import registry as registry

ROOT = vendorpaths.OPERATIONAL_ROOT
FIREMPC = vendorpaths.FIREMPC_ROOT
FIREMPC_FWI_MODULE = vendorpaths.FIREMPC_FWI / "fwi_module.py"


def firempc_source(relative: str) -> str:
    """Read a file out of the FireMPC checkout, or skip.

    These tests are drift guards against a reference tree that is gitignored,
    so on a fresh clone there is nothing to guard against and skipping is the
    honest result -- not a failure, and not a silent pass either.
    """
    path = FIREMPC / relative
    if not path.is_file():
        pytest.skip(f"{path} absent; the FireMPC checkout is gitignored")
    return path.read_text()


def test_ifs_bands_are_dense_and_ordered():
    assert [band.index for band in registry.IFS_BANDS] == list(range(1, 24))
    assert len(set(registry.IFS_BAND_NAMES)) == len(registry.IFS_BAND_NAMES)
    assert len({band.source for band in registry.IFS_BANDS}) == len(registry.IFS_BANDS)


def test_ifs_covers_thirteen_of_the_eighteen_era5_channels():
    """The whole point of the missing-band policy hinges on this split."""
    era5_named = {band.name for band in registry.IFS_BANDS if band.era5}
    slot = set(registry.ERA5_SLOT_BANDS)
    assert era5_named < slot
    assert len(era5_named) == 13
    assert slot - era5_named == set(registry.MISSING_ERA5_BANDS)
    assert len(registry.MISSING_ERA5_BANDS) == 5


def test_era5_slot_matches_the_model_channel_names():
    """`dataload_h5.CHANNEL_NAMES[1:19]` is the order the trained model reads."""
    source = firempc_source("../CanFireCast4AAAI-AISI/dataload_h5.py")
    tree = ast.parse(source)
    names = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "CHANNEL_NAMES":
            names = [ast.literal_eval(element) for element in node.value.elts]
            break
    assert names is not None, "CHANNEL_NAMES not found in dataload_h5.py"

    def normalise(text: str) -> str:
        return text.lower().replace(" ", "_")

    assert [normalise(name) for name in names[1:19]] == list(registry.ERA5_SLOT_BANDS)
    assert [normalise(name) for name in names[19:25]] == [
        name.lower() for name in registry.FWI_BAND_NAMES
    ]
    assert normalise(names[25]) == "vpd"


def test_reductions_are_known_and_accumulators_use_the_last_step():
    for band in registry.IFS_BANDS:
        assert band.reduction in {"mean", "max", "accumulated"}
    assert registry.ACCUMULATION_HOUR == registry.FORECAST_HOURS[-1]
    # Forecast hour 0 must stay out: its 3-hourly maximum band is a 0 K sentinel.
    assert 0 not in registry.FORECAST_HOURS
    assert registry.STEP_ZERO_IS_SENTINEL


def test_only_soil_temperature_keeps_its_served_kelvin():
    """GEE serves IFS air/dewpoint/skin temperature in Celsius, soil in Kelvin."""
    converted = {band.name for band in registry.IFS_BANDS if band.to_kelvin}
    assert converted == {
        "temperature_2m", "dewpoint_temperature_2m",
        "temperature_2m_max", "skin_temperature_max",
    }
    for band in registry.IFS_BANDS:
        if band.source.startswith("soil_temperature"):
            assert not band.to_kelvin


def _firempc_scale_default(flag: str) -> str:
    """The literal default of one `scale_int_tif.py` band-config argument."""
    source = firempc_source("newset_processing/scale_int_tif.py")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value != flag:
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                return ast.literal_eval(keyword.value)
    raise AssertionError(f"{flag} not found in scale_int_tif.py")


@pytest.mark.parametrize("product,flag", [("VPD", "--vpd-default"), ("FWI", "--fwi-default")])
def test_scale_ranges_are_the_frozen_firempc_values(product, flag):
    """The training cube's uint8 encoding depends on these exact bounds.

    Parsed from the FireMPC source rather than retyped, so a change upstream
    fails the test instead of silently re-quantising the operational product.
    """
    parsed = []
    for entry in _firempc_scale_default(flag).split(","):
        index, _scale, _offset, low, high = (part.strip() for part in entry.split(":"))
        parsed.append((int(index), float(low), float(high)))
    expected = [(index, low, high)
                for index, (low, high) in enumerate(registry.SCALE_RANGES[product])]
    assert parsed == expected


def test_scale_band_config_renders_one_entry_per_band():
    for product, route in registry.ROUTES.items():
        entries = registry.SCALE_BAND_CONFIG[product].split(",")
        assert len(entries) == len(route.band_names)
        assert len(registry.SCALE_RANGES[product]) == len(route.band_names)


def test_fwi_inputs_resolve_to_real_ifs_bands():
    """FireMPC addressed these by ERA5 position; IFS orders them differently."""
    for name in registry.FWI_INPUT_BANDS.values():
        assert name in registry.IFS_BAND_INDEX
    for name in registry.VPD_INPUT_BANDS.values():
        assert name in registry.IFS_BAND_INDEX
    # If IFS ever matched the ERA5 order, name resolution would be redundant.
    assert registry.IFS_BAND_INDEX["temperature_2m_max"] != 13


def test_fwi_startup_matches_the_pinned_top_level_equations():
    tree = ast.parse(FIREMPC_FWI_MODULE.read_text())
    constants = {
        node.target.id: ast.literal_eval(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "").endswith("_INIT")
    }
    assert registry.FWI_INIT == {
        "ffmc": constants["FFMC_INIT"],
        "dmc": constants["DMC_INIT"],
        "dc": constants["DC_INIT"],
    }


def test_route_stages_reflect_where_the_mosaic_happens():
    assert "merged" in registry.ROUTES["VPD"].stages
    # FWI mosaics inside its own derivation, so a merge stage would double-warp.
    assert "merged" not in registry.ROUTES["FWI"].stages
    for route in registry.ROUTES.values():
        assert route.stages[0] == "derived"
        assert route.stages[-1] == "scaled"
        assert "unified" in route.stages


def test_scale_directory_keywords_still_match_firempc_defaults():
    """`scale_int_tif.py` falls back to matching these substrings on the
    directory name; the band config is passed explicitly, but a mismatch here
    would mean the operational directory layout no longer mirrors training."""
    for route in registry.ROUTES.values():
        assert route.scale_keyword in route.output_dir.lower()


def test_ifs_stem_matches_the_historical_archive():
    assert registry.ifs_stem(date(2025, 7, 1), 0) == "IFS_2025_07_01_tile00"
    assert registry.date_token(date(2025, 12, 31)) == "2025_12_31"


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def test_the_two_sources_are_distinguishable_on_disk_and_by_shape():
    prefixes = {source.prefix for source in registry.SOURCES.values()}
    assert len(prefixes) == len(registry.SOURCES)
    counts = {len(source.bands) for source in registry.SOURCES.values()}
    # Band-count detection is only sound while the counts differ; if a third
    # source ever collides, `detect_source` must stop falling back to it.
    assert len(counts) == len(registry.SOURCES)
    for source in registry.SOURCES.values():
        assert [band.index for band in source.bands] == list(range(1, len(source.bands) + 1))
        assert len(set(source.band_names)) == len(source.bands)


def test_era5_source_is_exactly_the_model_slot_in_slot_order():
    """ERA5-Land is what the model was trained on, so it carries all 18."""
    assert registry.ERA5.band_names == registry.ERA5_SLOT_BANDS
    # DAILY_AGGR publishes these names verbatim, so nothing is renamed.
    assert all(band.name == band.source for band in registry.ERA5.bands)
    # And it needs no reduction: one image per day, taken as served.
    assert {band.reduction for band in registry.ERA5.bands} == {"daily"}
    assert not any(band.to_kelvin for band in registry.ERA5.bands)


def test_the_changeover_is_the_first_complete_ifs_cycle():
    """IFS has no archive behind it; before this date only ERA5-Land exists."""
    assert registry.SOURCE_SWITCH_DATE == registry.IFS.first_date
    assert registry.source_for(registry.IFS.first_date) is registry.IFS
    assert registry.source_for(registry.IFS.first_date - timedelta(days=1)) is registry.ERA5
    assert registry.source_for(date(2026, 8, 17)) is registry.IFS


def test_the_spinup_start_is_reachable_only_through_era5():
    """The whole point of the ERA5 route: IFS cannot reach the spin-up start."""
    start = date.fromisoformat(registry.FWI_SPINUP_START)
    assert start < registry.IFS.first_date
    assert registry.ERA5.first_date < start
    assert registry.source_for(start) is registry.ERA5


def test_both_sources_carry_every_band_fwi_reads():
    """FWI must survive the changeover, so neither source may be short a field."""
    for source in registry.SOURCES.values():
        for name in registry.FWI_INPUT_BANDS.values():
            assert name in source.band_index, (source.key, name)


def test_source_lookup_rejects_an_unknown_key():
    with pytest.raises(KeyError, match="unknown weather source"):
        registry.source_by_key("gfs")


def test_tile_stems_keep_the_historical_ifs_naming():
    assert registry.IFS.stem(date(2025, 7, 1), 0) == "IFS_2025_07_01_tile00"
    assert registry.ERA5.stem(date(2023, 1, 1), 3) == "ERA5_2023_01_01_tile03"
    assert registry.ifs_stem(date(2025, 7, 1), 0) == registry.IFS.stem(date(2025, 7, 1), 0)
