"""VPD and FWI derivation, on synthetic IFS tiles."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from weather import derive as derive
from weather import registry as registry

from tests.weather.synthetic import PIXEL, write_ifs_tile, write_source_tile  # noqa: F401


# ---------------------------------------------------------------------------
# VPD
# ---------------------------------------------------------------------------
def test_saturation_vapour_pressure_is_tetens():
    """0.611 * exp(17.27 T / (T + 237.3)), the FireMPC formula, in kPa."""
    for temp_c in (-30.0, 0.0, 20.0, 45.0):
        expected = 0.611 * np.exp(17.27 * temp_c / (temp_c + 237.3))
        assert derive.saturation_vapour_pressure(np.array([temp_c])) == pytest.approx(expected)
    # 0 C is the textbook anchor: ~0.611 kPa.
    assert derive.saturation_vapour_pressure(np.array([0.0]))[0] == pytest.approx(0.611)


def test_vpd_reads_the_named_bands_not_the_era5_positions(tmp_path):
    """Band 1 is air temperature and band 5 is dewpoint in IFS order.

    Reading FireMPC's ERA5 positions (0 and 6) instead would pick up wind and
    the fourth soil layer, and would not raise -- only produce nonsense.
    """
    tile = write_ifs_tile(tmp_path / "IFS_2026_07_01_tile00.tif", {
        "temperature_2m": 300.0,        # 26.85 C
        "dewpoint_temperature_2m": 285.0,  # 11.85 C
        "u_component_of_wind_10m": 999.0,
        "volumetric_soil_water_layer_4": 999.0,
    })
    vpd, profile = derive.vapour_pressure_deficit(tile)
    expected = (derive.saturation_vapour_pressure(np.float32(26.85))
                - derive.saturation_vapour_pressure(np.float32(11.85)))
    assert np.allclose(vpd, expected, atol=1e-4)
    assert profile["width"] == 8


def test_vpd_is_never_negative_and_gates_implausible_input(tmp_path):
    inverted = write_ifs_tile(tmp_path / "a.tif", {
        "temperature_2m": 280.0, "dewpoint_temperature_2m": 295.0,
    })
    vpd, _ = derive.vapour_pressure_deficit(inverted)
    assert np.all(vpd == 0.0)

    absurd = write_ifs_tile(tmp_path / "b.tif", {
        "temperature_2m": 0.0, "dewpoint_temperature_2m": 285.0,
    })
    vpd, _ = derive.vapour_pressure_deficit(absurd)
    assert np.all(np.isnan(vpd))


def test_nodata_becomes_nan_not_minus_infinity(tmp_path):
    path = tmp_path / "c.tif"
    write_ifs_tile(path, {"temperature_2m": 300.0, "dewpoint_temperature_2m": 285.0})
    with rasterio.open(path, "r+") as dataset:
        band = dataset.read([1])[0]
        band[0, 0] = float("-inf")
        dataset.write(band, 1)
    bands = derive.read_bands(path, ["temperature_2m"])
    assert np.isnan(bands["temperature_2m"][0, 0])
    assert np.isfinite(bands["temperature_2m"][1, 1])


def test_read_bands_rejects_a_tile_with_the_wrong_band_count(tmp_path):
    path = tmp_path / "short.tif"
    with rasterio.open(path, "w", driver="GTiff", dtype="float32", count=13,
                       height=4, width=4, crs="EPSG:4326",
                       transform=from_origin(-141.0, 64.0, PIXEL, PIXEL)) as dataset:
        dataset.write(np.ones((4, 4), dtype="float32"), 1)
    # No source claims 13 unnamed bands, so it cannot be identified at all.
    with pytest.raises(RuntimeError, match="match no weather source"):
        derive.read_bands(path, ["temperature_2m"])
    # Named explicitly, it fails on the band count instead.
    with pytest.raises(RuntimeError, match="expected 23"):
        derive.read_bands(path, ["temperature_2m"], registry.IFS)


# ---------------------------------------------------------------------------
# FWI
# ---------------------------------------------------------------------------
def test_fwi_state_starts_at_the_cwfis_constants():
    state = derive.FWIState({0: (3, 4), 1: (3, 4)})
    assert set(state.state) == {0, 1}
    assert np.all(state.state[0]["ffmc"] == registry.FWI_INIT["ffmc"])
    assert np.all(state.state[1]["dc"] == registry.FWI_INIT["dc"])
    assert state.day is None


def test_fwi_state_round_trips(tmp_path):
    state = derive.FWIState({0: (3, 4), 2: (5, 2)})
    state.state[0]["dc"][:] = 412.5
    state.day = date(2026, 3, 9)
    path = tmp_path / "fwi_state.npz"
    state.save(path)

    loaded = derive.FWIState.load(path)
    assert loaded.day == date(2026, 3, 9)
    assert set(loaded.state) == {0, 2}
    assert np.all(loaded.state[0]["dc"] == 412.5)
    assert loaded.shapes() == {0: (3, 4), 2: (5, 2)}
    # The staging file must not survive as a second checkpoint.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["fwi_state.npz"]


def test_advance_tile_moves_the_moisture_codes_and_returns_six_components(tmp_path):
    tile = write_ifs_tile(tmp_path / "IFS_2026_07_01_tile00.tif", {
        "temperature_2m_max": 298.15,      # 25 C
        "dewpoint_temperature_2m": 283.15,  # 12 C
        "u_component_of_wind_10m": 3.0,
        "v_component_of_wind_10m": 4.0,     # 5 m/s -> 18 km/h
        "total_precipitation_sum": 0.0,
    })
    state = derive.FWIState({0: (6, 8)}).state[0]
    before = {key: value.copy() for key, value in state.items()}
    components = derive.advance_tile(state, tile, date(2026, 7, 1))

    assert set(components) == set(derive.COMPONENT_KEYS)
    # A warm dry day with no rain dries the duff and builds drought.
    assert np.all(state["dmc"] > before["dmc"])
    assert np.all(state["dc"] > before["dc"])
    assert np.all(np.isfinite(components["fwi"]))
    assert np.all(components["fwi"] >= 0.0)
    # The state dict is advanced in place, which is what carries the recursion.
    assert np.all(state["ffmc"] == components["ffmc"])


def test_advance_tile_treats_the_zero_sentinel_as_outside_the_domain(tmp_path):
    tile = write_ifs_tile(tmp_path / "IFS_2026_07_01_tile00.tif", {
        "temperature_2m_max": 298.15, "dewpoint_temperature_2m": 283.15,
        "u_component_of_wind_10m": 1.0, "v_component_of_wind_10m": 1.0,
        "total_precipitation_sum": 0.0,
    })
    with rasterio.open(tile, "r+") as dataset:
        index = registry.IFS_BAND_INDEX["temperature_2m_max"]
        band = dataset.read([index])[0]
        band[0, :] = registry.FWI_OUTSIDE_SENTINEL
        dataset.write(band, index)

    state = derive.FWIState({0: (6, 8)}).state[0]
    components = derive.advance_tile(state, tile, date(2026, 7, 1))
    assert np.all(np.isnan(components["fwi"][0, :]))
    assert np.all(np.isfinite(components["fwi"][1:, :]))


def test_mosaic_averages_the_overlap_and_keeps_a_one_sided_pixel(tmp_path):
    """Two tiles sharing a footprint: the mean where both are valid."""
    shape = (4, 5)
    paths = {}
    for tile in (0, 1):
        paths[tile] = write_ifs_tile(tmp_path / f"t{tile}.tif", {}, shape=shape)
    context = derive.mosaic_context(paths)

    per_tile = {}
    for tile, value in ((0, 10.0), (1, 20.0)):
        per_tile[tile] = {key: np.full(shape, value, dtype="float32")
                          for key in derive.COMPONENT_KEYS}
    per_tile[1]["ffmc"][:] = np.nan

    bands = derive.mosaic_components(per_tile, context)
    ffmc, dmc = bands[0], bands[1]
    interior = np.s_[1:-1, 1:-1]
    # DMC is valid on both sides -> the mean; FFMC only on tile 0 -> tile 0.
    assert np.allclose(dmc[interior], 15.0, atol=1e-3)
    assert np.allclose(ffmc[interior], 10.0, atol=1e-3)


def test_write_fwi_labels_its_bands_and_leaves_nan_untagged(tmp_path):
    shape = (4, 5)
    paths = {0: write_ifs_tile(tmp_path / "t.tif", {}, shape=shape)}
    context = derive.mosaic_context(paths)
    bands = [np.full((context["height"], context["width"]), float(i), dtype="float32")
             for i in range(6)]
    out = tmp_path / "2026_07_01.tif"
    derive.write_fwi(bands, context, out)

    with rasterio.open(out) as dataset:
        assert dataset.count == 6
        assert dataset.descriptions == registry.FWI_BAND_NAMES
        # FireMPC left nodata untagged so NaN survives the alignment stage.
        assert dataset.nodata is None
    assert not list(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------------------
# Source identification
# ---------------------------------------------------------------------------
def test_detect_source_reads_the_band_names_first(tmp_path):
    """Band names are written at download time and outrank the band count."""
    for source in (registry.IFS, registry.ERA5):
        path = write_source_tile(tmp_path / f"{source.key}.tif", source, {}, name_bands=True)
        assert derive.detect_source(path) is source


def test_detect_source_falls_back_to_the_band_count(tmp_path):
    """Older tiles carry no descriptions; 23 and 18 are still unambiguous."""
    path = write_source_tile(tmp_path / "unnamed.tif", registry.ERA5, {})
    assert derive.detect_source(path) is registry.ERA5


def test_era5_bands_resolve_by_name_not_by_ifs_position(tmp_path):
    """Band 5 is dewpoint in IFS and precipitation in the ERA5 slot.

    Reading one source with the other's index is silent -- both are float32
    rasters of the right shape -- so this is the failure the name resolution
    exists to prevent.
    """
    assert registry.IFS.band_index["dewpoint_temperature_2m"] == 5
    assert registry.ERA5.band_index["total_precipitation_sum"] == 5

    path = write_source_tile(tmp_path / "era5.tif", registry.ERA5, {
        "dewpoint_temperature_2m": 280.0, "total_precipitation_sum": 0.004,
    }, name_bands=True)
    bands = derive.read_bands(path, ["dewpoint_temperature_2m", "total_precipitation_sum"])
    assert np.allclose(bands["dewpoint_temperature_2m"], 280.0)
    assert np.allclose(bands["total_precipitation_sum"], 0.004)


def test_read_bands_rejects_a_band_the_source_does_not_carry(tmp_path):
    path = write_source_tile(tmp_path / "ifs.tif", registry.IFS, {}, name_bands=True)
    with pytest.raises(RuntimeError, match="carries no band 'snow_cover'"):
        derive.read_bands(path, ["snow_cover"])


def test_fwi_advances_identically_from_either_source(tmp_path):
    """The two sources agree on units for the four fields FWI reads.

    That is what lets the recursion hand state across the changeover without
    rescaling, so the same weather must produce the same components.
    """
    weather = {
        "temperature_2m_max": 298.15, "dewpoint_temperature_2m": 283.15,
        "u_component_of_wind_10m": 3.0, "v_component_of_wind_10m": 4.0,
        "total_precipitation_sum": 0.002,
    }
    outputs = []
    for source in (registry.IFS, registry.ERA5):
        path = write_source_tile(tmp_path / f"{source.key}.tif", source, weather,
                                 name_bands=True)
        state = derive.FWIState({0: (6, 8)}).state[0]
        outputs.append(derive.advance_tile(state, path, date(2026, 7, 1)))
    for key in derive.COMPONENT_KEYS:
        assert np.allclose(outputs[0][key], outputs[1][key], equal_nan=True)
