from modis.processing_registry import (
    CHANNELS,
    LINEAGE,
    ROUTES,
    require_operational_lineage,
)
from modis.registry import OPERATIONAL_EXPORTS, PRODUCTS


def test_operational_exports_are_distinct_ten_source_profile():
    assert len(OPERATIONAL_EXPORTS) == 10
    assert set(OPERATIONAL_EXPORTS) <= set(PRODUCTS)
    assert "MOD13Q1" not in OPERATIONAL_EXPORTS
    assert "MYD13Q1" not in OPERATIONAL_EXPORTS


def test_channel_order_and_routes_are_frozen():
    assert [channel.index for channel in CHANNELS] == list(range(17))
    assert [channel.name for channel in CHANNELS[:4]] == ["NDVI", "EVI", "FPAR", "LAI"]
    assert ROUTES["MCD15A3H"].band_names == ("LAI", "FPAR")
    assert ROUTES["MCD09GA"].downsample_factor == 20
    assert ROUTES["MCD09GA"].downsample_stride == 10
    assert ROUTES["MCD11A1"].downsample_factor == 10
    assert ROUTES["MCD11A1"].fill_nodata == (32771.0, 0.0)
    assert ROUTES["MCD11A1"].downsample_invalid == -32768.0
    assert ROUTES["MCD12Q1"].downsample_method == "mode"
    assert ROUTES["MCD12Q1"].stages == (
        "masked", "merged", "downsampled", "unified", "daily_filled", "scaled"
    )


def test_operational_lineage_is_certified():
    assert LINEAGE["mcd12q1_right"]["status"] == "certified"
    assert LINEAGE["ndvi_evi_national"]["status"] == "certified"
    assert LINEAGE["ndvi_evi_national"]["route"] == (
        "scaled_national_mcd09ga_uint8_then_compute"
    )
    assert ROUTES["NDVIEVI"].stages == ("vegetation_indices", "scaled")
    require_operational_lineage()
