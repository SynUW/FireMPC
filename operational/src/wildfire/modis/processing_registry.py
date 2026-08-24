"""Frozen operational contract for the 17-channel MODIS package.

This module is intentionally data-only.  Runtime code imports it rather than
duplicating band order, scale ranges, routing, or output names.
"""

from __future__ import annotations

from dataclasses import dataclass

from wildfire.modis.registry import OPERATIONAL_EXPORTS
from wildfire.modis.registry import PRODUCTS as SOURCE_PRODUCTS

REGISTRY_VERSION = "modis-17-v2-model-order"
SCALING_POLICY_VERSION = "firempc-scale-int-v1"
PACKAGE_DTYPE = "uint8"
PACKAGE_NODATA = 255
LAND_COVER_CODES = frozenset(range(1, 18))


@dataclass(frozen=True, slots=True)
class Channel:
    index: int
    name: str
    product: str
    band: int
    raw_range: tuple[float, float] | None
    categorical: bool = False
    nodata: int = PACKAGE_NODATA


@dataclass(frozen=True, slots=True)
class Route:
    product: str
    exports: tuple[str, ...]
    cadence_days: int
    stages: tuple[str, ...]
    retained_bands: tuple[int, ...]
    band_names: tuple[str, ...]
    spatial_fill: bool
    fill_nodata: tuple[float, ...]
    downsample_method: str | None
    downsample_invalid: float | None
    downsample_factor: int | None
    downsample_stride: int | None
    output_dir: str
    # `exports` grouped by observing platform, Terra first -- the order the
    # FireMPC combiners break ties in (`--prefer mod`).  A date is processable
    # when ANY group is complete, not when every export is: all three combiners
    # key on `set(mod) | set(myd)` and take whichever side is valid per pixel,
    # so a single-platform day is a first-class case, not a degraded one.  The
    # 2000-2025 reference contains 879 of them -- 823 before Aqua launched, 26
    # Terra-only from the 2002-07-30 Aqua outage, 30 Aqua-only from the
    # 2003-12-17 Terra anomaly -- and every one was published as a full
    # national raster.  Routes with a single group (an already-combined product
    # or a single-platform one) keep the old all-exports semantics.
    platform_groups: tuple[tuple[str, ...], ...] = ()


# These are the literal vmin/vmax values from FireMPC scale_int_tif.py.  Its
# parsed physical scale/offset are deliberately not applied by the uint8 path.
CHANNELS = (
    Channel(0, "NDVI", "NDVIEVI", 1, (-0.5, 1.0)),
    Channel(1, "EVI", "NDVIEVI", 2, (-1.0, 1.0)),
    Channel(2, "FPAR", "MCD15A3H", 2, (0.0, 87.14)),
    Channel(3, "LAI", "MCD15A3H", 1, (0.0, 46.78)),
    Channel(4, "reflectance_band_1", "MCD09GA", 1, (-3.86, 8338.89)),
    Channel(5, "reflectance_band_2", "MCD09GA", 2, (4.53, 8620.19)),
    Channel(6, "reflectance_band_3", "MCD09GA", 3, (15.45, 9426.45)),
    Channel(7, "reflectance_band_7", "MCD09GA", 4, (-17.73, 1937.25)),
    Channel(8, "LST_day", "MCD11A1", 1, (7500.0, 20000.0)),
    Channel(9, "LST_night", "MCD11A1", 2, (7500.0, 20000.0)),
    Channel(10, "emissivity_31", "MCD11A1", 3, (1.0, 255.0)),
    Channel(11, "emissivity_32", "MCD11A1", 4, (1.0, 255.0)),
    Channel(12, "brightness_temperature_20", "MCD09CMG", 1, (219.0, 334.4)),
    Channel(13, "brightness_temperature_21", "MCD09CMG", 2, (193.0, 400.0)),
    Channel(14, "brightness_temperature_31", "MCD09CMG", 3, (204.0, 356.0)),
    Channel(15, "brightness_temperature_32", "MCD09CMG", 4, (202.0, 386.3)),
    Channel(16, "land_cover_type", "MCD12Q1", 1, None, categorical=True, nodata=0),
)

ROUTES = {
    "MCD09GA": Route(
        "MCD09GA",
        ("MOD09GA_b1237", "MYD09GA_b1237", "MOD09GA_QA_state1km", "MYD09GA_QA_state1km"),
        1,
        ("qa_combined", "masked", "spatial_filled", "downsampled", "merged", "unified", "daily_filled", "scaled"),
        (0, 1, 2, 3),
        ("reflectance_band_1", "reflectance_band_2", "reflectance_band_3", "reflectance_band_7"),
        True, (-32768.0,), "average", -32768.0, 20, 10, "MCD09GA",
        platform_groups=(
            ("MOD09GA_b1237", "MOD09GA_QA_state1km"),
            ("MYD09GA_b1237", "MYD09GA_QA_state1km"),
        ),
    ),
    "NDVIEVI": Route(
        "NDVIEVI", (), 1,
        ("vegetation_indices", "scaled"),
        (0, 1), ("NDVI", "EVI"), False, (-32768.0,), None, None, None, None, "NDVIEVI",
    ),
    "MCD11A1": Route(
        "MCD11A1", ("MOD11A1", "MYD11A1"), 1,
        ("masked", "spatial_filled", "downsampled", "merged", "unified", "daily_filled", "scaled"),
        (0, 1, 2, 3), ("LST_day", "LST_night", "emissivity_31", "emissivity_32"),
        # Spatial filling must recognize the wrapped 32771/zero sentinels,
        # while the historical downsampler used its literal -32768 setting.
        True, (32771.0, 0.0), "average", -32768.0, 10, 5, "MCD11A1",
        platform_groups=(("MOD11A1",), ("MYD11A1",)),
    ),
    "MCD09CMG": Route(
        "MCD09CMG", ("MOD09CMG", "MYD09CMG"), 1,
        ("masked", "spatial_filled", "merged", "unified", "daily_filled", "scaled"),
        (0, 1, 2, 3),
        ("brightness_temperature_20", "brightness_temperature_21", "brightness_temperature_31", "brightness_temperature_32"),
        True, (-32765.0, 0.0), None, None, None, None, "MCD09CMG",
        platform_groups=(("MOD09CMG",), ("MYD09CMG",)),
    ),
    "MCD15A3H": Route(
        "MCD15A3H", ("MCD15A3H",), 4,
        ("masked", "spatial_filled", "downsampled", "merged", "unified", "daily_filled", "scaled"),
        (0, 1), ("LAI", "FPAR"), True, (-32768.0,), "average", -32768.0, 20, 10, "MCD15A3H",
        # Already a combined Terra+Aqua product; the producer does the merge.
        platform_groups=(("MCD15A3H",),),
    ),
    "MCD12Q1": Route(
        "MCD12Q1", ("MCD12Q1",), 365,
        # "scaled" is the stage the package reads from, so every route must end
        # there -- but for MCD12Q1 it is a pass-through publish, not a min-max
        # scale: land-cover codes are categorical and must survive untouched.
        ("masked", "merged", "downsampled", "unified", "daily_filled", "scaled"),
        (0,), ("land_cover_type",), False, (0.0,), "mode", 0.0, 20, 10, "MCD12Q1",
        platform_groups=(("MCD12Q1",),),
    ),
}

# Reference artifact certification.  The historical ``*_right`` MCD12Q1 tree
# is byte-for-byte band 1 of its two-band producer: it only removes QC.  The
# national NDVI/EVI oracle is computed from FireMPC-scaled national uint8
# MCD09GA, not from the later surviving 1 km NDVI/EVI tiles.
LINEAGE = {
    "ndvi_evi_national": {
        "status": "certified",
        "route": "scaled_national_mcd09ga_uint8_then_compute",
        "input_dtype": "uint8",
        "input_conversion": "divide_by_255",
        "evidence": "2025-07-01 NDVI and EVI match every jointly valid pixel exactly",
    },
    "mcd12q1_right": {
        "status": "certified",
        "route": "retain_lc_type1_band_only",
        "evidence": "2000, 2010, 2020 and 2024 outputs exactly equal producer band 1",
    },
}


def require_operational_lineage() -> None:
    unresolved = [name for name, record in LINEAGE.items() if record["status"] != "certified"]
    if unresolved:
        raise RuntimeError(
            "MODIS preprocessing is not operationally certified; unresolved lineage: "
            + ", ".join(unresolved)
        )


def validate_registry() -> None:
    indices = [channel.index for channel in CHANNELS]
    names = [channel.name for channel in CHANNELS]
    if indices != list(range(17)):
        raise RuntimeError(f"MODIS channel indices must be exactly 0..16: {indices}")
    if len(names) != len(set(names)):
        raise RuntimeError("MODIS channel names are not unique")
    unknown = set(OPERATIONAL_EXPORTS) - set(SOURCE_PRODUCTS)
    referenced = {export for route in ROUTES.values() for export in route.exports}
    if unknown or referenced != set(OPERATIONAL_EXPORTS):
        raise RuntimeError(
            f"operational source mismatch: unknown={sorted(unknown)}, "
            f"missing={sorted(set(OPERATIONAL_EXPORTS) - referenced)}, "
            f"extra={sorted(referenced - set(OPERATIONAL_EXPORTS))}")
    if tuple(channel.product for channel in CHANNELS[:4]) != (
        "NDVIEVI", "NDVIEVI", "MCD15A3H", "MCD15A3H"
    ) or tuple(channel.name for channel in CHANNELS[:4]) != ("NDVI", "EVI", "FPAR", "LAI"):
        raise RuntimeError("NDVI/EVI or FPAR/LAI order changed")
    for route in ROUTES.values():
        if len(route.retained_bands) != len(route.band_names):
            raise RuntimeError(f"{route.product}: retained band/name count differs")
        if route.downsample_factor is None and route.downsample_stride is not None:
            raise RuntimeError(f"{route.product}: stride without a factor")
        if (route.downsample_factor is None) != (route.downsample_invalid is None):
            raise RuntimeError(
                f"{route.product}: downsample factor and invalid value must be configured together")
        # Every export must belong to exactly one platform group: an export left
        # out would silently stop gating a date, and one listed twice would make
        # "any group complete" satisfiable by half a platform.
        grouped = [export for group in route.platform_groups for export in group]
        if sorted(grouped) != sorted(route.exports) or len(set(grouped)) != len(grouped):
            raise RuntimeError(
                f"{route.product}: platform groups must partition exports; "
                f"groups={route.platform_groups} exports={route.exports}")
        # build_modis_package reads every channel from scaled_root/<output_dir>,
        # so a route that never reaches "scaled" produces a package input that
        # can never exist.  MCD12Q1 shipped that way and failed only at the
        # final build, long after its expensive stages had run.
        if route.stages[-1] != "scaled":
            raise RuntimeError(
                f"{route.product}: route must end at 'scaled' (the stage the "
                f"package reads); ends at {route.stages[-1]!r}")


validate_registry()

CHANNEL_NAMES = tuple(channel.name for channel in CHANNELS)
CHANNELS_BY_PRODUCT = {
    product: tuple(channel for channel in CHANNELS if channel.product == product)
    for product in ROUTES
}
