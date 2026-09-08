"""Frozen operational contract for the weather family and its derivatives.

This module is intentionally data-only.  Runtime code imports it rather than
duplicating band order, reductions, scale ranges, or output names.

Two *sources* and three *products* live here.

Sources
    ``IFS``   ``ECMWF/NRT_FORECAST/IFS/OPER``, the operational near-real-time
              forecast.  23 bands: 1-13 are the ERA5-Land training channels IFS
              actually carries, 14-23 are IFS auxiliaries the historical 2025
              archive also recorded.  Begins 2024-11-13.
    ``ERA5``  ``ECMWF/ERA5_LAND/DAILY_AGGR``, the analysis the model was trained
              on.  All 18 slot channels, one image per day, back to 1950 --
              which is the only way to spin the FWI recursion up before IFS
              exists.  It lags roughly a week, so it is history, not operations.

Both sources export the same tile geometry with the same band *names*, so
everything downstream resolves bands by name and never by position.

Products
    ``VPD``   Vapour pressure deficit, one band, derived pointwise from air and
              dewpoint temperature.  Model channel 25.
    ``FWI``   The six CFFDRS components, derived recursively from temperature,
              wind, humidity and precipitation.  Model channels 19-24.

The five ERA5-Land weather channels IFS does not carry (``MISSING_ERA5_BANDS``)
are deliberately NOT synthesised here.  Nothing in this module writes the
18-band ERA5 model slot; that assembly waits on an approved policy for those
five.  VPD and FWI need none of them, so both are complete today.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

REGISTRY_VERSION = "weather-v2"

CRS = "EPSG:4326"
RESOLUTION_M = 11132
TILES = (0, 1, 2, 3)

# GEE serves near-surface air, dewpoint and skin temperature from the IFS
# forecast collection in Celsius while serving soil temperature in Kelvin.
# Verified live on 2026-08-15 over British Columbia: temperature_2m_sfc = 16.4,
# soil_temperature_sol1 = 287.2.  ERA5-Land serves every temperature in Kelvin.
KELVIN_OFFSET = 273.15

# ---------------------------------------------------------------------------
# The 18-band ERA5 model slot
# ---------------------------------------------------------------------------
# `dataload_h5.CHANNEL_NAMES[1:19]` order.  These are also, verbatim, band names
# of ECMWF/ERA5_LAND/DAILY_AGGR (verified live 2026-08-17), which is what makes
# the ERA5 source a straight band selection with no renaming.
ERA5_SLOT_BANDS: tuple[str, ...] = (
    "temperature_2m", "u_component_of_wind_10m", "v_component_of_wind_10m",
    "snow_cover", "total_precipitation_sum", "surface_latent_heat_flux_sum",
    "dewpoint_temperature_2m", "surface_pressure",
    "volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3", "volumetric_soil_water_layer_4",
    "temperature_2m_max", "skin_temperature_max",
    "potential_evaporation_sum", "total_evaporation_sum",
    "skin_reservoir_content", "surface_net_solar_radiation_sum",
)

# The ERA5-Land channels the model consumes that IFS has no equivalent for.
# `dataload_h5.CHANNEL_NAMES` puts them at model channels 4, 6, 15, 16 and 17.
# They are recorded, not filled.
MISSING_ERA5_BANDS: tuple[str, ...] = (
    "snow_cover",
    "surface_latent_heat_flux_sum",
    "potential_evaporation_sum",
    "total_evaporation_sum",
    "skin_reservoir_content",
)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class WeatherBand:
    """One output band of a daily weather tile.

    ``source``    Earth Engine band name in the source collection.
    ``reduction`` ``mean`` and ``max`` run over the eight +3..+24 forecast
                  steps; ``accumulated`` takes the +24 image directly, because
                  the IFS accumulators run from the initialisation, so the +24
                  value is already the 24-hour total; ``daily`` means the
                  collection already publishes one image per day and the band
                  is taken as served.
    ``to_kelvin`` add 273.15 to convert the served Celsius to the training unit.
    ``era5``      the ERA5-Land training channel this band stands in for, or
                  None for an auxiliary with no ERA5 counterpart.
    """

    index: int
    name: str
    source: str
    reduction: str
    to_kelvin: bool = False
    era5: bool = True


# Backwards-compatible alias: this type was `IFSBand` when IFS was the only
# source.
IFSBand = WeatherBand


@dataclass(frozen=True, slots=True)
class WeatherSource:
    """One daily weather collection and everything needed to export it.

    ``forecast``  the collection is a forecast keyed by (initialisation, lead
                  time) and must be reduced over ``forecast_hours``; otherwise
                  it publishes one image per day and is taken directly.
    ``first_date`` the first date this source can produce.  IFS is a
                  near-real-time collection with no archive behind it, so this
                  is a hard floor, not a convention.
    """

    key: str
    collection: str
    prefix: str
    bands: tuple[WeatherBand, ...]
    first_date: date
    forecast: bool
    resolution_m: int = RESOLUTION_M

    @property
    def band_names(self) -> tuple[str, ...]:
        return tuple(band.name for band in self.bands)

    @property
    def band_index(self) -> dict[str, int]:
        return {band.name: band.index for band in self.bands}

    def stem(self, day: date, tile: int) -> str:
        """Tile filename stem, e.g. ``IFS_2025_07_01_tile00``."""
        return f"{self.prefix}_{date_token(day)}_tile{tile:02d}"

    def sidecar_stem(self, day: date) -> str:
        return f"{self.prefix}_{date_token(day)}"


# -- IFS --------------------------------------------------------------------
# ECMWF/NRT_FORECAST/IFS/OPER is a forecast collection: one image per
# (initialisation, lead time).  The 2025 archive at drivers/IFS fixed the daily
# convention as the 00 UTC initialisation reduced over forecast hours 3..24,
# assigned to the initialisation date.  `IFS_2025_07_01.json` records exactly
# this, and it is reproduced verbatim rather than re-derived.
INITIALISATION_HOUR = 0
FORECAST_HOURS = (3, 6, 9, 12, 15, 18, 21, 24)
ACCUMULATION_HOUR = 24

# Forecast hour 0 carries no 3-hourly maximum -- `max_2m_temperature_last_3h_sfc`
# reads exactly -273.15 C there, the 0 K sentinel for a window that has not
# happened yet.  That, not convention alone, is why the reduction starts at +3.
STEP_ZERO_IS_SENTINEL = True

IFS_BANDS: tuple[WeatherBand, ...] = (
    WeatherBand(1, "temperature_2m", "temperature_2m_sfc", "mean", to_kelvin=True),
    WeatherBand(2, "u_component_of_wind_10m", "u_component_of_wind_10m_sfc", "mean"),
    WeatherBand(3, "v_component_of_wind_10m", "v_component_of_wind_10m_sfc", "mean"),
    WeatherBand(4, "total_precipitation_sum", "total_precipitation_sfc", "accumulated"),
    WeatherBand(5, "dewpoint_temperature_2m", "dewpoint_temperature_2m_sfc", "mean", to_kelvin=True),
    WeatherBand(6, "surface_pressure", "surface_pressure_sfc", "mean"),
    WeatherBand(7, "volumetric_soil_water_layer_1", "volumetric_soil_moisture_sol1", "mean"),
    WeatherBand(8, "volumetric_soil_water_layer_2", "volumetric_soil_moisture_sol2", "mean"),
    WeatherBand(9, "volumetric_soil_water_layer_3", "volumetric_soil_moisture_sol3", "mean"),
    WeatherBand(10, "volumetric_soil_water_layer_4", "volumetric_soil_moisture_sol4", "mean"),
    WeatherBand(11, "temperature_2m_max", "max_2m_temperature_last_3h_sfc", "max", to_kelvin=True),
    WeatherBand(12, "skin_temperature_max", "skin_temperature_sfc", "max", to_kelvin=True),
    WeatherBand(13, "surface_net_solar_radiation_sum", "surface_net_solar_radiation_sfc", "accumulated"),
    WeatherBand(14, "ifs_snow_albedo_mean", "snow_albedo_sfc", "mean", era5=False),
    WeatherBand(15, "ifs_total_column_water_mean", "total_column_water_sfc", "mean", era5=False),
    WeatherBand(16, "ifs_total_column_water_vapour_mean", "total_column_water_vapour_sfc", "mean", era5=False),
    # Soil temperature is the one temperature GEE already serves in Kelvin.
    WeatherBand(17, "ifs_soil_temperature_layer_1", "soil_temperature_sol1", "mean", era5=False),
    WeatherBand(18, "ifs_soil_temperature_layer_2", "soil_temperature_sol2", "mean", era5=False),
    WeatherBand(19, "ifs_soil_temperature_layer_3", "soil_temperature_sol3", "mean", era5=False),
    WeatherBand(20, "ifs_soil_temperature_layer_4", "soil_temperature_sol4", "mean", era5=False),
    WeatherBand(21, "ifs_runoff_24h", "runoff_sfc", "accumulated", era5=False),
    WeatherBand(22, "ifs_surface_solar_radiation_downwards_24h", "surface_solar_radiation_downwards_sfc", "accumulated", era5=False),
    WeatherBand(23, "ifs_surface_net_thermal_radiation_24h", "surface_net_thermal_radiation_sfc", "accumulated", era5=False),
)

# -- ERA5-Land --------------------------------------------------------------
# DAILY_AGGR is already one image per day, so there is no forecast-hour
# reduction, no accumulation step and no Celsius offset: every band is taken as
# served, and every served name is already the training name.  Units match IFS
# on the four fields FWI reads -- K, m/s, m of precipitation -- which is why the
# recursion can hand state from one source to the other without rescaling.
ERA5_BANDS: tuple[WeatherBand, ...] = tuple(
    WeatherBand(index, name, name, "daily")
    for index, name in enumerate(ERA5_SLOT_BANDS, start=1)
)

# ---------------------------------------------------------------------------
# Rebuilding a missing ERA5-Land daily aggregate
# ---------------------------------------------------------------------------
# DAILY_AGGR is a derived product, and Earth Engine has been observed to skip a
# day of it while the hourly collection behind it is complete (2026-08-26, seen
# 2026-09-07: DAILY_AGGR 0 images, HOURLY 24).  A skipped day is not a cosmetic
# gap -- the FFMC/DMC/DC recursion cannot step over a date, so one absent
# aggregate stalls the certified ledger and every target that depends on it.
#
# The aggregation is recoverable because the band *name* carries the rule, and
# each rule was verified against a served day rather than assumed:
#
#   ``_max``   the hourly maximum.
#   ``_sum``   the sum of the day's hourly *increments*, which is not any single
#              hourly image.  ERA5-Land accumulates over forecast steps 1..24 --
#              valid 01:00 of the day through 00:00 of the next -- so the series
#              inside one calendar day RESETS at 01:00.  Increments for 01:00
#              through 23:00 telescope to the 23:00 image, and the 00:00 image
#              belongs to the previous day's run, contributing only its own last
#              hour.  Hence:
#
#                  sum(D) = v(D, 23:00) + v(D, 00:00) - v(D-1, 23:00)
#
#   otherwise  the hourly mean.
#
# Verified live 2026-09-07 against ECMWF/ERA5_LAND/DAILY_AGGR over Canada for
# 2026-08-25, 08-27 and 07-15: means and maxima reproduce to <1e-5, and the four
# accumulations to ~1e-16 -- exact in double precision.
# surface_latent_heat_flux_sum reaches 7.6e-5 on some days, which is float32
# cancellation between two large accumulations, not a rule mismatch.
#
# The near-misses are recorded because they are the trap.  Taking the 23:00
# image alone, or the next day's 00:00 image, agrees at a *point* to 4.5e-8 and
# still errs by ~15% elsewhere in the domain, which is why the gate reduces over
# all of Canada and not over a sample pixel.
#
# A reconstructed day is genuine ERA5-Land: same reanalysis, same aggregation, a
# different distribution product.  That is what keeps it admissible to the
# certified ledger, and why `weather download --from-hourly` re-verifies the
# rules against a neighbouring served day before it writes anything.
ERA5_HOURLY_COLLECTION = "ECMWF/ERA5_LAND/HOURLY"
ERA5_RECONSTRUCTION_VERSION = "era5-daily-from-hourly-v1"
ERA5_HOURLY_STEPS = 24


def era5_hourly_rule(slot: str) -> tuple[str, str]:
    """The hourly band and reduction that reproduce one DAILY_AGGR band."""
    if slot.endswith("_max"):
        return slot[: -len("_max")], "max"
    if slot.endswith("_sum"):
        return slot[: -len("_sum")], "accumulated"
    return slot, "mean"


# (daily band, hourly band, reduction) for all 18, resolved once so the mapping
# is inspectable rather than recomputed inside an Earth Engine loop.
ERA5_HOURLY_RULES: tuple[tuple[str, str, str], ...] = tuple(
    (slot, *era5_hourly_rule(slot)) for slot in ERA5_SLOT_BANDS
)


SOURCES: dict[str, WeatherSource] = {
    "ifs": WeatherSource(
        key="ifs",
        collection="ECMWF/NRT_FORECAST/IFS/OPER",
        prefix="IFS",
        bands=IFS_BANDS,
        # The collection begins 2024-11-12 12Z; the first complete 00 UTC cycle
        # is the next day.  There is no earlier IFS to download -- it is a
        # near-real-time collection, not an archive.
        first_date=date(2024, 11, 13),
        forecast=True,
    ),
    "era5": WeatherSource(
        key="era5",
        collection="ECMWF/ERA5_LAND/DAILY_AGGR",
        prefix="ERA5",
        bands=ERA5_BANDS,
        first_date=date(1950, 1, 2),
        forecast=False,
    ),
}

IFS = SOURCES["ifs"]
ERA5 = SOURCES["era5"]

# The changeover: ERA5-Land for history, IFS from its first complete cycle on.
# Choosing by date rather than by what happens to be on disk keeps a backfill
# and a daily run producing the same file for the same day.
SOURCE_SWITCH_DATE = IFS.first_date

# Names kept from the single-source era so importers do not all have to change
# at once.  New code should go through `SOURCES` / `source_for`.
GEE_COLLECTION = IFS.collection
IFS_BAND_NAMES: tuple[str, ...] = IFS.band_names
IFS_BAND_INDEX: dict[str, int] = IFS.band_index


def source_for(day: date) -> WeatherSource:
    """The source that owns a date: ERA5-Land before IFS exists, IFS after."""
    return ERA5 if day < SOURCE_SWITCH_DATE else IFS


def source_by_key(key: str) -> WeatherSource:
    try:
        return SOURCES[key.lower()]
    except KeyError:
        raise KeyError(f"unknown weather source {key!r}; have {sorted(SOURCES)}") from None


def source_by_band_count(count: int) -> WeatherSource | None:
    """Identify a source from a tile's band count, for files without names."""
    matches = [source for source in SOURCES.values() if len(source.bands) == count]
    return matches[0] if len(matches) == 1 else None


# ---------------------------------------------------------------------------
# Derived products
# ---------------------------------------------------------------------------
FWI_BAND_NAMES: tuple[str, ...] = ("FFMC", "DMC", "DC", "ISI", "BUI", "FWI")
VPD_BAND_NAMES: tuple[str, ...] = ("VPD",)

# CWFIS canonical spring startup, from FireMPC `fwi_module`.  The reference
# FWI_merged tree starts 2000-01-01 and carries one file for every calendar day
# including winter, so the operative convention is a single cold start followed
# by unbroken daily carry-over -- not the April-1 restart the fwi_module
# docstring describes.  The tree is the contract; the docstring is not.
FWI_INIT = {"ffmc": 85.0, "dmc": 6.0, "dc": 15.0}

# DC has a ~400-day time constant, so the cold start is still visible months
# later.  Starting from 2023-01-01 -- reachable only through ERA5-Land, since
# IFS does not exist before 2024-11-13 -- puts the residual startup error under
# 4% by 2026, smaller than the measured IFS-vs-ERA5 weather difference.  The
# earlier 2025-01-01 start left it at 22%.
FWI_SPINUP_START = "2023-01-01"

# Source bands the FWI recursion reads.  FireMPC's `fwi_module` addressed these
# by hardcoded ERA5 band number (2, 3, 5, 7, 13); IFS orders them differently,
# so they are resolved by name here and nowhere else.
FWI_INPUT_BANDS = {
    "u10": "u_component_of_wind_10m",
    "v10": "v_component_of_wind_10m",
    "precipitation": "total_precipitation_sum",
    "dewpoint": "dewpoint_temperature_2m",
    "temperature_max": "temperature_2m_max",
}

# `fwi_export_daily_components._process_one_tile_day` treats a pixel as outside
# the domain when temperature_2m_max or dewpoint reads exactly zero.  Preserved
# verbatim: it is what produced the reference tree's NaN footprint.
FWI_OUTSIDE_SENTINEL = 0.0

VPD_INPUT_BANDS = {
    "temperature": "temperature_2m",
    "dewpoint": "dewpoint_temperature_2m",
}

# Plausibility gates from FireMPC `climate_drought_calculation.py`, in Celsius
# and kPa.  Values outside these become NaN.
VPD_TEMP_RANGE = (-100.0, 70.0)
VPD_DEWPOINT_RANGE = (-100.0, 60.0)
VPD_RANGE = (0.0, 10.0)

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Route:
    product: str
    stages: tuple[str, ...]
    band_names: tuple[str, ...]
    # Directory name under each stage root.  `scale_int_tif.py` picks its band
    # configuration by case-insensitive substring match on this name, so
    # renaming these silently changes the quantisation.
    output_dir: str
    scale_keyword: str


ROUTES: dict[str, Route] = {
    # VPD is computed per tile, mosaicked by merge_patches, aligned to the
    # national template, then quantised.  It needs no spatial fill: both
    # sources are gap-free model fields, not swath composites.
    "VPD": Route(
        "VPD",
        ("derived", "merged", "unified", "scaled"),
        VPD_BAND_NAMES,
        "VPD_merged",
        "vpd",
    ),
    # FWI skips `merged`: the FireMPC exporter mosaics the four tiles itself,
    # as part of the same pass that advances the recursion, so `derived` is
    # already national.
    "FWI": Route(
        "FWI",
        ("derived", "unified", "scaled"),
        FWI_BAND_NAMES,
        "FWI_merged",
        "fwi",
    ),
}

# The frozen uint8 quantisation ranges, copied from the `--vpd-default` and
# `--fwi-default` arguments of FireMPC `scale_int_tif.py`.  The training cube's
# encoding depends on these exact numbers; they are never recomputed from data.
SCALE_RANGES: dict[str, tuple[tuple[float, float], ...]] = {
    "VPD": ((0.0, 1.63),),
    "FWI": ((0.0, 101.0), (0.0, 119.0), (0.0, 694.0),
            (0.0, 11.0), (0.0, 149.0), (0.0, 33.0)),
}

SCALE_BAND_CONFIG: dict[str, str] = {
    product: ", ".join(
        f"{index}:1:0:{low:g}:{high:g}"
        for index, (low, high) in enumerate(ranges)
    )
    for product, ranges in SCALE_RANGES.items()
}


def date_token(day) -> str:
    return day.strftime("%Y_%m_%d")


def ifs_stem(day, tile: int) -> str:
    """Historical IFS tile filename: ``IFS_2025_07_01_tile00``."""
    return IFS.stem(day, tile)
