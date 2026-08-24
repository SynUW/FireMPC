"""VPD and the six FWI components, computed from daily weather tiles.

Both derivations are ports of the FireMPC training code, kept numerically
identical to it and changed in exactly one respect: band resolution.

FireMPC addressed ERA5 bands by hardcoded position -- `climate_drought_calculation`
reads band 0 and band 6, `fwi_module` reads bands 2, 3, 5, 7, 13.  IFS carries
the same physical fields in a different order, so those positions are wrong for
it, and the 18-band ERA5-Land export used for the historical spin-up orders them
differently again.  Every band here is resolved by *name* through the source's
`weather_registry` entry -- identified from the tile itself when it is not
supplied -- so one derivation serves both sources.  Nothing else about the
arithmetic differs:

* VPD is Tetens saturation vapour pressure differenced between air and dewpoint
  temperature, with the same Celsius plausibility gates and the same non-negative
  clamp as `climate_drought_calculation.calc_vpd`.
* FWI calls top-level `FireMPC/fwi/fwi_module.py` -- the Van Wagner
  (1987) equations are imported, never reimplemented -- and mosaics the four
  tiles with the same bilinear-reproject-then-average rule as
  `fwi_export_daily_components._merge_components_geographic`.
"""

from __future__ import annotations

from wildfire import vendorpaths
import importlib.util
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine, array_bounds, from_bounds
from rasterio.warp import Resampling, reproject

from wildfire.weather import registry as registry

FIREMPC_FWI_MODULE_PATH = vendorpaths.FIREMPC_FWI / "fwi_module.py"
FWI_MODULE_PATH = FIREMPC_FWI_MODULE_PATH


def _load_fwi_module():
    """Import the FireMPC FWI equations by file path.

    The top-level FireMPC module is the single maintained implementation. Its
    checksum is pinned in ``operational/vendor/PROVENANCE.json``.
    """
    spec = importlib.util.spec_from_file_location("firempc_fwi_module", FWI_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {FWI_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("firempc_fwi_module", module)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Tile reading
# ---------------------------------------------------------------------------
def detect_source(path: Path) -> registry.WeatherSource:
    """Which source wrote this tile.

    The band descriptions are written at download time and are the primary
    evidence; the band count is the fallback for a tile that predates them.
    Guessing wrong is silent -- band 5 is dewpoint in IFS and precipitation in
    the ERA5 slot -- so an ambiguous tile raises instead.
    """
    with rasterio.open(path) as dataset:
        descriptions = tuple(dataset.descriptions or ())
        count = dataset.count
    for source in registry.SOURCES.values():
        if descriptions and tuple(descriptions) == source.band_names:
            return source
    source = registry.source_by_band_count(count)
    if source is None:
        raise RuntimeError(
            f"{path.name}: {count} bands named {descriptions or '(unnamed)'} match no "
            f"weather source; pass the source explicitly")
    return source


def read_bands(path: Path, names, source: registry.WeatherSource | None = None
               ) -> dict[str, np.ndarray]:
    """Read named source bands as float32, with nodata and -inf turned into NaN.

    The Earth Engine export tags nodata as -inf, so an untreated read leaves
    -inf in the arithmetic rather than NaN, and every downstream comparison
    against it silently succeeds.
    """
    source = source or detect_source(path)
    index = source.band_index
    out: dict[str, np.ndarray] = {}
    with rasterio.open(path) as dataset:
        if dataset.count != len(source.bands):
            raise RuntimeError(
                f"{path.name}: {dataset.count} bands, expected {len(source.bands)} "
                f"for source {source.key}")
        nodata = dataset.nodata
        for name in names:
            if name not in index:
                raise RuntimeError(f"{source.key} carries no band {name!r}")
            data = dataset.read([index[name]])[0].astype(np.float32)
            if nodata is not None and np.isfinite(nodata):
                data[data == nodata] = np.nan
            data[~np.isfinite(data)] = np.nan
            out[name] = data
    return out


def tile_profile(path: Path) -> dict:
    with rasterio.open(path) as dataset:
        return dataset.profile.copy()


# ---------------------------------------------------------------------------
# VPD
# ---------------------------------------------------------------------------
def saturation_vapour_pressure(temp_c: np.ndarray) -> np.ndarray:
    """Tetens, kPa.  `climate_drought_calculation.calc_saturation_vapor_pressure`."""
    return 0.611 * np.exp(17.27 * temp_c / (temp_c + 237.3))


def _gate(data: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    """NaN out values outside the plausibility range, as FireMPC does.

    FireMPC optionally ran `scipy.griddata` over the gated pixels.  That path
    exists because ERA5-Land carries coastal gaps; IFS is a gap-free model
    field, so on IFS the interpolation has nothing to fill and the fast path is
    the same computation.  It is not ported, so VPD derived from the ERA5
    source would keep ERA5-Land's coastal NaNs instead of interpolating over
    them.  That is why the ERA5 source exists for the FWI spin-up only -- FWI
    masks those pixels out through `FWI_OUTSIDE_SENTINEL` regardless -- and why
    operational VPD is IFS-only.
    """
    low, high = bounds
    out = np.array(data, dtype=np.float32, copy=True)
    out[~np.isfinite(out) | (out < low) | (out > high)] = np.nan
    return out


def vapour_pressure_deficit(path: Path, source: registry.WeatherSource | None = None
                            ) -> tuple[np.ndarray, dict]:
    """One tile's VPD in kPa, and the profile to write it with."""
    bands = read_bands(path, registry.VPD_INPUT_BANDS.values(), source)
    temp_c = _gate(
        bands[registry.VPD_INPUT_BANDS["temperature"]] - registry.KELVIN_OFFSET,
        registry.VPD_TEMP_RANGE,
    )
    dewpoint_c = _gate(
        bands[registry.VPD_INPUT_BANDS["dewpoint"]] - registry.KELVIN_OFFSET,
        registry.VPD_DEWPOINT_RANGE,
    )
    with np.errstate(invalid="ignore", over="ignore"):
        vpd = saturation_vapour_pressure(temp_c) - saturation_vapour_pressure(dewpoint_c)
        vpd = np.maximum(vpd, 0.0)
    return _gate(vpd, registry.VPD_RANGE), tile_profile(path)


def write_single_band(data: np.ndarray, profile: dict, out: Path) -> None:
    profile = profile.copy()
    profile.update(count=1, dtype="float32", compress="lzw", nodata=np.nan)
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", **profile) as dataset:
        dataset.write(data.astype("float32"), 1)
        dataset.descriptions = registry.VPD_BAND_NAMES


# ---------------------------------------------------------------------------
# FWI
# ---------------------------------------------------------------------------
class FWIState:
    """The per-tile FFMC/DMC/DC carried from one day to the next.

    FWI is the only stateful product in the pipeline.  DC has a roughly
    400-day time constant, so a checkpoint is not an optimisation -- a run that
    silently restarts from the cold-start constants produces a different, and
    wrong, drought code for months afterwards.
    """

    def __init__(self, shapes: dict[int, tuple[int, int]]):
        self.state: dict[int, dict[str, np.ndarray]] = {
            tile: {
                key: np.full(shape, value, dtype=np.float32)
                for key, value in registry.FWI_INIT.items()
            }
            for tile, shape in shapes.items()
        }
        self.day: date | None = None

    @classmethod
    def load(cls, path: Path) -> "FWIState":
        with np.load(path, allow_pickle=False) as archive:
            tiles = sorted({int(key.split("_")[0]) for key in archive.files if "_" in key})
            state = cls.__new__(cls)
            state.state = {
                tile: {key: archive[f"{tile}_{key}"] for key in registry.FWI_INIT}
                for tile in tiles
            }
            state.day = date.fromisoformat(str(archive["day"]))
        return state

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            f"{tile}_{key}": value
            for tile, codes in self.state.items()
            for key, value in codes.items()
        }
        payload["day"] = np.array(self.day.isoformat() if self.day else "")
        # np.savez appends `.npz` unless the name already ends in it, so the
        # staging name has to carry the extension itself.
        temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
        np.savez(temporary, **payload)
        temporary.replace(path)

    def shapes(self) -> dict[int, tuple[int, int]]:
        return {tile: codes["ffmc"].shape for tile, codes in self.state.items()}


def fwi_inputs(path: Path, source: registry.WeatherSource | None = None):
    """Convert one weather tile to the (T_c, RH, wind, rain) FireMPC expects.

    The band dict is keyed by FireMPC's ERA5 positions so
    `fwi_module.extract_fwi_inputs_from_bands` -- and its unit conversions and
    sanity assertions -- run unmodified on either source's data.  IFS and
    ERA5-Land agree on the units of all four fields it reads (K, m/s, and metres
    of precipitation), which is what lets the recursion carry state across the
    changeover without rescaling.
    """
    module = _load_fwi_module()
    names = registry.FWI_INPUT_BANDS
    bands = read_bands(path, names.values(), source)

    outside = (
        (bands[names["temperature_max"]] == registry.FWI_OUTSIDE_SENTINEL)
        | (bands[names["dewpoint"]] == registry.FWI_OUTSIDE_SENTINEL)
    )
    era5_positions = {
        2: bands[names["u10"]],
        3: bands[names["v10"]],
        5: bands[names["precipitation"]],
        7: bands[names["dewpoint"]],
        13: bands[names["temperature_max"]],
    }
    for position, data in era5_positions.items():
        era5_positions[position] = np.where(outside, np.nan, data)
    # FireMPC clamps precipitation before conversion; GRIB packing can leave
    # accumulators a hair below zero.
    era5_positions[5] = np.maximum(era5_positions[5], 0.0)
    return module, module.extract_fwi_inputs_from_bands(era5_positions)


def frozen_fraction(path: Path, source: registry.WeatherSource | None = None) -> float:
    """Fraction of pixels whose moisture codes cannot advance from this tile.

    Measured, not assumed: FireMPC's `np.where` guards swallow a NaN input
    rather than propagating it -- `dc_next` falls through to `dc_prev` when the
    rain comparison is false, which it is for NaN -- so a masked pixel does not
    become NaN, it *freezes* at yesterday's code and silently goes stale.  Only
    ISI and FWI, which read wind directly, come out NaN.

    That distinction matters at the ERA5-to-IFS changeover: ERA5-Land is masked
    over water and IFS is not, so those pixels carry frozen 2024 codes into the
    first IFS day and start evolving from a stale base.  Recording the fraction
    per date is what makes the effect visible in the manifests instead of
    invisible in the rasters.
    """
    names = registry.FWI_INPUT_BANDS
    bands = read_bands(path, (names["temperature_max"], names["dewpoint"]), source)
    outside = np.zeros(bands[names["temperature_max"]].shape, dtype=bool)
    for key in ("temperature_max", "dewpoint"):
        data = bands[names[key]]
        outside |= ~np.isfinite(data) | (data == registry.FWI_OUTSIDE_SENTINEL)
    return float(np.mean(outside))


def advance_tile(state: dict[str, np.ndarray], path: Path, day: date,
                 source: registry.WeatherSource | None = None) -> dict[str, np.ndarray]:
    """Advance one tile's moisture codes by a day and return all six components."""
    module, (temp_c, rh, wind, rain) = fwi_inputs(path, source)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ffmc = module.ffmc_next(state["ffmc"], temp_c, rh, wind, rain)
        dmc = module.dmc_next(state["dmc"], temp_c, rh, rain, day.month)
        dc = module.dc_next(state["dc"], temp_c, rain, day.month)
        isi = module.isi_calc(ffmc, wind)
        bui = module.bui_calc(dmc, dc)
        fwi = module.fwi_calc(isi, bui)
    state["ffmc"], state["dmc"], state["dc"] = ffmc, dmc, dc
    return {"ffmc": ffmc.copy(), "dmc": dmc.copy(), "dc": dc.copy(),
            "isi": isi.copy(), "bui": bui.copy(), "fwi": fwi.copy()}


# FWI values are legitimately large and positive, so NaN cannot be signalled
# through the warp; FireMPC used this sentinel and so does this port.
MOSAIC_SENTINEL = -99999.0
COMPONENT_KEYS = ("ffmc", "dmc", "dc", "isi", "bui", "fwi")


def mosaic_context(tile_paths: dict[int, Path]):
    """Common output grid for the four tiles: tile 1's resolution and CRS.

    `fwi_export_daily_components._build_mosaic_geo_context` picks tile 1
    deliberately -- it is the northern tile, and its pixel size sets the
    alignment the reference tree was built on.
    """
    transforms, shapes, crs_list = {}, {}, {}
    for tile, path in tile_paths.items():
        with rasterio.open(path) as dataset:
            transforms[tile] = dataset.transform
            shapes[tile] = (dataset.height, dataset.width)
            crs_list[tile] = dataset.crs

    boxes = [array_bounds(*shapes[tile], transforms[tile]) for tile in tile_paths]
    left = min(b[0] for b in boxes)
    bottom = min(b[1] for b in boxes)
    right = max(b[2] for b in boxes)
    top = max(b[3] for b in boxes)

    reference = transforms[1] if 1 in transforms else transforms[min(transforms)]
    xres, yres = abs(reference.a), abs(reference.e)
    width = max(1, int(np.ceil((right - left) / xres)))
    height = max(1, int(np.ceil((top - bottom) / yres)))
    crs = crs_list.get(1) or next(iter(crs_list.values()))
    return {
        "transforms": transforms,
        "affine": from_bounds(left, bottom, right, top, width, height),
        "width": width,
        "height": height,
        "crs": CRS.from_user_input(crs),
    }


def mosaic_components(per_tile: dict[int, dict[str, np.ndarray]], context) -> list[np.ndarray]:
    """Reproject each tile onto the common grid and average the overlap.

    Overlapping pixels take the arithmetic mean of the valid sides; a pixel
    valid on one side only keeps that side; valid nowhere stays NaN.
    """
    stacks = []
    for tile, components in per_tile.items():
        planes = []
        for key in COMPONENT_KEYS:
            source = components.get(key)
            if source is None:
                planes.append(np.full((context["height"], context["width"]), np.nan, dtype=np.float32))
                continue
            filled = np.where(np.isfinite(source), source, MOSAIC_SENTINEL).astype(np.float32)
            destination = np.full((context["height"], context["width"]), MOSAIC_SENTINEL, dtype=np.float32)
            reproject(
                source=filled,
                destination=destination,
                src_transform=context["transforms"][tile],
                src_crs=context["crs"],
                dst_transform=context["affine"],
                dst_crs=context["crs"],
                resampling=Resampling.bilinear,
                src_nodata=MOSAIC_SENTINEL,
                dst_nodata=MOSAIC_SENTINEL,
            )
            planes.append(np.where(destination == MOSAIC_SENTINEL, np.nan, destination).astype(np.float32))
        stacks.append(np.stack(planes, axis=0))

    cube = np.stack(stacks, axis=0).astype(np.float64, copy=False)
    valid = np.isfinite(cube)
    count = valid.sum(axis=0, dtype=np.float64)
    total = np.where(valid, cube, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        merged = np.where(count > 0, total / count, np.nan).astype(np.float32)
    return [merged[index].copy() for index in range(len(COMPONENT_KEYS))]


def write_fwi(bands: list[np.ndarray], context, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff", "dtype": "float32", "count": len(bands),
        "height": context["height"], "width": context["width"],
        "crs": context["crs"], "transform": context["affine"],
        "compress": "lzw", "tiled": True, "blockxsize": 512, "blockysize": 512,
        # FireMPC deliberately left nodata untagged so NaN stays NaN through
        # the alignment stage; the same choice is kept here.
    }
    temporary = out.with_suffix(f"{out.suffix}.tmp")
    with rasterio.open(temporary, "w", **profile) as dataset:
        for index, band in enumerate(bands, start=1):
            dataset.write(band.astype("float32"), index)
        dataset.descriptions = registry.FWI_BAND_NAMES
    temporary.replace(out)
