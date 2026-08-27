"""Build the aligned, physically encoded native55 window from daily IFS tiles."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import ExitStack
from datetime import date
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject
import typer

# Run directly as `python operational/canfirecast/weather_window.py`: put the application root on
# sys.path so the shared modules import by bare name, as elsewhere in FireMPC.
import sys
from pathlib import Path
OPERATIONAL = Path(__file__).resolve().parents[1]
if str(OPERATIONAL) not in sys.path:
    sys.path.insert(0, str(OPERATIONAL))

from canfirecast import input_registry as registry
import paths
from weather.registry import IFS, TILES, source_by_key

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

# IFS output names mapped explicitly to native model channels. Never positional.
SOURCE_TO_CHANNEL = {
    "temperature_2m": 1,
    "u_component_of_wind_10m": 2,
    "v_component_of_wind_10m": 3,
    "total_precipitation_sum": 5,
    "dewpoint_temperature_2m": 7,
    "surface_pressure": 8,
    "volumetric_soil_water_layer_1": 9,
    "volumetric_soil_water_layer_2": 10,
    "volumetric_soil_water_layer_3": 11,
    "volumetric_soil_water_layer_4": 12,
    "temperature_2m_max": 13,
    "skin_temperature_max": 14,
    "surface_net_solar_radiation_sum": 18,
}


def encode(values: np.ndarray, channel: int) -> np.ndarray:
    low, high = registry.PHYSICAL_RANGES[channel]
    return np.clip((values.astype(np.float32) - low) * 255.0 / (high - low), 0, 255)


def _tile_paths(drivers: Path, day: date, source=IFS) -> tuple[Path, ...]:
    paths_ = tuple(drivers / source.prefix / f"{source.stem(day, tile)}.tif" for tile in TILES)
    missing = [str(path) for path in paths_ if not path.is_file()]
    if missing:
        raise RuntimeError(f"{source.key} {day}: missing tiles: {missing}")
    return paths_


def _band_indices(datasets: list[rasterio.DatasetReader]) -> dict[str, int]:
    orders = []
    for dataset in datasets:
        names = tuple(name for name in dataset.descriptions if name)
        if len(names) != dataset.count:
            raise RuntimeError(f"{dataset.name}: every IFS band must have a description")
        orders.append(names)
    if any(order != orders[0] for order in orders[1:]):
        raise RuntimeError("IFS tile band-name orders differ")
    missing = set(SOURCE_TO_CHANNEL) - set(orders[0])
    if missing:
        raise RuntimeError(f"IFS tiles lack required named fields: {sorted(missing)}")
    return {name: orders[0].index(name) + 1 for name in SOURCE_TO_CHANNEL}


def _merge_masked(datasets, *, index: int):
    """Merge one band as a masked float array on Rasterio 1.3 and newer."""
    try:
        return merge(datasets, indexes=[index], method="first", masked=True)
    except TypeError:
        # ``masked=`` was added after FireMPC's pinned Rasterio 1.3.9. Asking
        # that version for a NaN output nodata and masking invalid cells gives
        # the same representation without upgrading the geospatial stack.
        values, transform = merge(
            datasets, indexes=[index], method="first", nodata=np.nan, dtype="float32"
        )
        return np.ma.masked_invalid(values), transform


def aligned_encoded_day(
    tile_paths: tuple[Path, ...], template: rasterio.DatasetReader
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Mosaic physical values, align to model grid, then encode."""
    with ExitStack() as stack:
        datasets = [stack.enter_context(rasterio.open(path)) for path in tile_paths]
        indices = _band_indices(datasets)
        # Spatial validity is separate from encoded values. A finite temperature
        # inside the source footprints defines source support; 255 remains data.
        temp_index = indices["temperature_2m"]
        source_valid, source_transform = _merge_masked(datasets, index=temp_index)
        source_valid = (~np.ma.getmaskarray(source_valid[0]) & np.isfinite(source_valid.data[0])).astype("uint8")
        validity = np.zeros((template.height, template.width), dtype="uint8")
        reproject(
            source_valid, validity, src_transform=source_transform,
            src_crs=datasets[0].crs, dst_transform=template.transform,
            dst_crs=template.crs, resampling=Resampling.nearest,
            src_nodata=0, dst_nodata=0,
        )
        result: dict[int, np.ndarray] = {}
        for name, channel in SOURCE_TO_CHANNEL.items():
            physical, transform = _merge_masked(datasets, index=indices[name])
            source = physical.filled(np.nan)[0].astype("float32")
            aligned = np.full((template.height, template.width), np.nan, dtype="float32")
            reproject(
                source, aligned, src_transform=transform, src_crs=datasets[0].crs,
                dst_transform=template.transform, dst_crs=template.crs,
                resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan,
            )
            valid = (validity != 0) & np.isfinite(aligned)
            # Preserve fractional encoded values through both correction
            # stages. The sole uint8 rounding boundary is after recursive
            # VPD/CFFDRS derivation.
            encoded = np.zeros(aligned.shape, dtype="float32")
            encoded[valid] = encode(aligned[valid], channel)
            result[channel] = encoded
        return result, validity.astype(bool)


def build_window(
    *, target: date, drivers: Path, template_path: Path, output: Path,
    overwrite: bool = False, source_key: str = "ifs",
) -> Path:
    registry.verify_pinned_artifacts()
    if output.exists() and not overwrite:
        raise FileExistsError(f"output exists: {output}")
    dates = registry.target_window_dates(target)
    source = source_by_key(source_key)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".weather-window-", suffix=".h5", dir=output.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    strings = h5py.string_dtype("utf-8")
    sources: list[list[str]] = []
    try:
        with rasterio.open(template_path) as template, h5py.File(tmp, "w") as h5:
            data = h5.create_dataset(
                "data", shape=(10, 55, template.height, template.width), dtype="f4",
                chunks=(1, 1, min(256, template.height), min(256, template.width)),
                compression="gzip", compression_opts=1, shuffle=True,
                fillvalue=0.0,
            )
            # HDF5 returns the fill value for unwritten chunks. Writing all 550
            # national planes here spent tens of minutes compressing zeros even
            # though IFS populates only 13 channels per date.
            validity = h5.create_dataset("spatial_validity", shape=(10, template.height, template.width), dtype="bool", compression="gzip")
            for time_index, day in enumerate(dates):
                tiles = _tile_paths(drivers, day, source)
                planes, mask = aligned_encoded_day(tiles, template)
                for channel, plane in planes.items():
                    data[time_index, channel] = plane
                validity[time_index] = mask
                sources.append([str(path) for path in tiles])
            h5.create_dataset("dates", data=np.asarray([d.isoformat() for d in dates], dtype=object), dtype=strings)
            h5.create_dataset("channel_names", data=np.asarray(registry.CHANNEL_NAMES, dtype=object), dtype=strings)
            h5.attrs.update({
                "target_date": target.isoformat(), "channel_registry_version": registry.REGISTRY_VERSION,
                "crs": template.crs.to_string(), "transform": tuple(template.transform)[:6],
                "height": template.height, "width": template.width,
                "weather_source": source.key,
                "source_weather_cycle": ("IFS 00UTC +3..+24 assigned to initialization date" if source.forecast else "ERA5-Land daily aggregate"),
                "source_tiles": json.dumps(sources), "encoding": "frozen physical ranges",
                "nodata_policy": "separate spatial_validity; encoded 0..255 are data",
            })
        os.replace(tmp, output)
    finally:
        tmp.unlink(missing_ok=True)
    return output


@app.command()
def main(
    target: Annotated[str, typer.Option("--target")],
    output: Annotated[Path, typer.Option("--output")],
    drivers: Annotated[Path, typer.Option("--drivers")] = Path(paths.DRIVERS),
    template: Annotated[Path, typer.Option("--grid-template")] = Path(paths.GRID_TEMPLATE),
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    source: Annotated[str, typer.Option("--source")] = "ifs",
) -> None:
    typer.echo(build_window(target=date.fromisoformat(target), drivers=drivers, template_path=template, output=output, overwrite=overwrite, source_key=source))


if __name__ == "__main__":
    app()
