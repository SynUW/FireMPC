"""Build the immutable 12-channel CanFireCast static package."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import rasterio
import typer

from wildfire.canfirecast import input_registry as registry
from wildfire import paths

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
STATIC_NAMES = tuple(registry.CHANNEL_NAMES[i] for i in registry.STATIC_CHANNELS)
DEFAULT_BASE = Path("")
DEFAULT_OSM = Path(os.environ.get("WILDFIRE_STATIC_OSM", ""))
DEFAULT_POPULATION = Path(os.environ.get("WILDFIRE_STATIC_WORLDPOP", ""))
DEFAULT_TERRAIN = Path(os.environ.get("WILDFIRE_STATIC_TERRAIN", ""))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_grid(source, template, path: Path) -> None:
    errors = []
    if source.shape != template.shape: errors.append(f"shape {source.shape} != {template.shape}")
    if source.crs != template.crs: errors.append(f"CRS {source.crs} != {template.crs}")
    if not source.transform.almost_equals(template.transform, precision=1e-12): errors.append("transform differs")
    if errors: raise RuntimeError(f"{path}: {'; '.join(errors)}")


def _encode_normalized(data: np.ndarray, source: Path) -> np.ndarray:
    valid = np.isfinite(data) & (data >= 0.0) & (data <= 1.0)
    if not np.any(valid):
        raise RuntimeError(f"{source}: no normalized values in [0,1]")
    out = np.full(data.shape, 255, dtype="uint8")
    out[valid] = np.rint(data[valid] * 255.0).astype("uint8")
    return out


def build_static_package(*, osm: Path, population: Path, terrain: Path, template_path: Path, output: Path, overwrite: bool = False) -> Path:
    if output.exists() and not overwrite: raise FileExistsError(f"output exists: {output}")
    for path in (osm, population, terrain, template_path):
        if not path.is_file(): raise RuntimeError(f"missing static source: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".statics-", suffix=".h5", dir=output.parent); os.close(fd)
    tmp = Path(name)
    try:
        with rasterio.open(template_path) as template, rasterio.open(osm) as o, rasterio.open(population) as p, rasterio.open(terrain) as t:
            for dataset, path, count in ((o, osm, 4), (p, population, 1), (t, terrain, 7)):
                _check_grid(dataset, template, path)
                if dataset.count != count: raise RuntimeError(f"{path}: expected {count} bands, got {dataset.count}")
            planes = np.concatenate((o.read().astype("float32"), p.read().astype("float32"), t.read().astype("float32")))
            encoded = _encode_normalized(planes, output)
            if any(np.all(encoded[index] == 255) for index in range(12)): raise RuntimeError("a static plane is all fill")
            strings = h5py.string_dtype("utf-8")
            with h5py.File(tmp, "w") as h5:
                h5.create_dataset("data", data=encoded, compression="gzip", shuffle=True)
                h5.create_dataset("channel_names", data=np.asarray(STATIC_NAMES, dtype=object), dtype=strings)
                h5.attrs.update({
                    "schema_version": 1, "channel_registry_version": registry.REGISTRY_VERSION,
                    "crs": template.crs.to_string(), "transform": tuple(template.transform)[:6],
                    "height": template.height, "width": template.width,
                    "nodata": 255, "encoding": "normalized [0,1] mapped to uint8",
                    "sources": json.dumps({str(x): sha256(x) for x in (osm, population, terrain)}, sort_keys=True),
                })
        os.replace(tmp, output)
    finally:
        tmp.unlink(missing_ok=True)
    return output


@app.command()
def main(
    output: Annotated[Path, typer.Option("--output")],
    osm: Annotated[Path, typer.Option("--osm")] = DEFAULT_OSM,
    population: Annotated[Path, typer.Option("--population")] = DEFAULT_POPULATION,
    terrain: Annotated[Path, typer.Option("--terrain")] = DEFAULT_TERRAIN,
    template: Annotated[Path, typer.Option("--grid-template")] = Path(paths.GRID_TEMPLATE),
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    typer.echo(build_static_package(osm=osm, population=population, terrain=terrain, template_path=template, output=output, overwrite=overwrite))


if __name__ == "__main__": app()
