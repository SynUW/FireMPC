"""Synthetic weather tiles shared by the weather tests.

These used to live in test_derive.py, which test_preprocess.py imported as
`from tests.test_weather_derive import ...` -- a test module importing another
test module, which only resolved because the repository root was on the path.
They are plain helpers rather than fixtures because callers pass arguments.
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_origin

from weather import registry as registry

PIXEL = registry.RESOLUTION_M / 111319.49079327358


def write_source_tile(path, source, values: dict[str, float], shape=(6, 8),
                      origin=(-141.0, 64.0), name_bands=False):
    """A synthetic tile for one source, with a constant value per named band."""
    profile = {
        "driver": "GTiff", "dtype": "float32", "count": len(source.bands),
        "height": shape[0], "width": shape[1], "crs": "EPSG:4326",
        "transform": from_origin(origin[0], origin[1], PIXEL, PIXEL),
        "nodata": float("-inf"),
    }
    with rasterio.open(path, "w", **profile) as dataset:
        for band in source.bands:
            data = np.full(shape, values.get(band.name, 1.0), dtype="float32")
            dataset.write(data, band.index)
        if name_bands:
            dataset.descriptions = source.band_names
    return path


def write_ifs_tile(path, values: dict[str, float], shape=(6, 8), origin=(-141.0, 64.0)):
    """A synthetic 23-band IFS tile with a constant value per named band."""
    return write_source_tile(path, registry.IFS, values, shape=shape, origin=origin)
