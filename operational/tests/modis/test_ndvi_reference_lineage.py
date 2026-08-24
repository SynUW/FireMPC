import os
from pathlib import Path

import h5py
import numpy as np
import pytest
import rasterio

from wildfire.modis.processing_registry import CHANNELS_BY_PRODUCT

REFERENCE_ROOT = Path(os.environ.get("WILDFIRE_MODIS_LINEAGE_ROOT", ""))
REFLECTANCE = (
    REFERENCE_ROOT
    / "MCD09GA_b1237_QAapplied_spatialFilled_downsampled_merged"
    / "2025_07_01.tif"
)
ORACLE = (
    REFERENCE_ROOT
    / "NDVI_EVI_QAapplied_spatialFilled_downsampled_merged_float"
    / "2025_07_01.tif"
)
REFERENCE_H5 = Path(os.environ.get("WILDFIRE_MODIS_LINEAGE_H5", ""))


def _firempc_uint8(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    invalid = (
        ~np.isfinite(values)
        | (values == -32768)
        | (values == 32768)
        | (values == 255)
    )
    result = np.full(values.shape, 255, dtype=np.uint8)
    scaled = (
        (values[~invalid].astype(np.float32) - np.float32(vmin))
        / np.float32(vmax - vmin)
        * np.float32(255.0)
    )
    result[~invalid] = np.clip(scaled, 0, 255).astype(np.uint8)
    return result


@pytest.mark.skipif(
    not REFLECTANCE.is_file() or not ORACLE.is_file(),
    reason="read-only FireMPC reference tree is unavailable",
)
def test_scaled_national_reflectance_reproduces_ndvi_evi_oracle_exactly():
    ranges = [channel.raw_range for channel in CHANNELS_BY_PRODUCT["MCD09GA"]]
    with rasterio.open(REFLECTANCE) as source:
        reflectance = source.read().astype(np.float32)
    scaled = np.stack(
        [
            _firempc_uint8(band, *value_range)
            for band, value_range in zip(reflectance, ranges, strict=True)
        ]
    ).astype(np.float32) / np.float32(255.0)
    red, nir, blue = scaled[:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = np.clip(
            ((nir - red).astype(np.float64) / (nir + red).astype(np.float64)).astype(np.float32),
            -1.0,
            1.0,
        )
        evi = np.clip((
            2.5 * (nir - red).astype(np.float64)
            / (
                nir.astype(np.float64) + 6.0 * red.astype(np.float64)
                - 7.5 * blue.astype(np.float64) + 1.0
            )
        ).astype(np.float32), -1.0, 1.0)
    with rasterio.open(ORACLE) as source:
        expected = source.read()
        nodata = source.nodata
    for actual, oracle in zip((ndvi, evi), expected, strict=True):
        valid = np.isfinite(actual) & np.isfinite(oracle) & (oracle != nodata)
        assert np.array_equal(actual[valid], oracle[valid])


@pytest.mark.skipif(
    not ORACLE.is_file() or not REFERENCE_H5.is_file(),
    reason="read-only FireMPC reference cube is unavailable",
)
def test_scaled_indices_match_reference_h5_channels():
    with rasterio.open(ORACLE) as source:
        indices = source.read()
    ranges = [channel.raw_range for channel in CHANNELS_BY_PRODUCT["NDVIEVI"]]
    scaled = np.stack(
        [
            _firempc_uint8(band, *value_range)
            for band, value_range in zip(indices, ranges, strict=True)
        ]
    )
    generator = np.random.default_rng(20250701)
    flat = np.sort(generator.choice(scaled.shape[1] * scaled.shape[2], size=100, replace=False))
    rows, columns = np.divmod(flat, scaled.shape[2])
    with h5py.File(REFERENCE_H5) as cube:
        assert cube.attrs[
            "channel_mapping_NDVI_EVI_QAapplied_spatialFilled_downsampled_merged_float"
        ] == "38-39"
        time_index = 181  # 2025-07-01 relative to the cube's 2025-01-01 start.
        for row, column in zip(rows, columns, strict=True):
            expected = cube[f"{int(row)}_{int(column)}"][38:40, time_index]
            assert np.array_equal(scaled[:, row, column], expected)
