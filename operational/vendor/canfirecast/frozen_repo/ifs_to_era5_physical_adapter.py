#!/usr/bin/env python3
"""Physics-aware IFS-to-ERA5-Land adapter used at inference time."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from full10_ifs_domain_transform import (
    CFFDRS,
    DIRECT,
    dc_next,
    decode,
    derived_codes,
    dmc_next,
    encode,
    ffmc_next,
    month_for_day,
    recompute_common_derived,
)


DIRECT_INDEX = {channel: index for index, channel in enumerate(DIRECT)}
SCALAR_CHANNELS = (1, 8, 9, 10, 11, 12, 13, 14, 18)
SOIL_CHANNELS = (9, 10, 11, 12)
MISSING_ERA5_ONLY_CHANNELS = (4, 6, 15, 16, 17)
PHYSICAL_LIMITS = {
    1: (180.0, 330.0),
    2: (-60.0, 60.0),
    3: (-60.0, 60.0),
    5: (0.0, 0.25),
    7: (170.0, 330.0),
    8: (50_000.0, 110_000.0),
    9: (0.0, 1.0),
    10: (0.0, 1.0),
    11: (0.0, 1.0),
    12: (0.0, 1.0),
    13: (180.0, 340.0),
    14: (180.0, 350.0),
    18: (0.0, 5.0e7),
}


def season_for_month(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def date_int_offset(target_date: int, offset_days: int) -> int:
    day = datetime.strptime(str(int(target_date)), "%Y%m%d")
    return int((day + timedelta(days=offset_days)).strftime("%Y%m%d"))


def latitude_band(rows: np.ndarray, boundaries: list[float]) -> np.ndarray:
    return np.searchsorted(
        np.asarray(boundaries[1:-1], dtype=np.float64),
        np.asarray(rows, dtype=np.float64),
        side="right",
    ).astype(np.int8)


def monotone_quantile_map(values: np.ndarray, spec: dict) -> np.ndarray:
    source = np.asarray(spec["source_quantiles"], dtype=np.float64)
    target = np.asarray(spec["target_quantiles"], dtype=np.float64)
    if source.size < 2 or target.size != source.size:
        raise ValueError("Invalid quantile-map knots")
    return np.interp(np.asarray(values, dtype=np.float64), source, target)


def _clip(values: np.ndarray, channel: int) -> np.ndarray:
    low, high = PHYSICAL_LIMITS[channel]
    return np.clip(values, low, high)


def blend_with_affine(
    source: np.ndarray,
    physical: np.ndarray,
    spec: dict,
) -> np.ndarray:
    if "affine_slope" not in spec:
        return physical
    affine = (
        float(spec["affine_slope"]) * np.asarray(source, dtype=np.float64)
        + float(spec["affine_intercept"])
    )
    alpha = float(spec.get("physics_blend_alpha", 1.0))
    return affine + alpha * (np.asarray(physical, dtype=np.float64) - affine)


def recompute_derived_with_initial_state(
    x: np.ndarray,
    target_date: int,
    initial_cffdrs_encoded: np.ndarray,
) -> np.ndarray:
    """Recompute VPD/CFFDRS from IFS using the state preceding the ten-day window."""
    result = np.asarray(x, dtype=np.float32).copy()
    initial = np.asarray(initial_cffdrs_encoded, dtype=np.float32)
    if initial.shape != (3, result.shape[1], result.shape[2]):
        raise ValueError(
            "Initial CFFDRS state must be (3,H,W), received "
            f"{initial.shape}"
        )
    valid = result[1, :, :, 0] > 0
    result[list(MISSING_ERA5_ONLY_CHANNELS), :, :, :] = 0.0
    ffmc_previous = decode(initial[0], 19)
    dmc_previous = decode(initial[1], 20)
    dc_previous = decode(initial[2], 21)
    ffmc_previous = np.where(valid, ffmc_previous, 85.0)
    dmc_previous = np.where(valid, dmc_previous, 6.0)
    dc_previous = np.where(valid, dc_previous, 15.0)

    for day in range(10):
        vpd_temperature_k = decode(result[1, :, :, day], 1)
        fire_temperature_k = decode(result[13, :, :, day], 13)
        dewpoint_k = decode(result[7, :, :, day], 7)
        vpd_temperature_c = vpd_temperature_k - 273.15
        fire_temperature_c = fire_temperature_k - 273.15
        vpd_dewpoint_c = np.minimum(dewpoint_k, vpd_temperature_k) - 273.15
        fire_dewpoint_c = np.minimum(dewpoint_k, fire_temperature_k) - 273.15
        saturation = 0.6108 * np.exp(
            17.27 * vpd_temperature_c
            / np.maximum(vpd_temperature_c + 237.3, 1e-3)
        )
        actual = 0.6108 * np.exp(
            17.27 * vpd_dewpoint_c
            / np.maximum(vpd_dewpoint_c + 237.3, 1e-3)
        )
        result[25, :, :, day] = np.where(
            valid,
            encode(np.maximum(saturation - actual, 0.0), 25),
            0.0,
        )

        log_ratio = (
            17.625
            * fire_dewpoint_c
            / np.maximum(243.04 + fire_dewpoint_c, 1e-3)
            - 17.625
            * fire_temperature_c
            / np.maximum(243.04 + fire_temperature_c, 1e-3)
        )
        humidity = np.clip(
            100 * np.exp(np.clip(log_ratio, -20, 5)),
            0,
            100,
        )
        u_wind = decode(result[2, :, :, day], 2)
        v_wind = decode(result[3, :, :, day], 3)
        wind = np.sqrt(u_wind * u_wind + v_wind * v_wind) * 3.6
        rain = np.maximum(decode(result[5, :, :, day], 5) * 1000, 0)
        month = month_for_day(target_date, day)
        ffmc = ffmc_next(
            ffmc_previous,
            fire_temperature_c,
            humidity,
            wind,
            rain,
        )
        dmc = dmc_next(
            dmc_previous,
            fire_temperature_c,
            humidity,
            rain,
            month,
        )
        dc = dc_next(dc_previous, fire_temperature_c, rain, month)
        isi, bui, fwi = derived_codes(ffmc, dmc, dc, wind)
        for channel, values in zip(
            CFFDRS,
            (ffmc, dmc, dc, isi, bui, fwi),
        ):
            result[channel, :, :, day] = np.where(
                valid,
                encode(values, channel),
                0.0,
            )
        ffmc_previous = np.where(valid, ffmc, ffmc_previous)
        dmc_previous = np.where(valid, dmc, dmc_previous)
        dc_previous = np.where(valid, dc, dc_previous)
    return result


def apply_stratum(
    values: np.ndarray,
    stratum: dict,
) -> np.ndarray:
    """Map a matrix in DIRECT channel order for one season/latitude stratum."""
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != len(DIRECT):
        raise ValueError(f"Expected (N,{len(DIRECT)}) direct matrix, got {source.shape}")
    mapped = source.copy()

    scalar = stratum["scalar"]
    for channel in SCALAR_CHANNELS:
        index = DIRECT_INDEX[channel]
        physical = monotone_quantile_map(source[:, index], scalar[str(channel)])
        mapped[:, index] = _clip(
            blend_with_affine(
                source[:, index],
                physical,
                scalar[str(channel)],
            ),
            channel,
        )

    # Preserve the physical relation between mean and maximum air temperature.
    mapped[:, DIRECT_INDEX[13]] = np.maximum(
        mapped[:, DIRECT_INDEX[13]],
        mapped[:, DIRECT_INDEX[1]],
    )

    dewpoint = stratum["dewpoint"]
    source_depression = np.maximum(
        source[:, DIRECT_INDEX[1]] - source[:, DIRECT_INDEX[7]],
        0.0,
    )
    mapped_depression = np.maximum(
        monotone_quantile_map(source_depression, dewpoint),
        0.0,
    )
    physical_dewpoint = mapped[:, DIRECT_INDEX[1]] - mapped_depression
    mapped_dewpoint = blend_with_affine(
        source[:, DIRECT_INDEX[7]],
        physical_dewpoint,
        dewpoint,
    )
    mapped[:, DIRECT_INDEX[7]] = _clip(
        np.minimum(mapped_dewpoint, mapped[:, DIRECT_INDEX[1]]),
        7,
    )

    precipitation = stratum["precipitation"]
    raw_precip = np.maximum(source[:, DIRECT_INDEX[5]], 0.0)
    wet = raw_precip > float(precipitation["ifs_wet_threshold"])
    corrected_precip = np.zeros_like(raw_precip)
    if np.any(wet):
        physical_amount = monotone_quantile_map(
            raw_precip[wet],
            precipitation,
        )
        corrected_precip[wet] = blend_with_affine(
            raw_precip[wet],
            physical_amount,
            precipitation,
        )
        corrected_precip[wet] = np.maximum(
            corrected_precip[wet],
            float(precipitation["era_wet_threshold"]) * 1.001,
        )
    mapped[:, DIRECT_INDEX[5]] = _clip(corrected_precip, 5)

    wind = stratum["wind"]
    design = np.column_stack(
        (
            source[:, DIRECT_INDEX[2]],
            source[:, DIRECT_INDEX[3]],
            np.ones(len(source), dtype=np.float64),
        )
    )
    vector = design @ np.asarray(wind["matrix"], dtype=np.float64)
    raw_speed = np.linalg.norm(vector, axis=1)
    corrected_speed = np.maximum(
        monotone_quantile_map(raw_speed, wind),
        0.0,
    )
    scale = np.divide(
        corrected_speed,
        raw_speed,
        out=np.zeros_like(corrected_speed),
        where=raw_speed > 1e-8,
    )
    vector *= scale[:, None]
    if "affine_matrix" in wind:
        affine_vector = design @ np.asarray(
            wind["affine_matrix"],
            dtype=np.float64,
        )
        alpha = float(wind.get("physics_blend_alpha", 1.0))
        vector = affine_vector + alpha * (vector - affine_vector)
    mapped[:, DIRECT_INDEX[2]] = _clip(vector[:, 0], 2)
    mapped[:, DIRECT_INDEX[3]] = _clip(vector[:, 1], 3)
    return mapped.astype(np.float32)


class IFSToERA5Adapter:
    def __init__(self, payload: dict):
        self.payload = payload
        self.boundaries = [
            float(value) for value in payload["protocol"]["latitude_band_boundaries"]
        ]
        self.strata = payload["strata"]
        self.global_stratum = payload["global"]

    @classmethod
    def from_json(cls, path: str | Path) -> "IFSToERA5Adapter":
        with open(path, "r", encoding="utf-8") as stream:
            return cls(json.load(stream))

    def map_pairs(
        self,
        direct_values: np.ndarray,
        date_ints: np.ndarray,
        rows: np.ndarray,
    ) -> np.ndarray:
        source = np.asarray(direct_values, dtype=np.float32)
        dates = np.asarray(date_ints, dtype=np.int64).reshape(-1)
        row_values = np.asarray(rows, dtype=np.int64).reshape(-1)
        if len(source) != len(dates) or len(source) != len(row_values):
            raise ValueError("Pair matrix, dates, and rows must have equal length")
        output = np.empty_like(source, dtype=np.float32)
        months = (dates // 100) % 100
        bands = latitude_band(row_values, self.boundaries)
        assigned = np.zeros(len(source), dtype=bool)
        for month in range(1, 13):
            season = season_for_month(month)
            for band in range(len(self.boundaries) - 1):
                mask = (months == month) & (bands == band)
                if not np.any(mask):
                    continue
                spec = self.strata.get(f"{season}|{band}", self.global_stratum)
                output[mask] = apply_stratum(source[mask], spec)
                assigned[mask] = True
        if not np.all(assigned):
            raise RuntimeError(f"Unassigned adapter rows: {int((~assigned).sum())}")
        return output

    def map_window(
        self,
        encoded_window: np.ndarray,
        target_date: int,
        center_row: int,
        initial_cffdrs_encoded: np.ndarray | None = None,
    ) -> np.ndarray:
        result = np.asarray(encoded_window, dtype=np.float32).copy()
        if result.ndim != 4 or result.shape[0] < 55 or result.shape[-1] != 10:
            raise ValueError(f"Unexpected window shape: {result.shape}")
        height, width = result.shape[1:3]
        valid = result[1, :, :, 0] > 0
        row_grid = (
            int(center_row)
            + np.arange(height, dtype=np.int64)[:, None]
            - height // 2
        )
        row_grid = np.broadcast_to(row_grid, (height, width))
        for time_index in range(10):
            date_int = date_int_offset(target_date, time_index - 10)
            matrix = np.stack(
                [
                    decode(result[channel, :, :, time_index], channel)
                    for channel in DIRECT
                ],
                axis=-1,
            )
            mapped = self.map_pairs(
                matrix[valid],
                np.full(int(valid.sum()), date_int, dtype=np.int64),
                row_grid[valid],
            )
            for direct_index, channel in enumerate(DIRECT):
                encoded = np.zeros((height, width), dtype=np.float32)
                encoded[valid] = encode(mapped[:, direct_index], channel)
                result[channel, :, :, time_index] = encoded
        if initial_cffdrs_encoded is None:
            result = recompute_common_derived(result, int(target_date))
        else:
            result = recompute_derived_with_initial_state(
                result,
                int(target_date),
                initial_cffdrs_encoded,
            )
        result[list(MISSING_ERA5_ONLY_CHANNELS), :, :, :] = 0.0
        return result
