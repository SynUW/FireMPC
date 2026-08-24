#!/usr/bin/env python3
"""Label-free multivariate weather and bounded CFFDRS state corrections."""

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
)
from ifs_to_era5_physical_adapter import (
    DIRECT_INDEX,
    MISSING_ERA5_ONLY_CHANNELS,
    PHYSICAL_LIMITS,
    latitude_band,
    season_for_month,
)


MULTIVARIATE_CHANNELS = (1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 18)
MULTIVARIATE_INDICES = tuple(DIRECT_INDEX[channel] for channel in MULTIVARIATE_CHANNELS)


def apply_physical_constraints(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    for channel in DIRECT:
        low, high = PHYSICAL_LIMITS[channel]
        index = DIRECT_INDEX[channel]
        result[:, index] = np.clip(result[:, index], low, high)
    result[:, DIRECT_INDEX[13]] = np.maximum(
        result[:, DIRECT_INDEX[13]],
        result[:, DIRECT_INDEX[1]],
    )
    result[:, DIRECT_INDEX[7]] = np.minimum(
        result[:, DIRECT_INDEX[7]],
        result[:, DIRECT_INDEX[1]],
    )
    result[:, DIRECT_INDEX[5]] = np.maximum(
        result[:, DIRECT_INDEX[5]],
        0.0,
    )
    return result.astype(np.float32)


def date_int_offset(target_date: int, offset_days: int) -> int:
    day = datetime.strptime(str(int(target_date)), "%Y%m%d")
    return int((day + timedelta(days=offset_days)).strftime("%Y%m%d"))


def relative_humidity(max_temperature_k: np.ndarray, dewpoint_k: np.ndarray) -> np.ndarray:
    temperature_c = np.asarray(max_temperature_k) - 273.15
    dewpoint_c = np.minimum(dewpoint_k, max_temperature_k) - 273.15
    log_ratio = (
        17.625 * dewpoint_c / np.maximum(243.04 + dewpoint_c, 1e-3)
        - 17.625
        * temperature_c
        / np.maximum(243.04 + temperature_c, 1e-3)
    )
    return np.clip(100 * np.exp(np.clip(log_ratio, -20, 5)), 0, 100)


def compute_codes(
    previous_states: np.ndarray,
    direct_weather: np.ndarray,
    date_ints: np.ndarray,
) -> np.ndarray:
    previous = np.asarray(previous_states, dtype=np.float64)
    weather = np.asarray(direct_weather, dtype=np.float64)
    dates = np.asarray(date_ints, dtype=np.int64)
    temperature_c = weather[:, DIRECT_INDEX[13]] - 273.15
    humidity = relative_humidity(
        weather[:, DIRECT_INDEX[13]],
        weather[:, DIRECT_INDEX[7]],
    )
    wind = np.sqrt(
        weather[:, DIRECT_INDEX[2]] ** 2
        + weather[:, DIRECT_INDEX[3]] ** 2
    ) * 3.6
    rain = np.maximum(weather[:, DIRECT_INDEX[5]] * 1000, 0)
    months = (dates // 100) % 100
    ffmc = np.empty(len(weather), dtype=np.float64)
    dmc = np.empty_like(ffmc)
    dc = np.empty_like(ffmc)
    for month in np.unique(months):
        selected = months == month
        ffmc[selected] = ffmc_next(
            previous[selected, 0],
            temperature_c[selected],
            humidity[selected],
            wind[selected],
            rain[selected],
        )
        dmc[selected] = dmc_next(
            previous[selected, 1],
            temperature_c[selected],
            humidity[selected],
            rain[selected],
            int(month),
        )
        dc[selected] = dc_next(
            previous[selected, 2],
            temperature_c[selected],
            rain[selected],
            int(month),
        )
    isi, bui, fwi = derived_codes(ffmc, dmc, dc, wind)
    return np.column_stack((ffmc, dmc, dc, isi, bui, fwi)).astype(np.float32)


class CFFDRSAwareCorrection:
    def __init__(self, payload: dict):
        self.payload = payload
        self.boundaries = [
            float(value)
            for value in payload["protocol"]["latitude_band_boundaries"]
        ]
        self.weather_alpha = float(payload["weather_alpha"])
        self.weather = payload["weather_models"]
        self.state_beta = float(payload.get("state_beta", 0.0))
        self.state = payload.get("state_models", {})

    @classmethod
    def from_json(cls, path: str | Path) -> "CFFDRSAwareCorrection":
        with open(path, "r", encoding="utf-8") as stream:
            return cls(json.load(stream))

    def keys(
        self,
        date_ints: np.ndarray,
        rows: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dates = np.asarray(date_ints, dtype=np.int64)
        row_values = np.asarray(rows, dtype=np.int64)
        months = (dates // 100) % 100
        seasons = np.asarray(
            [season_for_month(int(month)) for month in months]
        )
        bands = latitude_band(row_values, self.boundaries)
        keys = np.asarray(
            [f"{season}|{int(band)}" for season, band in zip(seasons, bands)]
        )
        return keys, months, bands

    def correct_weather(
        self,
        baseline_values: np.ndarray,
        date_ints: np.ndarray,
        rows: np.ndarray,
    ) -> np.ndarray:
        baseline = np.asarray(baseline_values, dtype=np.float64)
        output = baseline.copy()
        keys, _, _ = self.keys(date_ints, rows)
        for key in np.unique(keys):
            selected = keys == key
            selected_rows = np.flatnonzero(selected)
            spec = self.weather.get(key, self.weather["global"])
            x = baseline[selected][:, MULTIVARIATE_INDICES]
            z = (
                x - np.asarray(spec["x_mean"], dtype=np.float64)
            ) / np.asarray(spec["x_scale"], dtype=np.float64)
            residual_standardized = (
                z @ np.asarray(spec["coef"], dtype=np.float64)
                + np.asarray(spec["intercept"], dtype=np.float64)
            )
            residual = residual_standardized * np.asarray(
                spec["y_scale"], dtype=np.float64
            )
            output[np.ix_(selected_rows, MULTIVARIATE_INDICES)] += (
                self.weather_alpha * residual
            )
        return apply_physical_constraints(output)

    def correct_states(
        self,
        provisional_states: np.ndarray,
        direct_weather: np.ndarray,
        date_ints: np.ndarray,
        rows: np.ndarray,
    ) -> np.ndarray:
        states = np.asarray(provisional_states, dtype=np.float64)
        if self.state_beta <= 0 or not self.state:
            return states.astype(np.float32)
        weather = np.asarray(direct_weather, dtype=np.float64)
        output = states.copy()
        keys, _, _ = self.keys(date_ints, rows)
        for key in np.unique(keys):
            selected = keys == key
            spec = self.state.get(key, self.state["global"])
            features = np.column_stack(
                (
                    states[selected],
                    weather[selected][:, MULTIVARIATE_INDICES],
                )
            )
            z = (
                features - np.asarray(spec["x_mean"], dtype=np.float64)
            ) / np.asarray(spec["x_scale"], dtype=np.float64)
            residual_standardized = (
                z @ np.asarray(spec["coef"], dtype=np.float64)
                + np.asarray(spec["intercept"], dtype=np.float64)
            )
            residual = residual_standardized * np.asarray(
                spec["y_scale"], dtype=np.float64
            )
            bounds = np.asarray(spec["bounds"], dtype=np.float64)
            residual = np.clip(residual, -bounds, bounds)
            output[selected] += self.state_beta * residual
        output[:, 0] = np.clip(output[:, 0], 0, 101)
        output[:, 1:] = np.maximum(output[:, 1:], 0)
        return output.astype(np.float32)


def transform_corrected_window(
    encoded_window: np.ndarray,
    target_date: int,
    center_row: int,
    initial_cffdrs_encoded: np.ndarray,
    correction: CFFDRSAwareCorrection,
    use_state_correction: bool,
) -> np.ndarray:
    result = np.asarray(encoded_window, dtype=np.float32).copy()
    height, width = result.shape[1:3]
    valid = result[1, :, :, 0] > 0
    row_grid = (
        int(center_row) + np.arange(height, dtype=np.int64)[:, None] - height // 2
    )
    row_grid = np.broadcast_to(row_grid, (height, width))
    initial = np.asarray(initial_cffdrs_encoded, dtype=np.float32)
    previous = np.column_stack(
        [decode(initial[channel][valid], 19 + channel) for channel in range(3)]
    )

    for time_index in range(10):
        current_date = date_int_offset(target_date, time_index - 10)
        matrix = np.stack(
            [
                decode(result[channel, :, :, time_index], channel)
                for channel in DIRECT
            ],
            axis=-1,
        )
        dates = np.full(int(valid.sum()), current_date, dtype=np.int64)
        weather = correction.correct_weather(
            matrix[valid],
            dates,
            row_grid[valid],
        )
        for direct_index, channel in enumerate(DIRECT):
            encoded = np.zeros((height, width), dtype=np.float32)
            encoded[valid] = encode(weather[:, direct_index], channel)
            result[channel, :, :, time_index] = encoded

        codes = compute_codes(previous, weather, dates)
        states = codes[:, :3]
        if use_state_correction:
            states = correction.correct_states(
                states,
                weather,
                dates,
                row_grid[valid],
            )
            wind = np.sqrt(
                weather[:, DIRECT_INDEX[2]] ** 2
                + weather[:, DIRECT_INDEX[3]] ** 2
            ) * 3.6
            isi, bui, fwi = derived_codes(
                states[:, 0],
                states[:, 1],
                states[:, 2],
                wind,
            )
            codes = np.column_stack((states, isi, bui, fwi))
        for code_index, channel in enumerate(CFFDRS):
            encoded = np.zeros((height, width), dtype=np.float32)
            encoded[valid] = encode(codes[:, code_index], channel)
            result[channel, :, :, time_index] = encoded

        mean_temperature_k = weather[:, DIRECT_INDEX[1]]
        dewpoint_k = np.minimum(
            weather[:, DIRECT_INDEX[7]],
            mean_temperature_k,
        )
        temperature_c = mean_temperature_k - 273.15
        dewpoint_c = dewpoint_k - 273.15
        saturation = 0.6108 * np.exp(
            17.27 * temperature_c
            / np.maximum(temperature_c + 237.3, 1e-3)
        )
        actual = 0.6108 * np.exp(
            17.27 * dewpoint_c
            / np.maximum(dewpoint_c + 237.3, 1e-3)
        )
        encoded_vpd = np.zeros((height, width), dtype=np.float32)
        encoded_vpd[valid] = encode(
            np.maximum(saturation - actual, 0.0),
            25,
        )
        result[25, :, :, time_index] = encoded_vpd
        previous = states

    result[list(MISSING_ERA5_ONLY_CHANNELS), :, :, :] = 0.0
    return result
