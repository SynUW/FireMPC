#!/usr/bin/env python3
"""Shared transforms for a common ten-day IFS deployment domain."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np


DIRECT = (1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 18)
IMPUTED = (4, 6, 15, 16, 17)
CFFDRS = (19, 20, 21, 22, 23, 24)
RANGES = {
    1: (0.005193258648563279, 297.7777831997186),
    2: (-8.035555467392681, 9.75282316575327),
    3: (-8.884559637656066, 7.783394794594883),
    5: (1.9188684693810958e-7, 0.03603348170585861),
    7: (0.005005088835593607, 291.85923360787075),
    8: (1.7814644058506481, 102594.00284017395),
    9: (1.2825456326856735e-5, 0.7495786155920634),
    10: (1.2824859849846285e-5, 0.7429968109043169),
    11: (1.2793656195784068e-5, 0.7264044759278405),
    12: (1.2811615921890935e-5, 0.7151072856451576),
    13: (0.005317793384206413, 304.2753351358254),
    14: (0.005381247822265205, 305.6516323146297),
    18: (413.45007501667135, 28043026.67241847),
    19: (0.0, 101.0),
    20: (0.0, 119.0),
    21: (0.0, 694.0),
    22: (0.0, 11.0),
    23: (0.0, 149.0),
    24: (0.0, 33.0),
    25: (0.0, 1.63),
}


def decode(values: np.ndarray, channel: int) -> np.ndarray:
    low, high = RANGES[channel]
    return low + np.asarray(values, dtype=np.float32) * (high - low) / 255.0


def encode(values: np.ndarray, channel: int) -> np.ndarray:
    low, high = RANGES[channel]
    return np.clip((np.asarray(values, dtype=np.float32) - low) * 255.0 / (high - low), 0, 255)


def month_for_day(target_date: int, day: int) -> int:
    current = datetime.strptime(str(int(target_date)), "%Y%m%d") + timedelta(days=day - 9)
    return current.month


def ffmc_next(previous, temperature, humidity, wind, rain):
    moisture = 147.2 * (101 - previous) / (59.5 + previous)
    safe_rain = np.maximum(rain, 1e-6)
    wet = moisture + 42.5 * rain * np.exp(-100 / np.maximum(251 - moisture, 1e-6)) * (
        1 - np.exp(-6.93 / safe_rain)
    )
    wet = np.clip(np.where(rain > 0.5, wet, moisture), 0, 250)
    equilibrium_dry = (
        0.942 * humidity**0.679
        + 11 * np.exp((humidity - 100) / 10)
        + 0.18 * (21.1 - temperature) * (1 - np.exp(-0.115 * humidity))
    )
    equilibrium_wet = (
        0.618 * humidity**0.753
        + 10 * np.exp((humidity - 100) / 10)
        + 0.18 * (21.1 - temperature) * (1 - np.exp(-0.115 * humidity))
    )
    dry_rate = (
        0.424 * (1 - (humidity / 100) ** 1.7)
        + 0.0694 * np.sqrt(np.maximum(wind, 0)) * (1 - (humidity / 100) ** 8)
    ) * 0.581 * np.exp(0.0365 * temperature)
    wet_rate = (
        0.424 * (1 - ((100 - humidity) / 100) ** 1.7)
        + 0.0694 * np.sqrt(np.maximum(wind, 0)) * (1 - ((100 - humidity) / 100) ** 8)
    ) * 0.581 * np.exp(0.0365 * temperature)
    moisture = np.where(
        wet > equilibrium_dry,
        equilibrium_dry + (wet - equilibrium_dry) * np.exp(-dry_rate),
        np.where(
            wet < equilibrium_wet,
            equilibrium_wet - (equilibrium_wet - wet) * np.exp(-wet_rate),
            wet,
        ),
    )
    return np.clip(59.5 * (250 - moisture) / (147.2 + moisture), 0, 101)


def dmc_next(previous, temperature, humidity, rain, month):
    day_length = {4: 0.9, 5: 12.5, 6: 13.9, 7: 13.9, 8: 12.4, 9: 10.9, 10: 9.4}.get(month, 9.0)
    effective_rain = np.where(rain > 1.5, 0.92 * rain - 1.27, 0)
    moisture = 20 + np.exp(5.6348 - previous / 43.43)
    safe_previous = np.maximum(previous, 1e-6)
    coefficient = np.where(
        previous <= 33,
        100 / (0.5 + 0.3 * previous),
        np.where(
            previous <= 65,
            14 - 1.3 * np.log(safe_previous),
            6.2 * np.log(safe_previous) - 17.2,
        ),
    )
    rain_moisture = moisture + 1000 * effective_rain / (48.77 + coefficient * effective_rain)
    rain_code = 244.72 - 43.43 * np.log(np.maximum(rain_moisture - 20, 1e-6))
    rain_code = np.where(rain > 1.5, np.maximum(rain_code, 0), previous)
    drying = np.where(
        temperature > -1.1,
        1.894 * (temperature + 1.1) * (100 - humidity) * day_length * 1e-6,
        0,
    )
    return np.maximum(rain_code + 100 * drying, 0)


def dc_next(previous, temperature, rain, month):
    day_length = {4: -1.6, 5: 1.2, 6: 4.0, 7: 4.0, 8: 3.0, 9: 0.4, 10: -2.0}.get(month, -1.6)
    effective_rain = np.where(rain > 2.8, 0.83 * rain - 1.27, 0)
    moisture = 800 * np.exp(-previous / 400) + 3.937 * effective_rain
    rain_code = np.where(rain > 2.8, 400 * np.log(800 / np.maximum(moisture, 1e-6)), previous)
    drying = np.where(temperature > -2.8, 0.36 * (temperature + 2.8) + day_length, 0)
    return np.maximum(np.maximum(rain_code, 0) + 0.5 * np.maximum(drying, 0), 0)


def derived_codes(ffmc, dmc, dc, wind):
    moisture = 147.2 * (101 - ffmc) / (59.5 + ffmc)
    isi = (
        0.208
        * np.exp(0.05039 * wind)
        * 91.9
        * np.exp(-0.1386 * moisture)
        * (1 + moisture**5.31 / 4.93e7)
    )
    denominator = dmc + 0.4 * dc
    bui = np.where(
        denominator < 1e-6,
        0,
        np.where(
            dmc <= 0.4 * dc,
            0.8 * dmc * dc / np.maximum(denominator, 1e-6),
            dmc
            - (1 - 0.8 * dc / np.maximum(denominator, 1e-6))
            * (0.92 + (0.0114 * dmc) ** 1.7),
        ),
    )
    bui = np.maximum(bui, 0)
    base = 0.1 * isi * (0.626 * bui**0.809 + 2)
    fwi = base.copy()
    mask = base > 1
    fwi[mask] = np.exp(2.72 * (0.434 * np.log(base[mask])) ** 0.647)
    return np.maximum(isi, 0), bui, np.maximum(fwi, 0)


class MissingFieldImputer:
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        for key in ("x_mean", "x_std", "coef", "intercept", "y_low", "y_high"):
            setattr(self, key, data[key].astype(np.float32))

    def feature(self, x: np.ndarray, day: int, target_date: int) -> np.ndarray:
        height, width = x.shape[1:3]
        missing_last = x[list(IMPUTED), :, :, 4].transpose(1, 2, 0)
        missing_trend = (
            (x[list(IMPUTED), :, :, 4] - x[list(IMPUTED), :, :, 2]).transpose(1, 2, 0)
            / 2.0
        )
        direct_now = x[list(DIRECT), :, :, day].transpose(1, 2, 0)
        direct_anchor = x[list(DIRECT), :, :, 4].transpose(1, 2, 0)
        direct_change = direct_now - direct_anchor
        current = datetime.strptime(str(int(target_date)), "%Y%m%d") + timedelta(days=day - 9)
        angle = 2 * math.pi * current.timetuple().tm_yday / 365.25
        calendar = np.empty((height, width, 2), dtype=np.float32)
        calendar[..., 0], calendar[..., 1] = math.sin(angle), math.cos(angle)
        land_cover = (x[54, :, :, 4] / 17.0)[..., None]
        return np.concatenate(
            (
                missing_last,
                missing_trend,
                direct_now,
                direct_anchor,
                direct_change,
                calendar,
                land_cover,
            ),
            axis=-1,
        )

    def predict(self, x: np.ndarray, day: int, target_date: int) -> np.ndarray:
        feature = self.feature(x, day, target_date)
        lead = day - 5
        flat = feature.reshape(-1, feature.shape[-1])
        standardized = (flat - self.x_mean[lead]) / self.x_std[lead]
        prediction = standardized @ self.coef[lead] + self.intercept[lead]
        prediction = np.clip(prediction, self.y_low[lead], self.y_high[lead])
        return prediction.reshape(*feature.shape[:2], len(IMPUTED)).transpose(2, 0, 1)


def complete_window(x: np.ndarray, target_date: int, imputer: MissingFieldImputer) -> np.ndarray:
    result = np.asarray(x, dtype=np.float32).copy()
    valid = result[1, :, :, 4] > 0
    for day in range(5, 10):
        imputed = imputer.predict(result, day, target_date)
        result[list(IMPUTED), :, :, day] = np.where(
            valid[None],
            imputed,
            result[list(IMPUTED), :, :, day],
        )

        temperature_k = decode(result[13, :, :, day], 13)
        dewpoint_k = decode(result[7, :, :, day], 7)
        temperature_c = temperature_k - 273.15
        dewpoint_c = dewpoint_k - 273.15
        saturation = 0.6108 * np.exp(
            17.27 * temperature_c / np.maximum(temperature_c + 237.3, 1e-3)
        )
        actual = 0.6108 * np.exp(
            17.27 * dewpoint_c / np.maximum(dewpoint_c + 237.3, 1e-3)
        )
        vpd = np.maximum(saturation - actual, 0)
        result[25, :, :, day] = np.where(valid, encode(vpd, 25), result[25, :, :, day])

        log_ratio = (
            17.625 * dewpoint_c / np.maximum(243.04 + dewpoint_c, 1e-3)
            - 17.625 * temperature_c / np.maximum(243.04 + temperature_c, 1e-3)
        )
        humidity = np.clip(100 * np.exp(np.clip(log_ratio, -20, 5)), 0, 100)
        u = decode(result[2, :, :, day], 2)
        v = decode(result[3, :, :, day], 3)
        wind = np.sqrt(u * u + v * v) * 3.6
        rain = np.maximum(decode(result[5, :, :, day], 5) * 1000, 0)
        previous = day - 1
        ffmc = ffmc_next(decode(result[19, :, :, previous], 19), temperature_c, humidity, wind, rain)
        dmc = dmc_next(
            decode(result[20, :, :, previous], 20),
            temperature_c,
            humidity,
            rain,
            month_for_day(target_date, day),
        )
        dc = dc_next(
            decode(result[21, :, :, previous], 21),
            temperature_c,
            rain,
            month_for_day(target_date, day),
        )
        isi, bui, fwi = derived_codes(ffmc, dmc, dc, wind)
        for channel, values in zip(CFFDRS, (ffmc, dmc, dc, isi, bui, fwi)):
            result[channel, :, :, day] = np.where(
                valid,
                encode(values, channel),
                result[channel, :, :, day],
            )
    return result


def recompute_common_derived(x: np.ndarray, target_date: int) -> np.ndarray:
    """Drop unavailable fields and derive VPD/CFFDRS identically at both ends."""
    result = np.asarray(x, dtype=np.float32).copy()
    valid = result[1, :, :, 0] > 0
    result[list(IMPUTED), :, :, :] = 0.0
    height, width = valid.shape
    ffmc_previous = np.full((height, width), 85.0, dtype=np.float32)
    dmc_previous = np.full((height, width), 6.0, dtype=np.float32)
    dc_previous = np.full((height, width), 15.0, dtype=np.float32)

    for day in range(10):
        temperature_k = decode(result[13, :, :, day], 13)
        dewpoint_k = decode(result[7, :, :, day], 7)
        temperature_c = temperature_k - 273.15
        dewpoint_c = dewpoint_k - 273.15
        saturation = 0.6108 * np.exp(
            17.27 * temperature_c / np.maximum(temperature_c + 237.3, 1e-3)
        )
        actual = 0.6108 * np.exp(
            17.27 * dewpoint_c / np.maximum(dewpoint_c + 237.3, 1e-3)
        )
        vpd = np.maximum(saturation - actual, 0)
        result[25, :, :, day] = np.where(valid, encode(vpd, 25), 0.0)

        log_ratio = (
            17.625 * dewpoint_c / np.maximum(243.04 + dewpoint_c, 1e-3)
            - 17.625 * temperature_c / np.maximum(243.04 + temperature_c, 1e-3)
        )
        humidity = np.clip(100 * np.exp(np.clip(log_ratio, -20, 5)), 0, 100)
        u_wind = decode(result[2, :, :, day], 2)
        v_wind = decode(result[3, :, :, day], 3)
        wind = np.sqrt(u_wind * u_wind + v_wind * v_wind) * 3.6
        rain = np.maximum(decode(result[5, :, :, day], 5) * 1000, 0)
        month = month_for_day(target_date, day)
        ffmc = ffmc_next(ffmc_previous, temperature_c, humidity, wind, rain)
        dmc = dmc_next(dmc_previous, temperature_c, humidity, rain, month)
        dc = dc_next(dc_previous, temperature_c, rain, month)
        isi, bui, fwi = derived_codes(ffmc, dmc, dc, wind)
        for channel, values in zip(CFFDRS, (ffmc, dmc, dc, isi, bui, fwi)):
            result[channel, :, :, day] = np.where(valid, encode(values, channel), 0.0)
        ffmc_previous = np.where(valid, ffmc, ffmc_previous)
        dmc_previous = np.where(valid, dmc, dmc_previous)
        dc_previous = np.where(valid, dc, dc_previous)
    return result


def era5_to_ifs_domain(
    x: np.ndarray,
    target_date: int,
    calibration: dict,
) -> np.ndarray:
    """Map all ten historical ERA5 days into the calibrated IFS domain."""
    result = np.asarray(x, dtype=np.float32).copy()
    valid = result[1, :, :, 0] > 0
    for channel in DIRECT:
        mapping = calibration["mappings"][str(channel)]
        slope = float(mapping["slope_ifs_from_era"])
        intercept = float(mapping["intercept_ifs_from_era"])
        for day in range(10):
            physical = slope * decode(result[channel, :, :, day], channel) + intercept
            if channel == 5:
                physical = np.maximum(physical, 0.0)
            result[channel, :, :, day] = np.where(
                valid,
                encode(physical, channel),
                0.0,
            )
    return recompute_common_derived(result, target_date)


def insert_real_ifs_domain(
    x: np.ndarray,
    target_date: int,
    direct_maps: np.ndarray,
) -> np.ndarray:
    """Insert ten real IFS days and apply the common derived-field protocol."""
    if direct_maps.shape[0] != len(DIRECT) or direct_maps.shape[-1] != 10:
        raise ValueError(f"Unexpected real-IFS map shape: {direct_maps.shape}")
    result = np.asarray(x, dtype=np.float32).copy()
    valid = result[1, :, :, 0] > 0
    for index, channel in enumerate(DIRECT):
        for day in range(10):
            physical = direct_maps[index, :, :, day]
            if channel == 5:
                physical = np.maximum(physical, 0.0)
            result[channel, :, :, day] = np.where(
                valid,
                encode(physical, channel),
                0.0,
            )
    return recompute_common_derived(result, target_date)
