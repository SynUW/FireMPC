#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced downsampling script (high-performance version, with create_sample_config built in)
- average/sum: integral image (summed area table), O(1) window aggregation; sliding window is also fast
- mode: non-overlapping path uses full-block reshape + bincount; sliding-window path uses "per-class batching + integral image" sliding histograms
- nearest/bilinear: when stride == factor, use GDAL resampling directly (rasterio.warp.reproject)
- Intelligent invalid values: valid pixel = GeoTIFF per-band NoData (rasterio.nodatavals) AND configured invalid_value both apply (either match -> invalid); below threshold -> output invalid
- QA bands are auto-skipped
- Multiprocess imap_unordered for streaming progress
- Preserves original dtype with safe clipping

Usage examples
--------------
# Non-overlapping (recommended, fastest)
python run_downsample_fast.py --downsample-factor 10 --stride 10 --max-workers 8 --skip-existing --output-dir /mnt/raid/user/downsample_out

# Sliding window with overlap (e.g. stride=5)
python run_downsample_fast.py --downsample-factor 10 --stride 5 --max-workers 8 --skip-existing --output-dir /mnt/raid/user/downsample_out

# For sum (FIRMS counts) while keeping output size consistent with other strides
python run_downsample_fast.py --downsample-factor 10 --stride 5 --sum-nonoverlap-same-size --output-dir /mnt/raid/user/downsample_out

# Force rewrite (overwrites existing outputs; takes precedence over --skip-existing)
python run_downsample_fast.py --downsample-factor 10 --stride 10 --max-workers 8 --rewrite --output-dir /mnt/raid/user/downsample_out
"""

import os
import sys
import logging
import shutil
import signal
import atexit
import tempfile
import time
import threading
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, Future
from pathlib import Path
from queue import Queue, Empty
from typing import List, Dict, Tuple, Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from affine import Affine

# Environment and logging
os.environ.setdefault('GDAL_CACHEMAX', '10240')
os.environ.setdefault('GDAL_NUM_THREADS', 'ALL_CPUS')

# ---- Local scratch directory (prefer /data, fall back to /dev/shm) ----
def _pick_scratch() -> Path:
    for p in [Path("/data"), Path("/dev/shm")]:
        try:
            p.mkdir(parents=True, exist_ok=True)
            probe = p / f".probe_{os.getpid()}"
            probe.touch(); probe.unlink()
            return p
        except (PermissionError, OSError):
            continue
    return Path(tempfile.gettempdir())

_SCRATCH_DIR = _pick_scratch() / f"downsample_{os.getpid()}"

def _cleanup():
    if _SCRATCH_DIR.exists():
        shutil.rmtree(_SCRATCH_DIR, ignore_errors=True)

atexit.register(_cleanup)
signal.signal(signal.SIGINT,  lambda s, f: (_cleanup(), sys.exit(1)))
signal.signal(signal.SIGTERM, lambda s, f: (_cleanup(), sys.exit(1)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Advanced_Downsampler_Fast")


# ============== create_sample_config (built in, kept as-is) ==============
def create_sample_config() -> List[Dict]:
    return [
        # {
        #     # MCD09GA b1/2/3/7 (apparent reflectance, continuous) -> average
        #     # QA bands (e.g. Surface_Reflectance_QA, State QA) must be masked before aggregation
        #     'path': '/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled/MCD09GA_b1237_QAapplied_spatialFilled',
        #     'downsample_method': 'average',
        #     'invalid_value': -32768,
        #     'qa_bands': []  # composited cloud-free data has no QA band
        # },
        
        {
            # MCD09GA b1/2/3/7 (apparent reflectance, continuous) -> average
            # QA bands (e.g. Surface_Reflectance_QA, State QA) must be masked before aggregation
            'path': '/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/worldpop_v2_clipped/access_idw_norm',
            'downsample_method': 'average',
            'invalid_value': -32768,
            'qa_bands': []  # Composited cloud-free data has no QA band
        },
        
        # {
        #     # MCD11A1 (LST, continuous) -> average; mask with QA first
        #     'path': '/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled/MCD11A1_QAapplied_spatialFilled',
        #     'downsample_method': 'average',
        #     'invalid_value': -32768,
        #     'qa_bands': []  # was [1], changed to 0-based
        # },
        
        # {
        #   # MCD09CMG
        #   'path': '/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled/MCD09CMG_QAapplied_spatialFilled',
        #   'downsample_method': 'average',
        #   'invalid_value': -32765,  # set incorrectly in earlier processing, so the invalid value is -32765 here, not -32768
        #   'qa_bands': []
        #   # Native resolution is 5600 m, so no downsampling is needed; use the preprocessing output directly
        # },

        # {
        #     # MCD11A2 (LST, continuous) -> average; mask with QA first
        #     'path': '/mnt/raid/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip/MCD11A2_mosaic_filled',
        #     'downsample_method': 'average',
        #     'invalid_value': 0,
        #     'qa_bands': [4, 5, 6]  # likewise, 0-based (original config: [5, 6])
        # },

        # {
        #     # ERA5 (continuous variables: temperature, wind components, humidity, pressure, etc.) -> area average
        #     # * If accumulators (e.g. total_precipitation) are included, split them into a separate folder and use sum for that part (see the next optional block)
        #     'path': '/mnt/raid/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip/ERA5_consistent_mosaic_withnodata',
        #     'downsample_method': 'average',
        #     'invalid_value': -32768,  # to keep zero values out of interpolation, mask with the first band and set invalid to -32768
        #     'qa_bands': []  # ERA5 has no QA band
        # },
        # {
        #     # MCD12Q1 (annual land cover, discrete classes) -> mode
        #     'path': '/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/MCD12Q1_mosaic',
        #     'downsample_method': 'mode',
        #     'invalid_value': 0,
        #     'qa_bands': []  # was [1], changed to 0-based
        # },

        # {
        #     # MCD15A3H (LAI/FPAR, continuous) -> average; mask with QA first
        #     'path': '/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled/MCD15A3H_QAapplied_spatialFilled',
        #     'downsample_method': 'average',
        #     'invalid_value': -32768, # no invalid value, placeholder
        #     'qa_bands': []  # composited data has no QA band
        #     # native resolution: 500 m
        # },
        
        # {
        #     # MCD14A1 binary fire / detection raster: use sum (count); no QA band
        #     'path': '/mnt/raid/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip/MCD14A1_mosaic_binary',
        #     'downsample_method': 'sum',
        #     'invalid_value': 255,  # NoData value for the binarized data
        #     'qa_bands': []
        # }
        
        # {
        #     # MCD14A1 non-binary fire / detection raster: use mode (count); no QA band
        #     'path': '/mnt/raid/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic/MCD14A1_withoutMask_Binary',
        #     'downsample_method': 'mode',
        #     'invalid_value': 0,  # NoData value for the binarized data
        #     'qa_bands': []
        # }
    ]


# =================== Core math utilities (integral image, etc.) ===================
def _integral_image(arr: np.ndarray) -> np.ndarray:
    sat = arr.cumsum(axis=0).cumsum(axis=1)
    pad = np.zeros((arr.shape[0] + 1, arr.shape[1] + 1), dtype=sat.dtype)
    pad[1:, 1:] = sat
    return pad

def _window_sum(sat: np.ndarray, y0: np.ndarray, x0: np.ndarray, h: int, w: int) -> np.ndarray:
    y1 = y0 + h
    x1 = x0 + w
    return sat[y1, x1] - sat[y0, x1] - sat[y1, x0] + sat[y0, x0]

def _clip_cast_float_to_int(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    out = np.rint(arr)
    if dtype == np.uint8:
        out = np.clip(out, 0, 255)
    elif dtype == np.uint16:
        out = np.clip(out, 0, 65535)
    elif dtype == np.uint32:
        out = np.clip(out, 0, 4294967295)
    return out.astype(dtype, copy=False)


def _band_valid_mask(
    band: np.ndarray,
    dtype: np.dtype,
    file_nodata: Optional[float],
    config_invalid,
) -> np.ndarray:
    """
    Valid pixel = numerically legal (finite for floats), AND not equal to the GeoTIFF band's NoData (if any),
    AND not equal to the configured invalid_value (if config_invalid is float('nan'), treat NaN as invalid).
    Both checks are combined with UNION semantics: invalid if either matches.
    """
    band = np.asarray(band)
    if np.issubdtype(dtype, np.floating):
        valid = np.isfinite(band)
    else:
        valid = np.ones(band.shape, dtype=bool)

    if file_nodata is not None:
        fn = file_nodata
        if isinstance(fn, (float, np.floating)) and np.isnan(float(fn)):
            if np.issubdtype(dtype, np.floating):
                valid &= ~np.isnan(band)
        else:
            valid &= band != fn

    if config_invalid is not None:
        ci = config_invalid
        if isinstance(ci, float) and np.isnan(ci):
            if np.issubdtype(dtype, np.floating):
                valid &= ~np.isnan(band)
        else:
            valid &= band != ci

    return valid


# =================== average / sum: fast path ===================
def _downsample_avg_sum_integral(
    band: np.ndarray,
                         method: str, 
                         invalid_value: float,
    dtype: np.dtype,
    factor: int,
    stride: int,
    invalid_threshold: float,
    mirror_pad: int,
    sum_nonoverlap_same_size: bool,
    valid: Optional[np.ndarray] = None,
) -> np.ndarray:
    # Valid mask + zero out invalids (valid is generated by the caller from TIFF NoData + config invalid_value)
    if valid is None:
        valid = _band_valid_mask(band, dtype, None, invalid_value)
    else:
        valid = np.asarray(valid, dtype=bool)
    data = np.where(valid, band, 0.0).astype(np.float64, copy=False)

    # One-shot mirror padding
    if mirror_pad > 0:
        data = np.pad(data, ((mirror_pad, mirror_pad), (mirror_pad, mirror_pad)), mode='reflect')
        valid = np.pad(valid.astype(np.uint8), ((mirror_pad, mirror_pad), (mirror_pad, mirror_pad)), mode='reflect')
    else:
        valid = valid.astype(np.uint8)

    H, W = band.shape
    Hp, Wp = data.shape
    out_h = (H + stride - 1) // stride
    out_w = (W + stride - 1) // stride

    S = _integral_image(data.astype(np.float64, copy=False))
    # Cumulative count inside the integral image is at most Hp*Wp; use int32 within range to save memory and bandwidth
    cnt_dtype = np.int32 if (Hp * Wp) < (2**31 - 16) else np.int64
    C = _integral_image(valid.astype(cnt_dtype, copy=False))

    # Initialize with NaN to avoid clashes when a computed result happens to equal invalid_value
    out = np.full((out_h, out_w), np.nan, dtype=np.float64)
    area = float(factor * factor)
    # Process in row-chunks of the output to avoid the full-image meshgrid(y, x) using out_h*out_w-level memory and causing freezes / swapping
    target_chunk_elems = max(1, (256 * 1024 * 1024) // 64)
    chunk_h = max(1, min(out_h, target_chunk_elems // max(out_w, 1)))

    for i0 in range(0, out_h, chunk_h):
        i1 = min(i0 + chunk_h, out_h)
        yy = np.arange(i0, i1, dtype=np.int64)[:, None]
        xx = np.arange(out_w, dtype=np.int64)[None, :]
        y_nom = yy * stride
        x_nom = xx * stride

        if method == 'sum' and sum_nonoverlap_same_size:
            y0 = (y_nom // factor) * factor
            x0 = (x_nom // factor) * factor
        else:
            y0 = y_nom
            x0 = x_nom

        y0 = np.clip(y0, 0, Hp - factor)
        x0 = np.clip(x0, 0, Wp - factor)

        win_sum = _window_sum(S, y0, x0, factor, factor)
        win_cnt = _window_sum(C, y0, x0, factor, factor)

        valid_ratio = win_cnt.astype(np.float64) / area
        m = valid_ratio >= (1.0 - invalid_threshold)
        sl = slice(i0, i1)
        sub = out[sl]
        if method == 'average':
            sub[m] = win_sum[m] / np.maximum(win_cnt[m], 1)
        else:
            sub[m] = win_sum[m]

        if not np.isnan(invalid_value):
            computed_invalid = m & (sub == invalid_value)
            if np.any(computed_invalid):
                sub[computed_invalid] = invalid_value + 1.0

    # Set still-NaN (untouched) pixels to invalid_value
    out[np.isnan(out)] = invalid_value

    if np.issubdtype(dtype, np.integer):
        return _clip_cast_float_to_int(out, dtype)
    else:
        return out.astype(dtype, copy=False)


# =================== mode: non-overlapping / sliding-window paths ===================
def _downsample_mode_nonoverlap_fast(
    band: np.ndarray,
    invalid_value: int,
    dtype: np.dtype,
    factor: int,
    invalid_threshold: float
) -> np.ndarray:
    H, W = band.shape
    out_h = (H + factor - 1) // factor
    out_w = (W + factor - 1) // factor

    pad_h = out_h * factor - H
    pad_w = out_w * factor - W
    if pad_h or pad_w:
        band_pad = np.pad(band, ((0, pad_h), (0, pad_w)), mode='reflect')
    else:
        band_pad = band

    view = band_pad.reshape(out_h, factor, out_w, factor).transpose(0, 2, 1, 3).reshape(out_h, out_w, factor * factor)
    valid_mask = (view != invalid_value)
    valid_cnt = valid_mask.sum(axis=2).astype(np.int32)
    need_valid = (valid_cnt.astype(np.float32) / (factor * factor)) >= (1.0 - invalid_threshold)

    out = np.full((out_h, out_w), invalid_value, dtype=band.dtype)
    bins = 256 if band.dtype == np.uint8 else None

    for i in range(out_h):
        row = view[i]
        row_valid = valid_mask[i]
        for j in range(out_w):
            if not need_valid[i, j]:
                continue
            vals = row[j][row_valid[j]]
            if vals.size == 0:
                continue
            if bins is None:
                m = int(vals.max()) + 1
                hist = np.bincount(vals, minlength=m)
            else:
                hist = np.bincount(vals, minlength=bins)
            out[i, j] = hist.argmax()

    return out.astype(dtype, copy=False)


def _downsample_mode_sliding_histogram(
    band: np.ndarray,
    invalid_value: int,
    dtype: np.dtype,
    factor: int,
    stride: int,
    invalid_threshold: float,
    mirror_pad: int,
    max_class_value: int = 255,
    class_batch: int = 32
) -> np.ndarray:
    assert stride < factor, "Sliding-window mode is only used when stride < factor; non-overlapping should take the fast path."

    H, W = band.shape
    if mirror_pad > 0:
        band_p = np.pad(band, ((mirror_pad, mirror_pad), (mirror_pad, mirror_pad)), mode='reflect')
    else:
        band_p = band
    Hp, Wp = band_p.shape

    out_h = (H + stride - 1) // stride
    out_w = (W + stride - 1) // stride
    yy, xx = np.meshgrid(np.arange(out_h, dtype=np.int64),
                         np.arange(out_w, dtype=np.int64),
                         indexing='ij')
    y0 = np.clip(yy * stride, 0, Hp - factor)
    x0 = np.clip(xx * stride, 0, Wp - factor)

    valid = (band_p != invalid_value).astype(np.uint8)
    C = _integral_image(valid)
    win_cnt = _window_sum(C, y0, x0, factor, factor)
    valid_ratio = win_cnt.astype(np.float64) / float(factor * factor)
    need = valid_ratio >= (1.0 - invalid_threshold)

    best_counts = np.full((out_h, out_w), -1, dtype=np.int32)
    best_labels = np.full((out_h, out_w), invalid_value, dtype=np.int32)

    for start in range(0, max_class_value + 1, class_batch):
        end = min(start + class_batch - 1, max_class_value)
        num = end - start + 1
        cls_vals = np.arange(start, end + 1, dtype=band_p.dtype).reshape(num, 1, 1)
        masks = (band_p[None, :, :] == cls_vals) & (band_p[None, :, :] != invalid_value)
        masks = masks.astype(np.uint8)

        for k in range(num):
            Sk = _integral_image(masks[k])
            cnt = _window_sum(Sk, y0, x0, factor, factor).astype(np.int32)
            better = (cnt > best_counts) & need
            best_counts[better] = cnt[better]
            best_labels[better] = start + k

        del masks

    out = np.where(need, best_labels, invalid_value).astype(dtype, copy=False)
    return out


# =================== GDAL resampling (stride == factor) ===================
def _downsample_by_reproject(band: np.ndarray, method: str, dtype: np.dtype, factor: int) -> np.ndarray:
    H, W = band.shape
    out_h = (H + factor - 1) // factor
    out_w = (W + factor - 1) // factor
    dst = np.zeros((out_h, out_w), dtype=band.dtype)
    res_map = {
        'nearest': Resampling.nearest,
        'bilinear': Resampling.bilinear,
        'average': Resampling.average,
    }
    r = res_map[method]
    src_transform = Affine.identity()
    dst_transform = Affine.scale(factor, factor)
    reproject(
        source=band,
        destination=dst,
        src_transform=src_transform,
        src_crs=None,
        dst_transform=dst_transform,
        dst_crs=None,
        resampling=r,
        num_threads=os.environ.get('GDAL_NUM_THREADS', 'ALL_CPUS')
    )
    return dst.astype(dtype, copy=False)


# =================== Main class ===================
class AdvancedDownsampler:
    def __init__(
        self,
        downsample_factor: int = 10,
        invalid_threshold: float = 0.5,
        stride: Optional[int] = None,
        max_workers: Optional[int] = None,
        skip_existing: bool = True,
        sum_nonoverlap_same_size: bool = True,
        mode_max_class_value: int = 255,
        mode_class_batch: int = 32
    ):
        self.factor = int(downsample_factor)
        self.stride = int(stride) if stride is not None else max(1, self.factor // 2)
        self.invalid_threshold = float(invalid_threshold)
        self.max_workers = int(max_workers) if (max_workers and max_workers > 0) else min(8, os.cpu_count() or 4)
        self.skip_existing = bool(skip_existing)
        self.sum_nonoverlap_same_size = bool(sum_nonoverlap_same_size)
        self.mode_max_class_value = int(mode_max_class_value)
        self.mode_class_batch = int(mode_class_batch)

        logger.info(
            f"init: factor={self.factor}, stride={self.stride}, invalid_th={self.invalid_threshold}, "
            f"workers={self.max_workers}, skip_existing={self.skip_existing}, "
            f"sum_nonoverlap_same_size={self.sum_nonoverlap_same_size}, "
            f"mode_max_class={self.mode_max_class_value}, mode_class_batch={self.mode_class_batch}"
        )

    @staticmethod
    def _band_names(src: rasterio.io.DatasetReader) -> List[str]:
        return [src.descriptions[i] if src.descriptions[i] else f"Band_{i+1}" for i in range(src.count)]

    def _process_one_band(
        self,
        band: np.ndarray,
        method: str,
        invalid_value: float,
        dtype: np.dtype,
        file_nodata: Optional[float] = None,
    ) -> np.ndarray:
        method = method.lower()
        f, s, pad = self.factor, self.stride, self.factor // 2
        valid = _band_valid_mask(band, dtype, file_nodata, invalid_value)

        if method in ('average', 'sum'):
            return _downsample_avg_sum_integral(
                band, method, invalid_value, dtype,
                factor=f, stride=s,
                invalid_threshold=self.invalid_threshold,
                mirror_pad=pad,
                sum_nonoverlap_same_size=self.sum_nonoverlap_same_size,
                valid=valid,
            )

        if method in ('nearest', 'bilinear'):
            if s == f:
                return _downsample_by_reproject(band, method, dtype, factor=f)
            # Simplified sliding-window nearest / bilinear (not recommended for large-scale use)
            h, w = band.shape
            out_h = (h + s - 1) // s
            out_w = (w + s - 1) // s
            pad_w = f // 2
            band_p = np.pad(band, ((pad_w, pad_w), (pad_w, pad_w)), mode='reflect')
            out = np.empty((out_h, out_w), dtype=np.float64)
            ys, xs = np.indices((f, f))
            cy, cx = (f - 1) / 2.0, (f - 1) / 2.0
            dist = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
            wgt = 1.0 / (1.0 + dist)
            for i in range(out_h):
                y0 = min(i * s, h - 1)
                for j in range(out_w):
                    x0 = min(j * s, w - 1)
                    win = band_p[y0:y0 + f, x0:x0 + f]
                    if method == 'nearest':
                        out[i, j] = float(win[f // 2, f // 2])
                    else:
                        out[i, j] = float(np.average(win.astype(np.float64), weights=wgt))
            if np.issubdtype(dtype, np.integer):
                return _clip_cast_float_to_int(out, dtype)
            return out.astype(dtype, copy=False)

        if method == 'mode':
            iv = invalid_value
            if isinstance(iv, float) and np.isnan(iv):
                iv_int = 0
            else:
                iv_int = int(iv)
            band_m = np.asarray(band, dtype=dtype, order="C").copy()
            band_m[~valid] = iv_int
            if s == f:
                return _downsample_mode_nonoverlap_fast(
                    band=band_m, invalid_value=iv_int, dtype=dtype,
                    factor=f, invalid_threshold=self.invalid_threshold
                )
            return _downsample_mode_sliding_histogram(
                band=band_m, invalid_value=iv_int, dtype=dtype,
                factor=f, stride=s, invalid_threshold=self.invalid_threshold,
                mirror_pad=pad, max_class_value=self.mode_max_class_value,
                class_batch=self.mode_class_batch
            )

        raise ValueError(f"Unsupported downsampling method: {method}")

    def _process_local(self, local_in: str, local_out: str, config: Dict) -> bool:
        """Read from local scratch, process, then write to local scratch (pure CPU, no NFS IO)."""
        try:
            with rasterio.open(local_in) as src:
                names = self._band_names(src)
                qa_bands = set(config.get('qa_bands', []))
                proc_bands = [i for i in range(src.count) if i not in qa_bands]
                if not proc_bands:
                    return False

                data = src.read()
                nv = getattr(src, "nodatavals", None)
                nodata_per_band = (
                    [nv[i] if i < len(nv) else None for i in range(src.count)]
                    if nv is not None
                    else [getattr(src, "nodata", None)] * src.count
                )
                profile = src.profile.copy()
                h, w = profile['height'], profile['width']
                out_h = (h + self.stride - 1) // self.stride
                out_w = (w + self.stride - 1) // self.stride
                out_prof = profile.copy()
                out_prof.update({
                    'count': len(proc_bands),
                    'width': out_w, 'height': out_h,
                    'transform': src.transform * Affine.scale(self.stride, self.stride),
                })

                method = config['downsample_method'].lower()
                invalid_value = config['invalid_value']
                out_bands, out_desc = [], []
                for b in proc_bands:
                    band = data[b]
                    dtype = np.dtype(src.dtypes[b])
                    file_nd = nodata_per_band[b] if b < len(nodata_per_band) else None
                    out_band = self._process_one_band(band, method, invalid_value, dtype, file_nodata=file_nd)
                    out_bands.append(out_band.astype(dtype, copy=False))
                    out_desc.append(names[b])

                out_arr = np.stack(out_bands, axis=0)
                Path(local_out).parent.mkdir(parents=True, exist_ok=True)
                with rasterio.open(local_out, 'w', **out_prof) as dst:
                    dst.write(out_arr)
                    dst.descriptions = out_desc
            return True
        except Exception as e:
            logger.error(f"[Processing failed] {local_in}: {e}", exc_info=False)
            return False

    def process_directory(self, input_dir: str, output_dir: str, config: Dict) -> Tuple[int, int]:
        """
        Three-stage pipeline:
          prefetch thread pool  -> NFS read into scratch
          process process pool  -> in-memory compute on scratch
          writeback thread pool -> scratch write back to NFS (with semaphore backpressure to prevent scratch from overflowing)
        """
        inp = Path(input_dir)
        outp = Path(output_dir)
        if not inp.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            return 0, 0
        outp.mkdir(parents=True, exist_ok=True)

        tif_files = sorted(list(inp.glob("*.tif")) + list(inp.glob("*.TIF")))
        if not tif_files:
            logger.warning(f"No TIF found: {input_dir}")
            return 0, 0

        # Filter out existing
        if self.skip_existing:
            tif_files = [f for f in tif_files if not (outp / f.name).exists()]
            if not tif_files:
                logger.info("All files exist; skipping.")
                return 0, 0

        total = len(tif_files)
        logger.info(f"Starting to process: {input_dir} | {total} files | method={config['downsample_method']}")

        # scratch subdirectory (isolated per dataset)
        tag = Path(input_dir).name
        s_in  = _SCRATCH_DIR / tag / "input"
        s_out = _SCRATCH_DIR / tag / "output"
        s_in.mkdir(parents=True, exist_ok=True)
        s_out.mkdir(parents=True, exist_ok=True)

        # Parameters
        prefetch_threads  = max(16, self.max_workers * 2)
        writeback_threads = max(8,  self.max_workers)
        prefetch_ahead    = self.max_workers * 4   # Prefetch lookahead
        # Output is smaller than input (downsampling), so scratch output usage is bounded;
        # but still use a semaphore for extreme cases
        wb_slots = threading.Semaphore(self.max_workers * 3)

        prefetch_q = Queue(maxsize=prefetch_ahead)
        stats = {'ok': 0, 'fail': 0}
        stats_lock = threading.Lock()
        t_start = time.time()

        def _print_progress():
            with stats_lock:
                done = stats['ok'] + stats['fail']
            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            eta  = (total - done) / rate if rate > 0 else 0
            print(
                f"\r[{done}/{total}] OK:{stats['ok']} FAIL:{stats['fail']} "
                f"{rate:.1f}f/s ETA:{eta:.0f}s   ",
                end="", flush=True,
            )

        # --- Prefetch: NFS -> scratch ---
        def _copy_to_scratch(nfs_src: Path) -> str:
            dst = s_in / nfs_src.name
            if not dst.exists():
                tmp = dst.with_suffix(".tmp")
                try:
                    with open(nfs_src, 'rb') as fi, open(tmp, 'wb') as fo:
                        while chunk := fi.read(4 * 1024 * 1024):
                            fo.write(chunk)
                    tmp.rename(dst)
                except Exception:
                    tmp.unlink(missing_ok=True)
                    raise
            return str(dst)

        # --- Writeback: scratch -> NFS ---
        def _copy_to_nfs(scratch_out: str, nfs_out: Path, sem: threading.Semaphore):
            tmp = nfs_out.with_suffix(".tmp_wb")
            try:
                with open(scratch_out, 'rb') as fi, open(tmp, 'wb') as fo:
                    while chunk := fi.read(4 * 1024 * 1024):
                        fo.write(chunk)
                tmp.rename(nfs_out)
            except Exception as e:
                tmp.unlink(missing_ok=True)
                logger.warning(f"Writeback failed {nfs_out.name}: {e}")
            finally:
                Path(scratch_out).unlink(missing_ok=True)
                sem.release()

        # ---- Stage 1: prefetch ----
        def _prefetch_stage():
            with ThreadPoolExecutor(max_workers=prefetch_threads) as pool:
                pending: dict[Future, Path] = {}
                task_iter = iter(tif_files)
                exhausted = False
                while True:
                    while not exhausted and len(pending) < prefetch_ahead:
                        try:
                            f = next(task_iter)
                            pending[pool.submit(_copy_to_scratch, f)] = f
                        except StopIteration:
                            exhausted = True
                            break
                    if not pending:
                        break
                    done_futs, _ = concurrent.futures.wait(
                        list(pending), timeout=0.2,
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    for fut in done_futs:
                        src_file = pending.pop(fut)
                        try:
                            local_path = fut.result()
                            prefetch_q.put(local_path)   # Bounded, self-throttling
                        except Exception as e:
                            with stats_lock:
                                stats['fail'] += 1
                            logger.error(f"Prefetch failed {src_file.name}: {e}")
                            _print_progress()
            prefetch_q.put(None)  # Sentinel

        # ---- Stages 2+3: process + writeback ----
        def _process_and_writeback_stage():
            wb_pool = ThreadPoolExecutor(max_workers=writeback_threads)
            proc_futs: dict[Future, str] = {}

            def _drain():
                for f in [ff for ff in list(proc_futs) if ff.done()]:
                    local_in = proc_futs.pop(f)
                    fname = Path(local_in).name
                    try:
                        ok = f.result()
                    except Exception as e:
                        ok = False
                        logger.error(f"Processing crashed {fname}: {e}")
                    Path(local_in).unlink(missing_ok=True)
                    local_out = str(s_out / fname)
                    nfs_out   = outp / fname
                    if ok and Path(local_out).exists():
                        wb_slots.acquire()
                        wb_fut = wb_pool.submit(_copy_to_nfs, local_out, nfs_out, wb_slots)
                        with stats_lock:
                            stats['ok'] += 1
                    else:
                        Path(local_out).unlink(missing_ok=True)
                        with stats_lock:
                            stats['fail'] += 1
                    _print_progress()

            with ProcessPoolExecutor(max_workers=self.max_workers) as proc_pool:
                while True:
                    _drain()
                    while len(proc_futs) < self.max_workers * 2:
                        try:
                            local_in = prefetch_q.get(timeout=0.05)
                        except Empty:
                            break
                        if local_in is None:
                            while proc_futs:
                                _drain()
                                time.sleep(0.02)
                            wb_pool.shutdown(wait=True)
                            return
                        local_out = str(s_out / Path(local_in).name)
                        fut = proc_pool.submit(self._process_local, local_in, local_out, config)
                        proc_futs[fut] = local_in
                    time.sleep(0.01)

            wb_pool.shutdown(wait=True)

        t1 = threading.Thread(target=_prefetch_stage,              daemon=True)
        t2 = threading.Thread(target=_process_and_writeback_stage, daemon=True)
        t1.start(); t2.start()
        t1.join();  t2.join()

        print()
        logger.info(f"Done {tag}: OK={stats['ok']} FAIL={stats['fail']} / {total}")
        return stats['ok'], total


# =================== Single-dataset processing (can run in a child process) ===================
def _run_dataset_task(task_args: tuple) -> tuple:
    """
    Process a single dataset configuration. Can be called from the main process or a child process.
    task_args = (cfg, args_ns)
    Returns (dataset_name, ok, n)
    """
    cfg, args_ns = task_args

    method = cfg['downsample_method'].lower()
    if args_ns.stride is not None:
        stride = int(args_ns.stride)
    else:
        stride = args_ns.downsample_factor if method == 'sum' else max(1, args_ns.downsample_factor // 2)

    inp = cfg['path']
    if args_ns.output_dir:
        base_out = Path(args_ns.output_dir)
        base_out.mkdir(parents=True, exist_ok=True)
        out_dir = base_out / f"{Path(inp).name}_downsampled"
    else:
        out_dir = Path(inp).parent / f"{Path(inp).name}_downsampled"

    logger.info(f"\n=== Processing dataset ===")
    logger.info(f"in : {inp}")
    logger.info(f"out: {out_dir}")
    logger.info(f"method={method}, factor={args_ns.downsample_factor}, stride={stride}, invalid_th={args_ns.invalid_threshold}")

    ds = AdvancedDownsampler(
        downsample_factor=args_ns.downsample_factor,
        invalid_threshold=args_ns.invalid_threshold,
        stride=stride,
        max_workers=args_ns.max_workers,
        skip_existing=(args_ns.skip_existing and (not args_ns.rewrite)),
        sum_nonoverlap_same_size=args_ns.sum_nonoverlap_same_size,
        mode_max_class_value=args_ns.mode_max_class,
        mode_class_batch=args_ns.mode_class_batch,
    )

    ok, n = ds.process_directory(inp, str(out_dir), cfg)
    logger.info(f"[Done] {Path(inp).name}: {ok}/{n}")
    return (Path(inp).name, ok, n)


# =================== CLI ===================
def main():
    import argparse
    import multiprocessing
    parser = argparse.ArgumentParser(description="Advanced downsampling (high-performance version, built-in configuration)")
    parser.add_argument('--downsample-factor', type=int, default=10, # For non-mcd14a1: 5; for mcd14a1: 2
                        help='Window size / downsampling factor, default 10. Effective downsampling factor is downsample-factor / 2')
    parser.add_argument('--stride', type=int, default=None,
                        help='Output grid stride; default: sum=factor, otherwise factor//2')
    parser.add_argument('--invalid-threshold', type=float, default=0.5,
                        help='Valid-ratio threshold within the window; below this threshold the output is invalid; default 0.5')
    parser.add_argument('--max-workers', type=int, default=16,
                        help='Number of concurrent processes, default 8')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip if the output already exists')
    parser.add_argument(
        '--rewrite', '--overwrite',
        action='store_true',
        help='If the output exists, overwrite it (takes precedence over --skip-existing); --overwrite is an alias',
    )
    parser.add_argument('--sum-nonoverlap-same-size', action='store_true',
                        help='sum uses non-overlapping bucket counts, but the output size scales with stride (consistent with other methods)')
    parser.add_argument('--mode-max-class', type=int, default=255,
                        help='Class upper bound for sliding-window mode (typically 255 for uint8)')
    parser.add_argument('--mode-class-batch', type=int, default=32,
                        help='Sliding-window mode "per-class batch" size')
    parser.add_argument('--output-dir', default='/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled_5xdownsampled',
                        help='Top-level output directory (a *_downsampled subdirectory is created per dataset)')
    args = parser.parse_args()

    cfg_list = create_sample_config()
    n_datasets = len(cfg_list)

    task_args_list = [(cfg, args) for cfg in cfg_list]

    if n_datasets == 1:
        _run_dataset_task(task_args_list[0])
    else:
        logger.info(f"\nMulti-dataset parallel mode: starting {n_datasets} processes (one process per dataset)")
        ctx = multiprocessing.get_context("spawn")
        procs = []
        for task_args in task_args_list:
            p = ctx.Process(target=_run_dataset_task, args=(task_args,))
            p.start()
            procs.append((task_args[0]['path'], p))

        failed_datasets = []
        for inp_path, p in procs:
            p.join()
            name = Path(inp_path).name
            if p.exitcode == 0:
                logger.info(f"Dataset [{name}] processing complete (exitcode=0)")
            else:
                logger.error(f"Dataset [{name}] abnormal exit (exitcode={p.exitcode}）")
                failed_datasets.append(name)

        if failed_datasets:
            logger.warning(f"\nThe following datasets did not complete successfully: {', '.join(failed_datasets)}")
        else:
            logger.info(f"\nAll datasets processed")


if __name__ == "__main__":
    main()
