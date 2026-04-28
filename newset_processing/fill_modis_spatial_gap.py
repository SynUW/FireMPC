#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODIS time-series data filling script -- GDAL version
(high-performance + fast-path optimizations; falls back to the most recent processed image historically, and merges multiple future days for the first day)

This version (v5.4 compat fix)
- Resolves "-tap option cannot be used without using -tr.":
  WarpOptions now explicitly provides pixel size xRes/yRes (derived from the target geotransform),
  keeps outputBounds, drops width/height, lets GDAL derive the dimensions itself; keeps targetAlignedPixels=True.
- Retains v5.2/5.3's shape alignment and "common-window slicing" guard, plus all parameters and comments you requested.
- Adds support for YYYY_MM_DD_tileXX.tif: tiles are processed separately.
- Input filenames may have any prefix (e.g. MCD09GA_b1237_2002_07_04_tile00.tif); the date is taken from the trailing YYYY_MM_DD[_tileXX].tif.
- Writeback: the target GeoTIFF uses float32 bands; for UInt16/Byte etc. sources, a float32 buffer must be written (older versions used an integer buffer + Float32 buf_type, which corrupted integer products like MCD11A1).
- For inputs with malformed TIFF ExtraSamples tags, libtiff uses fprintf to **fd 2** (bypassing Python sys.stderr); before importing numpy/gdal, fd 2 is piped and filtered (set env var FILL_MODIS_NO_STDERR_FILTER=1 to disable).
- Permanently-invalid detection: block-wise spatial reads + only reading the bands under inspection + multi-window parallelism; optional stratified sampling by year (replacing whole-image ReadAsArray x processes).
- First-day "future merge": streamingly open subsequent images day by day; do not cache all future scenes at once (the old logic was prone to OOM / process kill with no Python traceback).
- Multi-tile parallelism: when multiple tile groups are detected, start as many child processes as there are tiles; each child runs permanent-invalid detection + filling + writeback independently.
- Skip optimization: when --overwrite is not given, first check whether each tile is already fully processed; completed tiles are skipped without entering the detection / filling flow.
"""

import os
import sys
import threading


def _install_fd2_extrasamples_filter() -> None:
    """
    libtiff fprintf(stderr)s for malformed ExtraSamples, i.e. writes to the OS-level fd 2 directly,
    bypassing Python's sys.stderr object -- so wrapping sys.stderr is ineffective. Here we dup the real terminal fd,
    redirect fd 2 to a pipe, and have a background thread filter line-by-line before writing back to the terminal.
    """
    try:
        saved = os.dup(2)
    except OSError:
        return
    try:
        r_fd, w_fd = os.pipe()
    except OSError:
        try:
            os.close(saved)
        except OSError:
            pass
        return
    try:
        os.dup2(w_fd, 2)
    except OSError:
        try:
            os.close(r_fd)
        except OSError:
            pass
        try:
            os.close(w_fd)
        except OSError:
            pass
        try:
            os.close(saved)
        except OSError:
            pass
        return
    try:
        os.close(w_fd)
    except OSError:
        pass

    def _reader() -> None:
        buf = b""
        try:
            while True:
                try:
                    chunk = os.read(r_fd, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
                while True:
                    nl = buf.find(b"\n")
                    if nl < 0:
                        break
                    line = buf[: nl + 1]
                    buf = buf[nl + 1 :]
                    if b"ExtraSamples" not in line and b"EXTRASAMPLES" not in line.upper():
                        try:
                            os.write(saved, line)
                        except OSError:
                            pass
        finally:
            if buf:
                if b"ExtraSamples" not in buf and b"EXTRASAMPLES" not in buf.upper():
                    try:
                        os.write(saved, buf)
                    except OSError:
                        pass
            try:
                os.close(r_fd)
            except OSError:
                pass

    threading.Thread(
        target=_reader, daemon=True, name="fd2-extrasamples-filter"
    ).start()


if os.environ.get("FILL_MODIS_NO_STDERR_FILTER", "").lower() not in (
    "1",
    "true",
    "yes",
):
    _install_fd2_extrasamples_filter()

import argparse
from bisect import bisect_right
from concurrent.futures import Future, ThreadPoolExecutor
import gc
import re
import time
import signal
import atexit
import random
import numpy as np
from osgeo import gdal, ogr
from pathlib import Path
from datetime import timedelta, date as Date
from typing import Dict, List, Optional, Tuple

# ===================== Basic setup =====================

gdal.PushErrorHandler("CPLQuietErrorHandler")

# ---------------------------------------------------------------------------
# Smart RAM detection + GDAL tuning
# ---------------------------------------------------------------------------

def _get_available_ram_bytes() -> int:
    """Return available system RAM in bytes.  Tries psutil first, then /proc/meminfo."""
    try:
        import psutil
        return psutil.virtual_memory().available
    except Exception:
        pass
    try:
        with open('/proc/meminfo') as _f:
            for _line in _f:
                if _line.startswith('MemAvailable:'):
                    return int(_line.split()[1]) * 1024   # kB → bytes
    except Exception:
        pass
    return 4 * 1024 ** 3   # conservative 4 GB fallback


def _configure_gdal() -> dict:
    """
    Read available RAM and tune GDAL accordingly.
    Returns a dict with the computed values so callers can print/log them.

    Strategy (leaves 25 % for the OS + other users):
      usable      = available_ram × 0.75
      gdal_cache  = min(usable × 0.30,  16 GB)   – tile block cache
      warp_mem    = min(usable × 0.20,   8 GB)   – per-warp working buffer
      swath_size  = min(usable × 0.05, 512 MB)   – stripe read buffer
    """
    avail = _get_available_ram_bytes()
    usable = int(avail * 0.75)

    gdal_cache_bytes = min(int(usable * 0.30), 16 * 1024 ** 3)
    warp_mem_bytes   = min(int(usable * 0.20),  8 * 1024 ** 3)
    swath_bytes      = min(int(usable * 0.05), 512 * 1024 ** 2)

    # SetCacheMax64 exists only in GDAL 3.6+; older bindings have only SetCacheMax (some versions are limited to 32-bit)
    if hasattr(gdal, "SetCacheMax64"):
        gdal.SetCacheMax64(gdal_cache_bytes)
    else:
        gdal.SetCacheMax(int(min(gdal_cache_bytes, 2**31 - 1)))
    gdal.SetConfigOption("GDAL_NUM_THREADS",  "ALL_CPUS")
    gdal.SetConfigOption("GDAL_SWATH_SIZE",   str(swath_bytes))
    gdal.SetConfigOption("VSI_CACHE",         "TRUE")
    gdal.SetConfigOption("VSI_CACHE_SIZE",    str(min(128 * 1024 ** 2, swath_bytes)))

    return dict(
        available_gb  = avail          / 1024 ** 3,
        usable_gb     = usable         / 1024 ** 3,
        gdal_cache_gb = gdal_cache_bytes / 1024 ** 3,
        warp_mem_bytes = warp_mem_bytes,
        warp_mem_gb   = warp_mem_bytes / 1024 ** 3,
    )


_RAM_CFG = _configure_gdal()   # run once at import time

# Supports:
#   - YYYY_MM_DD.tif / YYYY_MM_DD_tileXX.tif (the entire filename is the date)
#   - any-prefix_<YYYY>_<MM>_<DD>[_tileXX].tif, e.g. MCD09GA_b1237_2002_07_04_tile00.tif
# Take the date from the trailing segment of the filename (avoiding the requirement that the filename starts with the date)
DATE_TILE_RE = re.compile(
    r"(\d{4})_(\d{2})_(\d{2})(?:_(tile\d+))?\.tif$",
    re.IGNORECASE,
)

_global_executor = None
def _cleanup_executor():
    global _global_executor
    if _global_executor is not None:
        try:
            _global_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        _global_executor = None

def _signal_handler(signum, frame):
    _cleanup_executor()
    sys.exit(1)

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)
atexit.register(_cleanup_executor)

def _fmt_time(s: float) -> str:
    s = int(s)
    return f"{s//3600}:{(s%3600)//60:02d}:{s%60:02d}"

def _parse_date_tile(p: Path) -> Optional[Tuple[Date, Optional[str]]]:
    m = DATE_TILE_RE.search(p.name)
    if not m:
        return None
    y, M, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    tile = m.group(4)
    if tile is not None:
        tile = tile.lower()
    return Date(y, M, d), tile


def _output_filename_for_date(d: Date, tile_suffix: Optional[str]) -> str:
    if tile_suffix:
        return f"{d:%Y_%m_%d}_{tile_suffix}.tif"
    return f"{d:%Y_%m_%d}.tif"


def _build_index_by_tile(folder: Path) -> Dict[Optional[str], Dict[Date, Path]]:
    """Group by tile: None means the legacy naming without a _tile suffix; otherwise a lowercase suffix like 'tile12'."""
    groups: Dict[Optional[str], Dict[Date, Path]] = {}
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        parsed = _parse_date_tile(p)
        if not parsed:
            continue
        d, tile = parsed
        groups.setdefault(tile, {})
        if d in groups[tile]:
            print(
                f"Duplicate file for the same date + tile; the latter overrides the index: {groups[tile][d].name} → {p.name}",
                file=sys.stderr,
            )
        groups[tile][d] = p
    return groups


def _build_output_index_for_tile(folder: Path, tile_suffix: Optional[str]) -> Dict[Date, Path]:
    if not folder.exists():
        return {}
    idx: Dict[Date, Path] = {}
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        parsed = _parse_date_tile(p)
        if not parsed:
            continue
        d, tkey = parsed
        if tkey != tile_suffix:
            continue
        idx[d] = p
    return idx

# ===================== AOI / mask / validity check =====================

_aoi_mask_cache: dict = {}   # (shp_str, width, height) → ndarray

def load_aoi_mask(shp_path: Optional[Path], width: int, height: int,
                   geotransform: Tuple, projection: str) -> np.ndarray:
    """
    Load the AOI mask from the shapefile. Result is cached -- rasterizing the shapefile
    is expensive and the grid never changes within a single run.
    Returns:
        np.ndarray: boolean array; True means inside the AOI.
    """
    if shp_path is None or not Path(shp_path).exists():
        return np.ones((height, width), dtype=bool)
    cache_key = (str(shp_path), width, height)
    if cache_key in _aoi_mask_cache:
        return _aoi_mask_cache[cache_key]
    mem_driver = gdal.GetDriverByName('MEM')
    target_ds = mem_driver.Create('', width, height, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(geotransform)
    target_ds.SetProjection(projection)
    shp_ds = ogr.Open(str(shp_path))
    if shp_ds is None:
        raise RuntimeError(f"Cannot open shapefile: {shp_path}")
    layer = shp_ds.GetLayer()
    gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[1])
    mask_arr = _read_band_full_as_array(target_ds.GetRasterBand(1))
    if mask_arr is None:
        raise RuntimeError("Failed to read the AOI mask raster")
    mask = mask_arr.astype(bool)
    target_ds = None
    shp_ds = None
    _aoi_mask_cache[cache_key] = mask
    return mask

def is_nodata(arr: np.ndarray,
              builtin_nodata,
              custom_nodata_list=None,
              valid_min=None,
              valid_max=None) -> np.ndarray:
    """Returns True for invalid"""
    mask = np.zeros(arr.shape, dtype=bool)
    if builtin_nodata is not None:
        mask |= (arr == builtin_nodata)
    if custom_nodata_list:
        for nodata_val in custom_nodata_list:
            mask |= (arr == nodata_val)
    if arr.dtype.kind == 'f':
        mask |= np.isnan(arr)
    if valid_min is not None:
        mask |= (arr < valid_min)
    if valid_max is not None:
        mask |= (arr > valid_max)
    return mask


def _gdal_band_to_numpy_dtype(data_type: int):
    if data_type == gdal.GDT_Byte:
        return np.uint8
    if data_type == gdal.GDT_UInt16:
        return np.uint16
    if data_type == gdal.GDT_Int16:
        return np.int16
    if data_type == gdal.GDT_UInt32:
        return np.uint32
    if data_type == gdal.GDT_Int32:
        return np.int32
    if data_type == gdal.GDT_Float32:
        return np.float32
    if data_type == gdal.GDT_Float64:
        return np.float64
    if data_type == gdal.GDT_CFloat32:
        return np.complex64
    if data_type == gdal.GDT_CFloat64:
        return np.complex128
    return np.float32


def _read_band_window_as_array(band, xoff: int, yoff: int, xsize: int, ysize: int) -> Optional[np.ndarray]:
    """
    Use ReadRaster + np.frombuffer to avoid importing osgeo.gdal_array.
    This is important on environments where GDAL Python bindings are ABI-misaligned with NumPy.
    """
    raw = band.ReadRaster(
        xoff,
        yoff,
        xsize,
        ysize,
        buf_xsize=xsize,
        buf_ysize=ysize,
        buf_type=band.DataType,
    )
    if raw is None:
        return None
    dtype = _gdal_band_to_numpy_dtype(band.DataType)
    arr = np.frombuffer(raw, dtype=dtype)
    if arr.size != xsize * ysize:
        return None
    return arr.reshape((ysize, xsize))


def _read_band_full_as_array(band) -> Optional[np.ndarray]:
    return _read_band_window_as_array(band, 0, 0, band.XSize, band.YSize)


def _read_selected_bands_full_stack(
    ds,
    bands_to_process: List[int],
) -> Optional[Tuple[List[np.ndarray], List[Optional[float]]]]:
    arrays: List[np.ndarray] = []
    nodatas: List[Optional[float]] = []
    for bidx in bands_to_process:
        band = ds.GetRasterBand(bidx + 1)
        arr = _read_band_full_as_array(band)
        if arr is None:
            return None
        arrays.append(np.asarray(arr, dtype=np.float32))
        nodatas.append(band.GetNoDataValue())
    return arrays, nodatas


def _write_band_array(band, arr: np.ndarray, xoff: int = 0, yoff: int = 0) -> None:
    arr_np = np.ascontiguousarray(arr)
    if arr_np.ndim != 2:
        raise ValueError("Only 2D arrays are supported for raster band writes")
    band.WriteRaster(
        xoff,
        yoff,
        arr_np.shape[1],
        arr_np.shape[0],
        arr_np.tobytes(),
        buf_xsize=arr_np.shape[1],
        buf_ysize=arr_np.shape[0],
        buf_type=band.DataType,
    )

def spatial_fill(data: np.ndarray, need_fill: np.ndarray,
                 max_distance: int = 100) -> np.ndarray:
    """GDAL FillNodata fallback"""
    if not np.any(need_fill):
        return data
    h, w = data.shape
    mem_driver = gdal.GetDriverByName('MEM')
    temp = mem_driver.Create('', w, h, 1, gdal.GDT_Float32)
    work = data.astype(np.float32, copy=True)
    work[need_fill] = np.nan
    band = temp.GetRasterBand(1)
    band.SetNoDataValue(np.nan)
    _write_band_array(band, work)
    gdal.FillNodata(targetBand=band, maskBand=None,
                    maxSearchDist=max_distance, smoothingIterations=0)
    out = _read_band_full_as_array(band)
    if out is None:
        temp = None
        return data
    temp = None
    return np.asarray(out, dtype=np.float32)

# ===================== Resampling utilities (with alignment) =====================

def _force_align_to_target_shape(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Crop or pad an arbitrary 2D array to (target_h, target_w)."""
    h, w = arr.shape
    if (h, w) == (target_h, target_w):
        return arr.astype(np.float32, copy=False)
    out = np.full((target_h, target_w), np.nan, dtype=np.float32)
    hh = min(h, target_h)
    ww = min(w, target_w)
    out[:hh, :ww] = arr[:hh, :ww]
    return out


def _read_selected_bands_from_ds(
    ds,
    bands_to_process: List[int],
    target_height: int,
    target_width: int,
) -> List[np.ndarray]:
    """
    Read only the requested bands when possible.
    Reading the entire stack is only worth it when all bands are needed.
    """
    arrays: List[np.ndarray] = []
    for bidx in bands_to_process:
        arr = _read_band_full_as_array(ds.GetRasterBand(bidx + 1))
        if arr is None:
            raise RuntimeError(f"Failed to read band: {bidx + 1}")
        arrays.append(
            _force_align_to_target_shape(
                np.asarray(arr, dtype=np.float32),
                target_height,
                target_width,
            )
        )
    return arrays

def _read_aligned_bands_stack(data_path: Path,
                              bands_to_process: List[int],
                              target_width: int, target_height: int,
                              target_geotransform: Tuple, target_projection: str
                              ) -> Tuple[List[np.ndarray], Optional[float]]:
    """
    If the input differs from the target grid, Warp once to the target grid (compatible with older GDAL):
      - Supply outputBounds and xRes/yRes (derived from the target geotransform);
      - targetAlignedPixels=True;
      - Do not pass width/height; let GDAL derive the dimensions (but the dst_ds we pass already has the target size).
    Then read only the requested bands;
    return per-band arrays (float32, force-aligned to the target shape) and the source band1's NoData.
    """
    src_ds = gdal.Open(str(data_path), gdal.GA_ReadOnly)
    if src_ds is None:
        raise RuntimeError(f"Cannot open file: {data_path}")
    src_w, src_h = src_ds.RasterXSize, src_ds.RasterYSize
    src_gt = src_ds.GetGeoTransform()
    src_prj = src_ds.GetProjection()

    # Allow small floating-point error
    def _gt_equal(a, b, eps=1e-9):
        return all(abs(a[i]-b[i]) <= eps for i in range(6))

    need_resample = (
        (src_w != target_width) or
        (src_h != target_height) or
        (src_prj != target_projection) or
        (not _gt_equal(src_gt, target_geotransform))
    )

    builtin_nodata = src_ds.GetRasterBand(1).GetNoDataValue()

    if not need_resample:
        arrays = _read_selected_bands_from_ds(
            src_ds,
            bands_to_process,
            target_height,
            target_width,
        )
        src_ds = None
        return arrays, builtin_nodata

    # Resampling needed: warp to the target grid (all bands), then read the requested bands
    mem_driver = gdal.GetDriverByName('MEM')
    dst_ds = mem_driver.Create('', target_width, target_height,
                               src_ds.RasterCount, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(target_geotransform)
    dst_ds.SetProjection(target_projection)

    xRes = target_geotransform[1]
    yRes = abs(target_geotransform[5])

    opts = gdal.WarpOptions(
        format='MEM',
        dstSRS=target_projection,
        resampleAlg=gdal.GRA_NearestNeighbour,
        outputBounds=(
            target_geotransform[0],
            target_geotransform[3] + target_height * target_geotransform[5],
            target_geotransform[0] + target_width * target_geotransform[1],
            target_geotransform[3]
        ),
        xRes=xRes, yRes=yRes,
        multithread=True,
        targetAlignedPixels=True,
        warpMemoryLimit=_RAM_CFG['warp_mem_bytes'],   # sized to available RAM
    )
    gdal.Warp(dst_ds, src_ds, options=opts)

    arrays: List[np.ndarray] = []
    for b in bands_to_process:
        a = _read_band_full_as_array(dst_ds.GetRasterBand(b + 1))
        if a is None:
            dst_ds = None
            src_ds = None
            raise RuntimeError(f"Failed to read band after Warp: {b + 1}")
        a = np.asarray(a, dtype=np.float32)
        arrays.append(_force_align_to_target_shape(a, target_height, target_width))

    dst_ds = None
    src_ds = None
    return arrays, builtin_nodata

# ===================== Permanently-invalid pixels (optional) =====================

def _iter_spatial_windows(height: int, width: int, block: int) -> List[Tuple[int, int, int, int]]:
    """Generate (row_off, col_off, win_h, win_w) tuples covering the entire image."""
    out: List[Tuple[int, int, int, int]] = []
    for r in range(0, height, block):
        for c in range(0, width, block):
            wh = min(block, height - r)
            ww = min(block, width - c)
            out.append((r, c, wh, ww))
    return out


def _chunk_evenly(items: List, n_chunks: int) -> List[List]:
    """Split items into n_chunks as evenly as possible (the last few chunks may be slightly smaller)."""
    if not items:
        return []
    n_chunks = max(1, min(n_chunks, len(items)))
    base, extra = divmod(len(items), n_chunks)
    chunks: List[List] = []
    i = 0
    for k in range(n_chunks):
        sz = base + (1 if k < extra else 0)
        chunks.append(items[i : i + sz])
        i += sz
    return [c for c in chunks if c]


def _sample_dates_for_permanent_check(
    all_dates: List[Date], n_sample: int, seed: int, stratified: bool
) -> List[Date]:
    """
    Pick a number of days from the time series to participate in "permanently-invalid" detection.
    When stratified=True, round-robin one day per year, then cycle, to avoid samples clumping into a few years (the read volume is unchanged).
    """
    if n_sample >= len(all_dates):
        return list(all_dates)
    rng = random.Random(seed)
    if not stratified:
        return sorted(rng.sample(all_dates, n_sample))

    by_y: Dict[int, List[Date]] = {}
    for d in all_dates:
        by_y.setdefault(d.year, []).append(d)
    for y in by_y:
        rng.shuffle(by_y[y])
    years = sorted(by_y.keys())
    picked: List[Date] = []
    picked_set = set()
    while len(picked) < n_sample:
        progressed = False
        for y in years:
            if len(picked) >= n_sample:
                break
            if by_y[y]:
                d = by_y[y].pop()
                picked.append(d)
                picked_set.add(d)
                progressed = True
        if not progressed:
            break
    if len(picked) < n_sample:
        rest = [d for d in all_dates if d not in picked_set]
        rng.shuffle(rest)
        for d in rest:
            if len(picked) >= n_sample:
                break
            picked.append(d)
    return sorted(picked[:n_sample])


def _read_raster_size(path: str) -> Optional[Tuple[int, int]]:
    """Return (RasterXSize, RasterYSize); None if opening fails."""
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        return None
    w, h = ds.RasterXSize, ds.RasterYSize
    ds = None
    return (w, h)


def _majority_wh(counts: Dict[Tuple[int, int], int]) -> Tuple[int, int]:
    """The most frequent (width, height); break ties by larger area (w * h)."""
    if not counts:
        raise ValueError("empty counts")
    best_cnt = max(counts.values())
    candidates = [wh for wh, c in counts.items() if c == best_cnt]
    return max(candidates, key=lambda wh: wh[0] * wh[1])


def _filter_refill_permanent_sample_by_majority_size(
    index: Dict[Date, Path],
    all_dates: List[Date],
    initial_sampled: List[Date],
    n_target: int,
    sample_seed: int,
) -> Optional[Tuple[List[Date], int, int]]:
    """
    Compute the modal raster size from the initial stratified / random sample; drop dates that differ from the mode,
    then top up randomly from all_dates (with a reproducible seed), adding only dates matching the modal size.
    Returns None if none of the initial samples can be opened.
    """
    dated: List[Tuple[Date, int, int]] = []
    for d in initial_sampled:
        p = str(index[d])
        sz = _read_raster_size(p)
        if sz is None:
            continue
        w, h = sz
        dated.append((d, w, h))

    if not dated:
        return None

    counts: Dict[Tuple[int, int], int] = {}
    for _, w, h in dated:
        counts[(w, h)] = counts.get((w, h), 0) + 1

    mode_w, mode_h = _majority_wh(counts)
    mode_key = (mode_w, mode_h)

    kept: List[Date] = [d for d, w, h in dated if (w, h) == mode_key]
    kept_set = set(kept)
    n_fail_open = len(initial_sampled) - len(dated)
    n_wrong_size = len(dated) - len(kept)

    if n_wrong_size or n_fail_open:
        print(
            f"   Permanent-invalid sampling: modal raster size {mode_w}×{mode_h} px（{counts[mode_key]}/{len(dated)} votes)"
            + (f"; size-mismatched excluded {n_wrong_size} days" if n_wrong_size else "")
            + (f"; initial open failed {n_fail_open} days" if n_fail_open else ""),
            flush=True,
        )

    if len(kept) >= n_target:
        sampled = sorted(kept)[:n_target]
    else:
        need = n_target - len(kept)
        pool = [d for d in all_dates if d not in kept_set]
        rng = random.Random(sample_seed + 1337)
        rng.shuffle(pool)
        for d in pool:
            if need <= 0:
                break
            p = str(index[d])
            sz = _read_raster_size(p)
            if sz is None or sz != mode_key:
                continue
            kept.append(d)
            kept_set.add(d)
            need -= 1
        sampled = sorted(kept)
        if len(sampled) < n_target:
            print(
                f"   Days available with the modal size are only {len(sampled)} days (target {n_target}); exhausted",
                flush=True,
            )

    return (sampled, mode_w, mode_h)


def _permanent_invalid_worker_batch(args: Tuple) -> List[Tuple[int, int, np.ndarray]]:
    """
    Child process: for a batch of spatial windows, AND "per-band invalid" day by day across all sampled dates.
    Only ReadAsArray the current window + the bands under inspection, to avoid loading the whole image x all bands into memory.
    Returns [(row_off, col_off, acc), ...], where acc.shape == (len(bands_to_check), wh, ww).
    Uses a per-band permanent-invalid mask, so that one always-invalid band does not also mask out other still-fillable bands.
    """
    (
        paths,
        windows,
        bands_to_check,
        custom_nodata_values,
        valid_ranges,
    ) = args
    results: List[Tuple[int, int, np.ndarray]] = []
    n_b = len(bands_to_check)
    valid_ranges_for_checked: List[Tuple[Optional[float], Optional[float]]] = []
    for bidx in bands_to_check:
        vmin = vmax = None
        if valid_ranges and bidx < len(valid_ranges) and valid_ranges[bidx] is not None:
            vmin, vmax = valid_ranges[bidx]
        valid_ranges_for_checked.append((vmin, vmax))

    for (r, c, wh, ww) in windows:
        results.append((r, c, np.ones((n_b, wh, ww), dtype=bool)))

    for fp in paths:
        if not any(np.any(acc) for _, _, acc in results):
            break
        try:
            ds = gdal.Open(fp, gdal.GA_ReadOnly)
            if ds is None:
                continue
            w_ds = ds.RasterXSize
            h_ds = ds.RasterYSize
            for j, bidx in enumerate(bands_to_check):
                active_results = [(r, c, acc) for (r, c, acc) in results if acc[j].any()]
                if not active_results:
                    continue
                band = ds.GetRasterBand(bidx + 1)
                nd = band.GetNoDataValue()
                vmin, vmax = valid_ranges_for_checked[j]
                for r, c, acc in active_results:
                    wh, ww = acc.shape[1], acc.shape[2]
                    xsize = min(ww, max(0, w_ds - c))
                    ysize = min(wh, max(0, h_ds - r))
                    if xsize <= 0 or ysize <= 0:
                        acc[j].fill(False)
                        continue
                    arr = _read_band_window_as_array(band, c, r, xsize, ysize)
                    if arr is None:
                        acc[j].fill(False)
                        continue
                    arr = np.asarray(arr, dtype=np.float32)
                    m = is_nodata(arr, nd, custom_nodata_values, vmin, vmax)
                    acc[j, :ysize, :xsize] &= m
            ds = None
        except Exception:
            continue
    return results


def detect_permanently_invalid_pixels(
    index: Dict[Date, Path],
    sample_size: int = 1000,
    custom_nodata_values: Optional[List[float]] = None,
    valid_ranges: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
    top_k_bands: Optional[int] = None,
    band_indices: Optional[List[int]] = None,
    workers: int = 8,
    block_size: int = 2048,
    stratified_sample: bool = True,
    sample_seed: int = 42,
) -> Optional[np.ndarray]:
    """
    Permanently-invalid pixel: across the sampled dates, if a band's pixel is always invalid (NoData / out-of-range / etc.), it is treated as permanently invalid.
    """
    if not index:
        return None
    all_dates = sorted(index.keys())
    n_sample = min(sample_size, len(all_dates))
    initial_sampled = _sample_dates_for_permanent_check(
        all_dates, n_sample, sample_seed, stratified_sample
    )

    fr = _filter_refill_permanent_sample_by_majority_size(
        index, all_dates, initial_sampled, n_sample, sample_seed
    )
    if fr is None:
        return None
    sampled_dates, width, height = fr
    if not sampled_dates:
        return None

    first_path = str(index[sampled_dates[0]])
    first_ds = gdal.Open(first_path, gdal.GA_ReadOnly)
    if first_ds is None:
        return None
    total_bands = first_ds.RasterCount
    first_ds = None

    if band_indices is not None:
        bands_to_check = [b for b in band_indices if 0 <= b < total_bands]
    elif top_k_bands is not None:
        bands_to_check = list(range(min(top_k_bands, total_bands)))
    else:
        bands_to_check = list(range(total_bands))
    if not bands_to_check:
        return None

    paths = [str(index[d]) for d in sampled_dates]
    valid_ranges_for_checked: List[Tuple[Optional[float], Optional[float]]] = []
    for bidx in bands_to_check:
        vmin = vmax = None
        if valid_ranges and bidx < len(valid_ranges) and valid_ranges[bidx] is not None:
            vmin, vmax = valid_ranges[bidx]
        valid_ranges_for_checked.append((vmin, vmax))

    bytes_per_full_pass = height * width * max(1, len(bands_to_check)) * 4
    full_raster_budget = int(min(_RAM_CFG["warp_mem_bytes"], 2 * 1024 ** 3))
    use_full_raster = bytes_per_full_pass <= full_raster_budget

    print(
        f"   Permanent-invalid detection: sampled {len(sampled_dates)}/{len(all_dates)} days | "
        f"Stratified sampling={'on' if stratified_sample else 'off'} | "
        f"Bands under inspection {bands_to_check} | "
        f"Mode {'whole-image vectorized' if use_full_raster else 'tiled threads'}",
        flush=True,
    )

    permanently_invalid = np.zeros((len(bands_to_check), height, width), dtype=bool)

    if use_full_raster:
        permanently_invalid.fill(True)
        for idx_i, fp in enumerate(paths, start=1):
            if not np.any(permanently_invalid):
                break
            try:
                ds = gdal.Open(fp, gdal.GA_ReadOnly)
                if ds is None:
                    continue
                stack_info = _read_selected_bands_full_stack(ds, bands_to_check)
                ds = None
                if stack_info is None:
                    continue
                stack, nodatas = stack_info
                for j, arr in enumerate(stack):
                    if not permanently_invalid[j].any():
                        continue
                    vmin, vmax = valid_ranges_for_checked[j]
                    invalid = is_nodata(arr, nodatas[j], custom_nodata_values, vmin, vmax)
                    permanently_invalid[j] &= invalid
                if idx_i == 1 or idx_i % 32 == 0 or idx_i == len(paths):
                    remaining = int(sum(mask.any() for mask in permanently_invalid))
                    print(
                        f"   Permanent-invalid detection progress: {idx_i}/{len(paths)} days | bands still pending {remaining}",
                        flush=True,
                    )
            except Exception:
                continue
        return permanently_invalid

    windows = _iter_spatial_windows(height, width, block_size)
    if not windows:
        return None

    n_tasks = max(1, min(len(windows), max(workers * 12, workers)))
    window_batches = _chunk_evenly(windows, n_tasks)
    print(
        f"   Permanent-invalid detection fallback: spatial windows {len(windows)} (block {block_size}px) | "
        f"Parallel tasks {len(window_batches)} | thread count {workers}",
        flush=True,
    )

    worker_args = [
        (paths, batch, bands_to_check, custom_nodata_values, valid_ranges)
        for batch in window_batches
    ]

    from concurrent.futures import ThreadPoolExecutor, as_completed

    try:
        gc.collect()
        print(f"   Permanent-invalid detection executor: threads x {workers}", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_permanent_invalid_worker_batch, wa) for wa in worker_args]
            for fut in as_completed(futures):
                try:
                    batch_res = fut.result()
                except Exception:
                    continue
                for r, c, acc in batch_res:
                    wh, ww = acc.shape[1], acc.shape[2]
                    permanently_invalid[:, r : r + wh, c : c + ww] = acc
    except Exception:
        return None

    return permanently_invalid

# ===================== Core fill logic =====================

def _find_latest_processed_before(
    date: Date,
    output_folder: Path,
    max_past_days: int,
    out_idx: Optional[Dict[Date, Path]] = None,
    tile_suffix: Optional[str] = None,
) -> Optional[Date]:
    for offset in range(1, max_past_days + 1):
        cand = date - timedelta(days=offset)
        if out_idx is not None:
            if cand in out_idx:
                return cand
        else:
            p = output_folder / _output_filename_for_date(cand, tile_suffix)
            if p.exists():
                return cand
    return None

def _iter_future_inputs(
    date: Date,
    input_index: Dict[Date, Path],
    max_future_days: int,
    max_sources: Optional[int] = None,
    sorted_dates: Optional[List[Date]] = None,
) -> List[Date]:
    """
    Return the list of dates after `date` (within max_future_days calendar days) that exist in the index.
    When max_sources is non-empty, keep at most that many subsequent dates -- preventing the first-day merge from pulling in thousands of scenes for daily data (the original implementation OOMs).
    """
    max_date = date + timedelta(days=max_future_days)
    ordered_dates = sorted_dates if sorted_dates is not None else sorted(input_index.keys())
    start = bisect_right(ordered_dates, date)
    futs: List[Date] = []
    for cand in ordered_dates[start:]:
        if cand > max_date:
            break
        futs.append(cand)
        if max_sources is not None and len(futs) >= max_sources:
            break
    return futs


def _band_vmin_vmax(
    valid_ranges: Optional[List[Tuple[Optional[float], Optional[float]]]],
    bi: int,
) -> Tuple[Optional[float], Optional[float]]:
    if valid_ranges and bi < len(valid_ranges) and valid_ranges[bi] is not None:
        return valid_ranges[bi]
    return None, None


def _merge_source_plane_into_band(
    data_bi: np.ndarray,
    need_fill: np.ndarray,
    src_arr: np.ndarray,
    s_nd,
    custom_nodata_values,
    vmin,
    vmax,
) -> bool:
    """Fill the still-pending pixels using one band plane of a single source image; return whether any pixels still remain to fill."""
    if not np.any(need_fill):
        return False
    th, tw = data_bi.shape
    src_arr = np.asarray(src_arr, dtype=np.float32)
    sh, sw = src_arr.shape
    hh = min(th, sh)
    ww = min(tw, sw)
    if hh <= 0 or ww <= 0:
        return bool(np.any(need_fill))
    tgt_need = need_fill[:hh, :ww]
    src_sub = src_arr[:hh, :ww]
    src_valid = ~is_nodata(src_sub, s_nd, custom_nodata_values, vmin, vmax)
    fill_pos = tgt_need & src_valid
    if np.any(fill_pos):
        tgt = data_bi[:hh, :ww]
        tgt[fill_pos] = src_sub[fill_pos]
        data_bi[:hh, :ww] = tgt
        updated_need = need_fill[:hh, :ww]
        updated_need[fill_pos] = False
        need_fill[:hh, :ww] = updated_need
    return bool(np.any(need_fill))


def _build_need_fill_masks(
    data: np.ndarray,
    fillable_mask: np.ndarray,
    builtin_nodata,
    custom_nodata_values,
    valid_ranges: Optional[List[Tuple[Optional[float], Optional[float]]]],
    per_band_permanently_invalid_masks: Optional[List[np.ndarray]] = None,
) -> List[np.ndarray]:
    masks: List[np.ndarray] = []
    for bi in range(data.shape[0]):
        vmin, vmax = _band_vmin_vmax(valid_ranges, bi)
        nodata_mask = is_nodata(data[bi], builtin_nodata, custom_nodata_values, vmin, vmax)
        band_fillable_mask = fillable_mask
        if per_band_permanently_invalid_masks is not None:
            band_fillable_mask = fillable_mask & ~per_band_permanently_invalid_masks[bi]
        masks.append(nodata_mask & band_fillable_mask)
    return masks


def _prepare_per_band_permanent_invalid_masks(
    permanently_invalid_mask: Optional[np.ndarray],
    n_bands: int,
    height: int,
    width: int,
) -> Optional[List[np.ndarray]]:
    if permanently_invalid_mask is None:
        return None
    masks: List[np.ndarray] = []
    if permanently_invalid_mask.ndim == 2:
        shared = np.zeros((height, width), dtype=bool)
        ph, pw = permanently_invalid_mask.shape
        hh = min(height, ph)
        ww = min(width, pw)
        shared[:hh, :ww] = permanently_invalid_mask[:hh, :ww]
        return [shared] * n_bands

    if permanently_invalid_mask.ndim != 3:
        return None

    for bi in range(n_bands):
        mask = np.zeros((height, width), dtype=bool)
        if bi < permanently_invalid_mask.shape[0]:
            pm = permanently_invalid_mask[bi]
            ph, pw = pm.shape
            hh = min(height, ph)
            ww = min(width, pw)
            mask[:hh, :ww] = pm[:hh, :ww]
        masks.append(mask)
    return masks


def _has_remaining_need_fill(need_fill_masks: List[np.ndarray]) -> bool:
    return any(np.any(mask) for mask in need_fill_masks)


def _count_remaining_need_fill(need_fill_masks: List[np.ndarray]) -> int:
    return int(sum(int(np.count_nonzero(mask)) for mask in need_fill_masks))


def _adaptive_future_sources_cap(
    width: int,
    height: int,
    n_bands: int,
    requested: Optional[int],
) -> Optional[int]:
    """
    Limit first-day future merge IO for large products.
    The old static default (512 scenes) is too expensive for large tiles.
    """
    if requested is not None and requested <= 0:
        return requested
    per_scene_bytes = max(1, width * height * max(1, n_bands) * 4)
    io_budget_bytes = 16 * 1024 ** 3
    adaptive = max(365, min(512, io_budget_bytes // per_scene_bytes))
    if requested is None:
        return adaptive
    return max(1, min(requested, adaptive))

def _remove_if_exists(path: Path):
    """Ensure GDAL Create can overwrite: delete first if it exists"""
    try:
        if path.exists():
            os.remove(path)
    except Exception:
        pass


def load_input_snapshot(
    path: Path,
    keep_bands: Optional[List[int]],
) -> Optional[dict]:
    """
    Open a single scene and read all bands-to-process into memory (shared by the prefetch thread and the main thread).
    Returns a dict: width, height, geotransform, projection, bands_to_process, n_bands,
    builtin_nodata, dtypes, data_stack (float32, n_bands x H x W).
    """
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        return None
    try:
        width = ds.RasterXSize
        height = ds.RasterYSize
        total_bands = ds.RasterCount
        geotransform = ds.GetGeoTransform()
        projection = ds.GetProjection()
        bands_to_process = (
            list(range(total_bands))
            if keep_bands is None
            else [b for b in keep_bands if 0 <= b < total_bands]
        )
        n_bands = len(bands_to_process)
        if n_bands == 0:
            return None
        builtin_nodata = ds.GetRasterBand(bands_to_process[0] + 1).GetNoDataValue()
        dtypes = [ds.GetRasterBand(bidx + 1).DataType for bidx in bands_to_process]
        data_arrays = _read_selected_bands_from_ds(ds, bands_to_process, height, width)
        data_stack = np.stack(data_arrays, axis=0)
        return {
            "width": width,
            "height": height,
            "geotransform": geotransform,
            "projection": projection,
            "bands_to_process": bands_to_process,
            "n_bands": n_bands,
            "builtin_nodata": builtin_nodata,
            "dtypes": dtypes,
            "data_stack": data_stack,
        }
    finally:
        ds = None


class _InputPrefetcher:
    """
    Asynchronously prefetch the full input scenes for at most max_ahead following dates, overlapping reads with the current day's compute.
    Processing order is still day-by-day; only the read latency is hidden. By default ~10 images are in flight / cached at once.
    """

    def __init__(
        self,
        max_ahead: int,
        keep_bands: Optional[List[int]],
        input_index: Dict[Date, Path],
    ):
        self.max_ahead = max(0, int(max_ahead))
        self.keep_bands = keep_bands
        self.input_index = input_index
        self._executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=self.max_ahead) if self.max_ahead > 0 else None
        )
        self._futures: Dict[Date, Future] = {}
        self._lock = threading.Lock()

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        with self._lock:
            self._futures.clear()

    def ensure_submitted(self, d: Date) -> None:
        if self._executor is None or d not in self.input_index:
            return
        with self._lock:
            if d in self._futures:
                return
            path = self.input_index[d]
            self._futures[d] = self._executor.submit(
                load_input_snapshot, path, self.keep_bands
            )

    def take(self, d: Date) -> Optional[dict]:
        """Get the input snapshot for date d; if not prefetched, load synchronously."""
        if d not in self.input_index:
            return None
        path = self.input_index[d]
        if self._executor is None:
            return load_input_snapshot(path, self.keep_bands)
        fut: Optional[Future] = None
        with self._lock:
            fut = self._futures.pop(d, None)
        if fut is not None:
            try:
                snap = fut.result()
                if snap is not None:
                    return snap
            except Exception:
                pass
        return load_input_snapshot(path, self.keep_bands)

    def discard_if_pending(self, d: Date) -> None:
        """Scenarios like skipping an existing output: if d is in the prefetch queue, pull and discard it to avoid filling the window."""
        if self._executor is None:
            return
        fut: Optional[Future] = None
        with self._lock:
            fut = self._futures.pop(d, None)
        if fut is None:
            return
        try:
            _ = fut.result()
        except Exception:
            pass


def fill_single_date(
    target_date: Date,
    input_index: Dict[Date, Path],
    output_path: Path,
    output_folder: Path,
    max_past_days: int,
    max_future_days: int,
    mode: str,  # 'overwrite_first' | 'resume' | 'resume_first_future'
    is_first_of_run: bool,
    aoi_shp: Optional[Path],
    enable_spatial_interp: bool,
    spatial_interp_distance: int,
    compress: str,
    blocksize: int,
    custom_nodata_values: Optional[List[float]] = None,
    valid_ranges: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
    keep_bands: Optional[List[int]] = None,
    permanently_invalid_mask: Optional[np.ndarray] = None,
    out_idx: Optional[Dict[Date, Path]] = None,
    tile_suffix: Optional[str] = None,
    max_future_sources: Optional[int] = None,
    input_snapshot: Optional[dict] = None,
    past_override: Optional[Tuple[Date, np.ndarray, Optional[float]]] = None,
) -> Tuple[bool, Optional[Tuple[np.ndarray, Optional[float]]]]:
    """
    Fill target_date:
      - mode='overwrite_first': force the first day of this run to merge multiple future days; after that, fall back only on historical processed outputs.
      - mode='resume': fall back only on historical processed outputs; do not use future data.
      - mode='resume_first_future': merge multiple future days only on the first day; afterwards fall back on history.

    input_snapshot: if provided, do not reopen today's input file (prefetched result).
    past_override: (fallback_date, processed output stack float32 (n_bands, H, W), band1 nodata);
      used only when consistent with _find_latest_processed_before, to avoid re-reading the previous day's output.

    Returns (success?, (output stack float32 (n_bands, H, W), band1 nodata) or None).
    On success, the returned stack shares storage with the internal working buffer; the caller may only read-cache it (e.g. prev_cache), not write to it.
    """
    if target_date not in input_index:
        return False, None

    if input_snapshot is not None and input_snapshot.get("data_stack") is not None:
        width = int(input_snapshot["width"])
        height = int(input_snapshot["height"])
        geotransform = input_snapshot["geotransform"]
        projection = input_snapshot["projection"]
        bands_to_process = input_snapshot["bands_to_process"]
        n_bands = int(input_snapshot["n_bands"])
        builtin_nodata = input_snapshot["builtin_nodata"]
        dtypes = input_snapshot["dtypes"]
        data = np.asarray(input_snapshot["data_stack"], dtype=np.float32)
    else:
        src_ds = gdal.Open(str(input_index[target_date]), gdal.GA_ReadOnly)
        if src_ds is None:
            return False, None
        width = src_ds.RasterXSize
        height = src_ds.RasterYSize
        total_bands = src_ds.RasterCount
        geotransform = src_ds.GetGeoTransform()
        projection = src_ds.GetProjection()

        bands_to_process = (
            list(range(total_bands))
            if keep_bands is None
            else [b for b in keep_bands if 0 <= b < total_bands]
        )
        n_bands = len(bands_to_process)
        if n_bands == 0:
            src_ds = None
            return False, None

        builtin_nodata = src_ds.GetRasterBand(bands_to_process[0] + 1).GetNoDataValue()
        dtypes = [src_ds.GetRasterBand(bidx + 1).DataType for bidx in bands_to_process]
        data_arrays = _read_selected_bands_from_ds(src_ds, bands_to_process, height, width)
        src_ds = None
        data = np.stack(data_arrays, axis=0)

    aoi_mask = load_aoi_mask(aoi_shp, width, height, geotransform, projection)
    fillable_mask = aoi_mask.copy()
    per_band_permanently_invalid_masks = _prepare_per_band_permanent_invalid_masks(
        permanently_invalid_mask, n_bands, height, width
    )
    need_fill_masks = _build_need_fill_masks(
        data,
        fillable_mask,
        builtin_nodata,
        custom_nodata_values,
        valid_ranges,
        per_band_permanently_invalid_masks,
    )

    use_future_for_first = (mode in ("overwrite_first", "resume_first_future")) and is_first_of_run
    if use_future_for_first:
        effective_future_sources = _adaptive_future_sources_cap(
            width, height, n_bands, max_future_sources
        )
        future_dates = _iter_future_inputs(
            target_date,
            input_index,
            max_future_days,
            effective_future_sources,
            sorted_dates=sorted(input_index.keys()),
        )
        if is_first_of_run:
            initial_need_fill = _count_remaining_need_fill(need_fill_masks)
            print(
                f"  First-day future merge: streaming read {len(future_dates)} subsequent dates (cap="
                f"{effective_future_sources}; initial pixels to fill {initial_need_fill} pixels)",
                flush=True,
            )
        no_gain_streak = 0
        no_gain_limit = 12
        for idx_fd, fd in enumerate(future_dates, start=1):
            if not _has_remaining_need_fill(need_fill_masks):
                break
            before_fill = _count_remaining_need_fill(need_fill_masks)
            arrays, s_nd = _read_aligned_bands_stack(
                input_index[fd], bands_to_process, width, height, geotransform, projection
            )
            for bi in range(n_bands):
                vmin, vmax = _band_vmin_vmax(valid_ranges, bi)
                _merge_source_plane_into_band(
                    data[bi],
                    need_fill_masks[bi],
                    arrays[bi],
                    s_nd,
                    custom_nodata_values,
                    vmin,
                    vmax,
                )
            del arrays
            after_fill = _count_remaining_need_fill(need_fill_masks)
            gained = before_fill - after_fill
            if gained <= 0:
                no_gain_streak += 1
            else:
                no_gain_streak = 0
            if idx_fd == 1 or idx_fd % 8 == 0 or idx_fd == len(future_dates):
                print(
                    f"    future merge: {idx_fd}/{len(future_dates)} | newly filled {gained} | "
                    f"Remaining {after_fill} | consecutive no-gain {no_gain_streak}",
                    flush=True,
                )
            if no_gain_streak >= no_gain_limit:
                print(
                    f"    future merge: consecutive {no_gain_streak} scenes with no gain; early stop",
                    flush=True,
                )
                break
    else:
        if _has_remaining_need_fill(need_fill_masks):
            back = _find_latest_processed_before(
                target_date, output_folder, max_past_days, out_idx, tile_suffix
            )
            if back is not None:
                if (
                    past_override is not None
                    and past_override[0] == back
                    and past_override[1].shape[0] == n_bands
                ):
                    past_stack = past_override[1]
                    s_nd = past_override[2]
                    for bi in range(n_bands):
                        vmin, vmax = _band_vmin_vmax(valid_ranges, bi)
                        _merge_source_plane_into_band(
                            data[bi],
                            need_fill_masks[bi],
                            past_stack[bi],
                            s_nd,
                            custom_nodata_values,
                            vmin,
                            vmax,
                        )
                else:
                    source_path_out = output_folder / _output_filename_for_date(back, tile_suffix)
                    arrays, s_nd = _read_aligned_bands_stack(
                        source_path_out, bands_to_process, width, height, geotransform, projection
                    )
                    for bi in range(n_bands):
                        vmin, vmax = _band_vmin_vmax(valid_ranges, bi)
                        _merge_source_plane_into_band(
                            data[bi],
                            need_fill_masks[bi],
                            arrays[bi],
                            s_nd,
                            custom_nodata_values,
                            vmin,
                            vmax,
                        )
                    del arrays

    # ===== Spatial-interpolation fallback (per band) =====
    for bi in range(n_bands):
        if not enable_spatial_interp:
            continue
        vmin, vmax = _band_vmin_vmax(valid_ranges, bi)
        nodata_mask = is_nodata(data[bi], builtin_nodata, custom_nodata_values, vmin, vmax)
        band_fillable_mask = fillable_mask
        if per_band_permanently_invalid_masks is not None:
            band_fillable_mask = fillable_mask & ~per_band_permanently_invalid_masks[bi]
        need_fill = nodata_mask & band_fillable_mask
        if np.any(need_fill):
            try:
                data[bi] = spatial_fill(data[bi], need_fill, spatial_interp_distance)
            except Exception:
                pass

    # ===== Force still-invalid pixels (including those outside valid_ranges) to NoData =====
    for bi in range(n_bands):
        if valid_ranges and bi < len(valid_ranges) and valid_ranges[bi] is not None:
            vmin, vmax = valid_ranges[bi]
        else:
            vmin, vmax = None, None
        nodata_mask_final = is_nodata(data[bi], builtin_nodata, custom_nodata_values, vmin, vmax)
        if builtin_nodata is not None:
            data[bi][nodata_mask_final] = float(builtin_nodata)
        else:
            data[bi][nodata_mask_final] = np.nan

    # ===== Write results (overwrite / create) =====
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if output_path.exists():
            os.remove(output_path)
    except Exception:
        pass

    driver = gdal.GetDriverByName('GTiff')
    dst = driver.Create(
        str(output_path), width, height, n_bands, gdal.GDT_Float32,
        options=[
            f'COMPRESS={compress}', 'TILED=YES',
            f'BLOCKXSIZE={blocksize}', f'BLOCKYSIZE={blocksize}',
            'BIGTIFF=IF_SAFER', 'NUM_THREADS=ALL_CPUS',
        ]
    )
    dst.SetGeoTransform(geotransform)
    dst.SetProjection(projection)
    for i in range(n_bands):
        band = dst.GetRasterBand(i + 1)
        if builtin_nodata is not None:
            band.SetNoDataValue(float(builtin_nodata))
        if dtypes[i] in [gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32]:
            out = np.rint(data[i])
            if dtypes[i] == gdal.GDT_Byte:
                out = np.clip(out, 0, 255)
            elif dtypes[i] == gdal.GDT_UInt16:
                out = np.clip(out, 0, 65535)
            elif dtypes[i] == gdal.GDT_Int16:
                out = np.clip(out, -32768, 32767)
            elif dtypes[i] == gdal.GDT_UInt32:
                out = np.clip(out, 0, 2**32 - 1)
            else:
                out = np.clip(out, -(2**31), 2**31 - 1)
            _write_band_array(band, out.astype(np.float32))
        else:
            _write_band_array(band, np.asarray(data[i], dtype=np.float32))
    dst.FlushCache()
    dst = None
    return True, (data, builtin_nodata)

# ===================== Single-tile processing (can run in a child process) =====================

def _run_tile_task(task_args: Tuple) -> None:
    """
    Full pipeline for processing a single tile. Can be called from the main process or a child process.
    task_args = (tile_suffix, input_idx, args, custom_nodata_values,
                 valid_ranges, keep_bands, enable_spatial, max_future_sources)
    """
    (
        tile_suffix,
        input_idx,
        args,
        custom_nodata_values,
        valid_ranges,
        keep_bands,
        enable_spatial,
        max_future_sources,
    ) = task_args

    dates_all = sorted(input_idx.keys())
    if not dates_all:
        return

    tile_label = "No _tile suffix" if tile_suffix is None else tile_suffix
    print(f"\n{'=' * 60}\nTile [{tile_label}]：{len(dates_all)} dates\n{'=' * 60}")

    out_idx = _build_output_index_for_tile(args.output_folder, tile_suffix)

    # -- When --overwrite is not given, first check whether this tile is already fully processed --
    if not args.overwrite:
        if out_idx:
            last_done = max(out_idx.keys())
            # Check whether the largest date in the output already covers every input date
            if last_done >= dates_all[-1]:
                print(f"Block [{tile_label}] all processed (last output {last_done} >= input last date {dates_all[-1]}); skipping.")
                return

    # -- Permanently-invalid pixels (detected separately for the current tile) --
    permanently_invalid_mask = None
    if args.detect_permanent_invalid:
        print(f"\nEnable permanent-invalid pixel detection (tile {tile_label}; ignoring AOI)")
        detect_band_indices = None if args.detect_top_k_bands is not None else keep_bands
        permanently_invalid_mask = detect_permanently_invalid_pixels(
            input_idx,
            sample_size=args.permanent_invalid_sample_size,
            custom_nodata_values=custom_nodata_values,
            valid_ranges=valid_ranges,
            top_k_bands=args.detect_top_k_bands,
            band_indices=detect_band_indices,
            workers=args.permanent_invalid_workers,
            block_size=args.permanent_invalid_block,
            stratified_sample=not args.no_permanent_invalid_stratify,
        )
        if permanently_invalid_mask is None:
            print("   Detection failed; this tile continues with the normal flow (no permanent-invalid masking)")
        else:
            print("   Permanent-invalid detection complete for this tile")

    # -- Determine start date and mode --
    if args.overwrite:
        start_idx = 0
        mode = "overwrite_first"
        print("Mode: --overwrite (recompute from scratch; first day forced to merge multiple future days)")
    else:
        if out_idx:
            last_done = max(out_idx.keys())
            start_date = None
            for d in dates_all:
                if d > last_done:
                    start_date = d
                    break
            if start_date is None:
                print(f"Block [{tile_label}] output already in sync with input; skipping.")
                return
            start_idx = dates_all.index(start_date)
            mode = "resume"
            print(
                f"Mode: continue processing (from {start_date}; previous completed {last_done}; do not use future data)"
            )
        else:
            start_idx = 0
            mode = "resume_first_future"
            print("Mode: output is empty; starting from the earliest date; first day uses multi-future-day merge")

    n_total = len(dates_all) - start_idx
    if n_total <= 0:
        print("This tile has no dates to process.")
        return

    print(f"This tile pending {n_total} days | dates {dates_all[start_idx]} → {dates_all[-1]}")
    print(f"Maximum days to look back: {args.max_past_days}")
    print(f"Maximum days to look forward: {args.max_future_days} (first-day future merge only)")
    print("AOI: no (--aoi_shp disabled; the entire image participates in filling)")
    print(f"Spatial-interpolation fallback: {'enabled' if enable_spatial else 'disabled'}")
    print(f"Output: {args.output_folder}")

    t0 = time.time()
    first_of_run_idx = start_idx
    prefetcher = _InputPrefetcher(args.prefetch_ahead, keep_bands, input_idx)
    prev_cache: Optional[Tuple[Date, np.ndarray, Optional[float]]] = None
    ma = prefetcher.max_ahead
    try:
        if ma > 0:
            for k in range(1, ma + 1):
                j = start_idx + k
                if j < len(dates_all):
                    prefetcher.ensure_submitted(dates_all[j])

        for i in range(start_idx, len(dates_all)):
            d = dates_all[i]
            outp = args.output_folder / _output_filename_for_date(d, tile_suffix)

            is_first_of_run = i == first_of_run_idx

            if mode == "resume" and outp.exists():
                prefetcher.discard_if_pending(d)
                el = time.time() - t0
                finished = i - start_idx + 1
                eta = el * (len(dates_all) - finished) / max(1, finished)
                print(
                    f"\r[{tile_label}] [{finished}/{n_total}] {d} | exists (skip) | "
                    f"Elapsed: {_fmt_time(el)} ETA: {_fmt_time(eta)}",
                    end="",
                )
                continue

            if ma > 0:
                j = i + ma + 1
                if j < len(dates_all):
                    prefetcher.ensure_submitted(dates_all[j])

            back = _find_latest_processed_before(
                d, args.output_folder, args.max_past_days, out_idx, tile_suffix
            )
            past_ov: Optional[Tuple[Date, np.ndarray, Optional[float]]] = None
            if prev_cache is not None and back is not None and prev_cache[0] == back:
                past_ov = prev_cache

            snap = prefetcher.take(d)

            try:
                ok, cached = fill_single_date(
                    d,
                    input_idx,
                    outp,
                    args.output_folder,
                    args.max_past_days,
                    args.max_future_days,
                    mode,
                    is_first_of_run,
                    None,  # aoi_shp is disabled
                    enable_spatial,
                    args.spatial_interp_distance,
                    args.compress,
                    args.blocksize,
                    custom_nodata_values,
                    valid_ranges,
                    keep_bands,
                    permanently_invalid_mask,
                    out_idx=out_idx,
                    tile_suffix=tile_suffix,
                    max_future_sources=max_future_sources,
                    input_snapshot=snap,
                    past_override=past_ov,
                )
            except Exception as e:
                ok, cached = False, None
                print(f"\n❌ [{tile_label}] {d}: {e}")

            if ok:
                out_idx[d] = outp
                if cached is not None:
                    prev_cache = (d, cached[0], cached[1])
            else:
                prev_cache = None

            el = time.time() - t0
            finished = i - start_idx + 1
            eta = el * (len(dates_all) - finished) / max(1, finished)
            stat = "OK" if ok else "FAIL"
            print(
                f"\r[{tile_label}] [{finished}/{n_total}] {d} | {stat} | "
                f"Elapsed: {_fmt_time(el)} ETA: {_fmt_time(eta)}",
                end="",
            )

    finally:
        prefetcher.close()

    print()
    gc.collect()


# ===================== Main program =====================

def main():
    _ap_kw: dict = dict(
        description="MODIS filling: GDAL high-performance version (fast path + most-recent processed-image fallback + first-day multi-future-day merge)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Processing logic notes:
  - With --overwrite: recompute from the earliest date; the first day uses forced "future merge", streaming through subsequent dates' images and filling pixel-by-pixel (each scene loaded once),
    bounded jointly by --max-future-sources and --max_future_days.
  - Without --overwrite: first check whether each tile is already fully processed (output last date >= input last date); completed tiles are skipped.
      - If partial output exists: start from the day after the last completed day, falling back only on historical processed outputs.
      - If output is empty: start from the earliest date and use multi-future-day merge for the first day (as above; do not cache all future scenes at once).
  - --prefetch-ahead (default 10): asynchronously prefetch the input full scenes for some number of subsequent days, overlapping with the current day's compute; roughly equals the upper bound on input scenes simultaneously in memory / in flight.
    Set 0 to disable. After a successful write, the previous day's result in memory can serve as the historical fallback, avoiding rereading the output file.
  - Performance: if writes are slow, try --compress NONE or ZSTD; do not enable --enable_spatial_interp (GDAL FillNodata is very slow).
    Keeping input/output on the same grid as the previous day avoids Warp. With consecutive days, prev_cache provides the in-memory fallback, saving one output read.

Arguments with leading minus (important):
  Values for --valid_ranges / --custom_nodata that start with "-" (e.g. -100,16000) should NOT be written with a space as
  "--valid_ranges '-100,...'": argparse treats the next item as a new option and reports "expected one argument".
  Use an equals sign to join them and quote the whole semicolon-separated string, e.g.:
    --valid_ranges='-100,16000;-100,16000;None;None'
  Python 3.13+ enables allow_hyphen_values, in which case the space form is also accepted.
""",
    )
    if sys.version_info >= (3, 13):
        _ap_kw["allow_hyphen_values"] = True
    ap = argparse.ArgumentParser(**_ap_kw)

    ap.add_argument("--input_folder", type=Path, required=True,
                   help="Input folder: ..._YYYY_MM_DD[_tileXX].tif (any prefix allowed, e.g. MCD09GA_b1237_2002_07_04_tile00.tif)")
    ap.add_argument("--output_folder", type=Path, required=True,
                   help="Output folder")

    ap.add_argument("--max_past_days", type=int, default=60,
                   help="Maximum days to look back (default: 60)")
    ap.add_argument("--max_future_days", type=int, default=10240,
                   help="Maximum forward calendar-day window, used only during first-day future merging (default: 10240)")
    ap.add_argument(
        "--max-future-sources",
        type=int,
        default=512,
        help="Maximum number of \"actually existing\" subsequent images used during first-day future merging;"
        "set to 0 for no count limit (only --max_future_days constrains it). Default: 512.",
    )

    ap.add_argument("--enable_spatial_interp", action="store_true",
                   help="Enable spatial-interpolation fallback (default: disabled)")
    ap.add_argument("--spatial_interp_distance", type=int, default=100,
                   help="Maximum search distance for spatial interpolation (pixels, default: 100)")

    ap.add_argument("--custom_nodata", type=str, default=None,
                   help="Custom NoData value list, comma-separated, e.g. ='32768,-32768,1,0'")
    ap.add_argument(
        "--valid_ranges",
        type=str,
        default=None,
        help="Per-band valid range, semicolon-separated, like min,max or None. Use an equals sign for negative lower bounds:"
        "--valid_ranges='-100,16000;-100,16000;None'",
    )
    ap.add_argument("--keep_bands", type=str, default=None,
                   help="Band indices to keep (0-based), comma-separated, e.g. '0,1,2' (default: keep all bands)")

    ap.add_argument("--detect_permanent_invalid", action="store_true", default=False,
                   help="Enable permanent-invalid pixel detection and skip filling them (default: disabled)")
    ap.add_argument("--detect_top_k_bands", type=int, default=None,
                   help="Detect only the first k bands (0 to k-1) (default: detect all bands)")
    ap.add_argument("--permanent_invalid_sample_size", type=int, default=128,
                   help="Number of samples for permanent-invalid pixel detection (default: 128)")
    ap.add_argument("--permanent_invalid_workers", type=int, default=16,
                   help="Number of parallel threads for permanent-invalid detection (default: 16)")
    ap.add_argument(
        "--permanent-invalid-block",
        type=int,
        default=2048,
        help="Spatial-tile side length for permanent-invalid detection (pixels, default: 2048)",
    )
    ap.add_argument(
        "--no-permanent-invalid-stratify",
        action="store_true",
        help="Disable per-year stratified sampling (stratified is enabled by default)",
    )

    ap.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing output files (default: skip)")

    ap.add_argument(
        "--compress",
        type=str,
        default="LZW",
        help="GTiff compression (default: LZW). If writes are slow, try ZSTD, DEFLATE, or NONE",
    )
    ap.add_argument("--blocksize", type=int, default=512,
                   help="Block size (default: 512)")
    ap.add_argument(
        "--prefetch-ahead",
        type=int,
        default=10,
        help="Maximum in-flight days for asynchronously prefetching subsequent full-scene inputs; 0 disables (default: 10)",
    )

    args = ap.parse_args()

    max_future_sources: Optional[int] = (
        None if args.max_future_sources == 0 else args.max_future_sources
    )

    enable_spatial = args.enable_spatial_interp and not getattr(args, "disable_spatial_interp", False)

    custom_nodata_values = None
    if args.custom_nodata:
        try:
            custom_nodata_values = [float(x.strip()) for x in args.custom_nodata.split(',')]
        except ValueError:
            print(f"Invalid custom_nodata format: {args.custom_nodata}", file=sys.stderr)
            sys.exit(1)

    valid_ranges = None
    if args.valid_ranges:
        try:
            ranges = []
            for band_range in args.valid_ranges.split(';'):
                band_range = band_range.strip()
                if band_range.lower() == 'none':
                    ranges.append(None)
                else:
                    parts = band_range.split(',')
                    if len(parts) != 2:
                        raise ValueError(f"Band range must be in 'min,max' format: {band_range}")
                    vmin = float(parts[0].strip()) if parts[0].strip().lower() != 'none' else None
                    vmax = float(parts[1].strip()) if parts[1].strip().lower() != 'none' else None
                    ranges.append((vmin, vmax))
            valid_ranges = ranges
        except ValueError as e:
            print(f"Invalid valid_ranges format: {e}", file=sys.stderr)
            sys.exit(1)

    keep_bands = None
    if args.keep_bands:
        try:
            keep_bands = [int(x.strip()) for x in args.keep_bands.split(',')]
        except ValueError:
            print(f"Invalid keep_bands format: {args.keep_bands}", file=sys.stderr)
            sys.exit(1)

    # -- Build the input index (grouped by tile) --
    print(
        "Building input file index (supports YYYY_MM_DD.tif / YYYY_MM_DD_tileXX.tif / "
        "prefix_..._YYYY_MM_DD[_tileXX].tif, e.g. MCD09GA_b1237_2002_07_04_tile00.tif)..."
    )
    tile_groups = _build_index_by_tile(args.input_folder)
    if not tile_groups:
        print(
            f"In {args.input_folder}: no matching .tif found (filenames must end with "
            f"YYYY_MM_DD.tif or ..._YYYY_MM_DD_tileXX.tif etc.)",
            file=sys.stderr,
        )
        sys.exit(1)

    n_tiles = len(tile_groups)
    if n_tiles > 1:
        keys = [k or "(no _tile suffix)" for k in sorted(tile_groups.keys(), key=lambda x: (x is not None, x or ""))]
        print(f"Total {n_tiles} tile groups; each will be filled in time-series separately: {', '.join(keys)}")

    print(f"RAM: available {_RAM_CFG['available_gb']:.1f} GB | "
          f"available budget {_RAM_CFG['usable_gb']:.1f} GB | "
          f"GDAL block cache {_RAM_CFG['gdal_cache_gb']:.1f} GB | "
          f"Warp buffer {_RAM_CFG['warp_mem_gb']:.1f} GB")

    # -- Build the task argument bundle for each tile --
    tile_task_args = []
    for tile_suffix, input_idx in sorted(
        tile_groups.items(), key=lambda kv: (kv[0] is not None, kv[0] or "")
    ):
        tile_task_args.append((
            tile_suffix,
            input_idx,
            args,
            custom_nodata_values,
            valid_ranges,
            keep_bands,
            enable_spatial,
            max_future_sources,
        ))

    if n_tiles == 1:
        # Single tile: run directly in the main process
        _run_tile_task(tile_task_args[0])
    else:
        # Multi-tile: start as many parallel processes as there are tiles
        import multiprocessing
        n_procs = n_tiles
        print(f"\nMulti-tile parallel mode: starting {n_procs} processes (one process per tile)")
        ctx = multiprocessing.get_context("spawn")
        procs = []
        for task_args in tile_task_args:
            p = ctx.Process(target=_run_tile_task, args=(task_args,))
            p.start()
            procs.append((task_args[0], p))

        failed_tiles = []
        for tile_suffix, p in procs:
            p.join()
            tile_label = "No _tile suffix" if tile_suffix is None else tile_suffix
            if p.exitcode == 0:
                print(f"Block [{tile_label}] processing complete (exitcode=0)")
            else:
                print(f"Tile [{tile_label}] abnormal exit (exitcode={p.exitcode}）")
                failed_tiles.append(tile_label)

        if failed_tiles:
            print(f"\nThe following tiles did not complete successfully: {', '.join(failed_tiles)}")
        else:
            print("\nAll tiles processed")

    print("\n" + "=" * 60)
    print("[Done] All processing complete")
    print(f"[Output] {args.output_folder}")

if __name__ == "__main__":
    main()
