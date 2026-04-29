#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2-in-1: MOD13Q1 / MYD13Q1 masking + MCD13Q1 merge.

Input band order (8 bands; matches MOD13Q1 / MYD13Q1 in gee_download_fast_copy_parallel.py):
  1: NDVI
  2: EVI
  3: sur_refl_b01
  4: sur_refl_b02
  5: sur_refl_b03
  6: sur_refl_b07
  7: DetailedQA
  8: SummaryQA

---------------------------------------------------------------------------
DetailedQA (16 bits; bit-field summary below; masking rules in build_bad_mask_from_vi_qa)
---------------------------------------------------------------------------
Bits 0-1 -- VI quality (MODLAND QA):
  0: VI produced with good quality
  1: VI produced, but check other QA
  2: Pixel produced, but most probably cloudy
  3: Pixel not produced due to other reasons than clouds

Bits 2-5 -- VI usefulness (typically worse as the value grows; the implementation uses thresholds and sets):
  0: highest quality; 1/2/4/8/9/10/12: descending; 13: too low to use; 14: L1B faulty; 15: not processed, etc.

Bits 6-7 -- Aerosol Quantity: 0 Climatology, 1 Low, 2 Intermediate, 3 High

Bit 8 -- Adjacent cloud detected: 0 No, 1 Yes

Bit 9 -- Atmosphere BRDF correction: 0 No, 1 Yes

Bit 10 -- Mixed Clouds: 0 No, 1 Yes

Bits 11-13 -- Land/water mask (shallow ocean / land / coastline / inland water / deep ocean, etc.; by default no water masking; toggle with a flag)

Bit 14 -- Possible snow/ice; Bit 15 -- Possible shadow

---------------------------------------------------------------------------
SummaryQA (16 bits)
---------------------------------------------------------------------------
Bits 0-1 -- VI quality (MODLAND QA):
  0: Good data, use with confidence
  1: Marginal data, useful but look at detailed QA
  2: Pixel covered with snow/ice
  3: Pixel is cloudy

---------------------------------------------------------------------------
Masking rules (default conservative: drop obvious cloud / not-produced / very-low usefulness; keep medium quality)
---------------------------------------------------------------------------
  - SummaryQA bits 0-1 in {2, 3}: mask (snow/ice, cloud)
  - DetailedQA bits 0-1 in {2, 3}: mask (likely cloudy, not produced for non-cloud reasons)
  - DetailedQA bits 2-5 (usefulness) >= max_keep_usefulness + 1: default max_keep = 11, i.e. mask >= 12
    (covers 12 Lowest quality, 13-15 unusable / faulty, etc.)
  - Optional: mask Summary marginal (1), adjacent cloud (bit 8), mixed cloud (bit 10), high aerosol (6-7 == 3),
    Detailed snow/shadow (14/15), non-pure-land pixels (11-13), etc. (see CLI).

  Bands 1-6 share the same bad_mask uniformly; bands 7-8 (QA) are written back as-is (consistent with the template).

---------------------------------------------------------------------------
Merging rules (MOD + MYD -> MCD)
---------------------------------------------------------------------------
  - One side valid (not NoData) -> take all 8 bands from that side.
  - Both invalid -> all bands NoData.
  - Both valid -> pick by QC lexicographic order (smaller is better):
      (SummaryQA bits0-1, DetailedQA bits0-1, DetailedQA bits2-5 usefulness)
    The winning pixel copies NDVI / EVI / reflectance and the corresponding DetailedQA, SummaryQA from that side.
  - Ties broken by --prefer (default: mod).

NoData: uniformly -32765 across all bands (including placeholders in the QA bands when written as floats).

---------------------------------------------------------------------------
Interruption and memory (common Linux misunderstandings)
---------------------------------------------------------------------------
  - Prefer one Ctrl+C (SIGINT) in the terminal to end this script; avoid kill -9 (SIGKILL):
    SIGKILL does not run cleanup; child processes may linger as orphans holding memory; you'll need pkill yourself.
  - High "buff/cache" in `free` is mostly the OS caching previously-read GeoTIFFs and will be reclaimed on demand; it is not a leak.
  - To check whether python subprocesses still linger: `ps aux | grep mcd13q1` or `pgrep -af mcd13q1_mask_combine`.

  - Do not use osgeo.gdal ReadAsArray in the masking stage: it is often incompatible with the gdal_array extension under NumPy 2.x;
    this script uses only rasterio.read() in process_single_file.
"""

import re
import os
import sys
import math
import shutil
import signal
import argparse
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np
import rasterio
from rasterio import windows
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds, Resampling
from rasterio.errors import NotGeoreferencedWarning
import warnings

warnings.simplefilter("ignore", NotGeoreferencedWarning)

# ---------------------------------------------------------------------------
# GDAL / rasterio: tune via environment variables only; do not import osgeo.gdal
# ---------------------------------------------------------------------------
# Reason: conda's GDAL gdal_array is often compiled against NumPy 1.x, and under NumPy 2.x
# ReadAsArray() / WriteArray() raise errors. The masking stage uses rasterio.read() for the whole image, bypassing gdal_array.
def _configure_gdal_env():
    """Set in both the main and child processes (spawn re-executes the module top level); do not load osgeo Python bindings."""
    # GDAL_CACHEMAX is usually in MB (depending on the GDAL version); if not set, GDAL allocates a memory fraction automatically
    os.environ.setdefault("GDAL_CACHEMAX", "2048")
    os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")
    os.environ.setdefault("GDAL_SWATH_SIZE", str(200 * 1024 * 1024))
    os.environ.setdefault("VSI_CACHE", "TRUE")
    os.environ.setdefault("VSI_CACHE_SIZE", str(64 * 1024 * 1024))
    os.environ.setdefault("GDAL_TIFF_OVR_BLOCKSIZE", "512")


_configure_gdal_env()


# ---------------------------------------------------------------------------
# Worker initializer — called once in each subprocess at startup
# ---------------------------------------------------------------------------
def _worker_init():
    # Workers ignore SIGINT: Ctrl+C is handled exclusively by the parent,
    # which then explicitly sends SIGTERM to each worker PID.  Without this,
    # workers that are deep in C/GDAL I/O receive the signal but can't act on
    # it until the C call returns, leaving them as zombies after the parent exits.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _configure_gdal_env()


# ---------------------------------------------------------------------------
# Shutdown helpers
# ---------------------------------------------------------------------------
def _kill_executor_workers(ex: ProcessPoolExecutor | None,
                           sig: int = signal.SIGTERM) -> None:
    """Send *sig* to every live worker PID managed by *ex*."""
    if ex is None:
        return
    for pid in list(getattr(ex, "_processes", {}).keys()):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def _shutdown_executor(executor: ProcessPoolExecutor | None, *, aggressive: bool) -> None:
    if executor is None:
        return
    wait = not aggressive
    try:
        executor.shutdown(wait=wait, cancel_futures=aggressive)
    except TypeError:
        executor.shutdown(wait=wait)


N_BANDS = 8
N_VALUE = 6  # The first 6 are NDVI / EVI / reflectance

DATE_RE = re.compile(r"(?P<y>\d{4})[-_]?((?P<m>\d{2})[-_]?)(?P<d>\d{2})")
TILE_RE = re.compile(r"_tile(?P<tile>\d+)", re.IGNORECASE)


def parse_date_tile_key(p: Path) -> str | None:
    m_date = DATE_RE.search(p.stem)
    if not m_date:
        return None
    date_str = f"{m_date.group('y')}_{m_date.group('m')}_{m_date.group('d')}"
    m_tile = TILE_RE.search(p.stem)
    if m_tile:
        return f"{date_str}_tile{m_tile.group('tile')}"
    return date_str


def bitslice(arr, start, end):
    width = end - start + 1
    mask = (1 << width) - 1
    return (arr >> start) & mask


def print_progress(prefix: str, ok: int, total: int, fail: int):
    print(f"\r{prefix}: OK={ok}/{total}, FAIL={fail}", end="", flush=True)


def finish_progress():
    print("", flush=True)


def parse_detailed_qa(dqa):
    return dict(
        vi_quality=bitslice(dqa, 0, 1),
        usefulness=bitslice(dqa, 2, 5),
        aerosol=bitslice(dqa, 6, 7),
        adjacent_cloud=bitslice(dqa, 8, 8),
        brdf_corr=bitslice(dqa, 9, 9),
        mixed_clouds=bitslice(dqa, 10, 10),
        land_water=bitslice(dqa, 11, 13),
        possible_snow=bitslice(dqa, 14, 14),
        possible_shadow=bitslice(dqa, 15, 15),
    )


def parse_summary_qa(sqa):
    return dict(vi_quality=bitslice(sqa, 0, 1))


def build_bad_mask_from_vi_qa(
    detailed,
    summary,
    mask_summary_snow_cloud=True,
    mask_summary_marginal=False,
    mask_detailed_vi_23=True,
    max_keep_usefulness=11,
    mask_high_aerosol=False,
    mask_adjacent_cloud=False,
    mask_mixed_clouds=False,
    mask_detailed_snow_shadow=False,
    mask_non_land=False,
):
    d = parse_detailed_qa(detailed)
    s = parse_summary_qa(summary)

    bad = np.zeros(detailed.shape, dtype=bool)

    if mask_summary_snow_cloud:
        bad |= np.isin(s["vi_quality"], [2, 3])
    if mask_summary_marginal:
        bad |= (s["vi_quality"] == 1)

    if mask_detailed_vi_23:
        bad |= np.isin(d["vi_quality"], [2, 3])

    bad |= d["usefulness"] > max_keep_usefulness

    if mask_high_aerosol:
        bad |= (d["aerosol"] == 3)
    if mask_adjacent_cloud:
        bad |= (d["adjacent_cloud"] == 1)
    if mask_mixed_clouds:
        bad |= (d["mixed_clouds"] == 1)
    if mask_detailed_snow_shadow:
        bad |= (d["possible_snow"] == 1) | (d["possible_shadow"] == 1)
    if mask_non_land:
        bad |= (d["land_water"] != 1)

    return bad


def score_vi_for_choose(summary_arr, detailed_arr):
    s = parse_summary_qa(summary_arr)
    d = parse_detailed_qa(detailed_arr)
    return np.stack([s["vi_quality"], d["vi_quality"], d["usefulness"]], axis=0)


def lexicographic_choose(a_score, b_score, prefer_a=True):
    choose_a = np.zeros(a_score.shape[1:], dtype=bool)
    undecided = np.ones_like(choose_a, dtype=bool)
    for k in range(a_score.shape[0]):
        ak = a_score[k]
        bk = b_score[k]
        better_a = ak < bk
        better_b = ak > bk
        choose_a |= better_a & undecided
        undecided &= ~(better_a | better_b)
        if not undecided.any():
            break
    if undecided.any() and prefer_a:
        choose_a |= undecided
    return choose_a


def pick_target_grid(mod_src, myd_src):
    crs = mod_src.crs or myd_src.crs
    if crs is None:
        raise ValueError("Both inputs lack a valid CRS.")

    def res_of(src):
        a = src.transform
        return (abs(a.a), abs(a.e))

    mod_res = res_of(mod_src)
    myd_res = res_of(myd_src)
    xres = min(mod_res[0], myd_res[0])
    yres = min(mod_res[1], myd_res[1])

    mod_bounds = transform_bounds(mod_src.crs, crs, *mod_src.bounds, densify_pts=0) if mod_src.crs != crs else mod_src.bounds
    myd_bounds = transform_bounds(myd_src.crs, crs, *myd_src.bounds, densify_pts=0) if myd_src.crs != crs else myd_src.bounds
    xmin = min(mod_bounds[0], myd_bounds[0])
    ymin = min(mod_bounds[1], myd_bounds[1])
    xmax = max(mod_bounds[2], myd_bounds[2])
    ymax = max(mod_bounds[3], myd_bounds[3])

    from affine import Affine

    dst_transform = Affine(xres, 0.0, xmin, 0.0, -yres, ymax)
    width = int(math.ceil((xmax - xmin) / xres))
    height = int(math.ceil((ymax - ymin) / yres))

    return crs, dst_transform, width, height


def ensure_valid_blocksize(b, max_size=1024, min_size=16):
    b = max(min(b, max_size), min_size)
    return (b // 16) * 16 or 16


def process_single_file(
    path_in: Path,
    path_out: Path,
    out_nodata: float,
    overwrite: bool,
    mask_summary_snow_cloud: bool,
    mask_summary_marginal: bool,
    mask_detailed_vi_23: bool,
    max_keep_usefulness: int,
    mask_high_aerosol: bool,
    mask_adjacent_cloud: bool,
    mask_mixed_clouds: bool,
    mask_detailed_snow_shadow: bool,
    mask_non_land: bool,
    compress: str,
):
    if path_out.exists() and not overwrite:
        return True, path_in.name, None

    try:
        # Read all 8 bands at once with rasterio to avoid osgeo.gdal.Dataset.ReadAsArray -> gdal_array
        # (gdal_array in many environments still links against the NumPy 1.x ABI and ImportErrors under NumPy 2.x.)
        with rasterio.open(path_in) as src:
            if src.count < N_BANDS:
                return False, path_in.name, (
                    f"Required {N_BANDS} bands: NDVI, EVI, sur_refl_b01..b03, b07, DetailedQA, SummaryQA."
                )

            data = src.read(list(range(1, N_BANDS + 1)))
            detailed = data[6].astype(np.uint16, copy=False)
            summary = data[7].astype(np.uint16, copy=False)

            bad = build_bad_mask_from_vi_qa(
                detailed,
                summary,
                mask_summary_snow_cloud=mask_summary_snow_cloud,
                mask_summary_marginal=mask_summary_marginal,
                mask_detailed_vi_23=mask_detailed_vi_23,
                max_keep_usefulness=max_keep_usefulness,
                mask_high_aerosol=mask_high_aerosol,
                mask_adjacent_cloud=mask_adjacent_cloud,
                mask_mixed_clouds=mask_mixed_clouds,
                mask_detailed_snow_shadow=mask_detailed_snow_shadow,
                mask_non_land=mask_non_land,
            )

            profile = src.profile.copy()
            bx = ensure_valid_blocksize(profile.get("blockxsize", 512))
            by = ensure_valid_blocksize(profile.get("blockysize", 512))
            profile.update(
                dtype="float32",
                compress=compress,
                tiled=True,
                blockxsize=bx,
                blockysize=by,
                bigtiff="IF_SAFER",
                num_threads="all_cpus",   # multi-thread DEFLATE/ZSTD encode
            )

            # Build full 8-band output array in one shot, then write once.
            # For a tiled GeoTIFF this is much faster than 8 separate dst.write()
            # calls: GDAL can fill each tile across all bands simultaneously
            # instead of revisiting tiles on every band write.
            out_all = data.astype(np.float32)          # (8, H, W), one allocation
            out_all[:N_VALUE, bad] = out_nodata        # apply mask to bands 1-6 only

            path_out.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(path_out, "w", **profile) as dst:
                dst.write(out_all)                     # single 8-band GDAL RasterIO call
                for b in range(1, N_BANDS + 1):
                    dst.update_tags(b, nodata=str(int(out_nodata)))

        return True, path_in.name, None

    except Exception as e:
        return False, path_in.name, str(e)


def combine_one(
    date_tile_key: str,
    mod_path: Path | None,
    myd_path: Path | None,
    out_dir: Path,
    prefer_mod: bool,
    chunk: int,
    compress: str,
    out_nodata: float,
    overwrite: bool,
):
    out_filename = f"MCD13Q1_{date_tile_key}.tif"
    out_path = out_dir / out_filename

    if out_path.exists() and not overwrite:
        return True, "skipped"

    if (mod_path is None) ^ (myd_path is None):
        shutil.copy2(mod_path or myd_path, out_path)
        return True, "copied"

    with rasterio.open(mod_path) as mod, rasterio.open(myd_path) as myd:
        if mod.count < N_BANDS or myd.count < N_BANDS:
            raise ValueError(f"Input requires {N_BANDS} bands (MOD13Q1 / MYD13Q1 order).")

        crs, dst_transform, width, height = pick_target_grid(mod, myd)

        vrt_opts_float = dict(
            crs=crs, transform=dst_transform, width=width, height=height,
            resampling=Resampling.bilinear, add_alpha=False,
        )
        vrt_opts_near = dict(
            crs=crs, transform=dst_transform, width=width, height=height,
            resampling=Resampling.nearest, add_alpha=False,
        )

        bx = ensure_valid_blocksize(min(chunk, 1024))
        by = ensure_valid_blocksize(min(chunk, 1024))
        profile = dict(
            driver="GTiff", count=N_BANDS, crs=crs,
            transform=dst_transform, width=width, height=height,
            dtype="float32", tiled=True, blockxsize=bx, blockysize=by,
            compress=compress, bigtiff="IF_SAFER",
        )

        # Band index lists (1-based) for batch reads
        val_bands  = list(range(1, N_VALUE + 1))      # [1..6]
        qa_bands_n = [7, 8]                            # nearest-neighbour QA

        with (
            rasterio.open(out_path, "w", **profile) as dst,
            WarpedVRT(mod, **vrt_opts_float) as mod_vf,
            WarpedVRT(myd, **vrt_opts_float) as myd_vf,
            WarpedVRT(mod, **vrt_opts_near) as mod_vn,
            WarpedVRT(myd, **vrt_opts_near) as myd_vn,
        ):
            for row_off in range(0, height, chunk):
                h = min(chunk, height - row_off)
                for col_off in range(0, width, chunk):
                    w_px = min(chunk, width - col_off)
                    win = windows.Window(col_off, row_off, w_px, h)

                    # ── Read all value bands at once (single GDAL call each) ──
                    m_vals = mod_vf.read(val_bands, window=win)   # (6, h, w)
                    y_vals = myd_vf.read(val_bands, window=win)   # (6, h, w)

                    m_det = mod_vn.read(7, window=win).astype(np.uint16, copy=False)
                    m_sum = mod_vn.read(8, window=win).astype(np.uint16, copy=False)
                    y_det = myd_vn.read(7, window=win).astype(np.uint16, copy=False)
                    y_sum = myd_vn.read(8, window=win).astype(np.uint16, copy=False)

                    # ── Validity masks ────────────────────────────────────────
                    # Any band == out_nodata → invalid pixel
                    valid_m = ~np.any(m_vals == out_nodata, axis=0)  # (h, w)
                    valid_y = ~np.any(y_vals == out_nodata, axis=0)

                    # ── QA scoring ────────────────────────────────────────────
                    s_m = score_vi_for_choose(m_sum, m_det)
                    s_y = score_vi_for_choose(y_sum, y_det)

                    both = valid_m & valid_y
                    choose_mod = np.zeros(both.shape, dtype=bool)
                    if both.any():
                        choose_mod[both] = lexicographic_choose(
                            s_m[:, both], s_y[:, both], prefer_a=prefer_mod
                        )

                    # ── Merge value bands — fully vectorised, no Python loop ──
                    outs = np.full((N_VALUE, h, w_px), out_nodata, dtype=np.float32)
                    om = valid_m & ~valid_y
                    oy = ~valid_m & valid_y
                    am = both & choose_mod
                    bm = both & ~choose_mod

                    if om.any():
                        outs[:, om] = m_vals[:, om]
                    if oy.any():
                        outs[:, oy] = y_vals[:, oy]
                    if am.any():
                        outs[:, am] = m_vals[:, am]
                    if bm.any():
                        outs[:, bm] = y_vals[:, bm]

                    # ── Merge QA bands ────────────────────────────────────────
                    out_det = np.full((h, w_px), out_nodata, dtype=np.float32)
                    out_sum = np.full((h, w_px), out_nodata, dtype=np.float32)

                    if om.any():
                        out_det[om] = m_det[om]
                        out_sum[om] = m_sum[om]
                    if oy.any():
                        out_det[oy] = y_det[oy]
                        out_sum[oy] = y_sum[oy]
                    if am.any():
                        out_det[am] = m_det[am]
                        out_sum[am] = m_sum[am]
                    if bm.any():
                        out_det[bm] = y_det[bm]
                        out_sum[bm] = y_sum[bm]

                    # ── Write all 8 bands in one call ─────────────────────────
                    chunk_out = np.concatenate(
                        [outs, out_det[np.newaxis], out_sum[np.newaxis]], axis=0
                    )
                    dst.write(chunk_out, window=win)

            for b in range(1, N_BANDS + 1):
                dst.update_tags(b, nodata=str(int(out_nodata)))

    return True, "merged"


def list_tifs(d: Path):
    return sorted([p for p in d.iterdir() if p.suffix.lower() in (".tif", ".tiff")])


def build_pairs(mod_dir: Path, myd_dir: Path):
    mod_map, myd_map = {}, {}
    for p in list_tifs(mod_dir):
        k = parse_date_tile_key(p)
        if k:
            mod_map[k] = p
    for p in list_tifs(myd_dir):
        k = parse_date_tile_key(p)
        if k:
            myd_map[k] = p
    keys = sorted(set(mod_map) | set(myd_map))
    return [(k, mod_map.get(k), myd_map.get(k)) for k in keys]


def mask_dir(in_dir: Path, out_dir: Path, out_nodata: float, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    tifs = list_tifs(in_dir)
    total = len(tifs)
    prefix = f"[MASK] {in_dir.name} → {out_dir.name}"
    print_progress(prefix, 0, total, 0)

    tasks = [
        (
            p,
            out_dir / p.name,
            out_nodata,
            args.overwrite,
            args.mask_summary_snow_cloud,
            args.mask_summary_marginal,
            args.mask_detailed_vi_23,
            args.max_keep_usefulness,
            args.mask_high_aerosol,
            args.mask_adjacent_cloud,
            args.mask_mixed_clouds,
            args.mask_detailed_snow_shadow,
            args.mask_non_land,
            args.compress,
        )
        for p in tifs
    ]

    ok, fail = 0, 0
    # Each worker peaks at ~3 GB RAM (1.5 GB read + float32 copy).
    # Cap at 4 by default so 4 × 3 GB = 12 GB fits within a 35 GB budget
    # alongside GDAL caches and OS overhead.  Pass --mask-workers to override.
    default_mask = max(1, min(cpu_count() // 4, 4))
    num_processes = max(
        1, args.mask_workers if args.mask_workers is not None else default_mask
    )
    ex: ProcessPoolExecutor | None = None
    aggressive_shutdown = False
    try:
        ex = ProcessPoolExecutor(max_workers=num_processes, initializer=_worker_init)
        futs = {ex.submit(process_single_file, *task): task for task in tasks}
        for fut in as_completed(futs):
            try:
                success, fn, error_msg = fut.result()
                if success:
                    ok += 1
                else:
                    fail += 1
                    if error_msg:
                        print(f"\nError {fn}: {error_msg}")
            except Exception as e:
                fail += 1
                task = futs[fut]
                print(f"\nException {task[0].name}: {e}")
            finally:
                print_progress(prefix, ok, total, fail)
    except KeyboardInterrupt:
        aggressive_shutdown = True
        print("\n\nInterrupt: sending SIGTERM to child processes...", file=sys.stderr)
        _kill_executor_workers(ex)   # explicit SIGTERM — workers won't self-terminate
        raise                        # because they ignore SIGINT and may be in C I/O
    finally:
        _shutdown_executor(ex, aggressive=aggressive_shutdown)
        finish_progress()


def combine_dirs(mod_masked: Path, myd_masked: Path, out_dir: Path, out_nodata: float, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = build_pairs(mod_masked, myd_masked)
    total = len(pairs)
    prefix = f"[COMBINE] {mod_masked.name}+{myd_masked.name} → {out_dir.name}"
    print_progress(prefix, 0, total, 0)

    task = partial(
        combine_one,
        out_dir=out_dir,
        prefer_mod=(args.prefer == "mod"),
        chunk=args.chunk,
        compress=args.compress,
        out_nodata=out_nodata,
        overwrite=args.overwrite,
    )
    ok, fail = 0, 0
    ex: ProcessPoolExecutor | None = None
    aggressive_shutdown = False
    try:
        ex = ProcessPoolExecutor(max_workers=max(1, args.workers), initializer=_worker_init)
        futs = {ex.submit(task, k, mp, yp): k for (k, mp, yp) in pairs}
        for fut in as_completed(futs):
            try:
                fut.result()
                ok += 1
            except Exception as e:
                fail += 1
                print(f"\nException: {e}")
            finally:
                print_progress(prefix, ok, total, fail)
    except KeyboardInterrupt:
        aggressive_shutdown = True
        print("\n\nInterrupt: sending SIGTERM to child processes...", file=sys.stderr)
        _kill_executor_workers(ex)
        raise
    finally:
        _shutdown_executor(ex, aggressive=aggressive_shutdown)
        finish_progress()


def main():
    ap = argparse.ArgumentParser(
        description="MOD13Q1 / MYD13Q1 masking + MCD13Q1 merge (DetailedQA / SummaryQA; merges by QC lexicographic order)"
    )
    ap.add_argument("--mod_in", type=Path, default='/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MOD13Q1', help="MOD13Q1 input directory")
    ap.add_argument("--myd_in", type=Path, default='/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MYD13Q1', help="MYD13Q1 input directory")
    ap.add_argument("--mod_out", type=Path, default='/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MOD13Q1_QAapplied', help="MOD13Q1 masked-output directory")
    ap.add_argument("--myd_out", type=Path, default='/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MYD13Q1_QAapplied', help="MYD13Q1 masked-output directory")
    ap.add_argument("--mcd_out", type=Path, default='/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/MCD13Q1_QAapplied', help="MCD13Q1 merge-output directory")

    ap.add_argument("--prefer", choices=["mod", "myd"], default="mod", help="Tie-break priority when scores are equal (default: mod)")
    ap.add_argument("--chunk", type=int, default=2048, help="Tile size during merging (pixels, default 2048)")
    ap.add_argument("--workers", type=int, default=8, help="Number of merge parallel processes")
    ap.add_argument(
        "--mask-workers",
        type=int,
        default=None,
        help="Number of parallel processes in the masking stage (default cpu_count // 4); lower it to reduce peak memory",
    )
    ap.add_argument("--compress", default="LZW", help="Output compression (default: LZW)")
    ap.add_argument("--out-nodata", type=float, default=-32765, help="Unified NoData (default: -32765)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output")

    ap.add_argument(
        "--no-mask-summary-snow-cloud",
        action="store_true",
        help="Loosen: do not mask due to SummaryQA snow/cloud (2, 3) (not recommended)",
    )
    ap.add_argument(
        "--mask-summary-marginal",
        action="store_true",
        help="Tighten: also mask SummaryQA vi_quality == 1 (marginal)",
    )
    ap.add_argument(
        "--no-mask-detailed-vi-23",
        action="store_true",
        help="Loosen: do not mask due to DetailedQA vi_quality 2/3",
    )
    ap.add_argument(
        "--max-keep-usefulness",
        type=int,
        default=11,
        help="DetailedQA usefulness (bits 2-5) upper-bound to keep: mask if greater. Default 11 means mask >= 12",
    )
    ap.add_argument(
        "--mask-high-aerosol",
        action="store_true",
        help="Mask when DetailedQA bits 6-7 == 3 (high aerosol)",
    )
    ap.add_argument(
        "--mask-adjacent-cloud",
        action="store_true",
        help="Mask DetailedQA bit 8 (adjacent cloud)",
    )
    ap.add_argument(
        "--mask-mixed-clouds",
        action="store_true",
        help="Mask DetailedQA bit 10 (mixed cloud)",
    )
    ap.add_argument(
        "--mask-detailed-snow-shadow",
        action="store_true",
        help="Mask DetailedQA bits 14 / 15 (possible snow-ice / shadow)",
    )
    ap.add_argument(
        "--mask-non-land",
        action="store_true",
        help="Keep only land/water == 1 (pure land); mask water, coastlines, etc.",
    )

    args = ap.parse_args()

    args.mask_summary_snow_cloud = not args.no_mask_summary_snow_cloud
    args.mask_detailed_vi_23 = not args.no_mask_detailed_vi_23

    mask_dir(args.mod_in, args.mod_out, args.out_nodata, args)
    mask_dir(args.myd_in, args.myd_out, args.out_nodata, args)
    combine_dirs(args.mod_out, args.myd_out, args.mcd_out, args.out_nodata, args)


if __name__ == "__main__":
    main()
