#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MCD09GA batch low-confidence pixel masking and quality merging (6-band version: 4 observations + 2 QA).

Features:
1. Mask low-confidence pixels in MOD and MYD files separately.
2. Merge masked MOD and MYD files by quality (best-of, by QA score).
3. If only one side exists for a date+tile: mask only that side and output as an MCD09GA-named result (no two-source merging).

Merging strategy:
- One side valid -> take that side.
- Both invalid -> NoData.
- Both valid -> pick by QA score (lexicographic comparison).
- Ties broken by --prefer (default: mod).

New rule:
- cloud_only: filter only by cloud-related flags (default: Cloudy only); optionally include Mixed, cloud shadow, adjacent-to-cloud, internal cloud.

Performance notes:
- When MOD/MYD share CRS, size, and affine, take the "same-grid fast path": no WarpedVRT; each window does only 2 reads (6 bands each).
- Enable GDAL_NUM_THREADS=ALL_CPUS in the environment; use --workers for multiple files; if writes are slow, try --compress NONE.
"""

import os
import sys
import argparse
import re
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio import windows
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds, Resampling
from rasterio.errors import NotGeoreferencedWarning
import warnings
from tqdm import tqdm

warnings.simplefilter("ignore", NotGeoreferencedWarning)


def _gdal_thread_env():
    """Let decoding / resampling use multiple cores (within each process)."""
    return rasterio.Env(
        GDAL_NUM_THREADS="ALL_CPUS",
        NUM_THREADS="ALL_CPUS",
    )


def _transform_close(ta, tb, eps: float = 1e-6) -> bool:
    return all(
        abs(float(getattr(ta, k)) - float(getattr(tb, k))) <= eps
        for k in ("a", "b", "c", "d", "e", "f")
    )


def _rasters_same_grid(mod_src, myd_src) -> bool:
    """Whether MOD and MYD share the same CRS, size, and affine (common case: WarpedVRT resampling can be skipped)."""
    if mod_src.crs != myd_src.crs:
        return False
    if mod_src.width != myd_src.width or mod_src.height != myd_src.height:
        return False
    return _transform_close(mod_src.transform, myd_src.transform)


def _merge_mod_myd_chunk_to_dst(
    m_obs: np.ndarray,
    m_qa1: np.ndarray,
    m_qa2: np.ndarray,
    y_obs: np.ndarray,
    y_qa1: np.ndarray,
    y_qa2: np.ndarray,
    dst,
    win,
    args,
    out_nodata: float,
) -> None:
    """
    Process one window end-to-end: mask + quality merge + write 6 bands.
    m_obs / y_obs: (4, h, w) float32; QA is an integer array compatible with uint32 bitwise ops.
    """
    m_invalid = build_invalid_mask(
        m_qa1, m_qa2,
        rule=args.rule,
        allow_assumed_clear=args.allow_assumed_clear,
        max_aerosol=args.max_aerosol,
        max_cirrus=args.max_cirrus,
        ignore_adjacent_cloud=args.ignore_adjacent_cloud,
        check_bands=(1, 2, 3, 4),
        include_mixed=args.include_mixed,
        also_mask_shadow=args.also_mask_shadow,
        also_mask_adjacent=args.also_mask_adjacent,
        also_mask_internal_cloud=args.also_mask_internal_cloud,
    )
    y_invalid = build_invalid_mask(
        y_qa1, y_qa2,
        rule=args.rule,
        allow_assumed_clear=args.allow_assumed_clear,
        max_aerosol=args.max_aerosol,
        max_cirrus=args.max_cirrus,
        ignore_adjacent_cloud=args.ignore_adjacent_cloud,
        check_bands=(1, 2, 3, 4),
        include_mixed=args.include_mixed,
        also_mask_shadow=args.also_mask_shadow,
        also_mask_adjacent=args.also_mask_adjacent,
        also_mask_internal_cloud=args.also_mask_internal_cloud,
    )

    m_obs = np.asarray(m_obs, dtype=np.float32)
    y_obs = np.asarray(y_obs, dtype=np.float32)
    m_obs[:, m_invalid] = out_nodata
    y_obs[:, y_invalid] = out_nodata

    m_valid = m_obs != out_nodata
    y_valid = y_obs != out_nodata
    any_m = np.any(m_valid, axis=0)
    any_y = np.any(y_valid, axis=0)

    m_score = score_quality(
        np.asarray(m_qa1, dtype=np.uint32),
        np.asarray(m_qa2, dtype=np.uint32),
        check_bands=(1, 2, 3, 4),
    )
    y_score = score_quality(
        np.asarray(y_qa1, dtype=np.uint32),
        np.asarray(y_qa2, dtype=np.uint32),
        check_bands=(1, 2, 3, 4),
    )

    choose_mod = lexicographic_choose(m_score, y_score, prefer_a=(args.prefer == "mod"))

    only_m = any_m & ~any_y
    only_y = ~any_m & any_y
    both = any_m & any_y
    use_mod = only_m | (both & choose_mod)
    use_myd = only_y | (both & ~choose_mod)

    h, w = m_obs.shape[1], m_obs.shape[2]
    out_obs = np.full((4, h, w), out_nodata, dtype=np.float32)
    use_mod_3d = use_mod[np.newaxis, :, :]
    use_myd_3d = use_myd[np.newaxis, :, :]
    np.copyto(out_obs, m_obs, where=use_mod_3d)
    np.copyto(out_obs, y_obs, where=use_myd_3d)

    m_qa1f = m_qa1.astype(np.float32)
    m_qa2f = m_qa2.astype(np.float32)
    y_qa1f = y_qa1.astype(np.float32)
    y_qa2f = y_qa2.astype(np.float32)
    out_qa1 = np.where(use_mod, m_qa1f, np.float32(0))
    np.copyto(out_qa1, y_qa1f, where=use_myd)
    out_qa2 = np.where(use_mod, m_qa2f, np.float32(0))
    np.copyto(out_qa2, y_qa2f, where=use_myd)

    out_block = np.empty((6, h, w), dtype=np.float32)
    out_block[:4] = out_obs
    out_block[4] = out_qa1
    out_block[5] = out_qa2
    dst.write(out_block, indexes=(1, 2, 3, 4, 5, 6), window=win)


def bitslice(arr, start, end):
    width = end - start + 1
    mask = (1 << width) - 1
    return (arr >> start) & mask

def get_bit(arr, bit):
    return (arr >> bit) & 1

def qa1_fields(qa1):
    return dict(
        modland=bitslice(qa1, 0, 1),
        dq_b1=bitslice(qa1, 2, 5),
        dq_b2=bitslice(qa1, 6, 9),
        dq_b3=bitslice(qa1, 10, 13),
        dq_b4=bitslice(qa1, 14, 17),
        dq_b5=bitslice(qa1, 18, 21),
        dq_b6=bitslice(qa1, 22, 25),
        dq_b7=bitslice(qa1, 26, 29),
        ac=get_bit(qa1, 30),
        adj=get_bit(qa1, 31),
    )

def qa2_fields(qa2):
    return dict(
        cloud_state=bitslice(qa2, 0, 1),   # 0 clear, 1 cloudy, 2 mixed, 3 assumed clear
        cloud_shadow=get_bit(qa2, 2),
        land_water=bitslice(qa2, 3, 5),
        aerosol=bitslice(qa2, 6, 7),
        cirrus=bitslice(qa2, 8, 9),
        internal_cloud=get_bit(qa2, 10),
        internal_fire=get_bit(qa2, 11),
        mod35_snow=get_bit(qa2, 12),
        adjacent_cloud=get_bit(qa2, 13),
        brdf_corrected=get_bit(qa2, 14),
        internal_snow=get_bit(qa2, 15),
    )

def build_invalid_mask(
    qa1, qa2,
    rule="strict",
    allow_assumed_clear=False,
    max_aerosol=2,
    max_cirrus=0,
    ignore_adjacent_cloud=False,
    check_bands=(1, 2, 3, 4),
    include_mixed=False,
    also_mask_shadow=False,
    also_mask_adjacent=False,
    also_mask_internal_cloud=False,
):
    q1 = qa1_fields(qa1)
    q2 = qa2_fields(qa2)

    if rule == "cloud_only":
        invalid = (q2["cloud_state"] == 1)
        if include_mixed:
            invalid |= (q2["cloud_state"] == 2)
        if also_mask_internal_cloud:
            invalid |= (q2["internal_cloud"] == 1)
        if also_mask_shadow:
            invalid |= (q2["cloud_shadow"] == 1)
        if also_mask_adjacent:
            invalid |= (q2["adjacent_cloud"] == 1)
        return invalid

    # ===== Original strict / moderate rules =====
    if rule == "strict":
        ok_modland = (q1["modland"] == 0)
    elif rule == "moderate":
        ok_modland = np.isin(q1["modland"], [0, 1])
    else:
        raise ValueError(f"Unknown rule: {rule}")

    dq_map = {1: q1["dq_b1"], 2: q1["dq_b2"], 3: q1["dq_b3"], 4: q1["dq_b4"],
              5: q1["dq_b5"], 6: q1["dq_b6"], 7: q1["dq_b7"]}
    if rule == "strict":
        ok_dq = np.ones(qa1.shape, dtype=bool)
        for b in check_bands:
            ok_dq &= (dq_map[b] == 0)
    else:  # moderate
        ok_dq = np.ones(qa1.shape, dtype=bool)
        for b in check_bands:
            ok_dq &= (dq_map[b] <= 1)

    if allow_assumed_clear:
        ok_cloud_state = np.isin(q2["cloud_state"], [0, 3])
    else:
        ok_cloud_state = (q2["cloud_state"] == 0)

    ok_cloud_shadow = (q2["cloud_shadow"] == 0)
    ok_internal_cloud = (q2["internal_cloud"] == 0)
    ok_snow = (q2["mod35_snow"] == 0) & (q2["internal_snow"] == 0)
    ok_aerosol = (q2["aerosol"] <= max_aerosol)
    ok_cirrus = (q2["cirrus"] <= max_cirrus)
    ok_adjacent = (q2["adjacent_cloud"] == 0) if not ignore_adjacent_cloud else np.ones(qa2.shape, dtype=bool)

    ok_all = (ok_modland & ok_dq &
              ok_cloud_state & ok_cloud_shadow & ok_internal_cloud &
              ok_adjacent & ok_snow & ok_aerosol & ok_cirrus)

    return ~ok_all

def score_quality(qa1_arr, qa2_arr, check_bands=(1, 2, 3, 4)):
    """
    Build a quality-score array from QA1 and QA2.
    Smaller score is better; used for lexicographic comparison.
    
    Returns:
        score: array of shape (K, H, W)
        Score dimensions in order: [modland, max_dq_bands, cloud_state, aerosol, cirrus, cloud_shadow, internal_cloud]
    """
    q1 = qa1_fields(qa1_arr)
    q2 = qa2_fields(qa2_arr)
    
    # 1. MODLAND QA (0-1): smaller is better
    modland = q1["modland"]
    
    # 2. Maximum DQ value (across the checked bands): smaller is better
    dq_map = {1: q1["dq_b1"], 2: q1["dq_b2"], 3: q1["dq_b3"], 4: q1["dq_b4"]}
    max_dq = np.zeros_like(modland)
    for b in check_bands:
        max_dq = np.maximum(max_dq, dq_map[b])
    
    # 3. Cloud state (0-3): 0 = clear is best, 1 = cloudy is worst
    cloud_state = q2["cloud_state"]
    
    # 4. Aerosol (0-3): smaller is better
    aerosol = q2["aerosol"]
    
    # 5. Cirrus (0-3): smaller is better
    cirrus = q2["cirrus"]
    
    # 6. Cloud shadow (0-1): 0 is best
    cloud_shadow = q2["cloud_shadow"].astype(np.uint8)
    
    # 7. Internal cloud (0-1): 0 is best
    internal_cloud = q2["internal_cloud"].astype(np.uint8)
    
    return np.stack([
        modland, max_dq, cloud_state, aerosol, cirrus, cloud_shadow, internal_cloud
    ], axis=0)

def lexicographic_choose(a_score, b_score, prefer_a=True):
    """
    Lexicographic comparison to pick the better data (compare only on undecided pixels, reducing large temporary arrays).
    """
    choose_a = np.zeros(a_score.shape[1:], dtype=bool)
    undecided = np.ones(a_score.shape[1:], dtype=bool)
    for k in range(a_score.shape[0]):
        if not undecided.any():
            break
        ak = a_score[k]
        bk = b_score[k]
        better_a = undecided & (ak < bk)
        better_b = undecided & (ak > bk)
        choose_a |= better_a
        undecided &= ~(better_a | better_b)
    if prefer_a:
        choose_a |= undecided
    return choose_a

def pick_target_grid(mod_src, myd_src):
    """Select the target grid: prefer mod's CRS; take the finer resolution; take the union extent"""
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

def ensure_valid_blocksize(b, max_size=4096, min_size=16):
    b = max(min(b, max_size), min_size)
    return (b // 16) * 16 or 16

def extract_file_key(filename):
    """Extract the match key (date and tile) from a filename"""
    pattern = r'(\d{4}_\d{2}_\d{2})_tile(\d+)'
    match = re.search(pattern, filename)
    if match:
        date_str = match.group(1)
        tile_num = match.group(2)
        return (date_str, f'tile{tile_num}')
    return None

def pick_nodata(dtype, existing_nodata):
    if existing_nodata is not None:
        return existing_nodata
    if np.issubdtype(dtype, np.floating):
        return np.nan
    return -32768

def process_single_file_only(
    src_path: Path,
    dst_path: Path,
    args,
    out_nodata: float = -32768,
    chunk: int = 1024,
):
    """
    Process a single file: mask, then copy directly (used when the other file cannot be opened).
    """
    with _gdal_thread_env():
        with rasterio.open(src_path) as src:
            if src.count < 6:
                raise ValueError(f"{src_path.name}: 6 bands required (4 observations + 2 QA).")

            profile = src.profile.copy()
            bsz = ensure_valid_blocksize(chunk)
            profile.update(
                dtype='float32',
                compress=args.compress,
                tiled=True,
                blockxsize=bsz,
                blockysize=bsz,
                bigtiff="IF_SAFER",
                nodata=out_nodata
            )

            dst_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(dst_path, "w", **profile) as dst:
                for row_off in range(0, src.height, chunk):
                    h = min(chunk, src.height - row_off)
                    for col_off in range(0, src.width, chunk):
                        w = min(chunk, src.width - col_off)
                        win = windows.Window(col_off, row_off, w, h)

                        stk = src.read(indexes=(1, 2, 3, 4, 5, 6), window=win)
                        obs = stk[:4].astype(np.float32)
                        qa1 = stk[4].astype(np.uint32)
                        qa2 = stk[5].astype(np.uint32)

                        invalid = build_invalid_mask(
                            qa1, qa2,
                            rule=args.rule,
                            allow_assumed_clear=args.allow_assumed_clear,
                            max_aerosol=args.max_aerosol,
                            max_cirrus=args.max_cirrus,
                            ignore_adjacent_cloud=args.ignore_adjacent_cloud,
                            check_bands=(1, 2, 3, 4),
                            include_mixed=args.include_mixed,
                            also_mask_shadow=args.also_mask_shadow,
                            also_mask_adjacent=args.also_mask_adjacent,
                            also_mask_internal_cloud=args.also_mask_internal_cloud,
                        )

                        obs[:, invalid] = out_nodata

                        out_block = np.empty((6, h, w), dtype=np.float32)
                        out_block[:4] = obs
                        out_block[4] = qa1.astype(np.float32)
                        out_block[5] = qa2.astype(np.float32)
                        dst.write(out_block, indexes=(1, 2, 3, 4, 5, 6), window=win)

def combine_one_pair(
    mod_path: Path,
    myd_path: Path,
    dst_path: Path,
    args,
    out_nodata: float = -32768,
    chunk: int = 1024,
):
    """
    Merge a MOD/MYD pair: mask + quality merge.
    
    If one file cannot be opened, use the other directly.
    
    Args:
        mod_path: MOD file path
        myd_path: MYD file path
        dst_path: output file path
        args: command-line arguments
        out_nodata: unified NoData value
        chunk: tile size
    """
    # Try opening both files
    mod_src = None
    myd_src = None
    mod_error = None
    myd_error = None
    
    try:
        mod_src = rasterio.open(mod_path)
        if mod_src.count < 6:
            mod_src.close()
            mod_src = None
            mod_error = f"{mod_path.name}: 6 bands required"
    except Exception as e:
        mod_error = str(e)
    
    try:
        myd_src = rasterio.open(myd_path)
        if myd_src.count < 6:
            myd_src.close()
            myd_src = None
            myd_error = f"{myd_path.name}: 6 bands required"
    except Exception as e:
        myd_error = str(e)
    
    # If neither file can be opened, error out
    if mod_src is None and myd_src is None:
        raise RuntimeError(f"Cannot open either file. MOD error: {mod_error}; MYD error: {myd_error}")
    
    # If only one file is available, use it directly
    if mod_src is None:
        if myd_src:
            myd_src.close()
        process_single_file_only(myd_path, dst_path, args, out_nodata, chunk)
        return
    
    if myd_src is None:
        if mod_src:
            mod_src.close()
        process_single_file_only(mod_path, dst_path, args, out_nodata, chunk)
        return
    
    # Both files are available; merge them
    with mod_src, myd_src:
        if mod_src.count < 6 or myd_src.count < 6:
            raise ValueError("Input must have 6 bands (4 observations + 2 QA).")

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        bx = ensure_valid_blocksize(chunk)

        # ---------- Fast path: MOD/MYD already on the same grid (skip WarpedVRT, only 2 reads per block) ----------
        if _rasters_same_grid(mod_src, myd_src):
            width, height = mod_src.width, mod_src.height
            profile = dict(
                driver="GTiff",
                count=6,
                crs=mod_src.crs,
                transform=mod_src.transform,
                width=width,
                height=height,
                dtype="float32",
                tiled=True,
                blockxsize=bx,
                blockysize=bx,
                compress=args.compress,
                bigtiff="IF_SAFER",
                nodata=out_nodata,
            )
            with _gdal_thread_env():
                with rasterio.open(dst_path, "w", **profile) as dst:
                    for row_off in range(0, height, chunk):
                        h = min(chunk, height - row_off)
                        for col_off in range(0, width, chunk):
                            w = min(chunk, width - col_off)
                            win = windows.Window(col_off, row_off, w, h)
                            sm = mod_src.read(indexes=(1, 2, 3, 4, 5, 6), window=win)
                            sy = myd_src.read(indexes=(1, 2, 3, 4, 5, 6), window=win)
                            m_obs = sm[:4].astype(np.float32)
                            y_obs = sy[:4].astype(np.float32)
                            m_qa1 = sm[4].astype(np.uint32)
                            m_qa2 = sm[5].astype(np.uint32)
                            y_qa1 = sy[4].astype(np.uint32)
                            y_qa2 = sy[5].astype(np.uint32)
                            _merge_mod_myd_chunk_to_dst(
                                m_obs, m_qa1, m_qa2, y_obs, y_qa1, y_qa2,
                                dst, win, args, out_nodata,
                            )
            return

        # ---------- Slow path: grids differ, fall back to WarpedVRT resampling ----------
        crs, dst_transform, width, height = pick_target_grid(mod_src, myd_src)
        vrt_opts_float = dict(
            crs=crs,
            transform=dst_transform,
            width=width,
            height=height,
            resampling=Resampling.bilinear,
            add_alpha=False,
        )
        vrt_opts_near = dict(
            crs=crs,
            transform=dst_transform,
            width=width,
            height=height,
            resampling=Resampling.nearest,
            add_alpha=False,
        )
        profile = dict(
            driver="GTiff",
            count=6,
            crs=crs,
            transform=dst_transform,
            width=width,
            height=height,
            dtype="float32",
            tiled=True,
            blockxsize=bx,
            blockysize=bx,
            compress=args.compress,
            bigtiff="IF_SAFER",
            nodata=out_nodata,
        )
        with _gdal_thread_env():
            with rasterio.open(dst_path, "w", **profile) as dst, \
                 WarpedVRT(mod_src, **vrt_opts_float) as mod_vrt_f, \
                 WarpedVRT(myd_src, **vrt_opts_float) as myd_vrt_f, \
                 WarpedVRT(mod_src, **vrt_opts_near) as mod_vrt_n, \
                 WarpedVRT(myd_src, **vrt_opts_near) as myd_vrt_n:

                for row_off in range(0, height, chunk):
                    h = min(chunk, height - row_off)
                    for col_off in range(0, width, chunk):
                        w = min(chunk, width - col_off)
                        win = windows.Window(col_off, row_off, w, h)
                        m_obs = mod_vrt_f.read(
                            indexes=(1, 2, 3, 4), window=win
                        ).astype(np.float32)
                        y_obs = myd_vrt_f.read(
                            indexes=(1, 2, 3, 4), window=win
                        ).astype(np.float32)
                        m_qast = mod_vrt_n.read(
                            indexes=(5, 6), window=win
                        ).astype(np.uint32)
                        y_qast = myd_vrt_n.read(
                            indexes=(5, 6), window=win
                        ).astype(np.uint32)
                        m_qa1, m_qa2 = m_qast[0], m_qast[1]
                        y_qa1, y_qa2 = y_qast[0], y_qast[1]
                        _merge_mod_myd_chunk_to_dst(
                            m_obs, m_qa1, m_qa2, y_obs, y_qa1, y_qa2,
                            dst, win, args, out_nodata,
                        )


def _process_work_item(item, args_dict):
    """
    Process one work item (pair / mod_only / myd_only); used by multiprocess parallelism.
    args_dict is the dict form of an argparse.Namespace (Namespace is not reliably pickleable).
    Returns (label, ok: bool, error_msg: str | None)
    """
    args = argparse.Namespace(**args_dict)
    kind = item[0]
    try:
        if kind == "pair":
            _, key, mod_path, myd_path = item
            output_name = build_mcd_output_name(mod_path, args.suffix)
            dst_path = args.output_dir / output_name
            label = f"{mod_path.name} + {myd_path.name}"
            if dst_path.exists() and not args.overwrite:
                return (label, True, None)
            combine_one_pair(mod_path, myd_path, dst_path, args, args.out_nodata, args.chunk)
        elif kind == "mod_only":
            _, key, mod_path = item
            output_name = build_mcd_output_name(mod_path, args.suffix)
            dst_path = args.output_dir / output_name
            label = f"{mod_path.name} (MOD only)"
            if dst_path.exists() and not args.overwrite:
                return (label, True, None)
            process_single_file_only(mod_path, dst_path, args, args.out_nodata, args.chunk)
        else:
            _, key, myd_path = item
            output_name = build_mcd_output_name(myd_path, args.suffix)
            dst_path = args.output_dir / output_name
            label = f"{myd_path.name} (MYD only)"
            if dst_path.exists() and not args.overwrite:
                return (label, True, None)
            process_single_file_only(myd_path, dst_path, args, args.out_nodata, args.chunk)
        return (label, True, None)
    except Exception as e:
        label = str(item[1]) if len(item) > 1 else str(item)
        return (label, False, str(e))


def build_mcd_output_name(src_path: Path, suffix: str) -> str:
    """MOD09GA / MYD09GA input filename -> MCD09GA output filename (keeping the rest of the name and extension)."""
    name = src_path.name
    if "MOD09GA" in name:
        base = name.replace("MOD09GA", "MCD09GA", 1)
    elif "MYD09GA" in name:
        base = name.replace("MYD09GA", "MCD09GA", 1)
    else:
        base = name
    if suffix:
        stem = Path(base).stem
        ext = Path(base).suffix
        return stem + suffix + ext
    return base


def main():
    parser = argparse.ArgumentParser(description="Mask low-confidence pixels and quality-merge MOD/MYD09GA files (6-band output: 4 obs + 2 QA).")
    parser.add_argument("--mod_dir", default='/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MOD09GA_b1237_withQA', type=Path, help="MOD09GA input directory")
    parser.add_argument("--myd_dir", default='/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MYD09GA_b1237_withQA', type=Path, help="MYD09GA input directory")
    parser.add_argument("--output_dir", default='/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/MCD09GA_b1237_withQA', type=Path, help="Output directory")
    parser.add_argument("--suffix", default="", help="Output filename suffix")
    parser.add_argument(
        "--compress",
        default="LZW",
        help="GeoTIFF compression (LZW is slower; for speed, try NONE or ZSTD)",
    )

    parser.add_argument("--rule", choices=["strict", "moderate", "cloud_only"], default="cloud_only")

    # Masking rule details
    parser.add_argument("--allow_assumed_clear", action="store_true")
    parser.add_argument("--max_aerosol", type=int, default=2)
    parser.add_argument("--max_cirrus", type=int, default=0)
    parser.add_argument("--ignore_adjacent_cloud", action="store_true")

    # Cloud-specific options
    parser.add_argument("--include_mixed", action="store_true", help="Treat cloud_state = Mixed as cloud too")
    parser.add_argument("--also_mask_shadow", action="store_true", help="Also mask cloud-shadow pixels")
    parser.add_argument("--also_mask_adjacent", action="store_true", help="Also mask adjacent-to-cloud pixels")
    parser.add_argument("--also_mask_internal_cloud", action="store_true", help="Also mask internal-cloud pixels")

    # Merge parameters
    parser.add_argument("--prefer", choices=["mod", "myd"], default="mod", help="Tie-break preference for the score (default: mod)")
    parser.add_argument(
        "--chunk",
        type=int,
        default=2048,
        help="Read/write block side length (default 2048; on the same-grid fast path, larger means bigger reads and fewer loops)",
    )
    parser.add_argument("--out-nodata", type=float, default=-32768, help="Unified NoData value (default: -32768)")

    parser.add_argument("--overwrite", action="store_true")
    _ncpu = os.cpu_count() or 8
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(16, _ncpu)),
        help=f"Number of parallel processes (default min(32, CPU cores) ~= {max(1, min(32, _ncpu))}；"
        "On network storage like S3, very high concurrency can actually be slower; 4-8 may work better",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.mod_dir.exists():
        print(f"Note: MOD directory does not exist; will only process the MYD side (if any): {args.mod_dir}", file=sys.stderr)
        mod_tifs = []
    else:
        mod_tifs = sorted(
            [p for p in args.mod_dir.iterdir() if p.suffix.lower() in (".tif", ".tiff")]
        )

    if not args.myd_dir.exists():
        print(f"Note: MYD directory does not exist; will only process the MOD side (if any): {args.myd_dir}", file=sys.stderr)
        myd_tifs = []
    else:
        myd_tifs = sorted(
            [p for p in args.myd_dir.iterdir() if p.suffix.lower() in (".tif", ".tiff")]
        )

    if not mod_tifs and not myd_tifs:
        print("Neither the MOD nor the MYD directory has usable TIFs (or both directories are missing).", file=sys.stderr)
        sys.exit(1)

    # Build a file index keyed by (date + tile)
    mod_files = {}
    for f in mod_tifs:
        key = extract_file_key(f.name)
        if key:
            mod_files[key] = f

    myd_files = {}
    for f in myd_tifs:
        key = extract_file_key(f.name)
        if key:
            myd_files[key] = f

    all_keys = sorted(set(mod_files.keys()) | set(myd_files.keys()))
    if not all_keys:
        print("No valid keys parsed (filenames must match YYYY_MM_DD_tileNN.tif).", file=sys.stderr)
        sys.exit(1)

    work = []
    n_pair = n_mod_only = n_myd_only = 0
    for key in all_keys:
        has_m = key in mod_files
        has_y = key in myd_files
        if has_m and has_y:
            work.append(("pair", key, mod_files[key], myd_files[key]))
            n_pair += 1
        elif has_m:
            work.append(("mod_only", key, mod_files[key]))
            n_mod_only += 1
        else:
            work.append(("myd_only", key, myd_files[key]))
            n_myd_only += 1

    print(
        f"Task statistics: MOD+MYD pairs {n_pair} | MOD only {n_mod_only} | MYD only {n_myd_only} | total {len(work)}"
    )

    ok, fail = 0, 0
    failed_files = []

    n_workers = max(1, args.workers)
    args_dict = vars(args)

    if n_workers <= 1:
        for item in tqdm(work, desc="Processing"):
            label, success, err = _process_work_item(item, args_dict)
            if success:
                ok += 1
            else:
                fail += 1
                failed_files.append((label, err))
                print(f"[FAILED] {label}: {err}", file=sys.stderr)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from functools import partial

        fn = partial(_process_work_item, args_dict=args_dict)
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(fn, item): item for item in work}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Processing (x{n_workers})"):
                label, success, err = fut.result()
                if success:
                    ok += 1
                else:
                    fail += 1
                    failed_files.append((label, err))
                    print(f"[FAILED] {label}: {err}", file=sys.stderr)

    print(f"Done. OK={ok}, FAILED={fail}")
    if failed_files:
        print(f"\nFailed file list ({len(failed_files)}):")
        for filename, error in failed_files:
            print(f"  - {filename}: {error}")

if __name__ == "__main__":
    main()
