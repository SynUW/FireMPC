#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-performance GDAL raster processing (reference-grid alignment + final clipping version).
========================================================================
Root cause: data is on NFS (~241 ms latency); all 16 threads are stuck in IO wait (wa 30-77%).

Optimization strategy -- a true three-stage async pipeline:
  Stage 1 (prefetch)  : thread pool, NFS -> /data (local SSD), IO-bound, many threads
  Stage 2 (process)   : process pool, /data -> /data, CPU-bound, bypasses GIL
  Stage 3 (writeback) : thread pool, /data -> NFS, IO-bound, many threads

The three stages are connected by two bounded queues and overlap fully:
  prefetch_q (bounded)  : holds prefetched-and-pending filenames
  writeback_q (bounded) : holds processed-and-pending (shm_out, nfs_out) pairs

Timing sketch (-> means concurrent):
  batch1 prefetch -> batch1 process + batch2 prefetch -> batch1 writeback + batch2 process + batch3 prefetch -> ...

Other optimizations:
  - Eliminate the redundant copy in preclean_to_temp_no_modify (the original code does not modify pixels)
  - ProcessPoolExecutor bypasses the GIL
  - Removed gc.collect() inside the loop
  - In-memory intermediate files use COMPRESS=NONE; final output compression is optional
========================================================================
"""

import os, sys, argparse, shutil, tempfile, traceback, time, threading, signal, atexit
from datetime import datetime
from pathlib import Path
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future
from queue import Queue, Empty
import multiprocessing

import numpy as np
from osgeo import gdal, osr

# /data local SSD (7 TB ext4); much faster than NFS, used for all intermediate data
# If /data is not writable, automatically fall back to /dev/shm
def _pick_scratch_base() -> Path:
    for candidate in [Path("/data"), Path("/dev/shm")]:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / f".write_probe_{os.getpid()}"
            probe.touch()
            probe.unlink()
            return candidate
        except (PermissionError, OSError):
            continue
    return Path(tempfile.gettempdir())

SCRATCH_BASE = _pick_scratch_base()
SCRATCH_DIR  = SCRATCH_BASE / f"pre_proc_{os.getpid()}"  # Per-process unique to avoid conflicts between multiple instances
INPUT_DIR    = SCRATCH_DIR / "input"
OUTPUT_DIR   = SCRATCH_DIR / "output"
TMP_DIR      = SCRATCH_DIR / "tmp"

# ---- Auto-clean the temp directory on Ctrl+C / exception exit ----
def _cleanup_scratch():
    if SCRATCH_DIR.exists():
        try:
            shutil.rmtree(SCRATCH_DIR, ignore_errors=True)
            print(f"\n[CLEANUP] temp directory deleted: {SCRATCH_DIR}", flush=True)
        except Exception:
            pass

atexit.register(_cleanup_scratch)

# Registered by the main flow during process_all; first SIGINT only cooperatively stops to avoid deleting files in scratch still held by the process pool
_ACTIVE_PIPELINE_STOP = None  # Optional[threading.Event]


def _signal_handler(sig, frame):
    global _ACTIVE_PIPELINE_STOP
    if _ACTIVE_PIPELINE_STOP is not None and not _ACTIVE_PIPELINE_STOP.is_set():
        print(
            f"\n[INT] received signal {sig}; stopping the pipeline (waiting for current tasks to wrap up; press again to force exit)...",
            flush=True,
        )
        _ACTIVE_PIPELINE_STOP.set()
        return
    print(f"\n[INT] received signal {sig}; force exit and clean up the temp directory...", flush=True)
    _cleanup_scratch()
    os._exit(128 + int(sig) if int(sig) < 128 else 1)

signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def _init_gdal():
    gdal.UseExceptions()
    gdal.SetConfigOption("GDAL_PAM_ENABLED", "NO")
    gdal.SetConfigOption("GTIFF_INTERNAL_MASK", "YES")
    gdal.SetConfigOption("CPL_MAX_ERROR_REPORTS", "100")
    gdal.SetConfigOption("GDAL_NUM_THREADS", "4")
    gdal.SetConfigOption("NUM_THREADS", "4")
    gdal.SetConfigOption("GDAL_CACHEMAX", "256")

_init_gdal()


# =================== Utility functions ===================

def gdal_dtype_numpy(gdt):
    _MAP = {
        gdal.GDT_Byte: np.uint8, gdal.GDT_UInt16: np.uint16,
        gdal.GDT_Int16: np.int16, gdal.GDT_UInt32: np.uint32,
        gdal.GDT_Int32: np.int32, gdal.GDT_Float32: np.float32,
        gdal.GDT_Float64: np.float64,
    }
    if gdt not in _MAP:
        raise RuntimeError(f"Unsupported GDAL data type: {gdt}")
    return _MAP[gdt]


def dtype_range_for_gdt(gdt):
    _RANGES = {
        gdal.GDT_Byte: (0, 255), gdal.GDT_UInt16: (0, 65535),
        gdal.GDT_Int16: (-32768, 32767), gdal.GDT_UInt32: (0, 4294967295),
        gdal.GDT_Int32: (-2147483648, 2147483647),
    }
    return _RANGES.get(gdt, (None, None))


def choose_storage_gdt(src_gdt, nd_values):
    nd_values = list(nd_values) if nd_values else []
    if any(isinstance(v, float) and not float(v).is_integer() for v in nd_values):
        return gdal.GDT_Float32 if src_gdt != gdal.GDT_Float64 else gdal.GDT_Float64
    if src_gdt in (gdal.GDT_Float32, gdal.GDT_Float64):
        return src_gdt
    vmin, vmax = dtype_range_for_gdt(src_gdt)
    ok = True
    for v in nd_values:
        vv = int(v) if (isinstance(v, (int, np.integer)) or float(v).is_integer()) else v
        if isinstance(vv, float):
            return gdal.GDT_Float32
        if vmin is not None and (vv < vmin or vv > vmax):
            ok = False
            break
    if ok:
        return src_gdt
    promote = {
        gdal.GDT_Byte: gdal.GDT_Int16, gdal.GDT_UInt16: gdal.GDT_Int32,
        gdal.GDT_Int16: gdal.GDT_Int32, gdal.GDT_UInt32: gdal.GDT_Float64,
        gdal.GDT_Int32: gdal.GDT_Float64,
    }
    return promote.get(src_gdt, src_gdt)


def open_ds(path, readonly=True):
    ds = gdal.Open(str(path), gdal.GA_ReadOnly if readonly else gdal.GA_Update)
    if ds is None:
        raise RuntimeError(f"Cannot open image: {path}")
    return ds


def get_srs_wkt(ds):
    wkt = ds.GetProjection()
    if not wkt:
        return ""
    sref = osr.SpatialReference()
    sref.ImportFromWkt(wkt)
    return sref.ExportToWkt()


def get_bounds_size_gt(ds):
    gt = ds.GetGeoTransform()
    w, h = ds.RasterXSize, ds.RasterYSize
    x0, x1 = gt[0], gt[0] + w * gt[1]
    y0, y1 = gt[3], gt[3] + h * gt[5]
    xmin, xmax = min(x0, x1), max(x0, x1)
    ymin, ymax = min(y0, y1), max(y0, y1)
    return (xmin, ymin, xmax, ymax), (w, h), gt


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def close_ds(ds):
    try:
        if ds:
            ds.FlushCache()
    except Exception:
        pass
    return None


def unlink_safe(p):
    try:
        if p and Path(p).exists():
            Path(p).unlink()
    except Exception:
        pass


def resample_alg_for_subfolder(folder_name: str) -> str:
    n = folder_name.lower()
    return "near" if ("12q1" in n or "dems" in n or "mcd14a1" in n) else "bilinear"


def read_as_array_compat(band, xoff, yoff, xsize, ysize):
    try:
        return band.ReadAsArray(xoff=xoff, yoff=yoff, win_xsize=xsize, win_ysize=ysize)
    except TypeError:
        try:
            return band.ReadAsArray(xoff, yoff, xsize, ysize)
        except TypeError:
            buf = band.ReadRaster(xoff, yoff, xsize, ysize,
                                  buf_xsize=xsize, buf_ysize=ysize,
                                  buf_type=band.DataType)
            if buf is None:
                return None
            return np.frombuffer(buf, dtype=gdal_dtype_numpy(band.DataType)).reshape(ysize, xsize)


def write_fail_logs(csv_path, txt_path, records):
    if not records:
        return
    import csv
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['time', 'folder', 'file', 'step', 'error', 'gdal_last_error'])
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    with open(txt_path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(f"[{r['time']}] {r['folder']}/{r['file']}\nStep: {r['step']}\nError: {r['error']}\n")
            if r.get('gdal_last_error'):
                f.write(f"GDAL: {r['gdal_last_error']}\n")
            if r.get('traceback'):
                f.write("Traceback:\n" + r['traceback'])
            f.write("\n" + "-" * 80 + "\n")


# =================== Metadata loading ===================

def _get_src_metadata(src_path):
    src = open_ds(src_path, readonly=True)
    nbands = src.RasterCount
    src_gdt = src.GetRasterBand(1).DataType
    nd_values = []
    for b in range(nbands):
        nd = src.GetRasterBand(b + 1).GetNoDataValue()
        if nd is not None:
            nd_values.append(nd)
    nd_values = list(set(nd_values))
    out_gdt = choose_storage_gdt(src_gdt, nd_values)
    sp_lc = str(src_path).lower()
    if 'mod11a1' in sp_lc or 'myd11a1' in sp_lc:
        out_gdt = gdal.GDT_UInt16
    canonical_nd = nd_values[0] if nd_values else -32768
    src = close_ds(src)
    return nbands, out_gdt, canonical_nd, nd_values


# =================== VRT construction ===================

def translate_subset_to_vrt_disk(tmp_dir, name, src_ds, band_indices_0based):
    if not band_indices_0based:
        return None, None
    out_vrt = Path(tmp_dir) / f"{name}.vrt"
    unlink_safe(out_vrt)
    band_list_1b = [i + 1 for i in band_indices_0based]
    try:
        topts = gdal.TranslateOptions(format='VRT', bandList=band_list_1b)
        vrt = gdal.Translate(str(out_vrt), src_ds, options=topts)
    except TypeError:
        args = ['-of', 'VRT']
        for b in band_list_1b:
            args += ['-b', str(b)]
        vrt = gdal.Translate(str(out_vrt), src_ds, options=args)
    if vrt is None:
        raise RuntimeError("gdal.Translate failed")
    return vrt, out_vrt


def align_to_grid_vrt(tmp_dir, name, src_ds, grid_wkt, grid_bounds, grid_size_wh,
                      resample_alg, canonical_nd=None):
    out_vrt = Path(tmp_dir) / f"{name}.vrt"
    unlink_safe(out_vrt)
    xmin, ymin, xmax, ymax = grid_bounds
    width, height = grid_size_wh
    dst_nd = canonical_nd
    init_dest_val = -32768 if dst_nd is None else dst_nd
    warp_opts = [f'INIT_DEST={init_dest_val}', 'UNIFIED_SRC_NODATA=YES']
    wopts = gdal.WarpOptions(
        format='VRT', dstSRS=grid_wkt,
        outputBounds=(xmin, ymin, xmax, ymax), width=width, height=height,
        resampleAlg=resample_alg,
        warpOptions=warp_opts,
        srcNodata=dst_nd, dstNodata=dst_nd,
    )
    vrt = gdal.Warp(str(out_vrt), src_ds, options=wopts)
    if vrt is None:
        raise RuntimeError("Warp failed (aligning to reference grid)")
    return vrt, out_vrt


def clip_with_shapefile_vrt(tmp_dir, name, src_ds, grid_wkt, grid_bounds, grid_size_wh,
                            clipline_path, resample_alg, canonical_nd=None):
    out_vrt = Path(tmp_dir) / f"{name}.vrt"
    unlink_safe(out_vrt)
    dst_nd = canonical_nd
    init_dest_val = -32768 if dst_nd is None else dst_nd
    xmin, ymin, xmax, ymax = grid_bounds
    width, height = grid_size_wh
    wopts = gdal.WarpOptions(
        format='VRT', dstSRS=grid_wkt,
        # Cutline mask only; do not crop the extent; the output grid is strictly locked to the template grid
        outputBounds=(xmin, ymin, xmax, ymax), width=width, height=height,
        resampleAlg=resample_alg,
        cutlineDSName=str(clipline_path) if clipline_path else None,
        cropToCutline=False,
        warpOptions=[f'INIT_DEST={init_dest_val}', 'UNIFIED_SRC_NODATA=YES'],
        srcNodata=dst_nd, dstNodata=dst_nd,
    )
    vrt = gdal.Warp(str(out_vrt), src_ds, options=wopts)
    if vrt is None:
        raise RuntimeError("Warp failed (clipping by shapefile)")
    return vrt, out_vrt


# =================== Final write ===================

def write_final_with_blocks(out_path, gdt, grid_gt, grid_wkt, nbands,
                            qa_vrt, nonlai_vrt, lai_pos_vrt, lai_neg_vrt, lai_nn_vrt,
                            lai_indices, qa_indices, nonlai_indices,
                            canonical_nd, compress='NONE', blocksize=512):
    w, h = grid_gt['w'], grid_gt['h']
    drv = gdal.GetDriverByName('GTiff')
    ensure_dir(Path(out_path).parent)
    dst = drv.Create(str(out_path), w, h, nbands, gdt,
                     options=['TILED=YES', f'BLOCKXSIZE={blocksize}', f'BLOCKYSIZE={blocksize}',
                              f'COMPRESS={compress}', 'BIGTIFF=IF_SAFER'])
    if dst is None:
        raise RuntimeError(f"Cannot create output: {out_path}")
    dst.SetProjection(grid_wkt)
    dst.SetGeoTransform(grid_gt['gt'])

    qa_set = set(qa_indices)
    lai_set = set(lai_indices)
    qa_pos = {i: k + 1 for k, i in enumerate(qa_indices)} if qa_indices else {}
    nonlai_pos = {i: k + 1 for k, i in enumerate(nonlai_indices)} if nonlai_indices else {}
    lai_pos_map = {i: k + 1 for k, i in enumerate(lai_indices)} if lai_indices else {}
    nd_out = canonical_nd if canonical_nd is not None else -32768

    for i in range(nbands):
        if i not in qa_set:
            dst.GetRasterBand(i + 1).SetNoDataValue(nd_out)

    bx = by = blocksize
    for y0 in range(0, h, by):
        nrows = min(by, h - y0)
        for x0 in range(0, w, bx):
            ncols = min(bx, w - x0)
            for i in range(nbands):
                if i in qa_set:
                    arr = read_as_array_compat(qa_vrt.GetRasterBand(qa_pos[i]), x0, y0, ncols, nrows)
                    dst.GetRasterBand(i + 1).WriteArray(arr, xoff=x0, yoff=y0)
                elif i in lai_set:
                    bpos = lai_pos_map[i]
                    arr_pos = read_as_array_compat(lai_pos_vrt.GetRasterBand(bpos), x0, y0, ncols, nrows) if lai_pos_vrt else None
                    arr_neg = read_as_array_compat(lai_neg_vrt.GetRasterBand(bpos), x0, y0, ncols, nrows) if lai_neg_vrt else None
                    arr_nn = read_as_array_compat(lai_nn_vrt.GetRasterBand(bpos), x0, y0, ncols, nrows) if lai_nn_vrt else None
                    if arr_pos is None and arr_neg is None and arr_nn is None:
                        out = np.full((nrows, ncols), nd_out, dtype=gdal_dtype_numpy(gdt))
                    else:
                        base = arr_pos if arr_pos is not None else (arr_neg if arr_neg is not None else arr_nn)
                        out = np.full((nrows, ncols), nd_out, dtype=base.dtype)
                        if arr_nn is not None:
                            valid_nn = (arr_nn != nd_out)
                            if arr_pos is not None:
                                choose = (arr_nn > 0) & valid_nn & (arr_pos != nd_out)
                                out[choose] = arr_pos[choose]
                            if arr_neg is not None:
                                choose = (arr_nn < 0) & valid_nn & (arr_neg != nd_out)
                                out[choose] = arr_neg[choose]
                            zero_mask = (arr_nn == 0) & valid_nn
                            if zero_mask.any():
                                if arr_pos is not None:
                                    m = zero_mask & (arr_pos != nd_out)
                                    out[m] = arr_pos[m]
                                if arr_neg is not None:
                                    m = zero_mask & (arr_neg != nd_out)
                                    out[m] = arr_neg[m]
                    dst.GetRasterBand(i + 1).WriteArray(out, xoff=x0, yoff=y0)
                else:
                    arr = read_as_array_compat(nonlai_vrt.GetRasterBand(nonlai_pos[i]), x0, y0, ncols, nrows)
                    dst.GetRasterBand(i + 1).WriteArray(arr, xoff=x0, yoff=y0)

    dst.FlushCache()
    dst = close_ds(dst)
    return True


# =================== Single-file processing (pure in-memory path) ===================

def process_one_in_memory(local_src_path, local_out_path, grid_template_path,
                          clip_shp_path, resample_alg_nonqa,
                          compress='NONE', blocksize=512):
    """All paths live on /dev/shm (except grid_template and the shapefile); pure in-memory operations."""
    _init_gdal()

    local_src_path = Path(local_src_path)
    local_out_path = Path(local_out_path)

    nbands, out_gdt, canonical_nd, nd_values = _get_src_metadata(local_src_path)

    qa_indices = []
    lai_indices = []
    all_idx = list(range(nbands))
    nonlai_indices = [i for i in all_idx if i not in set(qa_indices) and i not in set(lai_indices)]

    tmp_dir = tempfile.mkdtemp(prefix="tri_", dir=str(TMP_DIR))
    tmp_dir = Path(tmp_dir)

    try:
        src_ds = open_ds(local_src_path, readonly=True)

        nonlai_vrt_src, _ = (
            translate_subset_to_vrt_disk(tmp_dir, "nonlai_src", src_ds, nonlai_indices)
            if nonlai_indices else (None, None)
        )
        qa_vrt_src, _ = (
            translate_subset_to_vrt_disk(tmp_dir, "qa_src", src_ds, qa_indices)
            if qa_indices else (None, None)
        )

        grid_ds = open_ds(grid_template_path, readonly=True)
        grid_bounds, grid_size, grid_gt_tuple = get_bounds_size_gt(grid_ds)
        grid_wkt = get_srs_wkt(grid_ds)
        grid_gt_info = {'gt': grid_gt_tuple, 'w': grid_size[0], 'h': grid_size[1]}
        grid_ds = close_ds(grid_ds)

        alg = (resample_alg_nonqa or "bilinear").strip().lower()
        warp_alg = "near" if alg in ("nearest", "near") else "bilinear"
        eff_nd = canonical_nd

        qa_vrt_aligned = None
        if qa_vrt_src:
            qa_vrt_aligned, _ = align_to_grid_vrt(
                tmp_dir, "qa_aligned", qa_vrt_src, grid_wkt, grid_bounds, grid_size,
                resample_alg='near', canonical_nd=None)
            qa_vrt_src = close_ds(qa_vrt_src)

        nonlai_vrt_aligned = None
        if nonlai_vrt_src:
            nonlai_vrt_aligned, _ = align_to_grid_vrt(
                tmp_dir, "nonlai_aligned", nonlai_vrt_src, grid_wkt, grid_bounds, grid_size,
                resample_alg=warp_alg, canonical_nd=eff_nd)
            nonlai_vrt_src = close_ds(nonlai_vrt_src)

        lai_pos_vrt = lai_neg_vrt = lai_nn_vrt = None
        src_ds = close_ds(src_ds)

        clip_shp = Path(clip_shp_path) if clip_shp_path else None
        if clip_shp and clip_shp.exists():
            if qa_vrt_aligned:
                qa_clip, _ = clip_with_shapefile_vrt(
                    tmp_dir, "qa_clip", qa_vrt_aligned, grid_wkt, grid_bounds, grid_size,
                    clip_shp, 'near', canonical_nd=None)
                qa_vrt_aligned = close_ds(qa_vrt_aligned)
                qa_vrt_aligned = qa_clip

            if nonlai_vrt_aligned:
                nonlai_clip, _ = clip_with_shapefile_vrt(
                    tmp_dir, "nonlai_clip", nonlai_vrt_aligned, grid_wkt, grid_bounds, grid_size,
                    clip_shp, warp_alg, canonical_nd=eff_nd)
                nonlai_vrt_aligned = close_ds(nonlai_vrt_aligned)
                nonlai_vrt_aligned = nonlai_clip

        out_gdt_final = out_gdt
        sp_lc = str(local_src_path).lower()
        if 'mod11a1' in sp_lc or 'myd11a1' in sp_lc:
            out_gdt_final = gdal.GDT_UInt16

        write_final_with_blocks(
            local_out_path, out_gdt_final, grid_gt_info, grid_wkt, nbands,
            qa_vrt_aligned, nonlai_vrt_aligned, lai_pos_vrt, lai_neg_vrt, lai_nn_vrt,
            lai_indices, qa_indices, nonlai_indices,
            canonical_nd, compress=compress, blocksize=blocksize)

        close_ds(qa_vrt_aligned)
        close_ds(nonlai_vrt_aligned)
        gdal.ErrorReset()
        return 'ok'
    except Exception:
        if local_out_path.exists():
            try:
                local_out_path.unlink()
            except Exception:
                pass
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# =================== Worker functions ===================

def _process_worker(args):
    """Runs in an isolated child process. Reads the local copy on /dev/shm and writes back to /dev/shm."""
    (filename, local_src, local_out, grid_template_path,
     clip_shp_path, warp_alg, compress, blocksize) = args
    try:
        process_one_in_memory(
            local_src_path=local_src,
            local_out_path=local_out,
            grid_template_path=grid_template_path,
            clip_shp_path=clip_shp_path,
            resample_alg_nonqa=warp_alg,
            compress=compress,
            blocksize=blocksize,
        )
        return ('ok', filename, None, None, None, None)
    except Exception as e:
        tb = traceback.format_exc()
        gdal_msg = gdal.GetLastErrorMsg()
        msg = str(e)
        if 'Warp' in msg or 'warp' in msg:
            step = 'warp'
        elif 'VRT' in msg or 'Translate' in msg:
            step = 'vrt'
        elif 'ReadAsArray' in msg:
            step = 'io_read'
        else:
            step = 'unknown'
        return ('failed', filename, msg, tb, gdal_msg, step)


# =================== Stage 1: NFS -> /dev/shm prefetch ===================

def _prefetch_one(nfs_src, shm_dst):
    """Copy a single file from NFS to /dev/shm. Uses a large buffer for speed."""
    shm_dst = Path(shm_dst)
    if shm_dst.exists():
        return
    shm_dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = shm_dst.with_suffix('.tmp')
    try:
        with open(str(nfs_src), 'rb') as fin, open(str(tmp), 'wb') as fout:
            while True:
                chunk = fin.read(4 * 1024 * 1024)
                if not chunk:
                    break
                fout.write(chunk)
        tmp.rename(shm_dst)
    except Exception:
        unlink_safe(tmp)
        raise


def _writeback_one(shm_src, nfs_dst):
    """Copy a single file from /dev/shm back to NFS."""
    nfs_dst = Path(nfs_dst)
    nfs_dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = nfs_dst.with_suffix('.tmp_wb')
    try:
        with open(str(shm_src), 'rb') as fin, open(str(tmp), 'wb') as fout:
            while True:
                chunk = fin.read(4 * 1024 * 1024)
                if not chunk:
                    break
                fout.write(chunk)
        tmp.rename(nfs_dst)
    except Exception:
        unlink_safe(tmp)
        raise
    finally:
        unlink_safe(shm_src)


# =================== CLI helpers ===================

def _normalize_include_folders(arg_list):
    # If argparse is mistakenly given a str as default, iterating that str will split it into hundreds of fake tokens by character
    if isinstance(arg_list, str):
        arg_list = [arg_list]
    if not arg_list:
        return None
    toks = []
    for item in arg_list:
        if item is None:
            continue
        s = str(item).strip()
        if not s:
            continue
        if ',' in s:
            toks.extend([t.strip() for t in s.split(',') if t.strip()])
        else:
            toks.append(s)
    return set(toks) if toks else None


def _make_tif_name_filter(tokens):
    if not tokens:
        return None
    exact = set()
    stems = set()
    for raw in tokens:
        s = str(raw).strip().lower()
        if not s:
            continue
        exact.add(s)
        if s.endswith(".tif") or s.endswith(".tiff"):
            stems.add(Path(s).stem.lower())
        else:
            stems.add(s)
    return exact, stems


def _tif_matches_name_filter(name, flt):
    if flt is None:
        return True
    exact, stems = flt
    low = name.lower()
    if low in exact:
        return True
    return Path(name).stem.lower() in stems


def _as_input_roots(input_roots):
    """Normalize into a list of resolved Path objects."""
    if isinstance(input_roots, (str, Path)):
        return [Path(input_roots).resolve()]
    return [Path(p).resolve() for p in input_roots]


def _root_output_tags(roots):
    """Used for output/scratch subpaths when there are multiple roots, to avoid same-name root directory conflicts."""
    names = [r.name for r in roots]
    tags = []
    for i, r in enumerate(roots):
        if names.count(r.name) > 1:
            tags.append(f"{r.name}_{i}")
        else:
            tags.append(r.name)
    return tags


def _warp_alg_for_source_rel(rel: Path, source_root: Path) -> str:
    """For files directly under the root, decide the algorithm by root directory name; for files in subdirectories, decide by relative path."""
    if len(rel.parts) == 1:
        return resample_alg_for_subfolder(source_root.name)
    return resample_alg_for_subfolder(str(rel))


def _oswalk_geo_tifs_under(root: Path):
    """
    Enumerate GeoTIFFs with os.walk, avoiding the overhead of pathlib.rglob + sorted on large directories.
    Yields (absolute path str, Path relative to root).
    """
    root_str = os.path.normpath(str(root))
    for dirpath, _dirnames, filenames in os.walk(root_str, topdown=True, followlinks=False):
        for fname in filenames:
            if ".__tmp__" in fname:
                continue
            lower = fname.lower()
            if not (lower.endswith(".tif") or lower.endswith(".tiff")):
                continue
            full = os.path.join(dirpath, fname)
            rel = os.path.relpath(full, root_str)
            yield full, Path(rel)


def _oswalk_existing_geo_rel_keys(output_root: Path):
    """Relative-path keys (lowercase posix) of existing .tif/.tiff files in the output tree, used to skip already-generated files."""
    out = set()
    root_str = os.path.normpath(str(output_root))
    if not os.path.isdir(root_str):
        return out
    for dirpath, _dirnames, filenames in os.walk(root_str, topdown=True, followlinks=False):
        for fname in filenames:
            lower = fname.lower()
            if not (lower.endswith(".tif") or lower.endswith(".tiff")):
                continue
            full = os.path.join(dirpath, fname)
            rel = os.path.relpath(full, root_str)
            out.add(Path(rel).as_posix().lower())
    return out


# =================== Main pipeline (true async three-stage) ===================

def process_all(input_roots, output_root, grid_template_path, shapefile_path,
                threads=16, compress='LZW', blocksize=512,
                force=False, resume_from=0, match_names=None,
                prefetch_threads=32, writeback_threads=16,
                prefetch_ahead=200, mem_compress='NONE'):
    """
    Three-stage fully-async pipeline:
      - prefetch thread pool: continuously pulls files from NFS to /dev/shm, prefetching prefetch_ahead ahead.
      - process process pool: processes files from /dev/shm and pushes completed files into the writeback queue.
      - writeback thread pool: continuously writes finished files from /dev/shm back to NFS.

    The three stages run concurrently and do not block each other.

    input_roots: one or more input root directories; every root is recursively processed for .tif/.tiff.
    match_names: optional, only process GeoTIFFs whose filename / basename matches (see --match-names).
    """
    input_roots = _as_input_roots(input_roots)
    output_root = Path(output_root).resolve()
    grid_template_path = Path(grid_template_path).resolve()

    if not grid_template_path.exists():
        raise FileNotFoundError(f"Grid template does not exist: {grid_template_path}")
    for ir in input_roots:
        if not ir.is_dir():
            print(f"[ERROR] input root must be a directory: {ir}")
            return

    name_filter = _make_tif_name_filter(match_names) if match_names else None
    if match_names:
        print(f"[INFO] --match-names: process only name-matched GeoTIFFs ({len(match_names)} items)")
    else:
        print(f"[INFO] will recursively process all GeoTIFFs under these input roots: ")
        for ir in input_roots:
            print(f"       {ir}")

    # Initialize the /dev/shm directory
    for d in [INPUT_DIR, OUTPUT_DIR, TMP_DIR]:
        ensure_dir(d)

    # Pre-copy grid-template and shapefile (small files, copy to RAM disk once)
    shm_grid = INPUT_DIR / "_grid_template.tif"
    if not shm_grid.exists():
        print(f"[INFO] pre-copying grid-template to RAM disk...")
        _prefetch_one(grid_template_path, shm_grid)

    shm_shp_dir = INPUT_DIR / "shp"
    clip_shp_str = None
    if shapefile_path and not Path(shapefile_path).exists():
        print(f"[WARN] shapefile does not exist; aligning with grid-template only (no vector clipping): {shapefile_path}")
    if shapefile_path and Path(shapefile_path).exists():
        ensure_dir(shm_shp_dir)
        shp_base = Path(shapefile_path)
        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg', '.sbn', '.sbx', '.qix']:
            src_f = shp_base.with_suffix(ext)
            if src_f.exists():
                dst_f = shm_shp_dir / src_f.name
                if not dst_f.exists():
                    _prefetch_one(src_f, dst_f)
        clip_shp_str = str(shm_shp_dir / shp_base.name)

    tags = _root_output_tags(input_roots) if len(input_roots) > 1 else None
    batch_label = ",".join(ir.name for ir in input_roots)

    print("[INFO] Enumerating GeoTIFFs (os.walk, no pixel reads; --force skips the output-tree scan)...")
    # (scratch_rel, nfs_src, nfs_out, warp_alg); scratch_rel is unique under INPUT_DIR/OUTPUT_DIR
    all_tasks = []
    skipped_count = 0
    ensure_dir(output_root)

    existing = set()
    if (not force) and output_root.exists():
        t0 = time.time()
        existing = _oswalk_existing_geo_rel_keys(output_root)
        print(f"[INFO] indexed existing GeoTIFFs in the output directory: {len(existing)} entries ({time.time() - t0:.1f}s）")

    t_enum = time.time()
    try:
        for ri, input_root in enumerate(input_roots):
            for full, rel in _oswalk_geo_tifs_under(input_root):
                name = rel.name
                if not _tif_matches_name_filter(name, name_filter):
                    continue
                if len(input_roots) == 1:
                    out_rel = rel
                else:
                    out_rel = Path(tags[ri]) / rel
                rel_key = out_rel.as_posix().lower()
                if (rel_key in existing) and (not force):
                    skipped_count += 1
                    continue
                wa = _warp_alg_for_source_rel(rel, input_root)
                all_tasks.append(
                    (out_rel, full, str(output_root / out_rel), wa))
    except FileNotFoundError as e:
        print(f"[ERROR] input path is unavailable: {e}")
        return
    print(f"[INFO] input enumeration complete: {len(all_tasks)} pending tasks ({time.time() - t_enum:.1f}s）")

    if skipped_count > 0:
        print(f"[SKIP] {skipped_count} files already exist (use --force to overwrite)")
    if not all_tasks:
        print("[INFO] No files to process.")
        return

    if resume_from > 0:
        print("[WARN] --resume-from cuts in os.walk traversal order, which differs from the old version's path-sorted order.", flush=True)
        if resume_from >= len(all_tasks):
            print(f"[WARN] --resume-from {resume_from} exceeds total task count {len(all_tasks)}")
            return
        all_tasks = all_tasks[resume_from:]

    total = len(all_tasks)
    # Explicit one-time assignment to avoid past UnboundLocalError caused by ordering bugs like "compute peak first, then assign wb_max"
    n_workers = max(1, min(int(threads), total))
    wb_max = max(n_workers * 2, min(int(prefetch_ahead), 50))

    # Estimate peak scratch usage: ~5 MB per input file, ~50 MB per output file (10x inflation when uncompressed)
    peak_gb = (float(prefetch_ahead) * 5.0 + float(wb_max) * 50.0) / 1024.0
    print(f"[INFO] Pending: {total} | processing processes: {n_workers} | prefetch threads: {prefetch_threads} | writeback threads: {writeback_threads}")
    print(f"[INFO] prefetch lookahead: {prefetch_ahead} | writeback queue cap: {wb_max} | in-memory compression: {mem_compress} | output compression: {compress}")
    print(f"[INFO] temp directory: {SCRATCH_DIR} (estimated peak ~{peak_gb:.1f} GB）")
    print("=" * 80)

    logdir = output_root / "_logs"
    logdir.mkdir(parents=True, exist_ok=True)

    stats = {'ok': 0, 'skipped': skipped_count, 'failed': 0}
    fail_records = []
    stats_lock = threading.Lock()
    t_start = time.time()

    # ---- Semaphore: true backpressure control ----
    # scratch_slots caps the number of output files currently on scratch.
    # Each completed file (output landed on scratch) consumes a slot,
    # released only after writeback to NFS finishes. This keeps the count of output files on scratch <= wb_max.
    # wb_max output files * ~50 MB = peak output footprint.
    scratch_slots = threading.Semaphore(wb_max)

    # prefetch_done_q: prefetch done -> consumed by process (bounded, controls prefetch backlog)
    prefetch_done_q = Queue(maxsize=prefetch_ahead)
    pipeline_stop = threading.Event()

    done_total = [0]

    def _print_progress():
        with stats_lock:
            d = done_total[0]
        elapsed = time.time() - t_start
        rate = d / elapsed if elapsed > 0 else 0
        eta = (total - d) / rate if rate > 0 else 0
        sys.stdout.write(
            f"\r[Progress] {d}/{total}  OK:{stats['ok']}  FAIL:{stats['failed']}  "
            f"{rate:.1f} files/s  ETA:{eta:.0f}s   ")
        sys.stdout.flush()

    # ================================================================
    # Stage 1: prefetch threads
    # Concurrently copy from NFS to scratch, then push into prefetch_done_q (bounded throttling).
    # ================================================================
    def _prefetch_stage():
        with ThreadPoolExecutor(max_workers=prefetch_threads) as pf_pool:
            pending: dict[Future, tuple] = {}
            task_iter = iter(all_tasks)
            in_flight = 0
            exhausted = False

            while True:
                # Top up submissions until prefetch_ahead is reached
                while not exhausted and in_flight < prefetch_ahead:
                    if pipeline_stop.is_set():
                        exhausted = True
                        break
                    try:
                        scratch_rel, nfs_src, nfs_out, wa = next(task_iter)
                    except StopIteration:
                        exhausted = True
                        break
                    fut = pf_pool.submit(_prefetch_one, nfs_src, INPUT_DIR / scratch_rel)
                    pending[fut] = (scratch_rel, nfs_out, wa)
                    in_flight += 1

                if not pending:
                    break

                # Wait for any one to complete
                done_futs, _ = concurrent.futures.wait(
                    list(pending.keys()), timeout=0.1,
                    return_when=concurrent.futures.FIRST_COMPLETED)
                if not done_futs:
                    continue

                for f in done_futs:
                    scratch_rel, nfs_out, wa = pending.pop(f)
                    in_flight -= 1
                    try:
                        f.result()
                        prefetch_done_q.put((scratch_rel, nfs_out, wa))  # Blocking = backpressure
                    except Exception as e:
                        with stats_lock:
                            stats['failed'] += 1
                            done_total[0] += 1
                        fail_records.append({
                            'time': datetime.now().isoformat(timespec='seconds'),
                            'folder': batch_label, 'file': scratch_rel.as_posix(),
                            'step': 'prefetch', 'error': str(e),
                            'gdal_last_error': '', 'traceback': traceback.format_exc(),
                        })
                        _print_progress()

        prefetch_done_q.put(None)  # Sentinel

    # ================================================================
    # Stage 2: process pool
    # Take a prefetched file -> acquire a scratch_slot (blocks = backpressure) -> submit processing.
    # Async writeback after processing; the scratch_slot is released only after writeback finishes.
    # ================================================================
    def _process_stage():
        wb_pool = ThreadPoolExecutor(max_workers=writeback_threads)
        try:

            def _on_writeback_done(fut: Future, scratch_out: str, filename: str):
                """Writeback-completion callback: releases the scratch slot and handles exceptions."""
                try:
                    fut.result()
                except Exception as e:
                    sys.stdout.write(f"\n[WARN] writeback failed {filename}: {e}")
                    sys.stdout.flush()
                finally:
                    unlink_safe(scratch_out)
                    scratch_slots.release()

            with ProcessPoolExecutor(max_workers=n_workers) as proc_pool:
                proc_futs: dict[Future, tuple] = {}

                def _drain_proc():
                    for f in [ff for ff in list(proc_futs) if ff.done()]:
                        scratch_rel, nfs_out, wa = proc_futs.pop(f)
                        task_key = scratch_rel.as_posix()
                        try:
                            ret = f.result()
                        except Exception as e:
                            ret = ('failed', task_key, str(e),
                                   traceback.format_exc(), '', 'process_crash')
                        with stats_lock:
                            done_total[0] += 1
                        scratch_out = str(OUTPUT_DIR / scratch_rel)
                        if ret[0] == 'ok':
                            with stats_lock:
                                stats['ok'] += 1
                            wb_fut = wb_pool.submit(_writeback_one, scratch_out, nfs_out)
                            wb_fut.add_done_callback(
                                lambda fut, s=scratch_out, fn=task_key:
                                    _on_writeback_done(fut, s, fn))
                        else:
                            _, fn, msg, tb, gdal_msg, step = ret
                            with stats_lock:
                                stats['failed'] += 1
                            sys.stdout.write(f"\n[FAIL] {fn} | step={step} | err={msg[:200]}")
                            sys.stdout.flush()
                            fail_records.append({
                                'time': datetime.now().isoformat(timespec='seconds'),
                                'folder': batch_label, 'file': fn, 'step': step,
                                'error': msg, 'gdal_last_error': gdal_msg, 'traceback': tb,
                            })
                            unlink_safe(scratch_out)
                            scratch_slots.release()
                        unlink_safe(INPUT_DIR / scratch_rel)
                        _print_progress()

                while True:
                    _drain_proc()
                    # When the process pool has spare capacity, take a task from the prefetch queue
                    while len(proc_futs) < n_workers * 2:
                        try:
                            pack = prefetch_done_q.get(timeout=0.05)
                        except Empty:
                            break
                        if pack is None:
                            # Wait for all process tasks to finish
                            while proc_futs:
                                _drain_proc()
                                time.sleep(0.02)
                            return

                        scratch_rel, nfs_out, wa = pack

                        # Acquire slot: blocks if scratch is at output capacity (core backpressure)
                        scratch_slots.acquire()
                        scratch_in = str(INPUT_DIR / scratch_rel)
                        scratch_out = str(OUTPUT_DIR / scratch_rel)
                        task_args = (scratch_rel.as_posix(), scratch_in, scratch_out,
                                     str(shm_grid), clip_shp_str,
                                     wa, mem_compress, blocksize)
                        fut = proc_pool.submit(_process_worker, task_args)
                        proc_futs[fut] = (scratch_rel, nfs_out, wa)

                    time.sleep(0.01)
        finally:
            wb_pool.shutdown(wait=True)

    # ================================================================
    # Start the two stages (writeback is embedded inside _process_stage's callback)
    # ================================================================
    t1 = threading.Thread(target=_prefetch_stage, name="prefetch", daemon=True)
    t2 = threading.Thread(target=_process_stage,  name="process",  daemon=True)

    global _ACTIVE_PIPELINE_STOP
    _ACTIVE_PIPELINE_STOP = pipeline_stop
    try:
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    finally:
        _ACTIVE_PIPELINE_STOP = None

    total_sec = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"[DONE] OK={stats['ok']}  SKIPPED={stats['skipped']}  FAILED={stats['failed']}  total elapsed: {total_sec:.0f}s")
    if total_sec > 0 and stats['ok'] > 0:
        print(f"[PERF] average {stats['ok']/total_sec:.2f} files/s")
    print(f"[OUT ] {output_root}")

    if fail_records:
        csv_path = logdir / "tri_pipeline_failures.csv"
        txt_path = logdir / "tri_pipeline_failures.log"
        write_fail_logs(csv_path, txt_path, fail_records)
        print(f"[LOG ] failure details: {csv_path}")

    _cleanup_scratch()


_FOLDER_MODE_OUTPUT_ROOT = Path(
    "/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/"
    "drivers/processed/spatial_filled_5xdownsampled_merged_unifiedEntireCan")


def main():
    ap = argparse.ArgumentParser(
        description="High-performance GDAL raster processing (three-stage pipeline: NFS prefetch -> in-memory processing -> async writeback)")
    ap.add_argument(
        "--root-folder",
        # default='/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled_5xdownsampled_merged',
        default=None,
        type=Path,
        metavar="DIR",
        help="Data root directory: contains multiple subdirectories, each with GeoTIFFs;"
             "recursively processed and relative paths preserved; output is fixed-written into "
             f"{_FOLDER_MODE_OUTPUT_ROOT}。"
             "If not specified, both --input-root and --output-root must be given.",
    )
    ap.add_argument(
        "--input-root",
        default='/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/worldpop_v2_clipped/access_idw_norm',
        type=Path,
        help="Used together with --output-root: a single input root directory (recursively process all GeoTIFFs beneath it)."
             "Required when --root-folder is not used.",
    )
    ap.add_argument(
        "--output-root",
        default='/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled_5xdownsampled_merged_unifiedEntireCan/access_idw_norm',
        type=Path,
        help="Used together with --input-root: the output root directory. Required when --root-folder is not used.",
    )
    ap.add_argument("--grid-template",
                    default="/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled_5xdownsampled_merged/MCD14A1_combined_5xMaximumDown_merged/2025_12_31.tif",
                    type=Path)
    ap.add_argument("--shapefile",
                    # default="/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/Ecozones/research_area_shpfile_merge.shp",
                    default=None,
                    type=Path,
                    help="Optional: shp used for clipping / masking. If not provided, only --grid-template is used as the single source of truth for grid alignment (no vector clipping).")
    ap.add_argument("--threads", type=int, default=16,
                    help="Number of processing processes (CPU-bound)")
    ap.add_argument("--prefetch-threads", type=int, default=32,
                    help="NFS prefetch concurrency thread count (IO-bound, can be high; 32-64 is recommended)")
    ap.add_argument("--writeback-threads", type=int, default=16,
                    help="NFS writeback concurrency thread count (16-32 is recommended)")
    ap.add_argument("--prefetch-ahead", type=int, default=50,
                    help="Maximum number of files prefetched into the RAM disk. Note: an uncompressed output file is ~50 MB,"
                         "so peak usage = (prefetch_ahead + n_workers * 2) * 55 MB, default 50 ~= 3 GB")
    ap.add_argument("--compress", type=str, default='LZW',
                    help="Compression for the final output file (NONE is fastest, LZW saves space)")
    ap.add_argument("--mem-compress", type=str, default='NONE',
                    help="Compression for in-memory intermediate files (strongly recommend NONE)")
    ap.add_argument("--blocksize", type=int, default=512)
    ap.add_argument("--force", action='store_true')
    ap.add_argument("--resume-from", type=int, default=0)
    ap.add_argument(
        "--match-names",
        nargs="*",
        default=None,
        metavar="NAME",
        help="Optional: only process GeoTIFFs whose filename or basename matches (space-separated for multiple)."
             "No restriction by default.",
    )
    args = ap.parse_args()

    if args.root_folder is not None:
        if args.input_root is not None or args.output_root is not None:
            print("[WARN] --root-folder was specified; ignoring --input-root / --output-root.", flush=True)
        input_roots = [args.root_folder.resolve()]
        output_root = _FOLDER_MODE_OUTPUT_ROOT
        print(f"[INFO] --root-folder={input_roots[0]}  output root={output_root}", flush=True)
    else:
        if args.input_root is None or args.output_root is None:
            ap.error(
                "Please specify --root-folder (a data root with multiple subdirectories),"
                "or explicitly specify both --input-root and --output-root."
            )
        input_roots = [args.input_root.resolve()]
        output_root = args.output_root.resolve()

    process_all(
        input_roots=input_roots,
        output_root=output_root,
        grid_template_path=args.grid_template,
        shapefile_path=args.shapefile,
        threads=args.threads,
        compress=args.compress,
        blocksize=args.blocksize,
        force=args.force,
        resume_from=args.resume_from,
        match_names=_normalize_include_folders(args.match_names),
        prefetch_threads=args.prefetch_threads,
        writeback_threads=args.writeback_threads,
        prefetch_ahead=args.prefetch_ahead,
        mem_compress=args.mem_compress,
    )


if __name__ == "__main__":
    main()
