# -*- coding: utf-8 -*-
"""
Generic script: scale + min-max normalize (without clipping to [0,1]) specified bands and unify NoData=-32768.
- Input: a folder containing TIF files (single directory, not recursive)
- Output: processed TIF files saved to a new folder
- Features:
    * Scale (multiply/add) and min-max normalize ((x - min)/(max-min)) for the specified bands
    * Skip any numerical transform on NoData pixels; uniformly write NoData as -32768 and set nodata=-32768
    * QA bands are detected via band-description keywords and skipped for scaling/normalization (NoData is still unified)
    * Before writing the output TIF, check whether pixel values lie within the configured min-max range; out-of-range pixels are set to NoData
- Order: scale & offset first, then min/max normalize (without clipping to [0,1]), finally range check and NoData setting
- dtype:
    * Output is uniformly float32

Additional:
- --band_config can be either an inline config string or a file path.
- Range check: for bands with configured min/max, pixels outside [vmin, vmax] can either be set to NoData or clipped to the range boundary.

Rule conventions (rules-excel dependency removed):
- NoData pixel set = file's native nodata UNION (--extra-nodata list)
  - Default --extra-nodata is [-32768, 32768]
  - If input path contains "mcd12q1" (case-insensitive), 0 is additionally treated as invalid (MCD12Q1's invalid value is 0)
- If input path contains "mcd14a1" (case-insensitive), 255 is additionally treated as invalid (MCD14A1's invalid value is 255), and **the image's native nodata metadata is ignored**
- If input path contains "mcd15a3h" (case-insensitive):
  - Map value==249 to 0, and these positions are not treated as NoData
  - Original value==0 positions always remain 0 (not subject to normalization/clipping/nodata setting)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import shutil
import uuid
import atexit
import signal
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

# ================== Configuration ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_QA_KEYWORDS = ['qa', 'cloud', 'quality', 'mask']  # Case-insensitive
UNIFIED_NODATA = -32768
BLOCK = 1024  # Window size (used for block-wise read/write to reduce memory usage)

_STAGING_DIRS: List[Path] = []

def _cleanup_staging_dirs() -> None:
    global _STAGING_DIRS
    for d in list(_STAGING_DIRS):
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass
    _STAGING_DIRS = []

def _install_cleanup_handlers() -> None:
    if getattr(_install_cleanup_handlers, "_installed", False):
        return
    setattr(_install_cleanup_handlers, "_installed", True)

    atexit.register(_cleanup_staging_dirs)

    def _handler(signum, frame):
        _cleanup_staging_dirs()
        raise KeyboardInterrupt()

    try:
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
    except Exception:
        pass


# ================== QA band detection ==================
def detect_qa_bands(band_names: List[str], qa_keywords: Optional[List[str]] = None) -> List[int]:
    if qa_keywords is None:
        qa_keywords = DEFAULT_QA_KEYWORDS
    qa_keys = [k.lower() for k in qa_keywords]
    qa_indices = []
    for i, name in enumerate(band_names):
        nm = (name or f'band_{i+1}').lower()
        if any(k in nm for k in qa_keys):
            qa_indices.append(i)
            logger.info(f"Detected QA band: {name or f'Band_{i+1}'} (index: {i})")
    return qa_indices


# ================== Read/parse band_config (supports file path + inline string) ==================
def _load_band_config_text(arg_value: str) -> str:
    """
    Return config text:
    - If arg_value points to an existing file, read its contents (allowing newline / space / comma as mixed separators)
    - Otherwise return the original string as-is
    """
    # Note: arg_value may be either a "file path" or a very long inline config string.
    # For overly long strings, Path.exists()/stat() may directly raise OSError: [Errno 36] File name too long.
    # In this case fall back to treating it as an "inline string" rather than aborting with an error.
    try:
        p = Path(arg_value)
        if p.exists() and p.is_file():
            return p.read_text(encoding='utf-8')
    except OSError:
        return arg_value
    return arg_value

def _normalize_config_text(text: str) -> str:
    """
    Normalize config text into a single comma-separated line.
    Supports newlines, Chinese commas/semicolons, etc.
    """
    if not text:
        return ""
    # Unify all separators to whitespace
    for sep in [',', '，', ';', '；', '、', '\n', '\r', '\t']:
        text = text.replace(sep, ' ')
    # Collapse multiple spaces into one, then join with commas
    parts = [t for t in text.split(' ') if t.strip()]
    return ','.join(parts)

def parse_band_config(config_str_or_path: str) -> Dict[int, Tuple[float, float, Optional[float], Optional[float]]]:
    """
    Config format (inline or file):
        "band_index:scale_factor:offset:min:max,band_index:scale_factor:offset:min:max,..."
    Example:
        "0:0.02:0:240:320,1:0.02:0:240:320,2:0.002:0:nan:nan,3:0.002:0:nan:nan"

    Returns:
        mapping: { band_idx(0-based): (scale, offset, min, max) }
                 where min/max == None means no normalization is performed
    """
    raw_text = _load_band_config_text(config_str_or_path)
    normalized = _normalize_config_text(raw_text)
    mapping: Dict[int, Tuple[float, float, Optional[float], Optional[float]]] = {}
    if not normalized.strip():
        return mapping

    for cfg in normalized.split(','):
        parts = cfg.strip().split(':')
        if len(parts) != 5:
            raise ValueError(f"Invalid config: {cfg} (5 segments required: band:scale:offset:min:max)")
        try:
            bidx = int(parts[0])
            # Compatible with the legacy format band:scale:offset:min:max, but by convention scale/offset are not used by default
            # (Still parse all 5 segments to avoid breaking old configs)
            scale = 1.0
            offset = 0.0

            def _to_opt_float(x):
                s = str(x).strip().lower()
                return None if s in ('nan', 'none', '') else float(x)

            vmin = _to_opt_float(parts[3])
            vmax = _to_opt_float(parts[4])
            if vmin is not None and vmax is not None and vmax <= vmin:
                raise ValueError(f"Normalization range must satisfy max > min: {cfg}")
            mapping[bidx] = (scale, offset, vmin, vmax)
        except Exception as e:
            raise ValueError(f"Invalid config: {cfg}; error: {e}")

    return mapping


# ================== NoData mask ==================
def build_nodata_mask(arr: np.ndarray, nd_values: List[float]) -> np.ndarray:
    """
    Returns True for NoData pixels.
    Supports multiple NoData values; for float arrays NaN is also handled.
    """
    if not nd_values:
        # If no list is provided, no value is treated as NoData (fall back to src.nodata)
        return np.zeros(arr.shape, dtype=bool)
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        has_nan = any(isinstance(v, float) and np.isnan(v) for v in nd_values)
        vals = [float(v) for v in nd_values if not (isinstance(v, float) and np.isnan(v))]
        if vals:
            mask = np.isin(arr, np.asarray(vals, dtype=np.float64))
        else:
            mask = np.zeros(arr.shape, dtype=bool)
        if has_nan:
            mask |= np.isnan(arr)
        return mask
    # integer
    vals_i = [int(v) for v in nd_values if not (isinstance(v, float) and np.isnan(v))]
    if not vals_i:
        return np.zeros(arr.shape, dtype=bool)
    return np.isin(arr, np.asarray(vals_i, dtype=arr.dtype))


# ================== dtype decision ==================
def _dtype_can_hold_unified_nodata(np_dtype_str: str) -> bool:
    try:
        info = np.iinfo(np_dtype_str)
        return (info.min <= UNIFIED_NODATA <= info.max)
    except ValueError:
        # All float types can represent -32768
        return True

def decide_output_dtype(src_dtype: str, band_ops: Dict[int, Tuple[float, float, Optional[float], Optional[float]]]) -> str:
    """
    Rules:
      - If any band needs normalization or fractional scale/offset -> float32
      - Otherwise keep the original dtype if possible; if it cannot represent -32768, upgrade to int16
    """
    # Per project convention: output is uniformly float32
    return "float32"
    # Any band requiring float-point computation?
    for _, (sf, off, vmin, vmax) in band_ops.items():
        if (vmin is not None and vmax is not None) or (abs(sf - round(sf)) > 1e-12) or (abs(off - round(off)) > 1e-12):
            return 'float32'
    # Try to keep the original dtype
    if _dtype_can_hold_unified_nodata(src_dtype):
        return src_dtype
    # Upgrade to int16 (can represent -32768)
    return 'int16'

def _safe_copy_then_replace(src_file: Path, dst_file: Path) -> None:
    """
    Cross-filesystem moves (e.g. /seal -> NFS) cannot use os.replace, so we:
      1) Copy to a same-directory temp file dst_file + '.tmp'
      2) os.replace the temp file to the final path (atomic replacement)
    """
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_file.with_suffix(dst_file.suffix + ".tmp")
    try:
        if tmp.exists():
            tmp.unlink()
    except Exception:
        pass
    with open(src_file, "rb") as fsrc, open(tmp, "wb") as fdst:
        shutil.copyfileobj(fsrc, fdst, length=16 * 1024 * 1024)
        try:
            fdst.flush()
            os.fsync(fdst.fileno())
        except Exception:
            pass
    os.replace(tmp, dst_file)


# ================== Single-file processing (block-wise) ==================
def process_tif_file(
    input_path: Path,
    output_path: Path,
    band_ops: Dict[int, Tuple[float, float, Optional[float], Optional[float]]],
    qa_keywords: Optional[List[str]],
    extra_nd_values: List[float],
    block_size: int = BLOCK,
    is_mcd15_context: bool = False,
    is_mcd12q1_context: bool = False,
    is_mcd14a1_context: bool = False,
    out_of_range_action: str = "nodata",
    local_stage_root: Optional[Path] = Path("/seal"),
) -> bool:
    """
    band_ops: {band_idx: (scale, offset, vmin, vmax)} (0-based)
    extra_nd_values: extra NoData value list (UNION with file's native nodata)
    is_mcd15_context: if in MCD15A3H context, perform 249->0 (without treating as NoData) and keep original 0 values as 0
    is_mcd12q1_context: if in MCD12Q1 context, additionally treat 0 as invalid
    is_mcd14a1_context: if in MCD14A1 context, additionally treat 255 as invalid and ignore the image's native nodata metadata
    out_of_range_action: how to handle out-of-range pixels: "nodata" sets to NoData, "clip" clips to range boundary.
    """
    try:
        _install_cleanup_handlers()
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(input_path) as src:
                src_profile = src.profile.copy()
                band_names = [src.descriptions[i] if src.descriptions[i] else f'Band_{i+1}' for i in range(src.count)]
                qa_indices = detect_qa_bands(band_names, qa_keywords)

                # Decide output dtype
                out_dtype = decide_output_dtype(src_profile['dtype'], band_ops)
                out_is_float = str(out_dtype).startswith('float')
                out_is_int = str(out_dtype).startswith('int')
                out_np_dtype = np.dtype(out_dtype)

                # Uniformly set nodata metadata to -32768
                profile = src_profile.copy()
                profile.update(dtype=out_dtype, nodata=UNIFIED_NODATA)
                # Keep original compress/tiling creation options: usually faster and more space-efficient; don't pop them without reason
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # -- Stage to local disk then transfer: write under /seal/<random_dir>, then copy to destination and atomically replace --
                # Note: each worker process generates its own random staging directory to avoid contention.
                staging_dir: Optional[Path] = None
                staged_output_path: Path = output_path
                if local_stage_root is not None:
                    try:
                        local_stage_root = Path(local_stage_root)
                        if local_stage_root.exists() and local_stage_root.is_dir():
                            staging_dir = local_stage_root / f"scale_tif_stage_{os.getpid()}_{uuid.uuid4().hex}"
                            staging_dir.mkdir(parents=True, exist_ok=True)
                            staged_output_path = staging_dir / output_path.name
                            _STAGING_DIRS.append(staging_dir)
                    except Exception:
                        staging_dir = None
                        staged_output_path = output_path

                # Build the NoData baseline list (fall back to image nodata; UNION with extra_nd_values)
                nd_set: Set[float] = set()
                # MCD14A1: ignore the image's native nodata metadata (invalid value is fixed at 255)
                if (not is_mcd14a1_context) and (src_profile.get('nodata') is not None):
                    try:
                        nd_set.add(float(src_profile['nodata']))
                    except Exception:
                        pass
                for v in (extra_nd_values or []):
                    try:
                        nd_set.add(float(v))
                    except Exception:
                        pass
                if is_mcd12q1_context:
                    nd_set.update([0.0])
                if is_mcd14a1_context:
                    nd_set.update([255.0])
                nd_vals = sorted(list(nd_set))

                # Output
                with rasterio.open(staged_output_path, 'w', **profile) as dst:
                    dst.descriptions = tuple(band_names)

                    height, width = src.height, src.width
                    for b in range(1, src.count + 1):
                        # Whether to process this band (only non-QA bands present in band_ops are scaled/normalized)
                        do_math = ((b - 1) in band_ops) and ((b - 1) not in qa_indices)

                        # Block-wise processing: prefer the GeoTIFF native block window (closer to the compression tiles, more IO-friendly)
                        # If the input is not tiled or reading fails, fall back to a manual block_size grid.
                        try:
                            windows_iter = (win for _, win in src.block_windows(b))
                        except Exception:
                            windows_iter = None

                        if windows_iter is None:
                            def _fallback_windows():
                                for y in range(0, height, block_size):
                                    h = min(block_size, height - y)
                                    for x in range(0, width, block_size):
                                        w = min(block_size, width - x)
                                        yield Window(x, y, w, h)
                            windows_iter = _fallback_windows()

                        for win in windows_iter:
                            data = src.read(b, window=win)

                            # Build NoData mask (standard pass first)
                            if nd_vals:
                                mask_nd = build_nodata_mask(data, nd_vals)
                            else:
                                mask_nd = np.isnan(data) if np.issubdtype(data.dtype, np.floating) else np.zeros(data.shape, dtype=bool)

                            # ====== Special rule: MCD15A3H: map value==249 to 0, and these positions are not treated as NoData ======
                            if is_mcd15_context:
                                to_zero = (data == 249)
                                if np.any(to_zero):
                                    mask_nd[to_zero] = False
                                    # In-place rewrite: avoid unnecessary dtype upgrades
                                    if out_is_float:
                                        data = data.astype(np.float32, copy=False)
                                    elif out_is_int:
                                        data = data.astype(np.int16 if out_dtype == 'int16' else np.int32, copy=False)
                                    data[to_zero] = 0

                            if do_math:
                                scale, offset, vmin, vmax = band_ops[b - 1]
                                valid_mask = ~mask_nd
                                if np.any(valid_mask):
                                    keep_zero_mask = None
                                    if is_mcd15_context:
                                        keep_zero_mask = valid_mask & (data == 0)

                                    # Use float32/float64 uniformly for computation (avoid repeated fancy indexing)
                                    work = data.astype(np.float32, copy=False)
                                    work = work * np.float32(scale) + np.float32(offset)

                                    if vmin is not None and vmax is not None:
                                        if out_of_range_action == "nodata":
                                            bad = valid_mask & ((work < vmin) | (work > vmax))
                                            if np.any(bad):
                                                mask_nd |= bad
                                                valid_mask = ~mask_nd
                                        else:
                                            work = np.clip(work, vmin, vmax)

                                        rng = (vmax - vmin)
                                        if rng != 0:
                                            work = (work - vmin) / rng

                                    if keep_zero_mask is not None and np.any(keep_zero_mask):
                                        work = work.copy()  # Avoid potential pitfalls of view / broadcasting
                                        work[keep_zero_mask] = 0.0

                                    # Write back: only write computed results for valid pixels
                                    if out_is_float:
                                        out_block = work.astype(out_np_dtype, copy=False)
                                    else:
                                        # Integer output: round and safely clip to dtype range
                                        out_block = np.rint(work).astype(np.float32, copy=False)
                                        if out_dtype == 'int16':
                                            out_block = np.clip(out_block, -32768, 32767).astype(np.int16, copy=False)
                                        else:
                                            # Other int types follow numpy conversion rules (unlikely to be used in practice)
                                            out_block = out_block.astype(out_np_dtype, copy=False)
                                    # Preserve original data shape/dtype for nodata write-back
                                    data = out_block
                                    # Release the work reference
                                    del work

                            # Uniformly write NoData pixels as -32768 (regardless of do_math)
                            if np.any(mask_nd):
                                if out_is_float:
                                    data = data.astype(np.float32, copy=False)
                                elif out_is_int:
                                    data = data.astype(np.int16 if out_dtype == 'int16' else np.int32, copy=False)
                                data[mask_nd] = UNIFIED_NODATA


                            # Write block
                            dst.write(data.astype(out_dtype, copy=False), b, window=win)

                # Move to destination directory (cross-FS): copy to a same-directory tmp file then atomically replace
                if staged_output_path != output_path:
                    try:
                        _safe_copy_then_replace(staged_output_path, output_path)
                    finally:
                        try:
                            staged_output_path.unlink(missing_ok=True)  # py3.8+ OK
                        except Exception:
                            try:
                                if staged_output_path.exists():
                                    staged_output_path.unlink()
                            except Exception:
                                pass
                        if staging_dir is not None:
                            try:
                                shutil.rmtree(staging_dir, ignore_errors=True)
                            except Exception:
                                pass
                            try:
                                if staging_dir in _STAGING_DIRS:
                                    _STAGING_DIRS.remove(staging_dir)
                            except Exception:
                                pass

                return True

    except Exception as e:
        logger.error(f"Processing file {input_path} failed: {e}")
        return False


# ================== Process folder ==================
def process_folder(
    input_folder: Path,
    output_folder: Path,
    band_ops: Dict[int, Tuple[float, float, Optional[float], Optional[float]]],
    qa_keywords: Optional[List[str]],
    extra_nd_values: List[float],
    max_workers: Optional[int] = None,
    out_of_range_action: str = "nodata",
    skip_existing: bool = False,
    verbose: bool = False
) -> Tuple[int, int]:
    if not input_folder.exists():
        logger.error(f"Input folder does not exist: {input_folder}")
        return 0, 0

    output_folder.mkdir(parents=True, exist_ok=True)

    tif_files = list(input_folder.glob("*.tif")) + list(input_folder.glob("*.TIF"))
    if not tif_files:
        logger.warning(f"In {input_folder}: no TIF files found")
        return 0, 0

    input_folder_str = str(input_folder.as_posix()).lower()
    is_mcd15_context = ("mcd15a3h" in input_folder_str)
    is_mcd12q1_context = ("mcd12q1" in input_folder_str)
    is_mcd14a1_context = ("mcd14a1" in input_folder_str)
    if is_mcd15_context:
        logger.info("Detected 'mcd15a3h' in input path: enabling rule 249->0 with original 0 values preserved")
    if is_mcd12q1_context:
        logger.info("Detected 'mcd12q1' in input path: 0 additionally treated as invalid")
    if is_mcd14a1_context:
        logger.info("Detected 'mcd14a1' in input path: 255 additionally treated as invalid, and the image's native nodata metadata is ignored")

    logger.info(f"Found {len(tif_files)} TIF files")

    # Filter out files that already exist (if skipping is enabled)
    files_to_process = []
    skipped_count = 0
    
    for tif in tif_files:
        out_file = output_folder / tif.name
        
        if skip_existing and out_file.exists() and out_file.stat().st_size > 0:
            skipped_count += 1
            if verbose:
                logger.info(f"Skip existing file: {out_file}")
        else:
            files_to_process.append(tif)
    
    if skipped_count > 0:
        logger.info(f"Skip {skipped_count} existing files")
    
    logger.info(f"Need to process {len(files_to_process)} files")

    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, len(files_to_process))

    # Per-file parallelism: this task is both IO-bound and numpy-heavy, multiprocessing saturates the CPU better
    # If rasterio/GDAL is unstable with multiprocessing in your environment, switch back to ThreadPoolExecutor.
    use_process_pool = True

    success_count = 0
    Executor = ProcessPoolExecutor if use_process_pool else ThreadPoolExecutor
    with Executor(max_workers=max_workers) as ex:
        futures = {}
        for tif in files_to_process:
            out_file = output_folder / tif.name
            fut = ex.submit(
                process_tif_file,
                tif,
                out_file,
                band_ops,
                qa_keywords,
                extra_nd_values,
                BLOCK,
                is_mcd15_context=is_mcd15_context,
                is_mcd12q1_context=is_mcd12q1_context,
                is_mcd14a1_context=is_mcd14a1_context,
                out_of_range_action=out_of_range_action,
                local_stage_root=Path("/seal"),
            )
            futures[fut] = tif

        with tqdm(total=len(files_to_process), desc="Process TIF files") as pbar:
            for fut in as_completed(futures):
                ok = False
                try:
                    ok = fut.result()
                except Exception as e:
                    logger.error(f"Processing file {futures[fut]} error: {e}")
                if ok:
                    success_count += 1
                pbar.update(1)

    return success_count, len(files_to_process)


# ================== CLI ==================
def main():
    parser = argparse.ArgumentParser(description='Scale and min-max normalize specified bands of TIF files, and unify NoData=-32768 (no clipping to [0,1])')
    parser.add_argument('--input_folder',
                        default='/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled_5xdownsampled_merged_clipped_unified/ERA5_merged_clipped_unified',
                        type=Path, help='Input folder path (single level, not recursive)')
    parser.add_argument('--output_folder',
                        default='/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled_5xdownsampled_merged_clipped_unified_scaled/ERA5_merged_clipped_unified_scaled',
                        type=Path, help='Output folder path')
    parser.add_argument('--band_config',
                        # band:scale:offset:min:max -- min/max as nan means no normalization
                        # MCD09GA_b1237_mosaic
                        # default='0:0.0001:0:-0.01:1.6, 1:0.0001:0:-0.01:1.6, 2:0.0001:0:-0.01:1.6, 3:0.0001:0:-0.01:1.11',
                        # default='0:0.1:0:-3.86:8338.89, 1:0.01:0:4.53:8620.19, 2:0.001:0:15.45:9426.45, 3:0.001:0:-17.73:1937.25',
                        # after mosaic downsampling
                        # default='0:0.0001:0:0:1.54, 1:0.0001:0:0:1.53, 2:0.0001:0:0:1.41, 3:0.0001:0:0:0.92',
                        # vpd
                        # default='0:1:0:0:1.63',
                        # MCD11A1_mosaic
                        # default='0:0.02:0:7500:20000, 1:0.02:0:7500:20000, 2:0.002:0.49:1:255, 3:0.002:0.49:1:255',
                        # downsampled
                        # default='0:0.02:0:5.1:333.7, 1:0.02:0:5.1:333.7, 2:0.002:0.49:0.75:1, 3:0.002:0.49:0.75:1',
                        # MCD13Q1_mosaic
                        # just for clip
                        # default='0:1:0:0:1, 1:1:0:0:1, 2:1:0:0:1, 3:1:0:0:1, 4:1:0:0:1, 5:1:0:0:1',
                        # real norm
                        # default='0:0.0001:0:-0.2:1, 1:0.0001:0:-0.2:1, 2:0.0001:0:0:1, 3:0.0001:0:0:1, 4:0.0001:0:0:1, 5:0.0001:0:0:1',
                        # MCD14A1_mosaic
                        # default='0:1:0:nan:nan, 1:0.1:0:0:18000',
                        # MCD15A3H_mosaic
                        # default='0:0.1:0:0:7, 1:0.01:0:0:1',
                        # default='0:0.1:0:0:46.78, 1:0.01:0:0:87.14',
                        # ERA5_mosaic
                        # # default='0:1:0:224:309.5, 1:1:0:-17:16.4, 2:1:0:-18.6:19.241, 3:1:0:0:100, 4:1:0:0:0.21, 5:1:0:-35157552:9870987, 6:1:0:220:295, 7:1:0:72353:104455, 8:1:0:0:0.7661, 9:1:0:0:0.7661,10:1:0:0:0.7661, 11:1:0:0:0.7661',
                        # default='0:1:0:224:310, 1:1:0:-17:16.4, 2:1:0:-18.6:19.241, 3:1:0:-3:100, 4:1:0:-6.4:0.21, 5:1:0:-39971252:9905048, 6:1:0:219.9:296, 7:1:0:71048.5:104608, 8:1:0:0:0.7661, 9:1:0:0:0.7661,10:1:0:0:0.7661, 11:1:0:0:0.7661',
                        default='0:1:0:0.005193258648563279:297.7777831997186, 1:1:0:-8.035555467392681:9.75282316575327, 2:1:0:-8.884559637656066:7.783394794594883, 3:1:0:0.00013918371392302747:99.94999949848244, 4:1:0:1.9188684693810958e-07:0.03603348170585861, 5:1:0:-14189138.410743382:2327057.3615739984, 6:1:0:0.005005088835593607:291.85923360787075, 7:1:0:1.7814644058506481:102594.00284017395, 8:1:0:1.2825456326856735e-05:0.7495786155920634, 9:1:0:1.2824859849846285e-05:0.7429968109043169, 10:1:0:1.2793656195784068e-05:0.7264044759278405, 11:1:0:1.2811615921890935e-05:0.7151072856451576, 12:1:0:0.005317793384206413:304.2753351358254, 13:1:0:0.005381247822265205:305.6516323146297, 14:1:0:-0.04698579626080153:0.00023615964263532034, 15:1:0:-0.005673370511294594:0.000825804130791964, 16:1:0:4.4439247570826976e-09:0.0005425088177093472, 17:1:0:413.45007501667135:28043026.67241847',
                        # MCD09CMG_mosaic
                        # default='0:1:0:0.01:400, 1:1:0:0.01:400, 2:1:0:0.01:400, 3:1:0:0.01:400',
                        # MCD09CMG_mosaic filled downsampled
                        # default='0:1:0:219:334.4, 1:1:0:193:400, 2:1:0:204:356, 3:1:0:202:386.3',
                        # ndvi evi
                        # default='0:0.0001:0:-0.35:0.95, 1:0.0001:0:-1:1',
                        # FWI
                        # default='0:0.0001:0:0:101, 1:0.0001:0:0:119, 2:0.0001:0:0:694, 3:0.0001:0:0:11, 4:0.0001:0:0:149, 5:0.0001:0:0:33',
                        help='Band config in the form "band:scale:offset:min:max"; comma-separate multiple entries; nan for min/max disables normalization (scale/offset are then ignored and treated as 1/0)')
    parser.add_argument(
        "--extra-nodata",
        nargs="+",
        type=float,
        default=[-32768, 32768],
        help=(
            "Extra NoData value list (UNION with file's native nodata by default); default: -32768 32768."
            "If path contains mcd12q1, 0 is added; if path contains mcd14a1, 255 is added (and tif's native nodata is ignored)"
        ),
    )
    parser.add_argument('--qa-keywords', nargs='+', default=DEFAULT_QA_KEYWORDS,
                        help='QA band keyword list (case-insensitive)')
    parser.add_argument('--max-workers', type=int, default=8,
                        help='Maximum concurrency (default: number of CPU cores)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--out-of-range-action', choices=['nodata', 'clip'], default='clip',
                        help='How to handle out-of-range pixels: nodata sets to NoData, clip clips to range boundary (default: clip)')
    parser.add_argument('--skip-existing', action='store_true', default=False,
                        help='Skip existing output files (default: False, reprocess all files)')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        band_ops = parse_band_config(args.band_config)
        if not band_ops:
            logger.warning("band_config is empty: no band will be scaled or normalized this run (NoData will simply be unified to -32768 and nodata=-32768 set)")
        else:
            logger.info("Band config: " + ", ".join(
                f"{b}:{sf}:{off}:{'nan' if vmn is None else vmn}:{'nan' if vmx is None else vmx}"
                for b,(sf,off,vmn,vmx) in band_ops.items()
            ))
    except ValueError as e:
        logger.error(e)
        sys.exit(1)

    logger.info(f"Starting to process folder: {args.input_folder}")
    logger.info(f"Output folder: {args.output_folder}")

    ok, total = process_folder(
        args.input_folder,
        args.output_folder,
        band_ops,
        args.qa_keywords,
        args.extra_nodata,
        args.max_workers,
        args.out_of_range_action,
        args.skip_existing,
        args.verbose
    )
    logger.info(f"Processing complete: {ok}/{total} files succeeded")


if __name__ == "__main__":
    main()
