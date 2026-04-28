# -*- coding: utf-8 -*-
from __future__ import annotations

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
import datetime
import time
import threading
import queue
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
import shutil
import uuid
import atexit
import signal
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set, Iterable, Iterator

import numpy as np

# Compat: unit tests use importlib to dynamically load modules without registering them in sys.modules.
# In Python 3.12, dataclasses accesses sys.modules[__name__].__dict__ when resolving string annotations.
try:  # pragma: no cover
    import types as _types

    if sys.modules.get(__name__) is None:
        _m = _types.ModuleType(__name__)
        _m.__dict__.update(globals())
        sys.modules[__name__] = _m
    else:
        try:
            sys.modules[__name__].__dict__.update(globals())
        except Exception:
            pass
except Exception:  # pragma: no cover
    pass

try:
    import rasterio  # type: ignore
    from rasterio.windows import Window  # type: ignore
except Exception:  # pragma: no cover
    rasterio = None  # type: ignore
    Window = None  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


def _transform_block_to_uint8(block: np.ndarray, vmin: float, vmax: float, is_mcd15a3h: bool) -> np.ndarray:
    """
    Pure function: min-max normalize the block and map to uint8 (0..255).

    Rules:
    - Generic datasets:
      - Input values of -32768 / 32768 / 255 / NaN do not participate in normalization; output is 255
      - Other values are normalized by (x - vmin)/(vmax - vmin) * 255 and cast to uint8
    - mcd15a3h:
      - Input -32768 / 32768 always output as 0 and do not participate in normalization
      - Input 255 / NaN output as 255
      - Other values follow the generic normalization rule
    - Any NaN/Inf produced for valid pixels during normalization must raise (memory protection).
    """
    arr = np.asarray(block)
    is_float = np.issubdtype(arr.dtype, np.floating)
    nan_mask = np.isnan(arr) if is_float else np.zeros(arr.shape, dtype=bool)

    # 255 may also appear in many integer datasets; uniformly treat as invalid (output 255)
    mask_255 = (arr == 255)
    mask_neg = (arr == -32768)
    mask_pos = (arr == 32768)

    if is_mcd15a3h:
        fixed_zero_mask = mask_neg | mask_pos
        invalid_to_255_mask = mask_255 | nan_mask
        valid_mask = ~(fixed_zero_mask | invalid_to_255_mask)

        out = np.full(arr.shape, 255, dtype=np.uint8)
        if np.any(fixed_zero_mask):
            out[fixed_zero_mask] = np.uint8(0)
    else:
        invalid_to_255_mask = mask_neg | mask_pos | mask_255 | nan_mask
        valid_mask = ~invalid_to_255_mask
        out = np.full(arr.shape, 255, dtype=np.uint8)

    if not np.any(valid_mask):
        return out

    if float(vmax) <= float(vmin):
        raise ValueError("normalization range must satisfy vmax > vmin for valid pixels")
    rng = float(vmax) - float(vmin)

    work = arr.astype(np.float32, copy=False)
    vals = work[valid_mask]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        scaled = (vals - np.float32(vmin)) / np.float32(rng) * np.float32(255.0)

    if not np.all(np.isfinite(scaled)):
        raise ValueError("Normalization produced NaN/Inf for valid pixels")

    # Avoid float -> uint8 wrap-around: clip in memory before cast.
    np.clip(scaled, 0.0, 255.0, out=scaled)
    out[valid_mask] = scaled.astype(np.uint8, copy=False)
    return out


def _map_special_and_cast_uint8(block: np.ndarray, is_mcd15a3h: bool) -> np.ndarray:
    """
    QA bands or unconfigured bands: skip normalization, only do "special-value mapping + safe cast to uint8".

    Rules (from spec):
    - Special-value set: -32768 / 32768 / NaN (and 255 is special but still maps to 255)
    - Generic datasets: -32768/32768/NaN -> 255
    - mcd15a3h: -32768/32768 -> 0; NaN -> 255
    - Other valid values: clip to [0,255], then safely cast to uint8 (avoid wrap-around)
    - Output does not set nodata metadata (controlled by upper-layer profile writer)
    """
    arr = np.asarray(block)
    is_float = np.issubdtype(arr.dtype, np.floating)
    nan_mask = np.isnan(arr) if is_float else np.zeros(arr.shape, dtype=bool)

    mask_neg = (arr == -32768)
    mask_pos = (arr == 32768)

    # First, safely cast "non-special values" to uint8 (clip + rint to avoid wrap-around)
    work = arr.astype(np.float32, copy=False)
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        np.clip(work, 0.0, 255.0, out=work)
    work = np.rint(work, out=work)
    out = work.astype(np.uint8, copy=False)

    if is_mcd15a3h:
        fixed_zero_mask = mask_neg | mask_pos
        if np.any(fixed_zero_mask):
            out[fixed_zero_mask] = np.uint8(0)
        if np.any(nan_mask):
            out[nan_mask] = np.uint8(255)
    else:
        invalid_to_255_mask = mask_neg | mask_pos | nan_mask
        if np.any(invalid_to_255_mask):
            out[invalid_to_255_mask] = np.uint8(255)

    return out


def _match_default_config_prefixes(folder_name: str, prefixes: list[str]) -> list[str]:
    """
    Return the list of matched config prefixes (in the original prefixes order). Match rules:
    - Subdirectory name contains the full prefix as a substring
    - case-insensitive
    """
    folder = (folder_name or "").lower()
    hits: list[str] = []
    for p in prefixes or []:
        if not p:
            continue
        if p.lower() in folder:
            hits.append(p)
    return hits


def _resolve_folder_default_config(folder_name: str, defaults: dict[str, str]) -> str | None:
    """
    Pure function: resolve a uniquely matching default config from `defaults` based on the subdirectory name.

    Conventions:
    - `defaults` keys are prefixes (e.g. "mcd09ga") and values are their config strings (e.g. from --mcd09ga-default)
    - Match rule uses _match_default_config_prefixes (case-insensitive substring match)
    - 0 matches: return None
    - 1 match: return that config string
    - Multiple matches: raise ValueError (to avoid ambiguity)
    """
    defaults = defaults or {}
    prefixes = list(defaults.keys())
    hits = _match_default_config_prefixes(folder_name, prefixes)
    if len(hits) == 0:
        return None
    if len(hits) == 1:
        return defaults[hits[0]]
    raise ValueError(f"Ambiguous default config match for folder={folder_name!r}: {hits}")


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


# ================== /beluga batch pipeline scaffolding (Task 4) ==================
@dataclass(frozen=True)
class BatchItem:
    remote_src: str
    remote_dst: str
    local_src: Optional[Path] = None
    local_dst: Optional[Path] = None
    band_config_text: str = ""
    is_mcd15a3h: bool = False


def _make_scratch_dirs(local_stage_root: Optional[Path]) -> Dict[str, Path]:
    """
    Generate and create the scratch directory layout:
      root/
        input/
        output/
        tmp/

    - If local_stage_root is given: use scale_int_tif_<timestamp>_<pid> beneath it
    - Otherwise default to: /beluga/scale_int_tif_<timestamp>_<pid>
    """
    root_parent = Path(local_stage_root) if local_stage_root is not None else Path("/beluga")
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    root = root_parent / f"scale_int_tif_{ts}_{os.getpid()}"
    inp = root / "input"
    out = root / "output"
    tmp = root / "tmp"
    for d in (inp, out, tmp):
        d.mkdir(parents=True, exist_ok=True)
    return {"root": root, "input": inp, "output": out, "tmp": tmp}


def _copy_large_atomic(src: Path, dst: Path, *, chunk_bytes: int = 4 * 1024 * 1024) -> None:
    """
    Atomic large-buffer copy: write to a same-directory .tmp file, then atomically replace via os.replace.
    - Atomic replacement of dst is only guaranteed within the same filesystem
    - Default chunk size: 4 MB
    """
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_name(dst.name + ".tmp")
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except Exception:
        pass

    try:
        with open(src, "rb") as fsrc, open(tmp_path, "wb") as fdst:
            while True:
                chunk = fsrc.read(chunk_bytes)
                if not chunk:
                    break
                fdst.write(chunk)
            try:
                fdst.flush()
                os.fsync(fdst.fileno())
            except Exception:
                pass
        os.replace(tmp_path, dst)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _process_one_local(item: BatchItem) -> BatchItem:
    """
    Task4: placeholder processing function (the pipeline runs, but no real tif numeric processing happens).
    Current implementation is a "direct copy input -> output", for end-to-end connectivity testing.
    """
    if item.local_src is None or item.local_dst is None:
        raise ValueError("BatchItem.local_src/local_dst must be set before processing")
    ok = process_tif_file_uint8_local(
        item.local_src,
        item.local_dst,
        item.band_config_text,
        is_mcd15_context=item.is_mcd15a3h,
        qa_keywords=DEFAULT_QA_KEYWORDS,
    )
    if not ok:
        raise RuntimeError(f"local tif uint8 processing failed: src={item.local_src} dst={item.local_dst}")
    return item


def run_batch_pipeline(
    items: Iterable[BatchItem],
    *,
    local_stage_root: Optional[Path] = None,
    scratch: Optional[Dict[str, Path]] = None,
    prefetch_workers: int = 16,
    process_workers: Optional[int] = None,
    writeback_workers: int = 16,
    prefetch_ahead: int = 100,
    writeback_max_inflight: int = 16,
    keep_scratch: bool = False,
) -> Dict[str, int]:
    """
    /beluga three-stage async pipeline scaffolding:
    - prefetch (thread pool): remote src -> local input
    - process (process pool): local input -> local output
    - writeback (thread pool): local output -> remote dst, then delete local output

    Constraints (Task4):
    - process is placeholder-only (no real numeric processing)
    - main() does not wire into this pipeline (Task6)
    """
    prefetch_ahead = int(prefetch_ahead)
    if prefetch_ahead > 100:
        prefetch_ahead = 100
    if prefetch_ahead < 0:
        raise ValueError("prefetch_ahead must be non-negative")
    writeback_max_inflight = int(writeback_max_inflight)
    if writeback_max_inflight < 1:
        raise ValueError("writeback_max_inflight must be >= 1")

    if scratch is None:
        scratch = _make_scratch_dirs(local_stage_root)
    scratch_root = scratch["root"]
    _STAGING_DIRS.append(scratch_root)
    _install_cleanup_handlers()

    stop_evt = threading.Event()
    # Critical backpressure: limit the number of "finished local-output files" in flight.
    # Must acquire before submitting process; this prevents the case where process is fast and writeback is slow,
    # in which output files would accumulate on disk indefinitely and lead to ENOSPC.
    output_slots = threading.Semaphore(writeback_max_inflight)

    def _best_effort_cleanup() -> None:
        if keep_scratch:
            return
        try:
            shutil.rmtree(scratch_root, ignore_errors=True)
        except Exception:
            pass
        try:
            if scratch_root in _STAGING_DIRS:
                _STAGING_DIRS.remove(scratch_root)
        except Exception:
            pass

    atexit.register(_best_effort_cleanup)

    def _materialize(item: BatchItem) -> BatchItem:
        if item.local_src is not None and item.local_dst is not None:
            return item
        src_name = Path(item.remote_src).name
        dst_name = Path(item.remote_dst).name or src_name
        return BatchItem(
            remote_src=item.remote_src,
            remote_dst=item.remote_dst,
            local_src=scratch["input"] / src_name,
            local_dst=scratch["output"] / dst_name,
            band_config_text=item.band_config_text,
            is_mcd15a3h=item.is_mcd15a3h,
        )

    def _prefetch_one(item: BatchItem) -> BatchItem:
        it = _materialize(item)
        assert it.local_src is not None
        try:
            _copy_large_atomic(Path(it.remote_src), it.local_src)
        except Exception as e:
            logger.warning(f"prefetch failed: src={it.remote_src} -> {it.local_src} err={e}")
            raise
        return it

    def _writeback_one(item: BatchItem) -> None:
        try:
            assert item.local_dst is not None
            _copy_large_atomic(item.local_dst, Path(item.remote_dst))
            try:
                item.local_dst.unlink(missing_ok=True)
            except Exception:
                try:
                    if item.local_dst.exists():
                        item.local_dst.unlink()
                except Exception:
                    pass
            # Delete the local input only after writeback succeeds (on failure, keep it for inspection / retry)
            try:
                if item.local_src is not None:
                    item.local_src.unlink(missing_ok=True)
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"writeback failed: local={item.local_dst} -> dst={item.remote_dst} err={e}")
            raise

    stats = {"planned": 0, "prefetched": 0, "processed": 0, "written": 0, "failed": 0}

    # Try not to crash on exceptions: log the failure and keep draining (best-effort)
    process_workers = process_workers if process_workers is not None else (os.cpu_count() or 1)

    prefetch_inflight: List = []
    writeback_futs: List = []

    def _submit_writeback(ex: ThreadPoolExecutor, item: BatchItem) -> None:
        fut = ex.submit(_writeback_one, item)

        def _release(_f):
            try:
                output_slots.release()
            except Exception:
                pass

        fut.add_done_callback(_release)
        writeback_futs.append(fut)

    def _submit_process(ex: ProcessPoolExecutor, item: BatchItem):
        """
        Reserve an output slot before submitting the process task.
        The slot is released after writeback completes; if process fails, it is released when the main thread harvests the result.
        """
        output_slots.acquire()
        return ex.submit(_process_one_local, item)

    def _release_output_slot_best_effort() -> None:
        try:
            output_slots.release()
        except Exception:
            pass

    try:
        with ThreadPoolExecutor(max_workers=prefetch_workers) as prefetch_ex, \
                ProcessPoolExecutor(max_workers=process_workers) as process_ex, \
                ThreadPoolExecutor(max_workers=writeback_workers) as writeback_ex:

            # prefetch window: at most prefetch_ahead inflight items
            it_items: Iterator[BatchItem] = iter(items)

            def _fill_prefetch_window() -> None:
                if prefetch_ahead == 0:
                    return
                while (not stop_evt.is_set()) and (len(prefetch_inflight) < prefetch_ahead):
                    try:
                        nxt = next(it_items)
                    except StopIteration:
                        return
                    stats["planned"] += 1
                    prefetch_inflight.append(prefetch_ex.submit(_prefetch_one, nxt))

            _fill_prefetch_window()

            # If prefetch_ahead==0: degrade to synchronous prefetch (structure preserved)
            def _iter_prefetched_sync() -> Iterator[BatchItem]:
                for raw in it_items:
                    if stop_evt.is_set():
                        break
                    stats["planned"] += 1
                    yield _prefetch_one(raw)

            def _iter_prefetched_async() -> Iterator[BatchItem]:
                nonlocal prefetch_inflight
                while prefetch_inflight:
                    # Wait for any prefetch to complete, freeing the window and continuing to fill
                    for fut in as_completed(prefetch_inflight, timeout=None):
                        try:
                            prefetch_inflight.remove(fut)
                        except ValueError:
                            pass
                        try:
                            got = fut.result()
                            stats["prefetched"] += 1
                            yield got
                        except Exception as _pf_exc:
                            stats["failed"] += 1
                            if isinstance(_pf_exc, OSError) and getattr(_pf_exc, "errno", None) == 28:
                                logger.error("Disk space exhausted (ENOSPC); stopping prefetch")
                                stop_evt.set()
                        if not stop_evt.is_set():
                            _fill_prefetch_window()
                        break

            prefetched_iter: Iterable[BatchItem]
            if prefetch_ahead == 0:
                prefetched_iter = _iter_prefetched_sync()
            else:
                prefetched_iter = _iter_prefetched_async()

            # process/writeback stage: stream writeback overlapping with process
            # - producer thread pushes prefetched_iter into the queue (queue maxsize provides additional backpressure)
            # - main thread maintains the in_flight_proc set; any process completion immediately triggers writeback (subject to the semaphore backpressure)
            prefetched_q: "queue.Queue[Optional[BatchItem]]" = queue.Queue(maxsize=max(1, prefetch_ahead))
            _SENTINEL = None

            def _prefetched_producer() -> None:
                try:
                    for pi in prefetched_iter:
                        if stop_evt.is_set():
                            break
                        prefetched_q.put(pi)  # may block: provides backpressure to prefetch stage
                except Exception as e:
                    logger.warning(f"prefetch producer failed: err={e}")
                    stop_evt.set()
                finally:
                    # Ensure consumer can terminate.
                    try:
                        prefetched_q.put(_SENTINEL)
                    except Exception:
                        pass

            producer_th = threading.Thread(target=_prefetched_producer, name="prefetch-producer", daemon=True)
            producer_th.start()

            # Process-pool inflight cap: avoid unbounded future accumulation (and let process/writeback truly overlap)
            max_inflight_proc = max(1, min(100, int(process_workers) * 2))
            in_flight_proc: Set = set()
            got_sentinel = False

            while (not got_sentinel) or in_flight_proc:
                if stop_evt.is_set():
                    break

                # Fill the process inflight as much as possible (don't block forever when the queue is empty)
                while (not got_sentinel) and (len(in_flight_proc) < max_inflight_proc):
                    try:
                        pi = prefetched_q.get_nowait()
                    except queue.Empty:
                        break
                    if pi is _SENTINEL:
                        got_sentinel = True
                        break
                    in_flight_proc.add(_submit_process(process_ex, pi))

                # Harvest completed process tasks and submit writeback as soon as possible
                if in_flight_proc:
                    # If the queue is temporarily empty, allow a short blocking wait for any process completion;
                    # Otherwise harvest quickly with timeout=0 to keep up the queue-consumption pace.
                    try:
                        q_empty = prefetched_q.empty()
                    except Exception:
                        q_empty = False
                    should_block = got_sentinel or q_empty or (len(in_flight_proc) >= max_inflight_proc)
                    timeout = None if should_block else 0
                    done, _ = wait(in_flight_proc, return_when=FIRST_COMPLETED, timeout=timeout)
                    for pf in done:
                        try:
                            in_flight_proc.remove(pf)
                        except KeyError:
                            pass
                        if stop_evt.is_set():
                            break
                        try:
                            out_item = pf.result()
                            stats["processed"] += 1
                        except Exception as e:
                            stats["failed"] += 1
                            logger.warning(f"process failed: err={e}")
                            # process failed: release the output slot and best-effort clean up the (possibly half-written) local output
                            _release_output_slot_best_effort()
                            continue
                        try:
                            _submit_writeback(writeback_ex, out_item)
                        except Exception as e:
                            stats["failed"] += 1
                            logger.warning(f"writeback submit failed: err={e}")
                            # writeback could not be submitted: release the output slot and best-effort clean up the local output file
                            try:
                                if out_item.local_dst is not None:
                                    out_item.local_dst.unlink(missing_ok=True)
                            except Exception:
                                pass
                            _release_output_slot_best_effort()
                else:
                    # No inflight process: wait for a new prefetched item or the termination signal
                    try:
                        pi = prefetched_q.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if pi is _SENTINEL:
                        got_sentinel = True
                    else:
                        in_flight_proc.add(_submit_process(process_ex, pi))

            # Ensure the producer has exited (best-effort)
            try:
                producer_th.join(timeout=5.0)
            except Exception:
                pass

            # Wait for writeback to drain
            for wf in as_completed(writeback_futs):
                try:
                    wf.result()
                    stats["written"] += 1
                except Exception:
                    stats["failed"] += 1

    finally:
        # Best-effort cleanup: prioritize not crashing during cleanup
        # Note: writeback was already drained inside the with-block; this is just a fallback
        try:
            if not keep_scratch:
                _best_effort_cleanup()
        except Exception:
            pass

    return stats


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
    Cross-filesystem moves (e.g. /beluga -> NFS) cannot use os.replace, so we:
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


# ================== uint8 local single-file processing (Task5) ==================
def process_tif_file_uint8_local(
    input_path: Path | str,
    output_path: Path | str,
    band_config_text: str,
    is_mcd15_context: bool,
    qa_keywords: List[str] = DEFAULT_QA_KEYWORDS,
) -> bool:
    """
    Task5: the real local-processing function (single read, single write; block-wise by window).

    Semantics (from spec):
    - For configured non-QA bands: min-max normalize (ignoring special values), output uint8
    - For QA bands or unconfigured bands: skip normalization, only do special-value mapping + safe cast to uint8
    - Output tif dtype=uint8 and do not set nodata metadata
    - Do not write an intermediate float tif; normalization and to-uint8 happen in memory in one pass
    """
    if rasterio is None or Window is None:
        raise ImportError("rasterio is required for process_tif_file_uint8_local")

    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Parse band_config: use vmin/vmax only; if missing or vmin/vmax=None, treat as an "unconfigured band"
    band_ops = parse_band_config(band_config_text or "")

    # Write the output to a same-directory temp file first, then atomically replace via os.replace (no double-write of data)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except Exception:
        pass

    try:
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(in_path) as src:
                src_profile = src.profile.copy()
                band_names = [
                    src.descriptions[i] if src.descriptions[i] else f"Band_{i+1}"
                    for i in range(src.count)
                ]
                qa_indices = detect_qa_bands(band_names, qa_keywords)

                profile = src_profile.copy()
                profile.update(dtype="uint8")
                profile.pop("nodata", None)  # spec: do not set nodata metadata

                with rasterio.open(tmp_path, "w", **profile) as dst:
                    dst.descriptions = tuple(band_names)

                    height, width = src.height, src.width
                    for b in range(1, src.count + 1):
                        bidx0 = b - 1
                        is_qa = (bidx0 in qa_indices)

                        cfg = band_ops.get(bidx0)
                        has_norm_cfg = False
                        vmin: float = 0.0
                        vmax: float = 0.0
                        if cfg is not None:
                            _, _, vmn, vmx = cfg
                            if vmn is not None and vmx is not None:
                                has_norm_cfg = True
                                vmin = float(vmn)
                                vmax = float(vmx)

                        do_norm = (not is_qa) and has_norm_cfg

                        try:
                            windows_iter = (win for _, win in src.block_windows(b))
                        except Exception:
                            windows_iter = None

                        if windows_iter is None:
                            def _fallback_windows() -> Iterator[Window]:
                                for y in range(0, height, BLOCK):
                                    h = min(BLOCK, height - y)
                                    for x in range(0, width, BLOCK):
                                        w = min(BLOCK, width - x)
                                        yield Window(x, y, w, h)

                            windows_iter = _fallback_windows()

                        for win in windows_iter:
                            block = src.read(b, window=win)
                            if do_norm:
                                out_block = _transform_block_to_uint8(block, vmin=vmin, vmax=vmax, is_mcd15a3h=is_mcd15_context)
                            else:
                                out_block = _map_special_and_cast_uint8(block, is_mcd15a3h=is_mcd15_context)
                            dst.write(out_block, b, window=win)

        os.replace(tmp_path, out_path)
        return True
    except Exception as e:
        logger.error(f"uint8 local processing failed: input={in_path} output={out_path} err={e}")
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        return False


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
    local_stage_root: Optional[Path] = Path("/beluga"),
) -> bool:
    """
    band_ops: {band_idx: (scale, offset, vmin, vmax)} (0-based)
    extra_nd_values: extra NoData value list (UNION with file's native nodata)
    is_mcd15_context: if in MCD15A3H context, perform 249->0 (without treating as NoData) and keep original 0 values as 0
    is_mcd12q1_context: if in MCD12Q1 context, additionally treat 0 as invalid
    is_mcd14a1_context: if in MCD14A1 context, additionally treat 255 as invalid and ignore the image's native nodata metadata
    out_of_range_action: how to handle out-of-range pixels: "nodata" sets to NoData, "clip" clips to range boundary.
    """
    if rasterio is None or Window is None:
        raise ImportError("rasterio is required for process_tif_file")
    if tqdm is None:
        raise ImportError("tqdm is required for process_tif_file")

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

                # -- Stage to local disk then transfer: write under /beluga/<random_dir>, then copy to destination and atomically replace --
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
                local_stage_root=Path("/beluga"),
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
                        default='/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled_5xdownsampled_merged_unifiedEntireCan',
                        type=Path, help='Input root directory; its immediate subdirectories contain tifs (one tif layer per subdir, no recursion)')
    parser.add_argument('--output_folder',
                        default='/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled_5xdownsampled_merged_unifiedEntireCan_scaledInt',
                        type=Path, help='Output root directory; preserves the same subdirectory structure as the input')

    # ===== band config overrides & defaults =====
    parser.add_argument(
        "--band-config",
        dest="band_config",
        type=str,
        default=None,
        help=(
            "Override band_config: may be an inline config string or a file path."
            "If provided, all subdirectories that uniquely match a default config will use this override instead of the default."
        ),
    )
    # Keep the original default string content, but rename to a valid CLI argument: --<prefix>-default
    parser.add_argument(
        "--mcd09ga-default",
        type=str,
        default="0:0.0001:0:-3.86:8338.89, 1:0.0001:0:4.53:8620.19, 2:0.0001:0:15.45:9426.45, 3:0.0001:0:-17.73:1937.25",
        help="MCD09GA default band_config (band:scale:offset:min:max)",
    )
    parser.add_argument(
        "--vpd-default",
        type=str,
        default="0:1:0:0:1.63",
        help="VPD default band_config (band:scale:offset:min:max)",
    )
    parser.add_argument(
        "--mcd11a1-default",
        type=str,
        default="0:0.02:0:7500:20000, 1:0.02:0:7500:20000, 2:0.002:0.49:1:255, 3:0.002:0.49:1:255",
        help="MCD11A1 default band_config (band:scale:offset:min:max)",
    )
    parser.add_argument(
        "--mcd15a3h-default",
        type=str,
        default="0:0.1:0:0:46.78, 1:0.01:0:0:87.14",
        help="MCD15A3H default band_config (band:scale:offset:min:max)",
    )
    parser.add_argument(
        "--era5-default",
        type=str,
        default=(
            "0:1:0:0.005193258648563279:297.7777831997186, 1:1:0:-8.035555467392681:9.75282316575327, "
            "2:1:0:-8.884559637656066:7.783394794594883, 3:1:0:0.00013918371392302747:99.94999949848244, "
            "4:1:0:1.9188684693810958e-07:0.03603348170585861, 5:1:0:-14189138.410743382:2327057.3615739984, "
            "6:1:0:0.005005088835593607:291.85923360787075, 7:1:0:1.7814644058506481:102594.00284017395, "
            "8:1:0:1.2825456326856735e-05:0.7495786155920634, 9:1:0:1.2824859849846285e-05:0.7429968109043169, "
            "10:1:0:1.2793656195784068e-05:0.7264044759278405, 11:1:0:1.2811615921890935e-05:0.7151072856451576, "
            "12:1:0:0.005317793384206413:304.2753351358254, 13:1:0:0.005381247822265205:305.6516323146297, "
            "14:1:0:-0.04698579626080153:0.00023615964263532034, 15:1:0:-0.005673370511294594:0.000825804130791964, "
            "16:1:0:4.4439247570826976e-09:0.0005425088177093472, 17:1:0:413.45007501667135:28043026.67241847"
        ),
        help="ERA5 default band_config (band:scale:offset:min:max)",
    )
    parser.add_argument(
        "--mcd09cmg-default",
        type=str,
        default="0:1:0:219:334.4, 1:1:0:193:400, 2:1:0:204:356, 3:1:0:202:386.3",
        help="MCD09CMG default band_config (band:scale:offset:min:max)",
    )
    parser.add_argument(
        "--ndvi-default",
        type=str,
        default="0:1:0:-0.5:1, 1:1:0:-1:1",
        help="NDVI/EVI default band_config (band:scale:offset:min:max)",
    )
    parser.add_argument(
        "--fwi-default",
        type=str,
        default="0:1:0:0:101, 1:1:0:0:119, 2:1:0:0:694, 3:1:0:0:11, 4:1:0:0:149, 5:1:0:0:33",
        help="FWI default band_config (band:scale:offset:min:max)",
    )

    # ===== /beluga pipeline knobs =====
    parser.add_argument(
        "--local-stage-root",
        type=Path,
        default=None,
        help="Local scratch root directory (recommended /beluga/...). If not specified, defaults to /beluga/scale_int_tif_<timestamp>_<pid>.",
    )
    parser.add_argument(
        "--prefetch-ahead",
        type=int,
        default=100,
        help="Maximum number of pre-fetched input files awaiting processing (max supported: 100).",
    )
    parser.add_argument(
        "--writeback-max-inflight",
        type=int,
        default=16,
        help="Maximum number of finished files in the local scratch output awaiting write-back to remote (backpressure).",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Scan / match / count only; do not write files.",
    )
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

    # Basic validation
    if args.prefetch_ahead is None:
        args.prefetch_ahead = 100
    if args.prefetch_ahead > 100:
        args.prefetch_ahead = 100
    if args.prefetch_ahead < 0:
        logger.error("--prefetch-ahead must be a non-negative integer")
        sys.exit(2)

    if args.writeback_max_inflight is None:
        args.writeback_max_inflight = 16
    if args.writeback_max_inflight < 1:
        logger.error("--writeback-max-inflight must be at least 1")
        sys.exit(2)

    # ===== Task6: root subdirectory batch processing + default matching + pipeline scheduling =====
    default_configs: dict[str, str] = {
        "mcd09ga": args.mcd09ga_default,
        "vpd": args.vpd_default,
        "mcd11a1": args.mcd11a1_default,
        "mcd15a3h": args.mcd15a3h_default,
        "era5": args.era5_default,
        "mcd09cmg": args.mcd09cmg_default,
        "ndvi": args.ndvi_default,
        "fwi": args.fwi_default,
    }

    input_root = Path(args.input_folder)
    output_root = Path(args.output_folder)
    if not input_root.exists() or not input_root.is_dir():
        logger.error(f"--input_folder is not a valid directory: {input_root}")
        sys.exit(2)

    # Override-policy notes:
    # - Even when --band-config is provided, the subdirectory name must still uniquely match one defaults prefix;
    #   otherwise (no-hit or multi-hit) the subdirectory is skipped, to avoid mis-applying the override to an unknown dataset.
    if args.band_config:
        logger.info("Detected --band-config: will override the config of all subdirectories that uniquely match a defaults prefix; no-hit / multi-hit are still skipped.")

    planned_items: List[BatchItem] = []
    skipped_no_hit = 0
    skipped_multi_hit = 0
    skipped_no_tif = 0
    skipped_existing = 0

    # Enumerate only the immediate subdirectories of input_root (no deeper recursion)
    try:
        subdirs = sorted([p for p in input_root.iterdir() if p.is_dir()])
    except Exception as e:
        logger.error(f"Failed to enumerate subdirectories: root={input_root} err={e}")
        sys.exit(1)

    for sd in subdirs:
        # First, perform defaults unique-matching (which decides whether this subdirectory joins the batch)
        default_cfg_text: str | None = None
        try:
            default_cfg_text = _resolve_folder_default_config(sd.name, default_configs)
        except ValueError as e:
            skipped_multi_hit += 1
            logger.warning(f"Skip subdirectory (multi-hit defaults): subdir={sd.name} err={e}")
            continue

        if default_cfg_text is None:
            skipped_no_hit += 1
            logger.warning(f"Skip subdirectory (no-hit defaults): subdir={sd.name}")
            continue

        cfg_text = str(args.band_config) if args.band_config else default_cfg_text

        # Within the subdirectory, only process direct .tif/.TIF files (no recursion)
        tif_files = sorted(list(sd.glob("*.tif")) + list(sd.glob("*.TIF")))
        if not tif_files:
            skipped_no_tif += 1
            logger.warning(f"Skip subdirectory (no-tif): subdir={sd.name}")
            continue

        # Structure preserved: output_root/<subdir>/<filename>
        # local paths: in non-dry-run mode, will later be built relative to the scratch input/output roots
        is_mcd15a3h = ("mcd15a3h" in sd.name.lower())
        for tif in tif_files:
            remote_dst = output_root / sd.name / tif.name
            if args.skip_existing and remote_dst.exists():
                skipped_existing += 1
                if args.verbose:
                    logger.info(f"Skip existing output: {remote_dst}")
                continue
            planned_items.append(
                BatchItem(
                    remote_src=str(tif),
                    remote_dst=str(remote_dst),
                    local_src=None,
                    local_dst=None,
                    band_config_text=cfg_text,
                    is_mcd15a3h=is_mcd15a3h,
                )
            )

    if args.dry_run:
        logger.info(
            "dry-run: planned_items=%d skipped_existing=%d skipped_no_hit=%d skipped_multi_hit=%d skipped_no_tif=%d",
            len(planned_items),
            skipped_existing,
            skipped_no_hit,
            skipped_multi_hit,
            skipped_no_tif,
        )
        return

    # Non-dry-run: create the scratch and map local_src/local_dst beneath scratch input/output (preserving the <subdir>/<filename> structure)
    scratch = _make_scratch_dirs(args.local_stage_root)
    scratch_inp = scratch["input"]
    scratch_out = scratch["output"]

    items_with_local: List[BatchItem] = []
    for it in planned_items:
        src_p = Path(it.remote_src)
        subdir_name = src_p.parent.name
        rel = Path(subdir_name) / src_p.name
        items_with_local.append(
            BatchItem(
                remote_src=it.remote_src,
                remote_dst=it.remote_dst,
                local_src=scratch_inp / rel,
                local_dst=scratch_out / rel,
                band_config_text=it.band_config_text,
                is_mcd15a3h=it.is_mcd15a3h,
            )
        )

    logger.info(f"Starting batch pipeline: planned_items={len(items_with_local)} input_root={input_root} output_root={output_root}")

    stats = run_batch_pipeline(
        items_with_local,
        prefetch_ahead=args.prefetch_ahead,
        writeback_max_inflight=args.writeback_max_inflight,
        local_stage_root=args.local_stage_root,
        scratch=scratch,
        process_workers=args.max_workers,
        prefetch_workers=min(16, max(1, int(args.max_workers or 1))),
        writeback_workers=min(16, max(1, int(args.max_workers or 1))),
    )

    ok = int(stats.get("written", 0))
    failed = int(stats.get("failed", 0))
    planned = int(stats.get("planned", len(items_with_local)))
    skipped = max(0, planned - ok - failed)
    logger.info(
        "batch complete: planned=%d ok=%d failed=%d skipped=%d (skipped_existing=%d; subdir_skips: no_hit=%d multi_hit=%d no_tif=%d)",
        planned,
        ok,
        failed,
        skipped,
        skipped_existing,
        skipped_no_hit,
        skipped_multi_hit,
        skipped_no_tif,
    )


if __name__ == "__main__":
    main()

