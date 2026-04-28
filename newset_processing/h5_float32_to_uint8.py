#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert H5 files produced by generate_full_dataset_new_int.py (per-pixel datasets are float32) to uint8.

Rounding rule: truncate the fractional part toward zero, matching numpy's float -> uint8 conversion; no scaling, clipping, or other transforms.

Performance notes (important):
- By default, datasets are processed in a stream: read one -> convert one -> write one, avoiding loading the whole h5 into memory at once.
- Optional `--dataset-fetch-workers`: prefetch / convert multiple datasets in parallel (each worker independently opens the input file read-only),
  controlled by `--dataset-readahead` for queue depth, overlapping IO/decompression with writes.

Usage:
  python newset_processing/h5_float32_to_uint8.py \\
    --input-dir /path/to/h5_dir --output-dir /path/to/out_uint8
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import logging
import os
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from typing import Iterable, Optional

import h5py
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_TQDM_NUMERIC = {"bar_format": "{desc}: {n_fmt}/{total_fmt}"}


def _h5_open_kwargs():
    return dict(
        libver="latest",
        rdcc_nbytes=512 * 1024 * 1024,
        rdcc_nslots=100003,
    )


def _float32_to_uint8_trunc(arr: np.ndarray) -> np.ndarray:
    """Truncate toward zero and cast to uint8 (matches arr.astype(np.uint8) behavior for floats)."""
    return np.trunc(arr).astype(np.uint8, copy=False)


@dataclasses.dataclass(frozen=True)
class _DatasetPayload:
    name: str
    data_uint8: np.ndarray
    chunks: Optional[tuple[int, ...]]
    compression: Optional[str]
    compression_opts: object
    shuffle: bool
    attrs: dict


def _read_convert_one_dataset(src_path: str, name: str) -> _DatasetPayload:
    """
    Independently open the input file to read a single dataset (suitable for multithreaded prefetch).
    Note: do not share the same h5py.File handle across threads.
    """
    with h5py.File(src_path, "r", **_h5_open_kwargs()) as f_in:
        ds_in = f_in[name]
        data = ds_in[()]
        out = _float32_to_uint8_trunc(np.asarray(data, dtype=np.float32))
        return _DatasetPayload(
            name=name,
            data_uint8=out,
            chunks=ds_in.chunks,
            compression=ds_in.compression,
            compression_opts=ds_in.compression_opts,
            shuffle=bool(ds_in.shuffle),
            attrs={ak: ds_in.attrs[ak] for ak in ds_in.attrs.keys()},
        )


def convert_one_h5(
    src_path: str,
    dst_path: str,
    *,
    tmp_suffix: str = ".partial",
    dataset_fetch_workers: int = 1,
    dataset_readahead: int = 1,
) -> None:
    """
    Single-file conversion: streaming write; optional parallel dataset prefetch / conversion.

    dataset_fetch_workers=1: strictly sequential reads (minimum memory, most conservative).
    dataset_fetch_workers>1: parallel prefetch, with queue depth controlled by dataset_readahead.
    """
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)) or ".", exist_ok=True)
    tmp_path = dst_path + tmp_suffix + f".{os.getpid()}"
    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    fetch_w = max(1, int(dataset_fetch_workers))
    ahead = max(1, int(dataset_readahead))

    try:
        with h5py.File(src_path, "r", **_h5_open_kwargs()) as f_in:
            root_attrs: dict = {}
            for k in f_in.attrs.keys():
                root_attrs[k] = "uint8" if k == "value_dtype" else f_in.attrs[k]
            keys = list(f_in.keys())

        with h5py.File(tmp_path, "w", **_h5_open_kwargs()) as f_out:
            for k, v in root_attrs.items():
                f_out.attrs[k] = v

            if fetch_w == 1:
                with h5py.File(src_path, "r", **_h5_open_kwargs()) as f_in:
                    for name in tqdm(keys, desc=os.path.basename(dst_path), **_TQDM_NUMERIC):
                        ds_in = f_in[name]
                        data = ds_in[()]
                        out = _float32_to_uint8_trunc(np.asarray(data, dtype=np.float32))
                        ds_out = f_out.create_dataset(
                            name,
                            data=out,
                            chunks=ds_in.chunks,
                            compression=ds_in.compression,
                            compression_opts=ds_in.compression_opts,
                            shuffle=bool(ds_in.shuffle),
                        )
                        for ak in ds_in.attrs.keys():
                            ds_out.attrs[ak] = ds_in.attrs[ak]
                f_out.flush()
            else:
                max_inflight = min(len(keys), fetch_w * ahead)
                inflight = min(fetch_w, len(keys))
                key_iter = iter(keys)
                pending: set = set()
                submitted = 0
                done = 0

                with ThreadPoolExecutor(max_workers=fetch_w) as pool:
                    def submit_next() -> bool:
                        nonlocal submitted
                        try:
                            n = next(key_iter)
                        except StopIteration:
                            return False
                        pending.add(pool.submit(_read_convert_one_dataset, src_path, n))
                        submitted += 1
                        return True

                    for _ in range(inflight):
                        if not submit_next():
                            break

                    with tqdm(total=len(keys), desc=os.path.basename(dst_path), **_TQDM_NUMERIC) as pbar:
                        while pending:
                            done_set, _ = wait(pending, return_when=FIRST_COMPLETED)
                            for fut in done_set:
                                pending.remove(fut)
                                payload: _DatasetPayload = fut.result()
                                ds_out = f_out.create_dataset(
                                    payload.name,
                                    data=payload.data_uint8,
                                    chunks=payload.chunks,
                                    compression=payload.compression,
                                    compression_opts=payload.compression_opts,
                                    shuffle=payload.shuffle,
                                )
                                for ak, av in payload.attrs.items():
                                    ds_out.attrs[ak] = av
                                done += 1
                                pbar.update(1)

                                while len(pending) < max_inflight:
                                    if not submit_next():
                                        break

                f_out.flush()

        os.replace(tmp_path, dst_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def _convert_worker(args: tuple[str, str, int, int]) -> str:
    """Multiprocess worker: must be a top-level module function so it can be pickled."""
    src, dst, dataset_fetch_workers, dataset_readahead = args
    convert_one_h5(
        src,
        dst,
        dataset_fetch_workers=dataset_fetch_workers,
        dataset_readahead=dataset_readahead,
    )
    return dst


def _iter_input_files(
    input_dir: Optional[str],
    pattern: str,
) -> list[str]:
    if input_dir:
        base = os.path.abspath(input_dir)
        paths = sorted(glob.glob(os.path.join(base, pattern)))
        return [p for p in paths if os.path.isfile(p)]
    raise ValueError("--input-dir is required")


def _default_output_path(
    input_path: str,
    output_dir: Optional[str],
    suffix: str,
) -> str:
    base = os.path.basename(input_path)
    if base.endswith(".h5"):
        stem = base[:-3]
    else:
        stem = base
    out_name = f"{stem}{suffix}.h5"
    if output_dir:
        return os.path.join(os.path.abspath(output_dir), out_name)
    return os.path.join(os.path.dirname(os.path.abspath(input_path)), out_name)


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Batch: H5 float32 pixel time series -> uint8 (truncated)")
    p.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory (contains many h5 files)",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default="*_year_dataset.h5",
        help="Input directory match pattern (default: *_year_dataset.h5)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory",
    )
    p.add_argument(
        "--suffix",
        type=str,
        default="_uint8",
        help="Output filename in batch mode: {stem}{suffix}.h5 (default: _uint8)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if the target file already exists and is non-empty",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of files processed in parallel (each process independently reads/writes one h5; default 2; 1 means sequential)",
    )
    p.add_argument(
        "--dataset-fetch-workers",
        type=int,
        default=8,
        help="Number of threads for prefetch/conversion of datasets within a file (default 8; 1 means fully sequential)",
    )
    p.add_argument(
        "--dataset-readahead",
        type=int,
        default=2,
        help="Prefetch queue-depth multiplier within a file: max_inflight ~= dataset_fetch_workers * dataset_readahead (default 2)",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    paths = _iter_input_files(args.input_dir, args.pattern)
    if not paths:
        logger.error("No input h5 files found.")
        return 1

    os.makedirs(os.path.abspath(args.output_dir), exist_ok=True)

    plan: list[tuple[str, str, int, int]] = []
    dfw = max(1, int(args.dataset_fetch_workers))
    dra = max(1, int(args.dataset_readahead))
    for src in paths:
        dst = _default_output_path(src, args.output_dir, args.suffix)
        if args.skip_existing and os.path.isfile(dst) and os.path.getsize(dst) > 0:
            logger.info("Already exists, skipping: %s", dst)
            continue
        plan.append((src, dst, dfw, dra))

    if not plan:
        logger.info("No files to process (everything may already exist).")
        return 0

    workers = max(1, int(args.workers))
    processed = 0
    if workers == 1:
        for src, dst, w, a in plan:
            logger.info("Convert: %s -> %s (dataset_fetch_workers=%d, dataset_readahead=%d)", src, dst, w, a)
            convert_one_h5(src, dst, dataset_fetch_workers=w, dataset_readahead=a)
            processed += 1
    else:
        logger.info("Parallel conversion: file_workers=%d, pending=%d", workers, len(plan))
        ex: ProcessPoolExecutor | None = None
        futs: list = []
        try:
            ex = ProcessPoolExecutor(max_workers=workers)
            futs = [ex.submit(_convert_worker, item) for item in plan]
            for fut in as_completed(futs):
                dst = fut.result()
                processed += 1
                logger.info("Done %d/%d: %s", processed, len(plan), dst)
        except KeyboardInterrupt:
            logger.warning("Interrupt received: cancelling unfinished tasks and terminating child processes...")
            for fut in futs:
                try:
                    fut.cancel()
                except Exception:
                    pass
            if ex is not None:
                # Python 3.9+: cancel_futures best-effort cancels tasks that have not yet started
                try:
                    ex.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    ex.shutdown(wait=False)
            raise
        finally:
            if ex is not None:
                try:
                    ex.shutdown(wait=True, cancel_futures=True)
                except TypeError:
                    ex.shutdown(wait=True)

    logger.info("Done; processed %d files in total.", processed)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
