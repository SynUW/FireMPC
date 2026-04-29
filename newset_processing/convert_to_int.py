#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recursively convert GeoTIFFs under root-in to uint8 and write them to root-out (mirroring the directory structure).

Rules:
- Pixel values == -32768, == 32768, or == 255 are mapped/kept as 255
- All other pixels are treated as valid: output floor(x * 255)
- Valid pixels must land in [0, 255] and be finite after mapping; otherwise fail fast and abort
- Output does not set nodata
- If any directory level in the path matches --skip-dirname, the file is not converted; it is copied to the output tree as-is
- Non-tif/tiff files are copied through by default to preserve the output tree structure

Examples:
- dry-run:
  python newset_processing/convert_to_uint8_gdal.py --root-in /path/in --root-out /path/out --dry-run
- Skip directory names:
  python newset_processing/convert_to_uint8_gdal.py --root-in /path/in --root-out /path/out --skip-dirname trash tmp
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import FIRST_COMPLETED, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    from osgeo import gdal
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Cannot import osgeo.gdal. Please install GDAL Python bindings first (e.g. conda install gdal / apt install python3-gdal).\n"
        f"Original error: {e}"
    )

# Avoid GDAL 4.0 behavior-change notices flooding multi-process logs; also explicitly enable exceptions so errors can be caught.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"Neither gdal\.UseExceptions\(\) nor gdal\.DontUseExceptions\(\) has been explicitly called\.",
        category=FutureWarning,
    )
    try:
        gdal.UseExceptions()
    except Exception:
        # Compatibility with very old bindings; does not affect core functionality
        pass


LOG = logging.getLogger("convert_to_uint8_gdal")


SPECIAL_A = -32768.0
SPECIAL_B = 32768.0
SPECIAL_C = 255.0


@dataclass(frozen=True)
class PlanItem:
    src: Path
    dst: Path
    action: str  # "convert" | "copy" | "skip"


def _is_tif(path: Path) -> bool:
    suf = path.suffix.lower()
    return suf in (".tif", ".tiff")


def _is_skipped_by_dirname(rel_parts: Tuple[str, ...], skip_dirnames: Sequence[str]) -> bool:
    if not skip_dirnames:
        return False
    s = set(skip_dirnames)
    return any(p in s for p in rel_parts[:-1])  # directory parts only


def _atomic_copy2(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _should_skip_existing(dst: Path) -> bool:
    try:
        return dst.exists() and dst.stat().st_size > 0
    except Exception:
        return False


def _index_existing_nonempty_files(
    root_out: Path,
    *,
    progress_every: int = 20000,
    progress_cb=None,
) -> set[Path]:
    """Build an in-memory index of existing non-empty outputs for fast skip checks."""
    existing: set[Path] = set()
    scanned = 0
    root_out = root_out.resolve()
    if not root_out.exists():
        return existing

    stack = [root_out]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        scanned += 1
                        st = entry.stat(follow_symlinks=False)
                        if st.st_size > 0:
                            existing.add(Path(entry.path).relative_to(root_out))
                        if progress_cb and progress_every > 0 and scanned % progress_every == 0:
                            progress_cb(scanned)
                    except OSError:
                        continue
        except OSError:
            continue
    return existing


def iter_plan_items(
    *,
    root_in: Path,
    root_out: Path,
    skip_dirnames: Sequence[str],
    skip_existing: bool,
    existing_relpaths: Optional[set[Path]] = None,
) -> Iterable[PlanItem]:
    """
    Streamingly iterate the input directory and yield PlanItems, avoiding the long stall caused by an upfront rglob + full-list construction.

    Key points:
    - Use os.walk so we can prune in place (when skip_dirname matches, skip the subtree entirely)
    - Yield while iterating so the main flow can submit tasks while scanning is still in progress
    """
    root_in = root_in.resolve()
    root_out = root_out.resolve()
    skip_set = set(skip_dirnames or [])

    # topdown=True lets us modify dirnames in place to prune
    for dirpath, dirnames, filenames in os.walk(root_in, topdown=True):
        if skip_set:
            dirnames[:] = [d for d in dirnames if d not in skip_set]

        dir_path = Path(dirpath)
        for fn in filenames:
            src = dir_path / fn
            # os.walk's filenames are practically always file names, but check once just in case
            try:
                if not src.is_file():
                    continue
            except Exception:
                continue

            rel = src.relative_to(root_in)
            dst = root_out / rel

            if skip_existing and existing_relpaths is not None:
                if rel in existing_relpaths:
                    yield PlanItem(src=src, dst=dst, action="skip")
                    continue
            elif skip_existing and _should_skip_existing(dst):
                yield PlanItem(src=src, dst=dst, action="skip")
                continue

            if _is_skipped_by_dirname(rel.parts, skip_dirnames):
                yield PlanItem(src=src, dst=dst, action="copy")
                continue

            if _is_tif(src):
                yield PlanItem(src=src, dst=dst, action="convert")
            else:
                yield PlanItem(src=src, dst=dst, action="copy")


def _infer_creation_options(src_ds: "gdal.Dataset") -> list[str]:
    """
    Best-effort inherit some creation options from the input.
    Note: GDAL cannot always reliably read every creation option from an existing file; we only handle the common ones here.
    """
    opts: list[str] = []
    md = src_ds.GetMetadata("IMAGE_STRUCTURE") or {}
    comp = md.get("COMPRESSION")
    if comp:
        opts.append(f"COMPRESS={comp}")
    interleave = md.get("INTERLEAVE")
    if interleave:
        opts.append(f"INTERLEAVE={interleave}")
    # If the input is tiled, try to keep it tiled; if blocksize cannot be read, let GDAL use its default
    if md.get("LAYOUT", "").upper() == "TILED":
        opts.append("TILED=YES")
    return opts


def _pick_block_size(band: "gdal.Band") -> Tuple[int, int]:
    try:
        bx, by = band.GetBlockSize()
        if bx and by and bx > 0 and by > 0:
            return int(bx), int(by)
    except Exception:
        pass
    return 1024, 1024


def _invalid_or_special_mask(a: np.ndarray) -> np.ndarray:
    """Treat NaN/Inf the same as reserved sentinel values."""
    return (~np.isfinite(a)) | (a == SPECIAL_A) | (a == SPECIAL_B) | (a == SPECIAL_C)


def _validate_scaled_block(
    *,
    src_path: Path,
    band_index_1based: int,
    window: Tuple[int, int, int, int],
    scaled_valid: np.ndarray,
) -> None:
    if scaled_valid.size == 0:
        return
    finite_mask = np.isfinite(scaled_valid)
    if not bool(finite_mask.all()):
        bad = scaled_valid[~finite_mask]
        example = bad.ravel()[0]
        raise RuntimeError(
            "NaN/Inf detected (not allowed in the valid-value region).\n"
            f"file={src_path}\n"
            f"band={band_index_1based}\n"
            f"window(xoff,yoff,xsize,ysize)={window}\n"
            f"example_bad_value={example}\n"
            "hint: valid values should be finite before multiplying by 255 and should land in [0, 255] after."
        )

    vmin = float(np.min(scaled_valid))
    vmax = float(np.max(scaled_valid))
    if vmin < 0.0 or vmax > 255.0:
        raise RuntimeError(
            "Valid-value mapping overflowed (the scaled result must land in [0, 255]).\n"
            f"file={src_path}\n"
            f"band={band_index_1based}\n"
            f"window(xoff,yoff,xsize,ysize)={window}\n"
            f"scaled_min={vmin}\n"
            f"scaled_max={vmax}\n"
            "hint: this usually means the raw values are outside [0, 1] (or there are negative / outlier values)."
        )


def convert_tif_to_uint8(src_path: Path, dst_path: Path) -> None:
    src_ds = gdal.Open(str(src_path), gdal.GA_ReadOnly)
    if src_ds is None:
        raise RuntimeError(f"Cannot open input tif: {src_path}")

    xsize = src_ds.RasterXSize
    ysize = src_ds.RasterYSize
    nbands = src_ds.RasterCount
    if nbands <= 0:
        raise RuntimeError(f"Input tif has no bands: {src_path}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_name(dst_path.name + ".tmp")
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except Exception:
            pass

    driver = gdal.GetDriverByName("GTiff")
    if driver is None:
        raise RuntimeError("Cannot obtain the GDAL GTiff driver")

    creation_options = _infer_creation_options(src_ds)
    dst_ds = driver.Create(str(tmp_path), xsize, ysize, nbands, gdal.GDT_Byte, options=creation_options)
    if dst_ds is None:
        raise RuntimeError(f"Cannot create output tif: {tmp_path}")

    # GeoTransform / Projection
    gt = src_ds.GetGeoTransform(can_return_null=True)
    if gt is not None:
        dst_ds.SetGeoTransform(gt)
    proj = src_ds.GetProjectionRef()
    if proj:
        dst_ds.SetProjection(proj)

    for b in range(1, nbands + 1):
        src_band = src_ds.GetRasterBand(b)
        dst_band = dst_ds.GetRasterBand(b)
        if src_band is None or dst_band is None:
            raise RuntimeError(f"Cannot get band: file={src_path} band={b}")

        # Inherit some band metadata
        try:
            dst_band.SetColorInterpretation(src_band.GetColorInterpretation())
        except Exception:
            pass
        try:
            desc = src_band.GetDescription()
            if desc:
                dst_band.SetDescription(desc)
        except Exception:
            pass

        bx, by = _pick_block_size(src_band)
        for yoff in range(0, ysize, by):
            ywin = min(by, ysize - yoff)
            for xoff in range(0, xsize, bx):
                xwin = min(bx, xsize - xoff)
                window = (xoff, yoff, xwin, ywin)

                arr = src_band.ReadAsArray(xoff, yoff, xwin, ywin)
                if arr is None:
                    raise RuntimeError(f"ReadAsArray failed: file={src_path} band={b} window={window}")

                # Compute uniformly in float64 to avoid precision / overflow edge instability
                a = np.asarray(arr, dtype=np.float64)
                special = _invalid_or_special_mask(a)
                scaled = np.rint(a * 255.0)
                valid = ~special

                if np.any(valid):
                    _validate_scaled_block(
                        src_path=src_path,
                        band_index_1based=b,
                        window=window,
                        scaled_valid=scaled[valid],
                    )

                out = scaled.astype(np.uint8, copy=False)
                if np.any(special):
                    out = out.copy()  # Copy only when overwriting is necessary
                    out[special] = 255

                dst_band.WriteArray(out, xoff, yoff)

        # Clear nodata (do not set)
        try:
            dst_band.DeleteNoDataValue()
        except Exception:
            # Older GDAL versions may not support this; just leave it unset
            pass

    dst_ds.FlushCache()
    # Close the dataset (ensure the file handle is released)
    dst_ds = None
    src_ds = None

    os.replace(tmp_path, dst_path)


def build_plan(
    *,
    root_in: Path,
    root_out: Path,
    skip_dirnames: Sequence[str],
    skip_existing: bool,
) -> list[PlanItem]:
    # Backward-compatible API: still return a list, but use the faster streaming iteration internally
    return list(
        iter_plan_items(
            root_in=root_in,
            root_out=root_out,
            skip_dirnames=skip_dirnames,
            skip_existing=skip_existing,
        )
    )


def _work_one(item: PlanItem) -> str:
    if item.action == "skip":
        return "skip"
    if item.action == "copy":
        _atomic_copy2(item.src, item.dst)
        return "copy"
    if item.action == "convert":
        convert_tif_to_uint8(item.src, item.dst)
        return "convert"
    raise RuntimeError(f"Unknown action: {item.action}")


def _consume_local_item(item: PlanItem) -> Optional[str]:
    """Handle trivial actions in the parent process to avoid process-pool overhead."""
    if item.action == "skip":
        return "skip"
    return None


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Recursively convert GeoTIFFs to uint8 (GDAL) and mirror the directory structure into a new root directory.")
    ap.add_argument("--root-in", type=Path, default='/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled_5xdownsampled_merged_clipped_unified_scaled', help="Input root directory (recursively traversed)")
    ap.add_argument("--root-out", type=Path, default='/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled_5xdownsampled_merged_clipped_unified_scaled_int', help="Output root directory (preserves the directory structure)")
    ap.add_argument("--skip-dirname", nargs="*", default=['MCD12Q1_mosaic_downsampled_right_clipped_unified_filled', 'MCD14A1_combined_5xMaximumDown_merged_clipped_binary_right', 'MCD14A1_combined_5xMaximumDown_merged_clipped_binary_right_aligned'], help="Skip by directory name (matching subtrees have their files copied through unchanged)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1)), help="Number of parallel processes (file-level parallelism)")
    ap.add_argument("--skip-existing", action="store_true", help="Skip if the output already exists and is non-empty")
    ap.add_argument("--dry-run", action="store_true", help="Count only; do not write files")
    ap.add_argument("--verbose", action="store_true", help="More verbose logging")
    args = ap.parse_args(argv)

    _configure_logging(args.verbose)

    root_in: Path = args.root_in
    root_out: Path = args.root_out

    if not root_in.exists():
        LOG.error("Input root directory does not exist: %s", root_in)
        return 2
    if not root_in.is_dir():
        LOG.error("root-in is not a directory: %s", root_in)
        return 2

    root_out.mkdir(parents=True, exist_ok=True)

    skip_dirnames = [str(x) for x in (args.skip_dirname or []) if str(x)]
    if skip_dirnames:
        LOG.info("skip-dirname: %s", skip_dirnames)

    LOG.info("Scanning input directory (recursive): %s", root_in)
    existing_relpaths: Optional[set[Path]] = None
    if args.skip_existing:
        LOG.info("Pre-scanning the output directory to accelerate skip-existing: %s", root_out)
        existing_relpaths = _index_existing_nonempty_files(
            root_out,
            progress_every=20000,
            progress_cb=lambda n: LOG.info("Output index scan progress: scanned=%d", n),
        )
        LOG.info("Output index complete: %d non-empty files", len(existing_relpaths))
    plan_iter = iter_plan_items(
        root_in=root_in,
        root_out=root_out,
        skip_dirnames=skip_dirnames,
        skip_existing=args.skip_existing,
        existing_relpaths=existing_relpaths,
    )

    # dry-run: count only, do not write files; still use streaming iteration to avoid long stretches of no output
    if args.dry_run:
        n_convert = n_copy = n_skip = total = 0
        for it in plan_iter:
            total += 1
            if it.action == "convert":
                n_convert += 1
            elif it.action == "copy":
                n_copy += 1
            elif it.action == "skip":
                n_skip += 1
            if total == 1 or total % 20000 == 0:
                LOG.info("Scan progress: total=%d (convert=%d copy=%d skip=%d)", total, n_convert, n_copy, n_skip)
        LOG.info("Plan: convert=%d copy=%d skip_existing=%d total=%d", n_convert, n_copy, n_skip, total)
        return 0

    converted = copied = skipped = 0

    workers = int(args.workers)
    if workers < 1:
        workers = 1

    # fail-fast: if any child task raises -> the main process cancels remaining tasks and exits
    with ProcessPoolExecutor(max_workers=workers) as ex:
        in_flight: set = set()
        max_in_flight = max(64, workers * 8)
        submitted = 0
        done = 0
        try:
            # Submit while iterating: avoid waiting for the full scan to finish before starting processing
            for it in plan_iter:
                submitted += 1
                local_res = _consume_local_item(it)
                if local_res is not None:
                    done += 1
                    if local_res == "skip":
                        skipped += 1
                    elif local_res == "copy":
                        copied += 1
                    elif local_res == "convert":
                        converted += 1
                    if done == 1 or done % 200 == 0:
                        LOG.info(
                            "Progress: done=%d submitted=%d in_flight=%d (convert=%d copy=%d skip=%d)",
                            done,
                            submitted,
                            len(in_flight),
                            converted,
                            copied,
                            skipped,
                        )
                    continue
                in_flight.add(ex.submit(_work_one, it))

                # Cap the concurrent queue length to keep memory usage bounded as file count grows
                if len(in_flight) >= max_in_flight:
                    done_set, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                    for fut in done_set:
                        res = fut.result()  # may raise
                        done += 1
                        if res == "convert":
                            converted += 1
                        elif res == "copy":
                            copied += 1
                        elif res == "skip":
                            skipped += 1
                        if done == 1 or done % 200 == 0:
                            LOG.info(
                                "Progress: done=%d submitted=%d in_flight=%d (convert=%d copy=%d skip=%d)",
                                done,
                                submitted,
                                len(in_flight),
                                converted,
                                copied,
                                skipped,
                            )

            # Wrap up: wait for remaining tasks to finish
            while in_flight:
                done_set, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done_set:
                    res = fut.result()  # may raise
                    done += 1
                    if res == "convert":
                        converted += 1
                    elif res == "copy":
                        copied += 1
                    elif res == "skip":
                        skipped += 1
                    if done == 1 or done % 200 == 0 or not in_flight:
                        LOG.info(
                            "Progress: done=%d submitted=%d in_flight=%d (convert=%d copy=%d skip=%d)",
                            done,
                            submitted,
                            len(in_flight),
                            converted,
                            copied,
                            skipped,
                        )
        except Exception:
            # Best-effort cancel of unfinished tasks
            for f in list(in_flight):
                f.cancel()
            raise

    LOG.info("Done: convert=%d copy=%d skip=%d total_done=%d total_submitted=%d", converted, copied, skipped, done, submitted)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)

