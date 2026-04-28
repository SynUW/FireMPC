#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCD14A1 (MODIS fire-mask classified raster) non-overlapping block downsampling: method 2 "custom priority".

Within each factor x factor window, pick **one** output pixel by priority (not mode, not average):
  Fire 9 > 8 > 7 > Cloud 4 > Non-fire land 5 > Non-fire water 3 > Unknown 6 > Not processed 2, 1

Pixel values not in the table (including common NoData) are assigned the lowest priority, and are only selected when the entire window consists of such values.

By default, **only band 1 is processed** (MODIS/MCD14A1 fire-mask band 0); the built-in priority table is only meaningful for that band. For multiple bands, pass `--bands` or `--all-bands` explicitly.

The input size is cropped to a multiple of factor (extra pixels on the right / bottom are discarded), matching typical block aggregation.
Output geotransform: transform *= Affine.scale(factor, factor); width and height are divisible by factor.

Performance notes (why "simple" can still be slow):
  - A whole-image `read` + priority array + `transpose/reshape` typically introduces 1-2 additional buffers of the same size as the input (mostly memory-bandwidth bound).
  - High-level ZSTD (especially 9) is CPU-heavy; use `--zstd-level` / `--compress NONE` to trade for speed.
  - For multiple files, use `-j`; single-file bottleneck is decompression on read, numpy, and compression on write.

Dependencies: numpy, rasterio, affine; tqdm optional
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rasterio
from affine import Affine

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("downsample_mcd14a1")

# Smaller number = higher priority (consistent with argmin)
# Fire (9>8>7) > Cloud (4) > Non-fire land (5) > Non-fire water (3) > Unknown (6) > Not processed (2, 1)
FIRE_MASK_PRIORITY: dict[int, int] = {
    9: 0,
    8: 1,
    7: 2,
    4: 3,
    5: 4,
    3: 5,
    6: 6,
    2: 7,
    1: 8,
}

DEFAULT_LUT_SIZE = 65536


def build_priority_lut(
    priority: dict[int, int] | None = None,
    lut_size: int = DEFAULT_LUT_SIZE,
    default_rank: int = 10_000,
) -> np.ndarray:
    """Lookup table over the range [0, lut_size); undefined classes get default_rank (lowest priority)."""
    priority = priority or FIRE_MASK_PRIORITY
    lut = np.full(lut_size, default_rank, dtype=np.int32)
    for k, r in priority.items():
        if 0 <= k < lut_size:
            lut[k] = r
    return lut


def downsample_fire_priority(
    arr: np.ndarray,
    factor: int,
    lut: np.ndarray,
    *,
    default_rank: int = 10_000,
    nodata_values: Optional[tuple[Any, ...]] = None,
) -> np.ndarray:
    """
    Non-overlapping factor window: take the raw class value by priority argmin.
    Pixels in nodata_values are forced to default_rank (do not compete unless the entire window is at the lowest tier).
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if factor == 1:
        return np.asarray(arr, copy=True)

    arr = np.asarray(arr)
    H, W = arr.shape
    Hn, Wn = H // factor, W // factor
    if Hn < 1 or Wn < 1:
        raise ValueError(f"Image {H}x{W} is smaller than factor={factor}; cannot downsample")

    sl = (slice(0, Hn * factor), slice(0, Wn * factor))
    arr = arr[sl]
    # Avoid unnecessary whole-image int64 copies (would explode memory and bandwidth for uint8/uint16 inputs)
    a = np.asarray(arr)
    pr_dtype = np.int32 if default_rank > 32767 else np.int16
    prio = np.full(a.shape, default_rank, dtype=pr_dtype)
    inside = (a >= 0) & (a < lut.size)
    prio[inside] = lut[a[inside].astype(np.intp, copy=False)]
    # Must overwrite after the lookup; otherwise NoData might share a numeric value with a class and be given high priority
    if nodata_values:
        for nd in nodata_values:
            if nd is None:
                continue
            if isinstance(nd, float) and np.isnan(nd):
                prio[np.isnan(a.astype(np.float64, copy=False))] = default_rank
            else:
                prio[a == int(nd)] = default_rank

    # Equivalent to (Hn, f, Wn, f).transpose(0, 2, 1, 3), usually saves one Python-level transpose call
    prio_blocks = prio.reshape(Hn, factor, Wn, factor).swapaxes(1, 2).reshape(Hn, Wn, -1)
    val_blocks = arr.reshape(Hn, factor, Wn, factor).swapaxes(1, 2).reshape(Hn, Wn, -1)

    idx = np.argmin(prio_blocks, axis=-1)
    out = np.take_along_axis(val_blocks, idx[..., None], axis=-1).squeeze(-1)

    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        out = np.clip(out, info.min, info.max).astype(arr.dtype, copy=False)
    else:
        out = out.astype(arr.dtype, copy=False)
    return out


def _parse_bands(s: Optional[str]) -> Optional[list[int]]:
    """
    None means process all bands; a list contains 1-based band numbers.
    Special: the string 'all' (case-insensitive) -> None.
    """
    if s is None or not str(s).strip():
        return [1]
    t = str(s).strip()
    if t.lower() == "all":
        return None
    out: list[int] = []
    for part in t.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out if out else [1]


def _apply_gdal_safe_blocksize(profile: dict, width: int, height: int, blocksize_req: int) -> None:
    """
    For ZSTD/LZW etc., GeoTIFF requires block width and height to be multiples of 16, otherwise it errors:
    "The height and width of dataset blocks must be multiples of 16"
    """
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    bx = (min(int(blocksize_req), int(width)) // 16) * 16
    by = (min(int(blocksize_req), int(height)) // 16) * 16
    if bx >= 16 and by >= 16:
        profile["tiled"] = True
        profile["blockxsize"] = bx
        profile["blockysize"] = by
    else:
        profile["tiled"] = False


def _nodata_tuple(src: Any, band_idx: int) -> tuple[Any, ...]:
    """band_idx is 0-based."""
    nv = getattr(src, "nodatavals", None)
    if nv is not None and band_idx < len(nv) and nv[band_idx] is not None:
        return (nv[band_idx],)
    n = getattr(src, "nodata", None)
    if n is not None:
        return (n,)
    return tuple()


def process_one_tif(
    input_path: Path,
    output_path: Path,
    factor: int,
    bands_1based: Optional[list[int]],
    lut: np.ndarray,
    default_rank: int,
    extra_nodata: tuple[Any, ...],
    compress: str,
    blocksize: int,
    zstd_level: int,
) -> dict[str, Any]:
    meta: dict[str, Any] = {"input": str(input_path), "output": str(output_path)}
    with rasterio.open(input_path) as src:
        count = src.count
        if bands_1based is None:
            band_list = list(range(1, count + 1))
        else:
            band_list = []
            for b in bands_1based:
                if b < 1 or b > count:
                    raise ValueError(f"Band {b} out-of-range (count={count})")
                band_list.append(b)

        profile = src.profile.copy()
        out_blocks: list[np.ndarray] = []
        descs: list[str] = []

        for b1 in band_list:
            b0 = b1 - 1
            arr = src.read(b1)
            nds = _nodata_tuple(src, b0) + tuple(extra_nodata)
            out_b = downsample_fire_priority(
                arr,
                factor,
                lut,
                default_rank=default_rank,
                nodata_values=nds if nds else None,
            )
            out_blocks.append(out_b)
            d = src.descriptions[b0] if b0 < len(src.descriptions) else None
            descs.append(d if d else f"band{b1}")

        stack = np.stack(out_blocks, axis=0)
        h, w = stack.shape[1], stack.shape[2]
        new_transform = src.transform * Affine.scale(factor, factor)

        profile.update(
            dtype=rasterio.dtypes.get_minimum_dtype(stack),
            count=stack.shape[0],
            width=w,
            height=h,
            transform=new_transform,
        )
        _apply_gdal_safe_blocksize(profile, w, h, blocksize)
        if compress and compress.upper() != "NONE":
            profile["compress"] = compress
            if compress.upper() == "ZSTD":
                profile["zstd_level"] = int(zstd_level)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(stack)
            dst.descriptions = tuple(descs)

    meta.update(
        {
            "factor": factor,
            "out_shape": [int(stack.shape[1]), int(stack.shape[2])],
            "bands_written": band_list,
        }
    )
    return meta


def _worker_job(task: dict) -> tuple[str, dict]:
    key = task["key"]
    try:
        lut = task["lut"]
        if not isinstance(lut, np.ndarray):
            lut = np.asarray(lut, dtype=np.int32)
        elif lut.dtype != np.int32:
            lut = np.asarray(lut, dtype=np.int32)
        m = process_one_tif(
            Path(task["input_path"]),
            Path(task["output_path"]),
            factor=int(task["factor"]),
            bands_1based=task.get("bands_1based"),
            lut=lut,
            default_rank=int(task["default_rank"]),
            extra_nodata=tuple(task.get("extra_nodata") or ()),
            compress=str(task["compress"]),
            blocksize=int(task["blocksize"]),
            zstd_level=int(task.get("zstd_level", 3)),
        )
        return key, {"ok": True, "meta": m}
    except Exception as e:
        return key, {"ok": False, "error": str(e)}


def parse_args():
    ap = argparse.ArgumentParser(
        description="MCD14A1 non-overlapping downsampling by fire-mask priority (pick the highest-priority class in each factor x factor window)"
    )
    ap.add_argument("--input-dir", type=Path, default='/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/MCD14A1_combined', help="Input GeoTIFF directory")
    ap.add_argument("--output-dir", type=Path, default='/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled_5xdownsampled/MCD14A1_combined_5xMaximumDown', help="Output directory")
    ap.add_argument("--factor", type=int, default=5, help="Block side length (non-overlapping), default 5")
    ap.add_argument("--pattern", default="*.tif", help="glob, relative to input-dir")
    ap.add_argument("--recursive", action="store_true", help="rglob(pattern)")
    ap.add_argument(
        "--bands",
        type=str,
        default="1",
        help="Bands to process, 1-based and comma-separated (default: just 1, the fire-mask band 0); pass 'all' for every band",
    )
    ap.add_argument(
        "--all-bands",
        action="store_true",
        help="Process all bands of the input (same as --bands all; the priority table is still semantically only meaningful for fire-mask classes)",
    )
    ap.add_argument(
        "--extra-nodata",
        type=str,
        default="",
        help="Additional integer values treated as NoData, comma-separated (combined with the file's nodata into the lowest priority)",
    )
    ap.add_argument(
        "--default-rank",
        type=int,
        default=10_000,
        help="Priority rank for undefined classes and NoData (larger = worse, default 10000)",
    )
    ap.add_argument(
        "--compress",
        default="ZSTD",
        choices=["NONE", "LZW", "DEFLATE", "ZSTD"],
    )
    ap.add_argument(
        "--zstd-level",
        type=int,
        default=3,
        help="ZSTD compression level; 9 is more space-efficient but much slower to write, default 3",
    )
    ap.add_argument(
        "--blocksize",
        type=int,
        default=512,
        help="Target block side length; actually rounded down to a multiple of 16 to satisfy GeoTIFF requirements",
    )
    ap.add_argument("-j", "--jobs", type=int, default=1, help="Number of files to process in parallel")
    ap.add_argument("--skip-existing", action="store_true", help="Skip if output already exists")
    ap.add_argument("--meta-json", type=Path, default=None, help="Write per-file metadata to JSON")
    ap.add_argument(
        "--priority-json",
        type=Path,
        default=None,
        help="Optional: custom priority JSON, format {\"class_value\": rank}; smaller rank = higher priority (overrides the built-in table)",
    )
    args = ap.parse_args()
    if args.all_bands:
        args.bands = "all"
    if args.factor < 1:
        ap.error("--factor must be >= 1")
    if args.jobs < 1:
        ap.error("-j must be >= 1")
    if args.zstd_level < 1 or args.zstd_level > 22:
        ap.error("--zstd-level must be in 1-22")
    return args


def list_tifs(input_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    if recursive:
        files = sorted(input_dir.rglob(pattern))
    else:
        files = sorted(input_dir.glob(pattern))
    return [f for f in files if f.is_file()]


def main():
    args = parse_args()
    in_dir = args.input_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    priority = dict(FIRE_MASK_PRIORITY)
    if args.priority_json is not None:
        custom = json.loads(args.priority_json.read_text(encoding="utf-8"))
        priority = {int(k): int(v) for k, v in custom.items()}

    lut = build_priority_lut(priority, lut_size=DEFAULT_LUT_SIZE, default_rank=args.default_rank)
    bands = _parse_bands(args.bands)

    extra_nd: list[Any] = []
    if args.extra_nodata.strip():
        for p in args.extra_nodata.split(","):
            p = p.strip()
            if p:
                extra_nd.append(int(p))

    files = list_tifs(in_dir, args.pattern, args.recursive)
    if not files:
        logger.warning("No matching TIFs found: %s / %s", in_dir, args.pattern)
        return

    jobs: list[dict] = []
    for fp in files:
        rel = fp.relative_to(in_dir) if fp.is_relative_to(in_dir) else Path(fp.name)
        outp = out_dir / rel
        key = str(rel.as_posix())
        if args.skip_existing and outp.exists():
            continue
        jobs.append(
            {
                "key": key,
                "input_path": str(fp),
                "output_path": str(outp),
                "factor": args.factor,
                "bands_1based": bands,
                "lut": np.ascontiguousarray(lut, dtype=np.int32),
                "default_rank": args.default_rank,
                "extra_nodata": extra_nd,
                "compress": args.compress,
                "blocksize": args.blocksize,
                "zstd_level": args.zstd_level,
            }
        )

    if not jobs:
        logger.info("No files to process (they may all exist; see --skip-existing)")
        return

    all_meta: dict[str, Any] = {}

    if args.jobs == 1:
        it = jobs
        if tqdm is not None:
            it = tqdm(jobs, desc="MCD14A1 priority-based downsampling", unit="file")
        for task in it:
            k, res = _worker_job(task)
            all_meta[k] = res
            if not res.get("ok"):
                logger.error("[%s] %s", k, res.get("error"))
            elif tqdm is None:
                logger.info("[%s] OK -> %s", k, res["meta"]["output"])
    else:
        worker_tasks = jobs
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(_worker_job, t) for t in worker_tasks]
            iter_f = as_completed(futs)
            if tqdm is not None:
                iter_f = tqdm(iter_f, total=len(futs), desc="MCD14A1 priority-based downsampling", unit="file")
            for fut in iter_f:
                k, res = fut.result()
                all_meta[k] = res
                if not res.get("ok"):
                    logger.error("[%s] %s", k, res.get("error"))

    if args.meta_json is not None:
        args.meta_json.parent.mkdir(parents=True, exist_ok=True)
        args.meta_json.write_text(
            json.dumps(all_meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Metadata written to %s", args.meta_json)

    n_ok = sum(1 for v in all_meta.values() if v.get("ok"))
    logger.info("Done: succeeded %d / %d", n_ok, len(all_meta))


if __name__ == "__main__":
    main()
