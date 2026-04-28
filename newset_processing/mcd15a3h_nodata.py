#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Replace "invalid values" in a raster with 0 and do NOT set nodata in the output (nodata=None).

Examples:
  python mcd15a3h_nodata.py --input-dir in_folder --output-dir out_folder --invalid -32768
  python mcd15a3h_nodata.py --input-dir in_folder --output-dir out_folder --recursive --invalid -32768 32768 255

Notes:
- Only numeric replacement: pixels equal to any value in the invalid list are uniformly changed to 0.
- Output files do not set nodata metadata; no internal mask is written to indicate invalid regions either.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import rasterio
from tqdm import tqdm


def _as_list(xs: Iterable[float]) -> List[float]:
    return list(xs)


def list_tifs(root: Path, recursive: bool) -> List[Path]:
    if recursive:
        return sorted(list(root.rglob("*.tif")) + list(root.rglob("*.TIF")))
    return sorted(list(root.glob("*.tif")) + list(root.glob("*.TIF")))


def replace_invalid_to_zero(
    input_path: Path,
    output_path: Path,
    invalid_values: List[float],
    quiet: bool = False,
) -> None:
    if not invalid_values:
        raise ValueError("--invalid requires at least one value")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        profile.update(
            nodata=None,  # Do not set nodata
        )

        # Some profiles may also carry mask/alpha; do not force-change them, keep as-is and just ensure nodata=None
        invalid_arr = np.asarray(invalid_values, dtype=np.float64)

        with rasterio.open(output_path, "w", **profile) as dst:
            for bidx in range(1, src.count + 1):
                for _, window in src.block_windows(bidx):
                    data = src.read(bidx, window=window, masked=False)
                    if data.size == 0:
                        continue

                    # Compare via float64 to avoid dtype issues (especially int16 vs Python float)
                    # Then write back in the original dtype
                    data_f = data.astype(np.float64, copy=False)
                    mask = np.isin(data_f, invalid_arr)
                    if mask.any():
                        data = data.copy()
                        data[mask] = 0

                    dst.write(data, bidx, window=window)

    if not quiet:
        print(f"[OK] wrote -> {output_path}（invalid={invalid_values} replaced with 0; nodata=None)")


def process_folder(
    input_dir: Path,
    output_dir: Path,
    invalid_values: List[float],
    recursive: bool,
    quiet: bool,
    skip_existing: bool,
) -> None:
    if not input_dir.exists() or not input_dir.is_dir():
        raise RuntimeError(f"Input directory does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tifs = list_tifs(input_dir, recursive=recursive)
    if not tifs:
        raise RuntimeError(f"No tifs found in the directory: {input_dir}")

    it = tifs
    if not quiet:
        it = tqdm(tifs, desc="Fix nodata->0", unit="file", ncols=100)

    for in_path in it:
        rel = in_path.relative_to(input_dir)
        out_path = output_dir / rel
        if skip_existing and out_path.exists():
            continue
        replace_invalid_to_zero(
            input_path=in_path,
            output_path=out_path,
            invalid_values=invalid_values,
            quiet=True,  # Do not flood the log per file
        )

    if not quiet:
        print(f"[OK] done {len(tifs)} files -> {output_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Replace invalid values with 0 and output GeoTIFFs with no nodata set")
    p.add_argument(
        "--input-dir",
        default='/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled_5xdownsampled_merged_clipped_unified/MCD15A3H_QAapplied_spatialFilled_downsampled_merged_clipped_unified',
        help="Input directory (contains tif / TIF)",
    )
    p.add_argument(
        "--output-dir",
        default='/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled_5xdownsampled_merged_clipped_unified/MCD15A3H_QAapplied_spatialFilled_downsampled_merged_clipped_unified_rightnodata',
        help="Output directory (preserves relative paths and filenames)",
    )
    p.add_argument("--recursive", action="store_true", help="Process subdirectories recursively")
    p.add_argument(
        "--invalid",
        type=float,
        nargs="+",
        default=[-32768],
        help="List of invalid values to replace with 0; default: -32768",
    )
    p.add_argument("--quiet", action="store_true", help="Do not print completion messages")
    p.add_argument("--skip-existing", action="store_true", help="Skip if output already exists")
    args = p.parse_args()

    process_folder(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        invalid_values=_as_list(args.invalid),
        recursive=bool(args.recursive),
        quiet=bool(args.quiet),
        skip_existing=bool(args.skip_existing),
    )


if __name__ == "__main__":
    main()

