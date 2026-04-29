import argparse
import glob
import json
import os
import sys
from typing import Any

import rasterio


def _is_positive(sample: dict[str, Any]) -> bool:
    st = sample.get("sample_type")
    if st is None:
        return bool(sample.get("firms_value", 0))
    return str(st).lower() == "positive"


def _is_negative(sample: dict[str, Any]) -> bool:
    st = sample.get("sample_type")
    if st is None:
        return not bool(sample.get("firms_value", 0))
    return str(st).lower() == "negative"


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    parser = argparse.ArgumentParser(
        description=(
            "Re-select samples based on MCD14A1 (fire detection) daily GeoTIFF values. "
            "Each sample uses (date, pixel_row, pixel_col) to lookup the MCD14A1 value. "
            "Positive samples must match pos-class; negative samples must match neg-class. "
            "Writes results to a new folder (no in-place modification)."
        )
    )
    parser.add_argument(
        "--json-dir",
        default="/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/sampling/sampling_calibrated",
        help="Input directory containing JSON files.",
    )
    parser.add_argument(
        "--mcd14a1-dir",
        default="/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled_5xdownsampled_merged_clipped/MCD14A1_combined_5xMaximumDown_merged_clipped_aligned",
        help="Directory containing daily MCD14A1 GeoTIFFs named like YYYY_MM_DD.tif.",
    )
    parser.add_argument(
        "--out-dir",
        default="/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/sampling/sampling_calibrated_selected",
        help="Output directory to write filtered JSON files into.",
    )
    parser.add_argument("--pos-class", type=int, default=9, help="Required mask value for positive samples.")
    parser.add_argument("--neg-class", type=int, default=5, help="Required mask value for negative samples.")
    parser.add_argument(
        "--copy-statistics",
        action="store_true",
        help="Copy files whose name contains 'statistics' into out-dir unchanged.",
    )
    parser.add_argument(
        "--missing-policy",
        choices=["drop", "keep"],
        default="drop",
        help="When the corresponding tif is missing / unreadable or (row,col) is out-of-bounds.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="Print progress every N unique dates processed within each JSON file.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print stats only; do not write files.")
    args = parser.parse_args()

    json_dir = args.json_dir.rstrip("/")
    mcd_dir = args.mcd14a1_dir.rstrip("/")
    out_dir = args.out_dir.rstrip("/")
    if not os.path.isdir(json_dir):
        print(f"[error] json-dir not found: {json_dir}", file=sys.stderr)
        return 2
    if not os.path.isdir(mcd_dir):
        print(f"[error] mcd14a1-dir not found: {mcd_dir}", file=sys.stderr)
        return 2
    if not args.dry_run:
        os.makedirs(out_dir, exist_ok=True)

    pattern = os.path.join(json_dir, "*.json")
    fnames = sorted(glob.glob(pattern))
    print(f"[info] glob pattern: {pattern}", flush=True)
    print(f"[info] matched files: {len(fnames)}", flush=True)
    print(f"[info] out-dir: {out_dir}{' [dry-run]' if args.dry_run else ''}", flush=True)
    print(f"[info] mcd14a1-dir: {mcd_dir}", flush=True)
    print(
        f"[info] pos-class={args.pos_class}, neg-class={args.neg_class}, missing-policy={args.missing_policy}",
        flush=True,
    )

    total_in = total_out = 0
    total_drop_pos = total_drop_neg = total_drop_other = 0
    total_drop_missing = 0
    copied_stats = 0

    for fname in fnames:
        base = os.path.basename(fname)
        if "statistics" in base and args.copy_statistics:
            copied_stats += 1
            if not args.dry_run:
                with open(fname, "rb") as r, open(os.path.join(out_dir, base), "wb") as w:
                    w.write(r.read())
            print(f"copied: {base}{' [dry-run]' if args.dry_run else ''}")
            continue

        with open(fname, "r") as f:
            payload = json.load(f)
        samples = payload.get("samples", [])
        n_in = len(samples)

        # Pre-compute MCD14A1 values for each sample by grouping per date.
        values: list[int | None] = [None] * n_in
        by_date: dict[str, list[int]] = {}
        for i, s in enumerate(samples):
            date_str = s.get("date")
            if isinstance(date_str, str) and date_str:
                by_date.setdefault(date_str, []).append(i)
            else:
                by_date.setdefault("", []).append(i)

        date_items = [(d, idxs) for d, idxs in by_date.items() if d]
        n_dates = len(date_items)
        for j, (date_str, idxs) in enumerate(date_items, start=1):
            if args.progress_every > 0 and (j == 1 or j % args.progress_every == 0 or j == n_dates):
                print(f"[info] {base}: reading MCD14A1 dates {j}/{n_dates}", flush=True)
            if not date_str:
                continue
            tif_name = date_str.replace("-", "_") + ".tif"
            tif_path = os.path.join(mcd_dir, tif_name)
            if not os.path.exists(tif_path):
                continue
            try:
                with rasterio.open(tif_path) as ds:
                    arr = ds.read(1)
                    h, w = arr.shape
                    for i in idxs:
                        s = samples[i]
                        r = s.get("pixel_row")
                        c = s.get("pixel_col")
                        if isinstance(r, int) and isinstance(c, int) and 0 <= r < h and 0 <= c < w:
                            values[i] = int(arr[r, c])
            except Exception:
                continue

        kept: list[dict[str, Any]] = []
        drop_pos = drop_neg = drop_other = 0
        drop_missing = 0
        for i, s in enumerate(samples):
            v = values[i]
            if v is None and args.missing_policy == "keep":
                kept.append(s)
                continue

            if v is None:
                drop_missing += 1
                continue

            if _is_positive(s):
                if v == args.pos_class:
                    kept.append(s)
                else:
                    drop_pos += 1
            elif _is_negative(s):
                if v == args.neg_class:
                    kept.append(s)
                else:
                    drop_neg += 1
            else:
                # Unknown sample_type; keep strict by dropping
                drop_other += 1

        n_out = len(kept)
        total_in += n_in
        total_out += n_out
        total_drop_pos += drop_pos
        total_drop_neg += drop_neg
        total_drop_other += drop_other
        total_drop_missing += drop_missing

        payload["samples"] = kept
        if "total_samples" in payload:
            payload["total_samples"] = n_out

        if not args.dry_run:
            out_path = os.path.join(out_dir, base)
            with open(out_path, "w") as f:
                json.dump(payload, f, separators=(",", ":"))

        print(
            f"filtered: {base} in={n_in} out={n_out} "
            f"drop_pos={drop_pos} drop_neg={drop_neg} drop_missing={drop_missing} drop_other={drop_other}"
            f"{' [dry-run]' if args.dry_run else ''}"
        )

    print(
        "[done] "
        f"files={len(fnames)} copied_stats={copied_stats} "
        f"samples_in={total_in} samples_out={total_out} "
        f"drop_pos={total_drop_pos} drop_neg={total_drop_neg} drop_missing={total_drop_missing} drop_other={total_drop_other}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

