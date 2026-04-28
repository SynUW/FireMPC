#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MOD11A1 batch low-quality pixel masking (6 bands: LST_Day, LST_Night, Emis_31, Emis_32, QC_Day, QC_Night).

Default strategy (conservative): "mask only especially-low quality; keep medium quality".
- Mandatory QA (bits 0-1): {2, 3} -> mask
- LST error (bits 6-7): 3 (> 3 K) -> mask; 0/1/2 (<= 1/2/3 K) -> keep
- Emissivity error (bits 4-5): 3 (> 0.04) -> mask; 0/1/2 (<= 0.01/0.02/0.04) -> keep
- Data quality (bits 2-3): 0 = good, 1 = other -> keep; 2/3 = TBD -> mask (can be relaxed via flags)

Output:
- Band 1 LST_Day_1km: masked by QC_Day
- Band 2 LST_Night_1km: masked by QC_Night
- Band 3 Emis_31: masked when the corresponding QC (Day or Night, see --emis-qc-mode) flags "especially low quality"
- Band 4 Emis_32: same as above
- Band 5 QC_Day and Band 6 QC_Night are written back as-is.

Notes:
- Reuses the source file's nodata where possible; if missing, uses -32768 for integers and NaN for floats.
- Compression / block size / BigTIFF and other output options are configurable.
"""

import sys
import argparse
from pathlib import Path
import warnings

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from tqdm import tqdm

warnings.simplefilter("ignore", NotGeoreferencedWarning)

# ---------- Bit-operation utilities ----------
def bitslice(arr, start, end):
    """Extract inclusive bit slice [start, end]."""
    width = end - start + 1
    mask = (1 << width) - 1
    return (arr >> start) & mask

# ---------- QC field parsing ----------
def qc_fields(qc_arr):
    """
    Parse MOD11A1 QC_Day / QC_Night bit fields into a dict:
    0-1: mandatory
    2-3: data_quality
    4-5: emis_error
    6-7: lst_error
    """
    return dict(
        mandatory = bitslice(qc_arr, 0, 1),
        data_quality = bitslice(qc_arr, 2, 3),
        emis_error = bitslice(qc_arr, 4, 5),
        lst_error = bitslice(qc_arr, 6, 7),
    )

# ---------- Mask criteria ----------
def build_bad_mask_from_qc(
    qc,
    mask_if_mandatory_23=True,
    mask_if_dq_tbd=True,
    dq_keep_1=True,
    max_keep_lst_error=2,     # 0..2 keep; 3 (> 3 K) mask
    max_keep_emis_error=2,    # 0..2 keep; 3 (> 0.04) mask
):
    """
    Returns a boolean bad_mask: True means "especially low quality and should be masked".
    """
    q = qc_fields(qc)

    bad = np.zeros(qc.shape, dtype=bool)

    # 1) mandatory QA: 2/3 -> always mask (not produced)
    if mask_if_mandatory_23:
        bad |= np.isin(q["mandatory"], [2, 3])

    # 2) data quality: 0=good, 1=other quality (keep), 2/3=TBD -> mask by default
    if mask_if_dq_tbd:
        bad |= np.isin(q["data_quality"], [2, 3])
    if not dq_keep_1:
        # For stricter filtering, you can also mask data_quality==1
        bad |= (q["data_quality"] == 1)

    # 3) LST error: > 3 K (value=3) -> mask; <= 3 K (0/1/2) -> keep
    bad |= (q["lst_error"] > max_keep_lst_error)

    # 4) Emissivity error: > 0.04 (value=3) -> mask; <= 0.04 (0/1/2) -> keep
    bad |= (q["emis_error"] > max_keep_emis_error)

    return bad

def pick_nodata(dtype, existing_nodata):
    if existing_nodata is not None:
        return existing_nodata
    if np.issubdtype(dtype, np.floating):
        return np.nan
    return np.int64(-32768)

# ---------- Core processing ----------
def process_one(path_in: Path, path_out: Path, args) -> None:
    with rasterio.open(path_in) as src:
        if src.count < 6:
            raise ValueError(f"{path_in.name}: 6 bands required (LST_Day, LST_Night, Emis_31, Emis_32, QC_Day, QC_Night).")

        # Read band
        lst_day  = src.read(1)
        lst_nite = src.read(2)
        emis31   = src.read(3)
        emis32   = src.read(4)
        qc_day   = src.read(5)
        qc_nite  = src.read(6)

        # Build the "especially-low-quality" mask from QC
        bad_day  = build_bad_mask_from_qc(
            qc_day,
            mask_if_mandatory_23=not args.allow_mandatory_23,
            mask_if_dq_tbd=not args.allow_dq_tbd,
            dq_keep_1=not args.mask_dq1,
            max_keep_lst_error=args.max_keep_lst_error,
            max_keep_emis_error=args.max_keep_emis_error,
        )
        bad_nite = build_bad_mask_from_qc(
            qc_nite,
            mask_if_mandatory_23=not args.allow_mandatory_23,
            mask_if_dq_tbd=not args.allow_dq_tbd,
            dq_keep_1=not args.mask_dq1,
            max_keep_lst_error=args.max_keep_lst_error,
            max_keep_emis_error=args.max_keep_emis_error,
        )

        profile = src.profile.copy()
        profile.update(
            compress=args.compress,
            tiled=True,
            blockxsize=min(profile.get("blockxsize", 512), 1024),
            blockysize=min(profile.get("blockysize", 512), 1024),
            bigtiff="IF_SAFER",
        )

        # Per-band nodata (inferred if not set in source)
        band_nd = []
        for b in range(1, src.count + 1):
            nd = src.nodatavals[b-1]
            if nd is None:
                nd = pick_nodata(np.dtype(src.dtypes[b-1]), None)
            band_nd.append(nd)

        # Write out
        path_out.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path_out, "w", **profile) as dst:
            # 1) LST_Day: apply NoData based on bad_day
            arr = lst_day.copy()
            nd = band_nd[0]
            if np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float32, copy=False)
                arr[bad_day] = np.nan if (isinstance(nd, float) and np.isnan(nd)) else nd
            else:
                arr[bad_day] = np.int64(nd)
            dst.write(arr, 1)

            # 2) LST_Night: apply NoData based on bad_nite
            arr = lst_nite.copy()
            nd = band_nd[1]
            if np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float32, copy=False)
                arr[bad_nite] = np.nan if (isinstance(nd, float) and np.isnan(nd)) else nd
            else:
                arr[bad_nite] = np.int64(nd)
            dst.write(arr, 2)

            # 3/4) Emis_31 / Emis_32: per --emis-qc-mode
            if args.emis_qc_mode == "day":
                bad_emis = bad_day
            elif args.emis_qc_mode == "night":
                bad_emis = bad_nite
            else:  # "union" (default): mask if either day or night is severely bad
                bad_emis = (bad_day | bad_nite)

            arr = emis31.copy()
            nd = band_nd[2]
            if np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float32, copy=False)
                arr[bad_emis] = np.nan if (isinstance(nd, float) and np.isnan(nd)) else nd
            else:
                arr[bad_emis] = np.int64(nd)
            dst.write(arr, 3)

            arr = emis32.copy()
            nd = band_nd[3]
            if np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float32, copy=False)
                arr[bad_emis] = np.nan if (isinstance(nd, float) and np.isnan(nd)) else nd
            else:
                arr[bad_emis] = np.int64(nd)
            dst.write(arr, 4)

            # 5/6) Write QA back as-is
            dst.write(qc_day, 5)
            dst.write(qc_nite, 6)

            # If the source did not set nodata uniformly, write per-band tags for compatibility
            if profile.get("nodata", None) is None:
                for i, ndv in enumerate(band_nd, start=1):
                    if not (isinstance(ndv, float) and np.isnan(ndv)):
                        dst.update_tags(i, nodata=str(ndv))

def main():
    ap = argparse.ArgumentParser(
        description="Mask out especially-low-quality MOD11A1 pixels (keep medium quality)."
    )
    ap.add_argument("--input_dir", type=Path, required=True, help="Input directory (containing 6-band MOD11A1 GeoTIFFs)")
    ap.add_argument("--output_dir", type=Path, required=True, help="Output directory")
    ap.add_argument("--suffix", default="", help="Optional suffix appended to output filenames")
    ap.add_argument("--compress", default="LZW", help="Output compression, default LZW")

    # Loosen / tighten options (default conservatively masks "especially low quality")
    ap.add_argument("--allow-mandatory-23", action="store_true",
                    help="Loosen: do not mask due to mandatory=2/3 (not produced) (generally not recommended)")
    ap.add_argument("--allow-dq-tbd", action="store_true",
                    help="Loosen: do not mask DataQuality=2/3 (TBD)")
    ap.add_argument("--mask-dq1", action="store_true",
                    help="Tighten: also mask DataQuality=1 (Other quality) (not masked by default to retain medium quality)")
    ap.add_argument("--max-keep-lst-error", type=int, default=3, choices=[0,1,2,3],
                    help="Maximum LST error level allowed to keep (0..3); default 2 (<=3 K keep, >3 K mask)")
    ap.add_argument("--max-keep-emis-error", type=int, default=3, choices=[0,1,2,3],
                    help="Maximum emissivity error level allowed to keep (0..3); default 2 (<=0.04 keep, >0.04 mask)")

    ap.add_argument("--emis-qc-mode", choices=["union", "day", "night"], default="union",
                    help="Which QC to apply to emissivity bands: union = mask if either day or night is severely bad (default); day = QC_Day only; night = QC_Night only")

    ap.add_argument("--overwrite", action="store_true", help="Overwrite if it already exists")
    args = ap.parse_args()

    if not args.input_dir.exists():
        print(f"Input directory does not exist: {args.input_dir}", file=sys.stderr)
        sys.exit(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tifs = sorted([p for p in args.input_dir.iterdir() if p.suffix.lower() in (".tif", ".tiff")])
    if not tifs:
        print("No TIF files found in the input directory.", file=sys.stderr)
        sys.exit(1)

    ok, fail = 0, 0
    for src in tqdm(tifs, desc="Processing"):
        dst = args.output_dir / (src.stem + args.suffix + src.suffix)
        if dst.exists() and not args.overwrite:
            ok += 1
            continue
        try:
            process_one(src, dst, args)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"[FAILED] {src.name}: {e}", file=sys.stderr)

    print(f"Done. OK={ok}, FAILED={fail}")

if __name__ == "__main__":
    main()
