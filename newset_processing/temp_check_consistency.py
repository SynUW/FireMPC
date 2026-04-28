from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class GeoSignature:
    width: int
    height: int
    crs_wkt: Optional[str]
    transform: tuple[float, float, float, float, float, float]
    bounds: tuple[float, float, float, float]  # left, bottom, right, top

    @property
    def res(self) -> tuple[float, float]:
        # pixel width/height (sign of y is typically negative)
        return (self.transform[0], self.transform[4])


def _iter_subdirs(root: Path) -> Iterable[Path]:
    for p in sorted(root.iterdir()):
        if p.is_dir():
            yield p


def _list_tifs(folder: Path) -> list[Path]:
    # cover .tif and .tiff, case-insensitive
    out: list[Path] = []
    for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        out.extend(folder.glob(ext))
    # de-dup while keeping deterministic ordering
    out = sorted({p.resolve() for p in out})
    return out


def _read_signature(tif_path: Path) -> GeoSignature:
    try:
        import rasterio  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: rasterio is required to read GeoTIFFs."
        ) from e

    with rasterio.open(tif_path) as ds:
        crs_wkt = ds.crs.to_wkt() if ds.crs is not None else None
        t = ds.transform
        transform = (t.a, t.b, t.c, t.d, t.e, t.f)
        b = ds.bounds
        bounds = (b.left, b.bottom, b.right, b.top)
        return GeoSignature(
            width=int(ds.width),
            height=int(ds.height),
            crs_wkt=crs_wkt,
            transform=transform,
            bounds=bounds,
        )


def _diff_signature(a: GeoSignature, b: GeoSignature) -> dict:
    diffs: dict[str, dict] = {}
    if (a.width, a.height) != (b.width, b.height):
        diffs["shape"] = {"ref": [a.height, a.width], "cur": [b.height, b.width]}
    if a.crs_wkt != b.crs_wkt:
        diffs["crs_wkt"] = {"ref": a.crs_wkt, "cur": b.crs_wkt}
    if a.transform != b.transform:
        diffs["transform"] = {"ref": a.transform, "cur": b.transform}
    if a.bounds != b.bounds:
        diffs["bounds"] = {"ref": a.bounds, "cur": b.bounds}
    return diffs


def check_folder(folder: Path, sample_n: int) -> dict:
    tifs = _list_tifs(folder)
    if not tifs:
        return {
            "folder": str(folder),
            "tif_count": 0,
            "sampled": 0,
            "consistent": True,
            "note": "Empty folder or no tif/tiff",
            "mismatches": [],
        }

    sample = tifs if len(tifs) < sample_n else tifs[:sample_n]
    ref_path = sample[0]
    ref_sig = _read_signature(ref_path)

    mismatches: list[dict] = []
    for p in sample[1:]:
        cur_sig = _read_signature(p)
        diffs = _diff_signature(ref_sig, cur_sig)
        if diffs:
            mismatches.append(
                {
                    "file": str(p),
                    "diffs": diffs,
                    "ref_file": str(ref_path),
                }
            )

    return {
        "folder": str(folder),
        "tif_count": len(tifs),
        "sampled": len(sample),
        "consistent": len(mismatches) == 0,
        "reference": {"file": str(ref_path), "signature": asdict(ref_sig)},
        "mismatches": mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether the spatial size / extent / projection of GeoTIFFs in each subfolder is consistent."
            "Sample at most N files per folder (check all if fewer than N)."
        )
    )
    parser.add_argument(
        "--root",
        default="/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled_5xdownsampled_merged_unifiedEntireCan",
        help="Root directory (contains multiple subfolders)",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=10,
        help="Upper bound on tifs sampled per folder (check all if fewer)",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Output line-by-line as JSONL (convenient for redirection / post-processing)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"root is not a directory: {root}")

    results: list[dict] = []
    for folder in _iter_subdirs(root):
        try:
            res = check_folder(folder, sample_n=args.sample_n)
        except Exception as e:
            res = {
                "folder": str(folder),
                "tif_count": None,
                "sampled": None,
                "consistent": False,
                "error": repr(e),
                "mismatches": None,
            }
        results.append(res)

        if args.jsonl:
            print(json.dumps(res, ensure_ascii=False))
        else:
            status = "OK" if res.get("consistent") else "MISMATCH"
            print(f"[{status}] {res.get('folder')} (tifs={res.get('tif_count')}, sampled={res.get('sampled')})")
            if res.get("note"):
                print(f"  note: {res['note']}")
            if res.get("error"):
                print(f"  error: {res['error']}")
            if res.get("reference"):
                ref = res["reference"]
                sig = ref["signature"]
                print(
                    "  ref:"
                    f" file={ref['file']}"
                    f" shape=({sig['height']},{sig['width']})"
                    f" bounds={tuple(sig['bounds'])}"
                )
            mism = res.get("mismatches") or []
            if mism:
                for m in mism:
                    print(f"  - file: {m['file']}")
                    for k, v in (m.get("diffs") or {}).items():
                        print(f"    * {k}: ref={v.get('ref')} cur={v.get('cur')}")
            if not args.jsonl:
                print()

    # summary
    bad = [r for r in results if not r.get("consistent")]
    if not args.jsonl:
        print(f"Summary: checked {len(results)} subfolders; inconsistent / errored {len(bad)} items.")
    return 0 if len(bad) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
