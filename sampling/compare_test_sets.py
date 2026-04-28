#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare test-set identity across sample JSON files.

Expected input format (from sampling/generate_samples.py):
{
  "samples": [ { ... "split": "test", ... }, ... ],
  "total_samples": N
}

This script normalizes dtypes (date/pixel_row/pixel_col/sample_type) and compares:
- each variant's test split against the others
- optionally against a fixed_test_set.json (same format; split may be omitted)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Tuple


KeyRow = Tuple[str, int, int, str]


def _read_samples(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        obj = json.load(fh)
    samples = obj.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"{path}: missing/invalid 'samples' list")
    return samples


def _normalize_date(s: object) -> str:
    # generate_samples writes YYYY-MM-DD; be tolerant to YYYY_MM_DD
    v = str(s).strip()
    return v.replace("_", "-")


def _iter_test_keys(samples: Iterable[dict], *, require_split: bool) -> Iterable[KeyRow]:
    for r in samples:
        if require_split:
            if str(r.get("split", "")).lower() != "test":
                continue
        date = _normalize_date(r.get("date", ""))
        pr = int(r.get("pixel_row"))
        pc = int(r.get("pixel_col"))
        st = str(r.get("sample_type", "")).strip().lower()
        yield (date, pr, pc, st)


def _sorted_keys_and_hash(keys: Iterable[KeyRow]) -> tuple[list[KeyRow], str]:
    rows = sorted(keys)
    h = hashlib.sha256()
    for d, r, c, t in rows:
        h.update(f"{d}\t{r}\t{c}\t{t}\n".encode("utf-8"))
    return rows, h.hexdigest()


def _diff_example(a: list[KeyRow], b: list[KeyRow], n: int = 10) -> tuple[list[KeyRow], list[KeyRow]]:
    sa, sb = set(a), set(b)
    only_a = sorted(sa - sb)[:n]
    only_b = sorted(sb - sa)[:n]
    return only_a, only_b


def main() -> int:
    p = argparse.ArgumentParser(description="Compare test splits across sample JSON files.")
    p.add_argument(
        "--dir",
        type=Path,
        default=Path("/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/sampling"),
        help="Directory containing samples_variant_*.json",
    )
    p.add_argument(
        "--variants",
        nargs="*",
        default=list("ABCDE"),
        help="Variant letters to compare (default: A B C D E)",
    )
    p.add_argument(
        "--fixed",
        type=Path,
        default=None,
        help="Optional fixed_test_set.json path to compare against.",
    )
    p.add_argument(
        "--show-diff",
        action="store_true",
        help="Print up to 10 example rows that differ when a mismatch is found.",
    )
    args = p.parse_args()

    variant_paths: List[Path] = []
    for v in args.variants:
        vp = args.dir / f"samples_variant_{v}.json"
        if not vp.is_file():
            raise FileNotFoundError(vp)
        variant_paths.append(vp)

    variant_rows: dict[str, list[KeyRow]] = {}
    variant_hash: dict[str, str] = {}
    for vp in variant_paths:
        v = vp.stem.split("_")[-1]
        samples = _read_samples(vp)
        rows, hexd = _sorted_keys_and_hash(_iter_test_keys(samples, require_split=True))
        variant_rows[v] = rows
        variant_hash[v] = hexd
        print(f"{vp.name}: test_rows={len(rows):,} sha256={hexd}")

    # Pairwise compare
    vs = list(variant_rows.keys())
    base = vs[0]
    ok_all = True
    for v in vs[1:]:
        same = variant_hash[v] == variant_hash[base] and len(variant_rows[v]) == len(variant_rows[base])
        print(f"compare {base} vs {v}: {'OK' if same else 'DIFF'}")
        if not same:
            ok_all = False
            if args.show_diff:
                only_a, only_b = _diff_example(variant_rows[base], variant_rows[v])
                print(f"  only_in_{base} (first {len(only_a)}): {only_a}")
                print(f"  only_in_{v} (first {len(only_b)}): {only_b}")

    # Optional fixed test set
    if args.fixed is not None:
        fp = args.fixed
    else:
        fp = args.dir / "fixed_test_set.json"
        if not fp.is_file():
            fp = None
    if fp is not None:
        fixed_samples = _read_samples(fp)
        # fixed file might not include split; accept both
        fixed_rows, fixed_hex = _sorted_keys_and_hash(_iter_test_keys(fixed_samples, require_split=False))
        print(f"{Path(fp).name}: rows={len(fixed_rows):,} sha256={fixed_hex}")
        same = fixed_hex == variant_hash[base] and len(fixed_rows) == len(variant_rows[base])
        print(f"compare fixed vs {base}: {'OK' if same else 'DIFF'}")
        if not same and args.show_diff:
            only_f, only_v = _diff_example(fixed_rows, variant_rows[base])
            print(f"  only_in_fixed (first {len(only_f)}): {only_f}")
            print(f"  only_in_{base} (first {len(only_v)}): {only_v}")

    return 0 if ok_all else 2


if __name__ == "__main__":
    raise SystemExit(main())

