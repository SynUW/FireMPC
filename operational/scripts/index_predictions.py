#!/usr/bin/env python3
"""Build the by-valid-date view over a multi-horizon prediction tree.

`predict_multi_sweep.sh` writes one directory per target, because that is the
unit of compute (one publish, one data load, 15 leads) and because a flat tree
would let visualization_gen's resume probe silently skip leads whose valid date
another target already wrote.

The question people actually ask, though, is "what did we predict for this day,
at every lead we have?".  That is a transpose of the write layout, so this
builds it as relative symlinks -- no duplicated rasters -- plus a flat CSV for
anything that wants to join against it.

Idempotent: safe to re-run after each sweep chunk.  Stale links (a raster that
was deleted or rebuilt under a different target) are pruned.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from datetime import date, timedelta
from pathlib import Path

STEM = re.compile(r"^(\d{4})_(\d{2})_(\d{2})\.tif$")


def parse_target_dirs(by_target: Path):
    for entry in sorted(by_target.iterdir()):
        if entry.is_dir():
            try:
                yield date.fromisoformat(entry.name), entry
            except ValueError:
                continue


def build(root: Path, *, prune: bool) -> tuple[int, int]:
    by_target, by_valid = root / "by_target", root / "by_valid"
    if not by_target.is_dir():
        raise SystemExit(f"no by_target/ under {root}")
    by_valid.mkdir(exist_ok=True)

    rows, linked = [], 0
    for target, tdir in parse_target_dirs(by_target):
        for tif in sorted(tdir.glob("*.tif")):
            match = STEM.match(tif.name)
            if not match:          # skips *_classified.tif and sidecars
                continue
            valid = date(*(int(part) for part in match.groups()))
            lead = (valid - target).days + 1
            if lead < 1:
                print(f"  ! {tif}: valid {valid} precedes target {target}, skipping")
                continue

            vdir = by_valid / valid.isoformat()
            vdir.mkdir(exist_ok=True)
            for suffix in ("", "_classified"):
                src = tif if not suffix else tif.with_name(f"{tif.stem}{suffix}.tif")
                if not src.is_file():
                    continue
                link = vdir / f"lead{lead:02d}{suffix}.tif"
                rel = os.path.relpath(src, vdir)
                if link.is_symlink() and os.readlink(link) == rel:
                    continue
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(rel)
                linked += 1

            rows.append({
                "valid_date": valid.isoformat(),
                "lead": lead,
                "target": target.isoformat(),
                "window_start": (target - timedelta(days=10)).isoformat(),
                "window_end": (target - timedelta(days=1)).isoformat(),
                "path": str(tif.relative_to(root)),
            })

    pruned = 0
    if prune:
        for vdir in sorted(by_valid.glob("*")):
            if not vdir.is_dir():
                continue
            for link in sorted(vdir.iterdir()):
                if link.is_symlink() and not link.resolve().is_file():
                    link.unlink()
                    pruned += 1
            if not any(vdir.iterdir()):
                vdir.rmdir()

    rows.sort(key=lambda r: (r["valid_date"], r["lead"]))
    with (root / "index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else
                                ["valid_date", "lead", "target",
                                 "window_start", "window_end", "path"])
        writer.writeheader()
        writer.writerows(rows)

    return linked, pruned, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("/s3/quinn/prediction_vis/canfirecast_2026_multi"))
    parser.add_argument("--no-prune", action="store_true",
                        help="Keep symlinks whose target raster is gone.")
    args = parser.parse_args()

    linked, pruned, rows = build(args.root, prune=not args.no_prune)
    valid_days = {r["valid_date"] for r in rows}
    targets = {r["target"] for r in rows}
    print(f"{len(rows)} rasters | {len(targets)} targets | {len(valid_days)} valid dates")
    print(f"{linked} symlinks written, {pruned} stale pruned")
    print(f"index: {args.root / 'index.csv'}")

    if rows:
        depth = {}
        for row in rows:
            depth[row["valid_date"]] = depth.get(row["valid_date"], 0) + 1
        full = sum(1 for n in depth.values() if n == 15)
        print(f"valid dates with all 15 leads: {full}/{len(depth)}")


if __name__ == "__main__":
    main()
