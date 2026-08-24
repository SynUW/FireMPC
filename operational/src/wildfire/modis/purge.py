"""Reclaim earthaccess raw HDF granules once they are no longer needed on disk.

A date's raw granules for an Earthdata collection (e.g. MOD09GA) are only
deleted once every export that collection feeds
(:func:`modis_registry.download_products`) has a *complete* entry in the
tiling history -- all four driver tiles present, recorded by ``modis_to_tiles``
only after they passed :func:`modis_to_tiles.validate_tile` -- and every one of
those tiles is still on disk and openable. Granules for a date that fails
either check are kept, regardless of age, because that date has not yet
produced a trustworthy downstream product.

The tiles are what the preprocessor consumes, so an intact set of four is
exactly the condition under which the granules become redundant: preprocessing
can be re-run from tiles alone. Only re-*tiling* needs the granules back, which
is what ``--retention-days`` buys time for.

Even a validated date is kept if it falls inside ``--retention-days`` of
today, so a recent processing bug can still be diagnosed against the raw
source. Defaults to a dry run; pass ``--execute`` to actually delete.

See ``plans/MODIS_OPERATIONAL_PLAN.md`` section 5 for the storage budget this
job exists to hold.
"""

from __future__ import annotations

import fcntl
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

import rasterio
import typer
from rasterio.errors import RasterioError

from wildfire import paths
from wildfire.modis.preprocess import TileEntry, load_complete_tiles
from wildfire.modis.registry import download_products
from wildfire.modis.to_tiles import validate_raster

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _lock(work_root: Path):
    work_root.mkdir(parents=True, exist_ok=True)
    path = work_root / "modis_purge_raw.lock"
    handle = path.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"another modis_purge_raw run holds {path}") from exc
    return handle


def _report_paths(source: Path) -> list[Path]:
    """Every tiling report in `source`, oldest first; or just `source` if a file."""
    if not source.is_dir():
        return [source]
    return sorted(source.glob("*.json"), key=lambda path: (path.stat().st_mtime, path.name))


def load_tiling_history(source: Path) -> dict[tuple[str, date], tuple[TileEntry, ...]]:
    """Union the complete tile entries across the whole tiling report history.

    ``latest.json`` describes only the run that wrote it, so reading one report
    sees only the dates that run happened to touch -- today that is nine
    MCD15A3H jobs. Purging is a statement about every date we have ever tiled,
    so it reads the directory and lets the newest report win per (export, date).
    """
    merged: dict[tuple[str, date], tuple[TileEntry, ...]] = {}
    for report in _report_paths(source):
        try:
            merged.update(load_complete_tiles(report, strict=False))
        except (json.JSONDecodeError, KeyError, ValueError, OSError) as exc:
            # Skipping a report can only shrink the purge set, so an unreadable
            # one costs disk rather than data. Aborting the sweep over a stray
            # file in the reports directory would cost more.
            typer.echo(f"warning: skipped unreadable tiling report {report}: {exc}", err=True)
    return merged


def _tile_failures(entries: tuple[TileEntry, ...], *, deep: bool) -> list[str]:
    """Why this date's tiles cannot be trusted; empty means they can."""
    failures = []
    for entry in entries:
        if not entry.path.is_file():
            failures.append(f"{entry.path}: missing")
            continue
        if deep:
            failures.extend(
                f"{entry.path}: {error}"
                for error in validate_raster(str(entry.path), entry.product)
            )
            continue
        try:
            with rasterio.open(entry.path) as dataset:
                if dataset.width <= 0 or dataset.height <= 0:
                    failures.append(f"{entry.path}: empty raster")
        except (RasterioError, OSError) as exc:
            failures.append(f"{entry.path}: {exc}")
    return failures


def validated_dates(
    exports: list[str],
    complete_tiles: dict[tuple[str, date], tuple],
    *,
    before: date | None = None,
    deep: bool = False,
) -> set[date]:
    """Dates where every export of a collection has four present, valid tiles.

    ``before`` drops dates inside the retention window *first*. They are kept
    no matter what validation says, so checking them is pure work: on this tree
    it is over half the tiles, and the deep check reads every block of each one.

    ``deep`` holds each tile to the full driver-tile raster contract -- CRS,
    band count, dtype, pixel size, lattice origin, and a checksum of every
    block. It is off by default because it is redundant here rather than
    cheap-and-careful: ``modis_to_tiles`` only records a tile as ``existing``
    or ``written`` after it passes :func:`modis_to_tiles.validate_tile`, which
    is this contract *plus* extent, so a manifest entry already asserts it. The
    only thing a re-check adds is detection of corruption that appeared after
    the report was written -- and the remedy for that is a re-download, not a
    retained granule. Use it as a periodic tile audit, not as a purge gate.
    """
    candidates: set[date] | None = None
    for export in exports:
        days = {day for product, day in complete_tiles if product == export}
        candidates = days if candidates is None else candidates & days
    if not candidates:
        return set()
    if before is not None:
        candidates = {day for day in candidates if day < before}
    return {
        day for day in candidates
        if not any(
            _tile_failures(complete_tiles[(export, day)], deep=deep)
            for export in exports
        )
    }


def raw_date_dir(raw_root: Path, short_name: str, day: date) -> Path:
    return raw_root / short_name / day.strftime("%Y-%m-%d")


def purge_date(directory: Path, execute: bool) -> tuple[int, int]:
    """Delete a date's granules (if `execute`); return (count, bytes reclaimed)."""
    granules = sorted(directory.glob("*.hdf"))
    reclaimed = sum(g.stat().st_size for g in granules)
    if execute:
        for g in granules:
            g.unlink()
        try:
            directory.rmdir()
        except OSError:
            pass
    return len(granules), reclaimed


@app.command()
def main(
    tiling_manifest: Annotated[Path, typer.Option(
        "--tiling-manifest",
        help="A tiling report, or a directory of them to union (the default).",
    )] = Path(paths.REPORTS) / "tiling",
    raw_root: Annotated[Path, typer.Option("--raw-root")] = Path(paths.RAW),
    retention_days: Annotated[int, typer.Option(
        "--retention-days",
        help="Keep a validated date's raw granules until it is at least this "
             "many days old, even though it is safe to purge.",
    )] = 90,
    deep_validate: Annotated[bool, typer.Option(
        "--deep-validate/--no-deep-validate",
        help="Re-check every tile against the full raster contract, "
             "checksumming each block. Off by default: the tiling report "
             "already records a stricter check made at write time. Use it as "
             "a periodic audit for corruption that appeared since.",
    )] = False,
    execute: Annotated[bool, typer.Option(
        "--execute",
        help="Actually delete. Without this flag, only report what would be removed.",
    )] = False,
) -> None:
    if retention_days < 0:
        raise typer.BadParameter("--retention-days must not be negative")
    handle = _lock(Path(paths.WORK))
    try:
        complete_tiles = load_tiling_history(tiling_manifest)
        cutoff = datetime.now(timezone.utc).date() - timedelta(days=retention_days)
        total_dates, total_bytes = 0, 0
        for short_name, cfg in sorted(download_products().items()):
            eligible = validated_dates(
                cfg["exports"], complete_tiles, before=cutoff, deep=deep_validate,
            )
            for day in sorted(eligible):
                directory = raw_date_dir(raw_root, short_name, day)
                if not directory.is_dir():
                    continue
                count, reclaimed = purge_date(directory, execute)
                if count == 0:
                    continue
                total_dates += 1
                total_bytes += reclaimed
                verb = "purged" if execute else "would purge"
                typer.echo(
                    f"{short_name} {day}: {verb} {count} granules "
                    f"({reclaimed / 1024**3:.2f} GiB)"
                )
        mode = "executed" if execute else "dry run"
        suffix = "" if execute else " (pass --execute to delete)"
        typer.echo(
            f"{mode}: {total_dates} dates, {total_bytes / 1024**3:.2f} GiB reclaimed{suffix}"
        )
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


if __name__ == "__main__":
    app()
