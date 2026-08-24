"""Daily weather tiles from Earth Engine, in the historical GEE format.

Two sources, one output contract (`weather_registry.SOURCES`):

``ifs``   `ECMWF/NRT_FORECAST/IFS/OPER` -- the operational source.  ERA5-Land is
          an analysis product with a multi-day publication lag; IFS is a
          forecast, and the 00 timezone.utc cycle of a given day is in Earth Engine the
          same day.  23 bands, reduced from the 00 timezone.utc initialisation over
          forecast hours +3..+24 (see the registry for why +0 is excluded) and
          assigned to the initialisation date.  Nothing before 2024-11-13.

``era5``  `ECMWF/ERA5_LAND/DAILY_AGGR` -- history, and the only way to spin the
          FWI recursion up before IFS exists.  18 bands, one image per day, no
          reduction and no unit conversion: DAILY_AGGR band names are already
          the training channel names.

`--source auto` (the default) picks per date: ERA5-Land before the first
complete IFS cycle, IFS from then on, so a backfill spanning the changeover
does the right thing without being told.

The IFS output reproduces the 2025 archive at `drivers/IFS` exactly: four tiles
per day, 23 float32 bands in registry order, 11132 m, on the same 2x2 Canada
quadrant grid every other driver uses.  ERA5 tiles land beside them under
`drivers/ERA5` with the same geometry.

    python -m wildfire.cli weather download --start 2026-01-01 --end 2026-08-16
    python -m wildfire.cli weather download --source era5 --start 2023-01-01 --end 2024-11-12
    python -m wildfire.cli weather download --probe --start 2026-08-16 --end 2026-08-17

Downloads are idempotent: a tile that already validates is skipped, and a tile
is only moved to its archive path after it passes the raster contract, so a
half-written file can never be mistaken for a finished one.  Transient Earth
Engine failures are retried with exponential backoff -- a 594-day backfill hit
exactly one 503, and a daily job must not exit non-zero because of it.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

import typer

from wildfire import paths
from wildfire.weather import registry as registry

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

EE_PROJECT = os.environ.get("WILDFIRE_EARTH_ENGINE_PROJECT", os.environ.get("EE_PROJECT", ""))


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------
# Earth Engine's failure modes divide cleanly: a 5xx, a 429 or a dropped socket
# is the service, and succeeds on a second ask; a 400 or a missing asset is the
# request, and will fail identically forever.  Only the first kind is retried,
# so a genuine mistake still surfaces immediately.
RETRY_ATTEMPTS = 5
RETRY_BASE_SECONDS = 4.0
TRANSIENT_MARKERS = (
    "429", "500", "502", "503", "504",
    "too many requests", "service unavailable", "internal error",
    "backend error", "bad gateway", "gateway timeout", "deadline exceeded",
    "timed out", "timeout", "connection reset", "connection aborted",
    "connection refused", "temporarily unavailable", "eof occurred",
    "remote end closed", "incomplete read", "chunked",
)
TRANSIENT_TYPES = (
    "ConnectionError", "Timeout", "ConnectTimeout", "ReadTimeout",
    "ChunkedEncodingError", "ProtocolError", "IncompleteRead", "SSLError",
)


def is_transient(exc: BaseException) -> bool:
    names = {klass.__name__ for klass in type(exc).__mro__}
    if names & set(TRANSIENT_TYPES):
        return True
    text = str(exc).lower()
    return any(marker in text for marker in TRANSIENT_MARKERS)


def with_retry(what: str, call, attempts: int = RETRY_ATTEMPTS,
               base: float = RETRY_BASE_SECONDS, sleep=time.sleep):
    """Run `call`, retrying transient Earth Engine failures with backoff."""
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 - classified, then re-raised
            if attempt >= attempts or not is_transient(exc):
                raise
            # Jitter matters: four tiles of the same date otherwise retry in
            # lockstep and re-hit the same overloaded backend together.
            delay = base * 2 ** (attempt - 1) * (1.0 + random.random())
            print(f"    {what}: {type(exc).__name__}: {exc}; "
                  f"retry {attempt}/{attempts - 1} in {delay:.0f}s", file=sys.stderr)
            sleep(delay)
    raise RuntimeError(f"{what}: retries exhausted")  # pragma: no cover


# ---------------------------------------------------------------------------
# Earth Engine
# ---------------------------------------------------------------------------
def initialise():
    import ee

    if not EE_PROJECT:
        raise RuntimeError(
            "Earth Engine project is not configured; set WILDFIRE_EARTH_ENGINE_PROJECT"
        )
    ee.Initialize(project=EE_PROJECT)
    return ee


def cycle_images(ee, source: registry.WeatherSource, day: date):
    """The images backing one date, and the instant they are keyed to.

    For a forecast source that is the 00 timezone.utc initialisation: `creation_time` is
    the initialisation instant in epoch milliseconds and is the only property
    that separates one cycle from another, since `system:time_start` is the
    *valid* time and therefore repeats across cycles.  For a daily source it is
    simply the day's image.
    """
    if source.forecast:
        issue = datetime(day.year, day.month, day.day, registry.INITIALISATION_HOUR, tzinfo=timezone.utc)
        collection = ee.ImageCollection(source.collection).filter(
            ee.Filter.eq("creation_time", int(issue.timestamp() * 1000))
        )
    else:
        issue = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        collection = ee.ImageCollection(source.collection).filterDate(
            day.isoformat(), (day + timedelta(days=1)).isoformat()
        )
    return issue, collection


def available_steps(ee, collection) -> list[int]:
    return sorted(int(h) for h in with_retry(
        "forecast_hours", lambda: collection.aggregate_array("forecast_hours").getInfo()))


def missing_pieces(ee, source: registry.WeatherSource, collection) -> list[int] | None:
    """What this date still lacks, or None when it is complete.

    Forecast sources need every step in `FORECAST_HOURS`; a daily source needs
    its one image, reported as an empty list when absent so the caller has the
    same shape to record either way.
    """
    if source.forecast:
        missing = sorted(set(registry.FORECAST_HOURS) - set(available_steps(ee, collection)))
        return missing or None
    count = with_retry("image count", lambda: collection.size().getInfo())
    return None if count else []


def daily_composite(ee, source: registry.WeatherSource, collection):
    """Build the daily image for one date from one source.

    Each output band is produced independently so a reduction change touches
    one registry row rather than this function.
    """
    if source.forecast:
        window = collection.filter(
            ee.Filter.inList("forecast_hours", ee.List(list(registry.FORECAST_HOURS))))
        accumulated = ee.Image(
            collection.filter(
                ee.Filter.eq("forecast_hours", registry.ACCUMULATION_HOUR)).first())
    else:
        window = collection
        accumulated = ee.Image(collection.first())

    bands = []
    for band in source.bands:
        if band.reduction == "mean":
            image = window.select([band.source]).mean()
        elif band.reduction == "max":
            image = window.select([band.source]).max()
        elif band.reduction in ("accumulated", "daily"):
            image = accumulated.select([band.source])
        else:  # pragma: no cover - registry is frozen
            raise RuntimeError(f"unknown reduction {band.reduction!r} for {band.name}")
        if band.to_kelvin:
            image = image.add(registry.KELVIN_OFFSET)
        bands.append(image.rename([band.name]).toFloat())
    return ee.Image.cat(bands)


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------
def tile_geometries(tiles):
    """The quadrant polygons of the 2x2 Canada split, as shapely pieces.

    Reuses `modis_to_tiles.quadrant_grids`, which is verified to reproduce the
    reference IFS tile extents exactly -- including the geodesic apex Earth
    Engine puts on the northern edge of each request.
    """
    from wildfire.modis import to_tiles as modis_to_tiles

    return modis_to_tiles.quadrant_grids(registry.RESOLUTION_M, set(tiles))


def http_download(url: str, out: Path) -> None:
    """Stream one signed GEE URL to disk and reject a non-TIFF reply."""
    import requests

    part = out.with_suffix(out.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=(30, 900)) as response:
            response.raise_for_status()
            with open(part, "wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        with open(part, "rb") as handle:
            if handle.read(4) not in (b"II*\x00", b"MM\x00*"):
                raise RuntimeError("Earth Engine response is not a GeoTIFF")
        os.replace(part, out)
    finally:
        part.unlink(missing_ok=True)


def download_tile(ee, source: registry.WeatherSource, image, grid, out: Path,
                  attempts: int = RETRY_ATTEMPTS) -> None:
    """Fetch one tile and publish it only once it validates."""
    import rasterio

    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{source.key}-", dir=out.parent))
    try:
        staged = work / out.name

        def fetch() -> None:
            # `region` + `scale` is one pixel too generous here: the grid bounds
            # are already snapped to the lattice, and Earth Engine still rounds
            # the request outward, yielding 444x227 where the reference is
            # 443x226.  An explicit transform and size asks for the window we
            # computed.
            url = image.getDownloadURL({
                "crs": registry.CRS,
                "crs_transform": [grid.px, 0.0, grid.minx, 0.0, -grid.px, grid.maxy],
                "dimensions": f"{grid.width}x{grid.height}",
                "format": "GEO_TIFF",
            })
            http_download(url, staged)

        with_retry(f"{out.name}", fetch, attempts=attempts)
        errors = validate_tile(staged, source, grid)
        if errors:
            raise RuntimeError("; ".join(errors))
        with rasterio.open(staged, "r+") as dataset:
            dataset.descriptions = source.band_names
        os.replace(staged, out)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def validate_tile(path: Path, source: registry.WeatherSource, grid) -> list[str]:
    """The raster contract every weather tile must satisfy before it is archived."""
    import rasterio

    errors: list[str] = []
    try:
        with rasterio.open(path) as dataset:
            if dataset.count != len(source.bands):
                errors.append(f"{dataset.count} bands, expected {len(source.bands)}")
            if (dataset.width, dataset.height) != (grid.width, grid.height):
                errors.append(
                    f"{dataset.width}x{dataset.height}, expected {grid.width}x{grid.height}")
            transform = dataset.transform
            tolerance = max(1e-12, grid.px * 1e-9)
            if abs(transform.c - grid.minx) > tolerance or abs(transform.f - grid.maxy) > tolerance:
                errors.append(
                    f"origin ({transform.c}, {transform.f}), "
                    f"expected ({grid.minx}, {grid.maxy})")
            if abs(abs(transform.a) - grid.px) > tolerance:
                errors.append(f"pixel {abs(transform.a)}, expected {grid.px}")
    except Exception as exc:  # noqa: BLE001 - any read failure disqualifies the tile
        errors.append(f"unreadable: {exc}")
    return errors


# ---------------------------------------------------------------------------
# Day driver
# ---------------------------------------------------------------------------
def tile_path(dest: Path, source: registry.WeatherSource, day: date, tile: int) -> Path:
    return dest / source.prefix / f"{source.stem(day, tile)}.tif"


def sidecar_path(dest: Path, source: registry.WeatherSource, day: date) -> Path:
    return dest / source.prefix / f"{source.sidecar_stem(day)}.json"


def existing(dest: Path, source: registry.WeatherSource, day: date, tile: int, grids) -> bool:
    path = tile_path(dest, source, day, tile)
    return path.is_file() and not validate_tile(path, source, grids[tile])


def fetch_day(ee, dest: Path, source: registry.WeatherSource, day: date, tiles, grids,
              overwrite: bool, attempts: int = RETRY_ATTEMPTS) -> dict:
    """Download every requested tile for one date and write its sidecar."""
    issue, collection = cycle_images(ee, source, day)
    missing = missing_pieces(ee, source, collection)
    if missing is not None:
        return {
            "status": "unavailable",
            "source": source.key,
            "issue_time_utc": issue.isoformat().replace("+00:00", "Z"),
            "missing_forecast_hours": missing,
        }

    image = daily_composite(ee, source, collection)
    written, skipped = [], []
    for tile in sorted(tiles):
        out = tile_path(dest, source, day, tile)
        if not overwrite and existing(dest, source, day, tile, grids):
            skipped.append(str(out))
            continue
        download_tile(ee, source, image, grids[tile], out, attempts=attempts)
        written.append(str(out))

    record = {
        "status": "downloaded",
        "collection": source.collection,
        "source": source.key,
        "model": source.key,
        "registry_version": registry.REGISTRY_VERSION,
        "issue_time_utc": issue.isoformat().replace("+00:00", "Z"),
        "scale_meters": source.resolution_m,
        "output_bands": list(source.band_names),
        "missing_era5_bands": [
            name for name in registry.MISSING_ERA5_BANDS if name not in source.band_index],
        "tile_files": written + skipped,
        "tiles_written": written,
        "tiles_skipped": skipped,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if source.forecast:
        record.update({
            "valid_start_utc": (issue + timedelta(hours=registry.FORECAST_HOURS[0])
                                ).isoformat().replace("+00:00", "Z"),
            "valid_end_utc": (issue + timedelta(hours=registry.ACCUMULATION_HOUR)
                              ).isoformat().replace("+00:00", "Z"),
            "forecast_hours_used": list(registry.FORECAST_HOURS),
        })
    sidecar = sidecar_path(dest, source, day)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    temporary = sidecar.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record, indent=1, sort_keys=True))
    os.replace(temporary, sidecar)
    return record


def probe_day(ee, source: registry.WeatherSource, day: date) -> dict:
    """Metadata-only availability check; downloads nothing."""
    issue, collection = cycle_images(ee, source, day)
    missing = missing_pieces(ee, source, collection)
    record = {
        "date": day.isoformat(),
        "source": source.key,
        "collection": source.collection,
        "issue_time_utc": issue.isoformat().replace("+00:00", "Z"),
        "complete": missing is None,
        "missing_forecast_hours": [] if missing is None else missing,
        "observed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if source.forecast:
        record["n_steps"] = len(available_steps(ee, collection))
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _dates(start: date, end: date):
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def resolve_source(choice: str, day: date) -> registry.WeatherSource:
    """`auto` follows the registry changeover; anything else is taken literally."""
    return registry.source_for(day) if choice == "auto" else registry.source_by_key(choice)


@app.command()
def main(
    start: Annotated[datetime, typer.Option("--start", formats=["%Y-%m-%d"])],
    end: Annotated[datetime, typer.Option("--end", formats=["%Y-%m-%d"])],
    source: Annotated[str, typer.Option(
        "--source", help="ifs, era5, or auto (ERA5-Land before the first IFS cycle).",
    )] = "auto",
    dest: Annotated[Path, typer.Option("--dest")] = Path(paths.DRIVERS),
    tiles: Annotated[str, typer.Option("--tiles")] = "0,1,2,3",
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    retries: Annotated[int, typer.Option(
        "--retries", help="Attempts per Earth Engine call, including the first.",
    )] = RETRY_ATTEMPTS,
    probe: Annotated[bool, typer.Option("--probe", help="Report availability only.")] = False,
) -> None:
    choice = source.lower()
    if choice != "auto":
        try:
            registry.source_by_key(choice)
        except KeyError as exc:
            raise typer.BadParameter(str(exc)) from None
    wanted = tuple(int(token) for token in tiles.split(",") if token.strip())
    if not set(wanted).issubset(registry.TILES):
        raise typer.BadParameter(f"tiles must be a subset of {registry.TILES}")
    if retries < 1:
        raise typer.BadParameter("--retries must be at least 1")

    ee = initialise()
    grids = tile_geometries(wanted)

    first, last = start.date(), end.date()
    if last < first:
        raise typer.BadParameter("--end precedes --start")

    print(f"range {first} .. {last}   source={choice}   dest={dest}")
    failures = 0
    for day in _dates(first, last):
        try:
            day_source = resolve_source(choice, day)
            if day < day_source.first_date:
                raise RuntimeError(
                    f"{day_source.key} begins {day_source.first_date}; nothing earlier exists")
            if probe:
                record = probe_day(ee, day_source, day)
                mark = "ok" if record["complete"] else "incomplete"
                print(f"  {day}  {day_source.key:5s} {mark:11s} "
                      f"missing={record['missing_forecast_hours']}")
                continue
            record = fetch_day(ee, dest, day_source, day, wanted, grids, overwrite,
                               attempts=retries)
            if record["status"] != "downloaded":
                failures += 1
                print(f"  {day}  {day_source.key}  unavailable  missing "
                      f"{record['missing_forecast_hours']}", file=sys.stderr)
                continue
            print(f"  {day}  {day_source.key:5s} downloaded {len(record['tiles_written'])} "
                  f"skipped {len(record['tiles_skipped'])}")
        except Exception as exc:  # noqa: BLE001 - one bad date must not stop a backfill
            failures += 1
            print(f"  {day}  FAILED: {exc}", file=sys.stderr)

    if failures:
        print(f"{failures} date(s) failed", file=sys.stderr)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
