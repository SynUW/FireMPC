"""Reclaim the intermediate weather stages once a date is quantised.

`processed/weather` grows about 55 MB a day across VPD and FWI, and the
float32 `unified` tree is most of it -- 33 GB of the 46 GB on disk today
against 8.2 GB of `scaled`.  The model reads `scaled`; the earlier stages are
working products.

A date's `derived`, `merged` and `unified` rasters are deleted only once its
`scaled` raster exists and opens, and only once the date is older than
``--retention-days``.  Both conditions matter for different reasons: the first
says the pipeline finished, the second keeps a recent processing bug
diagnosable against the intermediates that produced it.

Nothing here touches the driver tiles under `drivers/IFS` and `drivers/ERA5`.
Those are the re-derivation source -- a purged date is rebuilt by re-running
`weather_preprocess.py`, which needs them -- and they are cheap, about 3.7 MB
a day against the 55 MB this job reclaims.

The FWI recursion is unaffected: its state lives in the `fwi_state` checkpoint,
never in the rasters, so deleting a `derived` FWI file loses no history.

Defaults to a dry run; pass ``--execute`` to actually delete.

    python operational/cli.py weather purge
    python operational/cli.py weather purge --retention-days 30 --execute
"""

from __future__ import annotations

import fcntl
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

import rasterio
import typer
from rasterio.errors import RasterioError

# Run directly as `python operational/weather/purge.py`: put the application root on
# sys.path so the shared modules import by bare name, as elsewhere in FireMPC.
import sys
from pathlib import Path
OPERATIONAL = Path(__file__).resolve().parents[1]
if str(OPERATIONAL) not in sys.path:
    sys.path.insert(0, str(OPERATIONAL))

import ops, paths
from weather import registry as registry

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

DATE_IN_NAME = re.compile(r"(\d{4})_(\d{2})_(\d{2})")


def stage_dir(data_root: Path, stage: str, product: str) -> Path:
    return data_root / stage / registry.ROUTES[product].output_dir


def date_of(path: Path) -> date | None:
    match = DATE_IN_NAME.search(path.name)
    if not match:
        return None
    try:
        return date(*(int(part) for part in match.groups()))
    except ValueError:
        return None


def files_by_date(folder: Path) -> dict[date, list[Path]]:
    found: dict[date, list[Path]] = {}
    if not folder.is_dir():
        return found
    for path in sorted(folder.glob("*.tif")):
        day = date_of(path)
        if day is not None:
            found.setdefault(day, []).append(path)
    return found


def is_readable(path: Path) -> bool:
    try:
        with rasterio.open(path) as dataset:
            return dataset.width > 0 and dataset.height > 0
    except (RasterioError, OSError):
        return False


def purgeable_dates(data_root: Path, product: str, *, before: date) -> set[date]:
    """Dates whose `scaled` output is present and readable, and old enough.

    An unreadable `scaled` raster keeps the whole date: that is exactly the
    case where the intermediates are needed to work out what went wrong.
    """
    scaled = files_by_date(stage_dir(data_root, "scaled", product))
    return {
        day for day, entries in scaled.items()
        if day < before and entries and all(is_readable(path) for path in entries)
    }


@app.command()
def main(
    data_root: Annotated[Path, typer.Option("--data-root")] = Path(paths.PROCESSED_WEATHER),
    products: Annotated[str, typer.Option(
        "--products", help="Comma-separated, or 'all'.",
    )] = "all",
    stages: Annotated[str, typer.Option(
        "--stages", help="Which pre-scaled stages to reclaim.",
    )] = "derived,merged,unified",
    retention_days: Annotated[int, typer.Option(
        "--retention-days",
        help="Keep a finished date's intermediates until it is at least this "
             "many days old, even though it is safe to purge.",
    )] = 30,
    execute: Annotated[bool, typer.Option(
        "--execute", help="Actually delete. Without this flag, only report.",
    )] = False,
) -> None:
    if retention_days < 0:
        raise typer.BadParameter("--retention-days must not be negative")
    wanted_products = (
        sorted(registry.ROUTES) if products == "all"
        else [token.strip() for token in products.split(",") if token.strip()]
    )
    unknown = [name for name in wanted_products if name not in registry.ROUTES]
    if unknown:
        raise typer.BadParameter(f"unknown product {unknown[0]!r}")
    wanted_stages = [token.strip() for token in stages.split(",") if token.strip()]
    if "scaled" in wanted_stages:
        raise typer.BadParameter("`scaled` is the product this job protects, not a purge target")

    handle = ops.lock(Path(paths.WORK), "weather_purge")
    try:
        cutoff = datetime.now(timezone.utc).date() - timedelta(days=retention_days)
        total_files, total_bytes = 0, 0
        for product in wanted_products:
            eligible = purgeable_dates(data_root, product, before=cutoff)
            route = registry.ROUTES[product]
            for stage in wanted_stages:
                if stage not in route.stages:
                    continue  # FWI has no `merged` stage; nothing to say about it.
                found = files_by_date(stage_dir(data_root, stage, product))
                files = [path for day in sorted(eligible & set(found)) for path in found[day]]
                if not files:
                    continue
                reclaimed = sum(path.stat().st_size for path in files)
                if execute:
                    for path in files:
                        path.unlink()
                total_files += len(files)
                total_bytes += reclaimed
                verb = "purged" if execute else "would purge"
                typer.echo(
                    f"{product} {stage}: {verb} {len(files)} files across "
                    f"{len(eligible & set(found))} dates ({reclaimed / 1024**3:.2f} GiB)"
                )
        mode = "executed" if execute else "dry run"
        suffix = "" if execute else " (pass --execute to delete)"
        typer.echo(
            f"{mode}: {total_files} files, {total_bytes / 1024**3:.2f} GiB reclaimed{suffix}"
        )
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


if __name__ == "__main__":
    app()
