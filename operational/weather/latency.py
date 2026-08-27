"""How long after a date the weather source is actually usable.

One probe showing same-day IFS availability is one probe, not a distribution.
This accumulates the distribution, the same discipline `modis_latency.py`
applies to MODIS, and is what an SLA should be written from.

    python operational/cli.py weather latency probe --source ifs --days 3
    python operational/cli.py weather latency report --source ifs

WHY THIS CAN ONLY GO FORWARD
  MODIS lag is reconstructable retrospectively because CMR preserves
  `ProviderDates[Insert]` -- when the archive could first return a granule.
  Earth Engine publishes no equivalent: an image carries its valid time, not
  its ingest time.  So first-seen has to be *observed*, by asking repeatedly and
  recording the first time the answer was yes.  Run `probe` on a schedule and
  let the store grow; a report over fewer than ~30 dates is a sample, not a
  distribution, and says so.

WHAT THE NUMBER MEANS
  Lag is reported in hours after 00:00 timezone.utc on the target date, so "6.2h" reads
  directly as "complete at 06:12 timezone.utc on the day itself" -- the number a cron
  schedule is written from.

  It is bounded below by the probe cadence: a date first *seen* complete at
  06:00 may have been complete at 04:30 if nothing asked in between.  Every
  first-seen lag is therefore an upper bound, and the report prints the observed
  cadence beside it so the resolution of the measurement is never implicit.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

import typer

# Run directly as `python operational/weather/latency.py`: put the application root on
# sys.path so the shared modules import by bare name, as elsewhere in FireMPC.
import sys
from pathlib import Path
OPERATIONAL = Path(__file__).resolve().parents[1]
if str(OPERATIONAL) not in sys.path:
    sys.path.insert(0, str(OPERATIONAL))

import paths
from weather import download as download
from weather import registry as registry

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def store_path(store_root: Path, source: registry.WeatherSource) -> Path:
    return store_root / "weather_latency" / f"{source.key}.jsonl"


def append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A partial final line means a probe was killed mid-write; the rest
            # of the store is still good evidence.
            continue
    return rows


def lag_hours(target: date, observed_at: str) -> float:
    """Hours between 00:00 timezone.utc on the target date and an observation."""
    moment = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    midnight = datetime(target.year, target.month, target.day, tzinfo=timezone.utc)
    return (moment - midnight).total_seconds() / 3600.0


def first_complete(rows: list[dict]) -> dict[date, str]:
    """Earliest observation that found each date complete."""
    seen: dict[date, str] = {}
    for row in rows:
        if not row.get("complete"):
            continue
        day = date.fromisoformat(row["date"])
        moment = row["observed_at_utc"]
        if day not in seen or moment < seen[day]:
            seen[day] = moment
    return seen


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile; no numpy, and exact on tiny samples."""
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    rank = max(1, min(len(ordered), round(fraction * len(ordered) + 0.5)))
    return ordered[rank - 1]


@app.command()
def probe(
    source: Annotated[str, typer.Option("--source")] = "ifs",
    days: Annotated[int, typer.Option(
        "--days", help="How many recent target dates to check, ending today.",
    )] = 3,
    store_root: Annotated[Path, typer.Option("--store-root")] = Path(paths.REPORTS),
) -> None:
    """Ask Earth Engine what is complete right now and append the answer."""
    weather = registry.source_by_key(source)
    ee = download.initialise()
    path = store_path(store_root, weather)
    today = datetime.now(timezone.utc).date()
    for offset in range(days):
        day = today - timedelta(days=offset)
        if day < weather.first_date:
            continue
        try:
            record = download.probe_day(ee, weather, day)
        except Exception as exc:  # noqa: BLE001 - a failed probe is itself evidence
            record = {
                "date": day.isoformat(), "source": weather.key, "complete": False,
                "error": str(exc),
                "observed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
        append(path, record)
        mark = "complete" if record["complete"] else "incomplete"
        typer.echo(f"  {day}  {mark:10s} +{lag_hours(day, record['observed_at_utc']):.1f}h")
    typer.echo(str(path))


@app.command()
def report(
    source: Annotated[str, typer.Option("--source")] = "ifs",
    store_root: Annotated[Path, typer.Option("--store-root")] = Path(paths.REPORTS),
    minimum_dates: Annotated[int, typer.Option(
        "--minimum-dates", help="Below this many dates, the report is labelled a sample.",
    )] = 30,
) -> None:
    """Print the first-seen lag distribution accumulated so far."""
    weather = registry.source_by_key(source)
    path = store_path(store_root, weather)
    rows = load(path)
    if not rows:
        raise typer.BadParameter(f"no observations in {path}; run `probe` first")

    seen = first_complete(rows)
    lags = [lag_hours(day, moment) for day, moment in sorted(seen.items())]
    probed = sorted({row["observed_at_utc"] for row in rows})
    gaps = [
        (datetime.fromisoformat(b.replace("Z", "+00:00"))
         - datetime.fromisoformat(a.replace("Z", "+00:00"))).total_seconds() / 3600.0
        for a, b in zip(probed, probed[1:])
    ]
    incomplete = sorted({row["date"] for row in rows} - {day.isoformat() for day in seen})

    typer.echo(f"{weather.key}  {weather.collection}")
    typer.echo(f"  store            {path}")
    typer.echo(f"  observations     {len(rows)} over {len(probed)} probe runs")
    typer.echo(f"  probe cadence    median {percentile(gaps, 0.5):.1f}h"
               if gaps else "  probe cadence    single run")
    typer.echo(f"  dates complete   {len(seen)}")
    if lags:
        typer.echo(
            f"  first-seen lag   min {min(lags):.1f}h  p50 {percentile(lags, 0.5):.1f}h  "
            f"p90 {percentile(lags, 0.9):.1f}h  max {max(lags):.1f}h"
        )
        typer.echo("  (upper bounds: a date is only seen complete when something asks)")
    if incomplete:
        typer.echo(f"  never complete   {len(incomplete)}: {', '.join(incomplete[:8])}"
                   + (" ..." if len(incomplete) > 8 else ""))
    if len(seen) < minimum_dates:
        typer.echo(
            f"  SAMPLE ONLY: {len(seen)} of {minimum_dates} dates. Not enough to set an SLA."
        )


if __name__ == "__main__":
    app()
