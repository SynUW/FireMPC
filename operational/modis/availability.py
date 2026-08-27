"""When each observation actually became available, from the CMR record.

`modis.package.resolve_source` picks the newest observation on or before a
requested date from whatever is on disk *now*.  That is the right answer for
an operational run -- what is on disk now is what has arrived -- but it is the
wrong answer for a backfill.  Rebuilding July today would resolve every date
against granules that landed weeks later, and a hindcast built on it sees
observations that had not been published when it claims to have run.

`modis latency probe` already writes the evidence needed to avoid that.  Each
record in ``reports/latency/<COLLECTION>_<stream>.jsonl`` carries the granule's
acquisition date and its CMR ``insert`` time -- the moment it entered the
archive and became downloadable:

    {"product": "MOD09GA", "a_date": "2026-05-14",
     "insert": "2026-05-16T03:34:16.853Z", ...}

This module turns that history into the question the packager needs to ask:
which acquisition dates had fully landed by a given as-of date?

Completion, not first sighting
------------------------------
A date is treated as available only once *every* recorded granule for it has
been inserted -- the max over its insert times, not the min.  The domain is 31
tiles; one tile arriving does not make the date usable, and `modis latency`
measures domain completion for the same reason.

A date with no records at all is reported as unknown rather than available.
Under an as-of query that means it is excluded: a backfill should not assume
availability it cannot evidence.  Callers that pass no as-of date keep the
current behaviour exactly.

The grace window
----------------
Strictness alone is unusable at the edges.  The probe history only reaches
back to 2026-05-13, so a January backfill would have no evidence for anything
and resolve nothing; and MCD12Q1 is an annual land-cover product whose sole
observation is 2024-01-01, which no recent sweep records at all.

Neither case is genuinely uncertain.  An observation older than
`grace_days` before the as-of date has unquestionably landed -- the observed
lags here run 2-6 days at the median, and nothing approaches three months.  So
a date at or beyond that horizon is available without needing a record, while
anything inside it must still prove itself.  The dates a window actually turns
on are one to ten days old, far inside the horizon, so the grace never
weakens the guarantee where it matters.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

# Run directly as `python operational/modis/availability.py`: put the application root on
# sys.path so the shared modules import by bare name, as elsewhere in FireMPC.
import sys
OPERATIONAL = Path(__file__).resolve().parents[1]
if str(OPERATIONAL) not in sys.path:
    sys.path.insert(0, str(OPERATIONAL))

import paths
from modis.processing_registry import ROUTES

# NDVIEVI is computed from scaled MCD09GA and has no collection of its own, so
# it inherits MCD09GA's availability -- the same substitution
# `modis.package.observation_dates` makes for its observation dates.
NDVIEVI_PROXY = "MCD09GA"


def _parse_instant(value: str) -> datetime:
    text = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@lru_cache(maxsize=None)
def route_collections(product: str) -> frozenset[str]:
    """The Earthdata collections a processing route is built from."""
    if product == "NDVIEVI":
        return route_collections(NDVIEVI_PROXY)
    route = ROUTES.get(product)
    if route is None:
        raise KeyError(f"unknown route {product!r}")
    from modis.registry import PRODUCTS
    collections = {
        PRODUCTS[export]["earthaccess"]["short_name"]
        for export in route.exports
        if export in PRODUCTS
    }
    if not collections:
        raise RuntimeError(f"route {product!r} maps to no Earthdata collection")
    return frozenset(collections)


def load_insert_times(
    reports_root: Path | None = None, *, streams: tuple[str, ...] = ("standard", "nrt"),
) -> dict[tuple[str, date], datetime]:
    """Latest insert time per (collection, acquisition date).

    Union across streams: a date is available once it has landed by any route,
    and NRT arriving first is a real availability event, not a different date.
    """
    root = Path(reports_root or Path(paths.REPORTS) / "latency")
    latest: dict[tuple[str, date], datetime] = {}
    if not root.is_dir():
        return latest
    for stream in streams:
        for path in sorted(root.glob(f"*_{stream}.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        collection = record["product"]
                        acquired = date.fromisoformat(record["a_date"])
                        inserted = _parse_instant(record["insert"])
                    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                        # A probe that failed is itself a record; it carries an
                        # error rather than an insert time and proves nothing
                        # about availability.
                        continue
                    key = (collection, acquired)
                    if key not in latest or inserted > latest[key]:
                        latest[key] = inserted
    return latest


def completion_times(
    product: str, insert_times: dict[tuple[str, date], datetime] | None = None,
) -> dict[date, datetime]:
    """When each acquisition date of a route became complete.

    A route fed by several collections is complete only once the slowest of
    them has landed, so the per-date answer is the max across collections.  A
    date missing from any contributing collection is omitted -- unknown, not
    available.
    """
    times = insert_times if insert_times is not None else load_insert_times()
    collections = route_collections(product)
    per_date: dict[date, dict[str, datetime]] = defaultdict(dict)
    for (collection, acquired), inserted in times.items():
        if collection in collections:
            per_date[acquired][collection] = inserted
    return {
        acquired: max(seen.values())
        for acquired, seen in per_date.items()
        if set(seen) == set(collections)
    }


DEFAULT_GRACE_DAYS = 90


def known_by(
    product: str, as_of: date,
    insert_times: dict[tuple[str, date], datetime] | None = None,
    *, candidates: set[date] | None = None, grace_days: int = DEFAULT_GRACE_DAYS,
) -> set[date]:
    """Acquisition dates of `product` that had fully landed by end of `as_of`.

    `candidates` is the set of dates actually on disk.  Any of them older than
    `grace_days` is accepted without evidence, which is what lets a backfill
    run over dates the probe history predates.
    """
    cutoff = datetime.combine(as_of, datetime.max.time(), tzinfo=timezone.utc)
    known = {
        acquired for acquired, complete in completion_times(product, insert_times).items()
        if complete <= cutoff
    }
    if candidates is not None and grace_days is not None:
        horizon = as_of - timedelta(days=grace_days)
        known |= {day for day in candidates if day <= horizon}
    return {day for day in known if day <= as_of}
