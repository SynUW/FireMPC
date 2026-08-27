"""As-of resolution must not show a backfill granules that had not landed."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from modis import availability as av
from modis.package import resolve_source


def _write_probe(root: Path, collection: str, records, stream: str = "standard") -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{collection}_{stream}.jsonl").open("a", encoding="utf-8") as handle:
        for acquired, inserted in records:
            handle.write(json.dumps({
                "product": collection, "a_date": acquired.isoformat(),
                "insert": inserted.isoformat().replace("+00:00", "Z"),
            }) + "\n")


def _landed(day: date, lag_days: int) -> datetime:
    return datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc) + timedelta(days=lag_days)


def test_a_route_maps_to_its_earthdata_collections():
    assert av.route_collections("MCD09GA") == frozenset({"MOD09GA", "MYD09GA"})
    assert av.route_collections("MCD15A3H") == frozenset({"MCD15A3H"})
    # NDVIEVI is derived from scaled MCD09GA and has no collection of its own.
    assert av.route_collections("NDVIEVI") == av.route_collections("MCD09GA")


def test_a_route_is_complete_only_when_its_slowest_collection_lands(tmp_path: Path):
    day = date(2026, 7, 10)
    _write_probe(tmp_path, "MOD09GA", [(day, _landed(day, 2))])
    _write_probe(tmp_path, "MYD09GA", [(day, _landed(day, 5))])
    times = av.load_insert_times(tmp_path)
    assert av.completion_times("MCD09GA", times)[day] == _landed(day, 5)
    assert day not in av.known_by("MCD09GA", date(2026, 7, 14), times)
    assert day in av.known_by("MCD09GA", date(2026, 7, 15), times)


def test_a_date_missing_from_one_collection_is_unknown(tmp_path: Path):
    day = date(2026, 7, 10)
    _write_probe(tmp_path, "MOD09GA", [(day, _landed(day, 1))])
    times = av.load_insert_times(tmp_path)
    assert av.completion_times("MCD09GA", times) == {}


def test_observations_beyond_the_grace_horizon_need_no_evidence(tmp_path: Path):
    """MCD12Q1's only observation is years old and predates every probe."""
    ancient = date(2024, 1, 1)
    known = av.known_by("MCD12Q1", date(2026, 7, 1), av.load_insert_times(tmp_path),
                        candidates={ancient})
    assert known == {ancient}


def test_resolve_source_refuses_an_observation_that_had_not_landed(tmp_path: Path):
    observations = {date(2026, 7, 8), date(2026, 7, 9)}
    available = {day: tmp_path / f"{day}.tif" for day in observations}
    _write_probe(tmp_path, "MOD09GA", [(d, _landed(d, 1)) for d in observations])
    _write_probe(tmp_path, "MYD09GA", [(d, _landed(d, 1)) for d in observations])
    times = av.load_insert_times(tmp_path)

    def known(as_of):
        return av.known_by("MCD09GA", as_of, times, candidates=observations)

    # On the 10th, the 9th has landed; resolving the 9th as of the 9th must not
    # reach for it, because it only appeared on the 10th.
    assert date(2026, 7, 9) in known(date(2026, 7, 10))
    assert date(2026, 7, 9) not in known(date(2026, 7, 9))

    choice = resolve_source("MCD09GA", date(2026, 7, 9), available, 30, observations)
    assert choice.source_date == date(2026, 7, 9)


def test_without_as_of_behaviour_is_unchanged(tmp_path: Path):
    observations = {date(2026, 7, 8), date(2026, 7, 9)}
    available = {day: tmp_path / f"{day}.tif" for day in observations}
    choice = resolve_source("MCD09GA", date(2026, 7, 9), available, 30, observations)
    assert choice.source_date == date(2026, 7, 9)
    assert choice.age_days == 0
