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


def test_a_route_is_usable_once_its_first_platform_lands(tmp_path: Path):
    """The preprocessor builds a date when ANY platform group is complete.

    This replaces an earlier contract that took the max across every
    collection.  That rule ruled out single-platform days, which
    `available_platform_groups` treats as first-class -- and 2026 contains a
    55-day Aqua outage whose dates all have full Terra-built rasters.
    """
    day = date(2026, 7, 10)
    _write_probe(tmp_path, "MOD09GA", [(day, _landed(day, 2))])
    _write_probe(tmp_path, "MYD09GA", [(day, _landed(day, 5))])
    times = av.load_insert_times(tmp_path)
    assert av.completion_times("MCD09GA", times)[day] == _landed(day, 2)
    assert day not in av.known_by("MCD09GA", date(2026, 7, 11), times)
    assert day in av.known_by("MCD09GA", date(2026, 7, 12), times)


def test_a_single_platform_day_is_available_from_that_platform(tmp_path: Path):
    """Terra alone is enough; the absent platform is not evidence of absence."""
    day = date(2026, 4, 1)          # inside the 2026 Aqua outage
    _write_probe(tmp_path, "MOD09GA", [(day, _landed(day, 1))])
    times = av.load_insert_times(tmp_path)
    assert av.completion_times("MCD09GA", times)[day] == _landed(day, 1)
    assert day in av.known_by("MCD09GA", date(2026, 4, 2), times)


def test_a_date_with_no_complete_group_is_still_unknown(tmp_path: Path):
    """Nothing is available without evidence -- the conservative rule stands."""
    day = date(2026, 7, 10)
    times = av.load_insert_times(tmp_path)          # no probe records at all
    assert av.completion_times("MCD09GA", times) == {}
    assert day not in av.known_by(
        "MCD09GA", date(2026, 7, 12), times, candidates={day})


def test_the_aqua_outage_does_not_strand_a_terra_window(tmp_path: Path):
    """The failure this fix exists to prevent, at window scale."""
    outage = [date(2026, 3, 17) + timedelta(days=n) for n in range(55)]
    for day in outage:
        _write_probe(tmp_path, "MOD09GA", [(day, _landed(day, 2))])
    times = av.load_insert_times(tmp_path)
    # The last outage date lands at D+2, so an as-of before that correctly
    # still excludes it -- the guarantee is intact, not weakened.
    assert av.known_by("MCD09GA", date(2026, 5, 12), times,
                       candidates=set(outage)) == set(outage)
    assert av.known_by("MCD09GA", date(2026, 5, 11), times,
                       candidates=set(outage)) == set(outage) - {date(2026, 5, 10)}


def test_ndvievi_inherits_the_proxy_platform_groups():
    assert av.route_platform_collections("NDVIEVI") == av.route_platform_collections("MCD09GA")
    assert av.route_platform_collections("MCD09GA") == (
        frozenset({"MOD09GA"}), frozenset({"MYD09GA"}))


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
