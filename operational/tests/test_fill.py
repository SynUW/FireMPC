"""The range fill's date arithmetic.

`fill` takes a range of store *dates* and runs a range of *targets*, and the
two are offset by one: a target reads D-10..D-1, so it is the first run that
can freeze D-1.  Getting that wrong is invisible -- the sweep still completes,
every date still gets written -- but each date is then written by the wrong
target and carries hindsight it should not have.  So it is asserted here.
"""

from __future__ import annotations

from datetime import date, timedelta

import fill


def _targets(first: date, last: date) -> list[date]:
    return [first + timedelta(days=n + fill.TARGET_OFFSET_DAYS)
            for n in range((last - first).days + 1)]


def test_each_date_is_filled_by_the_target_after_it():
    first, last = date(2026, 6, 1), date(2026, 6, 30)
    targets = _targets(first, last)
    assert targets[0] == date(2026, 6, 2)
    assert targets[-1] == date(2026, 7, 1)
    assert len(targets) == 30


def test_a_single_date_is_a_single_target():
    assert _targets(date(2026, 6, 1), date(2026, 6, 1)) == [date(2026, 6, 2)]


def test_the_ledger_is_advanced_through_the_last_target_minus_eleven():
    """The pipeline's preflight demands certified state at D-11, no exceptions."""
    last_target = _targets(date(2026, 6, 1), date(2026, 6, 30))[-1]
    assert last_target - timedelta(days=fill.STATE_LAG_DAYS) == date(2026, 6, 20)


def test_the_modis_ingest_covers_every_targets_lookback():
    """One pass over the range, not one 45-day lookback per target."""
    targets = _targets(date(2026, 6, 1), date(2026, 6, 30))
    lookback = (targets[-1] - (targets[0] - timedelta(days=fill.MODIS_LOOKBACK_DAYS))).days
    assert targets[-1] - timedelta(days=lookback) == targets[0] - timedelta(
        days=fill.MODIS_LOOKBACK_DAYS)


def test_relative_days_resolve_against_utc():
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).date()
    assert fill.resolve_day("today") == today
    assert fill.resolve_day("yesterday") == today - timedelta(days=1)
    assert fill.resolve_day("2026-06-01") == date(2026, 6, 1)
