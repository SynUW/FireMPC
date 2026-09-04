"""The package stage must be told when a run is a backfill rather than today."""

from __future__ import annotations

from pathlib import Path

from modis.pipeline import package_command


POLICY = Path("/assets/modis_fill_policy.json")


def test_an_operational_run_leaves_modis_resolution_unrestricted():
    """What is on disk is what has arrived, which is correct only for today."""
    command = package_command("2026-08-25", POLICY, as_of=None, overwrite=False)
    assert "--as-of" not in command
    assert command[-4:] == ["--target", "2026-08-25", "--fill-policy", str(POLICY)]


def test_a_backfill_pins_resolution_to_what_had_landed():
    command = package_command("2026-01-15", POLICY, as_of="2026-01-15", overwrite=False)
    assert command[command.index("--as-of") + 1] == "2026-01-15"


def test_overwrite_survives_alongside_the_as_of_flag():
    command = package_command("2026-01-15", POLICY, as_of="2026-01-15", overwrite=True)
    assert command[command.index("--as-of") + 1] == "2026-01-15"
    assert command[-1] == "--overwrite"
