"""The unattended daily run: what systemd invokes, and nothing else.

`cli.py canfirecast pipeline` is already the full operational chain, but it
assumes a person is watching: it has no lock, no retry, and it presumes the
certified D-11 state ledger has already been advanced to the day it needs.
Under a timer none of those hold, so this wraps it with the three things an
unattended run needs and keeps them out of the pipeline itself.

* One run at a time.  A flock on the data root means a timer firing while
  yesterday's slow run is still downloading queues behind it instead of
  fighting it over the same stage directories.
* Retries.  Every stage is resumable (`--resume` reuses completed outputs), so
  a retry after a GEE or earthaccess timeout costs only the failed stage.
* The ledger first.  `state-ledger advance` is a no-op once the ledger already
  reaches D-11, so it is safe to call every day, and calling it every day is
  what keeps the ledger from drifting behind the pipeline's requirement.

Credentials come from $HOME (`~/.netrc` for earthaccess, `~/.config/earthengine`
for GEE, `~/.conda/envs/firempc` for the vendored stages), so the unit must run
as the owning user, not as root or as a bare System=user with a stub home.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

import typer

import paths, vendorpaths

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

# The pipeline reads the certified state for D-11; see canfirecast.input_registry.
STATE_LAG_DAYS = 11


def resolve_target(value: str) -> date:
    return datetime.now(timezone.utc).date() if value.lower() == "today" else date.fromisoformat(value)


def run(command: list[str], log: Path) -> None:
    """Run one stage, appending its output to `log`, raising on failure."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write(f"\n=== {datetime.now(timezone.utc).isoformat()} {' '.join(command)}\n")
        handle.flush()
        result = subprocess.run(command, cwd=vendorpaths.OPERATIONAL_ROOT,
                                stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"exited {result.returncode}: {' '.join(command)} (see {log})")


def cli(*args: str) -> list[str]:
    return [sys.executable, str(vendorpaths.CLI), *args]


@app.command()
def main(
    target: Annotated[str, typer.Option("--target", help="Target date, or 'today'.")] = "today",
    release: Annotated[str, typer.Option("--release")] = os.environ.get("WILDFIRE_ASSET_RELEASE", "operational-v1"),
    attempts: Annotated[int, typer.Option("--attempts", min=1,
        help="Pipeline attempts before the run is reported failed.")] = 3,
    retry_wait: Annotated[int, typer.Option("--retry-wait",
        help="Seconds between attempts.")] = 900,
    purge: Annotated[bool, typer.Option("--purge/--no-purge",
        help="Delete raw granules and weather stages outside retention.")] = False,
    lock_timeout: Annotated[int, typer.Option("--lock-timeout",
        help="Seconds to wait for a run already in progress; 0 exits immediately.")] = 0,
) -> None:
    """Advance the ledger, run the operational pipeline, and record the result."""
    target_day = resolve_target(target)
    log_dir = Path(paths.REPORTS) / "daily" / target_day.isoformat()
    record = {
        "schema_version": 1, "target_date": target_day.isoformat(), "release": release,
        "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running", "attempts": [],
    }
    lock_path = Path(paths.DATA_ROOT) / ".daily.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("w")
    deadline = time.monotonic() + lock_timeout
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() >= deadline:
                typer.echo(f"another daily run holds {lock_path}; exiting", err=True)
                raise typer.Exit(75)  # EX_TEMPFAIL: try again at the next firing.
            time.sleep(30)
    lock.write(f"{os.getpid()} {target_day.isoformat()}\n")
    lock.flush()

    try:
        run(cli("canfirecast", "state-ledger", "advance",
                "--through", (target_day - timedelta(days=STATE_LAG_DAYS)).isoformat()),
            log_dir / "state-ledger.log")
        record["ledger_through"] = (target_day - timedelta(days=STATE_LAG_DAYS)).isoformat()

        for attempt in range(1, attempts + 1):
            try:
                run(cli("canfirecast", "pipeline", "--target", target_day.isoformat(),
                        "--release", release, "--resume"),
                    log_dir / "pipeline.log")
                record["attempts"].append({"attempt": attempt, "status": "complete"})
                break
            except RuntimeError as exc:
                record["attempts"].append({"attempt": attempt, "status": "failed", "error": str(exc)})
                if attempt == attempts:
                    raise
                typer.echo(f"attempt {attempt} failed, retrying in {retry_wait}s: {exc}", err=True)
                time.sleep(retry_wait)

        if purge:
            run(cli("modis", "purge", "--execute"), log_dir / "purge.log")
            run(cli("weather", "purge", "--execute"), log_dir / "purge.log")
            record["purged"] = True
        record["status"] = "complete"
    except Exception as exc:
        record.update({"status": "failed", "error": str(exc)})
        raise
    finally:
        record["ended_utc"] = datetime.now(timezone.utc).isoformat()
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "daily.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    typer.echo(f"daily run complete for {target_day.isoformat()}; log dir {log_dir}")


if __name__ == "__main__":
    app()
