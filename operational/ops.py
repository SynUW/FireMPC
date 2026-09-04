"""Shared plumbing for the stages this application runs.

Every domain here is a chain of stages driven as subprocesses through `cli.py`,
and each of them needs the same handful of things: a content hash, a durable
JSON write, a subprocess runner that logs, a way to slice a
`--from-stage`/`--to-stage` range out of an ordered tuple, and a run manifest
that records how the run ended.  Those had been written once per module -- five
copies of the hash, five of the JSON write, four of the runner, three of the
stage slice -- and they had drifted apart in ways that mattered:

* Two of the JSON writers renamed a file they had never fsynced, so a manifest
  could survive a crash describing a run that did not happen that way.
* Only one of the four runners put the log tail into the exception it raised,
  so the same failure was diagnosable from one pipeline and opaque from three.

One copy each, so a fix lands everywhere.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

import typer

import vendorpaths


# ---------------------------------------------------------------------------
# Identity and time
# ---------------------------------------------------------------------------
def sha256(path: str | Path, block_size: int = 1 << 20) -> str:
    """The content hash every manifest in this application records."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    """An instant, as manifests spell it: RFC 3339 with a `Z`."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_id() -> str:
    """The timestamp a run's report directory is named for."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# Durable writes
# ---------------------------------------------------------------------------
def atomic_json(path: Path, value: dict) -> None:
    """Write `value` to `path`; a reader sees the old file or the new one.

    The fsync before the rename is the point.  Without it the rename can reach
    the disk while the bytes behind it have not, which leaves a manifest that
    describes a run nobody performed -- the one failure mode a provenance
    record must not have.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Driving a stage
# ---------------------------------------------------------------------------
def cli(*args: str) -> list[str]:
    """The argv for one `cli.py` invocation.

    The pipelines re-invoke their stages rather than importing them, so a stage
    boundary is a process boundary: its failure arrives as an exit code and not
    as a half-mutated interpreter.  There is no installed console script and no
    package to `-m`, hence the absolute path.
    """
    return [sys.executable, str(vendorpaths.CLI), *args]


def run(
    command: Sequence[str],
    log: Path,
    *,
    cwd: Path = vendorpaths.OPERATIONAL_ROOT,
    append: bool = False,
    tolerate: tuple[int, ...] = (),
) -> int:
    """Run one stage into `log`, raising with its tail unless it exited cleanly.

    `append` keeps every stage of one target in a single log, which is what a
    range fill wants; the pipelines write one log per stage and overwrite.
    `tolerate` names exit codes that are information rather than failure -- the
    MODIS downloader's 2, "everything landed, but later than its measured p95".
    """
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a" if append else "w", encoding="utf-8") as handle:
        handle.write(f"=== {utc_now()} {' '.join(command)}\n\n")
        handle.flush()
        completed = subprocess.run(command, cwd=cwd, stdout=handle,
                                   stderr=subprocess.STDOUT, check=False)
    if completed.returncode and completed.returncode not in tolerate:
        raise RuntimeError(
            f"stage exited {completed.returncode}: {' '.join(command)} (see {log})\n"
            + log.read_text(encoding="utf-8", errors="replace")[-2000:]
        )
    return completed.returncode


def select_stages(stages: tuple[str, ...], from_stage: str, to_stage: str | None) -> tuple[str, ...]:
    """The slice of `stages` a `--from-stage`/`--to-stage` pair asks for.

    `to_stage` may be `None`, which means the last stage: a pipeline whose
    option defaults to `None` and one whose option names its own final stage
    both arrive here.
    """
    end = to_stage or stages[-1]
    if from_stage not in stages or end not in stages:
        raise typer.BadParameter(f"stages must be one of {stages}")
    lo, hi = stages.index(from_stage), stages.index(end)
    if hi < lo:
        raise typer.BadParameter("--to-stage precedes --from-stage")
    return stages[lo:hi + 1]


@contextmanager
def record(report_dir: Path, fields: dict, *, name: str = "manifest.json") -> Iterator[dict]:
    """Yield a run manifest that is written on the way out, however it exits.

    A run that crashed and a run that never started look identical on disk
    unless the failure is recorded, so the write is in `finally` and the status
    is set here rather than by each caller.  The body may still overwrite
    `status` -- a run that decides it was partial says so and is believed.
    """
    value = {"status": "running", **fields}
    try:
        yield value
    except Exception as exc:
        value.update({"status": "failed", "error": str(exc)})
        raise
    else:
        if value["status"] == "running":
            value["status"] = "complete"
    finally:
        value["ended_utc"] = utc_now()
        atomic_json(report_dir / name, value)


def lock(work_root: Path, name: str) -> IO[str]:
    """Take `work_root/<name>.lock` exclusively; the caller closes it to release.

    Purges delete files another run may be validating, so two of them must not
    overlap.  Non-blocking: a second run is told who holds the lock rather than
    waiting behind a job that may take hours.
    """
    work_root.mkdir(parents=True, exist_ok=True)
    path = work_root / f"{name}.lock"
    handle = path.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"another {name} run holds {path}") from exc
    return handle
