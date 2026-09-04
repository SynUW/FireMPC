"""Manifest-gated operational wrapper around the unmodified FireMPC scripts.

The wrapper deliberately owns orchestration, atomic publication, recursive
state continuity, and provenance.  Pixel algorithms remain in the checked-in
FireMPC lineage and run with its pinned interpreter.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

import numpy as np
import rasterio
import typer
from rasterio.errors import RasterioError

# Run directly as `python operational/modis/preprocess.py`: put the application root on
# sys.path so the shared modules import by bare name, as elsewhere in FireMPC.
import sys
from pathlib import Path
OPERATIONAL = Path(__file__).resolve().parents[1]
if str(OPERATIONAL) not in sys.path:
    sys.path.insert(0, str(OPERATIONAL))

import paths, vendorpaths
from ops import atomic_json, sha256, utc_now
from modis.processing_registry import ROUTES, require_operational_lineage

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
FIREMPC = vendorpaths.FIREMPC_NEWSET
DEFAULT_FIREMPC_PYTHON = Path(os.environ.get("FIREMPC_PYTHON", sys.executable))
ALL_STAGES = (
    "masked", "spatial_filled", "vegetation_indices", "downsampled",
    "merged", "unified", "daily_filled", "scaled",
)
DATE_RE = re.compile(r"(?<!\d)(\d{4})_(\d{2})_(\d{2})(?!\d)")


def spatial_fill_options(product: str) -> list[str]:
    """Product-specific flags for ``fill_modis_spatial_gap.py``.

    The convergence harness reproduces the operational recursive fill exactly,
    so both callers must build these flags from this one definition.
    """
    if product == "MCD09GA":
        return ["--valid_ranges=2,16000;2,16000;2,16000;2,16000"]
    if product == "MCD11A1":
        return ["--custom_nodata=32771,0"]
    if product == "MCD09CMG":
        # The CMG QA step writes 0 for masked pixels (registry masked_value),
        # and 0 is not the raster's nodata (-32765).  Without this the fill
        # reads a masked pixel as a valid 0 K observation and carries it
        # forward indefinitely instead of gap-filling it.  Verified against the
        # reference tree: at every zero-masked pixel the reference output holds
        # the previous day's value, which only happens if 0 counts as invalid.
        return ["--custom_nodata=0"]
    if product == "MCD15A3H":
        return ["--detect_permanent_invalid"]
    return []


# Stages that only resample, mosaic or copy.  None of them may produce a value
# outside the envelope of their own inputs: an average or a bilinear kernel over
# valid samples is bounded by those samples, and a mosaic and a copy are too.
# The registry's raw ranges cannot serve here -- they are scale_int_tif's
# training-period clipping extremes, which real observations legitimately
# exceed (the 2025 reference reaches 4321 on a band whose raw_range tops out at
# 1937), so they would flag correct data while missing the failure below.
GEOMETRY_STAGES = frozenset({"downsampled", "merged", "unified", "daily_filled"})
# Envelope tolerance for a date only one platform observed.  Set from the
# reference, not guessed: see Processor._tolerance_for.  Clear of the measured
# 3.09% with margin, still far under a fraction that would hide a real defect.
SINGLE_PLATFORM_RANGE_TOLERANCE = 0.05


@dataclass(frozen=True, slots=True)
class TileEntry:
    product: str
    day: date
    tile: int
    path: Path


def load_complete_tiles(
    report_path: Path, *, strict: bool = True,
) -> dict[tuple[str, date], tuple[TileEntry, ...]]:
    """Read only complete four-patch entries from an Earthaccess/GEE tiling report.

    ``strict`` rejects a report that recorded any failure at all. That is the
    right stance for preprocessing, which is asked to build one window and
    should not quietly proceed on a run known to be partial. It is the wrong
    stance for a reader sweeping the whole report history, where one bad
    granule in one old run would veto every date in the file -- failed jobs are
    already excluded entry-by-entry below, because ``run_manifest`` records
    them with ``status="failed"``.
    """
    with report_path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if strict and report.get("summary", {}).get("failures", 0):
        raise RuntimeError(f"tiling report contains failures: {report_path}")
    grouped: dict[tuple[str, date], dict[int, TileEntry]] = {}
    for result in report.get("results", []):
        if result.get("status") != "complete":
            continue
        product = result.get("product")
        if not product or product not in {item for route in ROUTES.values() for item in route.exports}:
            continue
        day = date.fromisoformat(result["date"])
        tiles = result.get("tiles", {})
        for tile_text, record in tiles.items():
            if record.get("status") not in {"existing", "written", "complete"}:
                continue
            tile = int(tile_text)
            path = Path(record["path"])
            if path.is_file():
                grouped.setdefault((product, day), {})[tile] = TileEntry(product, day, tile, path)
    complete = {}
    for key, entries in grouped.items():
        if set(entries) == {0, 1, 2, 3}:
            complete[key] = tuple(entries[index] for index in range(4))
    return complete


def available_platform_groups(
    product: str,
    day: date,
    tiles: dict[tuple[str, date], tuple[TileEntry, ...]],
) -> tuple[tuple[str, ...], ...]:
    """Platform groups whose every export has four complete tiles for `day`."""
    return tuple(
        group for group in ROUTES[product].platform_groups
        if all((export, day) in tiles for export in group)
    )


def complete_product_dates(
    product: str,
    tiles: dict[tuple[str, date], tuple[TileEntry, ...]],
) -> tuple[date, ...]:
    """Dates this route can process: any one complete platform is enough.

    Requiring every export would discard a day the reference pipeline
    published -- Terra-complete days with an Aqua granule hole are ordinary in
    the archive, and the combiners handle them by design.  See Route.
    """
    days = {day for _, day in tiles}
    return tuple(sorted(
        day for day in days if available_platform_groups(product, day, tiles)
    ))


@contextmanager
def exclusive_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".preprocess.lock"
    handle = path.open("a+")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another MODIS preprocessing run holds {path}") from exc
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


class Terminated(RuntimeError):
    """The run was asked to stop by a signal."""


@contextmanager
def terminate_on_signal() -> Iterator[None]:
    """Turn SIGTERM/SIGINT into an exception so cleanup actually runs.

    Python's default SIGTERM disposition kills the interpreter outright: no
    ``finally``, so the lock is dropped without the FireMPC child being reaped.
    A scheduler stopping a job sends exactly that signal, which is how the
    child ends up orphaned at ``ppid=1`` while a later run takes the lock and
    writes the same product and date concurrently.
    """
    def _raise(signum, _frame):
        raise Terminated(f"received {signal.Signals(signum).name}")

    previous = {
        signum: signal.signal(signum, _raise)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def terminate_group(process: subprocess.Popen, grace: float = 10.0) -> None:
    """Stop a child and everything it spawned, escalating TERM to KILL.

    Signalling the process alone is not enough: the FireMPC scripts fan out
    into their own workers, and those are what survive.
    """
    try:
        group = os.getpgid(process.pid)
    except ProcessLookupError:
        return
    for signum in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(group, signum)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue


class Processor:
    def __init__(
        self,
        *,
        data_root: Path,
        report_root: Path,
        template: Path,
        firempc_python: Path,
        resume: bool,
        overwrite: bool,
        range_tolerance: float = 1.0,
        bootstrap: bool = False,
        bridge_missing: bool = False,
    ) -> None:
        self.data_root = data_root
        self.report_root = report_root
        self.template = template
        self.python = firempc_python
        self.resume = resume
        self.overwrite = overwrite
        self.range_tolerance = range_tolerance
        self.bootstrap = bootstrap
        self.bridge_missing = bridge_missing
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def stage_dir(self, stage: str, product: str) -> Path:
        return self.data_root / stage / product

    def _manifest_path(self, product: str, day: date, stage: str) -> Path:
        return self.report_root / product / day.isoformat() / f"{stage}.json"

    def _validated_outputs(self, outputs: list[Path], bands: int | None = None) -> bool:
        if not outputs or not all(path.is_file() for path in outputs):
            return False
        try:
            for path in outputs:
                with rasterio.open(path) as dataset:
                    if bands is not None and dataset.count != bands:
                        return False
                    if dataset.width <= 0 or dataset.height <= 0:
                        return False
            return True
        except RasterioError:
            return False

    def _resumable_manifest(self, manifest: dict, product: str, day: date, stage: str) -> bool:
        if manifest.get("status") != "complete":
            return False
        if not all(
            Path(item["path"]).is_file()
            and sha256(Path(item["path"])) == item["sha256"]
            for item in manifest.get("outputs", [])
        ):
            return False
        if not all(
            Path(item["path"]).is_file()
            and sha256(Path(item["path"])) == item["sha256"]
            for item in manifest.get("inputs", [])
        ):
            return False
        if stage == "daily_filled" and not self._daily_filled_source_unchanged(manifest, product, day):
            return False
        if stage != "spatial_filled":
            return True
        predecessor = manifest.get("predecessor")
        if not predecessor:
            return False
        predecessor_path = Path(predecessor["path"])
        predecessor_day = _date_from_name(predecessor_path.name)
        return bool(
            predecessor_day
            and 0 < (day - predecessor_day).days <= ROUTES[product].cadence_days
            and predecessor_path.is_file()
            and sha256(predecessor_path) == predecessor["sha256"]
        )

    def _daily_filled_source_unchanged(self, manifest: dict, product: str, day: date) -> bool:
        """False if a closer ``unified`` source has appeared since this manifest ran.

        ``stage_daily_filled`` forward-copies from the newest ``unified``
        raster on or before ``day``.  A manifest recorded while that source
        was still stale (e.g. it copied 08-08 forward because 08-09 hadn't
        landed yet) must not be treated as resumable once 08-09 lands --
        otherwise the forward-copy from 08-08 survives forever.
        """
        command = manifest.get("command") or []
        if len(command) != 3 or command[0] != "atomic-copy":
            return False
        recorded_source_day = date.fromisoformat(command[1])
        available = {
            parsed: path for path in self.stage_dir("unified", product).glob("*.tif")
            if (parsed := _date_from_name(path.name)) is not None and parsed <= day
        }
        if not available:
            return False
        return max(available) == recorded_source_day

    @staticmethod
    def _sentinels(product: str) -> tuple[float, ...]:
        """Values this route uses to mark a pixel invalid, beyond declared nodata.

        The stages disagree about how invalidity is spelled: the maskers write
        the route's ``fill_nodata`` sentinels, the FireMPC downsampler writes
        its own ``invalid_value``, and the two-platform combiner leaves NaN with
        no nodata tag at all.  A range check that does not know all three
        measures the sentinels rather than the data.
        """
        route = ROUTES[product]
        values = set(route.fill_nodata)
        if route.downsample_invalid is not None:
            values.add(route.downsample_invalid)
        return tuple(sorted(values))

    @staticmethod
    def _valid_pixels(data, nodata: float | None, sentinels: tuple[float, ...]):
        """Pixels that carry a measurement: not nodata, not a sentinel, not NaN."""
        mask = np.ones(data.shape, dtype=bool)
        if data.dtype.kind == "f":
            # NaN is why this gate was vacuous: an envelope of (nan, nan) makes
            # every subsequent comparison False, so nothing is ever outside it.
            mask &= ~np.isnan(data)
        if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
            mask &= data != nodata
        for value in sentinels:
            mask &= data != value
        return data[mask]

    def _band_envelope(self, product: str, paths_in: list[Path]) -> dict[int, tuple[float, float]]:
        """Per-band (min, max) over the valid pixels of every raster given."""
        sentinels = self._sentinels(product)
        envelope: dict[int, tuple[float, float]] = {}
        for path in paths_in:
            with rasterio.open(path) as dataset:
                for index in range(1, dataset.count + 1):
                    data = dataset.read(index)
                    valid = self._valid_pixels(data, dataset.nodata, sentinels)
                    if not valid.size:
                        continue
                    low, high = float(valid.min()), float(valid.max())
                    if index in envelope:
                        low = min(low, envelope[index][0])
                        high = max(high, envelope[index][1])
                    envelope[index] = (low, high)
        return envelope

    def _range_report(
        self, product: str, stage: str, inputs: list[Path], outputs: list[Path],
    ) -> list[dict]:
        """Validation gate 2: per-band value range at every stage.

        On the geometry stages the recorded range is also checked against the
        input envelope.  Bilinear alignment blends nodata into partially
        observed output pixels, so a pixel no day has observed can leave
        ``unified`` as a large negative number instead of as nodata -- which
        scaling then silently clips back into range.

        Sentinels are excluded from both sides of that comparison.  A
        downsampled raster is *expected* to carry the route's invalid value
        wherever its input was invalid -- the reference tree is 22.6% -32768 on
        an ordinary MCD11A1 date, dominated by permanently invalid ocean -- so
        counting sentinels would flag every date.  The blending artifact this
        gate exists for (values near -31539) is not a sentinel and is still
        caught.
        """
        check = stage in GEOMETRY_STAGES
        sentinels = self._sentinels(product)
        envelope = self._band_envelope(product, inputs) if check else {}
        report: list[dict] = []
        for path in outputs:
            with rasterio.open(path) as dataset:
                for index in range(1, dataset.count + 1):
                    data = dataset.read(index)
                    nodata = dataset.nodata
                    declared = data if nodata is None else data[data != nodata]
                    valid = self._valid_pixels(data, nodata, sentinels)
                    entry: dict = {
                        "path": str(path),
                        "band": index,
                        # Declared-nodata only, unchanged so earlier manifests
                        # stay comparable; invalid_fraction is the honest one.
                        "nodata_fraction": round(float(1.0 - declared.size / data.size), 6),
                        "invalid_fraction": round(float(1.0 - valid.size / data.size), 6),
                        "min": None if not valid.size else float(valid.min()),
                        "max": None if not valid.size else float(valid.max()),
                    }
                    bounds = envelope.get(index)
                    if bounds is not None and valid.size:
                        low, high = bounds
                        outside = (valid < low) | (valid > high)
                        entry["input_envelope"] = [low, high]
                        entry["outside_envelope_fraction"] = round(float(outside.mean()), 6)
                    elif check:
                        # A geometry stage that could not be checked must say so
                        # rather than look like a stage that passed.
                        entry["envelope_unavailable"] = True
                    report.append(entry)
        return report

    def _single_platform(self, product: str, day: date) -> bool:
        """Did this date's masker combine only one platform?"""
        manifest = self._manifest_path(product, day, "masked")
        if not manifest.is_file():
            return False
        try:
            with manifest.open(encoding="utf-8") as handle:
                return bool(json.load(handle).get("single_platform"))
        except (OSError, json.JSONDecodeError):
            return False

    def _tolerance_for(self, product: str, day: date) -> float:
        """The envelope tolerance this date is held to.

        A single-platform date leaves far more interior nodata for the
        alignment warp to blend against, so it sits about fifty times higher
        than an ordinary date -- and the reference does too.  Measured on band
        3 of unified MCD11A1, in this gate's own denominator (fraction of valid
        pixels):

            reference 2003-12-18, Aqua-only   0.030916
            ours      2026-08-15, Terra-only  0.030891
            reference 2003-12-16, both        0.000453
            ours      2026-08-14, both        0.000696

        Holding a single-platform date to the two-platform tolerance rejects
        data FireMPC published.
        """
        if self.range_tolerance >= SINGLE_PLATFORM_RANGE_TOLERANCE:
            return self.range_tolerance
        return (
            SINGLE_PLATFORM_RANGE_TOLERANCE
            if self._single_platform(product, day) else self.range_tolerance
        )

    def _enforce_range(self, product: str, day: date, stage: str, report: list[dict]) -> None:
        tolerance = self._tolerance_for(product, day)
        offenders = [
            entry for entry in report
            if entry.get("outside_envelope_fraction", 0.0) > tolerance
        ]
        if offenders:
            worst = max(offenders, key=lambda entry: entry["outside_envelope_fraction"])
            raise RuntimeError(
                f"{product} {day} {stage}: {len(offenders)} band(s) left the input "
                f"envelope by more than the tolerance {tolerance}; worst "
                f"is band {worst['band']} of {Path(worst['path']).name} at "
                f"{worst['outside_envelope_fraction']} (min {worst['min']}, "
                f"max {worst['max']}, input envelope {worst['input_envelope']})"
            )

    def _run(self, command: list[str], cwd: Path, log: Path) -> None:
        log.parent.mkdir(parents=True, exist_ok=True)
        environment = {**os.environ, "PYTHONNOUSERSITE": "1"}
        prefix = self.python.parent.parent
        for variable, relative in (
            ("PROJ_LIB", "share/proj"),
            ("PROJ_DATA", "share/proj"),
            ("GDAL_DATA", "share/gdal"),
        ):
            data_dir = prefix / relative
            if data_dir.is_dir():
                environment[variable] = str(data_dir)
        environment["CONDA_PREFIX"] = str(prefix)
        environment["PATH"] = os.pathsep.join(
            (str(prefix / "bin"), environment.get("PATH", ""))
        )
        with log.open("w", encoding="utf-8") as handle:
            handle.write(" ".join(command) + "\n\n")
            handle.flush()
            # Own process group, so that abandoning this run can take the whole
            # FireMPC process tree with it rather than leaving workers behind
            # holding memory and writing the product we just released the lock on.
            process = subprocess.Popen(  # noqa: S603 - command is built from the registry.
                command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT,
                env=environment, start_new_session=True,
            )
            try:
                returncode = process.wait()
            except BaseException:
                terminate_group(process)
                raise
        if returncode:
            raise RuntimeError(f"command exited {returncode}; see {log}")

    @contextmanager
    def _work(self, product: str, day: date, stage: str) -> Iterator[Path]:
        base = Path(paths.WORK)
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f"modis-{product}-{day}-{stage}-", dir=base,
        ) as temporary:
            yield Path(temporary)

    @staticmethod
    def _farm(folder: Path, entries: list[Path]) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        for source in entries:
            target = folder / source.name
            if not target.exists():
                target.symlink_to(source.resolve())

    def _publish(self, sources: list[Path], destination: Path) -> list[Path]:
        destination.mkdir(parents=True, exist_ok=True)
        published = []
        for source in sources:
            target = destination / source.name
            if target.exists() and not self.overwrite:
                published.append(target)
                continue
            temporary = destination / f".{source.name}.{os.getpid()}.tmp"
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
            published.append(target)
        return published

    def _record(
        self,
        product: str,
        day: date,
        stage: str,
        started: str,
        status: str,
        inputs: list[Path],
        outputs: list[Path],
        *,
        command: list[str] | None = None,
        error: str | None = None,
        predecessor: Path | None = None,
        ranges: list[dict] | None = None,
        extra: dict | None = None,
    ) -> None:
        manifest = {
            "schema_version": 1, "run_id": self.run_id, "product": product,
            "date": day.isoformat(), "stage": stage, "started_utc": started,
            "ended_utc": utc_now(), "status": status,
            "inputs": [{"path": str(path), "sha256": sha256(path)} for path in inputs if path.is_file()],
            "outputs": [{"path": str(path), "sha256": sha256(path)} for path in outputs if path.is_file()],
            "command": command, "firempc_python": str(self.python), "error": error,
            "band_ranges": ranges,
            "predecessor": (
                {"path": str(predecessor), "sha256": sha256(predecessor)}
                if predecessor and predecessor.is_file() else None
            ),
        }
        manifest.update(extra or {})
        atomic_json(self._manifest_path(product, day, stage), manifest)

    def run_stage(
        self,
        product: str,
        day: date,
        stage: str,
        source_tiles: dict[tuple[str, date], tuple[TileEntry, ...]],
    ) -> None:
        started = utc_now()
        inputs: list[Path] = []
        outputs: list[Path] = []
        command = None
        predecessor = None
        ranges: list[dict] | None = None
        # Stages that resolve provenance a caller cannot infer from the command
        # (currently which platforms the masker actually combined) leave it here.
        self._stage_extra: dict = {}
        try:
            if stage == "spatial_filled":
                candidates = [
                    path for path in self.stage_dir(stage, product).glob("*.tif")
                    if (_date_from_name(path.name) or day) < day
                ]
                if candidates:
                    predecessor = max(candidates, key=lambda path: _date_from_name(path.name) or date.min)
            method = getattr(self, f"stage_{stage}")
            inputs, outputs, command = method(product, day, source_tiles)
            ranges = self._range_report(product, stage, inputs, outputs)
            if stage in GEOMETRY_STAGES:
                # Carry the masker's finding forward, so a stage held to the
                # looser tolerance records why.
                tolerance = self._tolerance_for(product, day)
                self._stage_extra["range_tolerance_applied"] = tolerance
                if tolerance != self.range_tolerance:
                    self._stage_extra["single_platform"] = True
            self._record(
                product, day, stage, started, "complete", inputs, outputs,
                command=command, predecessor=predecessor, ranges=ranges,
                extra=self._stage_extra or None,
            )
            self._enforce_range(product, day, stage, ranges)
        except Exception as exc:
            self._record(
                product, day, stage, started, "failed", inputs, outputs,
                command=command, error=str(exc), ranges=ranges,
                extra=self._stage_extra or None,
            )
            raise

    def _tile_paths(self, product: str, day: date, source_stage: str) -> list[Path]:
        folder = self.stage_dir(source_stage, product)
        token = day.strftime("%Y_%m_%d")
        return sorted(folder.glob(f"*{token}*tile*.tif"))

    def stage_masked(self, product, day, source_tiles):
        if product == "NDVIEVI":
            raise RuntimeError("NDVIEVI starts at vegetation_indices")
        route = ROUTES[product]
        groups = available_platform_groups(product, day, source_tiles)
        if not groups:
            raise RuntimeError(f"{product} {day}: no complete platform")
        available = {export for group in groups for export in group}
        # An absent platform contributes an empty directory, which every
        # combiner handles: they key on the union of MOD and MYD, so the
        # surviving platform's pixels pass through unchanged.
        raw = [entry.path for export in route.exports if export in available
               for entry in source_tiles[(export, day)]]
        self._stage_extra = {
            "platforms": ["MOD" if group[0].startswith("MOD") else
                          "MYD" if group[0].startswith("MYD") else "MCD"
                          for group in groups],
            "single_platform": len(groups) < len(route.platform_groups),
        }
        with self._work(product, day, "masked") as work:
            out = work / "out"
            if product == "MCD09GA":
                with_qa = []
                for platform in ("MOD", "MYD"):
                    reflectance = work / f"{platform}09GA_b1237"
                    qa = work / f"{platform}09GA_QA_state1km"
                    combined = work / f"{platform}09GA_b1237_withQA"
                    if f"{platform}09GA_b1237" not in available:
                        combined.mkdir(parents=True, exist_ok=True)
                        with_qa.append(combined)
                        continue
                    self._farm(reflectance, [e.path for e in source_tiles[(f"{platform}09GA_b1237", day)]])
                    self._farm(qa, [e.path for e in source_tiles[(f"{platform}09GA_QA_state1km", day)]])
                    cmd = [str(self.python), str(FIREMPC / "merge_mcd09ga_qa.py"),
                           "--input-dir1", str(reflectance), "--input-dir2", str(qa),
                           "--output-dir", str(combined), "--overwrite"]
                    self._run(cmd, work, work / f"{platform}-with-qa.log")
                    with_qa.append(combined)
                command = [str(self.python), str(FIREMPC / "mask_low_quality_data_and_combine/mcd09ga_maskout_combine.py"),
                           "--mod_dir", str(with_qa[0]), "--myd_dir", str(with_qa[1]),
                           "--output_dir", str(out), "--overwrite"]
            elif product in {"MCD11A1", "MCD09CMG"}:
                stem = "11A1" if product == "MCD11A1" else "09CMG"
                mod, myd = work / f"MOD{stem}", work / f"MYD{stem}"
                for platform, folder in (("MOD", mod), ("MYD", myd)):
                    if f"{platform}{stem}" in available:
                        self._farm(folder, [e.path for e in source_tiles[(f"{platform}{stem}", day)]])
                    else:
                        folder.mkdir(parents=True, exist_ok=True)
                script = "mcd11a1_mask_combine.py" if product == "MCD11A1" else "mcd09cmg_mask_combined.py"
                command = [str(self.python), str(FIREMPC / "mask_low_quality_data_and_combine" / script),
                           "--mod_in", str(mod), "--myd_in", str(myd),
                           "--mod_out", str(work / "mod-masked"), "--myd_out", str(work / "myd-masked"),
                           "--mcd_out", str(out), "--overwrite"]
            elif product == "MCD15A3H":
                source = work / "MCD15A3H"
                self._farm(source, raw)
                command = [str(self.python), str(FIREMPC / "mask_low_quality_data_and_combine/mcd15a3h_maskout.py"),
                           "--input_dir", str(source), "--output_dir", str(out), "--overwrite"]
            else:  # MCD12Q1: the historical ``*_right`` route is band extraction only.
                out.mkdir(parents=True)
                for entry in source_tiles[("MCD12Q1", day)]:
                    target = out / entry.path.name
                    with rasterio.open(entry.path) as source:
                        land = source.read(1)
                        profile = source.profile.copy()
                        profile.update(count=1, dtype="uint8", nodata=0)
                        with rasterio.open(target, "w", **profile) as dest:
                            dest.write(land, 1)
                command = ["native", "mcd12q1_retain_lc_type1_band_only"]
            if product != "MCD12Q1":
                self._run(command, work, work / "masked.log")
            made = sorted(out.glob("*.tif"))
            if len(made) != 4:
                raise RuntimeError(f"{product} {day}: masker produced {len(made)} tiles")
            outputs = self._publish(made, self.stage_dir("qa_combined", product))
        return raw, outputs, command

    def stage_spatial_filled(self, product, day, source_tiles):
        route = ROUTES[product]
        if not route.spatial_fill:
            raise RuntimeError(f"{product} has no spatial-fill stage")
        inputs = self._tile_paths(product, day, "qa_combined")
        if len(inputs) != 4:
            raise RuntimeError(f"{product} {day}: expected four masked tiles")
        destination = self.stage_dir("spatial_filled", product)
        destination.mkdir(parents=True, exist_ok=True)
        state_by_day: dict[date, list[Path]] = {}
        for path in destination.glob("*.tif"):
            parsed = _date_from_name(path.name)
            if parsed is not None and parsed < day:
                state_by_day.setdefault(parsed, []).append(path)
        if not state_by_day and not self.bootstrap:
            raise RuntimeError(
                f"{product} {day}: no recursive predecessor; bootstrap state is required"
            )
        # A cold start runs the fill into an empty state folder, which is the
        # same path FireMPC itself took for its first day.  The state it
        # produces is not converged: it is warm-up, not publishable output.
        predecessor = max(state_by_day) if state_by_day else None
        predecessor_tiles = state_by_day[predecessor] if predecessor else []
        if predecessor is not None:
            tile_numbers = {
                int(match.group(1))
                for path in predecessor_tiles
                if (match := re.search(r"tile(\d{2})", path.name))
            }
            if tile_numbers != {0, 1, 2, 3}:
                raise RuntimeError(
                    f"{product}: predecessor {predecessor} is incomplete; "
                    f"found tiles {sorted(tile_numbers)}"
                )
            if (day - predecessor).days > route.cadence_days:
                if not self.bridge_missing:
                    raise RuntimeError(
                        f"{product}: recursive state gap {predecessor} -> {day}; refusing to advance")
                predecessor, predecessor_tiles = self._bridge_missing(
                    product, predecessor, predecessor_tiles, day, source_tiles)
        with self._work(product, day, "spatial_filled") as work:
            incoming = work / "incoming"
            # The permanent-invalid detector and future seeding need the
            # complete masked history, even though only this date is advanced.
            historical_inputs = [
                path for path in self.stage_dir("qa_combined", product).glob("*.tif")
                if (_date_from_name(path.name) or day) <= day
            ]
            self._farm(incoming, historical_inputs)
            staged_state = work / "state"
            previous_outputs = predecessor_tiles
            self._farm(staged_state, previous_outputs)
            command = [str(self.python), str(FIREMPC / "fill_modis_spatial_gap.py"),
                       "--input_folder", str(incoming), "--output_folder", str(staged_state),
                       "--keep_bands", ",".join(str(band) for band in route.retained_bands)]
            command += spatial_fill_options(product)
            self._run(command, work, work / "spatial-fill.log")
            token = day.strftime("%Y_%m_%d")
            made = sorted(staged_state.glob(f"*{token}*tile*.tif"))
            outputs = self._publish(made, destination)
        if len(outputs) != 4:
            raise RuntimeError(f"{product} {day}: spatial fill produced {len(outputs)} tiles")
        state = {
            "date": day.isoformat(), "bootstrap": predecessor is None,
            "predecessor_date": predecessor.isoformat() if predecessor else None,
            "outputs": [{"path": str(path), "sha256": sha256(path)} for path in outputs],
        }
        atomic_json(destination / "state.json", state)
        return inputs, outputs, command

    def _bridge_missing(
        self, product, predecessor, predecessor_tiles, day, source_tiles,
    ) -> tuple[date, list[Path]]:
        """Carry the recursive state across a source date that will never exist.

        A day whose granules were never published is the degenerate case of what
        the spatial fill already does for every unobserved pixel: nothing is
        seen, so every pixel keeps its previous value.  Bridging it is therefore
        the same operation as the fill itself, not an exception to it.

        This does NOT weaken the continuity gate into "skip on gap".  A date is
        only bridged when the tiling manifest proves it has no complete source;
        a date that *could* have been processed still raises, because that gap
        means a processing failure rather than a publication hole.
        """
        route = ROUTES[product]
        destination = self.stage_dir("spatial_filled", product)
        observed = set(complete_product_dates(product, source_tiles))
        current, tiles = predecessor, list(predecessor_tiles)
        while (day - current).days > route.cadence_days:
            missing = current + timedelta(days=route.cadence_days)
            if missing in observed:
                raise RuntimeError(
                    f"{product}: {missing} has complete source data but no "
                    "spatial-fill state; process that date instead of bridging it")
            started = utc_now()
            token = missing.strftime("%Y_%m_%d")
            carried: list[Path] = []
            for source in sorted(tiles):
                target = destination / DATE_RE.sub(token, source.name, count=1)
                temporary = destination / f".{target.name}.{os.getpid()}.tmp"
                shutil.copy2(source, temporary)
                os.replace(temporary, target)
                carried.append(target)
            if len(carried) != 4:
                raise RuntimeError(
                    f"{product} {missing}: bridged {len(carried)} tiles, expected four")
            self._record(
                product, missing, "spatial_filled", started, "complete",
                list(tiles), carried,
                command=["bridge-unpublished-source", current.isoformat(), missing.isoformat()],
                predecessor=min(tiles),
                extra={"carried_forward": True, "carried_from": current.isoformat()},
            )
            typer.echo(
                f"{product} {missing} spatial_filled (bridged from {current}: "
                "no source granules were ever published)")
            current, tiles = missing, carried
        return current, tiles

    def stage_vegetation_indices(self, product, day, source_tiles):
        if product != "NDVIEVI":
            raise RuntimeError("vegetation_indices is only valid for NDVIEVI")
        token = day.strftime("%Y_%m_%d")
        inputs = sorted(self.stage_dir("scaled", "MCD09GA").glob(f"{token}.tif"))
        if len(inputs) != 1:
            raise RuntimeError(
                f"NDVIEVI {day}: expected one scaled national MCD09GA raster")
        with rasterio.open(inputs[0]) as reflectance, rasterio.open(self.template) as template:
            if reflectance.count != 4 or reflectance.dtypes != ("uint8",) * 4:
                raise RuntimeError(
                    f"{inputs[0]} must contain four uint8 reflectance bands")
            if (
                reflectance.shape != template.shape
                or reflectance.crs != template.crs
                or not reflectance.transform.almost_equals(template.transform)
            ):
                raise RuntimeError(f"{inputs[0]} is not on GRID_TEMPLATE")
        with self._work(product, day, "vegetation_indices") as work:
            source, out = work / "in", work / "out"
            self._farm(source, inputs)
            command = [str(self.python), str(FIREMPC / "compute_ndvi_evi.py"),
                       "--input-dir", str(source), "--output-dir", str(out),
                       "--max-workers", "1", "--overwrite"]
            self._run(command, work, work / "vegetation-indices.log")
            made = sorted(out.glob("*.tif"))
            if len(made) != 1:
                raise RuntimeError(f"NDVIEVI {day}: produced {len(made)} rasters")
            outputs = self._publish(made, self.stage_dir("vegetation_indices", product))
        with rasterio.open(outputs[0]) as result, rasterio.open(self.template) as template:
            if result.count != 2 or result.dtypes != ("float32", "float32"):
                raise RuntimeError(f"{outputs[0]} must contain two float32 index bands")
            if (
                result.shape != template.shape
                or result.crs != template.crs
                or not result.transform.almost_equals(template.transform)
            ):
                raise RuntimeError(f"{outputs[0]} is not on GRID_TEMPLATE")
        return inputs, outputs, command

    def stage_downsampled(self, product, day, source_tiles):
        route = ROUTES[product]
        if route.downsample_factor is None:
            raise RuntimeError(f"{product} skips downsampling")
        if product == "MCD12Q1":
            token = day.strftime("%Y_%m_%d")
            inputs = sorted(self.stage_dir("merged", product).glob(f"*{token}*.tif"))
            expected = 1
        else:
            inputs = self._tile_paths(product, day, "spatial_filled")
            expected = 4
        if len(inputs) != expected:
            raise RuntimeError(f"{product} {day}: expected {expected} inputs for downsampling")
        with self._work(product, day, "downsampled") as work:
            source, output_root = work / product, work / "out"
            self._farm(source, inputs)
            shim = work / "run_downsample.py"
            mode_max_class = 17 if product == "MCD12Q1" else 255
            shim.write_text(
                "import sys\nfrom pathlib import Path\n"
                f"sys.path.insert(0, {str(FIREMPC)!r})\n"
                "from downsample_drivers import AdvancedDownsampler\n"
                f"cfg={{'downsample_method':{route.downsample_method!r},'invalid_value':{route.downsample_invalid!r},'qa_bands':[]}}\n"
                f"ds=AdvancedDownsampler(downsample_factor={route.downsample_factor},"
                f"stride={route.downsample_stride},max_workers=8,skip_existing=False,"
                f"mode_max_class_value={mode_max_class})\n"
                "ok,n=ds.process_directory(sys.argv[1],sys.argv[2],cfg)\n"
                "raise SystemExit(0 if ok==n else 1)\n",
                encoding="utf-8",
            )
            output_root.mkdir()
            command = [str(self.python), str(shim), str(source), str(output_root)]
            self._run(command, work, work / "downsample.log")
            made = sorted(output_root.glob("*.tif"))
            if len(made) != expected:
                raise RuntimeError(f"{product} {day}: downsampler produced {len(made)} tiles")
            outputs = self._publish(made, self.stage_dir("downsampled", product))
        return inputs, outputs, command

    def stage_merged(self, product, day, source_tiles):
        if product == "NDVIEVI":
            raise RuntimeError("NDVIEVI is already national and must not be merged")
        elif product in {"MCD09CMG"}:
            source_stage = "spatial_filled"
        elif product == "MCD12Q1":
            source_stage = "qa_combined"
        else:
            source_stage = "downsampled"
        inputs = self._tile_paths(product, day, source_stage)
        if len(inputs) != 4:
            raise RuntimeError(f"{product} {day}: expected four merge patches")
        with self._work(product, day, "merged") as work:
            source, out = work / "in", work / "out"
            self._farm(source, inputs)
            command = [str(self.python), str(FIREMPC / "merge_patches.py"),
                       "--input-dir", str(source), "--output-dir", str(out),
                       "--date", day.strftime("%Y_%m_%d"), "--overwrite"]
            self._run(command, work, work / "merge.log")
            made = sorted(out.glob("*.tif"))
            if len(made) != 1:
                raise RuntimeError(f"{product} {day}: merge produced {len(made)} rasters")
            outputs = self._publish(made, self.stage_dir("merged", product))
        return inputs, outputs, command

    def stage_unified(self, product, day, source_tiles):
        token = day.strftime("%Y_%m_%d")
        source_stage = "downsampled" if product == "MCD12Q1" else "merged"
        inputs = sorted(self.stage_dir(source_stage, product).glob(f"*{token}*.tif"))
        if len(inputs) != 1:
            raise RuntimeError(f"{product} {day}: expected one merged raster")
        with self._work(product, day, "unified") as work:
            source, out = work / product, work / "out"
            self._farm(source, inputs)
            command = [str(self.python), str(FIREMPC / "pre_processing_fast.py"),
                       "--input-root", str(source), "--output-root", str(out),
                       "--grid-template", str(self.template), "--force"]
            self._run(command, work, work / "unify.log")
            made = sorted(out.rglob("*.tif"))
            if len(made) != 1:
                raise RuntimeError(f"{product} {day}: alignment produced {len(made)} rasters")
            outputs = self._publish(made, self.stage_dir("unified", product))
        with rasterio.open(outputs[0]) as result, rasterio.open(self.template) as template:
            if result.shape != template.shape or result.crs != template.crs or not result.transform.almost_equals(template.transform):
                raise RuntimeError(f"{outputs[0]} is not on GRID_TEMPLATE")
        return inputs, outputs, command

    def stage_daily_filled(self, product, day, source_tiles):
        available = {
            parsed: path for path in self.stage_dir("unified", product).glob("*.tif")
            if (parsed := _date_from_name(path.name)) is not None and parsed <= day
        }
        if not available:
            raise RuntimeError(f"{product} {day}: no unified source on or before date")
        source_day = max(available)
        source = available[source_day]
        destination = self.stage_dir("daily_filled", product)
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / f"{day.strftime('%Y_%m_%d')}.tif"
        if not target.exists() or self.overwrite:
            temporary = destination / f".{target.name}.{os.getpid()}.tmp"
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
        return [source], [target], ["atomic-copy", source_day.isoformat(), day.isoformat()]

    def stage_scaled(self, product, day, source_tiles):
        token = day.strftime("%Y_%m_%d")
        source_stage = "vegetation_indices" if product == "NDVIEVI" else "daily_filled"
        inputs = sorted(self.stage_dir(source_stage, product).glob(f"*{token}*.tif"))
        if len(inputs) != 1:
            raise RuntimeError(f"{product} {day}: expected one daily-filled raster")
        if product == "MCD12Q1":
            outputs = self._publish(inputs, self.stage_dir("scaled", product))
            return inputs, outputs, ["atomic-copy-categorical"]
        with self._work(product, day, "scaled") as work:
            root = work / "in"
            source = root / product
            out_root = work / "out"
            self._farm(source, inputs)
            command = [str(self.python), str(FIREMPC / "scale_int_tif.py"),
                       "--input_folder", str(root), "--output_folder", str(out_root),
                       "--local-stage-root", str(work / "scratch"), "--max-workers", "1"]
            self._run(command, work, work / "scale.log")
            made = sorted(out_root.rglob("*.tif"))
            if len(made) != 1:
                raise RuntimeError(f"{product} {day}: scaling produced {len(made)} rasters")
            outputs = self._publish(made, self.stage_dir("scaled", product))
        return inputs, outputs, command


def _date_from_name(name: str) -> date | None:
    match = DATE_RE.search(name)
    if match is None:
        return None
    return date(*(int(value) for value in match.groups()))


def _selected_stages(product: str, stage: str) -> tuple[str, ...]:
    configured = tuple(item for item in ROUTES[product].stages if item in ALL_STAGES)
    if stage == "all":
        return configured
    if stage not in configured:
        raise typer.BadParameter(f"{product} does not use stage {stage!r}")
    return (stage,)


@app.command()
def main(
    product: Annotated[str, typer.Option("--product")] = "all",
    start: Annotated[str | None, typer.Option("--start")] = None,
    end: Annotated[str | None, typer.Option("--end")] = None,
    stage: Annotated[str, typer.Option("--stage")] = "all",
    tiling_manifest: Annotated[Path, typer.Option("--tiling-manifest")] = Path(paths.REPORTS) / "tiling/latest.json",
    processed_root: Annotated[Path, typer.Option("--processed-root")] = Path(paths.PROCESSED_MODIS),
    report_root: Annotated[Path, typer.Option("--report-root")] = Path(paths.MODIS_PROCESSING_REPORTS),
    template: Annotated[Path, typer.Option("--grid-template")] = Path(paths.GRID_TEMPLATE),
    firempc_python: Annotated[Path, typer.Option("--firempc-python")] = DEFAULT_FIREMPC_PYTHON,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    range_tolerance: Annotated[float, typer.Option(
        "--range-tolerance",
        help="Fail a geometry stage when the fraction of pixels outside its "
             "input envelope exceeds this. Defaults to 1.0 (record only); a "
             "warmed-up operational run should pass a small value such as 0.001.",
    )] = 1.0,
    bootstrap: Annotated[bool, typer.Option(
        "--bootstrap",
        help="Allow a product's first date to cold-start its recursive spatial "
             "fill from empty state. The dates it produces are warm-up, not "
             "publishable output.",
    )] = False,
    bridge_missing: Annotated[bool, typer.Option(
        "--bridge-missing",
        help="Carry the recursive state across a source date the tiling "
             "manifest proves was never published, recording it as "
             "carried_forward. A date that has complete source data still "
             "raises rather than being bridged.",
    )] = False,
) -> None:
    if resume and overwrite:
        raise typer.BadParameter("--resume and --overwrite are mutually exclusive")
    if start is None or end is None:
        raise typer.BadParameter("both --start and --end are required")
    if product in {"all", "NDVIEVI"}:
        require_operational_lineage()
    products = tuple(ROUTES) if product == "all" else (product,)
    bad = set(products) - set(ROUTES)
    if bad:
        raise typer.BadParameter(f"unknown product(s): {sorted(bad)}")
    begin, finish = date.fromisoformat(start), date.fromisoformat(end)
    if finish < begin:
        raise typer.BadParameter("--end must not precede --start")
    tiles = load_complete_tiles(tiling_manifest)
    processor = Processor(
        data_root=processed_root, report_root=report_root, template=template,
        firempc_python=firempc_python, resume=resume, overwrite=overwrite,
        range_tolerance=range_tolerance, bootstrap=bootstrap,
        bridge_missing=bridge_missing,
    )
    with terminate_on_signal(), exclusive_lock(processed_root):
        for name in products:
            # Derived NDVIEVI follows dates proven complete for MCD09GA.
            source_name = "MCD09GA" if name == "NDVIEVI" else name
            observed_dates = [
                day for day in complete_product_dates(source_name, tiles)
                if begin <= day <= finish
            ]
            calendar_dates = [
                begin + timedelta(days=offset)
                for offset in range((finish - begin).days + 1)
            ]
            # Run a whole stage across its dates before advancing.  This is
            # required for recursive fill continuity and for daily/scaled
            # stages to see every preceding composite output.
            for selected in _selected_stages(name, stage):
                dates = (
                    calendar_dates
                    if selected in {"vegetation_indices", "daily_filled", "scaled"}
                    else observed_dates
                )
                for day in dates:
                    manifest = processor._manifest_path(name, day, selected)
                    if resume and manifest.is_file():
                        with manifest.open(encoding="utf-8") as handle:
                            old = json.load(handle)
                        if processor._resumable_manifest(old, name, day, selected):
                            continue
                    typer.echo(f"{name} {day} {selected}")
                    processor.run_stage(name, day, selected, tiles)


if __name__ == "__main__":
    app()
