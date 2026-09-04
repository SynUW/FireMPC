"""Build and validate the ten-day, 17-channel operational MODIS package."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import rasterio
import typer

# Run directly as `python operational/modis/package.py`: put the application root on
# sys.path so the shared modules import by bare name, as elsewhere in FireMPC.
import sys
from pathlib import Path
OPERATIONAL = Path(__file__).resolve().parents[1]
if str(OPERATIONAL) not in sys.path:
    sys.path.insert(0, str(OPERATIONAL))

import paths, vendorpaths
from ops import atomic_json, sha256
from modis.processing_registry import (
    CHANNEL_NAMES,
    CHANNELS,
    LAND_COVER_CODES,
    PACKAGE_DTYPE,
    REGISTRY_VERSION,
    ROUTES,
    SCALING_POLICY_VERSION,
    require_operational_lineage,
)

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
DATE_RE = re.compile(r"(?<!\d)(\d{4})[_-](\d{2})[_-](\d{2})(?!\d)")
DEFAULT_POLICY = vendorpaths.CONFIG / "modis_fill_policy.json"


@dataclass(frozen=True, slots=True)
class SourceChoice:
    product: str
    requested_date: date
    source_date: date
    path: Path
    age_days: int
    status: str


def target_dates(target: date | str) -> tuple[date, ...]:
    """Return D-10 through D-1, oldest first."""
    if isinstance(target, str):
        target = date.fromisoformat(target)
    return tuple(target - timedelta(days=offset) for offset in range(10, 0, -1))


def file_date(path: Path) -> date | None:
    match = DATE_RE.search(path.stem)
    if match is None:
        return None
    try:
        return date(*(int(value) for value in match.groups()))
    except ValueError:
        return None


def dated_rasters(folder: Path) -> dict[date, Path]:
    found: dict[date, Path] = {}
    if not folder.is_dir():
        return found
    for candidate in sorted((*folder.glob("*.tif"), *folder.glob("*.tiff"))):
        day = file_date(candidate)
        if day is not None:
            if day in found:
                raise RuntimeError(f"duplicate {day} rasters in {folder}")
            found[day] = candidate
    return found


def load_fill_policy(path: Path = DEFAULT_POLICY) -> dict:
    with path.open(encoding="utf-8") as handle:
        policy = json.load(handle)
    ages = policy.get("maximum_age_days", {})
    missing = set(ROUTES) - set(ages)
    if missing or any(not isinstance(value, int) or value < 0 for value in ages.values()):
        raise RuntimeError(f"invalid fill policy; missing={sorted(missing)}")
    return policy


def observation_dates(product: str, unified_root: Path) -> set[date]:
    """Dates on which this product was actually observed.

    The `scaled` tree is not the answer: `daily_filled` forward-copies each
    national raster across the days it covers, so a file named for a date only
    proves a raster exists, not that anything was observed then.  `unified` is
    the last stage before that copy, so its dates are the real anchors.

    NDVIEVI is computed per calendar date from scaled MCD09GA and has no
    unified stage of its own, so it inherits MCD09GA's observation dates.
    """
    source = "MCD09GA" if product == "NDVIEVI" else product
    return set(dated_rasters(unified_root / source))


def resolve_source(
    product: str,
    requested: date,
    available: dict[date, Path],
    maximum_age_days: int,
    observations: set[date] | None = None,
    as_of: date | None = None,
) -> SourceChoice:
    """Choose the raster a requested date resolves to.

    `as_of` restricts the choice to observations that had actually landed by
    that date, per the CMR insert history (`modis.availability`).  Without it
    the newest observation on disk wins, which is correct for an operational
    run -- what is on disk is what has arrived -- and wrong for a backfill,
    where "on disk now" silently includes granules published weeks after the
    date being rebuilt.
    """
    candidates = [day for day in available if day <= requested]
    if not candidates:
        raise RuntimeError(f"{product} {requested}: no observation on or before requested date")
    source = max(candidates)
    # The raster holding the pixels and the observation it descends from are
    # different things once daily_filled has run.  Age must be measured from
    # the observation, or a four-day composite forward-filled to every day in
    # its window reports itself as freshly observed.
    # `or` not `is not None`: an empty observation set means the unified tree
    # had nothing to say, and the available dates stand in for it, as before.
    known = observations or set(available)
    if as_of is not None:
        from modis import availability
        known = availability.known_by(product, as_of, candidates=known)
    observed = [day for day in known if day <= requested]
    if not observed:
        raise RuntimeError(
            f"{product} {requested}: no observation on or before requested date"
            + (f" that had landed by {as_of}" if as_of is not None else ""))
    origin = max(observed)
    if as_of is not None and product != "MCD12Q1":
        # Take the raster the chosen observation produced, not simply the
        # newest file on or before the requested date: under daily_filled the
        # two normally name the same pixels, but once an observation is ruled
        # out for landing too late, its forward-filled copies must be ruled
        # out with it. MCD12Q1 is the exception: its annual observation is
        # published under each requested daily date, so there is deliberately
        # no scaled raster named with the old observation date.
        origin_candidates = [day for day in available if day <= origin]
        if not origin_candidates:
            raise RuntimeError(
                f"{product} {requested}: no raster for observation {origin}")
        source = max(origin_candidates)
    age = (requested - origin).days
    if age > maximum_age_days:
        raise RuntimeError(
            f"{product} {requested}: newest source {source} is {age} days old; "
            f"maximum is {maximum_age_days}")
    # A composite is structurally older than the day it covers: forward-filling
    # a four-day anchor to the intervening days is what the training pipeline
    # does, so an age inside the cadence is by design.  Age beyond the cadence
    # means an anchor was missed and the staleness is real latency -- keep the
    # two apart rather than blessing any composite gap as expected.
    if product == "MCD12Q1":
        status = "stale_by_design"
    elif age == 0:
        status = "observed"
    elif age < ROUTES[product].cadence_days:
        status = "composite_fill"
    else:
        status = "latency_fill"
    return SourceChoice(product, requested, origin, available[source], age, status)


def code_revision() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=vendorpaths.GIT_ROOT,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty_cmd = ["git", "status", "--porcelain", *vendorpaths.operational_git_pathspec()]
        dirty = subprocess.run(
            dirty_cmd, cwd=vendorpaths.GIT_ROOT,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def _grid_errors(dataset: rasterio.DatasetReader, template: rasterio.DatasetReader) -> list[str]:
    errors = []
    if (dataset.height, dataset.width) != (template.height, template.width):
        errors.append(
            f"shape {dataset.height}x{dataset.width}, expected "
            f"{template.height}x{template.width}")
    if dataset.crs != template.crs:
        errors.append(f"CRS {dataset.crs}, expected {template.crs}")
    if not dataset.transform.almost_equals(template.transform, precision=1e-12):
        errors.append(f"transform {dataset.transform}, expected {template.transform}")
    return errors


def _read_plane(choice: SourceChoice, channel, template) -> np.ndarray:
    with rasterio.open(choice.path) as source:
        errors = _grid_errors(source, template)
        if source.count < channel.band:
            errors.append(f"has {source.count} bands, needs band {channel.band}")
        if source.dtypes[0] != PACKAGE_DTYPE:
            errors.append(f"dtype {source.dtypes[0]}, expected {PACKAGE_DTYPE}")
        if errors:
            raise RuntimeError(f"{choice.path}: {'; '.join(errors)}")
        plane = source.read(channel.band)
    if channel.categorical:
        values = {int(value) for value in np.unique(plane)}
        bad = values - LAND_COVER_CODES - {channel.nodata}
        if bad:
            raise RuntimeError(f"{choice.path}: invalid land-cover codes {sorted(bad)}")
    if np.all(plane == channel.nodata):
        raise RuntimeError(f"{choice.path}: {channel.name} is all nodata")
    return plane


def _semantic_manifest(
    target: date,
    dates: tuple[date, ...],
    choices: dict[tuple[date, str], SourceChoice],
    policy: dict,
    as_of: date | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "target_date": target.isoformat(),
        "dates": [day.isoformat() for day in dates],
        "channel_registry_version": REGISTRY_VERSION,
        "scaling_policy_version": SCALING_POLICY_VERSION,
        "fill_policy_version": policy["version"],
        "as_of": as_of.isoformat() if as_of else None,
        "channels": [
            {
                "index": channel.index,
                "name": channel.name,
                "product": channel.product,
                "band": channel.band,
                "nodata": channel.nodata,
            }
            for channel in CHANNELS
        ],
        "sources": [
            {
                "date": day.isoformat(),
                "channel_index": channel.index,
                "channel": channel.name,
                "product": channel.product,
                "source_date": choices[(day, channel.product)].source_date.isoformat(),
                "fill_age_days": choices[(day, channel.product)].age_days,
                "status": choices[(day, channel.product)].status,
                "path": str(choices[(day, channel.product)].path),
            }
            for day in dates for channel in CHANNELS
        ],
    }


def build_package(
    target: date,
    *,
    scaled_root: Path,
    package_root: Path,
    template_path: Path,
    unified_root: Path | None = None,
    policy_path: Path = DEFAULT_POLICY,
    overwrite: bool = False,
    as_of: date | None = None,
) -> Path:
    policy = load_fill_policy(policy_path)
    # The observation tree is a sibling of the scaled tree, so overriding one
    # in a test or a scratch run does not silently read the production other.
    unified_root = unified_root or scaled_root.parent / "unified"
    dates = target_dates(target)
    available = {
        product: dated_rasters(scaled_root / route.output_dir)
        for product, route in ROUTES.items()
    }
    observed = {
        product: observation_dates(product, unified_root)
        for product in ROUTES
    }
    choices = {
        (day, product): resolve_source(
            product, day, available[product], policy["maximum_age_days"][product],
            observed[product], as_of=as_of)
        for day in dates for product in ROUTES
    }
    # latest-available accepts delayed observations inside its explicit
    # discovery lookback, but makes age beyond the measured strict SLA visible.
    warning_ages = policy.get("warning_age_days", {})
    choices = {
        key: replace(choice, status="stale_warning")
        if choice.age_days > warning_ages.get(choice.product, 10**9)
        else choice
        for key, choice in choices.items()
    }
    destination = package_root / target.isoformat()
    destination.mkdir(parents=True, exist_ok=True)
    final_h5 = destination / "modis_inputs.h5"
    if final_h5.exists() and not overwrite:
        raise FileExistsError(f"package exists: {final_h5}; use --overwrite")

    manifest = _semantic_manifest(target, dates, choices, policy, as_of)
    creation = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    temp_fd, temp_name = tempfile.mkstemp(prefix=".modis_inputs.", suffix=".h5", dir=destination)
    os.close(temp_fd)
    temp_h5 = Path(temp_name)
    try:
        with rasterio.open(template_path) as template, h5py.File(temp_h5, "w") as h5:
            height, width = template.height, template.width
            data = h5.create_dataset(
                "data", shape=(10, 17, height, width), dtype="u1",
                chunks=(1, 1, min(256, height), min(256, width)),
                compression="gzip", compression_opts=1, shuffle=True,
            )
            source_dates = np.empty((10, 17), dtype=object)
            fill_ages = np.empty((10, 17), dtype=np.int16)
            for time_index, day in enumerate(dates):
                for channel in CHANNELS:
                    choice = choices[(day, channel.product)]
                    data[time_index, channel.index] = _read_plane(choice, channel, template)
                    source_dates[time_index, channel.index] = choice.source_date.isoformat()
                    fill_ages[time_index, channel.index] = choice.age_days
            strings = h5py.string_dtype("utf-8")
            h5.create_dataset("dates", data=np.asarray([d.isoformat() for d in dates], dtype=object), dtype=strings)
            h5.create_dataset("channel_names", data=np.asarray(CHANNEL_NAMES, dtype=object), dtype=strings)
            h5.create_dataset("source_dates", data=source_dates, dtype=strings)
            h5.create_dataset("fill_age_days", data=fill_ages)
            h5.attrs.update({
                "target_date": target.isoformat(), "crs": template.crs.to_string(),
                "transform": tuple(template.transform)[:6], "height": height, "width": width,
                "channel_registry_version": REGISTRY_VERSION,
                "scaling_policy_version": SCALING_POLICY_VERSION,
                "fill_policy_version": policy["version"],
                "fill_policy_mode": policy.get("mode", "strict"),
                "as_of": as_of.isoformat() if as_of else "",
                "code_revision": code_revision(),
                "creation_timestamp": creation,
                "nodata_encoding": json.dumps({str(c.index): c.nodata for c in CHANNELS}),
                "mcd12q1_source_year": max(
                    choices[(day, "MCD12Q1")].source_date.year for day in dates),
            })
            h5.flush()
        validate_h5(temp_h5, template_path)
        os.replace(temp_h5, final_h5)
    finally:
        temp_h5.unlink(missing_ok=True)

    manifest["created_utc"] = creation
    manifest["code_revision"] = code_revision()
    manifest["artifacts"] = {"modis_inputs.h5": {"sha256": sha256(final_h5), "bytes": final_h5.stat().st_size}}
    atomic_json(destination / "manifest.json", manifest)
    validation = validate_package(destination, template_path)
    atomic_json(destination / "validation.json", validation)
    return final_h5


def validate_h5(h5_path: Path, template_path: Path) -> None:
    with rasterio.open(template_path) as template, h5py.File(h5_path, "r") as h5:
        expected = (10, 17, template.height, template.width)
        if h5["data"].shape != expected or h5["data"].dtype != np.dtype("uint8"):
            raise RuntimeError(f"invalid /data contract: {h5['data'].shape} {h5['data'].dtype}")
        dates = [value.decode() if isinstance(value, bytes) else str(value) for value in h5["dates"][:]]
        channels = [value.decode() if isinstance(value, bytes) else str(value) for value in h5["channel_names"][:]]
        if len(dates) != 10 or dates != sorted(dates) or channels != list(CHANNEL_NAMES):
            raise RuntimeError("date or channel order is invalid")
        if h5["source_dates"].shape != (10, 17) or h5["fill_age_days"].shape != (10, 17):
            raise RuntimeError("source provenance arrays have invalid shapes")
        target = date.fromisoformat(str(h5.attrs["target_date"]))
        if dates != [day.isoformat() for day in target_dates(target)]:
            raise RuntimeError("dates are not D-10 through D-1")
        if tuple(h5.attrs["transform"]) != tuple(template.transform)[:6]:
            raise RuntimeError("HDF5 transform does not match GRID_TEMPLATE")
        ages = h5["fill_age_days"][:]
        if np.any(ages < 0):
            raise RuntimeError("negative fill age in package")
        for channel in CHANNELS:
            for time_index in range(10):
                plane = h5["data"][time_index, channel.index]
                if np.all(plane == channel.nodata):
                    raise RuntimeError(
                        f"all-nodata plane at time={time_index}, channel={channel.name}")
                if channel.categorical:
                    values = {int(value) for value in np.unique(plane)}
                    bad = values - LAND_COVER_CODES - {channel.nodata}
                    if bad:
                        raise RuntimeError(
                            f"invalid land-cover codes in package: {sorted(bad)}")


def validate_package(package_dir: Path, template_path: Path) -> dict:
    h5_path, manifest_path = package_dir / "modis_inputs.h5", package_dir / "manifest.json"
    validate_h5(h5_path, template_path)
    checks = {"hdf5_contract": "pass"}
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        expected = manifest.get("artifacts", {}).get("modis_inputs.h5", {}).get("sha256")
        actual = sha256(h5_path)
        if expected != actual:
            raise RuntimeError(f"package checksum mismatch: manifest={expected}, actual={actual}")
        checks["manifest_checksum"] = "pass"
    return {"valid": True, "checked_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "checks": checks}


# Everything a rebuild is allowed to change.  A package is a statement about
# which pixels the model sees, so only the record of when it was made may vary.
VOLATILE_ATTRS = frozenset({"creation_timestamp"})
PROVENANCE_DATASETS = ("dates", "channel_names", "source_dates", "fill_age_days")


def determinism_report(
    target: date,
    *,
    scaled_root: Path,
    template_path: Path,
    unified_root: Path | None = None,
    policy_path: Path = DEFAULT_POLICY,
) -> dict:
    """Validation gate 5: build the package twice and diff everything but time."""
    work = Path(paths.WORK)
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"modis-determinism-{target}-", dir=work) as scratch:
        builds = [
            build_package(
                target, scaled_root=scaled_root, package_root=Path(scratch) / name,
                template_path=template_path, unified_root=unified_root,
                policy_path=policy_path,
            )
            for name in ("first", "second")
        ]
        differences: list[str] = []
        with h5py.File(builds[0], "r") as first, h5py.File(builds[1], "r") as second:
            for time_index in range(first["data"].shape[0]):
                if not np.array_equal(first["data"][time_index], second["data"][time_index]):
                    differences.append(f"data plane t={time_index} differs")
            for name in PROVENANCE_DATASETS:
                if not np.array_equal(first[name][:], second[name][:]):
                    differences.append(f"{name} differs")
            for key in sorted(set(first.attrs) | set(second.attrs)):
                if key in VOLATILE_ATTRS:
                    continue
                if np.any(first.attrs.get(key) != second.attrs.get(key)):
                    differences.append(
                        f"attribute {key!r} differs: "
                        f"{first.attrs.get(key)!r} vs {second.attrs.get(key)!r}")
        identical_bytes = sha256(builds[0]) == sha256(builds[1])
    return {
        "deterministic": not differences,
        "differences": differences,
        # gzip stores no timestamp, so the files usually also match byte for
        # byte; that is a stronger result than required and not a failure when
        # it does not hold, since creation_timestamp legitimately differs.
        "identical_bytes": identical_bytes,
    }


@app.command()
def main(
    target: Annotated[str, typer.Option("--target", help="Prediction date D (YYYY-MM-DD).")],
    scaled_root: Annotated[Path, typer.Option("--scaled-root")] = Path(paths.MODIS_STAGE_DIRS["scaled"]),
    package_root: Annotated[Path, typer.Option("--package-root")] = Path(paths.MODIS_PACKAGES),
    template: Annotated[Path, typer.Option("--grid-template")] = Path(paths.GRID_TEMPLATE),
    unified_root: Annotated[Path | None, typer.Option(
        "--unified-root",
        help="Stage holding the true per-observation rasters, used for fill "
             "provenance (the scaled tree cannot supply it: daily_filled has "
             "already forward-copied composites across their window).",
    )] = None,
    fill_policy: Annotated[Path, typer.Option("--fill-policy")] = DEFAULT_POLICY,
    as_of: Annotated[str | None, typer.Option(
        "--as-of",
        help="Resolve each date only against observations that had landed by "
             "this date, per the CMR insert history. Use it for backfills; an "
             "operational run wants the default, which is whatever is on disk.",
    )] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    verify_determinism: Annotated[bool, typer.Option(
        "--verify-determinism",
        help="Also build the package twice into scratch and fail unless the two "
             "agree on every plane, every provenance entry and every attribute "
             "except creation_timestamp.",
    )] = False,
) -> None:
    require_operational_lineage()
    day = date.fromisoformat(target)
    output = build_package(
        day, scaled_root=scaled_root, package_root=package_root,
        template_path=template, unified_root=unified_root,
        policy_path=fill_policy, overwrite=overwrite,
        as_of=date.fromisoformat(as_of) if as_of else None,
    )
    if verify_determinism:
        report = determinism_report(
            day, scaled_root=scaled_root, template_path=template,
            unified_root=unified_root, policy_path=fill_policy,
        )
        atomic_json(package_root / target / "determinism.json", report)
        if not report["deterministic"]:
            raise RuntimeError(
                "package is not reproducible: " + "; ".join(report["differences"]))
        typer.echo(f"determinism: pass (identical_bytes={report['identical_bytes']})")
    typer.echo(output)


if __name__ == "__main__":
    app()
