"""Month-labelled, append-only store of per-date CanFireCast inputs.

A daily package is a ten-day window, so consecutive targets restate nine of
their ten dates.  On top of that every package carries the twelve static planes
ten times over and six channels that are zero by contract.  Of the 550 planes
in `canfirecast_inputs.h5`, thirty per date are load-bearing.

This module keeps one file per calendar month holding each date once:

    packages/canfirecast/2026-07.h5

`read_window` reassembles the exact `(10, 55, H, W)` uint8 tensor the package
contract specifies, so nothing downstream of it can tell the difference, and
`publish_window` writes that tensor out as an ordinary package wherever the
caller asks.  That pair is the store's read API: this repository fills the
store, and inference code elsewhere publishes the window it wants out of it.

What is stored, and what is not
-------------------------------
Only quantities that are a function of the date can be stored per date.  The
stage outputs settle the question directly -- comparing the 2026-08-24 and
2026-08-25 runs on their nine shared dates:

    encoded_raw55.h5     differing channels = []
    base_corrected55.h5  differing channels = []
    corrected55.h5       differing channels = [19, 20, 21, 22, 23, 24]

Everything is date-keyed right up to `cffdrs-derive`.  That stage seeds the
FFMC/DMC/DC recursion from the certified D-11 ledger state, which is
ERA5-Land-grade, and then walks the IFS path forward; a window built for D and
one built for D+1 therefore disagree on their shared dates by construction.
So the FWI codes and VPD are *derived at read*, per target, exactly as the
pipeline derives them -- not stored.

The stored weather is `base_corrected55`, in float32.  Rounding it to uint8
first and deriving from that does not reproduce the package: VPD moves by a
mean of 4.2 and ISI by up to 166, because the recursion amplifies the rounding.
The frozen code is explicit that the sole uint8 boundary is *after* recursive
derivation, and the store has to respect that to stay byte-exact.

MODIS is stored as uint8 because it is already uint8 by the time it is
packaged, but it is not a function of the date in the same way: `resolve_source`
picks the newest observation on or before the date from whatever has landed, so
a late granule changes the answer.  The store therefore freezes a date at first
write -- the date entering a window for the first time, target d+1 for date d.
A window read for any target D >= d+1 then sees MODIS knowledge as of d+1,
never later, so a hindcast can never be shown a granule that had not yet
arrived.  The cost is that a plane can be staler than the operational run of D
saw, since that run had until D for late granules to land; `modis_fill_age_days`
records exactly how much, per channel.  Holding both would mean a value per
(date, as-of) pair, which is the redundancy this store exists to remove, and
erring the other way would quietly flatter every hindcast.
"""

from __future__ import annotations

import calendar
from datetime import date, datetime, timezone
from pathlib import Path

import h5py
import numpy as np

# Run directly as `python operational/canfirecast/store.py`: put the application root on
# sys.path so the shared modules import by bare name, as elsewhere in FireMPC.
import sys
OPERATIONAL = Path(__file__).resolve().parents[1]
if str(OPERATIONAL) not in sys.path:
    sys.path.insert(0, str(OPERATIONAL))

from canfirecast import input_registry as registry
from canfirecast.state_ledger import DEFAULT_LEDGER, state_path
import paths

SCHEMA_VERSION = 1
WINDOW_DAYS = 10

# The thirteen direct weather channels, held pre-rounding.  19-24 (CFFDRS) and
# 25 (VPD) are deliberately absent: they come out of the recursion at read.
WEATHER_CHANNELS = registry.DIRECT_WEATHER_CHANNELS
MODIS_CHANNELS = registry.MODIS_CHANNELS
DERIVED_AT_READ = registry.DERIVED_WEATHER_CHANNELS  # 19..25


def month_path(root: Path, day: date) -> Path:
    return Path(root) / f"{day.year:04d}-{day.month:02d}.h5"


def day_index(day: date) -> int:
    """Position of a date within its own month's datasets."""
    return day.day - 1


def month_length(day: date) -> int:
    return calendar.monthrange(day.year, day.month)[1]


def is_written(root: Path, day: date) -> bool:
    """Has this date already been frozen into its month file?"""
    path = month_path(root, day)
    if not path.is_file():
        return False
    with h5py.File(path, "r") as h5:
        return bool(h5["complete"][day_index(day)])


def _strings(values) -> np.ndarray:
    """Decode an h5py string array without collapsing its shape.

    `canfirecast.package._strings` iterates row-wise and `str()`s whatever it
    gets, which turns a (10, 17) provenance array into ten numpy reprs.  Map
    over the elements instead so a 2-D array stays 2-D.
    """
    array = np.asarray(values, dtype=object)
    if not array.size:
        return array
    decode = np.vectorize(
        lambda value: value.decode() if isinstance(value, bytes) else str(value),
        otypes=[object],
    )
    return decode(array)


def create_month(
    path: Path, *, day: date, height: int, width: int,
    crs: str, transform, weather_source: str = "",
) -> None:
    """Lay out an empty month.  Every date is preallocated; `complete` gates reads."""
    days = month_length(day)
    strings = h5py.string_dtype("utf-8")
    tiles = (min(256, height), min(256, width))
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset(
            "weather", shape=(days, len(WEATHER_CHANNELS), height, width), dtype="f4",
            chunks=(1, 1, *tiles), compression="gzip", compression_opts=1, shuffle=True,
        )
        h5.create_dataset(
            "modis", shape=(days, len(MODIS_CHANNELS), height, width), dtype="u1",
            chunks=(1, 1, *tiles), compression="gzip", compression_opts=1, shuffle=True,
        )
        h5.create_dataset(
            "spatial_validity", shape=(days, height, width), dtype="bool",
            chunks=(1, *tiles), compression="gzip",
        )
        h5.create_dataset("weather_channels", data=np.asarray(WEATHER_CHANNELS, dtype="i2"))
        h5.create_dataset("modis_channels", data=np.asarray(MODIS_CHANNELS, dtype="i2"))
        h5.create_dataset("complete", data=np.zeros(days, dtype=bool))
        h5.create_dataset(
            "dates",
            data=np.asarray([date(day.year, day.month, n + 1).isoformat() for n in range(days)], dtype=object),
            dtype=strings,
        )
        h5.create_dataset("modis_source_dates", shape=(days, len(MODIS_CHANNELS)), dtype=strings)
        h5.create_dataset(
            "modis_fill_age_days", data=np.full((days, len(MODIS_CHANNELS)), -1, dtype="i2"))
        for name in ("written_utc", "code_revision", "written_for_target"):
            h5.create_dataset(name, shape=(days,), dtype=strings)
        h5.attrs.update({
            "schema_version": SCHEMA_VERSION,
            "channel_registry_version": registry.REGISTRY_VERSION,
            "year": day.year, "month": day.month,
            "crs": crs, "transform": tuple(transform)[:6],
            "height": height, "width": width,
            "weather_stage": "base_corrected55",
            "weather_dtype": "float32",
            "weather_source": weather_source,
            "derived_at_read": tuple(DERIVED_AT_READ),
            "stored_channel_note":
                "statics live once in the release asset; COMPACT49_OMITTED_NATIVE "
                "are zero by contract; 19-25 are derived per target from the D-11 "
                "ledger state because the CFFDRS recursion is not date-keyed",
        })


def write_day(
    root: Path, day: date, *,
    weather: np.ndarray, modis: np.ndarray, validity: np.ndarray,
    modis_source_dates=None, modis_fill_age_days=None,
    code_revision: str = "", written_for_target: date | None = None,
    crs: str = "", transform=None, weather_source: str = "",
    force: bool = False,
) -> Path:
    """Freeze one date.

    `weather` is a native 55-channel float32 plane stack from
    `base_corrected55.h5`; only the thirteen direct channels are kept.
    `modis` is the 17-channel uint8 stack from `modis_inputs.h5`.
    """
    if weather.shape[0] != len(registry.CHANNEL_NAMES):
        raise RuntimeError(f"weather must be (55, H, W), got {weather.shape}")
    if modis.shape[0] != len(MODIS_CHANNELS):
        raise RuntimeError(f"modis must be ({len(MODIS_CHANNELS)}, H, W), got {modis.shape}")
    height, width = weather.shape[1:]
    if modis.shape[1:] != (height, width):
        raise RuntimeError(f"modis grid {modis.shape[1:]} != weather grid {(height, width)}")
    path = month_path(root, day)
    if not path.exists():
        if not crs or transform is None:
            raise RuntimeError(f"{path} does not exist and no grid was supplied to create it")
        create_month(path, day=day, height=height, width=width,
                     crs=crs, transform=transform, weather_source=weather_source)
    index = day_index(day)
    with h5py.File(path, "r+") as h5:
        if (int(h5.attrs["height"]), int(h5.attrs["width"])) != (height, width):
            raise RuntimeError(
                f"{path}: grid is {h5.attrs['height']}x{h5.attrs['width']}, "
                f"planes are {height}x{width}")
        if bool(h5["complete"][index]) and not force:
            raise RuntimeError(
                f"{day} is already written. Rewriting it would backdate MODIS "
                "granules that had not arrived when it was frozen, and would "
                "silently change every window that already covers it. "
                "Pass force=True only to repair a bad write.")
        h5["weather"][index] = weather[list(WEATHER_CHANNELS)]
        h5["modis"][index] = modis
        h5["spatial_validity"][index] = validity
        if modis_source_dates is not None:
            h5["modis_source_dates"][index] = _strings(modis_source_dates)
        if modis_fill_age_days is not None:
            h5["modis_fill_age_days"][index] = np.asarray(modis_fill_age_days, dtype="i2")
        h5["written_utc"][index] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        h5["code_revision"][index] = code_revision
        h5["written_for_target"][index] = written_for_target.isoformat() if written_for_target else ""
        h5.flush()
        # Last, so a crash mid-write leaves a date readers skip rather than a
        # half-written one they trust.  os.replace is not available to an append.
        h5["complete"][index] = True
    return path


def read_window(
    root: Path, target: date, *, static_h5: Path, state_h5: Path,
    correction_artifact: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Reassemble the (10, 55, H, W) package tensor for a target date.

    Opens at most two month files -- a ten-day window crosses a month boundary
    once at most -- then runs the same recursion `cffdrs-derive` runs, seeded
    from the certified D-11 state, to produce channels 19-25.
    """
    from canfirecast.model.cffdrs_derive import (
        derive_corrected_weather_channels, load_correction)

    dates = registry.target_window_dates(target)
    with h5py.File(static_h5, "r") as static:
        statics = static["data"][...]
    if statics.shape[0] != len(registry.STATIC_CHANNELS):
        raise RuntimeError(
            f"static asset must hold {len(registry.STATIC_CHANNELS)} planes, got {statics.shape}")

    base = validity = None
    modis = None
    source_dates = np.empty((WINDOW_DAYS, len(MODIS_CHANNELS)), dtype=object)
    fill_ages = np.full((WINDOW_DAYS, len(MODIS_CHANNELS)), -1, dtype="i2")
    attributes: dict = {}
    handles: dict[Path, h5py.File] = {}
    try:
        for time_index, day in enumerate(dates):
            path = month_path(root, day)
            if path not in handles:
                if not path.is_file():
                    raise RuntimeError(f"{day}: no month store at {path}")
                handles[path] = h5py.File(path, "r")
            h5 = handles[path]
            index = day_index(day)
            if not bool(h5["complete"][index]):
                raise RuntimeError(f"{day} is not marked complete in {path}")
            if base is None:
                height, width = int(h5.attrs["height"]), int(h5.attrs["width"])
                if statics.shape[1:] != (height, width):
                    raise RuntimeError(
                        f"static asset is {statics.shape[1:]}, store grid is {(height, width)}")
                base = np.zeros((WINDOW_DAYS, len(registry.CHANNEL_NAMES), height, width), dtype="f4")
                modis = np.zeros((WINDOW_DAYS, len(MODIS_CHANNELS), height, width), dtype="u1")
                validity = np.zeros((WINDOW_DAYS, height, width), dtype=bool)
                attributes = {
                    key: h5.attrs[key]
                    for key in ("crs", "transform", "height", "width",
                                "channel_registry_version", "weather_source")
                    if key in h5.attrs
                }
            # Scatter by the channel list the file declares, not a module
            # constant, so a store driven by ERA5 -- which supplies the five
            # fields IFS lacks -- reads back through this same path.
            base[time_index, list(h5["weather_channels"][...])] = h5["weather"][index]
            modis[time_index] = h5["modis"][index]
            validity[time_index] = h5["spatial_validity"][index]
            source_dates[time_index] = _strings(h5["modis_source_dates"][index])
            fill_ages[time_index] = h5["modis_fill_age_days"][index]
    finally:
        for handle in handles.values():
            handle.close()

    with h5py.File(state_h5, "r") as state:
        state_day = str(state.attrs.get("date", ""))
        if state_day != registry.state_date(target).isoformat():
            raise RuntimeError(f"state date {state_day!r} is not target D-11")
        if not state.attrs.get("certified", False):
            raise RuntimeError("D-11 state is not marked certified")
        initial = state["state"][()]

    # A national window starts at row zero; center_row is only an adapter
    # representation of row identity for patch inputs.
    data = derive_corrected_weather_channels(
        base, target=target, center_row=base.shape[2] // 2,
        initial_cffdrs_encoded=initial,
        correction=load_correction(correction_artifact),
    )
    data[:, list(registry.STATIC_CHANNELS)] = statics
    data[:, list(MODIS_CHANNELS)] = modis
    data[:, registry.FIRE_DETECTION_CHANNEL] = 0
    data[:, list(registry.ERA5_ONLY_MISSING_CHANNELS)] = 0

    provenance = {
        "target_date": target.isoformat(),
        "dates": [day.isoformat() for day in dates],
        "modis_source_dates": source_dates,
        "modis_fill_age_days": fill_ages,
        "d11_state_date": state_day,
        "d11_state_sha256": registry.sha256(state_h5),
        **attributes,
    }
    return data, validity, provenance


def validate_window(data: np.ndarray, validity: np.ndarray, provenance: dict, target: date) -> None:
    """Validate an in-memory window reconstructed from the monthly store."""
    if data.dtype != np.dtype("uint8"):
        raise RuntimeError("monthly-store window data must be uint8")
    if data.ndim != 4 or data.shape[:2] != (WINDOW_DAYS, len(registry.CHANNEL_NAMES)):
        raise RuntimeError(
            f"monthly-store window must have shape (10,55,H,W), got {data.shape}")
    height, width = data.shape[2:]
    if validity.dtype != np.dtype("bool") or validity.shape != (WINDOW_DAYS, height, width):
        raise RuntimeError("monthly-store spatial_validity must be bool (10,H,W)")
    if tuple(provenance.get("dates", ())) != tuple(
        day.isoformat() for day in registry.target_window_dates(target)
    ):
        raise RuntimeError("monthly-store dates must be exactly D-10..D-1")
    for key in ("crs", "transform", "height", "width", "d11_state_date", "d11_state_sha256"):
        if key not in provenance:
            raise RuntimeError(f"monthly-store window is missing provenance field {key}")
    if int(provenance["height"]) != height or int(provenance["width"]) != width:
        raise RuntimeError("monthly-store height/width do not match reconstructed data")
    if np.any(data[:, registry.FIRE_DETECTION_CHANNEL] != 0):
        raise RuntimeError("Fire Detection channel must be zero")
    if np.any(data[:, list(registry.ERA5_ONLY_MISSING_CHANNELS)] != 0):
        raise RuntimeError("ERA5-only missing channels must be zero")


def ingest_window(
    root: Path, *, target: date, base_h5: Path, modis_h5: Path,
    force: bool = False,
) -> list[date]:
    """Fold one run's ten dates into their month files, skipping frozen dates.

    Reads `base_corrected55.h5` rather than the finished package: the package
    holds post-recursion uint8, which cannot reproduce the CFFDRS derivation.
    """
    written: list[date] = []
    with h5py.File(base_h5, "r") as base, h5py.File(modis_h5, "r") as modis:
        dates = [date.fromisoformat(value) for value in _strings(base["dates"][()])]
        if tuple(dates) != registry.target_window_dates(target):
            raise RuntimeError(f"{base_h5} does not cover D-10..D-1 for {target}")
        if base["data"].dtype != np.dtype("float32"):
            raise RuntimeError(
                f"{base_h5} is {base['data'].dtype}; the store needs pre-rounding "
                "float32 weather to reproduce the recursion")
        revision = str(base.attrs.get("code_revision", ""))
        crs, transform = str(base.attrs["crs"]), base.attrs["transform"]
        weather_source = str(base.attrs.get("source_weather_cycle", ""))
        source_dates = _strings(modis["source_dates"][()]) if "source_dates" in modis else None
        fill_ages = modis["fill_age_days"][()] if "fill_age_days" in modis else None
        validity = base["spatial_validity"] if "spatial_validity" in base else None
        for time_index, day in enumerate(dates):
            if is_written(root, day) and not force:
                continue
            planes = base["data"][time_index]
            write_day(
                root, day,
                weather=planes,
                modis=modis["data"][time_index],
                validity=(validity[time_index] if validity is not None
                          else np.ones(planes.shape[1:], dtype=bool)),
                modis_source_dates=source_dates[time_index] if source_dates is not None else None,
                modis_fill_age_days=fill_ages[time_index] if fill_ages is not None else None,
                code_revision=revision, written_for_target=target,
                crs=crs, transform=transform, weather_source=weather_source,
                force=force,
            )
            written.append(day)
    return written


def missing_inputs(root: Path, target: date, *, state_root: Path) -> list[str]:
    """What a target still lacks, or an empty list if its window can be read.

    A sweep across a month is worth pre-checking: reconstructing a window is
    minutes of work, and finding out on day nineteen that one date was never
    ingested wastes all of it.
    """
    missing = [str(day) for day in registry.target_window_dates(target)
               if not is_written(root, day)]
    state_day = registry.state_date(target)
    if not state_path(state_root, state_day).is_file():
        missing.append(f"D-11 state {state_day}")
    return missing


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
import typer  # noqa: E402  (kept below the library surface, as elsewhere here)
from typing import Annotated  # noqa: E402

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
DEFAULT_ROOT = Path(paths.CANFIRECAST_PACKAGES)


def _run_paths(target: date) -> tuple[Path, Path]:
    work = Path(paths.PROCESSED_MODEL_WEATHER) / target.isoformat()
    return work / "base_corrected55.h5", Path(paths.MODIS_PACKAGES) / target.isoformat() / "modis_inputs.h5"


@app.command("ingest")
def ingest_command(
    target: Annotated[str, typer.Option("--target", help="Prediction date D whose run to fold in.")],
    root: Annotated[Path, typer.Option("--root")] = DEFAULT_ROOT,
    base: Annotated[Path | None, typer.Option("--base", help="base_corrected55.h5; defaults to the run's.")] = None,
    modis: Annotated[Path | None, typer.Option("--modis", help="modis_inputs.h5; defaults to the run's.")] = None,
    force: Annotated[bool, typer.Option("--force", help="Rewrite dates that are already frozen.")] = False,
) -> None:
    """Fold one run's ten dates into their month files."""
    day = date.fromisoformat(target)
    default_base, default_modis = _run_paths(day)
    written = ingest_window(
        root, target=day, base_h5=base or default_base,
        modis_h5=modis or default_modis, force=force)
    for written_day in written:
        typer.echo(f"wrote {written_day}")
    typer.echo(f"{len(written)} new date(s) into {root}")


@app.command("read")
def read_command(
    target: Annotated[str, typer.Option("--target")],
    static: Annotated[Path, typer.Option("--static")],
    root: Annotated[Path, typer.Option("--root")] = DEFAULT_ROOT,
    state_root: Annotated[Path, typer.Option("--state-root")] = DEFAULT_LEDGER,
    correction_artifact: Annotated[Path | None, typer.Option("--correction-artifact")] = None,
    output: Annotated[Path | None, typer.Option("--output", help="Write the window out as a package.")] = None,
    against: Annotated[Path | None, typer.Option("--against", help="Diff against a daily package.")] = None,
) -> None:
    """Read a window back, optionally writing it out or diffing it against a package."""
    day = date.fromisoformat(target)
    data, validity, provenance = read_window(
        root, day, static_h5=static,
        state_h5=state_path(state_root, registry.state_date(day)),
        correction_artifact=correction_artifact,
    )
    typer.echo(f"window {data.shape} {data.dtype}")
    if against is not None:
        with h5py.File(against, "r") as h5:
            differences = [
                f"t={t} channel={c} ({registry.CHANNEL_NAMES[c]})"
                for t in range(WINDOW_DAYS) for c in range(data.shape[1])
                if not np.array_equal(data[t, c], h5["data"][t, c])
            ]
            if not np.array_equal(validity, h5["spatial_validity"][...]):
                differences.append("spatial_validity")
        for line in differences[:20]:
            typer.echo(f"  differs: {line}")
        if differences:
            raise RuntimeError(f"{len(differences)} difference(s) against {against}")
        typer.echo(f"identical to {against}")
    if output is not None:
        write_window_package(output, data, validity, provenance)
        typer.echo(str(output))


# Daily packages carry the CRS as WKT; the archive (and base_corrected55.h5
# upstream of it) stores the short authority code. visualization_gen.py parses a
# package's string CRS with rasterio's CRS.from_wkt and nothing else, so a bare
# "EPSG:4326" raises there and the GeoTIFFs land unprojected.
WGS84_WKT = (
    'GEOGCS["WGS 84",DATUM["World Geodetic System 1984",'
    'SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],'
    'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
    'AXIS["Latitude",NORTH],AXIS["Longitude",EAST]]'
)


def _package_crs(value) -> str:
    """Expand a short authority code to the WKT the package format expects."""
    text = value.decode() if isinstance(value, bytes) else str(value)
    if text.strip().upper() in {"EPSG:4326", "4326"}:
        return WGS84_WKT
    return text


def write_window_package(output: Path, data, validity, provenance: dict) -> Path:
    """Materialize a read window in the package layout, by atomic replace."""
    import os
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    strings = h5py.string_dtype("utf-8")
    with h5py.File(tmp, "w") as h5:
        h5.create_dataset("data", data=data, chunks=(1, 1, min(256, data.shape[2]), min(256, data.shape[3])),
                          compression="gzip", compression_opts=1, shuffle=True)
        h5.create_dataset("spatial_validity", data=validity, compression="gzip")
        h5.create_dataset("dates", data=np.asarray(provenance["dates"], dtype=object), dtype=strings)
        h5.create_dataset("channel_names", data=np.asarray(registry.CHANNEL_NAMES, dtype=object), dtype=strings)
        h5.create_dataset("modis_source_dates", data=provenance["modis_source_dates"], dtype=strings)
        h5.create_dataset("modis_fill_age_days", data=provenance["modis_fill_age_days"])
        h5.attrs.update({
            "schema_version": 1,
            "target_date": provenance["target_date"],
            "channel_registry_version": registry.REGISTRY_VERSION,
            "dtype": registry.PACKAGE_DTYPE,
            "encoded_nodata": "none",
            "source": "canfirecast.store.read_window",
            "creation_timestamp": datetime.now(timezone.utc).isoformat(),
            **{k: provenance[k] for k in ("transform", "height", "width",
                                          "d11_state_date", "d11_state_sha256")
               if k in provenance},
            **({"crs": _package_crs(provenance["crs"])} if "crs" in provenance else {}),
        })
    os.replace(tmp, output)
    return output


def publish_window(
    root: Path, target: date, *, static_h5: Path, state_h5: Path,
    correction_artifact: Path | None = None, output: Path,
) -> Path:
    """Materialize one target's model input out of the store.

    The store is the archive; a model reads one ten-day window.  This reads
    that window back, validates it, and lands it at `output` by atomic
    replace, so a reader sees either the previous file or the new one and
    never a half-written package.  Channels 19-25 are re-derived here from
    `target`'s own certified D-11 state, which is why a window is published
    per target rather than sliced out of a dense multi-date file.
    """
    data, validity, provenance = read_window(
        root, target, static_h5=static_h5, state_h5=state_h5,
        correction_artifact=correction_artifact,
    )
    validate_window(data, validity, provenance, target)
    return write_window_package(Path(output), data, validity, provenance)


@app.command("publish")
def publish_command(
    target: Annotated[str, typer.Option("--target", help="Prediction date D to publish.")],
    static: Annotated[Path, typer.Option("--static")],
    output: Annotated[Path, typer.Option("--output", help="Package path; replaced atomically.")],
    root: Annotated[Path, typer.Option("--root")] = DEFAULT_ROOT,
    state_root: Annotated[Path, typer.Option("--state-root")] = DEFAULT_LEDGER,
    correction_artifact: Annotated[Path | None, typer.Option("--correction-artifact")] = None,
) -> None:
    """Write one target's ten-day window out as a model-input package."""
    day = date.fromisoformat(target)
    typer.echo(str(publish_window(
        root, day, static_h5=static,
        state_h5=state_path(state_root, registry.state_date(day)),
        correction_artifact=correction_artifact, output=output,
    )))


@app.command("status")
def status_command(
    root: Annotated[Path, typer.Option("--root")] = DEFAULT_ROOT,
) -> None:
    """List each month file and which of its dates are written."""
    for path in sorted(Path(root).glob("????-??.h5")):
        with h5py.File(path, "r") as h5:
            complete = h5["complete"][...]
            days = [str(value) for value in _strings(h5["dates"][()])[complete]]
        typer.echo(f"{path.name}  {complete.sum():2d}/{len(complete)} dates  "
                   f"{path.stat().st_size / 1e6:8.1f} MB")
        if days:
            typer.echo(f"    {days[0]} .. {days[-1]}")


if __name__ == "__main__":
    app()
