"""Operational preprocessing for VPD and the six FWI components.

Both products start from the daily weather tiles `weather_download.py` archives
and end as national uint8 rasters on the same grid, through the same FireMPC
stage scripts the MODIS routes use:

    VPD   derived (per tile) -> merged -> unified -> scaled
    FWI   derived (national) -> unified -> scaled

FWI has no `merged` stage because its derivation mosaics the four tiles itself,
in the same pass that advances the moisture-code recursion.

    python operational/cli.py weather preprocess --product VPD --start 2026-08-01 --end 2026-08-16
    python operational/cli.py weather preprocess --product FWI --start 2023-01-01 --end 2026-08-16
    python operational/cli.py weather preprocess --product FWI --stage scaled --start 2026-08-16 --end 2026-08-16
    python operational/cli.py weather preprocess --product FWI --state-only --start 2023-01-01 --end 2024-11-12

The input source is chosen per date by `weather_registry.source_for`: ERA5-Land
before the first complete IFS cycle, IFS from then on.  Which one was used is
recorded in every manifest, so the changeover is auditable after the fact rather
than inferred from file dates.

FWI is recursive, so its `derived` stage refuses to advance across a date gap:
skipping a day silently changes every subsequent drought code.  `--bridge-missing`
carries the previous day's state across an absent date and records that it did.

`--state-only` advances the moisture codes without writing the six-band raster.
The historical spin-up leg needs the state and nothing else, and the rasters it
would otherwise write are ~47 MB a day of output nobody reads.

The 18-band ERA5 model slot is deliberately not built here.  IFS carries 13 of
its 18 channels and the policy for the other five is unresolved; see
`weather_registry.MISSING_ERA5_BANDS`.  Neither VPD nor FWI needs any of them.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

import numpy as np
import rasterio
import typer

# Run directly as `python operational/weather/preprocess.py`: put the application root on
# sys.path so the shared modules import by bare name, as elsewhere in FireMPC.
import sys
from pathlib import Path
OPERATIONAL = Path(__file__).resolve().parents[1]
if str(OPERATIONAL) not in sys.path:
    sys.path.insert(0, str(OPERATIONAL))

import paths, vendorpaths
from weather import derive as derive
from weather import registry as registry

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

FIREMPC = vendorpaths.FIREMPC_NEWSET
DEFAULT_FIREMPC_PYTHON = Path(os.environ.get("FIREMPC_PYTHON", sys.executable))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _dates(start: date, end: date):
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


class Processor:
    def __init__(
        self,
        *,
        drivers: Path,
        data_root: Path,
        report_root: Path,
        state_root: Path,
        template: Path,
        firempc_python: Path,
        overwrite: bool = False,
        bridge_missing: bool = False,
        state_only: bool = False,
        source: str = "auto",
    ) -> None:
        self.drivers = drivers
        self.data_root = data_root
        self.report_root = report_root
        self.state_root = state_root
        self.template = template
        self.python = firempc_python
        self.overwrite = overwrite
        self.bridge_missing = bridge_missing
        self.state_only = state_only
        self.source_choice = source.lower()
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # -- layout ------------------------------------------------------------
    def stage_dir(self, stage: str, product: str) -> Path:
        return self.data_root / stage / registry.ROUTES[product].output_dir

    def source_for(self, day: date) -> registry.WeatherSource:
        """Which collection owns this date.

        `auto` follows the registry changeover rather than what happens to sit
        on disk, so a backfill and a daily run produce the same file for the
        same day.
        """
        if self.source_choice == "auto":
            return registry.source_for(day)
        return registry.source_by_key(self.source_choice)

    def _tiles_of(self, source: registry.WeatherSource, day: date) -> dict[int, Path]:
        found = {}
        for tile in registry.TILES:
            path = self.drivers / source.prefix / f"{source.stem(day, tile)}.tif"
            if path.is_file():
                found[tile] = path
        return found

    def input_tiles(self, day: date, product: str) -> tuple[registry.WeatherSource, dict[int, Path]]:
        """The four source tiles for one date, or a failure that names the gap.

        A date whose own source is absent but whose *other* source is on disk is
        an error, not a silent substitution: mixing sources inside one run would
        put a discontinuity into the FWI recursion without recording it.
        """
        source = self.source_for(day)
        tiles = self._tiles_of(source, day)
        if len(tiles) == len(registry.TILES):
            return source, tiles
        missing = sorted(set(registry.TILES) - set(tiles))
        other = [
            candidate for candidate in registry.SOURCES.values()
            if candidate.key != source.key
            and len(self._tiles_of(candidate, day)) == len(registry.TILES)
        ]
        hint = (f"; {other[0].key} tiles for this date are present -- "
                f"pass --source {other[0].key} to use them") if other else ""
        raise RuntimeError(
            f"{product} {day}: {source.key} tiles {missing} are absent{hint}")

    def _manifest_path(self, product: str, day: date, stage: str) -> Path:
        return self.report_root / product / day.isoformat() / f"{stage}.json"

    # -- process plumbing --------------------------------------------------
    @contextlib.contextmanager
    def _work(self, product: str, day: date, stage: str):
        base = self.data_root / ".work"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f"weather-{product}-{day}-{stage}-", dir=base
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
            if not target.exists() or self.overwrite:
                temporary = destination / f".{source.name}.{os.getpid()}.tmp"
                shutil.copy2(source, temporary)
                os.replace(temporary, target)
            published.append(target)
        return published

    def _run(self, command: list[str], cwd: Path, log: Path) -> None:
        log.parent.mkdir(parents=True, exist_ok=True)
        # The FireMPC scripts run under their own conda interpreter, and the
        # ambient shell exports PROJ_DATA=/usr/share/proj -- an older PROJ
        # database that GDAL rejects outright.  Point the child at its own
        # prefix, exactly as modis_preprocess does.
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
        environment["PATH"] = os.pathsep.join((str(prefix / "bin"), environment.get("PATH", "")))
        with log.open("w", encoding="utf-8") as handle:
            handle.write(" ".join(command) + "\n\n")
            handle.flush()
            result = subprocess.run(
                command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT,
                env=environment, check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"{Path(command[1]).name} exited {result.returncode}; see {log}\n"
                + log.read_text(encoding="utf-8", errors="replace")[-4000:]
            )

    def _record(self, product, day, stage, started, status, inputs, outputs,
                *, command=None, error=None, extra=None) -> None:
        manifest = {
            "schema_version": 1,
            "registry_version": registry.REGISTRY_VERSION,
            "run_id": self.run_id,
            "product": product,
            "date": day.isoformat(),
            "stage": stage,
            "started_utc": started,
            "ended_utc": _utc_now(),
            "status": status,
            "inputs": [{"path": str(p), "sha256": sha256(p)} for p in inputs if p.is_file()],
            "outputs": [{"path": str(p), "sha256": sha256(p)} for p in outputs if p.is_file()],
            "command": command,
            "error": error,
            "band_ranges": self._range_report(outputs) if status == "complete" else None,
        }
        manifest.update(extra or {})
        _atomic_json(self._manifest_path(product, day, stage), manifest)

    @staticmethod
    def _range_report(outputs: list[Path]) -> list[dict]:
        """Per-band valid-pixel range, the same shape of evidence the MODIS
        manifests carry.  NaN is the only invalid marker these products use."""
        report = []
        for path in outputs:
            if not path.is_file():
                continue
            with rasterio.open(path) as dataset:
                for index in range(1, dataset.count + 1):
                    data = dataset.read([index])[0].astype("float64")
                    nodata = dataset.nodata
                    mask = np.isfinite(data)
                    if nodata is not None and np.isfinite(nodata):
                        mask &= data != nodata
                    valid = data[mask]
                    report.append({
                        "path": str(path),
                        "band": index,
                        "invalid_fraction": round(float(1.0 - valid.size / data.size), 6),
                        "min": None if not valid.size else float(valid.min()),
                        "max": None if not valid.size else float(valid.max()),
                    })
        return report

    # -- stages ------------------------------------------------------------
    def stage_derived(self, product: str, day: date):
        if product == "VPD":
            return self._derive_vpd(day)
        if product == "FWI":
            return self._derive_fwi(day)
        raise RuntimeError(f"unknown product {product!r}")

    def _derive_vpd(self, day: date):
        source, tiles = self.input_tiles(day, "VPD")
        destination = self.stage_dir("derived", "VPD")
        inputs, outputs = [], []
        for tile in sorted(tiles):
            tile_path = tiles[tile]
            target = destination / f"VPD_{registry.date_token(day)}_tile{tile:02d}.tif"
            if not target.exists() or self.overwrite:
                data, profile = derive.vapour_pressure_deficit(tile_path, source)
                derive.write_single_band(data, profile, target)
            inputs.append(tile_path)
            outputs.append(target)
        self._stage_extra["weather_source"] = source.key
        return inputs, outputs, ["weather_derive.vapour_pressure_deficit", day.isoformat()]

    def _fwi_state_path(self) -> Path:
        return self.state_root / "fwi_state.npz"

    def _derive_fwi(self, day: date):
        """Advance the moisture codes one day and mosaic the six components.

        The recursion is the reason this stage is not idempotent in the way the
        others are: re-running a date that has already been folded into the
        state would double-count it.  The checkpoint therefore gates the work,
        not the output file.
        """
        state_path = self._fwi_state_path()
        source, tiles = self.input_tiles(day, "FWI")

        extra: dict = {"weather_source": source.key}
        if state_path.is_file():
            state = derive.FWIState.load(state_path)
            gap = (day - state.day).days
            if gap <= 0:
                raise RuntimeError(
                    f"FWI {day}: checkpoint already stands at {state.day}; "
                    "the recursion only moves forward. Delete the checkpoint to rebuild."
                )
            if gap > 1:
                if not self.bridge_missing:
                    raise RuntimeError(
                        f"FWI {day}: checkpoint stands at {state.day}, a {gap - 1}-day gap. "
                        "Backfill the missing dates, or pass --bridge-missing to carry "
                        "the moisture codes across them."
                    )
                extra["bridged_days"] = gap - 1
                extra["bridged_from"] = state.day.isoformat()
        else:
            shapes = {}
            for tile, path in tiles.items():
                with rasterio.open(path) as dataset:
                    shapes[tile] = (dataset.height, dataset.width)
            state = derive.FWIState(shapes)
            extra["cold_start"] = True
            extra["cold_start_values"] = dict(registry.FWI_INIT)

        per_tile = {
            tile: derive.advance_tile(state.state[tile], tiles[tile], day, source)
            for tile in sorted(tiles)
        }

        outputs: list[Path] = []
        if self.state_only:
            # The historical spin-up leg only needs the recursion to advance.
            # Mosaicking and writing the six-band national raster is the bulk of
            # both the runtime and the ~47 MB/day this stage would otherwise
            # produce, and nothing reads those dates.
            extra["state_only"] = True
        else:
            context = derive.mosaic_context(tiles)
            bands = derive.mosaic_components(per_tile, context)
            # Bare-date naming, as `merge_patches` produces for every other
            # national raster, so one glob pattern serves every stage.  FireMPC
            # called this file `fwis_<date>.tif` only because its exporter wrote
            # into a directory shared with per-tile inputs.
            target = self.stage_dir("derived", "FWI") / f"{registry.date_token(day)}.tif"
            derive.write_fwi(bands, context, target)
            outputs = [target]

        state.day = day
        state.save(state_path)
        extra["checkpoint"] = str(state_path)
        # Masked pixels do not go NaN, they freeze -- see
        # `weather_derive.frozen_fraction`.  ERA5-Land is masked over water and
        # IFS is not, so this jumps at the changeover, and a frozen code is
        # invisible in the raster unless it is counted here.
        extra["frozen_fraction"] = {
            str(tile): round(derive.frozen_fraction(tiles[tile], source), 6)
            for tile in sorted(tiles)
        }
        self._stage_extra = extra
        return sorted(tiles.values()), outputs, ["weather_derive.advance_tile", day.isoformat()]

    def stage_merged(self, product: str, day: date):
        if product != "VPD":
            raise RuntimeError(f"{product} is already national and must not be merged")
        token = registry.date_token(day)
        inputs = sorted(self.stage_dir("derived", product).glob(f"*{token}*tile*.tif"))
        if len(inputs) != len(registry.TILES):
            raise RuntimeError(f"{product} {day}: expected four merge patches, found {len(inputs)}")
        with self._work(product, day, "merged") as work:
            source, out = work / "in", work / "out"
            self._farm(source, inputs)
            command = [str(self.python), str(FIREMPC / "merge_patches.py"),
                       "--input-dir", str(source), "--output-dir", str(out),
                       "--date", token, "--overwrite"]
            self._run(command, work, work / "merge.log")
            made = sorted(out.glob("*.tif"))
            if len(made) != 1:
                raise RuntimeError(f"{product} {day}: merge produced {len(made)} rasters")
            outputs = self._publish(made, self.stage_dir("merged", product))
        return inputs, outputs, command

    def stage_unified(self, product: str, day: date):
        source_stage = "derived" if product == "FWI" else "merged"
        token = registry.date_token(day)
        inputs = sorted(self.stage_dir(source_stage, product).glob(f"*{token}*.tif"))
        if len(inputs) != 1:
            raise RuntimeError(f"{product} {day}: expected one national raster, found {len(inputs)}")
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
            if (result.shape != template.shape or result.crs != template.crs
                    or not result.transform.almost_equals(template.transform)):
                raise RuntimeError(f"{outputs[0]} is not on GRID_TEMPLATE")
        return inputs, outputs, command

    def stage_scaled(self, product: str, day: date):
        token = registry.date_token(day)
        inputs = sorted(self.stage_dir("unified", product).glob(f"*{token}*.tif"))
        if len(inputs) != 1:
            raise RuntimeError(f"{product} {day}: expected one unified raster, found {len(inputs)}")
        route = registry.ROUTES[product]
        with self._work(product, day, "scaled") as work:
            root = work / "in"
            source = root / route.output_dir
            out_root = work / "out"
            self._farm(source, inputs)
            # The band configuration is passed explicitly rather than left to
            # scale_int_tif's directory-name matching, so the frozen vmin/vmax
            # travel with this registry instead of with a folder name.
            command = [str(self.python), str(FIREMPC / "scale_int_tif.py"),
                       "--input_folder", str(root), "--output_folder", str(out_root),
                       "--band-config", registry.SCALE_BAND_CONFIG[product],
                       "--local-stage-root", str(work / "scratch"), "--max-workers", "1"]
            self._run(command, work, work / "scale.log")
            made = sorted(out_root.rglob("*.tif"))
            if len(made) != 1:
                raise RuntimeError(f"{product} {day}: scaling produced {len(made)} rasters")
            outputs = self._publish(made, self.stage_dir("scaled", product))
        return inputs, outputs, command

    # -- driver ------------------------------------------------------------
    def run_stage(self, product: str, day: date, stage: str) -> None:
        started = _utc_now()
        inputs: list[Path] = []
        outputs: list[Path] = []
        command = None
        self._stage_extra: dict = {}
        try:
            method = getattr(self, f"stage_{stage}")
            inputs, outputs, command = method(product, day)
            self._record(product, day, stage, started, "complete", inputs, outputs,
                         command=command, extra=self._stage_extra or None)
        except Exception as exc:
            self._record(product, day, stage, started, "failed", inputs, outputs,
                         command=command, error=str(exc), extra=self._stage_extra or None)
            raise

    def stage_is_done(self, product: str, day: date, stage: str) -> bool:
        """Has this stage already produced a validated output for this date?

        `derived` for FWI is excluded on purpose: the checkpoint, not the file,
        decides whether that stage may run.
        """
        if product == "FWI" and stage == "derived":
            state_path = self._fwi_state_path()
            if not state_path.is_file():
                return False
            return derive.FWIState.load(state_path).day >= day
        token = registry.date_token(day)
        expected = len(registry.TILES) if (product == "VPD" and stage == "derived") else 1
        found = sorted(self.stage_dir(stage, product).glob(f"*{token}*.tif"))
        if len(found) != expected:
            return False
        try:
            for path in found:
                with rasterio.open(path) as dataset:
                    if dataset.width <= 0 or dataset.height <= 0:
                        return False
        except rasterio.errors.RasterioError:
            return False
        return True


@app.command()
def main(
    product: Annotated[str, typer.Option("--product", help="VPD or FWI")],
    start: Annotated[datetime, typer.Option("--start", formats=["%Y-%m-%d"])],
    end: Annotated[datetime, typer.Option("--end", formats=["%Y-%m-%d"])],
    stage: Annotated[str, typer.Option("--stage", help="One stage, or 'all'.")] = "all",
    drivers: Annotated[Path, typer.Option("--drivers")] = Path(paths.DRIVERS),
    data_root: Annotated[Path, typer.Option("--data-root")] = Path(paths.PROCESSED_WEATHER),
    report_root: Annotated[Path, typer.Option("--report-root")] = Path(paths.WEATHER_PROCESSING_REPORTS),
    state_root: Annotated[Path, typer.Option("--state-root")] = Path(paths.FWI_STATE),
    template: Annotated[Path, typer.Option("--grid-template")] = Path(paths.GRID_TEMPLATE),
    firempc_python: Annotated[Path, typer.Option("--firempc-python")] = DEFAULT_FIREMPC_PYTHON,
    source: Annotated[str, typer.Option(
        "--source", help="ifs, era5, or auto (ERA5-Land before the first IFS cycle).",
    )] = "auto",
    resume: Annotated[bool, typer.Option("--resume")] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    bridge_missing: Annotated[bool, typer.Option("--bridge-missing")] = False,
    state_only: Annotated[bool, typer.Option(
        "--state-only",
        help="FWI only: advance the moisture codes without writing rasters.",
    )] = False,
) -> None:
    if product not in registry.ROUTES:
        raise typer.BadParameter(f"--product must be one of {sorted(registry.ROUTES)}")
    if source.lower() != "auto":
        try:
            registry.source_by_key(source)
        except KeyError as exc:
            raise typer.BadParameter(str(exc)) from None
    route = registry.ROUTES[product]
    stages = route.stages if stage == "all" else (stage,)
    unknown = [name for name in stages if name not in route.stages]
    if unknown:
        raise typer.BadParameter(f"{product} has no stage {unknown[0]!r}; it has {route.stages}")
    if state_only:
        # Without a `derived` raster there is nothing for the later stages to
        # align or quantise, so state-only is the `derived` stage alone.  It is
        # refused rather than silently narrowed when another stage was asked for.
        if product != "FWI":
            raise typer.BadParameter("--state-only applies to FWI, the only recursive product")
        if stage not in ("all", "derived"):
            raise typer.BadParameter(
                f"--state-only writes no raster, so stage {stage!r} would have no input")
        stages = ("derived",)

    processor = Processor(
        drivers=drivers, data_root=data_root, report_root=report_root,
        state_root=state_root, template=template, firempc_python=firempc_python,
        overwrite=overwrite, bridge_missing=bridge_missing,
        state_only=state_only, source=source,
    )

    first, last = start.date(), end.date()
    if last < first:
        raise typer.BadParameter("--end precedes --start")

    print(f"{product}  {first} .. {last}  stages={list(stages)}  source={source}"
          + ("  state-only" if state_only else ""))
    failures = 0
    for day in _dates(first, last):
        for name in stages:
            if resume and not overwrite and processor.stage_is_done(product, day, name):
                continue
            try:
                processor.run_stage(product, day, name)
                print(f"  {day}  {name}")
            except Exception as exc:  # noqa: BLE001 - one bad date must not stop a backfill
                failures += 1
                print(f"  {day}  {name}  FAILED: {exc}", file=sys.stderr)
                # FWI is recursive: continuing past a failed advance would
                # silently drop a day from every later moisture code.
                if product == "FWI" and name == "derived":
                    raise typer.Exit(1)
                break

    if failures:
        print(f"{failures} stage run(s) failed", file=sys.stderr)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
