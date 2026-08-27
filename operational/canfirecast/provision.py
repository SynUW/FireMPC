"""Create and verify an immutable CanFireCast input-asset bundle."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import typer

# Run directly as `python operational/canfirecast/provision.py`: put the application root on
# sys.path so the shared modules import by bare name, as elsewhere in FireMPC.
import sys
from pathlib import Path
OPERATIONAL = Path(__file__).resolve().parents[1]
if str(OPERATIONAL) not in sys.path:
    sys.path.insert(0, str(OPERATIONAL))

from canfirecast import input_registry as registry
import paths
from canfirecast import static_package
from canfirecast.static_package import build_static_package

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
DEFAULT_ROOT = Path(paths.CANFIRECAST_ASSETS)


def _tree_sha256(root: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    for path in sorted(
        p for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    ):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(bytes.fromhex(registry.sha256(path)))
    return digest.hexdigest()


def _copy(source: Path, destination: Path) -> dict:
    if not source.is_file(): raise RuntimeError(f"missing release source: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {"path": destination.name, "sha256": registry.sha256(destination), "bytes": destination.stat().st_size, "source": str(source)}


def verify_release(release_dir: Path) -> dict:
    manifest_path = release_dir / "release.json"
    if not manifest_path.is_file(): raise RuntimeError(f"missing release manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    for name, item in manifest["artifacts"].items():
        path = release_dir / item["path"]
        if item.get("kind") == "tree":
            if not path.is_dir(): raise RuntimeError(f"release {name} missing: {path}")
            actual = _tree_sha256(path)
        else:
            if not path.is_file(): raise RuntimeError(f"release {name} missing: {path}")
            actual = registry.sha256(path)
        if actual != item["sha256"]: raise RuntimeError(f"release {name} hash mismatch")
    return manifest


def provision(
    *,
    release: str,
    overwrite: bool = False,
    grid_template: Path | None = None,
    osm: Path | None = None,
    population: Path | None = None,
    terrain: Path | None = None,
) -> Path:
    registry.verify_pinned_artifacts()
    grid_template = Path(grid_template or paths.GRID_TEMPLATE)
    osm = Path(osm or static_package.DEFAULT_OSM)
    population = Path(population or static_package.DEFAULT_POPULATION)
    terrain = Path(terrain or static_package.DEFAULT_TERRAIN)
    root = DEFAULT_ROOT
    final = root / release
    if final.exists() and not overwrite: raise FileExistsError(f"release exists: {final}")
    root.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{release}-", dir=root))
    try:
        artifacts = {
            "grid": _copy(grid_template, temp / "national_grid.tif"),
        }
        statics = build_static_package(
            osm=osm,
            population=population,
            terrain=terrain,
            template_path=grid_template, output=temp / "statics.h5",
        )
        artifacts["statics"] = {"path": statics.name, "sha256": registry.sha256(statics), "bytes": statics.stat().st_size, "source": "pinned 2025 OSM/WorldPop + MERIT"}
        for name, meta in registry.ARTIFACTS.items(): artifacts[name] = _copy(Path(meta["path"]), temp / name)
        manifest = {
            "schema_version": 1, "release_id": release,
            "registry_version": registry.REGISTRY_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(), "artifacts": artifacts,
            "weather_alpha": registry.WEATHER_ALPHA, "state_correction_enabled": False,
        }
        (temp / "release.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        verify_release(temp)
        if final.exists(): shutil.rmtree(final)
        os.replace(temp, final)
    finally:
        if temp.exists(): shutil.rmtree(temp)
    return final


@app.command("create")
def create(
    release: Annotated[str, typer.Option("--release")],
    grid_template: Annotated[Path | None, typer.Option("--grid-template")] = None,
    osm: Annotated[Path | None, typer.Option("--osm")] = None,
    population: Annotated[Path | None, typer.Option("--population")] = None,
    terrain: Annotated[Path | None, typer.Option("--terrain")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    typer.echo(provision(
        release=release, overwrite=overwrite, grid_template=grid_template,
        osm=osm, population=population, terrain=terrain,
    ))


@app.command("verify")
def verify(release: Annotated[str, typer.Option("--release")]) -> None:
    verify_release(DEFAULT_ROOT / release); typer.echo(DEFAULT_ROOT / release)


if __name__ == "__main__": app()
