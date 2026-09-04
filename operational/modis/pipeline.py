"""Top-level operational MODIS orchestration.

Each child stage remains directly runnable; this command only connects their
validated handoffs for one target date.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated, Literal

import typer

# Run directly as `python operational/modis/pipeline.py`: put the application root on
# sys.path so the shared modules import by bare name, as elsewhere in FireMPC.
OPERATIONAL = Path(__file__).resolve().parents[1]
if str(OPERATIONAL) not in sys.path:
    sys.path.insert(0, str(OPERATIONAL))

import ops, paths, vendorpaths
from modis.processing_registry import require_operational_lineage
from modis.registry import OPERATIONAL_EXPORTS, PRODUCTS

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
STAGES = ("download", "preprocess", "package")


def operational_collections() -> tuple[str, ...]:
    return tuple(dict.fromkeys(PRODUCTS[name]["earthaccess"]["short_name"] for name in OPERATIONAL_EXPORTS))


def package_command(target: str, fill_policy: Path, *, as_of: str | None, overwrite: bool) -> list[str]:
    """The `modis package` invocation for one target.

    `as_of` is what separates a backfill from an operational run: without it
    `resolve_source` takes the newest observation on disk, which for a rebuilt
    date silently includes granules published weeks after it.  Kept as a
    function so the flag's presence is testable without running a stage.
    """
    command = ops.cli(
        "modis", "package",
        "--target", target, "--fill-policy", str(fill_policy),
    )
    if as_of is not None:
        command += ["--as-of", as_of]
    if overwrite:
        command.append("--overwrite")
    return command


@app.command()
def main(
    target: Annotated[str, typer.Option("--target", help="Prediction date D (YYYY-MM-DD).")],
    backend: Annotated[Literal["ea", "gee"], typer.Option("--backend")] = "ea",
    stream: Annotated[Literal["standard", "nrt"], typer.Option("--stream")] = "standard",
    from_stage: Annotated[str, typer.Option("--from-stage")] = "download",
    to_stage: Annotated[str, typer.Option("--to-stage")] = "package",
    resume: Annotated[bool, typer.Option("--resume")] = True,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    tiling_manifest: Annotated[Path | None, typer.Option("--tiling-manifest")] = None,
    modis_policy: Annotated[Literal["strict", "latest-available"], typer.Option("--modis-policy")] = "strict",
    lookback_days: Annotated[int, typer.Option("--lookback-days", min=10)] = 45,
    fill_policy: Annotated[Path | None, typer.Option("--fill-policy")] = None,
    as_of: Annotated[str | None, typer.Option(
        "--as-of",
        help="Resolve MODIS only against observations that had landed by this "
             "date (YYYY-MM-DD). Use it for backfills; an operational run wants "
             "the default, which is whatever is on disk.",
    )] = None,
) -> None:
    require_operational_lineage()
    selected = ops.select_stages(STAGES, from_stage, to_stage)
    target_day = date.fromisoformat(target)
    if as_of is not None:
        try:
            date.fromisoformat(as_of)
        except ValueError:
            raise typer.BadParameter("--as-of must be YYYY-MM-DD") from None
    if fill_policy is None:
        fill_policy = vendorpaths.CONFIG / (
            "modis_latest_available_policy.json"
            if modis_policy == "latest-available" else "modis_fill_policy.json"
        )
    start = target_day - timedelta(days=lookback_days if modis_policy == "latest-available" else 10)
    end = target_day - timedelta(days=1)
    run_id = ops.run_id()
    report_dir = Path(paths.MODIS_PROCESSING_REPORTS) / "pipeline" / run_id
    with ops.record(report_dir, {
        "schema_version": 1, "run_id": run_id, "target_date": target,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "backend": backend, "stream": stream, "modis_policy": modis_policy,
        "lookback_days": (lookback_days if modis_policy == "latest-available" else 10),
        "as_of": as_of,
        "stages": {},
    }) as record:
        if "download" in selected:
            command = ops.cli(
                "modis", "download", backend,
                "--start", start.isoformat(), "--end", end.isoformat(),
                "--stream", stream, "--products", ",".join(operational_collections()),
            )
            if overwrite:
                # The downloader is already idempotent and exposes no global
                # overwrite switch; tiling/preprocessing own replacement.
                pass
            # The downloader separates "the manifest records errors" (1) from
            # "everything landed, but a product is later than its measured p95"
            # (2). Only the first is a broken window; lateness is what the
            # latest-available fill policy exists to absorb, so it must not stop
            # the run -- it is recorded instead.
            code = ops.run(command, report_dir / "download.log", tolerate=(2,))
            record["stages"]["download"] = "complete with latency alerts" if code == 2 else "complete"
            tiling_manifest = Path(paths.REPORTS) / "tiling" / "latest.json"
        if "preprocess" in selected:
            if tiling_manifest is None:
                tiling_manifest = Path(paths.REPORTS) / "tiling" / "latest.json"
            command = ops.cli(
                "modis", "preprocess",
                "--product", "all", "--start", start.isoformat(), "--end", end.isoformat(),
                "--stage", "all", "--tiling-manifest", str(tiling_manifest),
            )
            if resume and not overwrite:
                command.append("--resume")
            if overwrite:
                command.append("--overwrite")
            ops.run(command, report_dir / "preprocess.log")
            record["stages"]["preprocess"] = "complete"
        if "package" in selected:
            command = package_command(target, fill_policy, as_of=as_of, overwrite=overwrite)
            ops.run(command, report_dir / "package.log")
            record["stages"]["package"] = "complete"
    typer.echo(Path(paths.MODIS_PACKAGES) / target)


if __name__ == "__main__":
    app()
