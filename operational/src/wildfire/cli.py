"""The single entry point for every operational stage.

Each domain module already carried its own `typer.Typer()`; this assembles
them into one tree so `wildfire --help` is the index of what is runnable.
Before this existed, README.md was that index, and it listed twelve
invocations across three naming conventions (`run_*`, `build_*`, and bare
nouns) that gave no hint of which were pipelines and which were single stages.

Two shapes are wired differently:

* A module whose app holds exactly one command contributes that command
  directly, so `wildfire modis preprocess` is a command and not a group with a
  single member underneath it.
* A module with several commands is attached whole, keeping its existing
  sub-names (`wildfire canfirecast provision create`).

`wildfire.modis.latency` and `wildfire.modis.to_tiles` predate typer and drive
argparse themselves; they are wrapped so the tree is complete rather than
rewritten, which would change their flags.
"""

from __future__ import annotations

import sys

import typer

from wildfire import daily as daily_run
from wildfire.canfirecast import package as canfirecast_package
from wildfire.canfirecast import pipeline as canfirecast_pipeline
from wildfire.canfirecast import provision as canfirecast_provision
from wildfire.canfirecast import state_ledger as canfirecast_state_ledger
from wildfire.canfirecast import static_package as canfirecast_static_package
from wildfire.canfirecast import weather_window as canfirecast_weather_window
from wildfire.canfirecast.model import cffdrs_derive as model_cffdrs_derive
from wildfire.canfirecast.model import weather_correct as model_weather_correct
from wildfire.modis import download as modis_download
from wildfire.modis import latency as modis_latency
from wildfire.modis import package as modis_package
from wildfire.modis import pipeline as modis_pipeline
from wildfire.modis import preprocess as modis_preprocess
from wildfire.modis import purge as modis_purge
from wildfire.modis import to_tiles as modis_to_tiles
from wildfire.weather import download as weather_download
from wildfire.weather import latency as weather_latency
from wildfire.weather import pipeline as weather_pipeline
from wildfire.weather import preprocess as weather_preprocess
from wildfire.weather import purge as weather_purge

app = typer.Typer(
    add_completion=False, pretty_exceptions_enable=False,
    help="Operational near-real-time wildfire model inputs for Canada.",
)


def _only_command(module):
    """The single registered command of a one-command module's app."""
    commands = module.app.registered_commands
    if len(commands) != 1:
        raise RuntimeError(
            f"{module.__name__} has {len(commands)} commands, expected exactly one; "
            "attach it with add_typer instead")
    return commands[0]


def _attach(group: typer.Typer, name: str, module, help_text: str) -> None:
    """Hoist a one-command module in as `group name`, not `group name name`."""
    command = _only_command(module)
    group.command(name, help=help_text or command.help)(command.callback)


# ---------------------------------------------------------------------------
# MODIS
# ---------------------------------------------------------------------------
modis = typer.Typer(add_completion=False, pretty_exceptions_enable=False,
                    help="MODIS granule ingest through the frozen 17-channel contract.")
app.add_typer(modis, name="modis")

_attach(modis, "download", modis_download, "Fetch granules and convert them to GEE-format tiles.")
_attach(modis, "preprocess", modis_preprocess, "Run the per-product processing routes.")
_attach(modis, "package", modis_package, "Assemble the ten-day HDF5 package for a target date.")
_attach(modis, "purge", modis_purge, "Delete raw granules whose tiles are complete and valid.")
_attach(modis, "pipeline", modis_pipeline, "Download, preprocess and package in one run.")


@modis.command("latency", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def _modis_latency(ctx: typer.Context) -> None:
    """Measure per-product domain-completion lag (argparse; pass its own flags)."""
    sys.argv = ["wildfire modis latency", *ctx.args]
    modis_latency.main()


@modis.command("to-tiles", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def _modis_to_tiles(ctx: typer.Context) -> None:
    """Convert HDF granules to GEE-format tiles (argparse; pass its own flags)."""
    sys.argv = ["wildfire modis to-tiles", *ctx.args]
    modis_to_tiles.main()


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------
weather = typer.Typer(add_completion=False, pretty_exceptions_enable=False,
                      help="IFS and ERA5 ingest, VPD, and the recursive FWI codes.")
app.add_typer(weather, name="weather")

_attach(weather, "download", weather_download, "Fetch IFS or ERA5 tiles from Earth Engine.")
_attach(weather, "preprocess", weather_preprocess, "Derive VPD and advance the FWI recursion.")
_attach(weather, "purge", weather_purge, "Delete weather tiles outside the retention window.")
_attach(weather, "pipeline", weather_pipeline, "Download and preprocess in one run.")
weather.add_typer(weather_latency.app, name="latency", help="Measure and report IFS/ERA5 availability lag.")


# ---------------------------------------------------------------------------
# CanFireCast
# ---------------------------------------------------------------------------
canfirecast = typer.Typer(add_completion=False, pretty_exceptions_enable=False,
                          help="Assembly of the operational 55-channel model input.")
app.add_typer(canfirecast, name="canfirecast")

_attach(canfirecast, "package", canfirecast_package, "Assemble the 55-channel model input package.")
_attach(canfirecast, "static-package", canfirecast_static_package, "Build the pinned OSM/terrain/population statics.")
_attach(canfirecast, "weather-window", canfirecast_weather_window, "Align and encode the ten-day weather window.")
_attach(canfirecast, "pipeline", canfirecast_pipeline, "The full daily operational run.")
canfirecast.add_typer(canfirecast_provision.app, name="provision", help="Create and verify immutable input assets.")
canfirecast.add_typer(canfirecast_state_ledger.app, name="state-ledger", help="Manage the certified D-11 state ledger.")

model = typer.Typer(add_completion=False, pretty_exceptions_enable=False,
                    help="Stages that run against the frozen training code in vendor/.")
canfirecast.add_typer(model, name="model")
_attach(model, "correct", model_weather_correct, "Apply the IFS-to-ERA5 physical correction.")
_attach(model, "derive", model_cffdrs_derive, "Derive the CFFDRS-aware corrected window.")


# ---------------------------------------------------------------------------
# Unattended
# ---------------------------------------------------------------------------
_attach(app, "daily", daily_run, "The unattended daily run, as invoked by the systemd timer.")


if __name__ == "__main__":
    app()
