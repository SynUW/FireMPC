"""The command tree, and the stage invocations the pipelines build from it.

The pipelines do not import the stages they drive -- they re-invoke them as
subprocesses.  Every test that exercises a pipeline stubs the `run` helper, so
a wrong command path inside one of those argv lists is invisible to the rest of
the suite and would surface only as a failed operational run.  These tests read
the argv lists back out of the source and check them against the real tree.
"""

from __future__ import annotations

import ast
import subprocess
import sys

import pytest
import typer
from typer.testing import CliRunner

from wildfire import vendorpaths
from wildfire.cli import app

PIPELINES = [
    "src/wildfire/modis/pipeline.py",
    "src/wildfire/weather/pipeline.py",
    "src/wildfire/canfirecast/pipeline.py",
    "src/wildfire/canfirecast/state_ledger.py",
]


def _command_paths(typer_app, prefix=()):
    """Every runnable leaf in the tree, as a tuple of command names."""
    for command in typer_app.registered_commands:
        yield (*prefix, command.name or command.callback.__name__.replace("_", "-"))
    for group in typer_app.registered_groups:
        yield from _command_paths(group.typer_instance, (*prefix, group.name))


ALL_PATHS = sorted(_command_paths(app))


def _invocations(path):
    """The ("-m", "wildfire.cli", ...) argv lists built in one source file."""
    tree = ast.parse((vendorpaths.REPO_ROOT / path).read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.List):
            continue
        literals = [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        if "-m" in literals and "wildfire.cli" in literals:
            tail = literals[literals.index("wildfire.cli") + 1:]
            # Stop at the first option; the rest are arguments, not command names.
            names = []
            for token in tail:
                if token.startswith("-"):
                    break
                names.append(token)
            yield tuple(names)


def test_the_tree_has_the_expected_shape():
    assert ("modis", "pipeline") in ALL_PATHS
    assert ("weather", "pipeline") in ALL_PATHS
    assert ("canfirecast", "pipeline") in ALL_PATHS
    assert ("canfirecast", "model", "correct") in ALL_PATHS
    assert ("canfirecast", "state-ledger", "advance") in ALL_PATHS
    assert ("canfirecast", "inference") not in ALL_PATHS
    assert not any(path[0] == "dashboard" for path in ALL_PATHS)
    assert len(ALL_PATHS) == len(set(ALL_PATHS)), "duplicate command path"


@pytest.mark.parametrize("path", PIPELINES)
def test_every_pipeline_invocation_names_a_real_command(path):
    """The check that would have caught a stale `HERE / "script.py"` rewrite."""
    found = list(_invocations(path))
    assert found, f"{path} builds no wildfire.cli invocation"
    for names in found:
        assert names in ALL_PATHS, (
            f"{path} invokes `wildfire {' '.join(names)}`, which is not a command. "
            f"Known: {[' '.join(p) for p in ALL_PATHS]}")


def test_no_pipeline_still_invokes_a_bare_script():
    for path in PIPELINES:
        source = (vendorpaths.REPO_ROOT / path).read_text()
        assert 'HERE / "' not in source, f"{path} still resolves a sibling script by path"


@pytest.mark.parametrize("path", ALL_PATHS, ids=lambda p: " ".join(p))
def test_every_command_renders_help(path):
    """A bad signature or a missing dependency shows up here, not in production."""
    result = CliRunner().invoke(app, [*path, "--help"])
    assert result.exit_code == 0, result.output


def test_the_console_script_is_installed():
    out = subprocess.run([sys.executable, "-m", "wildfire.cli", "--help"],
                         capture_output=True, text=True, check=True).stdout
    for domain in ("modis", "weather", "canfirecast"):
        assert domain in out


def test_every_readme_invocation_names_a_real_command():
    """Keep documented ``python -m wildfire.cli`` commands synchronized."""
    readme = (vendorpaths.OPERATIONAL_ROOT / "README.md").read_text()
    invocations = []
    prefix = "python -m wildfire.cli "
    for line in readme.splitlines():
        line = line.strip().removesuffix("\\").strip()
        if not line.startswith(prefix):
            continue
        names = []
        for token in line[len(prefix):].split():
            if token.startswith("-"):
                break
            names.append(token)
        if names:  # top-level ``--help`` is not a command leaf
            invocations.append((line, tuple(names)))
    assert invocations, "README documents no wildfire CLI invocations"
    for line, names in invocations:
        assert names in ALL_PATHS, f"README line is not a command: {line}"
