"""The repository-relative anchors the packaged modules depend on.

These guard the reorganisation in plans/PACKAGE_REORG_PLAN.md.  Every module
used to resolve `vendor/`, `training_code/` and the policy files relative to its
own `__file__`, which held only while all of them sat at the repository root.
Nothing in the existing suite would have noticed those anchors going stale --
`code_revision()` in particular degrades to the string "unknown" rather than
raising, which would have silently stripped provenance from published packages.
"""

from __future__ import annotations

import subprocess

import pytest

from canfirecast import package as build_canfirecast_package
from modis import package as build_modis_package
import paths, vendorpaths


def test_monorepo_anchors():
    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=vendorpaths.OPERATIONAL_ROOT,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert vendorpaths.GIT_ROOT == type(vendorpaths.GIT_ROOT)(top)
    assert vendorpaths.FIREMPC_ROOT == vendorpaths.GIT_ROOT
    assert vendorpaths.OPERATIONAL_ROOT == vendorpaths.FIREMPC_ROOT / "operational"
    assert vendorpaths.OPERATIONAL_ROOT.relative_to(vendorpaths.GIT_ROOT).as_posix() == "operational"


@pytest.mark.parametrize(
    "anchor",
    ["VENDOR", "FROZEN_REPO", "CANFIRECAST_VENDOR", "CONFIG", "FIREMPC_NEWSET", "FIREMPC_FWI"],
)
def test_required_anchors_exist(anchor):
    assert getattr(vendorpaths, anchor).is_dir(), f"{anchor} does not resolve"


def test_policy_files_resolve():
    for name in ("modis_fill_policy.json", "modis_latest_available_policy.json"):
        assert (vendorpaths.CONFIG / name).is_file()
    assert build_modis_package.DEFAULT_POLICY.is_file()


def test_ensure_frozen_repo_is_idempotent():
    import sys
    frozen = str(vendorpaths.ensure_frozen_repo())
    before = sys.path.count(frozen)
    vendorpaths.ensure_frozen_repo()
    assert sys.path.count(frozen) == before == 1


@pytest.mark.parametrize(
    "module", [build_modis_package, build_canfirecast_package], ids=["modis", "canfirecast"]
)
def test_code_revision_reports_the_real_head(module):
    """A stale `cwd` would return "unknown" and quietly de-provenance a package."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=vendorpaths.GIT_ROOT,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    revision = module.code_revision()
    assert revision != "unknown"
    assert revision.startswith(head)


# Values as merged before the dependency inversion, when `paths` imported
# `modis_registry` directly.  Frozen here so the registration hook cannot change
# the contract silently.
EXPECTED_NATIVE_RES_M = {
    "ERA5": 11132, "IFS": 11132,
    "MCD12Q1": 500, "MCD15A3H": 500,
    "MOD09A1": 500, "MYD09A1": 500,
    "MOD09CMG": 5600, "MYD09CMG": 5600,
    "MOD09GA_QA_state1km": 1000, "MYD09GA_QA_state1km": 1000,
    "MOD09GA_b1237": 500, "MYD09GA_b1237": 500,
    "MOD11A1": 1000, "MYD11A1": 1000,
    "MOD11A2": 1000, "MYD11A2": 1000,
    "MOD13Q1": 250, "MYD13Q1": 250,
    "MOD14A1": 1000, "MYD14A1": 1000,
}


def test_native_resolutions_survive_the_dependency_inversion():
    from modis import registry  # noqa: F401  -- registers the MODIS half
    assert paths.NATIVE_RES_M == EXPECTED_NATIVE_RES_M


def test_registering_a_conflicting_resolution_is_an_error():
    with pytest.raises(ValueError, match="conflicting native resolution"):
        paths.register_native_resolutions({"IFS": 9999})
    assert paths.NATIVE_RES_M["IFS"] == 11132


def test_paths_does_not_import_a_domain_package():
    source = (vendorpaths.OPERATIONAL_ROOT / "paths.py").read_text()
    for domain in ("modis", "weather", "canfirecast"):
        assert f"import {domain}" not in source, (
            f"paths must not depend on the {domain} domain")


DOMAINS = ("modis", "weather", "canfirecast")


def application_sources():
    """Every .py file this application ships, excluding vendored code and tests."""
    root = vendorpaths.OPERATIONAL_ROOT
    files = sorted(root.glob("*.py"))
    for domain in DOMAINS:
        files.extend(sorted((root / domain).rglob("*.py")))
    return files


def test_the_source_sweep_actually_finds_sources():
    """A moved tree would make every sweep below vacuously pass."""
    found = application_sources()
    assert len(found) > 25, f"only {len(found)} sources found; the layout moved"


def test_runtime_has_no_source_checkout_assumptions():
    forbidden = ("/home/quinn/wildfire_prediction", "training_code/FireMPC")
    for path in application_sources():
        text = path.read_text(encoding="utf-8")
        for value in forbidden:
            assert value not in text, f"{path} contains forbidden path {value}"


def test_nothing_still_imports_the_retired_package():
    """The tree is plain directories on sys.path now, not a `wildfire` package."""
    for path in application_sources():
        text = path.read_text(encoding="utf-8")
        for form in ("from wildfire", "import wildfire", '"wildfire.'):
            assert form not in text, f"{path} still references the wildfire package"


def test_every_runnable_domain_script_bootstraps_the_root():
    """`python operational/modis/pipeline.py` only imports if the root is added.

    Nothing else notices a missing bootstrap: the CLI reaches these modules by
    import, where sys.path is already correct, so the break shows up only when
    someone runs the script the way the rest of FireMPC is run.
    """
    for path in application_sources():
        if path.parent == vendorpaths.OPERATIONAL_ROOT:
            continue  # a root-level script's own directory is already sys.path[0]
        text = path.read_text(encoding="utf-8")
        if "__main__" not in text:
            continue
        depth = len(path.relative_to(vendorpaths.OPERATIONAL_ROOT).parts) - 1
        expected = f"Path(__file__).resolve().parents[{depth}]"
        assert expected in text, f"{path} is runnable but never puts the root on sys.path"
