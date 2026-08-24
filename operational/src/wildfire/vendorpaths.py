"""Repository and vendor anchors for the operational application.

The operational code runs from ``FireMPC/operational``. Keep these anchors distinct:
``OPERATIONAL_ROOT`` is this application directory, ``FIREMPC_ROOT`` is the parent
FireMPC checkout containing legacy scripts, and ``GIT_ROOT`` is where Git
provenance is queried.  ``REPO_ROOT`` remains as a compatibility alias for the
operational project root.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_DEPTH = 2
OPERATIONAL_ROOT = Path(__file__).resolve().parents[_DEPTH]
REPO_ROOT = OPERATIONAL_ROOT  # Backwards-compatible name used by older tests/code.


def _looks_like_firempc(root: Path) -> bool:
    return (root / "newset_processing").is_dir() and (root / "fwi").is_dir()


def _resolve_firempc_root() -> Path:
    override = os.environ.get("FIREMPC_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if not _looks_like_firempc(root):
            raise RuntimeError(
                f"FIREMPC_ROOT={root} does not look like a FireMPC checkout; "
                "expected newset_processing/ and fwi/."
            )
        return root
    parent = OPERATIONAL_ROOT.parent
    if _looks_like_firempc(parent):
        return parent
    raise RuntimeError(
        f"cannot locate FireMPC checkout above {OPERATIONAL_ROOT}; expected "
        "newset_processing/ and fwi/, or set FIREMPC_ROOT explicitly."
    )


def _git_root() -> Path:
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=OPERATIONAL_ROOT,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        return Path(top).resolve()
    except (OSError, subprocess.CalledProcessError):
        return OPERATIONAL_ROOT


FIREMPC_ROOT = _resolve_firempc_root()
GIT_ROOT = _git_root()

# Vendored third-party sources, pinned and checksummed in vendor/PROVENANCE.json.
VENDOR = OPERATIONAL_ROOT / "vendor"

# The frozen CanFireCast training repo: cffdrs_aware_correction,
# full10_ifs_domain_transform and ifs_to_era5_physical_adapter import from here.
FROZEN_REPO = VENDOR / "canfirecast" / "frozen_repo"
CANFIRECAST_VENDOR = VENDOR / "canfirecast"

# Top-level FireMPC legacy code.
FIREMPC_NEWSET = FIREMPC_ROOT / "newset_processing"
FIREMPC_FWI = FIREMPC_ROOT / "fwi"

# Versioned policy files that are inputs to the pipeline, not code.
CONFIG = OPERATIONAL_ROOT / "config"


def operational_git_pathspec() -> list[str]:
    """Pathspec limiting dirty checks to operational files when in FireMPC."""
    try:
        rel = OPERATIONAL_ROOT.relative_to(GIT_ROOT)
    except ValueError:
        return []
    return [] if rel == Path(".") else [rel.as_posix() + "/"]


def git_provenance() -> dict[str, str | bool]:
    """Return FireMPC and operational-tree identities with scoped dirtiness."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=GIT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        tree_result = subprocess.run(
            ["git", "rev-parse", "HEAD:operational"], cwd=GIT_ROOT,
            capture_output=True, text=True,
        )
        tree = tree_result.stdout.strip() if tree_result.returncode == 0 else "uncommitted"
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", *operational_git_pathspec()],
            cwd=GIT_ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"firempc_commit": "unknown", "operational_tree": "unknown", "operational_dirty": True}
    return {"firempc_commit": commit, "operational_tree": tree, "operational_dirty": dirty}


def imported_source_provenance() -> dict[str, str]:
    """Read the source snapshot identity recorded in IMPORT_PROVENANCE.md."""
    import re

    text = (OPERATIONAL_ROOT / "IMPORT_PROVENANCE.md").read_text(encoding="utf-8")
    commit = re.search(r"Source commit: `([0-9a-f]{40})`", text)
    snapshot = re.search(r"Snapshot UTC: `([^`]+)`", text)
    identity = re.search(r"snapshot identity\n`([0-9a-f]{64})`", text)
    if not commit or not snapshot or not identity:
        raise RuntimeError("IMPORT_PROVENANCE.md lacks source commit, snapshot UTC, or identity")
    return {
        "source_commit": commit.group(1),
        "snapshot_utc": snapshot.group(1),
        "snapshot_identity": identity.group(1),
    }


def ensure_frozen_repo() -> Path:
    """Put the frozen CanFireCast repo on `sys.path` and return it."""
    if not FROZEN_REPO.is_dir():
        raise FileNotFoundError(
            f"vendored CanFireCast frozen repo is missing: {FROZEN_REPO}. "
            "The model correction and CFFDRS stages cannot run without it."
        )
    entry = str(FROZEN_REPO)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return FROZEN_REPO
