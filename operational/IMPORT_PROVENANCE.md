# Operational source snapshot provenance

- Source project: `GSIL-UCalgary/wildfire_prediction` (`origin` uses the project's SSH GitHub URL)
- Source branch: `master`
- Source commit: `4f374c3dde9a6c36c03ca3906439b5974e44b3d6`
- Snapshot UTC: `2026-08-24T19:24:20Z`
- Import destination: `FireMPC/operational/`
- Snapshot method: reviewed direct file copy; no subtree, submodule, symlink, editable dependency, or runtime source-checkout reference

The source work tree contained the reviewed operational-integration changes at
snapshot time and was therefore not clean relative to the commit above. A
sorted SHA-256 manifest of the selected source files (`src/wildfire`, `config`,
`deploy`, `tests`, `vendor`, and `README.md`, excluding dashboard and cache
files) has snapshot identity
`0fd489a2f4810942240755407f1827667b4120fc2aba2fe8c29dfa95d8f487ad`.
This record is intentionally explicit rather than claiming a clean source tree.
FireMPC-specific path, environment, FWI, documentation, and test adaptations
were then made natively in this branch.

Excluded from the import:

- `.git/`, `.venv/`, caches, and bytecode;
- `training_code/`, `report/`, `plans/`, and `tools/`;
- `pyproject.toml`, `uv.lock`, and `.python-version`;
- dashboard source and tests;
- generated data and runtime products.

The five frozen CanFireCast dependencies, including redistribution decisions,
source experiment identifiers, byte sizes, and SHA-256 values, are recorded in
`vendor/PROVENANCE.json`. The imported FireMPC commit becomes authoritative
once this snapshot is committed.
