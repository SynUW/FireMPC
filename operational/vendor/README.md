# Vendored operational dependencies

Vendored files are copied **verbatim** and have SHA-256 entries in
`PROVENANCE.json`. The same manifest pins the top-level FireMPC FWI dependency.

## `canfirecast/`

The frozen CanFireCast inference helpers and fitted JSON correction parameters
needed to map IFS into the ERA5-Land training domain and apply the residual
CFFDRS-aware correction. Operational runs never refit these parameters.

## Top-level FWI dependency

Operational code loads `FireMPC/fwi/fwi_module.py` directly. It is not
maintained as a second copy here; its expected hash is pinned in
`PROVENANCE.json` and tested.

## What is *not* vendored

The legacy `newset_processing` stage scripts. In FireMPC integration those are
invoked from the top-level checkout under the configured `FIREMPC_PYTHON`
interpreter, so the `merged`, `unified` and `scaled` stages still need that
checkout present.
