"""Operational near-real-time wildfire model inputs for Canada.

This directory is a plain FireMPC stage folder, like `sampling/` and
`newset_processing/`: nothing is installed, and the modules import each other
by bare name off `operational/` rather than through a package prefix.  Scripts
run below a domain directory add that root to `sys.path` themselves; `cli.py`
and `fill.py` sit at the top, where the interpreter has already done it.

Three domain directories sit under this one:

* `modis`       -- earthaccess granule ingest through the frozen 17-channel
                   processing contract to a daily package.
* `weather`     -- IFS/ERA5 download, VPD and the FWI recursion.
* `canfirecast` -- assembly of the 55-channel model input and the certified
                   state ledger.

`fill.py` is the application: it drives all three over a range of dates and
lands the result in the monthly store.  Three modules at this level are shared
by every domain: `paths` holds the filesystem contract, `vendorpaths` resolves
the repository-relative trees (vendor/, config/) the domains read, and `ops`
holds the plumbing every stage needs -- hashing, durable manifest writes, the
subprocess runner, and the `--from-stage`/`--to-stage` slice.  A stage is
re-invoked as a subprocess through `ops.cli`, which is `vendorpaths.CLI`.
"""
