"""Operational near-real-time wildfire model inputs for Canada.

Three domain subpackages sit under this one:

* `wildfire.modis`       -- earthaccess granule ingest through the frozen
                            17-channel processing contract to a daily package.
* `wildfire.weather`     -- IFS/ERA5 download, VPD and the FWI recursion.
* `wildfire.canfirecast` -- assembly of the 55-channel model input and the
                            certified state ledger.

`wildfire.paths` holds the filesystem contract shared by all three;
`wildfire.vendorpaths` resolves the repository-relative trees (vendor/,
training_code/, config/) that the domains read.
"""
