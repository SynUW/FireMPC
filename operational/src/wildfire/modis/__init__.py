"""MODIS ingest: earthaccess granules through the frozen 17-channel contract.

Importing this package registers the MODIS export resolutions with
`wildfire.paths`; see `wildfire.paths.register_native_resolutions`.
"""

from wildfire.modis import registry as registry  # noqa: F401  -- registration side effect
