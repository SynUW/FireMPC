"""Test-session fixes for the ambient geospatial environment.

The login shell on this machine exports ``PROJ_DATA=/usr/share/proj``, a PROJ 6
database that the PROJ shipped inside rasterio's wheel refuses to read: any
``CRS.from_epsg`` raises ``DATABASE.LAYOUT.VERSION.MINOR = 2 whereas a number
>= 6 is expected``.  Reading an existing raster survives it, because the WKT is
embedded in the file, so this only bites tests that build a raster from an EPSG
code.  Point the process at the database that ships with the wheel it is using.
"""

from __future__ import annotations

import os
from pathlib import Path

import rasterio

_BUNDLED = Path(rasterio.__file__).parent / "proj_data"
if (_BUNDLED / "proj.db").is_file():
    os.environ["PROJ_DATA"] = str(_BUNDLED)
    os.environ["PROJ_LIB"] = str(_BUNDLED)
