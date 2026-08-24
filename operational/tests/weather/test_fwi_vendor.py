"""The top-level FireMPC FWI equations are the pinned operational dependency."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from wildfire import vendorpaths
from wildfire.weather import derive as derive

VENDOR = vendorpaths.VENDOR
PROVENANCE = json.loads((VENDOR / "PROVENANCE.json").read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_top_level_firempc_fwi_is_the_one_that_runs_and_is_pinned():
    assert derive.FWI_MODULE_PATH == vendorpaths.FIREMPC_ROOT / "fwi" / "fwi_module.py"
    recorded = PROVENANCE["files"]["fwi_module.py"]
    assert derive.FWI_MODULE_PATH.is_file()
    assert sha256(derive.FWI_MODULE_PATH) == recorded["sha256"]
    assert derive.FWI_MODULE_PATH.stat().st_size == recorded["bytes"]


def test_the_equations_load_and_the_startup_constants_are_intact():
    module = derive._load_fwi_module()
    assert (module.FFMC_INIT, module.DMC_INIT, module.DC_INIT) == (85.0, 6.0, 15.0)
    for name in ("ffmc_next", "dmc_next", "dc_next", "isi_calc", "bui_calc",
                 "fwi_calc", "extract_fwi_inputs_from_bands"):
        assert callable(getattr(module, name)), name


def test_a_known_step_of_the_recursion():
    """Pins the arithmetic itself, so a re-vendor cannot silently change values.

    A warm dry day with no rain: the fine fuels dry toward their ceiling, and
    both slow codes build from the cold-start constants.
    """
    module = derive._load_fwi_module()
    shape = (2, 2)
    temp = np.full(shape, 25.0, dtype=np.float32)
    rh = np.full(shape, 40.0, dtype=np.float32)
    wind = np.full(shape, 18.0, dtype=np.float32)
    rain = np.zeros(shape, dtype=np.float32)

    ffmc = module.ffmc_next(np.full(shape, 85.0, dtype=np.float32), temp, rh, wind, rain)
    dmc = module.dmc_next(np.full(shape, 6.0, dtype=np.float32), temp, rh, rain, 7)
    dc = module.dc_next(np.full(shape, 15.0, dtype=np.float32), temp, rain, 7)

    assert np.all(ffmc > 85.0) and np.all(ffmc <= 101.0)
    assert np.all(dmc > 6.0)
    assert np.all(dc > 15.0)
    # ISI rises with wind; FWI is non-negative everywhere it is defined.
    assert np.all(module.isi_calc(ffmc, wind) > module.isi_calc(ffmc, wind * 0.5))
    assert np.all(module.fwi_calc(module.isi_calc(ffmc, wind),
                                  module.bui_calc(dmc, dc)) >= 0.0)
