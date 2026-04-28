"""MODIS MCD14A1 firemask reading and confidence filtering utilities.

Firemask band values (MODIS Collection 6.1 Active Fire Product User Guide):
  0   = not processed (missing input data)
  1   = not processed (obsolete; not used)
  2   = not processed (other reason)
  3   = non-fire water pixel
  4   = cloud (EXCLUDE from all negative candidates)
  5   = non-fire land pixel (valid negative candidate)
  6   = unknown (exclude)
  7   = fire pixel, low confidence  (EXCLUDE from positives; include in exclusion buffer)
  8   = fire pixel, nominal confidence (EXCLUDE from positives; include in exclusion buffer)
  9   = fire pixel, high confidence (POSITIVE LABEL -- training targets)

Ternary label decision (documented in decisions/ternary_label_decision.md):
  - firemask == 9  -> positive training label
  - firemask == 8  -> exclusion buffer (not a negative, not a positive)
  - firemask == 7  -> exclusion buffer (not a negative, not a positive)
  - firemask == 4  -> cloud (exclude from negative candidates)
  - firemask == 5  -> valid negative candidate (after Layer 1 reliability buffer applied)

Reference: MODIS Collection 6.1 Active Fire Products User's Guide, Giglio et al. (2020)
"""
import numpy as np
from osgeo import gdal

# Firemask value constants
FIREMASK_NOT_PROCESSED = 0
FIREMASK_WATER         = 3
FIREMASK_CLOUD         = 4
FIREMASK_NONFIRE_LAND  = 5
FIREMASK_LOW_CONF      = 7   # Fire, low confidence -> exclusion buffer only
FIREMASK_NOM_CONF      = 8   # Fire, nominal confidence -> exclusion buffer only
FIREMASK_HIGH_CONF     = 9   # Fire, high confidence -> positive label


def read_modis_tile(tile_path: str) -> tuple:
    """Read MCD14A1_combined GeoTIFF tile.

    Args:
        tile_path: path to daily MCD14A1_combined tile.

    Returns:
        (firemask, maxfrp, qa) as 2D numpy arrays.
        firemask: uint8 (values 0-9)
        maxfrp:   float32 (maximum fire radiative power)
        qa:       uint8 (quality assessment flags)

    Note: GDAL band indices are 1-based. Band 1 = firemask, Band 2 = maxfrp, Band 3 = qa.
    """
    ds = gdal.Open(tile_path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open MODIS tile: {tile_path}")
    firemask = ds.GetRasterBand(1).ReadAsArray()   # uint8, values 0-9
    maxfrp   = ds.GetRasterBand(2).ReadAsArray().astype(np.float32)
    qa       = ds.GetRasterBand(3).ReadAsArray()
    ds = None
    return firemask, maxfrp, qa


def get_positive_mask(firemask: np.ndarray) -> np.ndarray:
    """Return boolean mask of high-confidence fire pixels (training positive labels).

    Only firemask == 9 (high confidence) pixels are positive labels.
    Medium-confidence (8) and low-confidence (7) are excluded per ternary label decision.

    Args:
        firemask: 2D uint8 array of firemask values.

    Returns:
        bool array, True where firemask == FIREMASK_HIGH_CONF (9).
    """
    return firemask == FIREMASK_HIGH_CONF


def get_exclusion_mask(firemask: np.ndarray) -> np.ndarray:
    """Return boolean mask of pixels to exclude from negative candidates.

    Excludes: all fire pixels (confidence 7/8/9) and cloud pixels (4).
    These pixels must NOT be selected as negative training samples.

    Args:
        firemask: 2D uint8 array of firemask values.

    Returns:
        bool array, True where pixel should be excluded from negative pool.
    """
    return (
        (firemask == FIREMASK_HIGH_CONF) |
        (firemask == FIREMASK_NOM_CONF)  |
        (firemask == FIREMASK_LOW_CONF)  |
        (firemask == FIREMASK_CLOUD)
    )


def get_cloud_mask(firemask: np.ndarray) -> np.ndarray:
    """Return boolean mask of cloud pixels only.

    Args:
        firemask: 2D uint8 array.

    Returns:
        bool array, True where firemask == FIREMASK_CLOUD (4).
    """
    return firemask == FIREMASK_CLOUD


def get_valid_negative_candidates(firemask: np.ndarray) -> np.ndarray:
    """Return boolean mask of pixels that are valid negative candidates.

    Valid negative candidates: non-fire, non-cloud land pixels (firemask == 5).
    The Layer 1 reliability buffer (Plan 02 of Phase 2) further filters these
    based on proximity to fire events.

    Args:
        firemask: 2D uint8 array.

    Returns:
        bool array, True where firemask == FIREMASK_NONFIRE_LAND (5).
    """
    return firemask == FIREMASK_NONFIRE_LAND
