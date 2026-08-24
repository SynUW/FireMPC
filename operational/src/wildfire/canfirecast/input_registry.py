"""Authoritative CanFireCast 55-channel model-input contract.

This module centralizes the retrained ``CanFireCastIFSCommon`` input layout so
weather correction and package assembly do not carry independent channel
tuples. The native data tensor is always 55 channels; the compact 49-channel
projection is only an interchange layout and must be expanded in the published
package.
"""

from __future__ import annotations

from wildfire import vendorpaths
import hashlib
import ast
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

REGISTRY_VERSION = "canfirecast-ifs-common-55-v1"
PACKAGE_DTYPE = "uint8"
# In CanFireCast corrected meteorology caches, 0..255 are valid encoded values.
# Spatial validity is a separate mask/provenance decision, not implicit nodata.
ENCODED_NODATA = None
VALID_ENCODED_RANGE = (0, 255)
WINDOW_DAYS = 10
WINDOW_OFFSETS = tuple(range(-10, 0))
STATE_OFFSET = -11
WEATHER_ALPHA = 0.625
ENABLE_STATE_CORRECTION = False

FROZEN_EXPERIMENT_ID = "timesformer_ifs_cffdrs_aware_20260729"
VENDORED_ROOT = vendorpaths.CANFIRECAST_VENDOR

ARTIFACTS = {
    "ifs_to_era5_physical_adapter.json": {
        "path": VENDORED_ROOT / "artifacts" / "ifs_to_era5_physical_adapter.json",
        "sha256": "9a388b6b7c8efd6af21977750803519048ec7546a206a0ab63a6a53aa6962efe",
        "source": "timesformer_ifs_to_era5_physical_qm_20260729/artifacts/ifs_to_era5_physical_adapter.json",
    },
    "cffdrs_aware_correction.json": {
        "path": VENDORED_ROOT / "artifacts" / "cffdrs_aware_correction.json",
        "sha256": "dd11aa6bae71e08dcd449c9c56d2179c4832e54689328c3c66d3af27e0a1b128",
        "source": "timesformer_ifs_cffdrs_aware_20260729/artifacts/cffdrs_aware_correction.json",
    },
}

VENDORED_CODE = {
    "full10_ifs_domain_transform.py": "3b8b2c56cdd7a669e845fbfc8840539f8036c596b1a1f822f5375702a74e48b2",
    "ifs_to_era5_physical_adapter.py": "de908bb48944e2316d69c26ad7c4c2e1d168a2fedcfcf5e5049b6112c342fc7e",
    "cffdrs_aware_correction.py": "0a7d764a6ed7300452459c6b11fa84b6f8f5dcaead1f4d6383270feb7955a793",
}

CHANNEL_NAMES: tuple[str, ...] = (
    "Fire Detection",
    "Temperature 2m", "U Component Of Wind 10m", "V Component Of Wind 10m",
    "Snow Cover", "Total Precipitation Sum", "Surface Latent Heat Flux Sum",
    "Dewpoint Temperature 2m", "Surface Pressure",
    "Volumetric Soil Water Layer 1", "Volumetric Soil Water Layer 2",
    "Volumetric Soil Water Layer 3", "Volumetric Soil Water Layer 4",
    "Temperature 2m Max", "Skin Temperature Max",
    "Potential Evaporation Sum", "Total Evaporation Sum",
    "Skin Reservoir Content", "Surface Net Solar Radiation Sum",
    "FFMC", "DMC", "DC", "ISI", "BUI", "FWI", "VPD",
    "Road Density", "Powerline Density", "Buildings Density",
    "Water Density", "Population",
    "DEM", "Slope", "Aspect Sine", "Aspect Cosine", "Hillshade", "TPI", "TWI",
    "NDVI", "EVI", "FPAR", "LAI",
    "Band 1", "Band 2", "Band 3", "Band 7",
    "LST Day", "LST Night", "Emis 31", "Emis 32",
    "Brightness Temperature Band 20", "Brightness Temperature Band 21",
    "Brightness Temperature Band 31", "Brightness Temperature Band 32",
    "Land Cover Types",
)
CHANNEL_INDEX = {name: index for index, name in enumerate(CHANNEL_NAMES)}

FIRE_DETECTION_CHANNEL = 0
DIRECT_WEATHER_CHANNELS = (1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 18)
ERA5_ONLY_MISSING_CHANNELS = (4, 6, 15, 16, 17)
DERIVED_WEATHER_CHANNELS = (19, 20, 21, 22, 23, 24, 25)
CFFDRS_CHANNELS = (19, 20, 21, 22, 23, 24)
VPD_CHANNEL = 25
WEATHER_CHANNELS = DIRECT_WEATHER_CHANNELS + DERIVED_WEATHER_CHANNELS
STATIC_CHANNELS = tuple(range(26, 38))
MODIS_CHANNELS = tuple(range(38, 55))
MODEL_MODIS_ORDER = ("NDVI", "EVI", "FPAR", "LAI")

COMPACT49_OMITTED_NATIVE = (0, 4, 6, 15, 16, 17)
COMPACT49_TO_NATIVE55 = tuple(i for i in range(55) if i not in COMPACT49_OMITTED_NATIVE)
NATIVE55_TO_COMPACT49 = {native: compact for compact, native in enumerate(COMPACT49_TO_NATIVE55)}

# Exact physical decode/encode ranges from frozen full10_ifs_domain_transform.py.
PHYSICAL_RANGES: dict[int, tuple[float, float]] = {
    1: (0.005193258648563279, 297.7777831997186),
    2: (-8.035555467392681, 9.75282316575327),
    3: (-8.884559637656066, 7.783394794594883),
    5: (1.9188684693810958e-7, 0.03603348170585861),
    7: (0.005005088835593607, 291.85923360787075),
    8: (1.7814644058506481, 102594.00284017395),
    9: (1.2825456326856735e-5, 0.7495786155920634),
    10: (1.2824859849846285e-5, 0.7429968109043169),
    11: (1.2793656195784068e-5, 0.7264044759278405),
    12: (1.2811615921890935e-5, 0.7151072856451576),
    13: (0.005317793384206413, 304.2753351358254),
    14: (0.005381247822265205, 305.6516323146297),
    18: (413.45007501667135, 28043026.67241847),
    19: (0.0, 101.0),
    20: (0.0, 119.0),
    21: (0.0, 694.0),
    22: (0.0, 11.0),
    23: (0.0, 149.0),
    24: (0.0, 33.0),
    25: (0.0, 1.63),
}

@dataclass(frozen=True, slots=True)
class Channel:
    index: int
    name: str
    group: str

CHANNELS = tuple(
    Channel(
        index,
        name,
        "fire" if index == FIRE_DETECTION_CHANNEL else
        "direct_weather" if index in DIRECT_WEATHER_CHANNELS else
        "era5_only_missing" if index in ERA5_ONLY_MISSING_CHANNELS else
        "derived_weather" if index in DERIVED_WEATHER_CHANNELS else
        "static" if index in STATIC_CHANNELS else
        "modis",
    )
    for index, name in enumerate(CHANNEL_NAMES)
)


def target_window_dates(target: date) -> tuple[date, ...]:
    """Return D-10 through D-1, oldest first."""
    return tuple(target + timedelta(days=offset) for offset in WINDOW_OFFSETS)


def state_date(target: date) -> date:
    """Return the antecedent FFMC/DMC/DC state date, D-11."""
    return target + timedelta(days=STATE_OFFSET)


def expand_compact49(compact, *, fill_value: int = 0):
    """Expand an array with channel axis 49 into native 55-channel order."""
    import numpy as np

    arr = np.asarray(compact)
    if arr.shape[-3] == 49:  # (..., C, H, W)
        out_shape = arr.shape[:-3] + (55,) + arr.shape[-2:]
        out = np.full(out_shape, fill_value, dtype=arr.dtype)
        out[..., COMPACT49_TO_NATIVE55, :, :] = arr
        return out
    if arr.shape[-1] == 49:  # (..., H, W, C)
        out = np.full(arr.shape[:-1] + (55,), fill_value, dtype=arr.dtype)
        out[..., COMPACT49_TO_NATIVE55] = arr
        return out
    raise ValueError(f"cannot find a 49-channel axis in shape {arr.shape}")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance_files() -> dict:
    manifest = vendorpaths.VENDOR / "PROVENANCE.json"
    return json.loads(manifest.read_text(encoding="utf-8"))["files"]


def expected_vendor_hash(relative_path: str) -> str:
    try:
        return provenance_files()[relative_path]["sha256"]
    except KeyError as exc:
        raise RuntimeError(f"{relative_path} is missing from vendor provenance manifest") from exc


def verify_file_hash(path: str | Path, expected_sha256: str, *, label: str | None = None) -> None:
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"missing {label or 'file'}: {path}")
    actual = sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(f"hash mismatch for {label or path.name}: {actual} != {expected_sha256}")


def verify_vendored_code() -> None:
    for name, expected in VENDORED_CODE.items():
        rel = f"canfirecast/frozen_repo/{name}"
        manifest_expected = expected_vendor_hash(rel)
        if manifest_expected != expected:
            raise RuntimeError(f"registry/provenance hash disagreement for {rel}")
        verify_file_hash(VENDORED_ROOT / "frozen_repo" / name, expected, label=name)


def verify_pinned_artifacts() -> None:
    for name, meta in ARTIFACTS.items():
        rel = f"canfirecast/artifacts/{name}"
        expected = expected_vendor_hash(rel)
        if expected != meta["sha256"]:
            raise RuntimeError(f"registry/provenance hash disagreement for {rel}")
        verify_file_hash(meta["path"], expected, label=name)
    verify_vendored_code()


def _load_frozen_channel_names() -> tuple[str, ...] | None:
    path = VENDORED_ROOT / "frozen_repo" / "dataload_h5.py"
    if not path.is_file():
        return None
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "CHANNEL_NAMES":
            return tuple(ast.literal_eval(node.value))
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "CHANNEL_NAMES" for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    return None


def validate_registry(*, check_frozen: bool = True) -> None:
    if len(CHANNEL_NAMES) != 55:
        raise RuntimeError(f"expected 55 channels, found {len(CHANNEL_NAMES)}")
    if tuple(range(55)) != tuple(channel.index for channel in CHANNELS):
        raise RuntimeError("channel indices must be exactly 0..54")
    if len(CHANNEL_INDEX) != 55:
        raise RuntimeError("channel names must be unique")
    if len(COMPACT49_TO_NATIVE55) != 49 or set(COMPACT49_TO_NATIVE55) & set(COMPACT49_OMITTED_NATIVE):
        raise RuntimeError("invalid compact49 mapping")
    if set(PHYSICAL_RANGES) != set(range(1, 26)) - set(ERA5_ONLY_MISSING_CHANNELS):
        raise RuntimeError("physical ranges must cover channels 1-25 except ERA5-only missing slots")
    if tuple(CHANNEL_NAMES[i] for i in range(38, 42)) != MODEL_MODIS_ORDER:
        raise RuntimeError("MODIS model order must be NDVI, EVI, FPAR, LAI")
    if check_frozen:
        frozen = _load_frozen_channel_names()
        if frozen is not None and frozen != CHANNEL_NAMES:
            raise RuntimeError("registry channel names differ from frozen dataload_h5.CHANNEL_NAMES")


validate_registry(check_frozen=True)
