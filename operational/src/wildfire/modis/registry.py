"""Canonical contract for GEE-export-compatible MODIS Collection 6.1 inputs.

All executable registries are derived from :data:`PRODUCTS`. Entries are at
the *project export* level because MOD09GA is intentionally exported twice at
different resolutions. :data:`DOWNLOAD_EXPORTS` is the one definition of what
the operational pipeline downloads; collection-level consumers derive their
plans by grouping those exports by Earthdata short name.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from wildfire.paths import register_native_resolutions

COLLECTION_VERSION = "061"
STANDARD_PROVIDER = "LPCLOUD"
OUTPUT_CRS = "EPSG:4326"
OUTPUT_TILES = [0, 1, 2, 3]
CANADA_LAND_TILES_30 = [
    "h08v03", "h08v04", "h09v03", "h09v04", "h10v02", "h10v03", "h10v04",
    "h11v02", "h11v03", "h11v04", "h12v02", "h12v03", "h12v04",
    "h13v01", "h13v02", "h13v03", "h13v04", "h14v01", "h14v02", "h14v03",
    "h14v04", "h15v00", "h15v01", "h15v02", "h15v03", "h16v00", "h16v01",
    "h16v02", "h17v00", "h17v01",
]
CANADA_LAND_TILES_29 = [tile for tile in CANADA_LAND_TILES_30 if tile != "h15v00"]

# The single operational allowlist. MOD14A1/MYD14A1 remain in PRODUCTS because
# parity and historical-label tooling need their contracts, but forecasting
# deliberately does not download fire labels.
DOWNLOAD_EXPORTS = (
    "MOD09GA_b1237", "MYD09GA_b1237",
    "MOD09GA_QA_state1km", "MYD09GA_QA_state1km",
    "MOD11A1", "MYD11A1", "MOD09CMG", "MYD09CMG",
    "MCD15A3H", "MCD12Q1",
    "MOD09A1", "MYD09A1", "MOD11A2", "MYD11A2",
    "MOD13Q1", "MYD13Q1",
)

# Forecasting uses only the sources which feed the 17-channel MODIS handoff.
# Keep this separate from DOWNLOAD_EXPORTS: the latter deliberately retains the
# larger historical/parity profile and is used by existing archive jobs.
OPERATIONAL_EXPORTS = (
    "MOD09GA_b1237", "MYD09GA_b1237",
    "MOD09GA_QA_state1km", "MYD09GA_QA_state1km",
    "MOD11A1", "MYD11A1", "MOD09CMG", "MYD09CMG",
    "MCD15A3H", "MCD12Q1",
)


def _bands(gee: list[str], hdf: list[str], qa: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    assert len(gee) == len(hdf)
    return [
        {"gee": g, "hdf": h, "qa": g in qa, "scale": 1.0, "offset": 0.0}
        for g, h in zip(gee, hdf, strict=True)
    ]


def _entry(
    short_name: str,
    *,
    gee_bands: list[str],
    hdf_bands: list[str],
    resolution: int,
    cadence_days: int,
    start: str,
    dtype: str,
    qa: tuple[str, ...] = (),
    offset: int = 0,
    global_source: bool = False,
    reference_subdir: str | None = None,
    composites: bool = False,
    subdivision: int | list[int] = 1,
    source_span_days: int = 1,
) -> dict[str, Any]:
    return {
        "gee_collection": f"MODIS/061/{short_name}",
        "earthaccess": {
            "short_name": short_name,
            "version": COLLECTION_VERSION,
            "provider": STANDARD_PROVIDER,
        },
        "bands": _bands(gee_bands, hdf_bands, qa),
        "output": {
            "crs": OUTPUT_CRS,
            "resolution_m": resolution,
            "resampling": "nearest",
            "dtype": dtype,
            "nodata": {
                "int16": -32768,
                "int32": -2147483648,
                "uint16": 0,
                "uint8": 0,
                "float64": None,
            }[dtype],
            "masked_value": 0,
            "mask_valid_range": True,
            "dtype_policy": "first_band",
        },
        "cadence": {"days": cadence_days, "offset": offset, "start": start},
        "source_layout": "global" if global_source else "modis_tiles",
        # Some daily GEE images are slices of a multi-day Earthdata granule.
        # MOD14A1/MYD14A1 package eight daily planes in one HDF whose A-date is
        # the first day; the requested output day selects the matching plane.
        "source_span_days": source_span_days,
        "expected_source_keys": ["global"] if global_source else None,
        # Number of subrequests per axis used by the historical GEE exporter.
        # GEE makes GeoJSON edges geodesic, so this affects the output envelope.
        "export_subdivision": subdivision,
        "reference_subdir": reference_subdir,
        "tiles": OUTPUT_TILES,
        "optional_composite": composites,
    }


_REFL_GEE = ["sur_refl_b01", "sur_refl_b02", "sur_refl_b03", "sur_refl_b07", "QC_500m"]
_REFL_HDF = ["sur_refl_b01_1", "sur_refl_b02_1", "sur_refl_b03_1", "sur_refl_b07_1", "QC_500m_1"]
_LST = ["LST_Day_1km", "LST_Night_1km", "Emis_31", "Emis_32", "QC_Day", "QC_Night"]
_CMG_GEE = [
    "Coarse_Resolution_Brightness_Temperature_Band_20",
    "Coarse_Resolution_Brightness_Temperature_Band_21",
    "Coarse_Resolution_Brightness_Temperature_Band_31",
    "Coarse_Resolution_Brightness_Temperature_Band_32",
    "Coarse_Resolution_QA", "Coarse_Resolution_State_QA", "Coarse_Resolution_Internal_CM",
]
_CMG_HDF = [name.replace("_", " ") for name in _CMG_GEE]
_VI_GEE = ["NDVI", "EVI", "sur_refl_b01", "sur_refl_b02", "sur_refl_b03", "sur_refl_b07", "DetailedQA", "SummaryQA"]
_VI_HDF = [
    "250m 16 days NDVI", "250m 16 days EVI", "250m 16 days red reflectance",
    "250m 16 days NIR reflectance", "250m 16 days blue reflectance",
    "250m 16 days MIR reflectance", "250m 16 days VI Quality",
    "250m 16 days pixel reliability",
]


PRODUCTS: dict[str, dict[str, Any]] = {
    "MOD09GA_b1237": _entry("MOD09GA", gee_bands=_REFL_GEE, hdf_bands=_REFL_HDF,
        resolution=500, cadence_days=1, start="2000-02-24", dtype="int16", qa=("QC_500m",), subdivision=[4, 3, 4, 3]),
    "MYD09GA_b1237": _entry("MYD09GA", gee_bands=_REFL_GEE, hdf_bands=_REFL_HDF,
        resolution=500, cadence_days=1, start="2002-07-04", dtype="int16", qa=("QC_500m",), subdivision=[4, 3, 4, 3]),
    "MOD09GA_QA_state1km": _entry("MOD09GA", gee_bands=["state_1km"], hdf_bands=["state_1km_1"],
        resolution=1000, cadence_days=1, start="2000-02-24", dtype="uint16", qa=("state_1km",)),
    "MYD09GA_QA_state1km": _entry("MYD09GA", gee_bands=["state_1km"], hdf_bands=["state_1km_1"],
        resolution=1000, cadence_days=1, start="2002-07-04", dtype="uint16", qa=("state_1km",)),
    "MOD11A1": _entry("MOD11A1", gee_bands=_LST, hdf_bands=_LST, resolution=1000,
        cadence_days=1, start="2000-03-05", dtype="uint16", qa=("QC_Day", "QC_Night")),
    "MYD11A1": _entry("MYD11A1", gee_bands=_LST, hdf_bands=_LST, resolution=1000,
        cadence_days=1, start="2002-07-04", dtype="uint16", qa=("QC_Day", "QC_Night")),
    "MOD09CMG": _entry("MOD09CMG", gee_bands=_CMG_GEE, hdf_bands=_CMG_HDF,
        resolution=5600, cadence_days=1, start="2000-02-24", dtype="float64",
        qa=tuple(_CMG_GEE[4:]), global_source=True),
    "MYD09CMG": _entry("MYD09CMG", gee_bands=_CMG_GEE, hdf_bands=_CMG_HDF,
        resolution=5600, cadence_days=1, start="2002-07-04", dtype="float64",
        qa=tuple(_CMG_GEE[4:]), global_source=True),
    "MOD14A1": _entry("MOD14A1", gee_bands=["FireMask", "MaxFRP", "QA"],
        hdf_bands=["FireMask", "MaxFRP", "QA"], resolution=1000, cadence_days=1,
        start="2000-02-24", dtype="uint8", qa=("FireMask", "QA"),
        subdivision=2, source_span_days=8),
    "MYD14A1": _entry("MYD14A1", gee_bands=["FireMask", "MaxFRP", "QA"],
        hdf_bands=["FireMask", "MaxFRP", "QA"], resolution=1000, cadence_days=1,
        start="2002-07-04", dtype="uint8", qa=("FireMask", "QA"),
        subdivision=2, source_span_days=8),
    "MCD15A3H": _entry("MCD15A3H", gee_bands=["Lai", "Fpar", "FparLai_QC", "FparExtra_QC"],
        hdf_bands=["Lai_500m", "Fpar_500m", "FparLai_QC", "FparExtra_QC"],
        resolution=500, cadence_days=4, start="2002-07-04", dtype="uint8",
        qa=("FparLai_QC", "FparExtra_QC"), subdivision=2),
    "MCD12Q1": _entry("MCD12Q1", gee_bands=["LC_Type1", "QC"], hdf_bands=["LC_Type1", "QC"],
        resolution=500, cadence_days=365, start="2000-01-01", dtype="uint8", qa=("QC",),
        reference_subdir="processed/MCD12Q1", subdivision=2),
    "MOD09A1": _entry("MOD09A1", gee_bands=["sur_refl_b01", "sur_refl_b02", "sur_refl_b03", "sur_refl_b07", "QA", "StateQA"],
        hdf_bands=["sur_refl_b01", "sur_refl_b02", "sur_refl_b03", "sur_refl_b07", "sur_refl_qc_500m", "sur_refl_state_500m"],
        resolution=500, cadence_days=8, start="2000-02-18", dtype="int16", qa=("QA", "StateQA"), composites=True, subdivision=[4, 3, 4, 3]),
    "MYD09A1": _entry("MYD09A1", gee_bands=["sur_refl_b01", "sur_refl_b02", "sur_refl_b03", "sur_refl_b07", "QA", "StateQA"],
        hdf_bands=["sur_refl_b01", "sur_refl_b02", "sur_refl_b03", "sur_refl_b07", "sur_refl_qc_500m", "sur_refl_state_500m"],
        resolution=500, cadence_days=8, start="2002-07-04", dtype="int16", qa=("QA", "StateQA"), composites=True, subdivision=[4, 3, 4, 3]),
    "MOD11A2": _entry("MOD11A2", gee_bands=_LST + ["Clear_sky_days"], hdf_bands=_LST + ["Clear_sky_days"],
        resolution=1000, cadence_days=8, start="2000-02-18", dtype="uint16",
        qa=("QC_Day", "QC_Night", "Clear_sky_days"), composites=True),
    "MYD11A2": _entry("MYD11A2", gee_bands=_LST + ["Clear_sky_days"], hdf_bands=_LST + ["Clear_sky_days"],
        resolution=1000, cadence_days=8, start="2002-07-04", dtype="uint16",
        qa=("QC_Day", "QC_Night", "Clear_sky_days"), composites=True),
    "MOD13Q1": _entry("MOD13Q1", gee_bands=_VI_GEE, hdf_bands=_VI_HDF, resolution=250,
        cadence_days=16, start="2000-01-01", dtype="int16", qa=("DetailedQA", "SummaryQA"), composites=True, subdivision=[6, 4, 6, 4]),
    "MYD13Q1": _entry("MYD13Q1", gee_bands=_VI_GEE, hdf_bands=_VI_HDF, resolution=250,
        cadence_days=16, offset=8, start="2002-07-04", dtype="int16", qa=("DetailedQA", "SummaryQA"), composites=True, subdivision=[6, 4, 6, 4]),
}

# GEE exposes physical values for these CMG bands.  This exception is pinned
# because Coarse Resolution QA's 0.0001 factor is absent from the HDF metadata.
for _name in ("MOD09CMG", "MYD09CMG"):
    PRODUCTS[_name]["output"]["mask_valid_range"] = False
    for _band in PRODUCTS[_name]["bands"][:4]:
        _band["scale"] = 0.01
    PRODUCTS[_name]["bands"][4]["scale"] = 0.0001

for _cfg in PRODUCTS.values():
    if _cfg["source_layout"] == "global":
        continue
    _short = _cfg["earthaccess"]["short_name"]
    _cfg["expected_source_keys"] = list(
        CANADA_LAND_TILES_29
        if _short.startswith(("MOD11", "MYD11")) or _short in {"MCD15A3H", "MCD12Q1"}
        else CANADA_LAND_TILES_30
    )


def validate_registry() -> None:
    """Fail fast when the frozen contract is internally inconsistent."""
    if len(PRODUCTS) != 18:
        raise RuntimeError(f"canonical registry has {len(PRODUCTS)} exports; expected 18")
    collections = {p["earthaccess"]["short_name"] for p in PRODUCTS.values()}
    if len(collections) != 16:
        raise RuntimeError(f"canonical registry has {len(collections)} collections; expected 16")
    unknown_downloads = set(DOWNLOAD_EXPORTS) - set(PRODUCTS)
    if unknown_downloads or len(DOWNLOAD_EXPORTS) != len(set(DOWNLOAD_EXPORTS)):
        raise RuntimeError(
            f"invalid DOWNLOAD_EXPORTS: unknown={sorted(unknown_downloads)}, "
            f"duplicates={len(DOWNLOAD_EXPORTS) - len(set(DOWNLOAD_EXPORTS))}")
    unknown_operational = set(OPERATIONAL_EXPORTS) - set(PRODUCTS)
    if unknown_operational or len(OPERATIONAL_EXPORTS) != 10:
        raise RuntimeError(
            "invalid OPERATIONAL_EXPORTS: "
            f"unknown={sorted(unknown_operational)}, count={len(OPERATIONAL_EXPORTS)}")
    for name, cfg in PRODUCTS.items():
        ea = cfg["earthaccess"]
        if ea["version"] != COLLECTION_VERSION or ea["provider"] != STANDARD_PROVIDER:
            raise RuntimeError(f"{name}: standard provenance is not pinned")
        if not cfg["bands"] or cfg["tiles"] != OUTPUT_TILES:
            raise RuntimeError(f"{name}: incomplete band/tile contract")
        cadence = cfg["cadence"]
        if cadence["days"] < 365 and not on_cadence(date.fromisoformat(cadence["start"]), cfg):
            raise RuntimeError(f"{name}: start date is off its cadence")


def on_cadence(day: date, cfg: dict[str, Any]) -> bool:
    cadence = cfg["cadence"]
    days, offset = cadence["days"], cadence["offset"]
    if days >= 365:
        return day.month == 1 and day.day == 1
    return (day.timetuple().tm_yday - 1 - offset) % days == 0


def converter_products() -> dict[str, dict[str, Any]]:
    """Legacy converter shape, derived losslessly from the canonical entries."""
    out = {}
    for name, cfg in PRODUCTS.items():
        row = {
            "short_name": cfg["earthaccess"]["short_name"],
            "res": cfg["output"]["resolution_m"],
            "sds": [b["hdf"] for b in cfg["bands"]],
            "mask_valid_range": cfg["output"]["mask_valid_range"],
            "export_subdivision": cfg["export_subdivision"],
            "source_span_days": cfg["source_span_days"],
        }
        scales = {b["hdf"]: b["scale"] for b in cfg["bands"] if b["scale"] != 1.0}
        if scales:
            row["band_scales"] = scales
        out[name] = row
    return out


def _download_lookback(cadence_days: int) -> int:
    """Operational retry window; source selection still lives in DOWNLOAD_EXPORTS."""
    if cadence_days >= 365:
        return 900
    return {4: 30, 8: 40, 16: 60}.get(cadence_days, 0)


# Display-only estimates for dry-run output. They do not select sources.
_APPROX_GRANULE_MB = {
    "MOD09GA": 90, "MYD09GA": 90,
    "MOD11A1": 5, "MYD11A1": 5,
    "MOD09CMG": 540, "MYD09CMG": 540,
    "MCD15A3H": 17, "MCD12Q1": 5,
    "MOD09A1": 60, "MYD09A1": 60,
    "MOD11A2": 6, "MYD11A2": 6,
    "MOD13Q1": 254, "MYD13Q1": 254,
}


def download_products() -> dict[str, dict[str, Any]]:
    """Collection-level downloader plans derived from DOWNLOAD_EXPORTS.

    Multiple requested exports may share one downloaded collection. In
    particular, each MOD09GA/MYD09GA HDF produces both the 500 m reflectance
    export and the 1 km state-QA export.
    """
    out: dict[str, dict[str, Any]] = {}
    for export_name in DOWNLOAD_EXPORTS:
        cfg = PRODUCTS[export_name]
        ea = cfg["earthaccess"]
        short = ea["short_name"]
        cadence = cfg["cadence"]
        nrt = ((short, "LANCEMODIS")
               if short in {"MOD09GA", "MYD09GA", "MOD09CMG", "MYD09CMG"}
               else None)
        candidate = {
            "std": (short, ea["version"]),
            "nrt": nrt,
            "cadence_days": cadence["days"],
            "cadence_offset": cadence["offset"],
            "lookback": _download_lookback(cadence["days"]),
            "global": cfg["source_layout"] == "global",
            "approx_mb": _APPROX_GRANULE_MB[short],
            "exports": [],
        }
        if short not in out:
            out[short] = candidate
        else:
            # Shared-collection exports must never disagree about search policy.
            current = {k: v for k, v in out[short].items() if k != "exports"}
            check = {k: v for k, v in candidate.items() if k != "exports"}
            if current != check:
                raise RuntimeError(f"{short}: requested exports disagree on download metadata")
        out[short]["exports"].append(export_name)
    return out


validate_registry()


# The shared filesystem contract records export resolutions for every grid this
# project touches, MODIS and otherwise.  Registering ours here -- rather than
# having `wildfire.paths` import this module -- keeps the shared module free of
# domain dependencies.
register_native_resolutions({
    name: cfg["output"]["resolution_m"] for name, cfg in PRODUCTS.items()
})
