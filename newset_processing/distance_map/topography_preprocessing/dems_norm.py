#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Normalize DEM / Slope / Aspect / Hillshade / (TPI/TWI) / Distance rasters.

Two input modes (--input-mode):
1) files (default): find rasters separately by filename (matches the original logic)
   - DEM: ASTER_GDEM_elevation_1km.tif
   - Slope: *_slope_degree.tif
   - Aspect (deg): *_aspect.tif -> outputs cos/sin normalization
   - Hillshade: *_hillshade.tif
   - Distance: distance_*.tif

2) stack: multi-band GeoTIFF produced by dems_cal.py (5 or 7 bands)
   - Band 1 DEM, Band 2 Slope (deg), Band 3 Aspect_sin, Band 4 Aspect_cos, Band 5 Hillshade
   - Band 6 TPI, Band 7 TWI (if present)
   Each physical quantity is written as its own single-band *_<layer>_norm.tif, for compatibility with legacy pipelines.

Defaults (configurable from the CLI):
- DEM: minmax_p2p98 or robust_z
- TPI (default symmetric): TPI' = TPI / max(|TPI|), TPI'' = (TPI' + 1) / 2 -> [0, 1], 0 -> 0.5 (ridges -> 1, valleys -> 0)
- TWI (default clip_p1p99): clip by percentile (p1-p99 or p2-p98), then min-max to [0, 1]; optional log-then-clip (for skewed distributions)
- Slope: minmax_p2p98 (default), or cap / auto_p98
- Single-file Aspect (deg): convert to cos/sin, then map to [0, 1]
- Stack Aspect_sin/cos: already in [-1, 1], only apply (x + 1) / 2 -> [0, 1] (use --aspect-keep-neg1_1 to keep [-1, 1])
- Hillshade: stretch_p1p99 or div255
- Distance: log1p

Outputs:
- files mode: *_norm.tif, *_aspect_*_norm.tif, etc.
- stack mode: per-layer single files {stem}_*_norm.tif, plus a multi-band {stem}_norm_stack.tif (band descriptions reuse the input where possible; fall back to DEM / Slope_deg / ...)
- norm_params.json
"""

import argparse, json, math
from pathlib import Path
import numpy as np
import rasterio

# ------------- I/O helpers -------------

def read_band(path: Path, band=1):
    with rasterio.open(path) as src:
        arr = src.read(band).astype(np.float32, copy=False)
        profile = src.profile.copy()
        nodata = src.nodata  # may be None
        # treat nodata as NaN internally
        if nodata is not None:
            arr[np.isclose(arr, nodata, equal_nan=True)] = np.float32(np.nan)
        return arr, profile, nodata

def profile_single_band(base_profile: dict, nodata=None) -> dict:
    """Generate a single-band float32 profile from a multi-band profile."""
    p = base_profile.copy()
    p.update(dtype="float32", count=1)
    if nodata is not None:
        p["nodata"] = nodata
    return p


def write_tif(path: Path, arr: np.ndarray, profile: dict,
              nodata=None, compress="ZSTD", blocksize=512, overwrite=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and overwrite:
        path.unlink()
    prof = profile.copy()
    prof.update(dtype="float32", count=1, tiled=True,
                blockxsize=blocksize, blockysize=blocksize)
    # keep original nodata if provided; if nodata=None we simply store NaNs
    if nodata is not None:
        prof.update(nodata=nodata)
    if compress != "NONE":
        prof.update(compress=compress)
        if compress == "ZSTD":
            prof.update(zstd_level=9)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr, 1)


DEFAULT_STACK_BAND_NAMES = (
    "DEM",
    "Slope_deg",
    "Aspect_sin",
    "Aspect_cos",
    "Hillshade",
    "TPI",
    "TWI",
)


def write_multiband_tif(
    path: Path,
    arrays: list,
    base_profile: dict,
    band_descriptions: list,
    nodata,
    compress: str,
    blocksize: int,
    overwrite: bool = True,
):
    """Write several single-band float32 arrays into a multi-band GeoTIFF and set each band's description."""
    if len(arrays) != len(band_descriptions):
        raise ValueError("arrays and band_descriptions must have the same length")
    n = len(arrays)
    if n == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and overwrite:
        path.unlink()
    prof = base_profile.copy()
    prof.update(dtype="float32", count=n, tiled=True, blockxsize=blocksize, blockysize=blocksize)
    if nodata is not None:
        prof.update(nodata=nodata)
    if compress != "NONE":
        prof.update(compress=compress)
        if compress == "ZSTD":
            prof.update(zstd_level=9)
    with rasterio.open(path, "w", **prof) as dst:
        for i, arr in enumerate(arrays, start=1):
            dst.write(arr.astype(np.float32, copy=False), i)
            desc = band_descriptions[i - 1]
            if desc:
                dst.set_band_description(i, str(desc))


def nanpercentile(a, q):
    v = a[~np.isnan(a)]
    return float(np.percentile(v, q)) if v.size else np.nan

# ------------- Normalizers -------------

def _log1p_minmax(arr: np.ndarray, v: np.ndarray, shift: float, q_lo: float, q_hi: float):
    """
    log1p(x - shift) -> clip[p_lo, p_hi] -> [0, 1].
    shift is usually min(v) to ensure the input is always >= 0; pass 0 manually if v >= 0 already.
    """
    shifted = np.where(np.isfinite(arr), arr - shift, np.nan)
    shifted = np.where(shifted < 0, 0.0, shifted)
    log_arr = np.log1p(shifted)
    log_v = np.log1p(np.maximum(v - shift, 0.0))
    lo = float(np.percentile(log_v, q_lo))
    hi = float(np.percentile(log_v, q_hi))
    if hi <= lo:
        hi = lo + 1e-6
    out = (np.clip(log_arr, lo, hi) - lo) / (hi - lo)
    out[~np.isfinite(arr)] = np.nan
    return out, lo, hi


# DEM
def normalize_dem(arr, params, mode="log1p_p2p98"):
    # Use only finite values to estimate percentiles; inf pollutes percentile (e.g. p98 = inf collapses the entire image to a constant / NaN)
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        params.update({"type":"dem","mode":mode,"info":"empty"})
        return arr
    if mode == "log1p_p2p98":
        # Canada's terrain is right-skewed (lots of plains, few high mountains); log1p brings the distribution closer to normal
        shift = float(np.percentile(v, 1))   # Use P1 as shift to avoid extreme negative values
        out, lo, hi = _log1p_minmax(arr, v, shift, 2, 98)
        params.update({"type":"dem","mode":"log1p_p2p98","shift":shift,"log_p2":lo,"log_p98":hi})
        return out.astype(np.float32)
    if mode == "minmax_p2p98":
        p2, p98 = float(np.percentile(v, 2)), float(np.percentile(v, 98))
        if p98 <= p2: p98 = p2 + 1e-6
        out = (np.clip(arr, p2, p98) - p2) / (p98 - p2)
        out[~np.isfinite(arr)] = np.nan
        params.update({"type":"dem","mode":"minmax_p2p98","p2":p2,"p98":p98})
        return out
    elif mode == "robust_z":
        med = float(np.median(v))
        q25, q75 = float(np.percentile(v, 25)), float(np.percentile(v, 75))
        iqr = max(q75 - q25, 1e-6)
        out = (arr - med) / iqr
        out = 0.5 * (np.tanh(out) + 1.0)
        out[~np.isfinite(arr)] = np.nan
        params.update({"type":"dem","mode":"robust_z","median":med,"iqr":iqr,"nonlinear":"tanh_to_01"})
        return out
    else:
        raise ValueError(f"Unknown DEM mode: {mode}")

# Slope (degrees)
def normalize_slope_deg(arr, params, mode="log1p_p2p98", cap_deg=60.0):
    v = arr[np.isfinite(arr)]
    v = v[v >= 0]
    if v.size == 0:
        params.update({"type":"slope_degree","mode":mode,"info":"empty"})
        return arr
    if mode == "log1p_p2p98":
        # Slope is severely right-skewed (skew ~3.5); log1p significantly improves the distribution
        out, lo, hi = _log1p_minmax(arr, v, 0.0, 2, 98)
        params.update({"type":"slope_degree","mode":"log1p_p2p98","log_p2":lo,"log_p98":hi})
        return out.astype(np.float32)
    if mode == "cap":
        out = np.clip(arr, 0.0, float(cap_deg)) / float(cap_deg)
        params.update({"type":"slope_degree","mode":"cap","cap_deg":float(cap_deg)})
        return out
    elif mode == "auto_p98":
        cap = float(np.percentile(v, 98))
        cap = max(cap, 1e-6)
        out = np.clip(arr, 0.0, cap) / cap
        params.update({"type":"slope_degree","mode":"auto_p98","cap_p98":cap})
        return out
    elif mode == "minmax_p2p98":
        p2, p98 = float(np.percentile(v, 2)), float(np.percentile(v, 98))
        if p98 <= p2: p98 = p2 + 1e-6
        out = (np.clip(arr, p2, p98) - p2) / (p98 - p2)
        params.update({"type":"slope_degree","mode":"minmax_p2p98","p2":p2,"p98":p98})
        return out
    else:
        raise ValueError(f"Unknown slope mode: {mode}")

def normalize_aspect_trig(arr, params, map_to_01=True):
    """
    Input is already sin(theta) or cos(theta) in roughly [-1, 1] (Band 3/4 of the dems_cal stack).
    map_to_01=True linearly maps to [0, 1]; False keeps [-1, 1].
    """
    out = arr.astype(np.float32, copy=True)
    if map_to_01:
        out = (out + 1.0) * 0.5
        params.update({"type": "aspect_trig", "source": "precalc_sin_or_cos", "range": "[0,1]"})
    else:
        params.update({"type": "aspect_trig", "source": "precalc_sin_or_cos", "range": "[-1,1]"})
    return out


# Aspect (deg) -> cos/sin channels mapped to [0, 1]
def normalize_aspect(aspect_deg, params, map_to_01=True):
    theta = np.deg2rad(aspect_deg)
    cosv = np.cos(theta).astype(np.float32)
    sinv = np.sin(theta).astype(np.float32)
    if map_to_01:
        cosv = (cosv + 1.0) * 0.5
        sinv = (sinv + 1.0) * 0.5
        params.update({"type":"aspect","encode":"cos_sin","range":"[0,1]"})
    else:
        params.update({"type":"aspect","encode":"cos_sin","range":"[-1,1]"})
    return cosv, sinv

# Hillshade
def normalize_hillshade(hs, params, mode="div255"):
    """
    Hillshade normalization:
      - 'div255' | '/255' | '255'  -> divide by 255 to get [0, 1]
      - 'stretch_p1p99' | 'p1p99' | 'stretch' | 'auto' -> linearly stretch P1-P99 to [0, 1]
    Case-insensitive and tolerant of leading/trailing whitespace.
    """
    m = (mode or "div255").strip().lower()
    v = hs[~np.isnan(hs)]
    if v.size == 0:
        params.update({"type": "hillshade", "mode": m, "info": "empty"})
        return hs

    if m in ("div255", "/255", "255"):
        out = np.clip(hs, 0.0, 255.0) / 255.0
        params.update({"type": "hillshade", "mode": "div255"})
        return out

    if m in ("stretch_p1p99", "p1p99", "stretch", "auto"):
        p1, p99 = float(np.percentile(v, 1)), float(np.percentile(v, 99))
        if p99 <= p1:
            p99 = p1 + 1e-6
        out = (np.clip(hs, p1, p99) - p1) / (p99 - p1)
        out = np.clip(out, 0.0, 1.0)
        params.update({"type": "hillshade", "mode": "stretch_p1p99", "p1": p1, "p99": p99})
        return out

    # Reaching here means an unknown spelling
    raise ValueError(f"Unknown hillshade mode: {mode}")


# Distance
def normalize_distance(arr, params,
                       transform="log1p",
                       k_mode="median", k_value=None,
                       cap_quantile=0.98):
    v = arr[~np.isnan(arr)]
    if v.size == 0:
        params.update({"type":"distance","mode":transform,"info":"empty"})
        return arr

    # cap for far tail
    q = float(cap_quantile)
    q = min(max(q, 0.5), 0.999)  # sanity
    cap = float(np.percentile(v, q*100.0))

    # k scale
    if k_value is not None:
        k = float(k_value)
    elif k_mode.lower() == "median":
        k = float(np.median(v))
    elif k_mode.lower() == "p75":
        k = float(np.percentile(v, 75))
    else:
        raise ValueError("k_mode must be 'median' or 'p75' (or set k_value).")
    k = max(k, 1e-6)
    cap = max(cap, k)

    if transform == "log1p":
        out = np.log1p(np.clip(arr, 0, cap) / k) / math.log1p(cap / k)
        params.update({"type":"distance","mode":"log1p","k":k,"cap_q":q,"cap":cap})
    elif transform == "exp":
        # proximity-style: exp(-d/k) then re-range by cap if you want contrast
        prox = np.exp(-np.clip(arr, 0, cap) / k)
        # optional min-max to [0,1] is already satisfied (0..1], so just pass
        out = prox.astype(np.float32)
        params.update({"type":"distance","mode":"exp","k":k,"cap_q":q,"cap":cap})
    else:
        raise ValueError("distance transform must be 'log1p' or 'exp'.")

    out = np.clip(out, 0.0, 1.0)
    return out


def normalize_tpi(arr, params, mode="quantile_symmetric"):
    """
    TPI normalization:
      - quantile_symmetric (default): clip to a symmetric [P2, P98] range, then (x + 1) / 2 -> [0, 1].
        More robust against outlier peaks than symmetric (which uses extreme abs_max), and uses the [0, 1] range fully.
      - symmetric: TPI' = TPI / max(|TPI|), TPI'' = (TPI' + 1) / 2 in [0, 1] (legacy; sensitive to extremes).
      - minmax_p2p98 / robust_z: same as DEM (legacy-compatible).
    """
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        params.update({"type": "tpi", "mode": mode, "info": "empty"})
        return arr

    if mode == "quantile_symmetric":
        # Use max(|P2|, |P98|) as the symmetric range radius; more noise-resistant than abs_max
        p2, p98 = float(np.percentile(v, 2)), float(np.percentile(v, 98))
        half = max(abs(p2), abs(p98), 1e-6)
        clipped = np.clip(arr, -half, half)
        out = (clipped / half + 1.0) * 0.5
        out[~np.isfinite(arr)] = np.nan
        params.update({"type": "tpi", "mode": "quantile_symmetric", "half_range": half, "p2": p2, "p98": p98})
        return out.astype(np.float32)

    if mode == "symmetric":
        abs_max = float(np.max(np.abs(v)))
        if abs_max <= 0.0:
            abs_max = 1e-6
        tpi_prime = arr / abs_max
        out = (tpi_prime + 1.0) * 0.5
        out[~np.isfinite(arr)] = np.nan
        params.update({"type": "tpi", "mode": "symmetric", "abs_max": abs_max})
        return out.astype(np.float32)

    if mode in ("minmax_p2p98", "robust_z"):
        out = normalize_dem(arr, params, mode=mode)
        params["type"] = "tpi"
        return out

    raise ValueError(f"Unknown TPI mode: {mode}")


def normalize_twi(
    arr,
    params,
    mode="clip_p2p98",
    *,
    log_before_clip: bool = False,
    log_eps: float = 1e-6,
):
    """
    TWI normalization:
      - clip_p2p98 (default): clip to [P2, P98], then linearly rescale to [0, 1]. TWI is already a logarithmic quantity and not severely right-skewed,
        so clip + minmax suffices. Canada's flat terrain has high TWI (mean ~21); P2/P98 clipping is more stable than P1/P99.
      - clip_p1p99: same as above but looser.
      - minmax_p2p98 / robust_z: same as the legacy DEM-style normalization (compatibility).
    log_before_clip=True first applies log(max(TWI, log_eps)) to finite pixels, then clip + min-max.
    """
    base = arr.astype(np.float64, copy=False)
    if log_before_clip:
        work = np.where(
            np.isfinite(base) & (base > log_eps),
            np.log(np.maximum(base, log_eps)),
            np.nan,
        )
    else:
        work = np.array(base, copy=True)
        work[~np.isfinite(base)] = np.nan

    v = work[np.isfinite(work)]
    if v.size == 0:
        params.update(
            {
                "type": "twi",
                "mode": mode,
                "log_before_clip": log_before_clip,
                "info": "empty",
            }
        )
        return arr

    if mode in ("minmax_p2p98", "robust_z"):
        out = normalize_dem(arr, params, mode=mode)
        params["type"] = "twi"
        params["log_before_clip"] = log_before_clip
        return out

    if mode == "clip_p1p99":
        q_lo, q_hi = 1.0, 99.0
    elif mode == "clip_p2p98":
        q_lo, q_hi = 2.0, 98.0
    else:
        raise ValueError(f"Unknown TWI mode: {mode}")

    lo = float(np.percentile(v, q_lo))
    hi = float(np.percentile(v, q_hi))
    if hi <= lo:
        hi = lo + 1e-6
    clipped = np.clip(work, lo, hi)
    out = (clipped - lo) / (hi - lo)
    out[~np.isfinite(base)] = np.nan
    params.update(
        {
            "type": "twi",
            "mode": mode,
            "clip_quantiles": (q_lo, q_hi),
            "clip_lo": lo,
            "clip_hi": hi,
            "log_before_clip": log_before_clip,
        }
    )
    return out.astype(np.float32)


def read_stack_bands(path: Path, max_band: int = 7):
    """Open once and read bands 1..max_band (capped at src.count); return each band's description (may be None)."""
    with rasterio.open(path) as src:
        nb = min(src.count, max_band)
        nod = src.nodata
        prof = src.profile.copy()
        bands = []
        descriptions = []
        for b in range(1, nb + 1):
            arr = src.read(b).astype(np.float32, copy=False)
            if nod is not None:
                arr = arr.copy()
                arr[np.isclose(arr, nod, equal_nan=True)] = np.nan
            bands.append(arr)
            d = src.descriptions[b - 1] if src.descriptions else None
            descriptions.append(d if d else None)
        return bands, prof, nod, src.count, descriptions


def _band_label(src_desc, index: int) -> str:
    if src_desc:
        return str(src_desc)
    if index < len(DEFAULT_STACK_BAND_NAMES):
        return DEFAULT_STACK_BAND_NAMES[index]
    return f"Band{index + 1}"


def process_terrain_stack(
    path: Path,
    out_dir: Path,
    args,
    meta: dict,
    compress: str,
    blocksize: int,
):
    """Process dems_cal's multi-band output: per-layer single-band TIFs + a multi-band *_norm_stack.tif (preserving / filling in band names)."""
    bands, prof_full, nod, raw_count, src_desc = read_stack_bands(path, max_band=7)
    if raw_count < 5:
        print(f"[SKIP stack] {path.name}: at least 5 bands required; actual {raw_count}")
        return

    stem = path.stem
    sub = profile_single_band(prof_full, nodata=nod)
    entry = meta.setdefault(path.name, {"kind": "stack", "source_bands": raw_count, "layers": {}})
    combined_arrays = []
    combined_names = []

    b = bands[0]
    params = {}
    out = normalize_dem(b, params, mode=args.dem_mode)
    out[np.isnan(b)] = np.nan
    write_tif(out_dir / f"{stem}_dem_norm.tif", out, sub, nodata=nod, compress=compress, blocksize=blocksize)
    entry["layers"]["dem"] = params
    combined_arrays.append(out)
    combined_names.append(_band_label(src_desc[0], 0))
    print(f"[OK] stack {path.name} -> {stem}_dem_norm.tif")

    b = bands[1]
    params = {}
    out = normalize_slope_deg(b, params, mode=args.slope_mode, cap_deg=args.slope_cap)
    out[np.isnan(b)] = np.nan
    write_tif(out_dir / f"{stem}_slope_norm.tif", out, sub, nodata=nod, compress=compress, blocksize=blocksize)
    entry["layers"]["slope"] = params
    combined_arrays.append(out)
    combined_names.append(_band_label(src_desc[1], 1))
    print(f"[OK] stack {path.name} -> {stem}_slope_norm.tif")

    b = bands[2]
    params = {}
    out = normalize_aspect_trig(b, params, map_to_01=args.aspect_to_01)
    out[np.isnan(b)] = np.nan
    write_tif(out_dir / f"{stem}_aspect_sin_norm.tif", out, sub, nodata=nod, compress=compress, blocksize=blocksize)
    entry["layers"]["aspect_sin"] = params
    combined_arrays.append(out)
    combined_names.append(_band_label(src_desc[2], 2))
    print(f"[OK] stack {path.name} -> {stem}_aspect_sin_norm.tif")

    b = bands[3]
    params = {}
    out = normalize_aspect_trig(b, params, map_to_01=args.aspect_to_01)
    out[np.isnan(b)] = np.nan
    write_tif(out_dir / f"{stem}_aspect_cos_norm.tif", out, sub, nodata=nod, compress=compress, blocksize=blocksize)
    entry["layers"]["aspect_cos"] = params
    combined_arrays.append(out)
    combined_names.append(_band_label(src_desc[3], 3))
    print(f"[OK] stack {path.name} -> {stem}_aspect_cos_norm.tif")

    b = bands[4]
    params = {}
    out = normalize_hillshade(b, params, mode=args.hillshade_mode)
    out[np.isnan(b)] = np.nan
    write_tif(out_dir / f"{stem}_hillshade_norm.tif", out, sub, nodata=nod, compress=compress, blocksize=blocksize)
    entry["layers"]["hillshade"] = params
    combined_arrays.append(out)
    combined_names.append(_band_label(src_desc[4], 4))
    print(f"[OK] stack {path.name} -> {stem}_hillshade_norm.tif")

    if len(bands) >= 6:
        b = bands[5]
        params = {}
        out = normalize_tpi(b, params, mode=args.tpi_mode)
        out[np.isnan(b)] = np.nan
        write_tif(out_dir / f"{stem}_tpi_norm.tif", out, sub, nodata=nod, compress=compress, blocksize=blocksize)
        entry["layers"]["tpi"] = params
        combined_arrays.append(out)
        combined_names.append(_band_label(src_desc[5], 5))
        print(f"[OK] stack {path.name} -> {stem}_tpi_norm.tif")
    else:
        entry["layers"]["tpi"] = {"info": "missing_band6"}

    if len(bands) >= 7:
        b = bands[6]
        params = {}
        out = normalize_twi(
            b,
            params,
            mode=args.twi_mode,
            log_before_clip=getattr(args, "twi_norm_log", False),
        )
        out[np.isnan(b)] = np.nan
        write_tif(out_dir / f"{stem}_twi_norm.tif", out, sub, nodata=nod, compress=compress, blocksize=blocksize)
        entry["layers"]["twi"] = params
        combined_arrays.append(out)
        combined_names.append(_band_label(src_desc[6], 6))
        print(f"[OK] stack {path.name} -> {stem}_twi_norm.tif")
    else:
        entry["layers"]["twi"] = {"info": "missing_band7"}

    if not args.no_stack_combined and combined_arrays:
        stack_path = out_dir / f"{stem}_norm_stack.tif"
        write_multiband_tif(
            stack_path,
            combined_arrays,
            prof_full,
            combined_names,
            nod,
            compress,
            blocksize,
        )
        entry["combined_stack"] = {
            "path": stack_path.name,
            "band_count": len(combined_arrays),
            "band_descriptions": combined_names,
        }
        print(f"[OK] stack {path.name} -> {stack_path.name} ({len(combined_arrays)} bands)")


# ------------- Main -------------

def main():
    ap = argparse.ArgumentParser("Normalize DEM/Slope/Aspect/Hillshade/TPI/TWI/Distance rasters")
    ap.add_argument("--input-folder", default='/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/merit_dem/dem_slope_aspect_hillshade', help="Input directory")
    ap.add_argument("--output-folder", default='/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/merit_dem/dem_slope_aspect_hillshade', help="Output directory")
    ap.add_argument(
        "--input-mode",
        choices=["files", "stack"],
        default="files",
        help="files = process separately by filename pattern; stack = dems_cal multi-band stack (>=5 bands): single-band *_<layer>_norm.tif + {stem}_norm_stack.tif (band descriptions reuse the input)",
    )
    ap.add_argument(
        "--stack-pattern",
        default="*.tif",
        help="Glob to match when input-mode=stack (relative to input-folder)",
    )
    ap.add_argument(
        "--no-stack-combined",
        action="store_true",
        help="Stack mode: do not write the combined multi-band file (default writes {stem}_norm_stack.tif and preserves band names)",
    )
    ap.add_argument("--band", type=int, default=1)

    # DEM
    ap.add_argument("--dem-mode", choices=["minmax_p2p98","robust_z"], default="minmax_p2p98")

    # Slope
    ap.add_argument("--slope-mode", choices=["cap","auto_p98","minmax_p2p98"], default="minmax_p2p98")
    ap.add_argument("--slope-cap", type=float, default=10.0)

    # Aspect
    ap.add_argument("--aspect-to-01", action="store_true", default=True,
                    help="map cos/sin from [-1,1] to [0,1] (default True)")
    ap.add_argument("--aspect-keep-neg1_1", dest="aspect_to_01", action="store_false")

    # Hillshade
    ap.add_argument("--hillshade-mode", choices=["div255","stretch_p1p99"], default="stretch_p1p99")

    # TPI / TWI (Band 6/7 in stack mode)
    ap.add_argument(
        "--tpi-mode",
        choices=["symmetric", "minmax_p2p98", "robust_z"],
        default="symmetric",
        help="TPI: symmetric = TPI / max(|TPI|) then (x+1)/2 -> [0, 1]; minmax/robust same as DEM",
    )
    ap.add_argument(
        "--twi-mode",
        choices=["clip_p1p99", "clip_p2p98", "minmax_p2p98", "robust_z"],
        default="clip_p1p99",
        help="TWI: percentile-clip then min-max; clip_p2p98 is more conservative; minmax/robust are legacy",
    )
    ap.add_argument(
        "--twi-norm-log",
        action="store_true",
        help="Apply log(max(TWI, eps)) before percentile clipping for TWI (only when values are positive and still skewed)",
    )

    # Distance
    ap.add_argument("--distance-transform", choices=["log1p","exp"], default="log1p")
    ap.add_argument("--distance-k-mode", choices=["median","p75"], default="median")
    ap.add_argument("--distance-k-value", type=float, default=None,
                    help="override k with a constant (same units as distance)")
    ap.add_argument("--distance-cap-quantile", type=float, default=0.98,
                    help="cap tail at this quantile before scaling (default 0.98)")

    # Write
    ap.add_argument("--compress", default="ZSTD", choices=["NONE","LZW","DEFLATE","ZSTD"])
    ap.add_argument("--blocksize", type=int, default=512)
    ap.add_argument("--save-meta", default="norm_params.json")

    args = ap.parse_args()
    in_dir = Path(args.input_folder)
    out_dir = Path(args.output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {}

    if args.input_mode == "stack":
        stack_files = sorted(in_dir.glob(args.stack_pattern))
        if not stack_files:
            print(f"[WARN] no match found in stack mode: {in_dir}/{args.stack_pattern}")
        for f in stack_files:
            if f.is_file():
                process_terrain_stack(
                    f, out_dir, args, meta, args.compress, args.blocksize
                )
        (out_dir / args.save_meta).write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(f"[DONE] Params saved to {out_dir / args.save_meta}")
        print("[ALL DONE]")
        return

    # ---------- files mode (original logic) ----------
    # gather files (case-insensitive by pattern)
    f_dem      = list(in_dir.glob("ASTER_GDEM_elevation_1km.tif"))
    f_slope    = list(in_dir.glob("*_slope_degree.tif"))
    f_aspect   = list(in_dir.glob("*_aspect.tif"))
    f_hill     = list(in_dir.glob("*_hillshade.tif"))
    f_dist     = sorted(list(in_dir.glob("distance_*.tif")))

    # DEM
    for f in f_dem:
        arr, prof, nod = read_band(f, args.band)
        params = {}
        out = normalize_dem(arr, params, mode=args.dem_mode)
        out[np.isnan(arr)] = np.nan
        write_tif(out_dir/(f.stem+"_norm.tif"), out, prof, nodata=nod,
                  compress=args.compress, blocksize=args.blocksize)
        meta[f.name] = params
        print(f"[OK] DEM -> {f.stem}_norm.tif")

    # Slope
    for f in f_slope:
        arr, prof, nod = read_band(f, args.band)
        params = {}
        out = normalize_slope_deg(arr, params, mode=args.slope_mode, cap_deg=args.slope_cap)
        out[np.isnan(arr)] = np.nan
        write_tif(out_dir/(f.stem+"_norm.tif"), out, prof, nodata=nod,
                  compress=args.compress, blocksize=args.blocksize)
        meta[f.name] = params
        print(f"[OK] Slope -> {f.stem}_norm.tif")

    # Aspect
    for f in f_aspect:
        arr, prof, nod = read_band(f, args.band)
        params = {}
        cosv, sinv = normalize_aspect(arr, params, map_to_01=args.aspect_to_01)
        cosv[np.isnan(arr)] = np.nan
        sinv[np.isnan(arr)] = np.nan
        write_tif(out_dir/(f.stem+"_cos_norm.tif"), cosv, prof, nodata=nod,
                  compress=args.compress, blocksize=args.blocksize)
        write_tif(out_dir/(f.stem+"_sin_norm.tif"), sinv, prof, nodata=nod,
                  compress=args.compress, blocksize=args.blocksize)
        meta[f.name] = params
        print(f"[OK] Aspect -> {f.stem}_cos_norm.tif / _sin_norm.tif")

    # Hillshade
    for f in f_hill:
        arr, prof, nod = read_band(f, args.band)
        params = {}
        out = normalize_hillshade(arr, params, mode=args.hillshade_mode)
        out[np.isnan(arr)] = np.nan
        write_tif(out_dir/(f.stem+"_norm.tif"), out, prof, nodata=nod,
                  compress=args.compress, blocksize=args.blocksize)
        meta[f.name] = params
        print(f"[OK] Hillshade -> {f.stem}_norm.tif")

    # Distance (multiple)
    for f in f_dist:
        arr, prof, nod = read_band(f, args.band)
        params = {}
        out = normalize_distance(arr, params,
                                 transform=args.distance_transform,
                                 k_mode=args.distance_k_mode,
                                 k_value=args.distance_k_value,
                                 cap_quantile=args.distance_cap_quantile)
        out[np.isnan(arr)] = np.nan
        write_tif(out_dir/(f.stem+"_norm.tif"), out, prof, nodata=nod,
                  compress=args.compress, blocksize=args.blocksize)
        meta[f.name] = params
        print(f"[OK] Distance -> {f.stem}_norm.tif")

    # Save parameter log
    (out_dir/args.save_meta).write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[DONE] Params saved to {out_dir/args.save_meta}")
    print("[ALL DONE]")
    
if __name__ == "__main__":
    main()
