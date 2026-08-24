#!/usr/bin/env python3
"""
earthaccess MODIS HDF -> GEE-format per-tile GeoTIFF.

This is the one component the operational pipeline was missing. Everything
downstream of it is FireMPC/newset_processing, unmodified -- literally the code
that produced the training cube.

WHY THIS IS SMALL
-----------------
The GEE download this replaces is a dumb pipe. Reading
gee_download_fast_copy_parallel.py:786-811, the entire server-side chain is:

    filterDate([D, D+1) timezone.utc) -> filterBounds -> select(bands)
      -> mosaic() -> select(band) -> reproject("EPSG:4326", scale) -> GeoTIFF

No QA masking, no scale factors, no offsets, no clamping, no NDVI/EVI, no
compositing. Raw DN straight out of the granule. So parity is a reprojection
and mosaic contract, not a reimplementation of MODIS science. All the masking
and scaling people expect to find here lives downstream in
mask_low_quality_data_and_combine/ and scale_int_tif.py.

THE CONTRACT (recorded in reference_grid.json)
-------------------------------------------------------------------------
  * EPSG:4326, pixel size = native_res / 111319.49079327358
  * origins are exact integer multiples of the pixel size -- one global lattice
    anchored at lon 0 / lat 0, which is why tiles co-register across time and
    across products with no explicit alignment step
  * nearest neighbour (EE never calls .resample(), and near is its default)
  * mosaic is paint-order, last-unmasked-wins, ascending acquisition time --
    not a mean, not a max, not quality-preferring
  * the day is the timezone.utc day [D, D+1), keyed off the granule A-date
  * bands in GEE order; the stack takes BAND 1's dtype for every band

EXTENT IS NOT PART OF THE CONTRACT. Earth Engine clips a download to the
intersection of the requested region with the mosaic footprint, so extents vary
by product and by date (at 500 m tile00, MOD09GA starts at lattice row 13927,
MCD15A3H at 14000). That washes out downstream -- pre_processing_fast.py warps
every driver onto one national template. Only the lattice has to agree.

THE DTYPE COLLAPSE IS DELIBERATE
--------------------------------
stack_bands_to_multitiff (gee_download_fast_copy_parallel.py:871-885) copies
band 1's profile to every band, so MOD09GA_b1237 band 5 (QC_500m, natively
uint32) is stored as Int16 -- confirmed by gdalinfo on the reference tile, and
true for every multi-band product. It never broke training because the default
cloud mask reads state_1km, which is downloaded as its own 1-band product and
so escapes truncation. We reproduce it by default because the model was trained
on truncated data; --correct-dtype turns it off.

Usage:
    # match a reference tile's grid exactly (parity testing)
    python modis_to_tiles.py --product MOD09GA_b1237 --date 2025-07-01 \
        --like /path/to/drivers/MOD09GA_b1237/MOD09GA_b1237_2025_07_01_tile00.tif

    # operational: compute the quadrant from the Canada shapefile
    python modis_to_tiles.py --product MOD09GA_b1237 --date 2026-08-05 --tiles 0,1,2,3

    # operational handoff: tile every newly complete date in a download manifest
    python modis_to_tiles.py \
        --manifest /caribou/quinn/wildfire/reports/manifests/latest.json
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import rasterio

from wildfire import paths
from wildfire.modis.registry import PRODUCTS as CANONICAL_PRODUCTS
from wildfire.modis.registry import converter_products

# ---------------------------------------------------------------------------
# Product registry
#
# gee_name          the directory name the FireMPC pipeline expects
# short_name        the CMR/earthaccess short name the granules live under
# sds               subdataset names in GEE BAND ORDER (physical band order of
#                   the output stack -- load-bearing, downstream indexes by it)
#
# MOD09GA appears twice on purpose: GEE pulled the 500 m reflectance bands and
# the 1 km state_1km QA band as two separate products because they need
# different export scales. merge_mcd09ga_qa.py re-joins them downstream.
# ---------------------------------------------------------------------------
# Derived losslessly from the frozen parity contract in modis_registry, which is
# also where the CMG scale/mask exceptions are pinned (modis_registry.py:169).
# Deriving it rather than restating it here is what keeps the two from drifting.
PRODUCTS = converter_products()

OUTPUTS_BY_SHORT_NAME: dict[str, list[str]] = {}
for _product, _cfg in PRODUCTS.items():
    OUTPUTS_BY_SHORT_NAME.setdefault(_cfg["short_name"], []).append(_product)
for _outputs in OUTPUTS_BY_SHORT_NAME.values():
    _outputs.sort()

# Earth Engine's masked pixels are VALUE 0, in every product and every dtype.
# Measured on the reference: MOD09GA_b1237 tile00 2025-07-01 contains zero
# -28672 (the MODIS sur_refl fill) and zero -32768, but 1.27% exact zeros. EE
# masks on the band's own _FillValue and writes 0 in its place.
EE_MASKED_VALUE = 0

# The nodata TAG Earth Engine puts in the GeoTIFF header, by dtype. Note it does
# not agree with the value above: an int16 tile is tagged -32768 while its masked
# pixels hold 0. That inconsistency is in the training data, so downstream never
# masked on the tag -- fill_modis_spatial_gap.py catches these through
# valid_ranges instead. We reproduce both halves rather than "fixing" one.
EE_NODATA_TAG = {
    "int16": -32768, "int32": -2147483648,
    "uint16": 0, "uint8": 0, "byte": 0,
    "float32": None, "float64": None,
}


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {p.stderr.strip()[:400]}")
    return p.stdout


def granule_adate(filename):
    """Observation date from the AYYYYDDD token. Never the production stamp --
    MOD11A1.A2026216.*.2026217225031 was produced Aug 5 but observes Aug 4."""
    for tok in filename.split("."):
        if len(tok) == 8 and tok.startswith("A") and tok[1:].isdigit():
            return datetime.strptime(tok[1:], "%Y%j").date()
    return None


def subdatasets(hdf):
    """{sds name: full GDAL subdataset spec} for one granule."""
    info = json.loads(run(["gdalinfo", "-json", hdf]))
    out = {}
    for sd in info.get("metadata", {}).get("SUBDATASETS", {}).items():
        key, val = sd
        if not key.endswith("_NAME"):
            continue
        # CMG products quote the SDS name because it contains spaces:
        #   ...:MOD09CMG:"Coarse Resolution Brightness Temperature Band 20"
        name = val.rsplit(":", 1)[-1].strip('"')
        out[name] = val
        # Register aliases so one registry entry matches every naming variant:
        # the L2G-lite "_1" suffix (MOD09GA), and CMG's spaces-vs-underscores.
        bare = re.sub(r"_\d+$", "", name)
        out.setdefault(bare, val)
        out.setdefault(name.replace(" ", "_"), val)
        out.setdefault(bare.replace(" ", "_"), val)
    return out


def resolve_sds(avail, want):
    """Find `want` among a granule's subdatasets, tolerating naming variants."""
    for cand in (want, want.replace(" ", "_"), want.replace("_", " "),
                 re.sub(r"_\d+$", "", want)):
        if cand in avail:
            return avail[cand]
    # last resort: unique case-insensitive prefix match
    low = re.sub(r"_\d+$", "", want).lower()
    hits = [v for k, v in avail.items() if k.lower() == low]
    return hits[0] if hits else None


def sds_meta(spec):
    """
    _FillValue, valid_range and scale_factor for one subdataset.

    valid_range is the one that matters and the one that is easy to miss.
    Earth Engine masks every pixel outside it, not just the fill value, and
    MODIS uses the space above valid_range for class codes rather than a single
    sentinel: MCD15A3H Lai declares valid_range 0-100 with _FillValue 255, but
    also emits 249 (unclassified), 250 (urban), 251 (wetland), 252 (snow/ice),
    253 (barren), 254 (water). Masking on _FillValue alone leaves all six in the
    raster, which on one tile was 3,030,556 pixels disagreeing with the
    reference -- every one of them a class code where GEE had written 0.
    """
    meta = {"fill": None, "valid_range": None, "scale": None}
    try:
        info = json.loads(run(["gdalinfo", "-json", spec]))
    except RuntimeError:
        return meta

    for band in info.get("bands", []):
        if band.get("noDataValue") is not None:
            meta["fill"] = band["noDataValue"]
            break

    md = info.get("metadata", {}).get("", {})
    if meta["fill"] is None and "_FillValue" in md:
        try:
            meta["fill"] = float(md["_FillValue"])
        except ValueError:
            pass
    if "valid_range" in md:
        try:
            lo, hi = [float(x) for x in md["valid_range"].split(",")]
            meta["valid_range"] = (lo, hi)
        except ValueError:
            pass
    if "scale_factor" in md:
        try:
            meta["scale"] = float(md["scale_factor"])
        except ValueError:
            pass
    return meta


# ---------------------------------------------------------------------------
# Target grid
# ---------------------------------------------------------------------------
def _lattice_floor(v, px, tol=1e-6):
    """floor(v/px), but snap to the integer when v is already on the lattice.

    Bare floor() is wrong here: a coordinate that IS on the lattice divides to
    something like -31393.000000000004, and floor takes that to -31394, adding a
    spurious pixel. Every reference origin is on the lattice, so this case is
    the common one, not the edge case."""
    n = v / px
    r = round(n)
    return r if abs(n - r) < tol else math.floor(n)


def _lattice_ceil(v, px, tol=1e-6):
    n = v / px
    r = round(n)
    return r if abs(n - r) < tol else math.ceil(n)


class Grid:
    """A target window on the global lattice."""

    def __init__(self, res_m, minx, miny, maxx, maxy):
        self.res_m = res_m
        self.px = paths.pixel_size(res_m)
        # Snap outward so the window covers the request and lands on the lattice.
        self.minx = _lattice_floor(minx, self.px) * self.px
        self.maxy = _lattice_ceil(maxy, self.px) * self.px
        self.maxx = _lattice_ceil(maxx, self.px) * self.px
        self.miny = _lattice_floor(miny, self.px) * self.px
        self.width = int(round((self.maxx - self.minx) / self.px))
        self.height = int(round((self.maxy - self.miny) / self.px))

    @classmethod
    def from_reference(cls, path, res_m):
        """Adopt an existing tile's exact window. Used for parity testing so a
        comparison measures our warp, not our extent arithmetic."""
        with rasterio.open(path) as src:
            b = src.bounds
            g = cls(res_m, b.left, b.bottom, b.right, b.top)
            if (g.width, g.height) != (src.width, src.height):
                raise RuntimeError(
                    f"reference {os.path.basename(path)} is {src.width}x{src.height} "
                    f"but its bounds snap to {g.width}x{g.height}; "
                    "it is off the lattice")
            return g

    def warp_args(self):
        return ["-t_srs", "EPSG:4326",
                "-te", repr(self.minx), repr(self.miny),
                repr(self.maxx), repr(self.maxy),
                "-ts", str(self.width), str(self.height),
                "-r", "near"]

    def __repr__(self):
        return (f"Grid({self.res_m}m {self.width}x{self.height} "
                f"@({self.minx:.9f},{self.maxy:.9f}) px={self.px:.15f})")


def _geodesic_edge_apex(latitude, longitude_span):
    """Northern apex of a spherical great-circle edge with equal-latitude ends."""
    lat = math.radians(latitude)
    half_span = math.radians(longitude_span) / 2.0
    return math.degrees(math.atan(math.tan(lat) / math.cos(half_span)))


def quadrant_grids(res_m, tiles, export_subdivision=1):
    """
    The 2x2 tile split of the Canada region, per make_fixed_grid /
    intersect_tiles (gee_download_fast_copy_parallel.py:506-520): split
    total_bounds into equal halves, emit column-major with j increasing north,
    intersect with the dissolved region, drop fragments under 0.02 deg^2.
    """
    import geopandas as gpd
    from shapely.geometry import box

    gdf = gpd.read_file(paths.CANADA_GRIDS_SHP)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    region = gdf.geometry.union_all()
    minx, miny, maxx, maxy = region.bounds
    dx, dy = (maxx - minx) / 2.0, (maxy - miny) / 2.0

    cells = []
    for i in range(2):          # column-major, matching the original
        for j in range(2):      # j increasing north
            cell = box(minx + i * dx, miny + j * dy,
                       minx + (i + 1) * dx, miny + (j + 1) * dy)
            piece = cell.intersection(region)
            if not piece.is_empty and piece.area >= 0.02:
                cells.append(piece)

    out = {}
    for idx, piece in enumerate(cells):
        if idx in tiles:
            subdivisions = (export_subdivision[idx]
                            if isinstance(export_subdivision, list)
                            else export_subdivision)
            left, bottom, right, top = piece.bounds
            # Earth Engine interprets each GeoJSON subrequest edge as geodesic.
            # The historical downloader split large requests into N columns;
            # the union envelope is therefore the apex of a width/N edge.
            geodesic_top = _geodesic_edge_apex(
                top, (right - left) / subdivisions)
            out[idx] = Grid(res_m, left, bottom, right, geodesic_top)
    return out


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
def band_to_tif(granule_specs, fills, grid, out_path, source_band=1):
    """
    Warp one band from every granule onto the target grid and mosaic.

    gdalwarp paints sources in the order given, later over earlier, skipping
    each source's nodata -- which is exactly EE's mosaic(). Granules are passed
    in ascending acquisition order so the paint order matches.
    """
    selected_specs = list(granule_specs)
    selection_vrts = []
    if source_band != 1:
        selected_specs = []
        for index, spec in enumerate(granule_specs):
            vrt = f"{out_path}.source{index:02d}.vrt"
            run(["gdal_translate", "-q", "-of", "VRT", "-b", str(source_band),
                 spec, vrt])
            selection_vrts.append(vrt)
            selected_specs.append(vrt)

    cmd = ["gdalwarp", "-q", "-overwrite", "-of", "GTiff",
           "-co", "COMPRESS=LZW", *grid.warp_args()]
    # Source fill is skipped when painting; anything left unpainted becomes 0,
    # which is what Earth Engine writes for a masked pixel. Uncovered ground and
    # genuine fill are therefore indistinguishable in the output -- exactly as in
    # the training tiles.
    fill = next((f for f in fills if f is not None), None)
    if fill is not None:
        cmd += ["-srcnodata", repr(fill)]
    cmd += ["-dstnodata", str(EE_MASKED_VALUE)]
    cmd += selected_specs + [out_path]
    try:
        run(cmd)
    finally:
        for vrt in selection_vrts:
            try:
                os.remove(vrt)
            except FileNotFoundError:
                pass
    return out_path


# ---------------------------------------------------------------------------
# GEE pyramid emulation
#
# Earth Engine served part of tile 1 on the 500 m optical exports from pyramid
# level 1 -- each output pixel is round(mean(aligned 2x2 SOURCE block)) instead
# of the level-0 pixel. The training cube carries that, so reproducing it is the
# same call the band-1 dtype collapse already makes: the model was trained on it.
#
# The window is MEASURED against the reference tiles, not derived. The export
# chunk grid predicts the boundary at row 3095, but the truth is 3060, because
# GEE's per-chunk downloads overlap and the neighbouring level-0 chunk was merged
# over the edge. Both edges are razor sharp -- col 3279 is 100% pyramid, col 3280
# is 100% level-0 -- and being one pixel out costs ~3,000 wrong pixels, so the
# window is pinned to the shape it was measured on and refuses to guess.
#
# Measured 2026-08-12 on MOD09GA_b1237/MYD09GA_b1237 2025-07-01 and
# MOD09A1/MYD09A1 2025-07-04: all four identical.
GEE_PYRAMID_BLOCKS = {
    # (product, tile): (expected (height, width), (row0, row1, col0, col1))
    ("MOD09GA_b1237", 1): ((4637, 9839), (0, 3060, 0, 3280)),
    ("MYD09GA_b1237", 1): ((4637, 9839), (0, 3060, 0, 3280)),
    ("MOD09A1", 1): ((4637, 9839), (0, 3060, 0, 3280)),
    ("MYD09A1", 1): ((4637, 9839), (0, 3060, 0, 3280)),
}


def pyramid_bands(product):
    """
    Band indices EE served from pyramid level 1.

    Earth Engine pyramids continuous bands with MEAN and bit-packed QA bands
    with SAMPLE.  Earlier code excluded QA because applying the reflectance
    mean to QC_500m made parity worse; the missing distinction was the reducer,
    not whether the QA band had a pyramid at all.
    """
    return list(range(len(CANONICAL_PRODUCTS[product]["bands"])))


def pyramid_policy(product, band_index):
    """Earth Engine ingestion policy for one band in a level-1 block."""
    band = CANONICAL_PRODUCTS[product]["bands"][band_index]
    return "sample" if band["qa"] else "mean"


def pyramid_window(product, tile, grid):
    """The output window GEE served from pyramid level 1, or None."""
    entry = GEE_PYRAMID_BLOCKS.get((product, tile))
    if entry is None:
        return None
    shape, window = entry
    if (grid.height, grid.width) != shape:
        raise RuntimeError(
            f"{product} tile{tile:02d}: GEE pyramid window was measured on a "
            f"{shape[0]}x{shape[1]} grid but this grid is "
            f"{grid.height}x{grid.width}. The window is measured, not derived, "
            "so it cannot be transferred to a different extent -- re-measure "
            "against the reference or pass --no-gee-pyramid.")
    return window


def _reduce_level1_blocks(q, fill, policy):
    """Reduce an ``(H, 2, W, 2)`` source view using an EE policy."""
    if policy == "sample":
        # EE SAMPLE is the upper-left member of each aligned 2x2 block;
        # its mask is sampled with it rather than averaged.
        return q[:, 0, :, 0].copy()
    if policy != "mean":
        raise ValueError(f"unsupported pyramid policy: {policy}")
    if fill is None:
        return np.floor(q.mean(axis=(1, 3)) + 0.5)
    valid = q != fill
    n = valid.sum(axis=(1, 3))
    s = np.where(valid, q, 0.0).sum(axis=(1, 3))
    reduced = np.full(n.shape, float(fill))
    ok = n > 0
    reduced[ok] = np.floor(s[ok] / n[ok] + 0.5)
    return reduced


def _level1_source(spec, fill, out, source_band=1, policy="mean"):
    """
    One granule resampled to pyramid level 1: the aligned 2x2 mean, exactly.

    Done in numpy rather than `gdalwarp -r average` on purpose. The means that
    matter land exactly on .5 (that is what makes them differ from level-0 at
    all), and gdalwarp returns 5577.4999999 where the true mean is 5577.5, which
    rounds the wrong way and was worth 1.2M wrong pixels. Sums of four int16 fit
    in float64 exactly and x/4 is exact in binary, so numpy is exact here.
    """
    l0 = out + ".l0.tif"
    run(["gdal_translate", "-q", "-of", "GTiff", "-b", str(source_band), spec, l0])
    try:
        with rasterio.open(l0) as src:
            a = src.read([1])[0]
            profile = src.profile.copy()
            t = src.transform
        h, w = a.shape
        h -= h % 2
        w -= w % 2
        q = a[:h, :w].astype(np.float64).reshape(h // 2, 2, w // 2, 2)
        m = _reduce_level1_blocks(q, fill, policy)
        profile.update(count=1, height=m.shape[0], width=m.shape[1],
                       dtype="float64", nodata=fill, compress="LZW",
                       transform=rasterio.Affine(t.a * 2, t.b, t.c,
                                                 t.d, t.e * 2, t.f))
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(m, 1)
    finally:
        try:
            os.remove(l0)
        except FileNotFoundError:
            pass
    return out


def apply_gee_pyramid(band_tif, specs, fill, grid, window, tmp_root, stem,
                      source_band=1, policy="mean"):
    """Overwrite `window` in band_tif with what EE served from pyramid level 1."""
    l1_specs = []
    try:
        for i, spec in enumerate(specs):
            l1_specs.append(_level1_source(
                spec, fill, os.path.join(tmp_root, f"{stem}.l1_{i:03d}.tif"),
                source_band=source_band, policy=policy))
        warped = os.path.join(tmp_root, f"{stem}.l1warp.tif")
        # source_band is already selected in the level-1 rasters.
        band_to_tif(l1_specs, [fill] * len(l1_specs), grid, warped, source_band=1)
        r0, r1, c0, c1 = window
        win = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
        with rasterio.open(warped) as src:
            patch = src.read([1], window=win)[0]
        with rasterio.open(band_tif, "r+") as dst:
            patch = np.asarray(patch, dtype=dst.dtypes[0])
            dst.write(patch, 1, window=win)
    finally:
        for path in [*l1_specs, os.path.join(tmp_root, f"{stem}.l1warp.tif")]:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
    return band_tif


def apply_ee_semantics(arr, meta, mask_valid_range, scale):
    """
    Turn a warped MODIS band into what Earth Engine would have served.

    Masking after the warp rather than before is safe precisely because the
    resampling is nearest neighbour: NN only ever copies a source value, so
    mask-then-warp and warp-then-mask give identical rasters. With bilinear it
    would not, and this would have to move.
    """
    if mask_valid_range and meta.get("valid_range") is not None:
        lo, hi = meta["valid_range"]
        out_of_range = (arr < lo) | (arr > hi)
        if out_of_range.any():
            arr = arr.copy()
            arr[out_of_range] = EE_MASKED_VALUE
    if scale not in (None, 1):
        arr = arr.astype(np.float64) * scale
    return arr


def stack(band_tifs, out_path, band_names, metas, band_scales=None,
          mask_valid_range=True, faithful_dtype=True):
    """
    Stack single-band rasters into the multi-band GeoTIFF the pipeline expects.

    Faithful mode reproduces the training-time bug: band 1's profile is applied
    to every band, so a uint32 QA word lands in an int16 raster and wraps. The
    model was trained on that, so it is the default.
    """
    opens = [rasterio.open(p) for p in band_tifs]
    try:
        band_scales = band_scales or {}
        # Rasterio's scalar-band read(1) mutates ndarray.shape internally and
        # warns under NumPy 2.5. The list-band path avoids that deprecated
        # operation; index 0 restores the same 2-D array.
        arrays = [apply_ee_semantics(src.read([1])[0], metas[i], mask_valid_range,
                                     band_scales.get(band_names[i]))
                  for i, src in enumerate(opens)]

        profile = opens[0].profile.copy()
        if faithful_dtype:
            dtype = arrays[0].dtype.name          # band 1 wins, bug and all
        else:
            dtype = np.result_type(*[a.dtype for a in arrays]).name
        nodata = EE_NODATA_TAG.get(dtype.lower(), None)
        profile.update(count=len(opens), dtype=dtype, compress="LZW", nodata=nodata)

        with rasterio.open(out_path, "w", **profile) as dst:
            for i, arr in enumerate(arrays, start=1):
                dst.write(np.asarray(arr, dtype=dtype), i)
    finally:
        for s in opens:
            s.close()
    return out_path


def is_nrt_granule(filename):
    """NRT granules are explicitly suffixed by modis_download.py."""
    return ".NRT." in filename


def granules_for_day(src_dir, day, stream, source_span_days=1):
    """Select one provenance stream and reject duplicate tile keys."""
    granules = sorted(f for f in os.listdir(src_dir) if f.endswith(".hdf"))
    granules = [
        f for f in granules
        if (source_day := granule_adate(f)) is not None
        and source_day <= day < source_day + timedelta(days=source_span_days)
    ]
    if stream == "standard":
        granules = [f for f in granules if not is_nrt_granule(f)]
    elif stream == "nrt":
        granules = [f for f in granules if is_nrt_granule(f)]

    keys = {}
    for filename in granules:
        match = re.search(r"\.h\d{2}v\d{2}\.", filename)
        # Global CMG products have no MODIS tile token: one granule is the key.
        key = match.group(0).strip(".") if match else "global"
        if key in keys:
            raise RuntimeError(
                f"duplicate {stream} granules for {day} {key}: "
                f"{keys[key]}, {filename}")
        keys[key] = filename
    return [os.path.join(src_dir, f) for f in granules]


def output_path(out_root, product, day, tile_idx):
    stem = f"{product}_{day.strftime('%Y_%m_%d')}_tile{tile_idx:02d}"
    return os.path.join(out_root, product, f"{stem}.tif")


def validate_raster(path, product):
    """
    Raster-contract violations that hold for a driver tile from ANY source.

    Deliberately says nothing about extent. Earth Engine clips a download to the
    intersection of the request with the mosaic footprint, so a GEE-sourced tile
    and our warp of the same day legitimately differ in width, height and origin
    -- see EXTENT IS NOT PART OF THE CONTRACT above. What both must satisfy is
    the lattice: same CRS, same pixel size, origin an exact multiple of it. That
    is what makes tiles co-register, and it is checkable without a target grid,
    so the GEE backend can run these too.
    """
    cfg = PRODUCTS[product]
    px = paths.pixel_size(cfg["res"])
    tol = max(1e-12, px * 1e-9)
    errors = []
    try:
        with rasterio.open(path) as src:
            # The host currently has mismatched GDAL/PROJ databases, so WGS84
            # WKT written by GDAL may not resolve back to the numeric EPSG code.
            # Accept that exact geographic WGS84 definition as equivalent.
            crs_text = str(src.crs) if src.crs is not None else ""
            is_wgs84 = (src.crs is not None and src.crs.is_geographic and
                        "WGS 84" in crs_text)
            if not is_wgs84:
                errors.append(f"CRS is {src.crs}, expected EPSG:4326")
            if src.count != len(cfg["sds"]):
                errors.append(f"bands={src.count}, expected {len(cfg['sds'])}")
            # Historical GEE exports carry no GeoTIFF band descriptions. Band
            # order is pinned by the canonical registry and checked by count.
            if any(description is not None for description in src.descriptions):
                errors.append("band descriptions must be absent like the GEE exports")
            if len(set(src.dtypes)) != 1:
                errors.append(f"mixed output dtypes: {src.dtypes}")
            transform = src.transform
            if abs(transform.a - px) > tol or abs(transform.e + px) > tol:
                errors.append(
                    f"pixel size is ({transform.a}, {transform.e}), "
                    f"expected +/-{px}")
            if not paths.is_on_lattice(transform.c, px):
                errors.append("x origin is off the global lattice")
            if not paths.is_on_lattice(transform.f, px):
                errors.append("y origin is off the global lattice")
            # Force GDAL to read every block. Opening metadata alone does not
            # detect a GeoTIFF truncated after its directory was written.
            for band in range(1, src.count + 1):
                src.checksum(band)
    except Exception as exc:  # noqa: BLE001 - corrupt rasters must become a verdict.
        errors.append(f"cannot open raster: {exc}")
    return errors


def validate_tile(path, product, grid):
    """
    validate_raster plus the extent we control when we do the warp ourselves.

    Only meaningful on the earthaccess path: there the target window is ours to
    choose, so landing off it is a real defect rather than EE footprint clipping.
    """
    errors = validate_raster(path, product)
    try:
        with rasterio.open(path) as src:
            tol = max(1e-12, grid.px * 1e-9)
            if (src.width, src.height) != (grid.width, grid.height):
                errors.append(
                    f"shape is {src.width}x{src.height}, expected "
                    f"{grid.width}x{grid.height}")
            transform = src.transform
            if (abs(transform.c - grid.minx) > tol or
                    abs(transform.f - grid.maxy) > tol):
                errors.append(
                    f"origin is ({transform.c}, {transform.f}), expected "
                    f"({grid.minx}, {grid.maxy})")
    except Exception as exc:  # noqa: BLE001 - validate_raster already recorded the open failure.
        if not errors:
            errors.append(f"cannot open raster: {exc}")
    return errors


def build(product, day, grids, raw_root, out_root, tmp_root,
          faithful_dtype=True, dry_run=False, stream="standard",
          emulate_pyramid=True):
    cfg = PRODUCTS[product]
    src_dir = os.path.join(raw_root, cfg["short_name"], day.strftime("%Y-%m-%d"))
    if not os.path.isdir(src_dir):
        raise RuntimeError(f"no granule directory: {src_dir}")

    source_span_days = int(cfg.get("source_span_days", 1))
    granules = granules_for_day(src_dir, day, stream, source_span_days)
    if not granules:
        raise RuntimeError(
            f"no {stream} granules with A-date {day} in {src_dir}")

    print(f"  {len(granules)} granule(s), {len(cfg['sds'])} band(s)")

    # Resolve every SDS up front so a naming miss fails before any warping.
    per_granule = []
    for g in granules:
        avail = subdatasets(g)
        specs = {}
        for name in cfg["sds"]:
            spec = resolve_sds(avail, name)
            if spec is None:
                raise RuntimeError(
                    f"{os.path.basename(g)}: no subdataset for {name!r}.\n"
                    f"  available: {sorted(k for k in avail if ' ' not in k)[:25]}")
            specs[name] = spec
        per_granule.append(specs)

    if dry_run:
        for name in cfg["sds"]:
            print(f"    {name} -> {per_granule[0][name].rsplit(':', 2)[-2:]}")
        return []

    # Band semantics come from the granule itself, read once per band.
    metas = [sds_meta(per_granule[0][name]) for name in cfg["sds"]]
    source_dates = {granule_adate(os.path.basename(path)) for path in granules}
    if len(source_dates) != 1 or None in source_dates:
        raise RuntimeError(f"source granules do not share one A-date: {source_dates}")
    source_band = (day - source_dates.pop()).days + 1

    os.makedirs(tmp_root, exist_ok=True)
    written = []
    for tile_idx, grid in sorted(grids.items()):
        stem = f"{product}_{day.strftime('%Y_%m_%d')}_tile{tile_idx:02d}"
        band_tifs = []
        out_dir = os.path.join(out_root, product)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{stem}.tif")
        partial = os.path.join(out_dir, f".{stem}.{os.getpid()}.part.tif")
        try:
            window = (pyramid_window(product, tile_idx, grid)
                      if emulate_pyramid else None)
            pyr_bands = set(pyramid_bands(product)) if window else set()
            if window:
                print(f"    tile{tile_idx:02d} emulating GEE pyramid level 1 over "
                      f"rows {window[0]}:{window[1]} cols {window[2]}:{window[3]} "
                      f"on bands {sorted(b+1 for b in pyr_bands)}")
            for bi, name in enumerate(cfg["sds"]):
                specs = [pg[name] for pg in per_granule]
                bt = os.path.join(tmp_root, f"{stem}.{os.getpid()}_b{bi:02d}.tif")
                band_to_tif(specs, [metas[bi]["fill"]] * len(specs), grid, bt,
                            source_band=source_band)
                if window and bi in pyr_bands:
                    apply_gee_pyramid(bt, specs, metas[bi]["fill"], grid, window,
                                      tmp_root, f"{stem}.{os.getpid()}_b{bi:02d}",
                                      source_band=source_band,
                                      policy=pyramid_policy(product, bi))
                band_tifs.append(bt)

            stack(band_tifs, partial, cfg["sds"], metas,
                  band_scales=cfg.get("band_scales"),
                  mask_valid_range=cfg.get("mask_valid_range", True),
                  faithful_dtype=faithful_dtype)
            errors = validate_tile(partial, product, grid)
            if errors:
                raise RuntimeError(f"{stem}: " + "; ".join(errors))
            os.replace(partial, out_path)
            with rasterio.open(out_path) as src:
                print(f"    tile{tile_idx:02d} -> {src.width}x{src.height} "
                      f"{src.count}x{src.dtypes[0]}  {out_path}")
            written.append(out_path)
        finally:
            for scratch in [*band_tifs, partial]:
                try:
                    os.remove(scratch)
                except FileNotFoundError:
                    pass
    return written


def load_manifest(path):
    with open(path) as f:
        manifest = json.load(f)
    required = {"run_utc", "stream", "products"}
    missing = required - manifest.keys()
    if missing:
        raise RuntimeError(
            f"manifest {path} is missing: {', '.join(sorted(missing))}")
    if manifest["stream"] not in {"standard", "nrt"}:
        raise RuntimeError(f"unsupported manifest stream {manifest['stream']!r}")
    # A download manifest describes GRANULES and is keyed by CMR short name. The
    # GEE backend writes a tiling report instead, but older runs of it left
    # export-name-keyed manifests at this path, and those look valid while
    # matching zero products -- a silent no-op. Fail loudly on them.
    if manifest.get("source") not in (None, "earthaccess"):
        raise RuntimeError(
            f"manifest {path} was written by the {manifest['source']!r} backend; "
            "only an earthaccess download manifest can be tiled")
    if manifest.get("errors"):
        raise RuntimeError("refusing a download manifest that contains errors")
    return manifest


def manifest_jobs(manifest):
    """Yield (raw product, output product, day, source status)."""
    jobs = []
    for short_name, block in sorted(manifest["products"].items()):
        outputs = OUTPUTS_BY_SHORT_NAME.get(short_name, [])
        entries = block.get("dates") or block.get("anchors") or {}
        for day_text, entry in sorted(entries.items()):
            status = entry.get("status")
            ready = status == "complete"
            # Annual land cover is intentionally old and cannot be latency-gated
            # against a current-year publication. Its manifest marks that policy.
            ready |= (status == "present" and block.get("stale_by_design") is True)
            if not ready:
                continue
            for product in outputs:
                jobs.append((short_name, product,
                             datetime.strptime(day_text, "%Y-%m-%d").date(), status))
    return jobs


def write_tiling_report(report):
    out_dir = os.path.join(paths.REPORTS, "tiling")
    os.makedirs(out_dir, exist_ok=True)
    stamp = report["run_utc"].replace(":", "").replace("-", "")
    stamp = stamp.replace(".", "").replace("+0000", "Z")
    body = json.dumps(report, indent=1, sort_keys=True)
    history = os.path.join(out_dir, f"{stamp}.json")
    for path in (history, os.path.join(out_dir, "latest.json")):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write(body)
        os.replace(tmp, path)
    return history


def tile_manifest(manifest, *, raw=None, out=paths.DRIVERS, tmp=None,
                  tiles="0,1,2,3", stream="standard", overwrite=False,
                  correct_dtype=False, dry_run=False, product=None, date=None,
                  no_gee_pyramid=False):
    """
    Library entry point: tile every ready date in a download manifest.

    Exists so modis_download.py can finish an earthaccess run in-process instead
    of shelling out. Returns run_manifest's exit code (1 if any job failed).
    """
    return run_manifest(argparse.Namespace(
        manifest=manifest, raw=raw, out=out,
        tmp=tmp or os.path.join(paths.DATA_ROOT, "tmp"),
        tiles=tiles, stream=stream, overwrite=overwrite,
        correct_dtype=correct_dtype, dry_run=dry_run,
        no_gee_pyramid=no_gee_pyramid,
        product=product, date=date))


def run_manifest(args):
    manifest = load_manifest(args.manifest)
    raw_root = args.raw or manifest.get("dest") or paths.RAW
    jobs = manifest_jobs(manifest)
    if args.product:
        jobs = [job for job in jobs if job[1] == args.product]
    if args.date:
        wanted_day = datetime.strptime(args.date, "%Y-%m-%d").date()
        jobs = [job for job in jobs if job[2] == wanted_day]

    print(f"manifest {args.manifest}")
    print(f"stream={manifest['stream']}  ready output-dates={len(jobs)}")
    report = {
        "run_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": "earthaccess",
        "source_manifest": os.path.abspath(args.manifest),
        "source_run_utc": manifest["run_utc"],
        "stream": manifest["stream"],
        "gee_pyramid_emulated": not args.no_gee_pyramid,
        "raw": raw_root,
        "out": args.out,
        "results": [],
    }
    grid_cache = {}
    failures = 0
    selected_tiles = {int(t) for t in args.tiles.split(",") if t.strip()}

    for short_name, product, day, source_status in jobs:
        cfg = PRODUCTS[product]
        subdivisions = cfg["export_subdivision"]
        subdivisions_key = tuple(subdivisions) if isinstance(subdivisions, list) else subdivisions
        key = (cfg["res"], subdivisions_key, tuple(sorted(selected_tiles)))
        if key not in grid_cache:
            grid_cache[key] = quadrant_grids(
                cfg["res"], selected_tiles, cfg["export_subdivision"])
        grids = grid_cache[key]
        result = {
            "raw_product": short_name,
            "product": product,
            "date": day.isoformat(),
            "source_status": source_status,
            "status": "planned" if args.dry_run else None,
            "tiles": {},
        }
        report["results"].append(result)

        pending = {}
        for tile_idx, grid in sorted(grids.items()):
            path = output_path(args.out, product, day, tile_idx)
            errors = (validate_tile(path, product, grid)
                      if os.path.isfile(path) else ["missing"])
            if not errors and not args.overwrite:
                result["tiles"][str(tile_idx)] = {"status": "existing", "path": path}
            else:
                pending[tile_idx] = grid
                result["tiles"][str(tile_idx)] = {
                    "status": "planned" if args.dry_run else "pending",
                    "path": path,
                    "replaces": errors if errors != ["missing"] else [],
                }

        print(f"{product:24s} {day}  existing={len(grids) - len(pending)} "
              f"build={len(pending)}")
        if args.dry_run or not pending:
            result["status"] = "planned" if args.dry_run else "complete"
            continue

        try:
            written = build(product, day, pending, raw_root, args.out, args.tmp,
                            faithful_dtype=not args.correct_dtype,
                            stream=manifest["stream"],
                            emulate_pyramid=not args.no_gee_pyramid)
            if len(written) != len(pending):
                raise RuntimeError(
                    f"wrote {len(written)} of {len(pending)} requested tiles")
            for path in written:
                match = re.search(r"_tile(\d+)\.tif$", path)
                if match is None:
                    raise RuntimeError(f"cannot parse tile index from {path}")
                tile_idx = int(match.group(1))
                result["tiles"][str(tile_idx)]["status"] = "written"
            result["status"] = "complete"
        except Exception as exc:  # noqa: BLE001 - continue independent jobs and report all failures.
            failures += 1
            result["status"] = "failed"
            result["error"] = str(exc)
            print(f"  ERROR {product} {day}: {exc}", file=sys.stderr)

    counts = {}
    for result in report["results"]:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    report["summary"] = {"jobs": len(jobs), "failures": failures,
                         "status_counts": counts}
    if not args.dry_run:
        print(f"report -> {write_tiling_report(report)}")
    return 1 if failures else 0


def run_single(args):
    if not args.product or not args.date:
        raise SystemExit("single-date mode requires both --product and --date")
    day = datetime.strptime(args.date, "%Y-%m-%d").date()
    res = PRODUCTS[args.product]["res"]

    if args.like:
        tile_idx = 0
        match = re.search(r"_tile(\d+)\.tif$", args.like)
        if match:
            tile_idx = int(match.group(1))
        grids = {tile_idx: Grid.from_reference(args.like, res)}
    else:
        tiles = {int(t) for t in args.tiles.split(",") if t.strip()}
        grids = quadrant_grids(res, tiles, PRODUCTS[args.product]["export_subdivision"])

    print(f"{args.product}  {day}  res={res}m  stream={args.stream}  "
          f"dtype={'faithful' if not args.correct_dtype else 'corrected'}  "
          f"pyramid={'emulated' if not args.no_gee_pyramid else 'off'}")
    for tile_idx, grid in sorted(grids.items()):
        print(f"  tile{tile_idx:02d} {grid}")

    build(args.product, day, grids, args.raw or paths.RAW, args.out, args.tmp,
          faithful_dtype=not args.correct_dtype, dry_run=args.dry_run,
          stream=args.stream, emulate_pyramid=not args.no_gee_pyramid)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest",
                    help="download manifest; tile every ready date incrementally")
    ap.add_argument("--product", choices=sorted(PRODUCTS),
                    help="output product (required with --date; optional "
                         "manifest filter)")
    ap.add_argument("--date",
                    help="observation date YYYY-MM-DD (timezone.utc; optional manifest filter)")
    ap.add_argument("--like", help="match this reference tile's grid exactly "
                                   "(parity testing); implies a single tile")
    ap.add_argument("--tiles", default="0,1,2,3", help="tile indices (default all)")
    ap.add_argument("--raw",
                    help=f"raw HDF root (single-date default: {paths.RAW}; "
                         "manifest mode default: manifest dest)")
    ap.add_argument("--out", default=paths.DRIVERS)
    ap.add_argument("--tmp", default=os.path.join(paths.DATA_ROOT, "tmp"))
    ap.add_argument("--stream", choices=["standard", "nrt"], default="standard",
                    help="single-date provenance (manifest mode uses the manifest)")
    ap.add_argument("--overwrite", action="store_true",
                    help="manifest mode: rebuild outputs that already validate")
    ap.add_argument("--no-gee-pyramid", action="store_true",
                    help="do NOT reproduce the GEE pyramid-level-1 chunk on tile 1 "
                         "of the 500 m optical exports. Diverges from the data the "
                         "model was trained on.")
    ap.add_argument("--correct-dtype", action="store_true",
                    help="do NOT reproduce the band-1 dtype collapse; widen instead. "
                         "Diverges from the data the model was trained on.")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve subdatasets and print the plan, warp nothing")
    args = ap.parse_args()

    if args.manifest and args.like:
        ap.error("--like is only valid in single-date mode")
    return run_manifest(args) if args.manifest else run_single(args)


if __name__ == "__main__":
    sys.exit(main())
