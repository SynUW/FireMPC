#!/usr/bin/env python3
"""
Use a shapefile to clip every GeoTIFF in a directory and write to a new directory; outputs are uniformly EPSG:4326 (WGS84).
Vector and raster CRSs may differ (aligned in their own CRSs first, then clipped, then reprojected to WGS84).
Supports multiprocess parallelism.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.mask import mask
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import mapping
from shapely.ops import unary_union

try:
    from shapely import make_valid as _shapely_make_valid
except ImportError:  # Shapely < 2
    _shapely_make_valid = None


def _make_valid_geometry(geom):
    """Repair self-intersections / topology errors to avoid rasterio.mask raising TopologyException."""
    if geom.is_empty:
        return geom
    if _shapely_make_valid is not None:
        return _shapely_make_valid(geom)
    if not geom.is_valid:
        return geom.buffer(0)
    return geom


def _keep_polygonal(geom):
    """
    Keep only polygonal features (Polygon / MultiPolygon). make_valid can return a GeometryCollection
    containing LineString / Point, which would interfere with subsequent masking.
    """
    if geom.is_empty:
        return geom
    gt = getattr(geom, "geom_type", None)
    if gt in ("Polygon", "MultiPolygon"):
        return geom
    if gt == "GeometryCollection":
        polys = [g for g in geom.geoms if getattr(g, "geom_type", None) in ("Polygon", "MultiPolygon")]
        if not polys:
            return geom.__class__()  # empty
        return unary_union(polys)
    # Other types (LineString / Point, etc.) are discarded
    return geom.__class__()  # empty


# Define WGS84 via PROJ4 to avoid from_epsg failures in environments where the EPSG database is out of sync with the GDAL/PROJ version
WGS84_CRS = CRS.from_string("+proj=longlat +datum=WGS84 +no_defs")
_GDF: Optional[gpd.GeoDataFrame] = None


def _init_worker(shp_path: str) -> None:
    global _GDF
    _GDF = gpd.read_file(shp_path)


def _clip_one_tif(
    tif_path: Path,
    out_dir: Path,
    resampling: Resampling,
) -> Tuple[str, Optional[str]]:
    """
    Returns (input_path, error_message); error_message is None on success.
    """
    assert _GDF is not None
    tif_path = Path(tif_path)
    out_dir = Path(out_dir)
    out_path = out_dir / tif_path.name

    try:
        with rasterio.open(tif_path) as src:
            if src.crs is None:
                return str(tif_path), "Raster has no CRS; cannot align with the vector"

            gdf_r = _GDF.to_crs(src.crs)
            if gdf_r.geometry.is_empty.all():
                return str(tif_path), "Vector clip is empty"

            # Important: do not union a "country-wide highly complex geometry collection" directly (it easily triggers TopologyException).
            # Repair / filter feature-by-feature first, then union -- much more stable.
            geoms = []
            for g in gdf_r.geometry:
                if g is None or g.is_empty:
                    continue
                try:
                    gv = _make_valid_geometry(g)
                    gv = _keep_polygonal(gv)
                except Exception:
                    continue
                if gv is None or gv.is_empty:
                    continue
                geoms.append(gv)

            if not geoms:
                return str(tif_path), "No valid polygon features remain after repair / filtering"

            try:
                geom = unary_union(geoms)
            except Exception as e:
                return str(tif_path), f"unary_union failed: {e}"

            geom = _keep_polygonal(_make_valid_geometry(geom))
            if geom is None or geom.is_empty:
                return str(tif_path), "Merged-and-repaired geometry is empty"

            shapes = [mapping(geom)]
            nodata = src.nodata

            try:
                clipped, out_transform = mask(
                    src,
                    shapes,
                    crop=True,
                    nodata=nodata,
                    filled=True,
                )
            except ValueError as e:
                return str(tif_path), f"mask failed (possibly no overlap): {e}"

            if clipped.size == 0 or clipped.shape[1] == 0 or clipped.shape[2] == 0:
                return str(tif_path), "Clip result is empty"

            height, width = clipped.shape[1], clipped.shape[2]
            left, bottom, right, top = array_bounds(height, width, out_transform)

            dst_transform, dst_w, dst_h = calculate_default_transform(
                src.crs,
                WGS84_CRS,
                width,
                height,
                left,
                bottom,
                right,
                top,
            )

            fill = nodata if nodata is not None else 0
            dst = np.full((clipped.shape[0], dst_h, dst_w), fill, dtype=clipped.dtype)

            reproj_kw = {"resampling": resampling}
            if nodata is not None:
                reproj_kw["src_nodata"] = nodata
                reproj_kw["dst_nodata"] = nodata

            for i in range(clipped.shape[0]):
                reproject(
                    source=clipped[i],
                    destination=dst[i],
                    src_transform=out_transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=WGS84_CRS,
                    **reproj_kw,
                )

            out_meta = src.meta.copy()
            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": dst_h,
                    "width": dst_w,
                    "transform": dst_transform,
                    "crs": WGS84_CRS,
                    "nodata": nodata,
                }
            )

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out_path, "w", **out_meta) as dst_ds:
                dst_ds.write(dst)
                if src.tags():
                    dst_ds.update_tags(**src.tags())

        return str(tif_path), None
    except Exception as e:  # noqa: BLE001
        return str(tif_path), str(e)


def _fmt_hms(seconds: float) -> str:
    """Format as H:MM:SS; under one hour is 0:MM:SS (matching common progress bars)."""
    s = int(max(0, round(seconds)))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h}:{m:02d}:{sec:02d}"


def _parse_resampling(name: str) -> Resampling:
    key = name.lower().replace("-", "_")
    mapping_r = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
        "mode": Resampling.mode,
    }
    if key not in mapping_r:
        raise argparse.ArgumentTypeError(
            f"Unknown resampling: {name}; valid options: {', '.join(mapping_r)}"
        )
    return mapping_r[key]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Use a shapefile to clip TIFs in a folder in parallel; output EPSG:4326"
    )
    parser.add_argument("--shp", 
                        default='/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/canada_boundary/lpr_000b21a_e_merge_simplified.shp', type=Path, help="Path to the shapefile used for clipping")
    parser.add_argument(
        "--input-dir", default='/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/vis/drivers_vis/drivers', type=Path, help="Directory containing input TIFs"
    )
    parser.add_argument(
        "--output-dir", default='/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/vis/drivers_vis/drivers_clipped', type=Path, help="Output directory (will be created automatically)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of processes, default: min(CPU cores, task count)",
    )
    parser.add_argument(
        "--resampling",
        type=_parse_resampling,
        default=Resampling.nearest,
        help="Resampling used when reprojecting to WGS84: nearest, bilinear, cubic, average, mode",
    )
    parser.add_argument(
        "--pattern",
        default="*.tif",
        help="Filename glob, default *.tif",
    )
    args = parser.parse_args()

    shp = args.shp.expanduser().resolve()
    in_dir = args.input_dir.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()

    if not shp.exists():
        print(f"Error: shapefile not found: {shp}", file=sys.stderr)
        return 1
    if not in_dir.is_dir():
        print(f"Error: input directory does not exist: {in_dir}", file=sys.stderr)
        return 1

    tifs = sorted(in_dir.glob(args.pattern))
    if not tifs:
        print(f"Not in {in_dir} found matches {args.pattern!r} files", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    n_workers = args.workers
    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, len(tifs))
    n_workers = max(1, n_workers)

    print(f"shapefile: {shp}")
    print(f"Input: {len(tifs)} files -> {out_dir}")
    print(f"Process count: {n_workers}, resampling: {args.resampling.name}")

    ok, fail = 0, 0
    failures: list[tuple[str, str]] = []
    total = len(tifs)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(str(shp),),
    ) as ex:
        futures = {
            ex.submit(_clip_one_tif, tif, out_dir, args.resampling): tif for tif in tifs
        }
        done = 0
        for fut in as_completed(futures):
            path, err = fut.result()
            done += 1
            if err:
                failures.append((path, err))
                fail += 1
            else:
                ok += 1

            elapsed = time.perf_counter() - t0
            if done > 0 and elapsed > 0:
                proj_total = elapsed * total / done
                eta = max(0.0, (total - done) * elapsed / done)
            else:
                proj_total = 0.0
                eta = 0.0
            line = (
                f"[{done}/{total}], {_fmt_hms(elapsed)}/{_fmt_hms(proj_total)}, "
                f"ETA: {_fmt_hms(eta)}"
            )
            print(f"\r{line}", end="", file=sys.stderr, flush=True)

    print(file=sys.stderr)
    for path, err in failures:
        print(f"[FAIL] {path}\n        {err}", file=sys.stderr)

    print(f"Done: succeeded {ok}, failed {fail}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
