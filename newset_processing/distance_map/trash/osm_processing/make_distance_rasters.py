import os
import math
import argparse
from typing import List, Iterable, Dict, Any, Set, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt, binary_dilation
import fiona

# Layer categories
layers_needed = {
    "distance_buildings": ["buildings"],
    "distance_water": ["water_lines"],  # water_areas would make almost the entire image water and requires further filtering, so it is omitted for now
    "distance_power": ["power_lines", "power_substations", "power_points"],
}

# For each theme, if these fields exist, filter for "non-empty"; if absent, do not filter (keep all)
non_null_fields_map = {
    "distance_buildings": ["building"],
    "distance_water": ["waterway", "water", "natural", "landuse"],
    "distance_power": ["power"],
}

# Value filter: applied only when specified; here we keep only waterway == 'river'
value_filters_map: Dict[str, Dict[str, Set[str]]] = {
    "distance_water": {"waterway": {"river"}},
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate distance rasters from one or more GPKG thematic layers (computed on a lon/lat grid; pixel value is distance to the nearest target in km)"
    )
    ap.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/osm_map/canada-us-mosaic-thematics.gpkg",
        ],
        help="List of input GPKG paths (same-named layers are auto-aggregated)",
    )
    ap.add_argument(
        "--output-dir",
        default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/osm_map/distance_map",
        help="Output directory",
    )
    ap.add_argument(
        "--resolution-m",
        type=float,
        default=1000.0,
        help="Resolution (meters), used to convert to lon/lat pixel size",
    )
    return ap.parse_args()


# Streaming read: filter by non-empty attributes and emit GeoJSON geometries directly for rasterization
def iter_shapes_from_layers(
    gpkg_paths: List[str],
    layer_names: List[str],
    candidate_cols: List[str],
) -> Iterable[Dict[str, Any]]:
    any_filter = False
    candidate_set = set(c.lower() for c in candidate_cols)
    for path in gpkg_paths:
        for layer in layer_names:
            try:
                with fiona.open(path, layer=layer) as src:
                    # Accept WGS84 / no CRS only (defaulting to WGS84); otherwise skip to avoid projection-conversion overhead
                    try:
                        crs_wkt = src.crs_wkt or ""
                    except Exception:
                        crs_wkt = ""
                    if src.crs and str(src.crs).upper() not in ("EPSG:4326", "{'init': 'epsg:4326'}") and "WGS 84" not in crs_wkt:
                        # Could also add on-the-fly reprojection here if needed (skipped for speed)
                        pass

                    # Decide whether to filter by field
                    present_cols = set(k.lower() for k in (src.schema.get("properties", {}) or {}).keys())
                    do_filter = len(candidate_set & present_cols) > 0
                    any_filter = any_filter or do_filter

                    for feat in src:  # Streaming read
                        geom = feat.get("geometry")
                        if not geom:
                            continue
                        # Filter empty coordinates
                        coords = geom.get("coordinates", None)
                        if coords is None or (isinstance(coords, (list, tuple)) and len(coords) == 0):
                            continue
                        if do_filter:
                            props = feat.get("properties") or {}
                            keep = False
                            for col in candidate_cols:
                                v = props.get(col)
                                if v is None:
                                    continue
                                s = str(v)
                                if len(s) > 0 and s.lower() != "none":
                                    keep = True
                                    break
                            if not keep:
                                continue
                        # Value filter (e.g. waterway == 'river' only)
                        # If the outer caller needs this filter, wrap a value-filtering generator in main
                        yield geom
            except Exception:
                continue
    # If no layer has the field to filter on, this is equivalent to no filtering
    if not any_filter:
        # All geometries have already been yielded in the loop
        pass


# Compute the union extent via fiona.bounds, avoiding loading all features into memory
def compute_union_bounds(gpkg_paths: List[str], all_layer_names: List[str]) -> tuple:
    minx, miny, maxx, maxy = (np.inf, np.inf, -np.inf, -np.inf)
    found = False
    for path in gpkg_paths:
        for layer in all_layer_names:
            try:
                with fiona.open(path, layer=layer) as src:
                    b = src.bounds  # (minx, miny, maxx, maxy)
                    if b is None:
                        continue
                    bx1, by1, bx2, by2 = b
                    if not (np.isfinite(bx1) and np.isfinite(by1) and np.isfinite(bx2) and np.isfinite(by2)):
                        continue
                    minx = min(minx, bx1)
                    miny = min(miny, by1)
                    maxx = max(maxx, bx2)
                    maxy = max(maxy, by2)
                    found = True
            except Exception:
                continue
    if not found:
        raise ValueError("No usable target layer found in the input GPKGs")
    return (minx, miny, maxx, maxy)


def main() -> None:
    args = parse_args()
    inputs: List[str] = args.inputs
    output_dir: str = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    RESOLUTION_DEG = float(args.resolution_m) / 111320.0

    # Compute the union extent (across all layers involved)
    all_layer_names = sorted({ln for v in layers_needed.values() for ln in v})
    minx, miny, maxx, maxy = compute_union_bounds(inputs, all_layer_names)

    width = int(math.ceil((maxx - minx) / RESOLUTION_DEG))
    height = int(math.ceil((maxy - miny) / RESOLUTION_DEG))
    width = max(width, 1)
    height = max(height, 1)

    maxx = minx + width * RESOLUTION_DEG
    maxy = miny + height * RESOLUTION_DEG
    transform = from_origin(minx, maxy, RESOLUTION_DEG, RESOLUTION_DEG)

    print(f"Unified extent: ({minx}, {miny}, {maxx}, {maxy}) -> {width}x{height} pixels")

    # Generate the distance map per category (streaming rasterization -> binarization -> EDT)
    for out_name, layer_list in layers_needed.items():
        candidate_cols = non_null_fields_map.get(out_name, [])

        # Rasterize (streaming shapes directly). Wrap a filter here if value-based filtering is needed
        base_iter = iter_shapes_from_layers(inputs, layer_list, candidate_cols)
        vf: Optional[Dict[str, Set[str]]] = value_filters_map.get(out_name)
        if vf:
            vf_lower = {k.lower(): {v.lower() for v in vals} for k, vals in vf.items()}
            def filtered_iter():
                # Reopening is needed to access attributes; iterate the data source again here
                for path in inputs:
                    for layer in layer_list:
                        try:
                            with fiona.open(path, layer=layer) as src:
                                for feat in src:
                                    geom = feat.get("geometry")
                                    if not geom:
                                        continue
                                    coords = geom.get("coordinates", None)
                                    if coords is None or (isinstance(coords, (list, tuple)) and len(coords) == 0):
                                        continue
                                    props = feat.get("properties") or {}
                                    ok = True
                                    for fld, allowed in vf_lower.items():
                                        val = props.get(fld)
                                        if val is None or str(val).lower() not in allowed:
                                            ok = False
                                            break
                                    if not ok:
                                        continue
                                    # Keep the basic non-empty-field filter logic at the same time
                                    if candidate_cols:
                                        present_cols = set(k.lower() for k in (src.schema.get("properties", {}) or {}).keys())
                                        do_filter = len(set(c.lower() for c in candidate_cols) & present_cols) > 0
                                        if do_filter:
                                            keep = False
                                            for col in candidate_cols:
                                                v = props.get(col)
                                                if v is None:
                                                    continue
                                                s = str(v)
                                                if len(s) > 0 and s.lower() != "none":
                                                    keep = True
                                                    break
                                            if not keep:
                                                continue
                                    yield geom
                        except Exception:
                            continue
            shapes_iter = ((geom, 1) for geom in filtered_iter())
        else:
            shapes_iter = ((geom, 1) for geom in base_iter)

        mask = features.rasterize(
            shapes=shapes_iter,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,
        )

        # If everything is 0, output all NaN directly
        if mask.max() == 0:
            out_path = os.path.join(output_dir, f"{out_name}.tif")
            with rasterio.open(
                out_path,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype=np.float32,
                crs="EPSG:4326",
                transform=transform,
                nodata=np.nan,
                compress="DEFLATE",
            ) as dst:
                dst.write(np.full((height, width), np.nan, dtype=np.float32), 1)
            print(f"{out_name}: no features; output all NaN -> {out_path}")
            continue

        # Dilate by 1 pixel in raster space to better capture thin features
        mask = binary_dilation(mask.astype(bool), iterations=1).astype(np.uint8)

        # Distance (km): note we are still on a lon/lat grid, so km is approximate
        inv = (mask == 0)
        dist_deg = distance_transform_edt(inv) * RESOLUTION_DEG
        dist_km = (dist_deg * 111320.0) / 1000.0

        out_path = os.path.join(output_dir, f"{out_name}.tif")
        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=np.float32,
            crs="EPSG:4326",
            transform=transform,
            nodata=np.nan,
            compress="DEFLATE",
        ) as dst:
            dst.write(dist_km.astype(np.float32), 1)

        print(f"{out_name} → {out_path}")

    print("Distance map complete (units: km); extent and pixels are fully aligned.")


if __name__ == "__main__":
    main()
