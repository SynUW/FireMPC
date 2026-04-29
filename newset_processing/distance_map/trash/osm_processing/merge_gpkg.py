"""
Merge multiple gpkg files produced by extract_osm_thematics.py into a single gpkg, for later use by make_distance_rasters.py.
"""
import argparse
import os
import sys
from typing import List, Optional, Iterable, Set

import geopandas as gpd
from pyproj import CRS
import fiona


def ensure_directory_exists(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)


def read_layer_from_gpkg(gpkg_path: str, layer_name: str) -> gpd.GeoDataFrame:
    if not os.path.isfile(gpkg_path):
        raise FileNotFoundError(f"File not found: {gpkg_path}")
    gdf = gpd.read_file(gpkg_path, layer=layer_name)
    return gdf


def reproject_if_needed(gdf: gpd.GeoDataFrame, target_crs: CRS) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("Input data is missing a CRS; cannot align coordinates. Please assign the correct CRS first.")
    if not CRS.from_user_input(gdf.crs).equals(target_crs):
        return gdf.to_crs(target_crs)
    return gdf


def align_columns(geodataframes: List[gpd.GeoDataFrame]) -> List[gpd.GeoDataFrame]:
    # Unify the field set (keeping geometry)
    all_columns = []
    all_column_set = set()
    for gdf in geodataframes:
        for col in gdf.columns:
            if col == gdf.geometry.name:
                continue
            if col not in all_column_set:
                all_column_set.add(col)
                all_columns.append(col)

    aligned = []
    for gdf in geodataframes:
        cols = [c for c in all_columns if c in gdf.columns]
        # Add missing columns as None
        for missing in (set(all_columns) - set(cols)):
            gdf[missing] = None
        # Reorder so that geometry stays last (managed by GeoPandas)
        ordered = gdf[[*all_columns, gdf.geometry.name]]
        aligned.append(ordered)
    return aligned


def dissolve_geometries(
    gdf: gpd.GeoDataFrame,
    dissolve_by: Optional[str] = None,
) -> gpd.GeoDataFrame:
    if dissolve_by:
        # Aggregate other fields by taking the first non-null value
        return gdf.dissolve(by=dissolve_by, aggfunc="first", dropna=False)
    # No fields: dissolve everything into a single geometry
    tmp = gdf.copy()
    tmp["__all__"] = 1
    res = tmp.dissolve(by="__all__", aggfunc="first", dropna=False)
    res = res.drop(columns=[c for c in res.columns if c == "__all__"], errors="ignore")
    return res


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge same-named layers from multiple GPKGs into one GPKG layer; or group-merge all same-named layers")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=['/mnt/raid/user/wildfire_dataset/self_built_materials/osm_map/canada-thematics.gpkg',
                 '/mnt/raid/user/wildfire_dataset/self_built_materials/osm_map/us-west-thematics.gpkg'],
        help="Input GPKG file path list (space-separated)",
    )
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Group-merge all layers (merge same-named layers into the output's same-named layer)",
    )
    parser.add_argument(
        "--layer",
        required=False,
        default=None,
        help="Layer name to merge; if not provided and --all-layers is not specified, defaults to group-merging all layers by name",
    )
    parser.add_argument(
        "--output",
        default='/mnt/raid/user/wildfire_dataset/self_built_materials/osm_map/canada-us-mosaic-thematics.gpkg',
        help="Output GPKG path",
    )
    parser.add_argument(
        "--out-layer",
        default=None,
        help="Output layer name (default: same as --layer)",
    )
    parser.add_argument(
        "--target-crs",
        default=None,
        help="Target CRS (e.g. EPSG:4326). If not specified, the first input's CRS is used",
    )
    parser.add_argument(
        "--add-source-field",
        default="source_file",
        help="Add a source-filename field in the output (empty string disables)",
    )
    parser.add_argument(
        "--dissolve",
        action="store_true",
        help="Dissolve all features into a single one (or group-dissolve by --dissolve-by field)",
    )
    parser.add_argument(
        "--dissolve-by",
        default=None,
        help="Group-dissolve by this field (requires --dissolve)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_paths: List[str] = args.inputs
    output_path: str = args.output
    add_source_field: Optional[str] = (args.add_source_field or "").strip() or None

    if len(input_paths) == 0:
        raise ValueError("No input GPKG file provided.")

    # Determine whether this is "merge one layer" or "merge all layers"
    merge_all_layers: bool = bool(args.all_layers or not args.layer)

    if not merge_all_layers:
        # ===== Merge a single specified layer =====
        layer_name: str = args.layer  # type: ignore[assignment]
        output_layer_name: str = args.out_layer or layer_name

        # Determine the target CRS
        first_gdf = read_layer_from_gpkg(input_paths[0], layer_name)
        if first_gdf.crs is None and not args.target_crs:
            raise ValueError("The first input layer has no CRS and --target-crs was not provided. Please supply the CRS.")
        if args.target_crs:
            target_crs = CRS.from_user_input(args.target_crs)
        else:
            target_crs = CRS.from_user_input(first_gdf.crs)

        geodataframes: List[gpd.GeoDataFrame] = []
        total = len(input_paths)
        for index, gpkg_path in enumerate(input_paths, start=1):
            sys.stdout.write(f"\r[{index}/{total}] reading {os.path.basename(gpkg_path)} …")
            sys.stdout.flush()
            try:
                gdf = read_layer_from_gpkg(gpkg_path, layer_name)
            except Exception:
                continue
            if gdf.empty:
                continue
            gdf = reproject_if_needed(gdf, target_crs)
            if add_source_field:
                gdf[add_source_field] = os.path.basename(gpkg_path)
            geodataframes.append(gdf)

        sys.stdout.write("\n")

        if not geodataframes:
            raise ValueError("All input layers are empty or unreadable.")

        geodataframes = align_columns(geodataframes)
        merged_gdf = gpd.GeoDataFrame(pd_concat(geodataframes), crs=target_crs)
        if args.dissolve:
            merged_gdf = dissolve_geometries(merged_gdf, args.dissolve_by)

        ensure_directory_exists(output_path)
        # The first write will create the file
        if os.path.exists(output_path):
            os.remove(output_path)
        merged_gdf.to_file(output_path, driver="GPKG", layer=output_layer_name)
        print(f"Output written: {output_path} (layer={output_layer_name})")

    else:
        # ===== Merge all same-named layers =====
        # Compute the union of layer names across all inputs
        layer_names: Set[str] = set()
        for gpkg_path in input_paths:
            try:
                for lyr in fiona.listlayers(gpkg_path):
                    layer_names.add(lyr)
            except Exception:
                continue

        if not layer_names:
            raise ValueError("No layers found in the input files.")

        # Determine the target CRS per layer: use --target-crs or the CRS of the first file that contains the layer
        if os.path.exists(output_path):
            os.remove(output_path)

        names_sorted = sorted(layer_names)
        total_layers = len(names_sorted)
        for layer_idx, layer_name in enumerate(names_sorted, start=1):
            # Find the first file containing the layer to determine the default CRS
            target_crs: Optional[CRS] = None
            if args.target_crs:
                target_crs = CRS.from_user_input(args.target_crs)
            else:
                for p in input_paths:
                    try:
                        probe = read_layer_from_gpkg(p, layer_name)
                    except Exception:
                        continue
                    if probe is not None and probe.crs is not None:
                        target_crs = CRS.from_user_input(probe.crs)
                        break
            if target_crs is None:
                # All inputs lack a CRS; skip this layer
                sys.stdout.write(f"\r[{layer_idx}/{total_layers}] skipping {layer_name} (no CRS)")
                sys.stdout.flush()
                continue

            # Collect all data for this layer and reproject / align
            geodataframes: List[gpd.GeoDataFrame] = []
            for idx, gpkg_path in enumerate(input_paths, start=1):
                sys.stdout.write(
                    f"\r[{layer_idx}/{total_layers}] {layer_name}: read {idx}/{len(input_paths)} {os.path.basename(gpkg_path)} …"
                )
                sys.stdout.flush()
                try:
                    gdf = read_layer_from_gpkg(gpkg_path, layer_name)
                except Exception:
                    continue
                if gdf.empty:
                    continue
                try:
                    gdf = reproject_if_needed(gdf, target_crs)
                except Exception:
                    continue
                if add_source_field:
                    gdf[add_source_field] = os.path.basename(gpkg_path)
                geodataframes.append(gdf)

            sys.stdout.write("\n")

            if not geodataframes:
                continue

            geodataframes = align_columns(geodataframes)
            merged_gdf = gpd.GeoDataFrame(pd_concat(geodataframes), crs=target_crs)
            if args.dissolve:
                merged_gdf = dissolve_geometries(merged_gdf, args.dissolve_by)

            # Write to the same-named layer
            ensure_directory_exists(output_path)
            merged_gdf.to_file(output_path, driver="GPKG", layer=layer_name)

        print(f"Output written: {output_path} (merge {total_layers} layers)")


def pd_concat(gdfs: List[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    # Deferred import to avoid unclear error messages when pandas is not installed in the user's environment
    import pandas as pd

    return pd.concat(gdfs, ignore_index=True)


if __name__ == "__main__":
    main()

