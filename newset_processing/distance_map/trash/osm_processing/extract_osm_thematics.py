"""
Extract thematic layers from a pre-clipped OSM .osm.pbf into a single GeoPackage (buildings, water, roads, power).
"""
import argparse
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Tuple


def check_ogr2ogr() -> None:
    if shutil.which("ogr2ogr") is None:
        print("ogr2ogr not found; please install gdal-bin first: sudo apt-get install -y gdal-bin", file=sys.stderr)
        sys.exit(1)


def run_and_stream(cmd: List[str]) -> int:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in proc.stdout:
            print(line.rstrip())
    finally:
        return proc.wait()


def ogr2ogr_extract(
    input_pbf: str,
    output_gpkg: str,
    source_layer: str,
    output_layer_name: str,
    where_clause: str,
    disable_spatial_index: bool,
    transaction_size: int,
    is_first_layer: bool,
) -> int:
    cmd: List[str] = [
        "ogr2ogr",
        "-f",
        "GPKG",
    ]

    if not is_first_layer:
        cmd += ["-update"]

    # Overwrite same-named layer (does not affect other layers)
    cmd += ["-overwrite"]

    if disable_spatial_index:
        cmd += ["-lco", "SPATIAL_INDEX=NO"]

    if transaction_size and transaction_size > 0:
        cmd += ["-gt", str(transaction_size)]

    cmd += [
        "-nln",
        output_layer_name,
        "-progress",
        output_gpkg,
        input_pbf,
        source_layer,
        "-where",
        where_clause,
    ]

    return run_and_stream(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract thematic layers from a pre-clipped OSM .osm.pbf into a single GeoPackage (buildings, water, roads, power)"
    )
    parser.add_argument(
        "--input_pbf",
        default="/mnt/raid/user/wildfire_dataset/self_built_materials/osm_map/canada-latest-clip.osm.pbf",
        help="Input (already-clipped) .osm.pbf path",
    )
    parser.add_argument(
        "--output_gpkg",
        default="/mnt/raid/user/wildfire_dataset/self_built_materials/osm_map/canada-thematics.gpkg",
        help="Output GeoPackage file path (contains multiple layers)",
    )
    parser.add_argument(
        "--disable_spatial_index",
        action="store_true",
        help="Disable GPKG spatial index during writing (faster); create it later if needed",
    )
    parser.add_argument(
        "--gt",
        dest="transaction_size",
        type=int,
        default=65536,
        help="Transaction batch size (-gt), e.g. 65536, to speed up writes",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="If the output .gpkg already exists, delete and recreate it",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_ogr2ogr()

    if os.path.exists(args.output_gpkg):
        if args.force:
            print(f"Deleting existing output file: {args.output_gpkg}")
            try:
                os.remove(args.output_gpkg)
            except Exception as exc:
                print(f"Delete failed: {exc}", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"Output file exists; will overwrite same-named layers in it: {args.output_gpkg}")

    # Define themes: (source layer, output layer name, filter condition)
    thematics: Dict[str, Tuple[str, str]] = {
        # Buildings (polygons)
        "buildings": ("multipolygons", "building IS NOT NULL OR other_tags LIKE '%\"building\"=>%'"),
        # Water: lines (rivers / waterways)
        "water_lines": (
            "lines",
            "waterway IN ('river','stream','canal','ditch','drain')",
        ),
        # Water: polygons (lakes / reservoirs / wetlands)
        "water_areas": (
            "multipolygons",
            "natural='water' OR landuse='reservoir' OR natural='wetland'",
        ),
        # Roads (all categories)
        "roads": ("lines", "highway IS NOT NULL"),
        # Power lines (matched only in other_tags)
        "power_lines": (
            "lines",
            "other_tags LIKE '%\"power\"=>\"line\"%' OR other_tags LIKE '%\"power\"=>\"minor_line\"%' OR other_tags LIKE '%\"power\"=>\"cable\"%'",
        ),
        # Substations (polygons, matched only in other_tags)
        "power_substations": (
            "multipolygons",
            "other_tags LIKE '%\"power\"=>\"substation\"%'",
        ),
        # Towers / poles / substations (points, matched only in other_tags)
        "power_points": (
            "points",
            "other_tags LIKE '%\"power\"=>\"tower\"%' OR other_tags LIKE '%\"power\"=>\"pole\"%' OR other_tags LIKE '%\"power\"=>\"substation\"%'",
        ),
    }

    # Stable order for clean progress display
    order: List[str] = [
        "buildings",
        "water_lines",
        "water_areas",
        "roads",
        "power_lines",
        "power_substations",
        "power_points",
    ]

    total = len(order)
    for idx, name in enumerate(order, start=1):
        source_layer, where_clause = thematics[name]
        print(f"[{idx}/{total}] extracting {name} -> {args.output_gpkg}")
        ret = ogr2ogr_extract(
            input_pbf=args.input_pbf,
            output_gpkg=args.output_gpkg,
            source_layer=source_layer,
            output_layer_name=name,
            where_clause=where_clause,
            disable_spatial_index=args.disable_spatial_index,
            transaction_size=args.transaction_size,
            is_first_layer=(idx == 1 and not os.path.exists(args.output_gpkg)),
        )
        if ret != 0:
            print(f"Layer {name} extraction failed (exit code {ret})", file=sys.stderr)
            sys.exit(ret)

    print("All thematic layers generated.")


if __name__ == "__main__":
    main()