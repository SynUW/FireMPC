"""
Clip an OSM .osm.pbf with a shapefile and output the clipped .osm.pbf (using GDAL and osmium).
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile


def check_ogr2ogr() -> None:
    if shutil.which("ogr2ogr") is None:
        print("ogr2ogr not found; please install gdal-bin first: sudo apt-get install -y gdal-bin", file=sys.stderr)
        sys.exit(1)


def check_osmium() -> None:
    if shutil.which("osmium") is None:
        print("osmium not found; please install osmium-tool first: sudo apt-get install -y osmium-tool", file=sys.stderr)
        sys.exit(1)


def run_and_stream(cmd: list) -> int:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in proc.stdout:
            print(line.rstrip())
    finally:
        return proc.wait()


def shapefile_to_geojson_wgs84(
    input_shapefile: str,
    output_geojson: str,
    assume_wgs84: bool = False,
) -> None:
    """Convert a shapefile to a WGS84 GeoJSON. If assume_wgs84=True, assign EPSG:4326 first."""
    cmd = [
        "ogr2ogr",
        "-f",
        "GeoJSON",
        output_geojson,
        input_shapefile,
        "-t_srs",
        "EPSG:4326",
    ]
    if assume_wgs84:
        # If the source has no .prj but coordinates are clearly lon/lat (WGS84), assign EPSG:4326 to it
        cmd += ["-a_srs", "EPSG:4326"]

    ret = run_and_stream(cmd)
    if ret != 0:
        print(f"Shapefile -> GeoJSON conversion failed (exit code {ret})", file=sys.stderr)
        sys.exit(ret)


def clip_pbf_with_polygon(
    input_pbf: str,
    polygon_geojson: str,
    output_pbf: str,
) -> None:
    """Use osmium to clip the PBF using a polygonal GeoJSON."""
    os.makedirs(os.path.dirname(output_pbf) or ".", exist_ok=True)

    cmd = [
        "osmium",
        "extract",
        "-p",
        polygon_geojson,
        input_pbf,
        "-o",
        output_pbf,
        "--overwrite",
    ]
    ret = run_and_stream(cmd)
    if ret != 0:
        print(f"PBF clipping failed (exit code {ret})", file=sys.stderr)
        sys.exit(ret)


def parse_args():
    parser = argparse.ArgumentParser(description="Clip an OSM .osm.pbf with a shapefile and output the clipped .osm.pbf (using GDAL and osmium)")
    parser.add_argument("--input_pbf", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/osm_map/us-west-latest.osm.pbf', help="Input .osm.pbf path")
    parser.add_argument("--clipsrc", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_boundary/british_columnbia_no_crs_boundingBox.shp', help="Shapefile path used for clipping (boundary data)")
    parser.add_argument("--output_pbf", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/osm_map/us-west-latest-clip.osm.pbf', help="Output clipped .osm.pbf path")
    parser.add_argument("--assume_wgs84", action="store_true", help="If the shapefile lacks .prj but its coordinates are valid WGS84 lon/lat, assign EPSG:4326 to it")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    check_ogr2ogr()
    check_osmium()

    # Place the temporary GeoJSON on a RAM disk (if available) for speed
    tmp_dir = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()
    tmp_geojson = os.path.join(tmp_dir, f"clip_polygon_{os.getpid()}.geojson")

    print(f"[1/2] converting clip boundary to WGS84 GeoJSON -> {tmp_geojson}")
    shapefile_to_geojson_wgs84(args.clipsrc, tmp_geojson, assume_wgs84=args.assume_wgs84)

    print(f"[2/2] clipping PBF with polygon -> {args.output_pbf}")
    clip_pbf_with_polygon(args.input_pbf, tmp_geojson, args.output_pbf)

    # Clean up temporary files
    try:
        if os.path.exists(tmp_geojson):
            os.remove(tmp_geojson)
    except Exception:
        pass