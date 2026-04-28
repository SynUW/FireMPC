"""
Download BC province's buildings and roads data, in tiled chunks by year.
The previous version downloaded directly without partitioning by year.
"""
import requests
import geopandas as gpd
import pandas as pd
import logging
from shapely.geometry import Polygon, LineString
import time
import os
import math

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Coordinate system (WGS84)
DEFAULT_CRS = "EPSG:4326"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# BC province tile-wise download configuration
BC_BBOX = (48.2, -139.0, 60.0, -114.1)  # Entire BC province boundary
CHUNK_SIZE = 2.0  # Size of each tile (degrees)

def split_bbox(bbox, chunk_size):
    """Split a large bounding box into smaller squares"""
    south, west, north, east = bbox
    chunks = []

    lat_steps = math.ceil((north - south) / chunk_size)
    lon_steps = math.ceil((east - west) / chunk_size)

    for i in range(lat_steps):
        for j in range(lon_steps):
            chunk_south = south + i * chunk_size
            chunk_west = west + j * chunk_size
            chunk_north = min(chunk_south + chunk_size, north)
            chunk_east = min(chunk_west + chunk_size, east)

            # Ensure a minimum valid area
            if chunk_north - chunk_south > 0.1 and chunk_east - chunk_west > 0.1:
                chunks.append((chunk_south, chunk_west, chunk_north, chunk_east))

    return chunks

def build_query(bbox, query_type, date=None):
    """Build an optimized query that can target a specific date"""
    south, west, north, east = bbox

    # If a date was provided, add a date filter
    date_str = f'[date:"{date}"]' if date else ''

    if query_type == "buildings":
        return f"""
        [out:json][timeout:1800]{date_str};
        (
            way["building"]({south},{west},{north},{east});
            relation["building"]({south},{west},{north},{east});
        );
        (._;>;);
        out body geom;
        """
    else:  # roads
        return f"""
        [out:json][timeout:1800]{date_str};
        (
            way["highway"]["highway"!~"footway|cycleway|path"]({south},{west},{north},{east});
        );
        (._;>;);
        out body geom;
        """

def download_chunk(bbox, query_type, date=None, max_retries=3):
    """Download a single tile, optionally for a specific date"""
    for attempt in range(max_retries):
        try:
            query = build_query(bbox, query_type, date)
            response = requests.post(OVERPASS_URL, data={'data': query}, timeout=1800)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Tile download failed (attempt {attempt + 1}): {str(e)}")
            time.sleep(30 * (attempt + 1))
    return None

def process_data(data, data_type):
    """Process raw OSM data"""
    features = []
    for element in data.get('elements', []):
        if element['type'] == 'way' and 'geometry' in element:
            coords = [(n['lon'], n['lat']) for n in element['geometry']]
            if data_type == "buildings" and len(coords) >= 4:
                geom = Polygon(coords) if coords[0] == coords[-1] else LineString(coords)
            else:
                geom = LineString(coords)

            features.append({
                'type': 'Feature',
                'geometry': geom,
                'properties': element.get('tags', {})
            })

    if not features:
        return None

    gdf = gpd.GeoDataFrame.from_features(features, crs=DEFAULT_CRS)
    gdf['data_type'] = data_type
    return gdf

def download_region(region_name, bbox, data_type, date=None):
    """Download the entire region (auto-tiled), optionally for a specific date"""
    chunks = split_bbox(bbox, CHUNK_SIZE)
    all_gdfs = []

    for i, chunk in enumerate(chunks):
        logger.info(f"Downloading {region_name} {data_type} tile {i + 1}/{len(chunks)} for date {date}...")

        data = download_chunk(chunk, data_type, date)
        if data and data['elements']:
            gdf = process_data(data, data_type)
            if gdf is not None:
                all_gdfs.append(gdf)

        time.sleep(10)  # Delay between tiles

    if all_gdfs:
        return gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True))
    return None

def main():
    # Ensure output directory exists
    os.makedirs("../osm_data", exist_ok=True)

    # Define the year range to download, e.g. 2010 to 2020
    years = range(2010, 2021)  # Includes 2010-2020

    # Download data per year
    for year in years:
        # Set the date to the last day of that year
        date = f"{year}-12-31T23:59:59Z"

        for data_type in ["buildings", "roads"]:
            logger.info(f"\n=== Starting BC province download {data_type} for year {year} ===")
            gdf = download_region("British_Columbia", BC_BBOX, data_type, date)

            if gdf is not None:
                output_file = f"osm_data/British_Columbia_{data_type}_{year}.geojson"
                gdf.to_file(output_file, driver='GeoJSON')
                logger.info(f"Successfully saved {len(gdf)} records to {output_file}")
                logger.info(f"Geometry type statistics:\n{gdf.geometry.type.value_counts()}")
            else:
                logger.warning(f"No BC data obtained for {data_type} data for year {year}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Program error: {str(e)}", exc_info=True)