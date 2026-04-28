"""
Split the Canada boundary into 200 grid cells and save them as a shapefile,
for subsequent data downloads.
"""
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
import os

def split_shapefile_to_grid(input_shapefile, output_shapefile, grid_size=400):
    """
    Split a shapefile into the specified number of grid cells.
    
    Args:
    input_shapefile: input shapefile path
    output_shapefile: output shapefile path (None means do not save)
    grid_size: total number of grid cells, default 400
    """
    
    print(f"Reading shapefile: {input_shapefile}")
    
    # Read the original shapefile
    try:
        gdf = gpd.read_file(input_shapefile)
        print(f"Successfully read shapefile; contains {len(gdf)} features")
        print(f"Coordinate system: {gdf.crs}")
    except Exception as e:
        print(f"Failed to read shapefile: {e}")
        return None
    
    # Get the bounding box of the whole shapefile
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    minx, miny, maxx, maxy = bounds
    
    print(f"Bounding box: ({minx:.6f}, {miny:.6f}) to ({maxx:.6f}, {maxy:.6f})")
    
    # Compute the grid's row/column counts (as close to square as possible)
    total_area = (maxx - minx) * (maxy - miny)
    cell_area = total_area / grid_size
    cell_size = np.sqrt(cell_area)
    
    # Compute rows and columns
    cols = int(np.ceil((maxx - minx) / cell_size))
    rows = int(np.ceil((maxy - miny) / cell_size))
    
    # Adjust so that the total stays close to the target grid count
    while cols * rows < grid_size:
        if (maxx - minx) / cols > (maxy - miny) / rows:
            cols += 1
        else:
            rows += 1
    
    print(f"Grid configuration: {rows} rows x {cols} columns = {rows * cols} grid cells")
    
    # Compute the actual grid cell size
    cell_width = (maxx - minx) / cols
    cell_height = (maxy - miny) / rows
    
    print(f"Each grid cell size: {cell_width:.6f} × {cell_height:.6f}")
    
    # Create the grid
    grid_polygons = []
    grid_ids = []
    
    print("Creating grid...")
    
    for i in range(rows):
        for j in range(cols):
            # Compute the current grid cell's bounds
            left = minx + j * cell_width
            right = minx + (j + 1) * cell_width
            bottom = miny + i * cell_height
            top = miny + (i + 1) * cell_height
            
            # Create polygon
            polygon = Polygon([
                (left, bottom),
                (right, bottom),
                (right, top),
                (left, top),
                (left, bottom)
            ])
            
            grid_polygons.append(polygon)
            grid_ids.append(f"grid_{i:03d}_{j:03d}")
    
    # Create GeoDataFrame
    grid_gdf = gpd.GeoDataFrame({
        'grid_id': grid_ids,
        'row': [i for i in range(rows) for j in range(cols)],
        'col': [j for i in range(rows) for j in range(cols)],
        'geometry': grid_polygons
    }, crs=gdf.crs)
    
    print(f"Created {len(grid_gdf)} grid cells")
    
    # Save to a new shapefile (if an output file is specified)
    if output_shapefile is not None:
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_shapefile)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            grid_gdf.to_file(output_shapefile)
            print(f"Grid shapefile saved to: {output_shapefile}")
            
            # Print statistics
            print("\n=== Statistics ===")
            print(f"Original shapefile bounds: ({minx:.6f}, {miny:.6f}, {maxx:.6f}, {maxy:.6f})")
            print(f"Grid configuration: {rows} rows x {cols} columns")
            print(f"Total grid cells: {len(grid_gdf)}")
            print(f"Single grid cell size: {cell_width:.6f} × {cell_height:.6f}")
            print(f"Output file: {output_shapefile}")
            
        except Exception as e:
            print(f"Failed to save shapefile: {e}")
            return None
    
    return grid_gdf

def create_intersecting_grid(input_shapefile, output_shapefile, grid_size=400):
    """
    Build grid cells that intersect the original shapefile (keep only intersecting cells).
    
    Args:
    input_shapefile: input shapefile path
    output_shapefile: output shapefile path
    grid_size: target number of grid cells
    """
    
    print(f"Creating grid cells intersecting the original geometry...")
    
    # First, create the full grid
    full_grid = split_shapefile_to_grid(input_shapefile, None, grid_size)
    if full_grid is None:
        print("Failed to create the full grid")
        return None
    
    # Read the original shapefile
    try:
        original_gdf = gpd.read_file(input_shapefile)
    except Exception as e:
        print(f"Failed to read the original shapefile: {e}")
        return None
    
    # Find grid cells that intersect the original geometry
    print("Filtering intersecting grid cells...")
    intersecting_grids = []
    
    for idx, grid_cell in full_grid.iterrows():
        # Check whether the grid cell intersects any original feature
        intersects = original_gdf.geometry.intersects(grid_cell.geometry).any()
        if intersects:
            intersecting_grids.append(grid_cell)
    
    if intersecting_grids:
        # Create a new GeoDataFrame
        intersecting_gdf = gpd.GeoDataFrame(intersecting_grids, crs=full_grid.crs)
        
        # Renumber
        intersecting_gdf['grid_id'] = [f"grid_{i:04d}" for i in range(len(intersecting_gdf))]
        intersecting_gdf.reset_index(drop=True, inplace=True)
        
        # Save
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_shapefile)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            intersecting_gdf.to_file(output_shapefile)
            print(f"Saved {len(intersecting_gdf)} intersecting grid cells to: {output_shapefile}")
            return intersecting_gdf
        except Exception as e:
            print(f"Failed to save intersecting grid: {e}")
            return None
    else:
        print("No grid cells intersect the original geometry")
        return None

# Usage example
if __name__ == "__main__":
    # Configure file paths
    input_file = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_boundary/bc_boundary_without_sea.shp"  # Replace with your input shapefile path
    # output_file = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/Canada_boundary/Canada_Export_grids_200_1.shp"  # output file path
    output_intersecting = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_boundary/bc_boundary_without_sea_4.shp"  # Output file containing only intersecting grid cells
    
    # print("=== Shapefile grid splitting tool ===\n")
    
    # # Method 1: create the full 400 grid cells covering the entire boundary
    # print("Method 1: full grid coverage")
    # grid_result = split_shapefile_to_grid(input_file, output_file, 400)
    
    # print("\n" + "="*50 + "\n")
    
    # Method 2: keep only grid cells intersecting the original geometry
    print("Method 2: build intersecting grid")
    intersecting_result = create_intersecting_grid(input_file, output_intersecting, 4)
    
    print("\n=== Processing complete ===")
    print("Two files have been generated:")
    # print(f"1. {output_file} - the full 400 grid cells")
    print(f"2. {output_intersecting} - contains only grid cells intersecting the original geometry")