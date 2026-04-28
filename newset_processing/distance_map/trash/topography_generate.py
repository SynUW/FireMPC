"""
Use a shapefile (with a buffer) to clip TIF files
and name each clipped TIF after the shapefile's filename.
"""
import os
import glob
import tempfile
from osgeo import gdal, ogr, osr
import numpy as np


def get_processing_region(shapefile_path):
    """Get the processing extent; if the bounding box is smaller than 1 deg x 1 deg, expand it to 1 deg x 1 deg"""
    try:
        # Open shapefile
        shape_ds = ogr.Open(shapefile_path)
        if shape_ds is None:
            raise ValueError(f"Cannot open shapefile: {shapefile_path}")
        shape_layer = shape_ds.GetLayer()

        # Get the bounds of the entire layer
        extent = shape_layer.GetExtent()
        min_x, max_x, min_y, max_y = extent

        # Compute current width and height
        width_deg = max_x - min_x
        height_deg = max_y - min_y

        # If less than 1 degree, expand to 1 degree
        if width_deg < 1.0 or height_deg < 1.0:
            # Compute center
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2

            # New half-width and half-height
            half_size = 0.5  # Half of 1 degree

            # Compute new bounds
            new_min_x = center_x - half_size
            new_max_x = center_x + half_size
            new_min_y = center_y - half_size
            new_max_y = center_y + half_size

            # Create new rectangular geometry
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(new_min_x, new_min_y)
            ring.AddPoint(new_max_x, new_min_y)
            ring.AddPoint(new_max_x, new_max_y)
            ring.AddPoint(new_min_x, new_max_y)
            ring.AddPoint(new_min_x, new_min_y)

            polygon = ogr.Geometry(ogr.wkbPolygon)
            polygon.AddGeometry(ring)

            return polygon
        else:
            # If already >= 1 degree, return the original bounding box
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(min_x, min_y)
            ring.AddPoint(max_x, min_y)
            ring.AddPoint(max_x, max_y)
            ring.AddPoint(min_x, max_y)
            ring.AddPoint(min_x, min_y)

            polygon = ogr.Geometry(ogr.wkbPolygon)
            polygon.AddGeometry(ring)

            return polygon

    except Exception as e:
        print(f"Error obtaining processing extent: {str(e)}")
        return None
    finally:
        if 'shape_ds' in locals():
            shape_ds = None


def create_temp_shapefile_with_region(region_geom, original_shp, output_shp):
    """Create a temporary shapefile containing the processing extent"""
    try:
        # Get the driver and spatial reference of the original shapefile
        driver = ogr.GetDriverByName('ESRI Shapefile')
        original_ds = ogr.Open(original_shp)
        if original_ds is None:
            raise ValueError(f"Cannot open the original shapefile: {original_shp}")

        srs = original_ds.GetLayer().GetSpatialRef()

        # Create the output shapefile
        if os.path.exists(output_shp):
            driver.DeleteDataSource(output_shp)

        out_ds = driver.CreateDataSource(output_shp)
        out_layer = out_ds.CreateLayer('processing_region', srs, ogr.wkbPolygon)

        # Create fields (copy the first field of the original shapefile)
        in_layer = original_ds.GetLayer()
        in_feature_def = in_layer.GetLayerDefn()
        if in_feature_def.GetFieldCount() > 0:
            field_def = in_feature_def.GetFieldDefn(0)
            out_layer.CreateField(field_def)

        # Create feature
        out_feature = ogr.Feature(out_layer.GetLayerDefn())
        out_feature.SetGeometry(region_geom)

        # Set field values (if any)
        if in_feature_def.GetFieldCount() > 0:
            first_feature = in_layer.GetFeature(0)
            out_feature.SetField(0, first_feature.GetField(0))

        out_layer.CreateFeature(out_feature)

        return True
    except Exception as e:
        print(f"Error creating temporary shapefile: {str(e)}")
        return False
    finally:
        if 'original_ds' in locals():
            original_ds = None
        if 'out_ds' in locals():
            out_ds = None


def clip_raster_with_shapefile(raster_path, shapefile_path, output_path):
    """Use a shapefile to clip a raster"""
    temp_shp = None
    try:
        # Open the raster file
        raster_ds = gdal.Open(raster_path)
        if raster_ds is None:
            raise ValueError(f"Cannot open raster file: {raster_path}")

        # Get the processing extent
        region_geom = get_processing_region(shapefile_path)
        if region_geom is None:
            raise ValueError("Cannot get the processing extent")

        # Create temporary shapefile
        temp_dir = tempfile.gettempdir()
        temp_shp = os.path.join(temp_dir, f"temp_region_{os.getpid()}.shp")

        if not create_temp_shapefile_with_region(region_geom, shapefile_path, temp_shp):
            raise ValueError("Failed to create the temporary processing-extent shapefile")

        # Execute clipping
        options = gdal.WarpOptions(
            cutlineDSName=temp_shp,
            cropToCutline=True,
            dstNodata=np.nan
        )

        result = gdal.Warp(output_path, raster_ds, options=options)
        if result is None:
            raise ValueError("Clipping operation failed")

        return True
    except Exception as e:
        print(f"Error during clipping: {str(e)}")
        return False
    finally:
        # Ensure all datasets are properly closed
        if 'raster_ds' in locals():
            raster_ds = None

        # Clean up temporary files
        if temp_shp and os.path.exists(temp_shp):
            try:
                driver = ogr.GetDriverByName('ESRI Shapefile')
                driver.DeleteDataSource(temp_shp)
                for ext in ['.shx', '.dbf', '.prj']:
                    temp_file = temp_shp.replace('.shp', ext)
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            except Exception as e:
                print(f"Error deleting temporary file: {str(e)}")


def batch_clip_raster_with_shapefiles(raster_path, shapefile_folder, output_folder):
    """Batch raster clipping using shapefiles"""
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all shapefiles
    shapefiles = glob.glob(os.path.join(shapefile_folder, '*.shp'))

    total = len(shapefiles)
    success = 0

    for i, shp_path in enumerate(shapefiles, 1):
        # Derive the name from the shapefile path
        shp_name = os.path.splitext(os.path.basename(shp_path))[0]

        # Set output path
        output_path = os.path.join(output_folder, f"{shp_name}.tif")

        print(f"Processing ({i}/{total}): {shp_name}")

        try:
            # Execute clipping
            if clip_raster_with_shapefile(raster_path, shp_path, output_path):
                print(f"Successfully clipped and saved to: {output_path}")
                success += 1
            else:
                print(f"Process {shp_name} failed")
        except Exception as e:
            print(f"Process {shp_name} error: {str(e)}")

    print(f"\nProcessing complete! Succeeded: {success}/{total}")


if __name__ == '__main__':
    # Configure paths (modify according to actual setup)
    input_raster = r"D:\wildfire_dataset\self_built\drivers\topographies.tif"  # GeoTIFF file to be clipped
    shapefile_dir = r"D:\wildfire_dataset\self_built\GlobFire Fire Perimeters_BC_Filtered_FierEventsDate"  # Folder containing the shapefiles
    output_dir = r"D:\wildfire_dataset\self_built\drivers\Topography_clip"  # Output folder

    # Execute batch clipping
    batch_clip_raster_with_shapefiles(input_raster, shapefile_dir, output_dir)
