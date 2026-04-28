"""
Use GDAL to rename band names
"""
import os
from osgeo import gdal


def rename_bands(input_dir, output_dir):
    """
    Rename band names of GeoTIFF files.
    :param input_dir: input directory containing the TIFFs to process
    :param output_dir: output directory where modified files are saved
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define new band names
    band_names = ["Aspect", "DEM", "HillShade", "Slope"]

    # Iterate over all TIFF files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.tif'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            print(f"Processing: {filename}")

            try:
                # Open the original file
                ds = gdal.Open(input_path, gdal.GA_ReadOnly)
                if ds is None:
                    print(f"Cannot open file: {input_path}")
                    continue

                # Check band count
                num_bands = ds.RasterCount
                if num_bands < 4:
                    print(f"File {filename} has only {num_bands} bands; at least 4 required")
                    ds = None
                    continue

                # Create output file
                driver = gdal.GetDriverByName('GTiff')
                out_ds = driver.CreateCopy(output_path, ds, 0)

                # Set band name
                for i in range(1, 5):  # Band index is 1-based
                    band = out_ds.GetRasterBand(i)
                    band.SetDescription(band_names[i - 1])
                    band.FlushCache()

                print(f"Successfully processed: {filename}")

            except Exception as e:
                print(f"Processing file {filename} error: {str(e)}")
            finally:
                # Make sure the dataset is closed
                if 'ds' in locals():
                    ds = None
                if 'out_ds' in locals():
                    out_ds = None


if __name__ == '__main__':
    # Configure paths (modify according to actual setup)
    input_directory = r"D:\wildfire_dataset\self_built\drivers\Topography_everyday"  # Input TIFF file directory
    output_directory = r"D:\wildfire_dataset\self_built\drivers\Topography_everyday_rename"  # Output directory

    # Execute band renaming
    rename_bands(input_directory, output_directory)