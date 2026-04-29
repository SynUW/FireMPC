from osgeo import gdal
import numpy as np
import os

def read_tif(input_path):
    ds = gdal.Open(input_path)
    if ds is None:
        raise FileNotFoundError(f"Cannot open {input_path}")

    num_bands = ds.RasterCount
    data = []
    nodata_values = []

    for i in range(num_bands):
        band = ds.GetRasterBand(i+1)
        arr = band.ReadAsArray()
        nodata = band.GetNoDataValue()
        nodata_values.append(nodata)
        data.append(arr)

    data = np.stack(data)
    projection = ds.GetProjection()
    geotransform = ds.GetGeoTransform()

    return data, projection, geotransform, nodata_values

def save_tif(output_path, data, projection, geotransform, nodata_values):
    driver = gdal.GetDriverByName('GTiff')
    num_bands, rows, cols = data.shape
    ds = driver.Create(output_path, cols, rows, num_bands, gdal.GDT_Float32)

    ds.SetProjection(projection)
    ds.SetGeoTransform(geotransform)

    for i in range(num_bands):
        band = ds.GetRasterBand(i+1)
        band.WriteArray(data[i])
        if nodata_values[i] is not None:
            band.SetNoDataValue(nodata_values[i])
        band.FlushCache()

    ds = None

def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))

def apply_log_transform(input_path, output_path, signed=False):
    data, projection, geotransform, nodata_values = read_tif(input_path)
    log_data = np.zeros_like(data, dtype=np.float32)

    for i in range(data.shape[0]):
        nodata = nodata_values[i] if nodata_values[i] is not None else -9999
        mask = (data[i] != nodata)

        if signed:
            temp = signed_log1p(data[i][mask])
        else:
            temp = np.log1p(np.maximum(data[i][mask], 0))  # Apply log1p only to non-negative values

        band_out = np.full_like(data[i], nodata, dtype=np.float32)
        band_out[mask] = temp
        log_data[i] = band_out

    save_tif(output_path, log_data, projection, geotransform, nodata_values)
    print(f"Log transform saved to: {output_path}")

if __name__ == "__main__":
    input_tif = "/mnt/raid/user/wildfire_dataset/self_built_materials/all_data_masked/DEM_and_distance_map/Topo_and_distance_map_stack_nodata_homogenized.tif"
    output_tif = "/mnt/raid/user/wildfire_dataset/self_built_materials/all_data_masked/DEM_and_distance_map/Topo_and_distance_map_stack_nodata_homogenized_log.tif"
    apply_log_transform(input_tif, output_tif, signed=True)  # signed=True supports negative values
