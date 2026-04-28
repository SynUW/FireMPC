#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-dataset generator -- year-range version (keeps the original H5 structure; reference mask + tile streaming writes).

Global valid pixels: derived from band 1 of a random GeoTIFF inside the reference directory (default unifiedEntireCan/ERA5_merged),
deciding spatial validity by NoData / NaN only (no all-drivers cross-date AND/OR aggregation).
Pass-1 has been removed (no second filter by full-year channel x time ratio); per-driver value-range sampling has been removed.

Storage: write uint8 to HDF5; source rasters must be integer-typed (not float), otherwise abort with an error. No scaling or float -> integer conversion is performed.

Performance (matches the reference version, keeping the no-Pass-1 design):
- _build_day_tile_stack_window: parallel GDAL windowed reads across drivers (ThreadPoolExecutor)
- Tile: per-day prefetch_days; each batch reads with tile_read_workers in parallel; the background prefetches the "next batch" while the current batch is written to H5
- Years run strictly serially (avoiding NFS contention between "creating next year's many datasets" and writing the current year's tiles)
- H5 chunks=(C, T) - one chunk for the whole time series, reducing NFS write count
- Compression lzf (faster than gzip-6 with comparable size)
- HDF5 rdcc_nbytes=512MB to enlarge the chunk cache
- Defaults: max_workers=32, prefetch_days=32
"""

import os
import h5py
import numpy as np
import glob
from datetime import datetime, timedelta
from tqdm import tqdm
import logging

# tqdm shows only "description + numeric progress", without rendering the middle bar
_TQDM_NUMERIC = {"bar_format": "{desc}: {n_fmt}/{total_fmt}"}
from functools import lru_cache
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import ast
from collections import OrderedDict
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_REFERENCE_VALID_MASK_DIR = (
    '/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/'
    'drivers/processed/spatial_filled_5xdownsampled_merged_unifiedEntireCan/ERA5_merged'
)

class YearOnlyDatasetGenerator:
    def __init__(self, data_dir, output_dir, target_years=[2024],
                 max_workers=32, batch_size=10000, max_aggregation_dates=10, force_retrain=False,
                 prefetch_days=32, tile_read_workers=8,
                 reference_valid_mask_dir=None, reference_valid_mask_seed=None):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.target_years = target_years if isinstance(target_years, list) else [target_years]
        self.max_workers = max(1, int(max_workers))
        self.batch_size = batch_size  # Backward-compatibility parameter
        self.max_aggregation_dates = max_aggregation_dates  # Maximum number of dates when aggregating
        self.force_retrain = force_retrain  # Whether to force reprocessing
        self.prefetch_days = max(1, int(prefetch_days))
        self.tile_read_workers = max(1, int(tile_read_workers))
        self.reference_valid_mask_dir = (
            reference_valid_mask_dir
            if reference_valid_mask_dir is not None
            else DEFAULT_REFERENCE_VALID_MASK_DIR
        )
        self.reference_valid_mask_seed = reference_valid_mask_seed

        # Keep consistent with the existing order
        self.driver_order = [
            'MCD14A1_combined_5xMaximumDown_merged_binary', # checked and fixed  
            # 'MCD14A1_withoutMask_Binary_downsampled_masked', # checked and fixed
            'ERA5_merged', # checked and fixed
            'FWI_merged', # checked and fixed
            'VPD_merged',
            
            'OSMs_downsampled_tsfilled',  # checked
            'Worldpop_tsfilled',  # checked
            'DEMs_downsampled_unified_tsfilled',
            
            'NDVI_EVI_QAapplied_spatialFilled_downsampled_merged_float',
            'MCD15A3H_QAapplied_spatialFilled_downsampled_merged', # checked
            
            'MCD09GA_b1237_QAapplied_spatialFilled_downsampled_merged',  
            'MCD11A1_QAapplied_spatialFilled_downsampled_merged',
            
            'MCD09CMG_QAapplied_spatialFilled_merged',
            'MCD12Q1_mosaic_downsampled_right_tsfilled'
        ]

        os.makedirs(output_dir, exist_ok=True)
        self.driver_dirs = self._get_driver_directories()
        self.total_channels, self.channel_mapping = self._calculate_channels_and_mapping()

        # Runtime cache
        self._global_date_mapping = None
        self._global_valid_pixels = None
        self._global_geo_reference = None

        logger.info(f"Data directory: {data_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Target years: {self.target_years}")
        logger.info(f"Found {len(self.driver_dirs)} drivers")
        logger.info(f"Total channels: {self.total_channels}")
        logger.info(f"Parallel threads: {self.max_workers}")
        logger.info(f"Prefetch days (within tile): {self.prefetch_days}")
        logger.info(f"Tile parallel-prefetch count: {self.tile_read_workers}")
        logger.info(f"Force reprocess: {self.force_retrain}")
        logger.info(f"Global spatial mask reference directory: {self.reference_valid_mask_dir}")

    @staticmethod
    def _exit_if_float_raster(arr, context: str):
        """Driver write path: abort immediately on float rasters (no float -> integer conversion of any kind)."""
        if arr is None:
            return
        if np.issubdtype(arr.dtype, np.floating):
            logger.error(
                "Refusing to write: driver data is a float type (integer rasters are required so they can be written as uint8 as-is). %s dtype=%s",
                context,
                arr.dtype,
            )
            raise SystemExit(1)

    # ---------------- Basic utilities ----------------
    def _get_driver_directories(self):
        driver_dirs = {}
        for root, _, files in os.walk(self.data_dir):
            if any(f.endswith('.tif') for f in files):
                rel_path = os.path.relpath(root, self.data_dir)
                if rel_path == '.':
                    continue
                rel_path = rel_path.replace(os.sep, '/')
                driver_dirs[rel_path] = root
        logger.info("Found driver directory:")
        for name, path in driver_dirs.items():
            logger.info(f"  {name}: {path}")
        return driver_dirs

    def _safe_sample_file(self, driver_dir):
        files = glob.glob(os.path.join(driver_dir, '*.tif'))
        return files[0] if files else None

    def _calculate_channels_and_mapping(self):
        """Compute total number of channels and build channel mapping"""
        total_channels = 0
        channel_mapping = {}

        for driver_name in self.driver_order:
            if driver_name not in self.driver_dirs:
                logger.warning(f"Missing driver directory: {driver_name} (will fill these channels with NaN)")
                continue

            sample_file = self._safe_sample_file(self.driver_dirs[driver_name])
            if not sample_file:
                logger.warning(f"Driver directory has no samples: {driver_name} (will fill these channels with NaN)")
                continue

            data, _, _ = self._load_single_file_with_gdal(sample_file)
            if data is None:
                logger.warning(f"Cannot read sample: {driver_name} (will fill these channels with NaN)")
                continue

            channels = data.shape[0] if data.ndim == 3 else 1
            start_idx = total_channels
            end_idx_exclusive = start_idx + channels

            channel_mapping[driver_name] = {
                'start_idx': start_idx,
                'channels' : channels,
                'end_idx'  : end_idx_exclusive
            }

            logger.info(
                f"  {driver_name}: {channels} channels (index {start_idx}-{end_idx_exclusive - 1})"
            )

            total_channels = end_idx_exclusive

        return total_channels, channel_mapping

    def _check_existing_years(self):
        """Check existing yearly data in the output directory"""
        existing_years = set()
        
        if not os.path.exists(self.output_dir):
            logger.info("Output directory does not exist, a new one will be created")
            return existing_years
        
        # Scan H5 files in the output directory
        for filename in os.listdir(self.output_dir):
            if filename.endswith('_year_dataset.h5'):
                try:
                    # Extract year (format: YYYY_year_dataset.h5)
                    year_str = filename.split('_')[0]
                    year = int(year_str)
                    if 2000 <= year <= 2024:  # Reasonable year range
                        existing_years.add(year)
                except (ValueError, IndexError):
                    continue
        
        if existing_years:
            logger.info(f"Existing yearly data found: {sorted(existing_years)}")
        else:
            logger.info("No existing yearly data found")
        
        return existing_years

    def _filter_target_years(self):
        """Filter target years based on existing data"""
        if self.force_retrain:
            logger.info("Force-reprocess mode: all target years will be processed")
            return self.target_years
        
        existing_years = self._check_existing_years()
        remaining_years = [year for year in self.target_years if year not in existing_years]
        
        if not remaining_years:
            logger.info("All target years already have data; nothing to process")
            return []
        
        skipped_years = [year for year in self.target_years if year in existing_years]
        if skipped_years:
            logger.info(f"Skip existing year: {skipped_years}")
        
        logger.info(f"Years to process: {remaining_years}")
        return remaining_years

    @lru_cache(maxsize=1000)
    def _get_date_from_filename(self, filename):
        try:
            import re
            m = re.search(r'(\d{4})_(\d{2})_(\d{2})', filename)
            if m:
                y, M, d = m.groups()
                return datetime(int(y), int(M), int(d))
        except Exception as e:
            logger.debug(f"Failed to extract date from filename {filename}: {e}")
        return None

    def _load_single_file_with_gdal(self, file_path, block_size=2048):
        """
        Optimized file reader with block-wise reading to reduce memory usage.
        Also retrieves the image's NoData value information.
        """
        try:
            from osgeo import gdal
            ds = gdal.Open(file_path, gdal.GA_ReadOnly)
            if ds is None:
                return None, None, None
            
            bands = ds.RasterCount
            width, height = ds.RasterXSize, ds.RasterYSize
            
            # Get NoData value info
            nodata_values = []
            for i in range(1, bands + 1):
                band = ds.GetRasterBand(i)
                nodata_value = band.GetNoDataValue()
                nodata_values.append(nodata_value)
            
            # For large files, use block-wise reading
            if width * height > 10000000:  # More than 10M pixels
                logger.debug(f"Using block-wise read for large file: {file_path}")
                
                if bands == 1:
                    # Block-wise read for a single band
                    band = ds.GetRasterBand(1)
                    arr = np.zeros((height, width), dtype=band.ReadAsArray(0, 0, 1, 1).dtype)
                    
                    for y in range(0, height, block_size):
                        y_end = min(y + block_size, height)
                        for x in range(0, width, block_size):
                            x_end = min(x + block_size, width)
                            block = band.ReadAsArray(x, y, x_end - x, y_end - y)
                            arr[y:y_end, x:x_end] = block
                else:
                    # Multi-band block-wise read (dtype matches the band to avoid loading integers into float32)
                    band0 = ds.GetRasterBand(1)
                    dt0 = band0.ReadAsArray(0, 0, 1, 1).dtype
                    arr = np.zeros((bands, height, width), dtype=dt0)
                    for i in range(1, bands + 1):
                        band = ds.GetRasterBand(i)
                        for y in range(0, height, block_size):
                            y_end = min(y + block_size, height)
                            for x in range(0, width, block_size):
                                x_end = min(x + block_size, width)
                                block = band.ReadAsArray(x, y, x_end - x, y_end - y)
                                arr[i-1, y:y_end, x:x_end] = block
            else:
                # Read small files directly
                if bands == 1:
                    arr = ds.GetRasterBand(1).ReadAsArray()
                else:
                    arr = np.array([ds.GetRasterBand(i).ReadAsArray() for i in range(1, bands+1)])
            
            geo_info = {
                'geotransform': ds.GetGeoTransform(),
                'projection': ds.GetProjection(),
                'raster_size': (width, height),
                'raster_count': bands,
                'nodata_values': nodata_values
            }
            ds = None
            return arr, geo_info, nodata_values
            
        except Exception as e:
            logger.error(f"Failed to load file: {file_path}, error: {e}")
            return None, None, None

    def _get_all_required_dates(self, target_year):
        start = datetime(target_year, 1, 1)
        end   = datetime(target_year, 12, 31)
        dates = []
        cur = start
        while cur <= end:
            dates.append(cur)
            cur += timedelta(days=1)
        logger.info(f"Year {target_year} date range: {start:%Y-%m-%d} to {end:%Y-%m-%d}")
        logger.info(f"Total days: {len(dates)}")
        return dates

    def _build_global_date_mapping(self):
        logger.info("Building the global date-to-file mapping (all drivers, all dates)...")
        mp = {}
        with tqdm(total=len(self.driver_dirs), desc="Scanning drivers", **_TQDM_NUMERIC) as pbar:
            for driver_name, driver_dir in self.driver_dirs.items():
                pbar.set_description(f"Scan: {driver_name}")
                for fp in glob.glob(os.path.join(driver_dir, '*.tif')):
                    dt = self._get_date_from_filename(os.path.basename(fp))
                    if dt is None:
                        continue
                    mp.setdefault(dt, {})[driver_name] = fp
                pbar.update(1)
        logger.info(f"Mapped {len(mp)} dates")
        return mp

    # =========== Valid-mask utilities ===========
    @staticmethod
    def _valid_mask_from_array(arr, invalid_values=(-9999, 255, -32768), image_nodata_values=None, driver_name=None):
        """
        Use only the image's built-in NoData (no more preset invalid values).
        If driver_name is provided and that driver has a valid value range, constrain by that driver's range.
        Multi-band: judge by the first channel (consistent with the original style).
        
        Args:
            arr: input array
            invalid_values: backward-compat parameter, no longer used
            image_nodata_values: list of NoData values from the image
            driver_name: driver name (optional); if provided and the driver has a range, that range is used
        """
        if arr is None:
            return None
        if arr.ndim == 3:
            arr = arr[0]
            # If image NoData values are available, use the first band's (index 0) NoData value
            image_nodata = image_nodata_values[0] if image_nodata_values and len(image_nodata_values) > 0 else None
        else:
            # Single-band image: use the first (and only) NoData value
            image_nodata = image_nodata_values[0] if image_nodata_values and len(image_nodata_values) > 0 else None
            
        # Based on the image's built-in NoData
        if np.issubdtype(arr.dtype, np.floating):
            mask = ~np.isnan(arr)
            if image_nodata is not None and not np.isnan(image_nodata):
                mask &= (arr != image_nodata)
        else:
            if image_nodata is not None:
                mask = (arr != image_nodata)
            else:
                mask = np.ones_like(arr, dtype=bool)

        # Overlay this driver's valid value range (if available and driver name is provided)
        if driver_name is not None:
            driver_ranges = getattr(YearOnlyDatasetGenerator, "_GLOBAL_VALUE_RANGE_IN_EFFECT", None)
            if driver_ranges is not None and isinstance(driver_ranges, dict):
                driver_range = driver_ranges.get(driver_name)
                if driver_range is not None:
                    vmin, vmax = driver_range
                    if vmin is not None:
                        mask &= (arr >= vmin)
                    if vmax is not None:
                        mask &= (arr <= vmax)
        return mask

    # ---------------- Global valid pixels (reference: ERA5 band 1) ----------------
    def _get_global_valid_pixels(self, all_date_to_file):
        """
        Pick a random GeoTIFF from reference_valid_mask_dir and use band 1's NoData / NaN to
        define spatial validity (driver_name=None, no value-range constraints).
        all_date_to_file is kept only for signature compatibility and is unused here.
        """
        import random

        ref_dir = self.reference_valid_mask_dir
        if not ref_dir or not os.path.isdir(ref_dir):
            raise RuntimeError(f"Reference valid-mask directory does not exist: {ref_dir}")

        files = glob.glob(os.path.join(ref_dir, '*.tif'))
        if not files:
            raise RuntimeError(f"No .tif files under the reference valid-mask directory: {ref_dir}")

        rng = random.Random(self.reference_valid_mask_seed)
        sample_fp = rng.choice(files)
        logger.info(
            "Building the global spatial valid mask from band 1 of the reference raster (single scene, no intersection with other drivers/dates): "
            f"{sample_fp}"
        )

        arr, geo_reference, nodata_vals = self._load_single_file_with_gdal(sample_fp)
        if arr is None or geo_reference is None:
            raise RuntimeError(f"Cannot read reference raster: {sample_fp}")

        if arr.ndim == 3:
            band0 = arr[0]
            nd0 = nodata_vals[0] if nodata_vals else None
        else:
            band0 = arr
            nd0 = nodata_vals[0] if nodata_vals else None

        mask = YearOnlyDatasetGenerator._valid_mask_from_array(
            band0,
            image_nodata_values=[nd0],
            driver_name=None,
        )
        if mask is None:
            raise RuntimeError("Cannot build a valid mask from the reference raster")

        width, height = geo_reference['raster_size']
        height, width = int(height), int(width)
        if mask.shape != (height, width):
            raise RuntimeError(
                f"Reference mask shape {mask.shape} and raster size (height={height}, width={width}) mismatch"
            )

        rows, cols = np.where(mask)
        valid_pixels = [(int(r), int(c)) for r, c in zip(rows, cols)]
        logger.info(f"[Global] reference-mask valid pixel count: {len(valid_pixels)}")
        return valid_pixels, geo_reference

    # ---------------- Write (pixels inside the reference mask) ----------------
    class _DSCache(OrderedDict):
        def __init__(self, h5, capacity=4096):
            super().__init__()
            self.h5 = h5
            self.capacity = capacity
        def get_ds(self, name):
            ds = super().get(name)
            if ds is not None:
                self.move_to_end(name)
                return ds
            ds = self.h5[name]
            super().__setitem__(name, ds)
            if len(self) > self.capacity:
                self.popitem(last=False)
            return ds

    def _gather_vals_for_day(self, date_files, rows_keep, cols_keep):
        C = self.total_channels
        Pk = rows_keep.shape[0]
        vals = np.full((C, Pk), 0, dtype=np.uint8)

        def load_one(driver_name):
            fp = date_files.get(driver_name, None)
            if fp is None:
                return driver_name, None
            arr, _, _ = self._load_single_file_with_gdal(fp)
            return driver_name, arr

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {ex.submit(load_one, dn): dn for dn in self.driver_order}
            for fut in as_completed(futs):
                dn = futs[fut]
                info = self.channel_mapping.get(dn)
                if info is None:
                    continue
                s, e = info['start_idx'], info['end_idx']
                try:
                    _, arr = fut.result()
                except Exception as e:
                    logger.error(f"[{dn}] read failed: {e}")
                    arr = None
                if arr is None:
                    continue

                fp = date_files.get(dn)
                self._exit_if_float_raster(arr, f"[{dn}] {fp}" if fp else f"[{dn}]")

                if arr.ndim == 2:
                    # Save raw values directly
                    processed_data = arr[rows_keep, cols_keep]
                    vals[s, :] = processed_data
                elif arr.ndim == 3:
                    sub = arr[:, rows_keep, cols_keep]
                    # Save raw values directly
                    processed_sub = sub
                    vals[s:e, :] = processed_sub
                del arr
        return vals

    def _build_year_generation_context(self, target_year):
        """
        Return None if this year should be skipped; otherwise return the context dict needed for writing (excluding any open file handles).
        Assumes _global_date_mapping and the global valid pixels already exist.
        """
        output_file = os.path.join(self.output_dir, f'{target_year}_year_dataset.h5')
        if os.path.exists(output_file) and not self.force_retrain:
            logger.info(f"Year {target_year} dataset already exists; skipping: {output_file}")
            return None

        valid_pixels = self._global_valid_pixels
        geo_reference = self._global_geo_reference
        if not geo_reference:
            logger.error("Cannot determine image size information")
            return None

        all_dates = self._get_all_required_dates(target_year)
        width, height = geo_reference['raster_size']
        height, width = int(height), int(width)

        rows_all = np.array([r for r, _ in valid_pixels], dtype=np.int32)
        cols_all = np.array([c for _, c in valid_pixels], dtype=np.int32)
        P_all = rows_all.shape[0]
        T = len(all_dates)
        C = self.total_channels
        rows = rows_all
        cols = cols_all
        P = int(rows.shape[0])
        if P == 0:
            logger.error("Global valid-pixel set is empty; aborting")
            return None

        tile_h, tile_w = 512, 512
        tiles = []
        for y0, y1, x0, x1 in self._iter_tiles(height, width, tile_h, tile_w):
            mask = (rows >= y0) & (rows < y1) & (cols >= x0) & (cols < x1)
            idxs = np.where(mask)[0]
            if idxs.size == 0:
                continue
            local_rows = (rows[idxs] - y0).astype(np.int32, copy=False)
            local_cols = (cols[idxs] - x0).astype(np.int32, copy=False)
            tiles.append((y0, y1, x0, x1, idxs, local_rows, local_cols))

        return {
            'output_file': output_file,
            'target_year': target_year,
            'all_dates': all_dates,
            'rows': rows,
            'cols': cols,
            'tiles': tiles,
            'C': C,
            'T': T,
            'P': P,
            'P_all': P_all,
            'geo_reference': geo_reference,
        }

    def _h5_open_kwargs(self):
        return dict(
            libver='latest',
            rdcc_nbytes=512 * 1024 * 1024,
            rdcc_nslots=100003,
        )

    def _create_h5_skeleton_only(self, ctx):
        """Create the H5 file, write file-level attributes, and create all per-pixel datasets (no tile data written yet)."""
        output_file = ctx['output_file']
        target_year = ctx['target_year']
        all_dates = ctx['all_dates']
        rows, cols = ctx['rows'], ctx['cols']
        C, T, P, P_all = ctx['C'], ctx['T'], ctx['P'], ctx['P_all']
        geo_reference = ctx['geo_reference']

        logger.info(f"[{target_year}] creating output file and building pixel datasets: {output_file}")
        with h5py.File(output_file, 'w', **self._h5_open_kwargs()) as h5_file:
            h5_file.attrs['year'] = str(target_year)
            h5_file.attrs['total_time_steps'] = T
            h5_file.attrs['total_channels'] = C
            h5_file.attrs['data_format'] = 'channels_x_time'
            h5_file.attrs['data_type'] = 'year_pixel_time_series'
            h5_file.attrs['value_dtype'] = 'uint8'
            h5_file.attrs['driver_names'] = self.driver_order
            h5_file.attrs['total_pixels'] = int(P_all)
            h5_file.attrs['processed_pixels'] = int(P)
            h5_file.attrs['start_date'] = all_dates[0].strftime('%Y-%m-%d')
            h5_file.attrs['end_date'] = all_dates[-1].strftime('%Y-%m-%d')
            for driver_name, channel_info in self.channel_mapping.items():
                h5_file.attrs[f'channel_mapping_{driver_name}'] = (
                    f"{channel_info['start_idx']}-{channel_info['end_idx']-1}"
                )
            if geo_reference:
                h5_file.attrs['geotransform'] = geo_reference['geotransform']
                h5_file.attrs['projection'] = geo_reference['projection']
                h5_file.attrs['raster_size'] = geo_reference['raster_size']
                h5_file.attrs['raster_count'] = geo_reference['raster_count']

            for r, c in tqdm(
                zip(rows, cols), total=P, desc=f"{target_year} creating pixel datasets", **_TQDM_NUMERIC
            ):
                name = f"{int(r)}_{int(c)}"
                ds = h5_file.create_dataset(
                    name,
                    shape=(C, T),
                    dtype=np.uint8,
                    chunks=(C, T),
                    compression='lzf',
                    shuffle=True,
                )
                ds.attrs['pixel_coord'] = (int(r), int(c))
                ds.attrs['data_shape'] = f"{C}x{T}"
            h5_file.flush()

    def _read_tile_batch(self, batch, all_dates):
        """Read a batch of tiles into a buffer list in parallel (order matches the batch)."""
        if not batch:
            return []
        n = len(batch)
        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = [
                pool.submit(self._read_one_tile_to_buffer, pack, all_dates) for pack in batch
            ]
            return [f.result() for f in futures]

    def _write_year_tile_data(self, ctx):
        """Write tile-by-tile data into an existing-skeleton H5 (opened for append)."""
        output_file = ctx['output_file']
        target_year = ctx['target_year']
        all_dates = ctx['all_dates']
        rows, cols = ctx['rows'], ctx['cols']
        tiles = ctx['tiles']
        C, T, P, P_all = ctx['C'], ctx['T'], ctx['P'], ctx['P_all']

        n_tiles = len(tiles)
        w_read = min(self.tile_read_workers, n_tiles) if n_tiles else 1
        logger.info(
            f"[{target_year}] number of tiles to process: {n_tiles} (per-batch parallel reads <= {w_read}；"
            "Prefetch the next batch in the background while writing the current batch, to overlap NFS and H5 time)"
        )

        with h5py.File(output_file, 'a', **self._h5_open_kwargs()) as h5_file:
            with tqdm(total=n_tiles, desc=f"{target_year} Tile write", **_TQDM_NUMERIC) as pbar:
                ti = 0
                next_batch_fut = None
                with ThreadPoolExecutor(max_workers=1) as batch_read_pipe:
                    while ti < n_tiles:
                        batch_end = min(ti + w_read, n_tiles)
                        batch = tiles[ti:batch_end]

                        if next_batch_fut is not None:
                            bufs = next_batch_fut.result()
                            next_batch_fut = None
                        else:
                            bufs = self._read_tile_batch(batch, all_dates)

                        if batch_end < n_tiles:
                            next_end = min(batch_end + w_read, n_tiles)
                            next_batch = tiles[batch_end:next_end]
                            next_batch_fut = batch_read_pipe.submit(
                                self._read_tile_batch, next_batch, all_dates
                            )

                        for tile_pack, buf in zip(batch, bufs):
                            y0, y1, x0, x1, idxs, lrs, lcs = tile_pack
                            for k, idx in enumerate(idxs):
                                r = int(rows[idx]); c = int(cols[idx])
                                name = f"{r}_{c}"
                                ds = h5_file[name]
                                ds[:, :] = buf[:, :, k]
                            del buf
                        del bufs
                        pbar.update(batch_end - ti)
                        ti = batch_end
                        gc.collect()

        logger.info(f"Year {target_year} full dataset generation complete: {output_file}")
        logger.info(f"Successfully processed {P}/{P_all} pixels")
        self._print_dataset_info(output_file)

    def generate_dataset_for_year(self, target_year):
        """Single-year entry: build skeleton + write tiles (for multi-year, call generate_all_datasets to iterate years serially)."""
        logger.info(f"Starting to generate year {target_year} full dataset (preserving original H5 structure)...")

        if self._global_date_mapping is None:
            self._global_date_mapping = self._build_global_date_mapping()

        if self._global_valid_pixels is None or self._global_geo_reference is None:
            valid_pixels, geo_reference = self._get_global_valid_pixels(self._global_date_mapping)
            if not valid_pixels:
                logger.error("Global valid-pixel set is empty; aborting.")
                return
            self._global_valid_pixels = valid_pixels
            self._global_geo_reference = geo_reference

        ctx = self._build_year_generation_context(target_year)
        if ctx is None:
            return
        logger.info(f"Keeping all pixels inside the reference mask: {ctx['P']} (no per-year time-series ratio filtering)")

        self._create_h5_skeleton_only(ctx)
        self._write_year_tile_data(ctx)

    def generate_all_datasets(self):
        logger.info(f"Starting to generate {len(self.target_years)} years of full datasets...")
        
        # Filter the years to be processed
        years_to_process = self._filter_target_years()
        if not years_to_process:
            logger.info("All target years already have data; exiting.")
            return
        
        logger.info(f"Years actually to process: {years_to_process} (total {len(years_to_process)})")
        
        # First, build the global date mapping and global valid-pixel set once
        if self._global_date_mapping is None:
            self._global_date_mapping = self._build_global_date_mapping()
        YearOnlyDatasetGenerator._GLOBAL_VALUE_RANGE_IN_EFFECT = None

        if self._global_valid_pixels is None or self._global_geo_reference is None:
            vp, gr = self._get_global_valid_pixels(self._global_date_mapping)
            if not vp:
                raise RuntimeError("Global valid-pixel set is empty; cannot continue.")
            self._global_valid_pixels = vp
            self._global_geo_reference = gr

        for year_idx, target_year in enumerate(years_to_process):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing year {target_year} ({year_idx + 1}/{len(years_to_process)})")
            logger.info(f"{'='*60}")
            start_time = time.time()
            self.generate_dataset_for_year(target_year)
            end_time = time.time()
            logger.info(f"Year {target_year} complete, elapsed: {end_time - start_time:.2f} s")

        logger.info(f"\nAll yearly datasets generated!")

    def _print_dataset_info(self, output_file):
        try:
            with h5py.File(output_file, 'r') as f:
                print(f"\nDataset info:")
                print(f"="*50)
                print(f"File: {output_file}")
                print(f"Year: {f.attrs.get('year', 'N/A')}")
                print(f"Date range: {f.attrs.get('start_date', 'N/A')} ~ {f.attrs.get('end_date', 'N/A')}")
                print(f"Dataset count (pixel count): {len(f.keys())}")
                print(f"Total channels: {f.attrs.get('total_channels', 'N/A')}")
                print(f"Total time steps: {f.attrs.get('total_time_steps', 'N/A')}")
                print(f"Data format: {f.attrs.get('data_format', 'N/A')}")
                print(f"\nChannel mapping:")
                for key in f.attrs.keys():
                    if key.startswith('channel_mapping_'):
                        driver_name = key.replace('channel_mapping_', '')
                        mapping = f.attrs[key]
                        print(f"  {driver_name}: {mapping}")
                if len(f.keys()) > 0:
                    first_key = next(iter(f.keys()))
                    first_dataset = f[first_key]
                    print(f"\nSample dataset ({first_key}): shape={first_dataset.shape}, dtype={first_dataset.dtype}")
                    arr = first_dataset[:, :]
                    print(f"  Data range: {np.nanmin(arr)} ~ {np.nanmax(arr)}")
                print(f"="*50)
        except Exception as e:
            logger.error(f"Failed to print dataset info: {e}")
            
    def _iter_tiles(self, height, width, tile_h=96, tile_w=96):
        y = 0
        while y < height:
            x = 0
            y1 = min(y + tile_h, height)
            while x < width:
                x1 = min(x + tile_w, width)
                yield y, y1, x, x1
                x = x1
            y = y1

    def _build_day_tile_stack_window(self, date_files, y0, y1, x0, x1):
        """
        Read window [y0:y1, x0:x1] data for each driver on a given day and compose them as [C, h, w].
        GDAL windowed reads run in parallel across drivers (ThreadPoolExecutor), matching the reference implementation.
        """
        h = int(y1 - y0)
        w = int(x1 - x0)
        C = self.total_channels
        tile = np.zeros((C, h, w), dtype=np.uint8)
        ix0, iy0 = int(x0), int(y0)

        def read_driver_window(dn):
            info = self.channel_mapping.get(dn)
            if info is None:
                return dn, None, None, None
            fp = date_files.get(dn)
            if fp is None:
                return dn, None, None, None
            try:
                from osgeo import gdal
                ds = gdal.Open(fp, gdal.GA_ReadOnly)
                if ds is None:
                    return dn, None, None, None
                bands = ds.RasterCount
                if bands == 1:
                    arr = ds.GetRasterBand(1).ReadAsArray(ix0, iy0, int(w), int(h))
                else:
                    arr = np.array([
                        ds.GetRasterBand(i).ReadAsArray(ix0, iy0, int(w), int(h))
                        for i in range(1, bands + 1)
                    ])
                ds = None
                return dn, fp, info, arr
            except Exception as e:
                logger.debug(f"[{dn}] window read failed: {e}")
                return dn, None, None, None

        n_drv = max(1, len(self.driver_order))
        with ThreadPoolExecutor(max_workers=min(self.max_workers, n_drv)) as ex:
            futs = [ex.submit(read_driver_window, dn) for dn in self.driver_order]
            for fut in as_completed(futs):
                _dn, fp, info, arr = fut.result()
                if info is None or arr is None:
                    continue
                self._exit_if_float_raster(arr, f"[{_dn}] {fp}" if fp else f"[{_dn}]")
                s, e = info['start_idx'], info['end_idx']
                if arr.ndim == 2:
                    if arr.shape != (h, w):
                        continue
                    arr = arr[np.newaxis, ...]
                elif arr.ndim == 3:
                    if arr.shape[1:] != (h, w):
                        continue
                    arr = arr[: (e - s)]
                tile[s:s + arr.shape[0], :, :] = arr
                del arr
        return tile

    def _read_one_tile_to_buffer(self, tile_pack, all_dates):
        """
        NFS/GDAL only: read the full-year stack for one spatial tile into (C, T, P_tile); no HDF5 write here.
        Internally still uses per-day prefetch + per-driver parallel windowed reads.
        """
        y0, y1, x0, x1, idxs, lrs, lcs = tile_pack
        C = self.total_channels
        T = len(all_dates)
        P_tile = int(idxs.size)
        buf = np.empty((C, T, P_tile), dtype=np.uint8)
        prefetch = self.prefetch_days
        semaphore = threading.Semaphore(prefetch)

        def _fetch_day(t_idx, date):
            date_files = self._global_date_mapping.get(date, {})
            result = self._build_day_tile_stack_window(date_files, y0, y1, x0, x1)
            return t_idx, result

        with ThreadPoolExecutor(max_workers=prefetch) as day_ex:
            pending = {}
            submit_ptr = 0

            def _submit_next():
                nonlocal submit_ptr
                if submit_ptr < T:
                    semaphore.acquire()
                    fut = day_ex.submit(_fetch_day, submit_ptr, all_dates[submit_ptr])
                    pending[submit_ptr] = fut
                    submit_ptr += 1

            for _ in range(min(prefetch, T)):
                _submit_next()

            for t in range(T):
                fut = pending.pop(t)
                _t_idx, day_tile = fut.result()
                semaphore.release()
                _submit_next()
                buf[:, t, :] = day_tile[:, lrs, lcs]
                del day_tile
                if (t + 1) % 32 == 0:
                    gc.collect()
        return buf


# ---------------- CLI ----------------
def parse_year_list(year_string):
    try:
        return ast.literal_eval(year_string)
    except:
        try:
            return [int(x.strip()) for x in year_string.split(',')]
        except:
            return [int(year_string)]

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate year-range datasets: ERA5 reference spatial mask + tile streaming writes (no Pass-1)'
    )
    parser.add_argument('--data_dir', 
                       default='/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled_5xdownsampled_merged_unifiedEntireCan_scaledInt',
                       help='Data root directory')
    parser.add_argument('--output_dir',
                       default='/s3/zhengsen/wildfire_dataset/h5_dataset/entire_can_v1',
                       help='Output directory')
    parser.add_argument('--years', type=str, default='[2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]',
                       help='Target year list, format: [2000,2021,...] or 2000,2021,...')
    parser.add_argument('--max_workers', type=int, default=32,
                       help='Number of threads for driver reads (32-48 is recommended to saturate NFS concurrency)')
    parser.add_argument('--prefetch_days', type=int, default=32,
                       help='Days prefetched within each tile (controls memory / NFS concurrency, 16-48 is recommended)')
    parser.add_argument('--tile-read-workers', type=int, default=8,
                       help='Tiles prefetched in parallel per batch (<= total tile count; H5 is still written single-threaded, 4-8 is recommended)')
    parser.add_argument('--batch_size', type=int, default=10000,
                       help='Backward-compat parameter, no actual effect')
    parser.add_argument('--force-retrain', action='store_true',
                       help='Force reprocessing of all years even if data already exists')
    parser.add_argument(
        '--reference-valid-mask-dir',
        type=str,
        default=None,
        help='Reference directory for the global spatial valid mask (default: unifiedEntireCan/ERA5_merged)',
    )
    parser.add_argument(
        '--reference-valid-mask-seed',
        type=int,
        default=None,
        help='Random seed for picking the reference image (random on every run by default)',
    )
    args = parser.parse_args()

    target_years = parse_year_list(args.years)
    generator = YearOnlyDatasetGenerator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        target_years=target_years,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        force_retrain=args.force_retrain,
        prefetch_days=args.prefetch_days,
        tile_read_workers=args.tile_read_workers,
        reference_valid_mask_dir=args.reference_valid_mask_dir,
        reference_valid_mask_seed=args.reference_valid_mask_seed,
    )
    generator.generate_all_datasets()

if __name__ == "__main__":
    main()
