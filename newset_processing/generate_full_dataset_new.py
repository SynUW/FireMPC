#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-dataset generator -- year-range version (keeps the original H5 structure; two-stage filtering + streaming writes).
Global valid pixels: across all drivers (over all dates: AND per date, then OR over dates).

Update: pixel filtering uses only the image's built-in NoData values
- Read built-in NoData values from the TIFF images
- Sample 100 TIFs across all drivers to compute the intersection of the global valid-value range
- Storage: write raw values directly, with no scaling or conversion
"""

import os
import h5py
import numpy as np
import glob
from datetime import datetime, timedelta
from tqdm import tqdm
import logging
from functools import lru_cache
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import ast
from collections import OrderedDict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YearOnlyDatasetGenerator:
    def __init__(self, data_dir, output_dir, target_years=[2024], 
                 max_workers=12, batch_size=10000, max_aggregation_dates=10, force_retrain=False):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.target_years = target_years if isinstance(target_years, list) else [target_years]
        self.max_workers = max(1, int(max_workers))
        self.batch_size = batch_size  # Backward-compatibility parameter
        self.max_aggregation_dates = max_aggregation_dates  # Maximum number of dates when aggregating
        self.force_retrain = force_retrain  # Whether to force reprocessing

        # Keep consistent with the existing order
        self.driver_order = [
            'MCD14A1_mosaic_binary_downsampled_masked', # checked and fixed  
            # 'MCD14A1_withoutMask_Binary_downsampled_masked', # checked and fixed
            'ERA5_consistent_mosaic_withnodata_downsampled', # checked and fixed
            'distance_maps',  # checked
            'NDVI_EVI_withQA',
            'MCD15A3H_mosaic_249_filled_downsampled', # checked
            'MCD09GA_b1237_mosaic_withQA_QAapplied_filled_downsampled',
            'MCD11A1_mosaic_QAapplied_filled_downsampled_filled',
            'MCD09CMG_mosaic_QAapplied_filled_downsampled',
            'MCD12Q1_mosaic_downsampled'
        ]

        os.makedirs(output_dir, exist_ok=True)
        self.driver_dirs = self._get_driver_directories()
        self.total_channels, self.channel_mapping = self._calculate_channels_and_mapping()

        # Runtime cache
        self._global_date_mapping = None
        self._global_valid_pixels = None
        self._global_geo_reference = None
        self._global_value_range = None  # {driver_name: (vmin, vmax), ...}, each driver's own valid-value range

        logger.info(f"Data directory: {data_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Target years: {self.target_years}")
        logger.info(f"Found {len(self.driver_dirs)} drivers")
        logger.info(f"Total channels: {self.total_channels}")
        logger.info(f"Parallel threads: {self.max_workers}")
        logger.info(f"Force reprocess: {self.force_retrain}")

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
                    # Multi-band block read
                    arr = np.zeros((bands, height, width), dtype=np.float32)
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
        with tqdm(total=len(self.driver_dirs), desc="Scanning drivers") as pbar:
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

    # ---------------- Global valid pixels (across all dates) ----------------
    def _get_global_valid_pixels(self, all_date_to_file):
        """
        Global valid pixels: for dates where all drivers are present:
            per_day_mask = valid(all drivers) AND
        Then OR over the date dimension:
            global_mask = OR_over_days(per_day_mask)
        Returns the list of pixel coordinates in global_mask; shared by all years.
        """
        logger.info("Building a valid mask from all drivers (using only the images' built-in NoData + the global valid-value range) and OR-aggregating across dates...")
        driver_names = [dn for dn in self.driver_order if dn in self.driver_dirs]
        if not driver_names:
            raise RuntimeError("No driver directory found")

        # Pick a sample to determine size and georeferencing
        sample_fp = None
        for d, m in all_date_to_file.items():
            ok = all(dn in m for dn in driver_names)
            if ok:
                # Use the first driver as the reference
                sample_fp = m[driver_names[0]]
                break
        if sample_fp is None:
            raise RuntimeError("No single date has all drivers; cannot build the global valid mask.")

        _, geo_reference, _ = self._load_single_file_with_gdal(sample_fp)
        width, height = geo_reference['raster_size']
        height, width = int(height), int(width)

        global_mask = np.zeros((height, width), dtype=bool)
        pair_dates = [d for d in sorted(all_date_to_file.keys()) if all(dn in all_date_to_file[d] for dn in driver_names)]
        logger.info(f"Number of dates used for aggregation (all drivers present): {len(pair_dates)}")

        # Performance optimization: sampling strategy -- if there are too many dates, process only a subset
        max_dates = self.max_aggregation_dates
        if len(pair_dates) > max_dates:
            # Uniform sampling
            step = len(pair_dates) // max_dates
            pair_dates = pair_dates[::step][:max_dates]
            logger.info(f"Number of dates after sampling: {len(pair_dates)}")
        
        # Further optimization: if there are still many dates, use more aggressive sampling
        # if len(pair_dates) > 500:
        #     # Process only 1 out of every 10 dates
        #     pair_dates = pair_dates[::10]
        #     logger.info(f"Number of dates after aggressive sampling: {len(pair_dates)}")

        # Parallel-processing optimization
        def process_single_date(date_info):
            """Process a single date and return its mask (AND of all drivers)"""
            date, driver_to_file = date_info
            
            try:
                day_mask = None
                for dn in driver_names:
                    fp = driver_to_file.get(dn)
                    if not fp:
                        return None
                    arr, _, nodata_vals = self._load_single_file_with_gdal(fp)
                    if arr is None:
                        return None
                    mask = YearOnlyDatasetGenerator._valid_mask_from_array(arr, image_nodata_values=nodata_vals, driver_name=dn)
                    day_mask = mask if day_mask is None else (day_mask & mask)
                return day_mask
                
            except Exception as e:
                logger.warning(f"Processing date {date} failed: {e}")
                return None

        # Prepare data for parallel processing
        date_tasks = []
        for d in pair_dates:
            date_tasks.append((d, all_date_to_file[d]))

        # Parallel processing
        logger.info(f"Starting parallel processing {len(date_tasks)} dates, using {self.max_workers} threads")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_date = {
                executor.submit(process_single_date, task): task[0] 
                for task in date_tasks
            }
            
            # Collect results and aggregate -- with batched-processing optimization
            processed_count = 0
            batch_size = 10  # Batch-process masks
            mask_batch = []
            
            for future in tqdm(as_completed(future_to_date), 
                             total=len(future_to_date), 
                             desc="Aggregate global mask"):
                day_mask = future.result()
                if day_mask is not None:
                    mask_batch.append(day_mask)
                
                processed_count += 1
                
                # Aggregate masks in batch
                if len(mask_batch) >= batch_size or processed_count == len(future_to_date):
                    if mask_batch:
                        # Use numpy's efficient bitwise ops for batch aggregation
                        batch_array = np.array(mask_batch)
                        batch_or = np.any(batch_array, axis=0)
                        global_mask |= batch_or
                        mask_batch = []  # Clear the batch
                
                # Periodic garbage collection
                if processed_count % 50 == 0:
                    gc.collect()

        elapsed_time = time.time() - start_time
        logger.info(f"Aggregation complete, elapsed: {elapsed_time:.2f}s")

        rows, cols = np.where(global_mask)
        valid_pixels = [(int(r), int(c)) for r, c in zip(rows, cols)]
        logger.info(f"[Global] valid pixel count: {len(valid_pixels)}")
        return valid_pixels, geo_reference

    # ---------------- Sample drivers to compute "per-driver valid-value range" ----------------
    def _compute_global_value_range(self, max_samples_per_driver=100, seed=42):
        """
        For each driver directory, randomly pick up to 100 TIFs:
          - Identify valid pixels using only the images' built-in NoData;
          - Compute each driver's own valid-value range (multi-band uses the first channel, consistent with the mask);
          - Each driver is computed independently; no intersection across drivers is required.
        Returns a per-driver valid-range dict: {driver_name: (vmin, vmax), ...}.
        Drivers whose range cannot be computed are not included in the dict.
        """
        import random
        import glob
        random.seed(seed)
        
        # Record each driver's value range
        driver_ranges = {}
        
        for dn in [dn for dn in self.driver_order if dn in self.driver_dirs]:
            logger.debug(f"Processing driver {dn} valid-value range stats")
            
            folder = self.driver_dirs[dn]
            files = glob.glob(os.path.join(folder, '*.tif'))
            if not files:
                continue
            if len(files) > max_samples_per_driver:
                files = random.sample(files, max_samples_per_driver)
            
            driver_mins = []
            driver_maxs = []
            
            for fp in files:
                try:
                    arr, _, nodata_vals = self._load_single_file_with_gdal(fp)
                    if arr is None:
                        continue
                    mask = YearOnlyDatasetGenerator._valid_mask_from_array(arr, image_nodata_values=nodata_vals, driver_name=dn)
                    if mask is None or not np.any(mask):
                        continue
                    if arr.ndim == 3:
                        arr_use = arr[0]
                    else:
                        arr_use = arr
                    vals = arr_use[mask]
                    if vals.size == 0:
                        continue
                    
                    # Ignore extreme NaN (defensive)
                    if np.issubdtype(vals.dtype, np.floating):
                        vals = vals[~np.isnan(vals)]
                        if vals.size == 0:
                            continue
                    
                    file_min = np.nanmin(vals)
                    file_max = np.nanmax(vals)
                    
                    driver_mins.append(file_min)
                    driver_maxs.append(file_max)
                except Exception as e:
                    logger.debug(f"Sample {fp} failed: {e}")
                    continue
            
            # Compute this driver's valid-value range (the min of all sample minima, the max of all sample maxima)
            if driver_mins and driver_maxs:
                driver_vmin = min(driver_mins)
                driver_vmax = max(driver_maxs)
                
                if np.isfinite(driver_vmin) and np.isfinite(driver_vmax) and driver_vmin < driver_vmax:
                    driver_ranges[dn] = (float(driver_vmin), float(driver_vmax))
                    logger.info(f"Driver {dn} valid-value range: [{driver_vmin:.6f}, {driver_vmax:.6f}] (sample count: {len(driver_mins)})")
                else:
                    logger.warning(f"Driver {dn} cannot compute valid-value range: vmin={driver_vmin}, vmax={driver_vmax}")
        
        if not driver_ranges:
            logger.warning("Could not compute a valid-value range for any driver from the samples; range constraints will not be enabled.")
            return None
        
        logger.info(f"Successfully computed {len(driver_ranges)} drivers' valid-value ranges")
        return driver_ranges

    # ---------------- Pass 1: compute the per-channel x time non-NaN ratio and filter pixels ----------------
    def _count_valid_channels_for_day(self, date_files, rows, cols):
        P = rows.shape[0]
        per_day_counts = np.zeros(P, dtype=np.uint16)

        def load_one(driver_name):
            fp = date_files.get(driver_name, None)
            if fp is None:
                return driver_name, None
            arr, _, nodata_vals = self._load_single_file_with_gdal(fp)
            return driver_name, arr, nodata_vals

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {ex.submit(load_one, dn): dn for dn in self.driver_order}
            for fut in as_completed(futs):
                dn = futs[fut]
                try:
                    _, arr, nodata_vals = fut.result()
                except Exception as e:
                    logger.error(f"[{dn}] read failed: {e}")
                    arr = None
                    nodata_vals = None
                if arr is None:
                    continue

                # Use the same mask logic as the global valid pixels
                if arr.ndim == 2:
                    a = arr[rows, cols]
                    mask = YearOnlyDatasetGenerator._valid_mask_from_array(a, image_nodata_values=nodata_vals, driver_name=dn)
                    per_day_counts += mask.astype(np.uint16)
                elif arr.ndim == 3:
                    sub = arr[:, rows, cols]
                    valid_ch = np.zeros((sub.shape[0], sub.shape[1]), dtype=np.uint16)
                    for i in range(sub.shape[0]):
                        band_nodata = nodata_vals[i] if i < len(nodata_vals) else None
                        band_mask = YearOnlyDatasetGenerator._valid_mask_from_array(sub[i], image_nodata_values=[band_nodata], driver_name=dn)
                        valid_ch[i] = band_mask.astype(np.uint16)
                    per_day_counts += valid_ch.sum(axis=0)
                del arr
        return per_day_counts

    def _select_pixels_by_non_nan_ratio(self, all_dates, date_map, rows, cols, C):
        T = len(all_dates)
        total_needed = C * T
        total_counts = np.zeros(rows.shape[0], dtype=np.uint32)

        for d in tqdm(all_dates, desc="Pass 1: computing valid-ratio statistics"):
            date_files = date_map.get(d, {})
            per_day = self._count_valid_channels_for_day(date_files, rows, cols)
            total_counts += per_day.astype(np.uint32)
            del per_day
            if (d.timetuple().tm_yday % 3) == 0:
                gc.collect()

        ratio = total_counts / float(total_needed)
        keep_mask = ratio > 0.1
        kept = int(keep_mask.sum())
        logger.info(f"Pixel count after filtering: {kept}/{rows.shape[0]}（non_nan_ratio > 0.1）")
        return keep_mask, ratio

    # ---------------- Pass 2: write (kept pixels only) ----------------
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
        vals = np.full((C, Pk), 0, dtype=np.float32)

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

    def generate_dataset_for_year(self, target_year):
        logger.info(f"Starting to generate year {target_year} full dataset (preserving original H5 structure)...")

        # Check if output file already exists
        output_file = os.path.join(self.output_dir, f'{target_year}_year_dataset.h5')
        if os.path.exists(output_file) and not self.force_retrain:
            logger.info(f"Year {target_year} dataset already exists; skipping: {output_file}")
            return

        if self._global_date_mapping is None:
            self._global_date_mapping = self._build_global_date_mapping()
        date_map = self._global_date_mapping

        # -- Global valid pixels (computed only once):
        if self._global_valid_pixels is None or self._global_geo_reference is None:
            valid_pixels, geo_reference = self._get_global_valid_pixels(date_map)
            if not valid_pixels:
                logger.error("Global valid-pixel set is empty; aborting.")
                return
            self._global_valid_pixels = valid_pixels
            self._global_geo_reference = geo_reference
        else:
            valid_pixels = self._global_valid_pixels
            geo_reference = self._global_geo_reference

        all_dates = self._get_all_required_dates(target_year)

        # Image dimensions
        if not geo_reference:
            logger.error("Cannot determine image size information")
            return
        width, height = geo_reference['raster_size']
        height, width = int(height), int(width)

        rows_all = np.array([r for r, _ in valid_pixels], dtype=np.int32)
        cols_all = np.array([c for _, c in valid_pixels], dtype=np.int32)
        P_all = rows_all.shape[0]
        T = len(all_dates)
        C = self.total_channels

        # -------- Pass 1: compute per-channel x time non-NaN ratio and filter pixels --------
        keep_mask, ratio = self._select_pixels_by_non_nan_ratio(all_dates, date_map, rows_all, cols_all, C)
        rows = rows_all[keep_mask]
        cols = cols_all[keep_mask]
        P = rows.shape[0]
        if P == 0:
            logger.error("No pixels remain after filtering; aborting")
            return

        # Output file path (existence already checked above)
        output_file = os.path.join(self.output_dir, f'{target_year}_year_dataset.h5')

        logger.info(f"Creating output file: {output_file}")
        with h5py.File(output_file, 'w', libver='latest') as h5_file:
            # File-level attributes
            h5_file.attrs['year'] = str(target_year)
            h5_file.attrs['total_time_steps'] = T
            h5_file.attrs['total_channels'] = C
            h5_file.attrs['data_format'] = 'channels_x_time'
            h5_file.attrs['data_type'] = 'year_pixel_time_series'
            h5_file.attrs['driver_names'] = self.driver_order
            h5_file.attrs['total_pixels'] = int(P_all)
            h5_file.attrs['processed_pixels'] = int(P)
            h5_file.attrs['start_date'] = all_dates[0].strftime('%Y-%m-%d')
            h5_file.attrs['end_date'] = all_dates[-1].strftime('%Y-%m-%d')
            for driver_name, channel_info in self.channel_mapping.items():
                h5_file.attrs[f'channel_mapping_{driver_name}'] = f"{channel_info['start_idx']}-{channel_info['end_idx']-1}"
            if geo_reference:
                h5_file.attrs['geotransform'] = geo_reference['geotransform']
                h5_file.attrs['projection'] = geo_reference['projection']
                h5_file.attrs['raster_size'] = geo_reference['raster_size']
                h5_file.attrs['raster_count'] = geo_reference['raster_count']

            logger.info("Creating per-pixel datasets (kept pixels only)...")
            for r, c in tqdm(zip(rows, cols), total=P, desc="Create pixel datasets"):
                name = f"{int(r)}_{int(c)}"
                ds = h5_file.create_dataset(
                    name,
                    shape=(C, T),
                    dtype=np.float32,
                    chunks=(C, 1),  # Write per day
                    compression='gzip',
                    compression_opts=6,
                    shuffle=True
                )
                ds.attrs['pixel_coord'] = (int(r), int(c))
                ds.attrs['data_shape'] = f"{C}x{T}"
            h5_file.flush()

            # Tile writing
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

            logger.info(f"Number of tiles to process: {len(tiles)}")

            for (y0, y1, x0, x1, idxs, lrs, lcs) in tqdm(tiles, desc=f"{target_year} Tile write"):
                P_tile = idxs.size
                buf = np.empty((C, T, P_tile), dtype=np.float32)

                for t, d in enumerate(all_dates):
                    date_files = self._global_date_mapping.get(d, {})
                    day_tile = self._build_day_tile_stack_window(date_files, y0, y1, x0, x1)  # [C, h, w]
                    buf[:, t, :] = day_tile[:, lrs, lcs]
                    del day_tile
                    if (t + 1) % 8 == 0:
                        gc.collect()

                for k, idx in enumerate(idxs):
                    r = int(rows[idx]); c = int(cols[idx])
                    name = f"{r}_{c}"
                    ds = h5_file[name]
                    ds[:, :] = buf[:, :, k]

                del buf
                gc.collect()

        logger.info(f"Year {target_year} full dataset generation complete: {output_file}")
        logger.info(f"Successfully processed {P}/{P_all} pixels")
        self._print_dataset_info(output_file)

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
        # Compute each driver's valid-value range (once)
        if self._global_value_range is None:
            self._global_value_range = self._compute_global_value_range()
        # Inject the range dict into a class-level variable accessible from static methods (effective in the current process only)
        YearOnlyDatasetGenerator._GLOBAL_VALUE_RANGE_IN_EFFECT = self._global_value_range

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
        Read "single-day" data from all drivers within the [y0:y1, x0:x1] window and compose them as [C, h, w].
        On storage: already-normalized data (except MCD14A1 and MCD12Q1) is multiplied by 255 and rounded; MCD14A1 and MCD12Q1 keep raw values.
        """
        h = y0.__class__(y1 - y0) if isinstance(y0, np.generic) else (y1 - y0)
        w = x0.__class__(x1 - x0) if isinstance(x0, np.generic) else (x1 - x0)
        h = int(h); w = int(w)
        C = self.total_channels
        tile = np.full((C, h, w), 0, dtype=np.float32)

        def read_window(fp):
            from osgeo import gdal
            ds = gdal.Open(fp, gdal.GA_ReadOnly)
            if ds is None:
                return None
            bands = ds.RasterCount
            if bands == 1:
                rb = ds.GetRasterBand(1)
                arr = rb.ReadAsArray(int(x0), int(y0), int(w), int(h))
            else:
                arrs = []
                for i in range(1, bands + 1):
                    rb = ds.GetRasterBand(i)
                    arrs.append(rb.ReadAsArray(int(x0), int(y0), int(w), int(h)))
                arr = np.array(arrs)
            ds = None
            return arr

        for dn in self.driver_order:
            info = self.channel_mapping.get(dn)
            if info is None:
                continue
            s, e = info['start_idx'], info['end_idx']
            fp = date_files.get(dn)
            if fp is None:
                continue
            arr = read_window(fp)
            if arr is None:
                continue
            if arr.ndim == 2:
                if arr.shape != (h, w):
                    continue
                arr = arr[np.newaxis, ...]
            elif arr.ndim == 3:
                if arr.shape[1:] != (h, w):
                    continue
                if arr.shape[0] != (e - s):
                    arr = arr[:(e - s), ...]
            
            # Save raw values directly
            processed_arr = arr
            
            tile[s:s+processed_arr.shape[0], :, :] = processed_arr
            del arr
        return tile


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
    parser = argparse.ArgumentParser(description='Generate year-range datasets -- two-stage filtering + streaming writes (keeps the original H5 structure)')
    parser.add_argument('--data_dir', 
                       default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x_norm',
                       help='Data root directory')
    parser.add_argument('--output_dir',
                       default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/h5_dataset/newset_dataset_v8',
                       help='Output directory')
    parser.add_argument('--years', type=str, default='[2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]',
                       help='Target year list, format: [2000,2021,...] or 2000,2021,...')
    parser.add_argument('--max_workers', type=int, default=12,
                       help='Number of threads used for single-day driver reads')
    parser.add_argument('--batch_size', type=int, default=10000,
                       help='Backward-compat parameter, no actual effect')
    parser.add_argument('--force-retrain', action='store_true',
                       help='Force reprocessing of all years even if data already exists')
    args = parser.parse_args()

    target_years = parse_year_list(args.years)
    generator = YearOnlyDatasetGenerator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        target_years=target_years,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        force_retrain=args.force_retrain
    )
    generator.generate_all_datasets()

if __name__ == "__main__":
    main()
