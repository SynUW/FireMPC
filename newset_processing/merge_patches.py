# -*- coding: utf-8 -*-
"""
Use GDAL to merge multiple TIFF tile files of the same date into a single file:
- Input: a specified folder containing date-named tile files (e.g. ERA5_2013_09_11_tile00.tif)
- Output: a specified folder with one merged TIFF per date (e.g. 2013_09_11.tif)
- Skips already-processed files (checks existence and size > 0)
- Auto-groups by date; one output file per date

Usage:
    python merge_patches.py --input-dir /path/to/input --output-dir /path/to/output
    python merge_patches.py --input-dir /path/to/input --output-dir /path/to/output --overwrite
    python merge_patches.py --input-dir /path/to/input --output-dir /path/to/output --output-name single_file.tif

Advantages of the GDAL version:
- Faster I/O
- Better memory management
- Built-in multithreading support
- More efficient tile-merging algorithm
"""

import os
import sys
import argparse
import logging
import time
import re
import shutil
import signal
import atexit
import tempfile
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from pathlib import Path
from queue import Queue, Empty

from osgeo import gdal, gdalconst

# ================== Defaults ==================
DEFAULT_MAX_WORKERS = max(1, os.cpu_count() // 2)  # Merge concurrency

# GDAL optimization configuration (tuned for large-memory systems)
# Fully utilize 300 GB of memory: allocate 250 GB to the GDAL cache, leaving 50 GB for the OS and Python
GDAL_CACHE_SIZE    = 256000       # GDAL cache size (MB) = 250 GB, fully utilizing large memory
GDAL_NUM_THREADS   = "ALL_CPUS"   # GDAL thread count
COMPRESS           = "LZW"        # Output compression (set to None to disable compression for higher speed)
TILED              = True          # Output block-tiling flag
BIGTIFF            = "IF_SAFER"   # Auto-switch to BigTIFF
BLOCK_SIZE         = 512           # Output file block size (larger improves performance)
# Large-memory optimization: use larger processing blocks (fully utilize memory)
LARGE_MEMORY_BLOCK_SIZE = 16384   # Large-memory processing block size (pixels) = 16K x 16K, fully utilizing memory
MEMORY_THRESHOLD   = 50 * 1024 * 1024 * 1024  # 50 GB: files smaller than this are read fully into memory (more aggressive)

# ============ Logging (concise) ============
logging.basicConfig(
    level=logging.WARNING,  # Show only warnings and errors
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mosaic_gdal")

# Date match patterns - match various date formats
DATE_PATTERNS = [
    re.compile(r'(\d{4})[_-](\d{2})[_-](\d{2})'),  # YYYY_MM_DD or YYYY-MM-DD
    re.compile(r'(\d{4})(\d{2})(\d{2})'),          # YYYYMMDD
    re.compile(r'(\d{4})[_-](\d{1,2})[_-](\d{1,2})'),  # YYYY_M_D or YYYY-M-D
]

# ---- Local scratch (prefer /data, fall back to /dev/shm) ----
def _pick_scratch() -> Path:
    for p in [Path("/data"), Path("/dev/shm")]:
        try:
            p.mkdir(parents=True, exist_ok=True)
            probe = p / f".probe_{os.getpid()}"
            probe.touch(); probe.unlink()
            return p
        except (PermissionError, OSError):
            continue
    return Path(tempfile.gettempdir())

_SCRATCH_DIR = _pick_scratch() / f"merge_patches_{os.getpid()}"

def _cleanup_scratch():
    if _SCRATCH_DIR.exists():
        shutil.rmtree(_SCRATCH_DIR, ignore_errors=True)

atexit.register(_cleanup_scratch)
signal.signal(signal.SIGINT,  lambda s, f: (_cleanup_scratch(), sys.exit(1)))
signal.signal(signal.SIGTERM, lambda s, f: (_cleanup_scratch(), sys.exit(1)))

# Initialize GDAL
gdal.UseExceptions()  # Enable GDAL exception handling to suppress warnings
gdal.SetCacheMax(GDAL_CACHE_SIZE * 1024 * 1024)  # Convert to bytes
gdal.SetConfigOption('GDAL_NUM_THREADS', GDAL_NUM_THREADS)
gdal.SetConfigOption('GDAL_CACHEMAX', str(GDAL_CACHE_SIZE))
# Aggressive I/O optimization
gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')  # Disable directory scanning
gdal.SetConfigOption('CPL_VSIL_CURL_USE_HEAD', 'NO')  # Disable HTTP head check
gdal.SetConfigOption('GDAL_MAX_DATASET_POOL_SIZE', '1000')  # Increase dataset pool size
gdal.SetConfigOption('GDAL_SWATH_SIZE', '268435456')  # 256 MB swath size


def format_time(seconds):
    """Format time display"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def print_progress(current, total, start_time, last_print_time=[0]):
    """Print a custom-format progress bar -- single-line dynamic update"""
    current_time = time.time()
    
    # Throttle print frequency: at most once per second, or every 1% of items processed
    progress_percent = current / total * 100
    should_update = (
        current_time - last_print_time[0] >= 2.0 or  # At least 2 seconds interval
        progress_percent - int(progress_percent) == 0 or  # Every 1% progress
        current == total  # When complete
    )
    
    if not should_update and current < total:
        return
    
    last_print_time[0] = current_time
    
    elapsed = current_time - start_time
    
    if current > 0:
        # Compute average processing time
        avg_time_per_item = elapsed / current
        remaining_items = total - current
        eta = remaining_items * avg_time_per_item
        eta_str = format_time(eta)
    else:
        eta_str = "?:??:??"
    
    elapsed_str = format_time(elapsed)
    total_estimated = elapsed / current * total if current > 0 else 0
    total_str = format_time(total_estimated)
    
    # Use \r to return to the start of the line and update progress in place
    print(f"\r[{current}/{total}], {elapsed_str}/{total_str}, ETA: {eta_str}", end='', flush=True)
    
    if current == total:
        print()  # Newline after completion


def extract_date_from_filename(filename: str):
    """Extract the date from a filename and return a YYYY_MM_DD string"""
    for pattern in DATE_PATTERNS:
        match = pattern.search(filename)
        if match:
            year, month, day = match.groups()
            # Ensure month and day are two digits
            month = month.zfill(2)
            day = day.zfill(2)
            return f"{year}_{month}_{day}"
    return None


def validate_geospatial_info(file_paths):
    """Validate that input files have consistent geospatial info"""
    if not file_paths:
        return True
    
    # Use the first file as reference
    ref_ds = gdal.Open(str(file_paths[0]), gdal.GA_ReadOnly)
    if ref_ds is None:
        logger.error(f"Cannot open reference file: {file_paths[0]}")
        return False
    
    ref_gt = ref_ds.GetGeoTransform()
    ref_proj = ref_ds.GetProjection()
    ref_pixel_x = abs(ref_gt[1])
    ref_pixel_y = abs(ref_gt[5])
    
    inconsistent_files = []
    
    for i, file_path in enumerate(file_paths[1:], 1):
        ds = gdal.Open(str(file_path), gdal.GA_ReadOnly)
        if ds is None:
            continue
            
        gt = ds.GetGeoTransform()
        proj = ds.GetProjection()
        pixel_x = abs(gt[1])
        pixel_y = abs(gt[5])
        
        # Check whether pixel sizes are consistent
        if abs(pixel_x - ref_pixel_x) > 1e-8 or abs(pixel_y - ref_pixel_y) > 1e-8:
            inconsistent_files.append((file_path, f"Pixel size mismatch: {pixel_x:.8f} x {pixel_y:.8f}"))
        
        # Check whether projections are consistent
        if proj != ref_proj:
            inconsistent_files.append((file_path, "Projections are inconsistent"))
        
        ds = None
    
    ref_ds = None
    
    if inconsistent_files:
        logger.error("Found files with inconsistent geospatial info:")
        for file_path, reason in inconsistent_files:
            logger.error(f"  {file_path.name}: {reason}")
        return False
    
    return True


def group_files_by_date(input_dir: str):
    """Group TIFF files by date and return {date: [file_list]}"""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    patterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
    all_files = []
    
    for pattern in patterns:
        all_files.extend(input_path.glob(pattern))
    
    # Group by date
    date_groups = {}
    for file_path in all_files:
        date_str = extract_date_from_filename(file_path.name)
        if date_str:
            if date_str not in date_groups:
                date_groups[date_str] = []
            date_groups[date_str].append(file_path)
        else:
            logger.warning(f"Cannot extract date from filename: {file_path.name}")
    
    # Sort files within each date
    for date_str in date_groups:
        date_groups[date_str] = sorted(date_groups[date_str])
    
    return date_groups


def get_tiff_files(input_dir: str):
    """Get all TIFF files in the input directory (backward-compatible)"""
    date_groups = group_files_by_date(input_dir)
    all_files = []
    for files in date_groups.values():
        all_files.extend(files)
    return sorted(all_files)


def get_file_info(file_path: str):
    """Get TIFF file info"""
    ds = gdal.Open(str(file_path), gdalconst.GA_ReadOnly)
    if ds is None:
        return None
    
    info = {
        'ds': ds,
        'width': ds.RasterXSize,
        'height': ds.RasterYSize,
        'bands': ds.RasterCount,
        'geotransform': ds.GetGeoTransform(),
        'projection': ds.GetProjection(),
        'nodata': ds.GetRasterBand(1).GetNoDataValue(),
        'datatype': ds.GetRasterBand(1).DataType
    }
    return info


def merge_files_gdal(file_paths, out_path, overwrite=False):
    """
    Merge multiple TIFF files into a single file using GDAL.
    """
    if (not overwrite) and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True

    try:
        # Use the first file's info as reference
        ref_info = get_file_info(file_paths[0])
        if ref_info is None:
            logger.error(f"Cannot open reference file: {file_paths[0]}")
            return False
        
        ref_bands = ref_info['bands']
        ref_nodata = ref_info['nodata']
        ref_datatype = ref_info['datatype']
        
        # Compute the merged extent
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        
        file_infos = []
        for file_path in file_paths:
            info = get_file_info(file_path)
            if info is None:
                logger.error(f"Cannot open file: {file_path}")
                continue
            
            file_infos.append(info)
            gt = info['geotransform']
            
            # Compute the file's geographic bounds
            file_min_x = gt[0]
            file_max_x = gt[0] + info['width'] * gt[1]
            file_min_y = gt[3] + info['height'] * gt[5]
            file_max_y = gt[3]
            
            min_x = min(min_x, file_min_x)
            max_x = max(max_x, file_max_x)
            min_y = min(min_y, file_min_y)
            max_y = max(max_y, file_max_y)
        
        if not file_infos:
            logger.error(f"No valid files: {out_path}")
            return False
        
        # Compute output size
        pixel_size_x = abs(ref_info['geotransform'][1])
        pixel_size_y = abs(ref_info['geotransform'][5])
        
        out_width = int((max_x - min_x) / pixel_size_x)
        out_height = int((max_y - min_y) / pixel_size_y)
        
        # Create output dataset
        driver = gdal.GetDriverByName('GTiff')
        out_geotransform = (min_x, pixel_size_x, 0, max_y, 0, -pixel_size_y)
        
        # Creation options (optimized to fully utilize memory)
        creation_options = []
        if COMPRESS:
            creation_options.append(f'COMPRESS={COMPRESS}')
        creation_options.extend([
            'TILED=YES' if TILED else 'TILED=NO',
            f'BLOCKXSIZE={BLOCK_SIZE}',
            f'BLOCKYSIZE={BLOCK_SIZE}',
            f'BIGTIFF={BIGTIFF}',
            'NUM_THREADS=ALL_CPUS'
        ])
        
        # Create output file
        out_ds = driver.Create(
            out_path,
            out_width,
            out_height,
            ref_bands,
            ref_datatype,
            options=creation_options
        )
        
        if out_ds is None:
            logger.error(f"Cannot create output file: {out_path}")
            return False
        
        out_ds.SetGeoTransform(out_geotransform)
        out_ds.SetProjection(ref_info['projection'])
        
        # Set NoData value
        if ref_nodata is not None:
            for i in range(1, ref_bands + 1):
                out_ds.GetRasterBand(i).SetNoDataValue(ref_nodata)
        
        # Choose a processing strategy intelligently based on file size and available memory
        # For small files, read the entire file into memory (faster)
        # For large files, use large blocks (fully utilizing memory)
        
        # Estimate memory usage per file (bytes)
        # Assume 2 bytes per pixel (Int16), multi-band
        estimated_file_size = file_infos[0]['width'] * file_infos[0]['height'] * ref_bands * 2 if file_infos else 0
        
        # Estimate total memory requirement (all files)
        total_estimated_size = estimated_file_size * len(file_infos)
        
        # More aggressive strategy: fully utilize large memory
        # If total memory requirement is below the threshold, read all files into memory
        if total_estimated_size < MEMORY_THRESHOLD:
            # Read the entire file into memory (fastest)
            block_size = max(out_width, out_height)  # Use the full image size
        elif estimated_file_size < MEMORY_THRESHOLD // 4:  # Per-file size under 12.5 GB
            # Single file is small; read it entirely
            block_size = max(out_width, out_height)
        else:
            # Very large files: use very large blocks (fully utilizing memory)
            # Use LARGE_MEMORY_BLOCK_SIZE blocks (16K x 16K), fully utilizing large memory
            block_size = min(LARGE_MEMORY_BLOCK_SIZE, out_width, out_height)
        
        # Merge each band
        # Optimization: for files that can be read directly, load them all into memory at once
        use_direct_read = (block_size >= max([info['width'] for info in file_infos] + [info['height'] for info in file_infos]))
        
        for band_idx in range(1, ref_bands + 1):
            out_band = out_ds.GetRasterBand(band_idx)
            
            # Initialize output bands with the NoData value
            if ref_nodata is not None:
                out_band.Fill(ref_nodata)
            
            # Merge this band from each file
            for file_info in file_infos:
                file_ds = file_info['ds']
                file_band = file_ds.GetRasterBand(band_idx)
                
                # Compute the file's position in the output
                file_gt = file_info['geotransform']
                file_x = int((file_gt[0] - min_x) / pixel_size_x)
                file_y = int((max_y - file_gt[3]) / pixel_size_y)
                
                file_width = file_info['width']
                file_height = file_info['height']
                
                # If the entire file can be read directly, use the fastest path
                if use_direct_read or block_size >= max(file_width, file_height):
                    # Read the entire file (fastest, fully utilizing memory)
                    file_data = file_band.ReadAsArray()
                    out_band.WriteArray(file_data, file_x, file_y)
                else:
                    # Block-wise read/write (using very large blocks, fully utilizing memory)
                    # Use larger blocks to reduce I/O count
                    for y in range(0, file_height, block_size):
                        y_size = min(block_size, file_height - y)
                        for x in range(0, file_width, block_size):
                            x_size = min(block_size, file_width - x)
                            
                            # Read block data
                            file_data = file_band.ReadAsArray(x, y, x_size, y_size)
                            
                            # Write to the output band (considering the file's position in the output)
                            out_x = file_x + x
                            out_y = file_y + y
                            out_band.WriteArray(file_data, out_x, out_y)
        
        # Flush the cache once (rather than per band)
        out_ds.FlushCache()
        
        # Clean up resources
        out_ds = None
        for info in file_infos:
            info['ds'] = None
        
        return True

    except Exception as e:
        logger.error(f"GDAL merge failed: {out_path} - {e}")
        return False


def merge_files_gdal_warp(file_paths, out_path, overwrite=False):
    """
    Merge using GDAL's gdalwarp with the most-optimized strategy.
    Uses VRT as an intermediate format to improve performance.
    """
    if (not overwrite) and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True

    try:
        # Method 1: VRT + Warp directly (fastest)
        # Create a VRT as an intermediate format for significantly better performance
        vrt_path = '/vsimem/temp_mosaic.vrt'
        
        # Create the VRT (in memory, very fast)
        vrt_options = gdal.BuildVRTOptions(
            separate=False,
            addAlpha=False,
            resampleAlg=gdal.GRA_NearestNeighbour
        )
        
        vrt_ds = gdal.BuildVRT(vrt_path, [str(p) for p in file_paths], options=vrt_options)
        if vrt_ds is None:
            logger.error("Cannot create VRT file")
            return False
        
        # Optimization: minimize all unnecessary operations
        creation_opts = []
        if COMPRESS:
            creation_opts.append(f'COMPRESS={COMPRESS}')
        creation_opts.extend([
            'TILED=YES',
            f'BLOCKXSIZE={BLOCK_SIZE}',
            f'BLOCKYSIZE={BLOCK_SIZE}',
            f'BIGTIFF={BIGTIFF}',
        ])
        
        # Most aggressive Warp options
        warp_options = gdal.WarpOptions(
            format='GTiff',
            creationOptions=creation_opts,
            resampleAlg=gdal.GRA_NearestNeighbour,
            multithread=True,
            # Disable all unnecessary operations
            copyMetadata=False,
            dstNodata=None,
            # Use the most aggressive performance options
            options=[
                'NUM_THREADS=ALL_CPUS',
                'GDAL_CACHEMAX=256000',
                'GDAL_SWATH_SIZE=536870912',  # 512 MB swath (larger)
            ],
        )
        
        # Execute the merge (VRT -> final file)
        out_ds = gdal.Warp(out_path, vrt_ds, options=warp_options)
        
        # Clean up the VRT
        vrt_ds = None
        try:
            gdal.GetDriverByName('VRT').Delete(vrt_path)
        except Exception:
            pass
        
        if out_ds is None:
            logger.error(f"gdal.Warp merge failed: {out_path}")
            return False
        
        out_ds = None
        return True
        
    except Exception as e:
        logger.error(f"gdal.Warp merge failed: {out_path} - {e}")
        return False


def merge_files_precision(file_paths, out_path, overwrite=False):
    """
    High-precision merge method, ensuring perfect pixel alignment.
    """
    if (not overwrite) and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True

    try:
        # Use the first file as reference
        ref_ds = gdal.Open(str(file_paths[0]), gdal.GA_ReadOnly)
        if ref_ds is None:
            logger.error(f"Cannot open reference file: {file_paths[0]}")
            return False
        
        ref_gt = ref_ds.GetGeoTransform()
        ref_proj = ref_ds.GetProjection()
        ref_pixel_x = abs(ref_gt[1])
        ref_pixel_y = abs(ref_gt[5])
        
        logger.info("Using the high-precision alignment method")
        logger.info(f"Reference pixel size: {ref_pixel_x:.10f} x {ref_pixel_y:.10f}")
        
        # Use VRT for precise merging
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.vrt', delete=False) as tmp:
            vrt_path = tmp.name
        
        # Create the VRT
        vrt_options = gdal.BuildVRTOptions(
            separate=False,
            addAlpha=False,
            resampleAlg=gdal.GRA_NearestNeighbour
        )
        
        vrt_ds = gdal.BuildVRT(vrt_path, [str(p) for p in file_paths], options=vrt_options)
        if vrt_ds is None:
            logger.error("Cannot create VRT file")
            return False
        
        # Use precise Warp parameters
        warp_options = gdal.WarpOptions(
            format='GTiff',
            creationOptions=[
                f'COMPRESS={COMPRESS}',
                'TILED=YES' if TILED else 'TILED=NO',
                f'BLOCKXSIZE={BLOCK_SIZE}',
                f'BLOCKYSIZE={BLOCK_SIZE}',
                f'BIGTIFF={BIGTIFF}',
                'NUM_THREADS=ALL_CPUS'
            ],
            multithread=True,
            warpOptions=['NUM_THREADS=ALL_CPUS'],
            # Important: use the exact resolution
            xRes=ref_pixel_x,
            yRes=ref_pixel_y,
            # Use nearest-neighbor resampling
            resampleAlg=gdal.GRA_NearestNeighbour,
            # Ensure target CRS is consistent
            dstSRS=ref_proj if ref_proj else None,
            # High-precision control
            targetAlignedPixels=True,
            errorThreshold=0.0,
            # Disable auto-adjustment
            noData=None,
            initDest=None
        )
        
        # Execute merge
        out_ds = gdal.Warp(out_path, vrt_ds, options=warp_options)
        
        # Cleanup
        vrt_ds = None
        ref_ds = None
        try:
            os.unlink(vrt_path)
        except Exception:
            pass
        
        if out_ds is None:
            logger.error(f"High-precision merge failed: {out_path}")
            return False
        
        # Validate output
        out_gt = out_ds.GetGeoTransform()
        logger.debug(f"Output geotransform: {out_gt}")
        logger.debug(f"Output pixel size: {abs(out_gt[1]):.10f} x {abs(out_gt[5]):.10f}")
        
        out_ds = None
        return True
        
    except Exception as e:
        logger.error(f"High-precision merge failed: {out_path} - {e}")
        return False


def _read_bytes(path: Path) -> bytes:
    """Read file contents into memory in one shot (large buffer)."""
    with open(path, 'rb') as f:
        return f.read()


def _write_bytes(data: bytes, dst: Path):
    """Write in-memory bytes back to NFS (atomic write: first write .tmp, then rename)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp_wb")
    try:
        with open(tmp, 'wb') as f:
            f.write(data)
        tmp.rename(dst)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# /vsimem/ path lock: GDAL vsimem is not fully thread-safe; use a fine-grained lock to protect the same-name path
_vsimem_lock = threading.Lock()


def _vsimem_path(date_str: str, name: str) -> str:
    return f"/vsimem/merge_{os.getpid()}_{date_str}_{name}"


def _vsimem_free(path: str):
    """Safely release a /vsimem/ path."""
    try:
        gdal.Unlink(path)
    except Exception:
        pass


def _merge_in_memory(
    date_str: str,
    tile_paths: list,
    merge_func_name: str,
    skip_validation: bool,
) -> bytes | None:
    """
    All-in-memory merge:
      1. Read NFS tile bytes -> gdal.FileFromMemBuffer -> /vsimem/
      2. BuildVRT + Warp entirely inside /vsimem/ (zero disk I/O)
      3. Read result bytes; release all /vsimem/ paths
      4. Return result bytes (caller writes back to NFS)
    """
    vsimem_tiles = []
    try:
        # Stage 1: read tile bytes -> /vsimem/
        for tile_path in tile_paths:
            vpath = _vsimem_path(date_str, tile_path.name)
            data = _read_bytes(tile_path)
            gdal.FileFromMemBuffer(vpath, data)
            vsimem_tiles.append(vpath)
            del data  # Release the raw bytes immediately

        if not vsimem_tiles:
            return None

        if not skip_validation:
            vsimem_path_objs = [Path(v) for v in vsimem_tiles]
            validate_geospatial_info(vsimem_path_objs)

        # Stage 2: BuildVRT (pure in-memory)
        vrt_path = _vsimem_path(date_str, "mosaic.vrt")
        vrt_options = gdal.BuildVRTOptions(
            separate=False,
            addAlpha=False,
            resampleAlg=gdal.GRA_NearestNeighbour,
        )
        vrt_ds = gdal.BuildVRT(vrt_path, vsimem_tiles, options=vrt_options)
        if vrt_ds is None:
            return None
        vrt_ds.FlushCache()

        # Stage 3: Warp -> /vsimem/ (still in memory)
        out_vpath = _vsimem_path(date_str, "merged.tif")
        creation_opts = []
        if COMPRESS:
            creation_opts.append(f'COMPRESS={COMPRESS}')
        creation_opts.extend([
            'TILED=YES',
            f'BLOCKXSIZE={BLOCK_SIZE}',
            f'BLOCKYSIZE={BLOCK_SIZE}',
            f'BIGTIFF={BIGTIFF}',
        ])

        if merge_func_name == 'precision':
            ref_ds = gdal.Open(vsimem_tiles[0], gdal.GA_ReadOnly)
            ref_gt  = ref_ds.GetGeoTransform()
            ref_proj = ref_ds.GetProjection()
            px, py = abs(ref_gt[1]), abs(ref_gt[5])
            ref_ds = None
            warp_options = gdal.WarpOptions(
                format='GTiff',
                creationOptions=creation_opts,
                resampleAlg=gdal.GRA_NearestNeighbour,
                multithread=True,
                warpOptions=['NUM_THREADS=ALL_CPUS'],
                xRes=px, yRes=py,
                dstSRS=ref_proj or None,
                targetAlignedPixels=True,
                errorThreshold=0.0,
            )
        else:
            warp_options = gdal.WarpOptions(
                format='GTiff',
                creationOptions=creation_opts,
                resampleAlg=gdal.GRA_NearestNeighbour,
                multithread=True,
                copyMetadata=False,
                options=['NUM_THREADS=ALL_CPUS'],
            )

        out_ds = gdal.Warp(out_vpath, vrt_ds, options=warp_options)
        vrt_ds = None
        _vsimem_free(vrt_path)

        if out_ds is None:
            _vsimem_free(out_vpath)
            return None

        out_ds.FlushCache()
        out_ds = None

        # Stage 4: read bytes from /vsimem/
        f = gdal.VSIFOpenL(out_vpath, 'rb')
        if f is None:
            _vsimem_free(out_vpath)
            return None
        stat = gdal.VSIStatL(out_vpath)
        result_bytes = gdal.VSIFReadL(1, stat.size, f)
        gdal.VSIFCloseL(f)
        _vsimem_free(out_vpath)

        return bytes(result_bytes)

    except Exception as e:
        logger.error(f"In-memory merge failed {date_str}: {e}")
        return None
    finally:
        for vp in vsimem_tiles:
            _vsimem_free(vp)


def process_all_pipeline(
    date_groups: dict,
    output_dir: Path,
    merge_func,
    merge_func_name: str,
    overwrite: bool,
    skip_validation: bool,
    workers: int,
    mem_limit_gb: float,
):
    """
    All-in-memory pipeline (deadlock-free version):

    Each "date task" goes through three steps end-to-end, driven by the same ThreadPoolExecutor:
      Step A: prefetch -- NFS tiles -> /vsimem/   (IO-bound, multi-thread concurrent)
      Step B: merge    -- /vsimem/ BuildVRT + Warp -> result bytes (IO/CPU-bound)
      Step C: writeback -- bytes -> NFS            (IO-bound, multi-thread concurrent)

    Flow control:
      - inflight_sem (concurrency slots): caps the number of dates in flight to prevent OOM
        limit = min(mem_limit_gb * 1024 / 30 MB_per_date, workers * 8)
      - Prefetch and merge run in the same thread pool (avoiding producer/consumer queue deadlock)
      - Writeback runs asynchronously in a separate writeback pool; the slot is released after it finishes
    """
    # Read the output directory in one shot (avoiding per-file NFS stat, each of which may stall several seconds)
    print(f"[INFO] scanning output directory...", end="", flush=True)
    t_scan = time.time()
    try:
        existing_outputs = {
            e.name
            for e in os.scandir(output_dir)
            if e.is_file() and e.stat().st_size > 0
        }
    except FileNotFoundError:
        existing_outputs = set()
    print(f" {len(existing_outputs)} completed files ({time.time()-t_scan:.1f}s）", flush=True)

    # Filter out completed
    pending = {}
    for date_str, files in sorted(date_groups.items()):
        if (not overwrite) and f"{date_str}.tif" in existing_outputs:
            continue
        pending[date_str] = files

    total = len(pending)
    if total == 0:
        print("All dates already processed; no reprocessing needed.")
        return 0, 0

    # Concurrency slot: ~30 MB per date (input ~20 MB + output ~13 MB)
    avg_mb_per_date  = 30
    slots_by_ram     = max(1, int(mem_limit_gb * 1024 / avg_mb_per_date))
    inflight         = min(slots_by_ram, workers * 8)
    inflight_sem     = threading.Semaphore(inflight)

    # Two thread pools: merge pool (A+B), writeback pool (C)
    merge_pool   = ThreadPoolExecutor(max_workers=workers,          thread_name_prefix="merge")
    writeback_pool = ThreadPoolExecutor(max_workers=max(8, workers), thread_name_prefix="wb")

    stats      = {'ok': 0, 'fail': 0}
    stats_lock = threading.Lock()
    t_start    = time.time()

    def _print_progress():
        with stats_lock:
            done = stats['ok'] + stats['fail']
        elapsed = time.time() - t_start
        rate    = done / elapsed if elapsed > 0 else 0
        eta     = (total - done) / rate if rate > 0 else 0
        print(f"\r[{done}/{total}] OK:{stats['ok']} FAIL:{stats['fail']} "
              f"{rate:.1f}dates/s ETA:{eta:.0f}s   ", end="", flush=True)

    # ------------------------------------------------------------------ #
    # Step B: BuildVRT + Warp inside /vsimem/, return result bytes              #
    # ------------------------------------------------------------------ #
    def _merge_vsimem(date_str: str) -> bytes | None:
        tile_vpaths = [_vsimem_path(date_str, p.name) for p in pending[date_str]]
        try:
            vrt_path = _vsimem_path(date_str, "mosaic.vrt")
            vrt_ds = gdal.BuildVRT(vrt_path, tile_vpaths,
                                   options=gdal.BuildVRTOptions(
                                       separate=False, addAlpha=False,
                                       resampleAlg=gdal.GRA_NearestNeighbour))
            if vrt_ds is None:
                return None
            vrt_ds.FlushCache()

            out_vpath    = _vsimem_path(date_str, "merged.tif")
            creation_opts = []
            if COMPRESS:
                creation_opts.append(f'COMPRESS={COMPRESS}')
            creation_opts.extend([
                'TILED=YES', f'BLOCKXSIZE={BLOCK_SIZE}',
                f'BLOCKYSIZE={BLOCK_SIZE}', f'BIGTIFF={BIGTIFF}',
            ])

            if merge_func_name == 'precision':
                ref_ds   = gdal.Open(tile_vpaths[0], gdal.GA_ReadOnly)
                ref_gt   = ref_ds.GetGeoTransform()
                ref_proj = ref_ds.GetProjection()
                ref_ds   = None
                warp_opts = gdal.WarpOptions(
                    format='GTiff', creationOptions=creation_opts,
                    resampleAlg=gdal.GRA_NearestNeighbour, multithread=True,
                    warpOptions=['NUM_THREADS=ALL_CPUS'],
                    xRes=abs(ref_gt[1]), yRes=abs(ref_gt[5]),
                    dstSRS=ref_proj or None,
                    targetAlignedPixels=True, errorThreshold=0.0,
                )
            else:
                warp_opts = gdal.WarpOptions(
                    format='GTiff', creationOptions=creation_opts,
                    resampleAlg=gdal.GRA_NearestNeighbour, multithread=True,
                    copyMetadata=False, options=['NUM_THREADS=ALL_CPUS'],
                )

            out_ds = gdal.Warp(out_vpath, vrt_ds, options=warp_opts)
            vrt_ds = None
            _vsimem_free(vrt_path)

            if out_ds is None:
                _vsimem_free(out_vpath)
                return None
            out_ds.FlushCache()
            out_ds = None

            fh   = gdal.VSIFOpenL(out_vpath, 'rb')
            stat = gdal.VSIStatL(out_vpath)
            res  = bytes(gdal.VSIFReadL(1, stat.size, fh))
            gdal.VSIFCloseL(fh)
            _vsimem_free(out_vpath)
            return res

        except Exception as e:
            logger.error(f"vsimem merge failed {date_str}: {e}")
            return None
        finally:
            for vp in tile_vpaths:
                _vsimem_free(vp)

    # ------------------------------------------------------------------ #
    # Steps A+B combined into one task function, submitted to merge_pool       #
    # ------------------------------------------------------------------ #
    def _prefetch_and_merge(date_str: str, tile_paths: list) -> bytes | None:
        """A: read NFS -> vsimem; B: Warp -> bytes. None means failure."""
        try:
            for tile_path in tile_paths:
                vpath = _vsimem_path(date_str, tile_path.name)
                gdal.FileFromMemBuffer(vpath, _read_bytes(tile_path))
        except Exception as e:
            logger.error(f"Prefetch failed {date_str}: {e}")
            for tp in tile_paths:
                _vsimem_free(_vsimem_path(date_str, tp.name))
            return None
        return _merge_vsimem(date_str)

    # ------------------------------------------------------------------ #
    # Step C: write back to NFS, release the concurrency slot afterwards       #
    # ------------------------------------------------------------------ #
    def _on_merge_done(fut: Future, date_str: str):
        """Callback when a merge_pool task completes (runs in the calling thread)."""
        try:
            result = fut.result()
        except Exception as e:
            result = None
            logger.error(f"Merge crashed {date_str}: {e}")

        if result is not None:
            with stats_lock:
                stats['ok'] += 1
            # Async writeback; release the slot only after it completes
            wb_fut = writeback_pool.submit(_writeback, date_str, result)
            wb_fut.add_done_callback(lambda _: inflight_sem.release())
        else:
            with stats_lock:
                stats['fail'] += 1
            inflight_sem.release()
        _print_progress()

    def _writeback(date_str: str, data: bytes):
        try:
            _write_bytes(data, output_dir / f"{date_str}.tif")
        except Exception as e:
            logger.warning(f"Writeback failed {date_str}: {e}")
        finally:
            del data

    # ------------------------------------------------------------------ #
    # Main loop: submit all date tasks one by one, throttled by inflight_sem  #
    # ------------------------------------------------------------------ #
    print(f"[INFO] in-memory mode: concurrency slots={inflight} (cap {mem_limit_gb:.0f}GB）  "
          f"merge threads={workers}  writeback threads={max(8,workers)}")

    all_futs = []
    for date_str, tile_paths in pending.items():
        inflight_sem.acquire()          # Wait for a free slot (backpressure)
        fut = merge_pool.submit(_prefetch_and_merge, date_str, tile_paths)
        fut.add_done_callback(lambda f, d=date_str: _on_merge_done(f, d))
        all_futs.append(fut)

    # Wait for everything to complete
    concurrent.futures.wait(all_futs)
    merge_pool.shutdown(wait=False)
    writeback_pool.shutdown(wait=True)   # Ensure all writebacks finish

    print()
    return stats['ok'], stats['fail']


def main():
    # Declare global at the top of the function to avoid later-use syntax errors
    global GDAL_CACHE_SIZE, LARGE_MEMORY_BLOCK_SIZE
    
    parser = argparse.ArgumentParser(description='Merge multiple TIFF files within a folder')
    
    parser.add_argument('--input-dir',default=r'/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/MCD14A1_combined',
                       help='Input directory containing TIFF files')
    parser.add_argument('--output-dir', default=r'/s3/user/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/MCD14A1_combined_merged',
                       help='Output directory')
    parser.add_argument('--output-name', default=None,
                       help='Output filename (default: auto-extract the date from input filenames)')
    parser.add_argument('--date', default=None,
                       help='Manually specify a date (format: YYYY_MM_DD, used when filenames lack a full date)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing output files')
    parser.add_argument('--workers', type=int, default=max(1, os.cpu_count() // 2),
                       help=f'Parallel worker threads (default: {max(1, os.cpu_count() // 2)})')
    parser.add_argument('--method', default='warp',
                       choices=['gdal', 'warp', 'precision'],
                       help='Merge method: gdal=manual merge, warp=gdal.Warp, precision=high-precision alignment (default: warp)')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip geospatial-info validation for speed (not recommended, but a significant speedup)')
    # Save defaults for use in argument definitions
    default_gdal_cache = GDAL_CACHE_SIZE
    default_block_size = LARGE_MEMORY_BLOCK_SIZE
    
    parser.add_argument('--gdal-cache', type=int, default=default_gdal_cache,
                       help=f'GDAL cache size (MB, default: {default_gdal_cache}MB = {default_gdal_cache//1024}GB)')
    parser.add_argument('--block-size', type=int, default=default_block_size,
                       help=f'Processing block size (pixels, default: {default_block_size}x{default_block_size})')
    parser.add_argument('--no-compress', action='store_true',
                       help='Disable compression to speed up processing (files will be larger but processing is faster)')
    parser.add_argument('--mem-limit', type=float, default=100.0,
                       help='Memory mode: maximum memory allowed (GB, default 50 GB). Controls the data volume kept in /vsimem/ simultaneously.')
    parser.add_argument('--verbose', action='store_true',
                       help='Show verbose geospatial info (for debugging)')
    
    args = parser.parse_args()
    
    # Apply the user-specified GDAL cache size
    if args.gdal_cache != GDAL_CACHE_SIZE:
        GDAL_CACHE_SIZE = args.gdal_cache
        gdal.SetCacheMax(GDAL_CACHE_SIZE * 1024 * 1024)
        gdal.SetConfigOption('GDAL_CACHEMAX', str(GDAL_CACHE_SIZE))
        logger.info(f"GDAL cache size set to: {GDAL_CACHE_SIZE}MB ({GDAL_CACHE_SIZE//1024}GB)")
    
    # Apply the user-specified block size
    if args.block_size != LARGE_MEMORY_BLOCK_SIZE:
        LARGE_MEMORY_BLOCK_SIZE = args.block_size
        logger.info(f"Processing block size set to: {LARGE_MEMORY_BLOCK_SIZE}x{LARGE_MEMORY_BLOCK_SIZE}")
    
    # Apply compression settings
    global COMPRESS
    if args.no_compress:
        COMPRESS = None
        logger.info("Compression disabled to speed up processing")
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group TIFF files by date
    try:
        date_groups = group_files_by_date(args.input_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    if not date_groups:
        logger.error(f"In directory {args.input_dir}: no valid TIFF files found")
        sys.exit(1)

    total_dates = len(date_groups)
    total_files = sum(len(files) for files in date_groups.values())
    print(f"Found {total_dates} dates of data, total {total_files} files")

    # Choose the merge method
    if args.method == 'gdal':
        merge_func = merge_files_gdal
    elif args.method == 'precision':
        merge_func = merge_files_precision
    else:
        merge_func = merge_files_gdal_warp

    # --output-name processes only the first date (single-file mode, original semantics)
    if args.output_name:
        first_date = sorted(date_groups.keys())[0]
        date_groups = {first_date: date_groups[first_date]}
        # Override the output filename
        output_name = args.output_name if args.output_name.endswith('.tif') else args.output_name + '.tif'
        # Single date: just run sequentially
        tiff_files = date_groups[first_date]
        out_path = output_dir / output_name
        if (not args.overwrite) and out_path.exists() and out_path.stat().st_size > 0:
            print(f"Already exists; skipping: {out_path}")
        else:
            ok = merge_func(tiff_files, str(out_path), args.overwrite)
            print("done" if ok else "failed")
        return

    start_time = time.time()
    ok, fail = process_all_pipeline(
        date_groups=date_groups,
        output_dir=output_dir,
        merge_func=merge_func,
        merge_func_name=args.method,
        overwrite=args.overwrite,
        skip_validation=args.skip_validation,
        workers=args.workers,
        mem_limit_gb=args.mem_limit,
    )
    elapsed_time = time.time() - start_time

    print(f"\nProcessing complete! Total elapsed: {format_time(elapsed_time)}")
    print(f"Succeeded: {ok}/{total_dates} dates  failed: {fail}")
    if fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
