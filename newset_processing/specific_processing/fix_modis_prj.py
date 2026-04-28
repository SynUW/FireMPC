#!/usr/bin/env python3
"""
Check and fix MODIS projection.
The source is Sinusoidal but is written as a "pseudo-Sinusoidal" with +datum=WGS84 and is missing +R=6371007.181 (the spherical radius MODIS uses). This causes systematic offsets when reprojecting to WGS84.
"""
from osgeo import gdal, osr
from pathlib import Path
import time
import sys
from multiprocessing import Pool, Manager
from functools import partial

# Suppress SWIG memory-leak warnings
if hasattr(sys, 'stderr'):
    _stderr = sys.stderr
    sys.stderr = open('/dev/null', 'w')
    import warnings
    warnings.filterwarnings('ignore')
    sys.stderr = _stderr

ROOT = Path("/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/trash")
FIX_WKT_STR = None

def _fmt_time(s: float) -> str:
    """Format time display"""
    s = int(s)
    return f"{s//3600}:{(s%3600)//60:02d}:{s%60:02d}"

def need_fix(proj4: str) -> bool:
    if not proj4: return False
    p = proj4.lower()
    return ("+proj=sinu" in p) and (("+datum=wgs84" in p) or ("+r=" not in p))

def process_single_file(tif_path, fix_wkt_str):
    """
    Process a single TIFF file.
    Returns: (status, message)
        status: 'fixed', 'normal', 'skipped', 'error'
    """
    try:
        ds = gdal.Open(str(tif_path), gdal.GA_Update)
        if ds is None:
            return ('error', f"Cannot open: {tif_path.name}")
        
        s = osr.SpatialReference()
        try:
            s.ImportFromWkt(ds.GetProjection() or "")
            p4 = s.ExportToProj4() or ""
        except Exception as e:
            return ('error', f"Failed to read projection: {tif_path.name}")
        finally:
            del s
        
        if need_fix(p4):
            ds.SetProjection(fix_wkt_str)
            ds.FlushCache()
            ds = None
            return ('fixed', tif_path.name)
        else:
            ds = None
            return ('normal', tif_path.name)
            
    except Exception as e:
        return ('error', f"Processing exception: {tif_path.name} - {str(e)}")

def main():
    global FIX_WKT_STR
    
    # Prepare the projection string
    FIX_WKT = osr.SpatialReference()
    FIX_WKT.ImportFromProj4('+proj=sinu +R=6371007.181 +lon_0=0 +x_0=0 +y_0=0 +no_defs')
    FIX_WKT_STR = FIX_WKT.ExportToWkt()
    
    # Collect all TIFF files
    print("Scanning TIFF files...")
    print(f"Root directory: {ROOT}")
    
    subfolders = [d for d in ROOT.iterdir() if d.is_dir()]
    print(f"Found {len(subfolders)} subfolders")
    
    all_tifs = list(ROOT.rglob("*.tif"))
    total = len(all_tifs)
    
    if total == 0:
        print("No TIFF files found")
        exit(1)
    
    print(f"Total {total} TIFF files")
    
    # Determine the number of processes
    import os
    n_workers = min(os.cpu_count() or 4, 16)  # At most 16 processes
    print(f"Using {n_workers} processes in parallel")
    print("=" * 60)
    
    # Process files
    t0 = time.time()
    done = 0
    fixed = 0
    normal = 0
    error = 0
    
    # Use partial to fix the fix_wkt_str argument
    process_func = partial(process_single_file, fix_wkt_str=FIX_WKT_STR)
    
    with Pool(processes=n_workers) as pool:
        for status, msg in pool.imap_unordered(process_func, all_tifs):
            done += 1
            
            if status == 'fixed':
                fixed += 1
            elif status == 'normal':
                normal += 1
            elif status == 'error':
                error += 1
            
            # Update progress
            elapsed = time.time() - t0
            eta = elapsed * (total - done) / max(1, done)
            avg_time = elapsed / done
            
            print(f"\rProgress [{done}/{total}] | "
                  f"Fixed: {fixed} normal: {normal} errors: {error} | "
                  f"Avg: {avg_time:.3f} s/file | "
                  f"Elapsed: {_fmt_time(elapsed)} "
                  f"ETA: {_fmt_time(eta)}", end='', flush=True)
    
    print(f"\n\n{'='*60}")
    print(f"Processing complete!")
    print(f"   - subfolder: {len(subfolders)}")
    print(f"   - Total files: {total}")
    print(f"   - fixed: {fixed} (had issues and was fixed)")
    print(f"   - normal: {normal} (projection correct)")
    print(f"   - Errors: {error} (cannot process)")
    print(f"\nTotal elapsed: {_fmt_time(time.time() - t0)}")
    print(f"Average speed: {total / (time.time() - t0):.1f} files/sec")
    print("=" * 60)

if __name__ == '__main__':
    main()
