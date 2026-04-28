#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-in-one GDAL raster processing (reference-grid alignment + final clip version)
========================================================================
Goals and principles
------------------------------------------------------------------------
1) **Use the reference TIF (--grid-template) as the single source of truth**:
   - Only its CRS, pixel size (xRes/yRes), origin, and extent are used.
   - All input images are first reprojected / resampled and aligned to a grid identical to the reference TIF (same origin, same resolution, same extent).

2) **Do not change origin or extent during the alignment stage**:
   - Step 1 (align to grid) only does: reprojection + resampling to the **grid-template resolution**,
     and **forces the output bounds** to equal the **grid-template bounds**, producing pixel-aligned, size-equal, extent-equal output relative to the reference TIF.
   - This guarantees that the alignment stage preserves origin and extent (matching the reference TIF exactly).

3) **Use a shapefile for the final clip**:
   - Step 2 (clip) takes the aligned data from step 1 and clips it with a shapefile, producing the final extent-consistent output.
   - This step changes the output bounds (because of clipping) while keeping the alignment-stage resolution and pixel alignment (xRes/yRes and targetAlignedPixels).

4) **Resampling strategy (by input directory name)**:
   - If the **last folder** of the --input-root path contains `12Q1` or `DEMs` (case-insensitive) -> nearest neighbor `near`.
   - Otherwise -> bilinear.

5) **No QA / no LAI branches, no Excel**: NoData comes only from per-band GeoTIFF metadata; QA bands are not specially handled and there is no LAI sub-path composition.
   - LST (MOD11A1 / MYD11A1) output is still forced to UInt16.

The only differences from the old version
------------------------------------------------------------------------
- Removed `--proj-template` (no more "two Warps"); all alignment uses **--grid-template** as the single source of truth.
- The alignment stage **no longer** uses a cutline (no clipping); a separate clip Warp runs only **after alignment**.
- The alignment-stage output exactly matches the reference TIF's origin / extent / resolution.

Usage (default parameters preserved)
------------------------------------------------------------------------
Example:
    python tri_pipeline_align_then_clip.py \\
        --input-root /path/to/in\\
        --output-root /path/to/out\\
        --grid-template /path/to/reference.tif\\
        --shapefile /path/to/clip.shp\\
        --threads 4 --compress LZW --blocksize 512

Notes:
- If the input is in MODIS Sinusoidal, ensure its SRS is correctly defined (including R = 6371007.181).
- To verify alignment alone, omit --shapefile temporarily; the output will then match the reference TIF's bounds exactly.
========================================================================
"""

import os, sys, argparse, shutil, tempfile, threading, traceback, gc
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from osgeo import gdal, osr

# ----------------- Global GDAL configuration -----------------
gdal.UseExceptions()
gdal.SetConfigOption("GDAL_PAM_ENABLED", "NO")      # Disable writing / reading .aux.xml
gdal.SetConfigOption("GTIFF_INTERNAL_MASK", "YES")  # Write masks inside the TIF
gdal.SetConfigOption("CPL_MAX_ERROR_REPORTS", "100")
gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
gdal.SetConfigOption("NUM_THREADS", "ALL_CPUS")

progress_lock = threading.Lock()

# ----------------- Utility functions -----------------
def gdal_dtype_numpy(gdt):
    mapping = {
        gdal.GDT_Byte: np.uint8,
        gdal.GDT_UInt16: np.uint16,
        gdal.GDT_Int16: np.int16,
        gdal.GDT_UInt32: np.uint32,
        gdal.GDT_Int32: np.int32,
        gdal.GDT_Float32: np.float32,
        gdal.GDT_Float64: np.float64,
    }
    if gdt not in mapping: raise RuntimeError(f"Unsupported GDAL data type: {gdt}")
    return mapping[gdt]

def dtype_range_for_gdt(gdt):
    if gdt == gdal.GDT_Byte:    return (0, 255)
    if gdt == gdal.GDT_UInt16:  return (0, 65535)
    if gdt == gdal.GDT_Int16:   return (-32768, 32767)
    if gdt == gdal.GDT_UInt32:  return (0, 4294967295)
    if gdt == gdal.GDT_Int32:   return (-2147483648, 2147483647)
    if gdt == gdal.GDT_Float32: return (None, None)
    if gdt == gdal.GDT_Float64: return (None, None)
    return (None, None)

def choose_storage_gdt(src_gdt, nd_values):
    """Choose the output dtype based on nodata (no pixel-value modification).
       Keep the original nodata value; do not force a conversion to 0."""
    nd_values = list(nd_values) if nd_values else []

    # Keep float
    if any(isinstance(v, float) and not float(v).is_integer() for v in nd_values):
        return gdal.GDT_Float32 if src_gdt != gdal.GDT_Float64 else gdal.GDT_Float64
    if src_gdt in (gdal.GDT_Float32, gdal.GDT_Float64):
        return src_gdt

    vmin, vmax = dtype_range_for_gdt(src_gdt)
    ok = True
    for v in nd_values:
        vv = int(v) if (isinstance(v,(int,np.integer)) or float(v).is_integer()) else v
        if isinstance(vv, float): return gdal.GDT_Float32
        if vmin is not None and (vv < vmin or vv > vmax): ok = False; break
    if ok: return src_gdt

    # Promote
    if src_gdt == gdal.GDT_Byte:   return gdal.GDT_Int16
    if src_gdt == gdal.GDT_UInt16: return gdal.GDT_Int32
    if src_gdt == gdal.GDT_Int16:  return gdal.GDT_Int32
    if src_gdt == gdal.GDT_UInt32: return gdal.GDT_Float64
    if src_gdt == gdal.GDT_Int32:  return gdal.GDT_Float64
    return src_gdt

def open_ds(path: Path, readonly=True):
    ds = gdal.Open(str(path), gdal.GA_ReadOnly if readonly else gdal.GA_Update)
    if ds is None: raise RuntimeError(f"Cannot open image: {path}")
    return ds

def get_srs_wkt(ds):
    wkt = ds.GetProjection()
    if not wkt: return ""
    sref = osr.SpatialReference(); sref.ImportFromWkt(wkt)
    return sref.ExportToWkt()

def get_bounds_size_gt(ds):
    gt = ds.GetGeoTransform(); w, h = ds.RasterXSize, ds.RasterYSize
    xs = [0, w, 0, w]; ys = [0, 0, h, h]
    def pix2geo(px, py): return gt[0]+px*gt[1]+py*gt[2], gt[3]+px*gt[4]+py*gt[5]
    coords = [pix2geo(px,py) for px,py in zip(xs,ys)]
    xs = [c[0] for c in coords]; ys = [c[1] for c in coords]
    xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
    return (xmin, ymin, xmax, ymax), (w, h), gt

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def close_ds(ds):
    try:
        if ds: ds.FlushCache()
    except Exception:
        pass
    return None

def unlink_safe(p: Path):
    try:
        if p and p.exists(): p.unlink()
    except Exception: pass

def resample_alg_for_subfolder(folder_name: str) -> str:
    """If the folder name (typically the last level of --input-root) contains 12Q1 or DEMs (case-insensitive) -> near; otherwise bilinear."""
    n = folder_name.lower()
    if "12q1" in n or "dems" in n:
        return "near"
    return "bilinear"

def read_as_array_compat(band, xoff, yoff, xsize, ysize):
    try:
        return band.ReadAsArray(xoff=xoff, yoff=yoff, win_xsize=xsize, win_ysize=ysize)
    except TypeError:
        try:
            return band.ReadAsArray(xoff, yoff, xsize, ysize)
        except TypeError:
            buf = band.ReadRaster(xoff, yoff, xsize, ysize,
                                  buf_xsize=xsize, buf_ysize=ysize,
                                  buf_type=band.DataType)
            if buf is None: return None
            np_dtype = gdal_dtype_numpy(band.DataType)
            arr = np.frombuffer(buf, dtype=np_dtype)
            return arr.reshape(ysize, xsize)

def write_fail_logs(csv_path: Path, txt_path: Path, records: list):
    if not records: return
    import csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['time','folder','file','step','error','gdal_last_error'])
        w.writeheader()
        for r in records: w.writerow({k:r.get(k) for k in w.fieldnames})
    with open(txt_path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(f"[{r['time']}] {r['folder']}/{r['file']}\nStep: {r['step']}\nError: {r['error']}\n")
            if r.get('gdal_last_error'): f.write(f"GDAL: {r['gdal_last_error']}\n")
            if r.get('traceback'): f.write("Traceback:\n"+r['traceback'])
            f.write("\n"+"-"*80+"\n")

# ----------------- Pre-cleaning to a temporary GTiff -----------------
def preclean_to_temp_no_modify(
    src_path: Path,
    qa_bands_0based,
    temp_path: Path,
    compress='NONE',
    blocksize=512,
    apply_nd_to_bands=None
):
    src = open_ds(src_path, readonly=True)
    w, h = src.RasterXSize, src.RasterYSize
    src_gdt = src.GetRasterBand(1).DataType
    nbands = src.RasterCount

    # Use only the per-band NoData from GeoTIFF metadata
    tif_nodata_values = []
    for b in range(nbands):
        band = src.GetRasterBand(b + 1)
        nodata = band.GetNoDataValue()
        if nodata is not None:
            tif_nodata_values.append(nodata)
    nd_values = list(set(tif_nodata_values))

    out_gdt = choose_storage_gdt(src_gdt, nd_values)

    # LST forced to UInt16
    sp_lc = str(src_path).lower()
    if ('mod11a1' in sp_lc or 'myd11a1' in sp_lc):
        out_gdt = gdal.GDT_UInt16

    drv = gdal.GetDriverByName('GTiff')
    ensure_dir(temp_path.parent)
    creation = ['TILED=YES', f'BLOCKXSIZE={blocksize}', f'BLOCKYSIZE={blocksize}',
                f'COMPRESS={compress}', 'BIGTIFF=IF_SAFER']
    dst = drv.Create(str(temp_path), w, h, nbands, out_gdt, options=creation)
    if dst is None:
        src = close_ds(src)
        raise RuntimeError(f"Cannot create temp file: {temp_path}")
    dst.SetGeoTransform(src.GetGeoTransform())
    dst.SetProjection(src.GetProjection())

    qa_set = set(qa_bands_0based)
    apply_nd_set = set(apply_nd_to_bands) if apply_nd_to_bands is not None else set(i for i in range(nbands) if i not in qa_set)

    canonical_nd = nd_values[0] if len(nd_values)>0 else -32768
    # Keep the original nodata value; use -32768 if not set

    # Set NoData on non-QA bands only
    for b in range(nbands):
        if b in apply_nd_set:
            dst.GetRasterBand(b+1).SetNoDataValue(
                float(canonical_nd) if out_gdt in (gdal.GDT_Float32,gdal.GDT_Float64) else int(canonical_nd)
            )

    bx, by = src.GetRasterBand(1).GetBlockSize()
    bx = bx or blocksize; by = by or blocksize

    compare_as_float = any(isinstance(v, float) and not float(v).is_integer() for v in nd_values)
    nd_vals_np = np.array(nd_values, dtype=np.float64 if compare_as_float else np.int64) if len(nd_values)>0 else None

    if nd_vals_np is not None and nd_vals_np.size>0:
        for b in range(nbands):
            if b in apply_nd_set:
                rb = dst.GetRasterBand(b+1)
                try:
                    rb.CreateMaskBand(0)
                except Exception:
                    pass

    for y0 in range(0, h, by):
        nrows = by if y0 + by <= h else h - y0
        for x0 in range(0, w, bx):
            ncols = bx if x0 + bx <= w else w - x0
            for b in range(nbands):
                band = src.GetRasterBand(b+1)
                arr  = read_as_array_compat(band, x0, y0, ncols, nrows)
                if arr is None:
                    arr = np.zeros((nrows,ncols), dtype=gdal_dtype_numpy(src_gdt))
                # Integer & non-QA: keep raw values; no non-negative cleansing
                # Non-negative cleansing logic is commented out; keep raw pixel values
                dst.GetRasterBand(b+1).WriteArray(arr, xoff=x0, yoff=y0)

                if (nd_vals_np is not None and nd_vals_np.size>0) and (b in apply_nd_set):
                    rb = dst.GetRasterBand(b+1)
                    try:
                        rb.CreateMaskBand(0)
                    except Exception:
                        pass
                    mb = rb.GetMaskBand()
                    if compare_as_float: carr = arr.astype(np.float64, copy=False)
                    else: carr = arr.astype(np.int64, copy=False)
                    valid = np.ones_like(carr, dtype=np.uint8)
                    for v in nd_vals_np: valid &= (carr != v)
                    mask_block = valid * 255
                    mb.WriteArray(mask_block, xoff=x0, yoff=y0)
                    del mask_block, valid, carr
                del arr
        gc.collect()

    for b in range(nbands):
        dst.GetRasterBand(b+1).FlushCache()
        m = dst.GetRasterBand(b+1).GetMaskBand()
        if m: m.FlushCache()
    dst.FlushCache()
    src = close_ds(src); dst = close_ds(dst)
    return out_gdt, canonical_nd

# ----------------- Subset VRT (disk) -----------------
def translate_subset_to_vrt_disk(tmp_dir: Path, name: str, src_ds, band_indices_0based):
    if len(band_indices_0based) == 0: return None, None
    out_vrt = tmp_dir / f"{name}.vrt"
    if out_vrt.exists(): unlink_safe(out_vrt)
    band_list_1b = [i+1 for i in band_indices_0based]
    try:
        topts = gdal.TranslateOptions(format='VRT', bandList=band_list_1b)
        vrt = gdal.Translate(str(out_vrt), src_ds, options=topts)
    except TypeError:
        args = ['-of', 'VRT']; args += sum((['-b', str(b)] for b in band_list_1b), [])
        vrt = gdal.Translate(str(out_vrt), src_ds, options=args)
    if vrt is None: raise RuntimeError("gdal.Translate failed")
    return vrt, out_vrt

# ----------------- Build the LAI mask temp file -----------------
def build_lai_masked_temp(src_path: Path, lai_indices, nodata_val, mode, out_path: Path, compress='NONE', blocksize=512):
    """
    Extract the LAI band from src_path and build a masked version per mode ('pos' / 'neg'):
    - 'pos': keep values >= 0; set < 0 to nodata
    - 'neg': keep values < 0; set >= 0 to nodata
    """
    src_ds = open_ds(src_path, readonly=True)
    if src_ds is None:
        raise RuntimeError(f"Cannot open source file: {src_path}")
    
    width = src_ds.RasterXSize
    height = src_ds.RasterYSize
    n_lai_bands = len(lai_indices)
    
    if n_lai_bands == 0:
        src_ds = close_ds(src_ds)
        return
    
    # Get the dtype of the first LAI band
    first_lai_band = src_ds.GetRasterBand(lai_indices[0] + 1)
    gdt = first_lai_band.DataType
    np_dtype = gdal_dtype_numpy(gdt)
    
    # Create output file
    driver = gdal.GetDriverByName('GTiff')
    co = [f'COMPRESS={compress}', f'BLOCKXSIZE={blocksize}', f'BLOCKYSIZE={blocksize}', 'TILED=YES', 'BIGTIFF=IF_SAFER']
    dst_ds = driver.Create(str(out_path), width, height, n_lai_bands, gdt, options=co)
    if dst_ds is None:
        src_ds = close_ds(src_ds)
        raise RuntimeError(f"Cannot create output file: {out_path}")
    
    # Copy georeferencing
    dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
    dst_ds.SetProjection(src_ds.GetProjection())
    
    # Process each LAI band
    for out_band_idx, src_band_idx in enumerate(lai_indices):
        src_band = src_ds.GetRasterBand(src_band_idx + 1)
        dst_band = dst_ds.GetRasterBand(out_band_idx + 1)
        
        # Set nodata (keep raw values; do not constrain to >= 0)
        if nodata_val is not None:
            dst_band.SetNoDataValue(float(nodata_val))
        
        # Block-wise processing
        for y in range(0, height, blocksize):
            ysize = min(blocksize, height - y)
            for x in range(0, width, blocksize):
                xsize = min(blocksize, width - x)
                
                # Read data
                data = read_as_array_compat(src_band, x, y, xsize, ysize)
                if data is None:
                    continue
                
                # Apply mask
                if mode == 'pos':
                    # Keep values >= 0
                    mask = data < 0
                elif mode == 'neg':
                    # Keep values < 0
                    mask = data >= 0
                else:
                    raise ValueError(f"Unknown mode: {mode}; expected 'pos' or 'neg'")
                
                # Set the masked region to nodata (keep raw values elsewhere)
                if nodata_val is not None:
                    data[mask] = nodata_val
                
                # Write data
                dst_band.WriteArray(data, x, y)
        
        dst_band.FlushCache()
    
    dst_ds.FlushCache()
    src_ds = close_ds(src_ds)
    dst_ds = close_ds(dst_ds)

# ----------------- Align to the reference grid (no clipping; do not change origin or extent) -----------------
def align_to_grid_vrt(tmp_dir: Path, name: str, src_ds, grid_wkt, grid_bounds, grid_size_wh,
                      resample_alg, canonical_nd=None, extra_warp_opts=None):
    """
    Reproject / resample src_ds to a grid identical to the reference TIF:
      - dstSRS = grid_wkt
      - outputBounds = grid_bounds
      - width / height = grid_size_wh
      => so the output's origin / extent / resolution **exactly match** the reference TIF.
    """
    out_vrt = tmp_dir / f"{name}.vrt"
    if out_vrt.exists(): unlink_safe(out_vrt)

    xmin,ymin,xmax,ymax = grid_bounds
    width,height        = grid_size_wh
    dst_nd = canonical_nd if canonical_nd is not None else None

    # Use -32768 as the initialization value instead of 0
    init_dest_val = -32768 if dst_nd is None else dst_nd
    warp_opts = [f'INIT_DEST={init_dest_val}', 'UNIFIED_SRC_NODATA=YES']
    if extra_warp_opts: warp_opts.extend(extra_warp_opts)

    wopts = gdal.WarpOptions(
        format='VRT', dstSRS=grid_wkt,
        outputBounds=(xmin,ymin,xmax,ymax), width=width, height=height,
        resampleAlg=resample_alg,
        warpOptions=warp_opts,
        srcNodata=dst_nd, dstNodata=dst_nd
    )
    vrt = gdal.Warp(str(out_vrt), src_ds, options=wopts)
    if vrt is None: raise RuntimeError("Warp failed (aligning to reference grid)")
    return vrt, out_vrt

# ----------------- Final clip (keep resolution and pixel alignment; do not require bounds equal to the template) -----------------
def clip_with_shapefile_vrt(tmp_dir: Path, name: str, src_ds, grid_wkt, xRes, yRes,
                             clipline_path, resample_alg, canonical_nd=None):
    """
    Take a VRT "already aligned to the reference grid" and clip it with a shapefile.
    - Keep the same SRS (grid_wkt), the same pixel resolution (xRes / yRes), and targetAlignedPixels=True.
    - Do not specify outputBounds; GDAL uses the post-clip minimum bounding rectangle as the new bounds (so bounds shrink; this is expected).
    """
    out_vrt = tmp_dir / f"{name}.vrt"
    if out_vrt.exists(): unlink_safe(out_vrt)

    dst_nd = canonical_nd if canonical_nd is not None else None
    # Use -32768 as the initialization value instead of 0
    init_dest_val = -32768 if dst_nd is None else dst_nd
    wopts = gdal.WarpOptions(
        format='VRT', dstSRS=grid_wkt,
        xRes=xRes, yRes=yRes, targetAlignedPixels=True,
        resampleAlg=resample_alg,
        cutlineDSName=str(clipline_path) if clipline_path else None,
        cropToCutline=True if clipline_path else False,
        warpOptions=[f'INIT_DEST={init_dest_val}', 'UNIFIED_SRC_NODATA=YES'],
        srcNodata=dst_nd, dstNodata=dst_nd
    )
    vrt = gdal.Warp(str(out_vrt), src_ds, options=wopts)
    if vrt is None: raise RuntimeError("Warp failed (clipping by shapefile)")
    return vrt, out_vrt

# ----------------- Final block write -----------------
def write_final_with_blocks(out_path: Path, gdt, grid_gt, grid_wkt, nbands,
                            qa_vrt, nonlai_vrt, lai_pos_vrt, lai_neg_vrt, lai_nn_vrt, lai_indices,
                            qa_indices, nonlai_indices,
                            canonical_nd, compress='LZW', blocksize=512,
                            era5_mode=False, out_nodata_override=None):
    drv = gdal.GetDriverByName('GTiff')
    ensure_dir(out_path.parent)
    dst = drv.Create(str(out_path), grid_gt['w'], grid_gt['h'], nbands, gdt,
                     options=['TILED=YES', f'BLOCKXSIZE={blocksize}', f'BLOCKYSIZE={blocksize}',
                              f'COMPRESS={compress}', 'BIGTIFF=IF_SAFER'])
    if dst is None: raise RuntimeError(f"Cannot create output: {out_path}")
    dst.SetProjection(grid_wkt); dst.SetGeoTransform(grid_gt['gt'])

    qa_set=set(qa_indices); lai_set=set(lai_indices)
    qa_pos     = {i:k+1 for k,i in enumerate(qa_indices)} if qa_indices else {}
    nonlai_pos = {i:k+1 for k,i in enumerate(nonlai_indices)} if nonlai_indices else {}
    lai_pos_map= {i:k+1 for k,i in enumerate(lai_indices)} if lai_indices else {}

    nd_out = canonical_nd
    if nd_out is None: nd_out = -32768
    # Keep the original nodata value; use -32768 if not set

    # Non-QA: set NoData
    for i in range(nbands):
        if i in qa_set: continue
        dst.GetRasterBand(i+1).SetNoDataValue(nd_out)

    w,h = grid_gt['w'], grid_gt['h']; bx,by = blocksize, blocksize
    for y0 in range(0,h,by):
        nrows = by if y0+by<=h else h-y0
        for x0 in range(0,w,bx):
            ncols = bx if x0+bx<=w else w-x0

            # Remove the ERA5-specific logic; use standard processing uniformly

            for i in range(nbands):
                if i in qa_set:
                    arr = read_as_array_compat(qa_vrt.GetRasterBand(qa_pos[i]), x0,y0,ncols,nrows)
                    dst.GetRasterBand(i+1).WriteArray(arr, xoff=x0, yoff=y0); del arr
                elif i in lai_set:
                    bpos = lai_pos_map[i]
                    arr_pos = read_as_array_compat(lai_pos_vrt.GetRasterBand(bpos), x0,y0,ncols,nrows) if lai_pos_vrt else None
                    arr_neg = read_as_array_compat(lai_neg_vrt.GetRasterBand(bpos), x0,y0,ncols,nrows) if lai_neg_vrt else None
                    arr_nn  = read_as_array_compat(lai_nn_vrt.GetRasterBand(bpos),  x0,y0,ncols,nrows) if lai_nn_vrt  else None
                    if arr_pos is None and arr_neg is None and arr_nn is None:
                        out = np.full((nrows,ncols), nd_out, dtype=gdal_dtype_numpy(gdt))
                    else:
                        base = arr_pos if arr_pos is not None else (arr_neg if arr_neg is not None else arr_nn)
                        out = np.full((nrows,ncols), nd_out, dtype=base.dtype)
                        if arr_nn is not None:
                            valid_nn  = (arr_nn != nd_out)
                            if arr_pos is not None:
                                valid_pos = (arr_pos != nd_out); choose = (arr_nn>0)&valid_nn&valid_pos; out[choose]=arr_pos[choose]
                            if arr_neg is not None:
                                valid_neg = (arr_neg != nd_out); choose = (arr_nn<0)&valid_nn&valid_neg; out[choose]=arr_neg[choose]
                            zero_mask = (arr_nn==0)&valid_nn
                            if zero_mask.any():
                                if arr_pos is not None: out[zero_mask&(arr_pos!=nd_out)] = arr_pos[zero_mask&(arr_pos!=nd_out)]
                                if arr_neg is not None: out[zero_mask&(arr_neg!=nd_out)] = arr_neg[zero_mask&(arr_neg!=nd_out)]
                    # Keep raw values; do not apply a non-negative cleanup
                    dst.GetRasterBand(i+1).WriteArray(out, xoff=x0, yoff=y0)
                    del arr_pos,arr_neg,arr_nn,out
                else:
                    arr = read_as_array_compat(nonlai_vrt.GetRasterBand(nonlai_pos[i]), x0,y0,ncols,nrows)
                    # Keep raw values; do not apply a non-negative cleanup
                    dst.GetRasterBand(i+1).WriteArray(arr, xoff=x0, yoff=y0); del arr
            gc.collect()
    dst.FlushCache(); dst=close_ds(dst); return True

# ----------------- Process one file -----------------
def process_one(src_path: Path, out_path: Path,
                grid_template_ds, clip_shp_path: Path,
                resample_alg_nonqa: str,
                compress='LZW', blocksize=512,
                keep_temp=False, force=False):
    if out_path.exists() and not force:
        return 'skipped'

    qa_indices = []
    lai_indices = []

    src = open_ds(src_path, readonly=True)
    nbands = src.RasterCount
    src = close_ds(src)

    tmp_dir = Path(tempfile.mkdtemp(prefix="tri_lai_tmp_"))
    cleaned_path = tmp_dir / "cleaned.tif"
    cleaned_ds = None
    try:
        apply_nd_bands = [i for i in range(nbands) if i not in set(qa_indices)]

        out_gdt, canonical_nd = preclean_to_temp_no_modify(
            src_path, qa_indices, cleaned_path,
            compress='NONE', blocksize=blocksize,
            apply_nd_to_bands=apply_nd_bands
        )
        cleaned_ds = open_ds(cleaned_path, readonly=True)

        qa_vrt_src, qa_vrt_src_path = (translate_subset_to_vrt_disk(tmp_dir, "qa_src", cleaned_ds, qa_indices) if qa_indices else (None,None))
        all_idx = list(range(nbands)); qa_set=set(qa_indices); lai_set=set(lai_indices)
        nonlai_indices = [i for i in all_idx if i not in qa_set and i not in lai_set]
        nonlai_vrt_src, nonlai_vrt_src_path = (translate_subset_to_vrt_disk(tmp_dir, "nonlai_src", cleaned_ds, nonlai_indices) if nonlai_indices else (None,None))
        lai_vrt_src, lai_vrt_src_path = (translate_subset_to_vrt_disk(tmp_dir, "lai_src", cleaned_ds, lai_indices) if lai_indices else (None,None))

        # ---------------- Align to the grid-template grid (no clipping) ----------------
        grid_bounds, grid_size, grid_gt_tuple = get_bounds_size_gt(grid_template_ds)
        grid_wkt = get_srs_wkt(grid_template_ds)
        grid_gt_info = {'gt': grid_gt_tuple, 'w': grid_size[0], 'h': grid_size[1]}
        xRes = grid_gt_tuple[1]; yRes = grid_gt_tuple[5]

        alg = (resample_alg_nonqa or "bilinear").strip().lower()
        if alg in ("nearest", "near"):
            warp_alg_nonqa = "near"
        elif alg == "bilinear":
            warp_alg_nonqa = "bilinear"
        else:
            warp_alg_nonqa = "bilinear"
        eff_nd_align = canonical_nd

        qa_vrt_aligned  = qa_vrt_aligned_path  = None
        if qa_vrt_src:
            # QA bands: nearest; do not pass nodata
            qa_vrt_aligned, qa_vrt_aligned_path = align_to_grid_vrt(
                tmp_dir, "qa_aligned", qa_vrt_src, grid_wkt, grid_bounds, grid_size,
                resample_alg='near', canonical_nd=None
            )
            qa_vrt_src = close_ds(qa_vrt_src); unlink_safe(qa_vrt_src_path)

        nonlai_vrt_aligned = nonlai_vrt_aligned_path = None
        if nonlai_vrt_src:
            nonlai_vrt_aligned, nonlai_vrt_aligned_path = align_to_grid_vrt(
                tmp_dir, "nonlai_aligned", nonlai_vrt_src, grid_wkt, grid_bounds, grid_size,
                resample_alg=warp_alg_nonqa, canonical_nd=eff_nd_align
            )
            nonlai_vrt_src = close_ds(nonlai_vrt_src); unlink_safe(nonlai_vrt_src_path)

        lai_pos_vrt = lai_neg_vrt = lai_nn_vrt = None
        lai_pos_vrt_path = lai_neg_vrt_path = lai_nn_vrt_path = None

        if lai_indices:
            # Build LAI positive / negative / NN three paths
            lai_pos_path = tmp_dir / "lai_pos.tif"
            lai_neg_path = tmp_dir / "lai_neg.tif"
            build_lai_masked_temp(cleaned_path, lai_indices, canonical_nd, 'pos', lai_pos_path, compress='NONE', blocksize=blocksize)
            build_lai_masked_temp(cleaned_path, lai_indices, canonical_nd, 'neg', lai_neg_path, compress='NONE', blocksize=blocksize)
            cleaned_ds = close_ds(cleaned_ds)

            lai_pos_ds = open_ds(lai_pos_path, readonly=True)
            lai_neg_ds = open_ds(lai_neg_path, readonly=True)
            lai_src_ds = open_ds(lai_vrt_src_path, readonly=True) if lai_vrt_src else None

            lai_pos_vrt_src, lai_pos_vrt_src_path = translate_subset_to_vrt_disk(tmp_dir, "lai_pos_src", lai_pos_ds, list(range(len(lai_indices))))
            lai_neg_vrt_src, lai_neg_vrt_src_path = translate_subset_to_vrt_disk(tmp_dir, "lai_neg_src", lai_neg_ds, list(range(len(lai_indices))))
            lai_pos_ds = close_ds(lai_pos_ds); lai_neg_ds = close_ds(lai_neg_ds)
            unlink_safe(lai_pos_path); unlink_safe(lai_neg_path)

            eff_nd_align_lai = canonical_nd
            lai_pos_vrt, lai_pos_vrt_path = align_to_grid_vrt(tmp_dir,"lai_pos_aligned",lai_pos_vrt_src,grid_wkt,grid_bounds,grid_size,resample_alg=warp_alg_nonqa,canonical_nd=eff_nd_align_lai)
            lai_neg_vrt, lai_neg_vrt_path = align_to_grid_vrt(tmp_dir,"lai_neg_aligned",lai_neg_vrt_src,grid_wkt,grid_bounds,grid_size,resample_alg=warp_alg_nonqa,canonical_nd=eff_nd_align_lai)
            lai_nn_vrt,  lai_nn_vrt_path  = align_to_grid_vrt(tmp_dir,"lai_nn_aligned", lai_src_ds,     grid_wkt,grid_bounds,grid_size,resample_alg='near',canonical_nd=eff_nd_align_lai)

            if lai_vrt_src: lai_vrt_src = close_ds(lai_vrt_src); unlink_safe(lai_vrt_src_path)
            lai_pos_vrt_src = close_ds(lai_pos_vrt_src); unlink_safe(lai_pos_vrt_src_path)
            lai_neg_vrt_src = close_ds(lai_neg_vrt_src); unlink_safe(lai_neg_vrt_src_path)
            lai_src_ds = close_ds(lai_src_ds)
        else:
            cleaned_ds = close_ds(cleaned_ds)

        # ---------------- (Optional) clip by shapefile (preserves resolution and pixel alignment) ----------------
        if clip_shp_path and clip_shp_path.exists():
            # QA
            if qa_vrt_aligned:
                qa_vrt_clip, qa_vrt_clip_path = clip_with_shapefile_vrt(tmp_dir,"qa_clip",qa_vrt_aligned,grid_wkt,xRes,yRes,clip_shp_path,'near',canonical_nd=None)
                qa_vrt_aligned = close_ds(qa_vrt_aligned); unlink_safe(qa_vrt_aligned_path)
                qa_vrt_aligned, qa_vrt_aligned_path = qa_vrt_clip, qa_vrt_clip_path
            # Non-LAI
            if nonlai_vrt_aligned:
                nonlai_vrt_clip, nonlai_vrt_clip_path = clip_with_shapefile_vrt(tmp_dir,"nonlai_clip",nonlai_vrt_aligned,grid_wkt,xRes,yRes,clip_shp_path,warp_alg_nonqa,canonical_nd=eff_nd_align)
                nonlai_vrt_aligned = close_ds(nonlai_vrt_aligned); unlink_safe(nonlai_vrt_aligned_path)
                nonlai_vrt_aligned, nonlai_vrt_aligned_path = nonlai_vrt_clip, nonlai_vrt_clip_path
            # LAI
            if lai_indices:
                lai_pos_vrt_clip, lai_pos_vrt_clip_path = clip_with_shapefile_vrt(tmp_dir,"lai_pos_clip",lai_pos_vrt,grid_wkt,xRes,yRes,clip_shp_path,warp_alg_nonqa,canonical_nd=eff_nd_align)
                lai_neg_vrt_clip, lai_neg_vrt_clip_path = clip_with_shapefile_vrt(tmp_dir,"lai_neg_clip",lai_neg_vrt,grid_wkt,xRes,yRes,clip_shp_path,warp_alg_nonqa,canonical_nd=eff_nd_align)
                lai_nn_vrt_clip,  lai_nn_vrt_clip_path  = clip_with_shapefile_vrt(tmp_dir,"lai_nn_clip", lai_nn_vrt, grid_wkt,xRes,yRes,clip_shp_path,'near',canonical_nd=eff_nd_align)
                lai_pos_vrt = close_ds(lai_pos_vrt); unlink_safe(lai_pos_vrt_path); lai_pos_vrt, lai_pos_vrt_path = lai_pos_vrt_clip, lai_pos_vrt_clip_path
                lai_neg_vrt = close_ds(lai_neg_vrt); unlink_safe(lai_neg_vrt_path); lai_neg_vrt, lai_neg_vrt_path = lai_neg_vrt_clip, lai_neg_vrt_clip_path
                lai_nn_vrt  = close_ds(lai_nn_vrt ); unlink_safe(lai_nn_vrt_path ); lai_nn_vrt,  lai_nn_vrt_path  = lai_nn_vrt_clip,  lai_nn_vrt_clip_path

            # Update grid_gt_info to the "post-clip" geotransform and size
            # (Clip stage shrinks the bounds; this is expected behavior)
            ref_vrt_for_gt = nonlai_vrt_aligned or qa_vrt_aligned or lai_nn_vrt or lai_pos_vrt or lai_neg_vrt
            if ref_vrt_for_gt is None:
                raise RuntimeError("Cannot find a temporary VRT to read the post-clip geotransform from.")
            grid_bounds_clip, grid_size_clip, grid_gt_clip = get_bounds_size_gt(ref_vrt_for_gt)
            grid_gt_info = {'gt': grid_gt_clip, 'w': grid_size_clip[0], 'h': grid_size_clip[1]}

        # ---------------- Write out (keep LST UInt16; use canonical_nd uniformly) ----------------
        out_gdt_final = out_gdt
        sp_lc = str(src_path).lower()
        if ('mod11a1' in sp_lc or 'myd11a1' in sp_lc):
            out_gdt_final = gdal.GDT_UInt16

        write_final_with_blocks(out_path, out_gdt_final, grid_gt_info, grid_wkt, nbands,
                                qa_vrt_aligned, nonlai_vrt_aligned, lai_pos_vrt, lai_neg_vrt, lai_nn_vrt,
                                lai_indices,
                                qa_indices, nonlai_indices,
                                canonical_nd, compress=compress, blocksize=blocksize,
                                era5_mode=False, out_nodata_override=None)

        # Cleanup
        if qa_vrt_aligned:   qa_vrt_aligned   = close_ds(qa_vrt_aligned);   unlink_safe(qa_vrt_aligned_path)
        if nonlai_vrt_aligned: nonlai_vrt_aligned = close_ds(nonlai_vrt_aligned); unlink_safe(nonlai_vrt_aligned_path)
        if lai_indices:
            lai_pos_vrt = close_ds(lai_pos_vrt); unlink_safe(lai_pos_vrt_path)
            lai_neg_vrt = close_ds(lai_neg_vrt); unlink_safe(lai_neg_vrt_path)
            lai_nn_vrt  = close_ds(lai_nn_vrt ); unlink_safe(lai_nn_vrt_path)

        gc.collect(); gdal.ErrorReset()
        return 'ok'
    except Exception:
        if out_path.exists():
            try: out_path.unlink()
            except Exception: pass
        raise
    finally:
        try: cleaned_ds = close_ds(cleaned_ds)
        except Exception: pass
        shutil.rmtree(tmp_dir, ignore_errors=True)
        gc.collect()

# ----------------- Batch processing (with efficient scanning) -----------------
def process_all(input_root: Path, output_root: Path,
                grid_template_path: Path,
                shapefile_path: Path,
                threads=16, compress='LZW', blocksize=512,
                keep_temp=False, force=False, resume_from=0,
                include_folders=None):
    if not grid_template_path.exists(): raise FileNotFoundError(f"Grid template does not exist: {grid_template_path}")

    grid_template_ds = open_ds(grid_template_path, readonly=True)

    input_root = Path(input_root).resolve()
    if not input_root.is_dir():
        print(f"[ERROR] --input-root must be a directory (containing GeoTIFFs directly): {input_root}")
        close_ds(grid_template_ds)
        return

    name_filter = _make_tif_name_filter(include_folders) if include_folders else None
    if include_folders:
        inc_sorted = sorted(include_folders)
        print(f"[INFO] only process name-matched GeoTIFFs ({len(inc_sorted)} items): {', '.join(inc_sorted)}")
    else:
        print(f"[INFO] --folders not specified; will process all GeoTIFFs in the directory: {input_root}")

    print("[INFO] Scanning input files (single pass, no sort)...")
    warp_alg = resample_alg_for_subfolder(input_root.name)
    batch_label = input_root.name

    tasks = []; skipped_count = 0; tif_seen = 0; matched_by_filter = 0
    ensure_dir(output_root)
    existing = set()
    if output_root.exists():
        try:
            with os.scandir(output_root) as it_out:
                for e in it_out:
                    if e.is_file(): existing.add(e.name.lower())
        except FileNotFoundError:
            pass

    try:
        with os.scandir(input_root) as it_files:
            for fe in it_files:
                if not fe.is_file(): continue
                name = fe.name
                if ".__tmp__" in name: continue
                lower = name.lower()
                if not (lower.endswith(".tif") or lower.endswith(".tiff")): continue
                tif_seen += 1
                if not _tif_matches_name_filter(name, name_filter):
                    continue
                matched_by_filter += 1
                if (lower in existing) and (not force):
                    skipped_count += 1; continue
                tasks.append((batch_label, Path(fe.path), output_root / name, warp_alg))
    except FileNotFoundError:
        print(f"[ERROR] input directory does not exist: {input_root}")
        close_ds(grid_template_ds)
        return

    if name_filter and tif_seen > 0 and matched_by_filter == 0:
        print("[WARN] The directory has GeoTIFFs but none matches --folders; please check filenames or stems.")
    if skipped_count > 0: print(f"[SKIP] {skipped_count} files already exist (use --force to overwrite)")
    if not tasks:
        print("[INFO] No files to process.")
        close_ds(grid_template_ds); return

    if resume_from > 0:
        if resume_from >= len(tasks):
            print(f"[WARN] --resume-from {resume_from} exceeds total task count {len(tasks)}")
            close_ds(grid_template_ds); return
        print(f"[RESUME] from task index {resume_from} (total {len(tasks)} pending tasks)")
        tasks = tasks[resume_from:]

    total = len(tasks)
    stats = {'ok': 0, 'skipped': skipped_count, 'failed': 0}
    fail_records = []

    print(f"[INFO] Pending: {total} | threads: {threads} | compression: {compress} | Block：{blocksize}")
    print("=" * 80)

    clip_shp = Path(shapefile_path) if shapefile_path else None
    logdir = output_root / "_logs"
    logdir.mkdir(parents=True, exist_ok=True)

    def _wrap(t):
        folder_name, tif, out_path, warp_alg = t
        try:
            process_one(
                src_path=tif, out_path=out_path,
                grid_template_ds=grid_template_ds, clip_shp_path=clip_shp,
                resample_alg_nonqa=warp_alg,
                compress=compress, blocksize=blocksize,
                keep_temp=keep_temp, force=True,
            )
            return ('ok', folder_name, tif, None, None, None, None)
        except Exception as e:
            tb = traceback.format_exc()
            gdal_msg = gdal.GetLastErrorMsg()
            msg = str(e)
            if 'Write operation not permitted on dataset opened in read-only mode' in msg: step='readonly_write'
            elif 'LAI' in msg: step='lai'
            elif 'Warp' in msg or 'warp' in msg: step='warp'
            elif 'VRT' in msg or 'Translate' in msg: step='vrt'
            elif 'ReadAsArray' in msg: step='io_read'
            else: step='unknown'
            return ('failed', folder_name, tif, msg, tb, gdal_msg, step)

    done = 0
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futures = [ex.submit(_wrap, t) for t in tasks]
        for fut in as_completed(futures):
            ret = fut.result()
            with progress_lock:
                done += 1
                if ret[0]=='ok':
                    stats['ok'] += 1
                    sys.stdout.write(f"\r[Progress] {done}/{total}  OK: {stats['ok']}  FAILED: {stats['failed']}")
                else:
                    stats['failed'] += 1
                    _, folder_name, tif, msg, tb, gdal_msg, step = ret
                    sys.stdout.write(f"\n[FAIL] {folder_name}/{tif.name}  | step={step} | err={msg[:300]}")
                    sys.stdout.flush()
                    fail_records.append({
                        'time': datetime.now().isoformat(timespec='seconds'),
                        'folder': folder_name, 'file': tif.name, 'step': step,
                        'error': msg, 'gdal_last_error': gdal_msg, 'traceback': tb
                    })

    print("\n" + "=" * 80)
    print(f"[DONE] OK={stats['ok']}  SKIPPED={stats['skipped']}  FAILED={stats['failed']}")
    print(f"[OUT ] {output_root}")

    if fail_records:
        csv_path = logdir / "tri_pipeline_failures.csv"
        txt_path = logdir / "tri_pipeline_failures.log"
        write_fail_logs(csv_path, txt_path, fail_records)
        print(f"[LOG ] failure details written to: {csv_path}")
        print(f"[LOG ] full stack trace log: {txt_path}")

    close_ds(grid_template_ds)
    gc.collect(); gdal.ErrorReset()

# ----------------- CLI -----------------
def _normalize_include_folders(arg_list):
    if not arg_list: return None
    toks = []
    for item in arg_list:
        if item is None: continue
        s = str(item).strip()
        if not s: continue
        if ',' in s: toks.extend([t.strip() for t in s.split(',') if t.strip()])
        else: toks.append(s)
    return set(toks) if toks else None

def _make_tif_name_filter(tokens):
    """Return (lowercase full-filename set, lowercase stem set), used as the --folders allowlist."""
    if not tokens:
        return None
    exact = set()
    stems = set()
    for raw in tokens:
        s = str(raw).strip().lower()
        if not s:
            continue
        exact.add(s)
        if s.endswith(".tif") or s.endswith(".tiff"):
            stems.add(Path(s).stem.lower())
        else:
            stems.add(s)
    return exact, stems

def _tif_matches_name_filter(name, flt):
    if flt is None:
        return True
    exact, stems = flt
    low = name.lower()
    if low in exact:
        return True
    return Path(name).stem.lower() in stems

def main():
    ap = argparse.ArgumentParser(description="Three-in-one GDAL raster processing: align to the reference TIF grid (without changing origin / bounds) -> final shapefile clip (consolidating into a single grid-template)")
    ap.add_argument("--input-root",  default="/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled_5xdownsampled_merged_clipped/VPD_merged_clipped", type=Path)
    ap.add_argument("--output-root", default="/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled_5xdownsampled_merged_clipped_unified/VPD_merged_clipped_unified", type=Path)
    ap.add_argument("--grid-template", default="/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/spatial_filled_5xdownsampled_merged_clipped/MCD14A1_combined_5xMaximumDown_merged_clipped/2025_12_31.tif", type=Path)
    ap.add_argument("--shapefile",    default="/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/Ecozones/research_area_shpfile_merge.shp", type=Path)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--compress", type=str, default='LZW')
    ap.add_argument("--blocksize", type=int, default=512)
    ap.add_argument("--keep-temp", action='store_true')
    ap.add_argument("--force", action='store_true')
    ap.add_argument("--resume-from", type=int, default=0)
    ap.add_argument("--folders", nargs='*', default=None,
                    help="Optional: process only matching GeoTIFF filenames or stems, separated by space or comma. Examples: --folders 2025_01_01.tif 2025_01_02 or --folders a.tif,b.tif")
    args = ap.parse_args()
    include_folders = _normalize_include_folders(args.folders)

    process_all(
        input_root=args.input_root,
        output_root=args.output_root,
        grid_template_path=args.grid_template,
        shapefile_path=args.shapefile,
        threads=args.threads,
        compress=args.compress,
        blocksize=args.blocksize,
        keep_temp=args.keep_temp,
        force=args.force,
        resume_from=args.resume_from,
        include_folders=include_folders
    )

if __name__ == "__main__":
    main()
