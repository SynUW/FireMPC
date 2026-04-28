#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MODIS GeoTIFF -> true-color MP4 (standard true-color RGB composite)
--------------------------------------------------
- Assumes each GeoTIFF contains MCD09GA's 4 bands:
  - Band 1: red (620-670 nm)
  - Band 4: green (545-565 nm)  -- real green band
  - Band 3: blue (459-479 nm)
  - Band 2: near-infrared (841-876 nm) - optionally used for vegetation enhancement

- For each date file:
  1) Reproject to WGS84 (EPSG:4326), aligned to a common grid derived from the first frame.
  2) Standard true-color composite: R=Band1, G=Band4, B=Band3
  3) 5% linear stretch (5%-95% percentile) + optional gamma correction
  4) Write the video frame and overlay the date (from the filename) in white text with a black outline.

- Video parameters: default 2 seconds per frame (fps=0.5); H.264/MP4 with mp4v as fallback.

Dependencies:
  pip install gdal numpy opencv-python tqdm

Examples:
  # Standard true color + 5% linear stretch (default)
  python video_gen.py \
      --input-dir /path/to/tiffs \
      --output out.mp4 \
      --fps 0.5
  
  # Quick test (4 frames)
  python video_gen.py \
      --input-dir /path/to/tiffs \
      --output test.mp4 \
      --max-frames 4

Notes:
  - Uses GDAL instead of rasterio to avoid NumPy version conflicts.
  - Automatically skips TIFFs that cannot be opened.
  - Uses the real green band (Band 4) rather than synthesized green.

Filenames must be parseable as a date, e.g.:
  2003_07_15.tif / 20030715.tif / 2003-07-15.tif
"""

import re
import os
import math
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from osgeo import gdal, osr
import cv2
from tqdm import tqdm

# Disable GDAL error output to stderr
# Note: do not use UseExceptions() because it triggers a gdal_array import that causes NumPy version conflicts
gdal.PushErrorHandler('CPLQuietErrorHandler')

DATE_PATTERNS = [
    re.compile(r".*?(\d{4})[-_]?(\d{2})[-_]?(\d{2}).*?$"),  # YYYY[-_]MM[-_]DD or YYYYMMDD
]

def parse_date_from_name(name: str) -> Optional[str]:
    for pat in DATE_PATTERNS:
        m = pat.match(name)
        if m:
            y, mth, d = m.groups()
            return f"{y}-{mth}-{d}"
    return None

def percent_clip_stretch(arr: np.ndarray, p_low=2.0, p_high=98.0, eps=1e-6) -> np.ndarray:
    """Percentile-stretch a single channel; output in 0..1"""
    lo = np.nanpercentile(arr, p_low)
    hi = np.nanpercentile(arr, p_high)
    if not np.isfinite(lo): lo = np.nanmin(arr)
    if not np.isfinite(hi): hi = np.nanmax(arr)
    if hi - lo < eps:
        out = np.zeros_like(arr, dtype=np.float32)
    else:
        out = (arr - lo) / (hi - lo)
        out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)

def gamma_correct(arr01: np.ndarray, gamma=1.0) -> np.ndarray:
    """Apply gamma correction to a 0..1 image"""
    if abs(gamma - 1.0) < 1e-6:
        return arr01
    arr = np.clip(arr01, 0.0, 1.0)
    return np.power(arr, 1.0 / gamma)

def build_true_color(b1_red, b4_green, b3_blue, b2_nir=None,
                     use_nir_enhance: bool = False,
                     nir_weight: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standard true-color RGB composite:
      R = Band 1 (red, 620-670 nm)
      G = Band 4 (green, 545-565 nm)  -- using the real green band
      B = Band 3 (blue, 459-479 nm)
    
    Optional: enhance the green channel with NIR (good over vegetated regions)
      G_enhanced = G + nir_weight * NIR
    """
    R = b1_red.astype(np.float32)
    G = b4_green.astype(np.float32)
    B = b3_blue.astype(np.float32)
    
    # Optional: NIR enhancement (highlights vegetation)
    if use_nir_enhance and b2_nir is not None:
        G = G + nir_weight * b2_nir.astype(np.float32)
        G = np.clip(G, 0, None)  # Avoid negative values

    return R, G, B

def to_uint8_rgb(R01, G01, B01) -> np.ndarray:
    rgb = np.stack([R01, G01, B01], axis=-1)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return rgb

def compute_common_grid(first_ds, dst_crs="EPSG:4326", dst_res: Optional[Tuple[float, float]] = None):
    """
    Use the first image to determine the video's common grid (WGS84) and return (transform, width, height).
    Implemented with GDAL.
    """
    src_width = first_ds.RasterXSize
    src_height = first_ds.RasterYSize
    src_geotransform = first_ds.GetGeoTransform()
    src_projection = first_ds.GetProjection()
    
    # Create the target coordinate system
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(src_projection)
    
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(4326)
    
    # Use GDAL AutoCreateWarpedVRT to estimate the target size
    vrt = gdal.AutoCreateWarpedVRT(first_ds, src_projection, dst_srs.ExportToWkt())
    if vrt is None:
        raise RuntimeError("Cannot create WarpedVRT")
    
    width = vrt.RasterXSize
    height = vrt.RasterYSize
    transform = vrt.GetGeoTransform()
    
    # If a resolution was specified, recompute
    if dst_res is not None:
        # Get bounds
        minx = transform[0]
        maxy = transform[3]
        maxx = minx + transform[1] * width
        miny = maxy + transform[5] * height
        
        # Recompute size
        width = int((maxx - minx) / dst_res[0])
        height = int((miny - maxy) / abs(dst_res[1]))
        transform = (minx, dst_res[0], 0, maxy, 0, -abs(dst_res[1]))
    
    vrt = None
    return transform, width, height

def reproject_to_grid(src_ds, indexes, dst_crs_epsg, dst_transform, dst_shape, resampling=gdal.GRA_Bilinear) -> np.ndarray:
    """
    Reproject arrays at the given band indices to the common grid.
    Uses the GDAL Warp API (to avoid ReadAsArray's NumPy version issues).
    Performance optimization: process all bands at once.
    """
    bands = len(indexes)
    dst_height, dst_width = dst_shape
    out = np.zeros((bands, dst_height, dst_width), dtype=np.float32)
    
    # Create the target coordinate system
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(dst_crs_epsg)
    dst_wkt = dst_srs.ExportToWkt()
    
    # Create a VRT selecting the required bands
    vrt_options = gdal.BuildVRTOptions(bandList=indexes)
    vrt = gdal.BuildVRT('', src_ds, options=vrt_options)
    
    # Warp all bands at once (much faster than per-band)
    warp_options = gdal.WarpOptions(
        format='MEM',
        dstSRS=dst_wkt,
        resampleAlg=resampling,
        dstNodata=np.nan,
        multithread=True,  # Enable multithreading
        width=dst_width,
        height=dst_height
    )
    # Warp directly creates the target dataset
    result_ds = gdal.Warp('', vrt, options=warp_options)
    result_ds.SetGeoTransform(dst_transform)
    
    # Read all band data (using RasterIO, not ReadAsArray)
    for i in range(bands):
        band = result_ds.GetRasterBand(i + 1)
        buf = band.ReadRaster(0, 0, dst_width, dst_height, dst_width, dst_height, gdal.GDT_Float32)
        out[i] = np.frombuffer(buf, dtype=np.float32).reshape((dst_height, dst_width))
    
    # Cleanup
    vrt = None
    result_ds = None
    
    return out

def put_date_text(frame_bgr: np.ndarray, date_text: str, font_scale=1.2, thickness=2) -> np.ndarray:
    """
    Overlay the date in white text with a black outline at the lower-left corner.
    """
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    org = (int(0.02 * w), int(h - 0.02 * h))  # Slightly above the lower-left corner
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Black outline
    cv2.putText(img, date_text, org, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # White text
    cv2.putText(img, date_text, org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img

def main():
    ap = argparse.ArgumentParser(description="Generate true-color MP4 from MODIS GeoTIFFs (b1,b2,b3,b7).")
    ap.add_argument("--input-dir", type=str, required=True, help="Folder containing GeoTIFFs")
    ap.add_argument("--output", type=str, required=True, help="Output MP4 file path, e.g. out.mp4")
    ap.add_argument("--fps", type=float, default=0.5, help="Video frame rate (frames/second); 2 seconds per frame is 0.5 fps")
    ap.add_argument("--codec", type=str, default="h264", choices=["h264", "mp4v"], help="Video codec")
    ap.add_argument("--band-order", type=str, default="1,4,3,2",
                    help="Input GeoTIFF band order R, G, B, NIR (comma-separated); default 1=red, 4=green, 3=blue, 2=NIR (MCD09GA standard)")
    ap.add_argument("--dst-res", type=str, default=None,
                    help="Target resolution (degrees in lon/lat, e.g. '0.0045,0.0045'); default: auto-estimated from the first frame")
    ap.add_argument("--use-nir-enhance", action="store_true", help="Enhance the green channel with NIR (highlights vegetation)")
    ap.add_argument("--nir-weight", type=float, default=0.1, help="NIR enhancement weight, typically 0.05-0.15")
    ap.add_argument("--stretch-plow", type=float, default=5.0, help="Lower percentile for stretching (default: 5% linear stretch)")
    ap.add_argument("--stretch-phigh", type=float, default=95.0, help="Upper percentile for stretching (default: 5% linear stretch)")
    ap.add_argument("--gamma", type=float, default=1.0, help="Gamma correction; >1 brightens, <1 enhances contrast")
    ap.add_argument("--global-stretch", action="store_true",
                    help="Use a global (across all time) percentile stretch instead of per-frame stretching")
    ap.add_argument("--font-scale", type=float, default=4, help="Date text size")
    ap.add_argument("--thickness", type=int, default=2, help="Date text line width")
    ap.add_argument("--max-frames", type=int, default=None, help="Maximum frame count limit (for quick tests, e.g. --max-frames 4)")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    assert in_dir.is_dir(), f"Input directory does not exist: {in_dir}"

    tiffs = sorted([p for p in in_dir.glob("*.tif*")])
    if not tiffs:
        raise FileNotFoundError("No .tif/.tiff files found in the input directory")

    # Parse dates and sort (files without parseable dates go last)
    items = []
    for p in tiffs:
        ds = parse_date_from_name(p.stem)
        items.append((p, ds if ds else "9999-99-99"))
    items.sort(key=lambda x: x[1])
    
    # Limit frame count (for quick tests)
    if args.max_frames and args.max_frames > 0:
        items = items[:args.max_frames]
        print(f"Frame-limit mode: process only the first {len(items)} frames")

    # Parse the band mapping (R, G, B, NIR)
    try:
        r_idx, g_idx, b_idx, nir_idx = [int(x.strip()) for x in args.band_order.split(",")]
    except Exception:
        raise ValueError("--band-order should be in the form 'R,G,B,NIR', e.g. '1,4,3,2' (MCD09GA standard)")

    # Open the first frame and establish the common grid
    # Try opening the first valid TIFF file
    src0 = None
    first_valid_idx = 0
    for idx, (p, _) in enumerate(items):
        try:
            src0 = gdal.Open(str(p), gdal.GA_ReadOnly)
            if src0 is not None:
                first_valid_idx = idx
                break
        except Exception as e:
            print(f"Skip file that cannot be opened: {p.name} ({e})")
            continue
    
    if src0 is None:
        raise RuntimeError("None of the TIFF files can be opened!")
    
    dst_crs = "EPSG:4326"
    dst_res = None
    if args.dst_res:
        a, b = args.dst_res.split(",")
        dst_res = (float(a), float(b))
    dst_transform, dst_w, dst_h = compute_common_grid(src0, dst_crs=dst_crs, dst_res=dst_res)
    src0 = None  # Close

    # If global stretching is needed, first compute the global percentiles
    global_stats = None
    if args.global_stretch:
        vals_R, vals_G, vals_B = [], [], []
        # Performance optimization: sample only some frames for global stats (at most 50 frames or all)
        sample_items = items if len(items) <= 50 else items[::max(1, len(items)//50)]
        print(f"Global statistics: sampling {len(sample_items)}/{len(items)} frames")
        for p, _ in tqdm(sample_items, desc="Scanning global percentiles", ncols=100):
            try:
                src = gdal.Open(str(p), gdal.GA_ReadOnly)
                if src is None:
                    print(f"Skip: {p.name}")
                    continue
                
                arr = reproject_to_grid(
                    src, [r_idx, g_idx, b_idx, nir_idx], 4326, dst_transform, (dst_h, dst_w),
                    resampling=gdal.GRA_Bilinear
                )
                src = None
                
                band_r, band_g, band_b, band_nir = arr
                R, G, B = build_true_color(band_r, band_g, band_b, band_nir,
                                           use_nir_enhance=args.use_nir_enhance,
                                           nir_weight=args.nir_weight)
                # Collect samples (to save memory, randomly sample 10% of pixels)
                for ch, buf in zip([R, G, B], [vals_R, vals_G, vals_B]):
                    flat = ch.ravel()
                    n = flat.size
                    step = max(1, n // (n // 10 if n >= 1_000_000 else 10))  # Roughly 10% or fewer
                    buf.append(flat[::step])
            except Exception as e:
                print(f"Global statistics skipped {p.name}: {e}")
                continue
        
        R_all = np.concatenate(vals_R) if vals_R else np.array([0.0])
        G_all = np.concatenate(vals_G) if vals_G else np.array([0.0])
        B_all = np.concatenate(vals_B) if vals_B else np.array([0.0])
        def perc(a):
            lo = np.nanpercentile(a, args.stretch_plow)
            hi = np.nanpercentile(a, args.stretch_phigh)
            return lo, hi
        global_stats = (perc(R_all), perc(G_all), perc(B_all))

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*('avc1' if args.codec == 'h264' else 'mp4v'))
    out_video = cv2.VideoWriter(str(args.output), fourcc, args.fps, (dst_w, dst_h))
    if not out_video.isOpened():
        # avc1/h264 may be unavailable on some systems; fall back to mp4v
        fourcc_fallback = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(str(args.output), fourcc_fallback, args.fps, (dst_w, dst_h))
        if not out_video.isOpened():
            raise RuntimeError("Cannot open the video writer; try changing --codec or check OpenCV/FFmpeg support.")

    # Write frame
    skipped_count = 0
    for p, ds in tqdm(items, desc="Generate video frame", ncols=100):
        try:
            src = gdal.Open(str(p), gdal.GA_ReadOnly)
            if src is None:
                print(f"Skip file that cannot be opened: {p.name}")
                skipped_count += 1
                continue
            
            arr = reproject_to_grid(
                src, [r_idx, g_idx, b_idx, nir_idx], 4326, dst_transform, (dst_h, dst_w),
                resampling=gdal.GRA_Bilinear
            )
            src = None
            
            band_r, band_g, band_b, band_nir = arr

            # Standard true-color composite
            R, G, B = build_true_color(
                band_r, band_g, band_b, band_nir,
                use_nir_enhance=args.use_nir_enhance,
                nir_weight=args.nir_weight
            )

            # Stretch & gamma
            if args.global_stretch and global_stats is not None:
                (Rlo, Rhi), (Glo, Ghi), (Blo, Bhi) = global_stats
                def stretch_fix(ch, lo, hi, eps=1e-6):
                    if hi - lo < eps:
                        return np.zeros_like(ch, dtype=np.float32)
                    x = (ch - lo) / (hi - lo)
                    return np.clip(x, 0.0, 1.0).astype(np.float32)
                R01 = stretch_fix(R, Rlo, Rhi)
                G01 = stretch_fix(G, Glo, Ghi)
                B01 = stretch_fix(B, Blo, Bhi)
            else:
                R01 = percent_clip_stretch(R, args.stretch_plow, args.stretch_phigh)
                G01 = percent_clip_stretch(G, args.stretch_plow, args.stretch_phigh)
                B01 = percent_clip_stretch(B, args.stretch_plow, args.stretch_phigh)

            if args.gamma and abs(args.gamma - 1.0) > 1e-6:
                R01 = gamma_correct(R01, args.gamma)
                G01 = gamma_correct(G01, args.gamma)
                B01 = gamma_correct(B01, args.gamma)

            rgb = to_uint8_rgb(R01, G01, B01)
            # OpenCV uses BGR
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Overlay the date (if not parseable, show the filename)
            date_txt = ds if ds != "9999-99-99" else p.stem
            bgr = put_date_text(bgr, date_txt, font_scale=args.font_scale, thickness=args.thickness)

            out_video.write(bgr)
        
        except Exception as e:
            print(f"Processing failed; skipping {p.name}: {e}")
            skipped_count += 1
            continue

    out_video.release()
    print(f"\nVideo generation complete: {args.output}")
    print(f"Total file count: {len(items)}")
    print(f"Successfully processed: {len(items) - skipped_count}")
    print(f"Skipped files: {skipped_count}")

if __name__ == "__main__":
    main()
