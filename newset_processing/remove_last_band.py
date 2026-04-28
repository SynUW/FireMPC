#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to remove the last few bands of TIF images.
- Removes the last few bands from every TIF file in a folder.
- Supports specifying how many bands to remove (1 = last band, 2 = last two bands, etc.).
- Supports single-file or batch processing.
- Preserves original dtype, georeferencing, and NoData settings.
- Auto-skips already-processed files.
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple
import rasterio
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def process_single_file(input_file: str, output_file: str, bands_to_remove: int) -> bool:
    """
    Process a single TIF file by removing the last few bands.
    
    Args:
        input_file: input file path
        output_file: output file path
        bands_to_remove: number of bands to remove (1 = last band, 2 = last two bands)
        
    Returns:
        whether processing succeeded
    """
    try:
        with rasterio.open(input_file) as src:
            # Check band count
            if src.count <= bands_to_remove:
                logger.warning(f"File {input_file} has only {src.count} bands; cannot remove {bands_to_remove} bands")
                return False
            
            # Get original profile
            profile = src.profile.copy()
            
            # Update band count
            new_band_count = src.count - bands_to_remove
            profile.update({
                'count': new_band_count
            })
            
            # Read the bands to keep
            processed_data = []
            processed_descriptions = []
            
            for i in range(1, new_band_count + 1):  # Read only the bands to keep
                band_data = src.read(i)
                processed_data.append(band_data)
                
                # Preserve band descriptions (if any)
                if src.descriptions and i-1 < len(src.descriptions):
                    processed_descriptions.append(src.descriptions[i-1])
            
            # Write to output file
            with rasterio.open(output_file, 'w', **profile) as dst:
                for i, band_data in enumerate(processed_data):
                    dst.write(band_data, i + 1)
                
                # Set band descriptions
                if processed_descriptions:
                    dst.descriptions = tuple(processed_descriptions)
            
            return True
            
    except Exception as e:
        logger.error(f"Processing file {input_file} error: {str(e)}")
        return False

def process_folder(input_dir: str, output_dir: str, bands_to_remove: int, skip_existing: bool = True) -> Tuple[int, int]:
    """
    Process all TIF files in a folder.
    
    Args:
        input_dir: input folder path
        output_dir: output folder path
        bands_to_remove: number of bands to remove
        skip_existing: whether to skip files that already exist
        
    Returns:
        (success_count, total_count)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all TIF files
    tif_files = list(input_path.glob("*.tif")) + list(input_path.glob("*.TIF"))
    if not tif_files:
        logger.warning(f"In directory {input_dir}: no TIF files found")
        return 0, 0
    
    logger.info(f"Found {len(tif_files)} TIF files")
    
    success_count = 0
    
    for tif_file in tqdm(tif_files, desc="Process files", ncols=100):
        output_file = output_path / tif_file.name
        
        # Check whether to skip existing files
        if skip_existing and output_file.exists():
            success_count += 1
            continue
        
        if process_single_file(str(tif_file), str(output_file), bands_to_remove):
            success_count += 1
    
    return success_count, len(tif_files)

def main():
    parser = argparse.ArgumentParser(description="Remove the last few bands of TIF images")
    parser.add_argument("--input", default="/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/MCD14A1_combined_merged", help="Input file or folder path")
    parser.add_argument("--output", default="/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/MCD14A1_combined_merged_remove2bands", help="Output file or folder path")
    parser.add_argument("--bands", "-b", type=int, default=2, help="Number of bands to remove (1 = last band, 2 = last two bands, etc.)")
    parser.add_argument("--no-skip-existing", action="store_true", 
                       help="Do not skip existing output files")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    bands_to_remove = args.bands
    skip_existing = not args.no_skip_existing
    
    # Validate arguments
    if bands_to_remove <= 0:
        logger.error("Number of bands to remove must be greater than 0")
        return
    
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return
    
    logger.info(f"Will remove the last {bands_to_remove} bands")
    
    if input_path.is_file():
        # Process a single file
        if process_single_file(str(input_path), str(output_path), bands_to_remove):
            logger.info("File processing complete")
        else:
            logger.error("File processing failed")
    elif input_path.is_dir():
        # Process folder
        success_count, total_count = process_folder(
            str(input_path), str(output_path), bands_to_remove, skip_existing
        )
        logger.info(f"Processing complete: {success_count}/{total_count} files succeeded")
    else:
        logger.error(f"Input path is neither a file nor a folder: {input_path}")

if __name__ == "__main__":
    main()
