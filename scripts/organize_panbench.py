#!/usr/bin/env python3
"""
organize_panbench.py - Convert PanBench structure to prepare_h5.py format

PanBench structure:
  raw/ -> {satellite}/{MODALITY}_{SIZE}/*.tif

Expected structure:
  raw/ -> {satellite}/{train|valid|test}/[PAN|MS|GT]/*.tif
"""

import os
import glob
import shutil
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Satellites and their multi-spectral directories
SATELLITE_INFO = {
    "wv3": {"ms_dirs": ["RGB_256", "NIR_256"], "ms_bands": 8, "pan": "PAN_1024"},
    "gf2": {"ms_dirs": ["RGB_256"], "ms_bands": 4, "pan": "PAN_1024"},
    "qb": {"ms_dirs": ["RGB_256"], "ms_bands": 4, "pan": "PAN_1024"},
}

def organize_satellite(satellite_dir, satellite_name="wv3", split_ratios=(0.8, 0.1, 0.1)):
    """Organize one satellite's directory."""
    satellite_dir = Path(satellite_dir)
    info = SATELLITE_INFO.get(satellite_name)
    
    if not info:
        print(f"Unknown satellite: {satellite_name}")
        return
    
    # Find all PAN files
    pan_dir = satellite_dir / info["pan"]
    if not pan_dir.exists():
        print(f"PAN directory not found: {pan_dir}")
        return
    
    pan_files = sorted(glob.glob(str(pan_dir / "*.tif")))
    if not pan_files:
        print(f"No PAN files found in {pan_dir}")
        return
    
    print(f"\nFound {len(pan_files)} PAN images for {satellite_name}")
    
    # Split into train/valid/test
    train_files, temp_files = train_test_split(pan_files, train_size=split_ratios[0], random_state=42)
    valid_files, test_files = train_test_split(temp_files, train_size=split_ratios[1]/(split_ratios[1] + split_ratios[2]), random_state=42)
    
    print(f"Splits: train={len(train_files)}, valid={len(valid_files)}, test={len(test_files)}")
    
    # Create directory structure
    for split_name, files in [("train", train_files), ("valid", valid_files), ("test", test_files)]:
        # Create directories
        pan_out = satellite_dir / split_name / "PAN"
        ms_out = satellite_dir / split_name / "MS"
        gt_out = satellite_dir / split_name / "GT"
        
        pan_out.mkdir(parents=True, exist_ok=True)
        ms_out.mkdir(parents=True, exist_ok=True)
        gt_out.mkdir(parents=True, exist_ok=True)
        
        # Process each PAN file
        for pan_file in files:
            pan_file = Path(pan_file)
            base_name = pan_file.stem
            
            # Copy PAN
            shutil.copy2(pan_file, pan_out / pan_file.name)
            
            # Find and copy MS (if exists)
            ms_copied = False
            for ms_dir_name in info["ms_dirs"]:
                ms_dir = satellite_dir / ms_dir_name
                if not ms_dir.exists():
                    continue
                
                # Look for MS file with same base name (might have different extensions)
                ms_files = list(ms_dir.glob(f"{base_name}.*"))
                if ms_files:
                    ms_file = ms_files[0]
                    ext = ms_file.suffix
                    # Copy to both MS and GT (as GT is HR-MS, we'll use same file as proxy)
                    shutil.copy2(ms_file, ms_out / f"{base_name}{ext}")
                    shutil.copy2(ms_file, gt_out / f"{base_name}{ext}")
                    ms_copied = True
                    break
            
            if not ms_copied:
                print(f"Warning: No MS file found for {base_name}")
    
    print(f"[OK] Organized {satellite_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw/PanBench", help="Path to raw PanBench data")
    parser.add_argument("--satellites", nargs="+", default=["wv3", "gf2", "qb"], 
                       choices=["wv3", "gf2", "qb"])
    args = parser.parse_args()
    
    print(f"Organizing PanBench dataset from {args.raw_dir}")
    
    for satellite in args.satellites:
        sat_dir = Path(args.raw_dir) / satellite
        if sat_dir.exists():
            print(f"\n{'='*60}")
            organize_satellite(sat_dir, satellite)
        else:
            print(f"\nSatellite directory not found: {sat_dir}")
    
    print(f"\n{'='*60}")
    print("[OK] Organization complete!")
    print(f"\nNext step: Run prepare_h5.py")
    print(f"Example:")
    print(f"  python scripts/prepare_h5.py --dataset wv3 --raw_dir data/raw/PanBench/WV3")

if __name__ == "__main__":
    import argparse
    main()
