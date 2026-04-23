"""
prepare_h5.py - Convert raw satellite imagery (.tif) to PanCollection HDF5 format.
"""

import argparse
import os
import glob
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm

# Satellite config (adjusted for RGB data)
SATELLITE_CONFIG = {
    "wv3": {"ms_bands": 3, "max_val": 2047.0, "scale": 4, "pan_patch": 128, "ms_patch": 32},
    "gf2": {"ms_bands": 4, "max_val": 1023.0, "scale": 4, "pan_patch": 128, "ms_patch": 32},
    "qb": {"ms_bands": 4, "max_val": 2047.0, "scale": 4, "pan_patch": 128, "ms_patch": 32},
}

def load_tif(path: str) -> np.ndarray:
    """Load a GeoTIFF file as numpy array (C, H, W)."""
    import rasterio
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
    return data

def bicubic_upsample(ms: np.ndarray, scale: int) -> np.ndarray:
    """Upsample MS image by scale factor using bicubic interpolation."""
    import cv2
    C, H, W = ms.shape
    out = np.zeros((C, H * scale, W * scale), dtype=np.float32)
    for c in range(C):
        out[c] = cv2.resize(ms[c], (W * scale, H * scale), interpolation=cv2.INTER_CUBIC)
    return out

def extract_patches(pan: np.ndarray, ms: np.ndarray, gt: np.ndarray, pan_patch: int, ms_patch: int, stride: int, max_val: float) -> tuple:
    """Extract non-overlapping patches from full images."""
    _, H_pan, W_pan = pan.shape
    pan_patches, ms_patches, gt_patches, lrms_patches = [], [], [], []
    
    for y in range(0, H_pan - pan_patch + 1, stride):
        for x in range(0, W_pan - pan_patch + 1, stride):
            ys = y // (pan_patch // ms_patch)
            xs = x // (pan_patch // ms_patch)
            
            pan_p = pan[:, y:y+pan_patch, x:x+pan_patch]
            gt_p = gt[:, y:y+pan_patch, x:x+pan_patch]
            ms_p = ms[:, ys:ys+ms_patch, xs:xs+ms_patch]
            lrms_p = bicubic_upsample(ms_p, pan_patch // ms_patch)
            
            pan_patches.append((pan_p / max_val).astype(np.float32))
            ms_patches.append((ms_p / max_val).astype(np.float32))
            gt_patches.append((gt_p / max_val).astype(np.float32))
            lrms_patches.append((lrms_p / max_val).astype(np.float32))
    
    return (np.stack(pan_patches, axis=0), np.stack(ms_patches, axis=0),
            np.stack(gt_patches, axis=0), np.stack(lrms_patches, axis=0))

def create_h5(pan_dir: str, ms_dir: str, gt_dir: str, out_path: str, satellite: str, split: str, stride_factor: float = 0.5):
    """Create HDF5 from GeoTIFF directories."""
    cfg = SATELLITE_CONFIG[satellite]
    pan_patch = cfg["pan_patch"]
    ms_patch = cfg["ms_patch"]
    stride = int(pan_patch * stride_factor)
    
    pan_files = sorted(glob.glob(os.path.join(pan_dir, "*.tif")))
    ms_files = sorted(glob.glob(os.path.join(ms_dir, "*.tif")))
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.tif")))
    
    if not pan_files:
        print(f"No .tif files found in {pan_dir}")
        return
    
    print(f"Processing {len(pan_files)} images -> {out_path}")
    
    all_pan, all_ms, all_gt, all_lrms = [], [], [], []
    
    for pan_f, ms_f, gt_f in tqdm(zip(pan_files, ms_files, gt_files), total=len(pan_files), desc=f"{satellite}/{split}"):
        try:
            pan = load_tif(pan_f)
            ms = load_tif(ms_f)
            gt = load_tif(gt_f)
            
            if pan.ndim == 2:
                pan = pan[np.newaxis]
            if ms.ndim == 2:
                ms = ms[np.newaxis]
            if gt.ndim == 2:
                gt = gt[np.newaxis]
            
            pans, mss, gts, lrmss = extract_patches(pan, ms, gt, pan_patch, ms_patch, stride, cfg["max_val"])
            all_pan.append(pans)
            all_ms.append(mss)
            all_gt.append(gts)
            all_lrms.append(lrmss)
            
        except Exception as e:
            print(f"Error processing {pan_f}: {e}")
    
    if not all_pan:
        print("No valid patches extracted.")
        return
    
    pan_arr = np.concatenate(all_pan, axis=0)
    ms_arr = np.concatenate(all_ms, axis=0)
    gt_arr = np.concatenate(all_gt, axis=0)
    lrms_arr = np.concatenate(all_lrms, axis=0)
    
    print(f"Patches: {pan_arr.shape[0]} | PAN: {pan_arr.shape[1:]} | GT: {gt_arr.shape[1:]}")
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("pan", data=pan_arr, compression="gzip", chunks=True)
        f.create_dataset("ms", data=ms_arr, compression="gzip", chunks=True)
        f.create_dataset("gt", data=gt_arr, compression="gzip", chunks=True)
        f.create_dataset("lrms", data=lrms_arr, compression="gzip", chunks=True)
        f.attrs["satellite"] = satellite
        f.attrs["split"] = split
        f.attrs["n_samples"] = pan_arr.shape[0]
        f.attrs["ms_bands"] = cfg["ms_bands"]
    
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"Saved {out_path} ({size_mb:.1f} MB, {pan_arr.shape[0]} patches)")

def main():
    parser = argparse.ArgumentParser(description="Convert raw .tif to HDF5")
    parser.add_argument("--dataset", type=str, required=True, choices=["wv3", "gf2", "qb"])
    parser.add_argument("--raw_dir", type=str, required=True, help="Root dir with PAN/, MS/, GT/ subdirs")
    parser.add_argument("--out_dir", type=str, default="data/h5")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--stride", type=float, default=0.5)
    args = parser.parse_args()
    
    raw = Path(args.raw_dir)
    for split in ["train", "valid", "test"]:
        split_dir = raw / split
        if not split_dir.exists():
            print(f"Skipping {split} (directory not found)")
            continue
        
        create_h5(
            pan_dir=str(split_dir / "PAN"),
            ms_dir=str(split_dir / "MS"),
            gt_dir=str(split_dir / "GT"),
            out_path=f"{args.out_dir}/{split}_{args.dataset}.h5",
            satellite=args.dataset,
            split=split,
            stride_factor=args.stride,
        )
    
    print("\nHDF5 preparation complete!")
    print("Next: python train.py --config configs/panbench.yaml")

if __name__ == "__main__":
    main()
