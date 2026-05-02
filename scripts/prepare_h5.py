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
    "wv3": {"ms_bands": 8, "max_val": 2047.0, "scale": 4, "pan_patch": 256, "ms_patch": 64},
    "gf2": {"ms_bands": 4, "max_val": 1023.0, "scale": 4, "pan_patch": 256, "ms_patch": 64},
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

def extract_patches(pan, ms, gt, pan_patch, ms_patch, stride, max_val):
    """
    Extract patch pairs from PAN (high-res) and MS/GT (low-res) images.
    PAN: (1, H_pan, W_pan)   e.g. 1024x1024
    MS/GT: (C, H_ms, W_ms)  e.g. 256x256 (scale=4)
    GT is upsampled to PAN resolution as the HR-MS training target.
    """
    scale = pan_patch // ms_patch
    _, H_pan, W_pan = pan.shape
    pan_patches, ms_patches, gt_patches, lrms_patches = [], [], [], []

    for y in range(0, H_pan - pan_patch + 1, stride):
        for x in range(0, W_pan - pan_patch + 1, stride):
            ys = y // scale
            xs = x // scale
            pan_p  = pan[:, y:y+pan_patch,   x:x+pan_patch]
            ms_p   = ms[:,  ys:ys+ms_patch,  xs:xs+ms_patch]
            gt_p   = gt[:,  ys:ys+ms_patch,  xs:xs+ms_patch]
            gt_hr  = np.clip(bicubic_upsample(gt_p, scale), 0.0, max_val)
            lrms_p = np.clip(bicubic_upsample(ms_p, scale), 0.0, max_val)
            pan_patches.append( (pan_p  / max_val).astype(np.float32))
            ms_patches.append(  (ms_p   / max_val).astype(np.float32))
            gt_patches.append(  (gt_hr  / max_val).astype(np.float32))
            lrms_patches.append((lrms_p / max_val).astype(np.float32))

    if not pan_patches:
        C = ms.shape[0]
        return (np.zeros((0,1,pan_patch,pan_patch),np.float32),
                np.zeros((0,C,ms_patch, ms_patch), np.float32),
                np.zeros((0,C,pan_patch,pan_patch),np.float32),
                np.zeros((0,C,pan_patch,pan_patch),np.float32))

    return (np.stack(pan_patches,axis=0), np.stack(ms_patches,axis=0),
            np.stack(gt_patches,axis=0),  np.stack(lrms_patches,axis=0))

def create_h5(pan_dir: str, ms_dir: str, gt_dir: str, out_path: str, satellite: str, split: str, stride_factor: float = 0.5):
    """
    Create HDF5 from GeoTIFF directories.

    STREAMING MODE — patches are written to disk image-by-image using
    HDF5 resizable datasets. No full array is ever held in RAM.
    RAM usage is bounded to ~1 image worth of patches at a time (~50–200 MB).
    """
    cfg = SATELLITE_CONFIG[satellite]
    pan_patch  = cfg["pan_patch"]
    ms_patch   = cfg["ms_patch"]
    ms_bands   = cfg["ms_bands"]
    stride     = int(pan_patch * stride_factor)

    pan_files = sorted(glob.glob(os.path.join(pan_dir,  "*.tif")))
    ms_files  = sorted(glob.glob(os.path.join(ms_dir,   "*.tif")))
    gt_files  = sorted(glob.glob(os.path.join(gt_dir,   "*.tif")))

    if not pan_files:
        print(f"No .tif files found in {pan_dir}")
        return

    print(f"Processing {len(pan_files)} images -> {out_path}  [streaming — low RAM]")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Chunk shape: one patch per chunk for efficient random-access reads during training
    pan_chunk  = (1, 1,        pan_patch, pan_patch)
    ms_chunk   = (1, ms_bands, ms_patch,  ms_patch)
    gt_chunk   = (1, ms_bands, pan_patch, pan_patch)
    lrms_chunk = (1, ms_bands, pan_patch, pan_patch)

    total_written = 0

    with h5py.File(out_path, "w") as f:
        # Create resizable datasets — maxshape=None means unlimited rows
        ds_pan  = f.create_dataset("pan",  shape=(0, 1,        pan_patch, pan_patch),
                                   maxshape=(None, 1,        pan_patch, pan_patch),
                                   dtype="float32", compression="gzip", compression_opts=4,
                                   chunks=pan_chunk)
        ds_ms   = f.create_dataset("ms",   shape=(0, ms_bands, ms_patch,  ms_patch),
                                   maxshape=(None, ms_bands, ms_patch,  ms_patch),
                                   dtype="float32", compression="gzip", compression_opts=4,
                                   chunks=ms_chunk)
        ds_gt   = f.create_dataset("gt",   shape=(0, ms_bands, pan_patch, pan_patch),
                                   maxshape=(None, ms_bands, pan_patch, pan_patch),
                                   dtype="float32", compression="gzip", compression_opts=4,
                                   chunks=gt_chunk)
        ds_lrms = f.create_dataset("lrms", shape=(0, ms_bands, pan_patch, pan_patch),
                                   maxshape=(None, ms_bands, pan_patch, pan_patch),
                                   dtype="float32", compression="gzip", compression_opts=4,
                                   chunks=lrms_chunk)

        for pan_f, ms_f, gt_f in tqdm(
                zip(pan_files, ms_files, gt_files),
                total=len(pan_files),
                desc=f"{satellite}/{split}"):
            try:
                pan = load_tif(pan_f)
                ms  = load_tif(ms_f)
                gt  = load_tif(gt_f)

                if pan.ndim == 2: pan = pan[np.newaxis]
                if ms.ndim  == 2: ms  = ms[np.newaxis]
                if gt.ndim  == 2: gt  = gt[np.newaxis]

                # Ensure ms has the expected number of bands
                if ms.shape[0] < ms_bands:
                    # Pad missing bands by repeating the last available band
                    pad = np.repeat(ms[[-1]], ms_bands - ms.shape[0], axis=0)
                    ms  = np.concatenate([ms, pad], axis=0)
                    gt  = np.concatenate([gt[:ms_bands] if gt.shape[0] >= ms_bands
                                          else np.concatenate([gt, pad[:ms_bands - gt.shape[0]]], axis=0),
                                         ], axis=0)[:ms_bands]
                elif ms.shape[0] > ms_bands:
                    ms = ms[:ms_bands]
                    gt = gt[:ms_bands]

                pans, mss, gts, lrmss = extract_patches(
                    pan, ms, gt, pan_patch, ms_patch, stride, cfg["max_val"])

                n = pans.shape[0]
                if n == 0:
                    continue

                # ── Append to HDF5 datasets (streaming write) ──────────────
                start = total_written
                end   = start + n

                ds_pan.resize( end, axis=0); ds_pan[start:end]  = pans
                ds_ms.resize(  end, axis=0); ds_ms[start:end]   = mss
                ds_gt.resize(  end, axis=0); ds_gt[start:end]   = gts
                ds_lrms.resize(end, axis=0); ds_lrms[start:end] = lrmss

                total_written = end

                # Free memory immediately — critical for large datasets
                del pans, mss, gts, lrmss, pan, ms, gt

            except Exception as e:
                print(f"\nError processing {pan_f}: {e}")
                continue

        if total_written == 0:
            print("No valid patches extracted.")
            return

        # Write metadata
        f.attrs["satellite"] = satellite
        f.attrs["split"]     = split
        f.attrs["n_samples"] = total_written
        f.attrs["ms_bands"]  = ms_bands

    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"\nSaved {out_path}  ({size_mb:.1f} MB  |  {total_written} patches)")

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