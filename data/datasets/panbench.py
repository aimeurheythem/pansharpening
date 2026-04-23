"""
panbench.py — PanBench Dataset Loader (Remote Sensing 2024)

PanBench is a multi-satellite benchmark dataset (MS: 256×256, PAN: 1024×1024).
Download: https://drive.google.com/file/d/1fjwvRrCmExk02c5sxGXMoSvGdGL0FbYR/view

HDF5 file structure (PanCollection standard):
    f['pan']   → (N, 1, H_pan, W_pan)     PAN image  (high resolution)
    f['lrms']  → (N, C, H_ms,  W_ms)      LR-MS image (upsampled to PAN size)
    f['gt']    → (N, C, H_pan, W_pan)      HR-MS ground truth
    f['ms']    → (N, C, H_ms,  W_ms)      Original LR-MS (no upsampling)

Supported satellites:
    - WorldView-3 (wv3): 8 MS bands, max pixel = 2047 (11-bit)
    - GaoFen-2 (gf2):    4 MS bands, max pixel = 1023 (10-bit)
    - QuickBird (qb):    4 MS bands, max pixel = 2047 (11-bit)
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Literal


# Bit-depth normalization constants per satellite
NORMALIZATION = {
    "wv3": 2047.0,  # 11-bit WorldView-3
    "gf2": 1023.0,  # 10-bit GaoFen-2
    "qb":  2047.0,  # 11-bit QuickBird
}

NUM_BANDS = {
    "wv3": 8,
    "gf2": 4,
    "qb":  4,
}


class PanBenchDataset(Dataset):
    """
    PyTorch Dataset for PanBench HDF5 files.

    Args:
        h5_path:   Path to the .h5 file
        satellite: One of 'wv3', 'gf2', 'qb'
        split:     'train', 'val', or 'test'
        augment:   Apply random flips/rotations (train only)

    Returns per sample:
        pan:   (1, H_pan, W_pan) float tensor [0, 1]
        lrms:  (C, H_pan, W_pan) float tensor [0, 1]  (bicubic upsampled)
        gt:    (C, H_pan, W_pan) float tensor [0, 1]
        ms:    (C, H_ms,  W_ms)  float tensor [0, 1]  (original LR)
    """

    def __init__(
        self,
        h5_path: str,
        satellite: Literal["wv3", "gf2", "qb"] = "wv3",
        split: Literal["train", "val", "test"]  = "train",
        augment: bool = False,
    ):
        super().__init__()
        self.h5_path   = Path(h5_path)
        self.satellite = satellite.lower()
        self.split     = split
        self.augment   = augment and (split == "train")
        self.norm      = NORMALIZATION.get(self.satellite, 2047.0)

        if not self.h5_path.exists():
            raise FileNotFoundError(
                f"HDF5 file not found: {self.h5_path}\n"
                f"Download PanBench from: "
                f"https://drive.google.com/file/d/1fjwvRrCmExk02c5sxGXMoSvGdGL0FbYR/view"
            )

        # Load all data into RAM (fast training, standard in pansharpening research)
        with h5py.File(self.h5_path, "r") as f:
            self.pan  = np.array(f["pan"],  dtype=np.float32)   # (N, 1, H, W)
            self.lrms = np.array(f["lrms"], dtype=np.float32)   # (N, C, H, W)
            self.gt   = np.array(f["gt"],   dtype=np.float32)   # (N, C, H, W)
            # Some files have 'ms', some don't
            self.ms   = np.array(f["ms"], dtype=np.float32) if "ms" in f else self.lrms

        # Normalize to [0, 1]
        self.pan  /= self.norm
        self.lrms /= self.norm
        self.gt   /= self.norm
        self.ms   /= self.norm

        self._len = self.pan.shape[0]

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict:
        pan  = torch.from_numpy(self.pan[idx])   # (1, H_pan, W_pan)
        lrms = torch.from_numpy(self.lrms[idx])  # (C, H_pan, W_pan)
        gt   = torch.from_numpy(self.gt[idx])    # (C, H_pan, W_pan)
        ms   = torch.from_numpy(self.ms[idx])    # (C, H_ms,  W_ms)

        if self.augment:
            pan, lrms, gt, ms = self._augment(pan, lrms, gt, ms)

        return {"pan": pan, "lrms": lrms, "gt": gt, "ms": ms}

    @staticmethod
    def _augment(
        pan: torch.Tensor,
        lrms: torch.Tensor,
        gt: torch.Tensor,
        ms: torch.Tensor
    ) -> tuple:
        """Random horizontal/vertical flip and 90° rotation."""
        if torch.rand(1) > 0.5:
            pan  = torch.flip(pan,  dims=[-1])
            lrms = torch.flip(lrms, dims=[-1])
            gt   = torch.flip(gt,   dims=[-1])
            ms   = torch.flip(ms,   dims=[-1])
        if torch.rand(1) > 0.5:
            pan  = torch.flip(pan,  dims=[-2])
            lrms = torch.flip(lrms, dims=[-2])
            gt   = torch.flip(gt,   dims=[-2])
            ms   = torch.flip(ms,   dims=[-2])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            pan  = torch.rot90(pan,  k, dims=[-2, -1])
            lrms = torch.rot90(lrms, k, dims=[-2, -1])
            gt   = torch.rot90(gt,   k, dims=[-2, -1])
            ms   = torch.rot90(ms,   k, dims=[-2, -1])
        return pan, lrms, gt, ms

    def get_normalization(self) -> float:
        return self.norm

    def __repr__(self) -> str:
        return (
            f"PanBenchDataset(satellite={self.satellite}, split={self.split}, "
            f"n_samples={self._len}, bands={self.gt.shape[1]})"
        )


def get_panbench_loaders(
    h5_train: str,
    h5_val: str,
    h5_test: Optional[str] = None,
    satellite: str = "wv3",
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict:
    """
    Convenience function to create train/val/test DataLoaders for PanBench.

    Usage:
        loaders = get_panbench_loaders(
            h5_train="./data/h5/train_wv3.h5",
            h5_val="./data/h5/valid_wv3.h5",
            satellite="wv3",
            batch_size=32
        )
        for batch in loaders["train"]:
            pan, lrms, gt = batch["pan"], batch["lrms"], batch["gt"]
    """
    train_ds = PanBenchDataset(h5_train, satellite=satellite, split="train", augment=True)
    val_ds   = PanBenchDataset(h5_val,   satellite=satellite, split="val",   augment=False)

    loaders = {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True
        ),
        "val": DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        ),
    }

    if h5_test:
        test_ds = PanBenchDataset(h5_test, satellite=satellite, split="test", augment=False)
        loaders["test"] = DataLoader(
            test_ds, batch_size=1, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

    return loaders
