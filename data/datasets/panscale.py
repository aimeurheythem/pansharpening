"""
panscale.py — PanScale Dataset Loader

PanScale is the multi-scene, cross-scale pansharpening benchmark used in ScaleFormer
(CVPR 2026). Downloaded from: hf://kecao/PanScale

Dataset structure (after HuggingFace download):
    datasets/panscale/
        train/
            <scene_id>/
                PAN/   *.tif  (high-resolution panchromatic)
                MS/    *.tif  (low-resolution multispectral)
                GT/    *.tif  (high-resolution multispectral, ground truth)
        val/
            <scene_id>/...
        test/
            <scene_id>/...

The dataset supports variable scale ratios (4×, 8×, 16×) — each scene may
have a different PAN/MS spatial ratio. ScaleFormer is designed to handle this.

Normalization:
    Images are stored as uint16 (0–65535). We normalize to [0, 1] by dividing
    by the theoretical max of 65535 (or the actual per-dataset max if specified).
"""

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Literal, List, Tuple
import random


# =============================================================================
# HELPERS
# =============================================================================

def _load_image(path: str) -> np.ndarray:
    """
    Load a satellite image from .tif, .png, or .npy.
    Returns (C, H, W) float32 array.
    """
    path = str(path)
    ext  = Path(path).suffix.lower()

    if ext in (".tif", ".tiff"):
        try:
            import rasterio
            with rasterio.open(path) as src:
                data = src.read().astype(np.float32)
        except ImportError:
            # Fallback to PIL/imageio for plain 3-band TIFFs
            import imageio
            data = np.array(imageio.imread(path)).astype(np.float32)
            if data.ndim == 2:
                data = data[np.newaxis]
            elif data.ndim == 3 and data.shape[-1] <= 4:
                data = data.transpose(2, 0, 1)
    elif ext == ".npy":
        data = np.load(path).astype(np.float32)
        if data.ndim == 2:
            data = data[np.newaxis]
    elif ext in (".png", ".jpg", ".jpeg"):
        import imageio
        data = np.array(imageio.imread(path)).astype(np.float32)
        if data.ndim == 2:
            data = data[np.newaxis]
        else:
            data = data.transpose(2, 0, 1)
    else:
        raise ValueError(f"Unsupported image format: {ext}")

    return data   # (C, H, W)


def _bicubic_upsample(ms: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Upsample MS image to (C, target_h, target_w) using bicubic interpolation."""
    t = torch.from_numpy(ms).unsqueeze(0)      # (1, C, H, W)
    t = F.interpolate(t, size=(target_h, target_w), mode="bicubic", align_corners=False)
    return t.squeeze(0).numpy()                 # (C, target_h, target_w)


def _find_image_files(directory: str) -> List[str]:
    """Recursively find all image files in a directory."""
    exts = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.npy")
    files = []
    for ext in exts:
        files += glob.glob(os.path.join(directory, "**", ext), recursive=True)
    return sorted(files)


# =============================================================================
# SCENE DISCOVERY — handles multiple possible folder layouts
# =============================================================================

def discover_panscale_scenes(root: str, split: str) -> List[dict]:
    """
    Discover all (pan, ms, gt) triplets in the PanScale dataset.
    Handles different possible folder layouts from HuggingFace.

    Layout A (scene-based):
        root/split/scene_001/PAN/img.tif
        root/split/scene_001/MS/img.tif
        root/split/scene_001/GT/img.tif

    Layout B (flat):
        root/split/PAN/img_001.tif
        root/split/MS/img_001.tif
        root/split/GT/img_001.tif

    Layout C (HuggingFace parquet / image folders):
        root/split/pan/img_001.tif
        root/split/ms/img_001.tif
        root/split/gt/img_001.tif

    Returns list of dicts: {"pan": path, "ms": path, "gt": path}
    """
    split_dir = Path(root) / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"PanScale {split} split not found at: {split_dir}\n"
            f"Run: python setup_and_train.py --download-only  to fetch the dataset."
        )

    scenes = []

    # Try Layout A: scene-based subdirectories
    scene_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
    for sd in sorted(scene_dirs):
        # Find PAN subfolder (case-insensitive)
        pan_dir = _find_subdir(sd, ["PAN", "pan", "Pan"])
        ms_dir  = _find_subdir(sd, ["MS",  "ms",  "Ms", "LMS", "lms"])
        gt_dir  = _find_subdir(sd, ["GT",  "gt",  "Gt", "HRMS", "hrms"])

        if pan_dir and ms_dir and gt_dir:
            pan_files = _find_image_files(str(pan_dir))
            ms_files  = _find_image_files(str(ms_dir))
            gt_files  = _find_image_files(str(gt_dir))

            # Pair files by sorted order (assumes 1:1 correspondence)
            for pf, mf, gf in zip(pan_files, ms_files, gt_files):
                scenes.append({"pan": pf, "ms": mf, "gt": gf})
            continue

    # Try Layout B/C: PAN/MS/GT directly under split_dir
    if not scenes:
        pan_dir = _find_subdir(split_dir, ["PAN", "pan", "Pan"])
        ms_dir  = _find_subdir(split_dir, ["MS",  "ms",  "Ms", "LMS", "lms"])
        gt_dir  = _find_subdir(split_dir, ["GT",  "gt",  "Gt", "HRMS", "hrms"])

        if pan_dir and ms_dir and gt_dir:
            pan_files = _find_image_files(str(pan_dir))
            ms_files  = _find_image_files(str(ms_dir))
            gt_files  = _find_image_files(str(gt_dir))
            for pf, mf, gf in zip(pan_files, ms_files, gt_files):
                scenes.append({"pan": pf, "ms": mf, "gt": gf})

    if not scenes:
        raise RuntimeError(
            f"Could not find any (PAN, MS, GT) image triplets under {split_dir}.\n"
            f"Directory contents: {[d.name for d in split_dir.iterdir()]}"
        )

    return scenes


def _find_subdir(parent: Path, names: list) -> Optional[Path]:
    """Return the first subdirectory that matches one of the given names."""
    for name in names:
        p = parent / name
        if p.is_dir():
            return p
    return None


# =============================================================================
# PANSCALE DATASET
# =============================================================================

class PanScaleDataset(Dataset):
    """
    PyTorch Dataset for the PanScale benchmark (ScaleFormer, CVPR 2026).

    Loads satellite image triplets (PAN, MS, GT) from the downloaded
    HuggingFace dataset, applies normalization, optional patch cropping
    for training, and returns tensors compatible with the project's models.

    Args:
        root:        Root directory of the downloaded PanScale dataset
        split:       One of 'train', 'val', 'test'
        patch_size:  PAN patch size for training (None = use full image)
        augment:     Apply random flips/rotations (train only)
        max_val:     Normalization divisor (65535 for uint16, 1.0 if pre-normalized)
        scale_ratio: Expected PAN/MS spatial ratio (used to compute lrms size)

    Returns per sample (dict):
        pan:  (1, P, P)   PAN patch, float [0, 1]
        lrms: (C, P, P)   Bicubic-upsampled MS to PAN size, float [0, 1]
        gt:   (C, P, P)   Ground truth HR-MS, float [0, 1]
        ms:   (C, p, p)   Original LR-MS (p = P // scale_ratio), float [0, 1]
    """

    def __init__(
        self,
        root:        str,
        split:       Literal["train", "val", "test"] = "train",
        patch_size:  Optional[int]  = 128,
        augment:     bool           = False,
        max_val:     float          = 65535.0,
        scale_ratio: int            = 4,
    ):
        super().__init__()
        self.root        = root
        self.split       = split
        self.patch_size  = patch_size
        self.augment     = augment and (split == "train")
        self.max_val     = max_val
        self.scale_ratio = scale_ratio

        # Discover all image triplets
        self.scenes = discover_panscale_scenes(root, split)
        print(f"[PanScaleDataset] {split}: {len(self.scenes)} image triplets found")

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx: int) -> dict:
        scene = self.scenes[idx]

        # Load images
        pan = _load_image(scene["pan"])    # (1, H_pan, W_pan)
        ms  = _load_image(scene["ms"])     # (C, H_ms, W_ms)
        gt  = _load_image(scene["gt"])     # (C, H_pan, W_pan)

        # Ensure PAN is single-channel
        if pan.shape[0] > 1:
            pan = pan[:1]

        # Normalize to [0, 1]
        # Auto-detect max_val from actual pixel range if data appears pre-normalized
        actual_max = max(pan.max(), ms.max(), gt.max())
        norm = self.max_val if actual_max > 1.0 else 1.0
        pan = pan / norm
        ms  = ms  / norm
        gt  = gt  / norm

        # Upsample MS to PAN resolution → lrms
        H_pan, W_pan = pan.shape[-2], pan.shape[-1]
        lrms = _bicubic_upsample(ms, H_pan, W_pan)

        # Patch extraction for training
        if self.patch_size is not None and (H_pan > self.patch_size or W_pan > self.patch_size):
            pan, lrms, gt, ms = self._random_crop(pan, lrms, gt, ms)

        # Augmentation
        if self.augment:
            pan, lrms, gt, ms = self._augment(pan, lrms, gt, ms)

        return {
            "pan":  torch.from_numpy(pan.copy()),
            "lrms": torch.from_numpy(lrms.copy()),
            "gt":   torch.from_numpy(gt.copy()),
            "ms":   torch.from_numpy(ms.copy()),
        }

    def _random_crop(
        self,
        pan:  np.ndarray,   # (1, H, W)
        lrms: np.ndarray,   # (C, H, W)
        gt:   np.ndarray,   # (C, H, W)
        ms:   np.ndarray,   # (C, h, w)
    ) -> Tuple:
        """Random crop: PAN patch_size × patch_size, MS patch_size//scale × patch_size//scale."""
        P    = self.patch_size
        H, W = pan.shape[-2], pan.shape[-1]

        # Random top-left for PAN crop
        top  = random.randint(0, max(0, H - P))
        left = random.randint(0, max(0, W - P))

        # Corresponding MS crop
        s    = self.scale_ratio
        ms_top  = top  // s
        ms_left = left // s
        ms_size = P // s

        pan_crop  = pan[:,  top:top+P,       left:left+P]
        lrms_crop = lrms[:, top:top+P,       left:left+P]
        gt_crop   = gt[:,   top:top+P,       left:left+P]
        ms_crop   = ms[:,   ms_top:ms_top+ms_size, ms_left:ms_left+ms_size]

        return pan_crop, lrms_crop, gt_crop, ms_crop

    @staticmethod
    def _augment(
        pan:  np.ndarray,
        lrms: np.ndarray,
        gt:   np.ndarray,
        ms:   np.ndarray
    ) -> Tuple:
        """Random horizontal flip, vertical flip, and 90° rotation."""
        if random.random() > 0.5:
            pan  = pan[...,  ::-1]
            lrms = lrms[..., ::-1]
            gt   = gt[...,   ::-1]
            ms   = ms[...,   ::-1]
        if random.random() > 0.5:
            pan  = pan[...,  ::-1, :]
            lrms = lrms[..., ::-1, :]
            gt   = gt[...,   ::-1, :]
            ms   = ms[...,   ::-1, :]
        k = random.randint(0, 3)
        if k > 0:
            pan  = np.rot90(pan,  k, axes=(-2, -1))
            lrms = np.rot90(lrms, k, axes=(-2, -1))
            gt   = np.rot90(gt,   k, axes=(-2, -1))
            ms   = np.rot90(ms,   k, axes=(-2, -1))
        return pan, lrms, gt, ms


# =============================================================================
# DATALOADER FACTORY
# =============================================================================

def get_panscale_loaders(
    root:        str,
    batch_size:  int  = 16,
    patch_size:  int  = 128,
    num_workers: int  = 4,
    max_val:     float = 65535.0,
    scale_ratio: int  = 4,
) -> dict:
    """
    Create train/val/test DataLoaders for PanScale.

    Args:
        root:        Root directory of PanScale download (datasets/panscale)
        batch_size:  Training batch size
        patch_size:  PAN patch size for training crops
        num_workers: DataLoader workers
        max_val:     Normalization constant (65535 for uint16 TIFFs)
        scale_ratio: PAN/MS scale ratio

    Returns:
        dict with keys 'train', 'val', 'test' → DataLoader
    """
    train_ds = PanScaleDataset(root, "train", patch_size=patch_size,
                               augment=True,  max_val=max_val, scale_ratio=scale_ratio)
    val_ds   = PanScaleDataset(root, "val",   patch_size=None,
                               augment=False, max_val=max_val, scale_ratio=scale_ratio)

    loaders = {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        ),
        "val": DataLoader(
            val_ds, batch_size=1, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
    }

    # Test split (falls back to val if not present)
    try:
        test_ds = PanScaleDataset(root, "test", patch_size=None,
                                    augment=False, max_val=max_val, scale_ratio=scale_ratio)
        loaders["test"] = DataLoader(
            test_ds, batch_size=1, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
    except (FileNotFoundError, RuntimeError):
        loaders["test"] = loaders["val"]

    return loaders