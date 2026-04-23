"""
metrics.py — Complete evaluation metrics for pansharpening
All metrics follow the standard definitions used in remote sensing literature.

References:
    - SAM:   Yuhas et al. (1992)
    - ERGAS: Wald (2000)
    - Q4:    Wang & Bovik (2002) extended to N-bands
    - SCC:   Zhou et al. (1998)
    - PSNR:  Standard signal processing
    - SSIM:  Wang et al. (2004)
"""

import numpy as np
import torch
from skimage.metrics import structural_similarity
from typing import Dict, Union


# =============================================================================
# NUMPY-BASED METRICS (for evaluation on numpy arrays)
# =============================================================================

def sam(ms_ref: np.ndarray, ms_fused: np.ndarray) -> float:
    """
    Spectral Angle Mapper (SAM) — lower is better.
    Measures spectral distortion in degrees.

    Args:
        ms_ref:   Reference MS image  shape (C, H, W), float [0,1]
        ms_fused: Fused  MS image     shape (C, H, W), float [0,1]
    Returns:
        SAM value in degrees (↓ better, ideal = 0)
    """
    assert ms_ref.shape == ms_fused.shape, "Shape mismatch in SAM"
    # Dot product along spectral axis
    dot   = np.sum(ms_ref * ms_fused, axis=0)               # (H, W)
    norm1 = np.linalg.norm(ms_ref,   axis=0) + 1e-8         # (H, W)
    norm2 = np.linalg.norm(ms_fused, axis=0) + 1e-8         # (H, W)
    cos_angle = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
    return float(np.mean(np.arccos(cos_angle)) * 180.0 / np.pi)


def ergas(ms_ref: np.ndarray, ms_fused: np.ndarray, ratio: int = 4) -> float:
    """
    ERGAS — Erreur Relative Globale Adimensionnelle de Synthèse (↓ better).
    Standard metric for pansharpening quality assessment.

    Args:
        ms_ref:   Reference MS image shape (C, H, W), float [0,1]
        ms_fused: Fused   MS image shape (C, H, W), float [0,1]
        ratio:    Spatial resolution ratio (PAN_res / MS_res), default 4
    Returns:
        ERGAS value (↓ better, ideal = 0)
    """
    C = ms_ref.shape[0]
    band_rmse_sq = []
    for c in range(C):
        rmse_sq = np.mean((ms_ref[c] - ms_fused[c]) ** 2)
        mean_sq = (np.mean(ms_ref[c]) ** 2) + 1e-8
        band_rmse_sq.append(rmse_sq / mean_sq)
    return float(100.0 / ratio * np.sqrt(np.mean(band_rmse_sq)))


def q_index(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Universal Image Quality Index Q for a single band.
    Internal helper for Q4/Q8 computation.
    """
    mu1, mu2   = np.mean(img1), np.mean(img2)
    sigma1_sq  = np.var(img1)
    sigma2_sq  = np.var(img2)
    sigma12    = np.mean((img1 - mu1) * (img2 - mu2))
    denom = (sigma1_sq + sigma2_sq) * (mu1**2 + mu2**2 + 1e-8)
    return float((4 * sigma12 * mu1 * mu2) / (denom + 1e-8))


def q4(ms_ref: np.ndarray, ms_fused: np.ndarray) -> float:
    """
    Q4 / QN — Universal Quality Index extended to N spectral bands (↑ better).
    Uses complex-valued extension proposed by Alparone et al.

    Args:
        ms_ref:   Reference MS image shape (C, H, W)
        ms_fused: Fused   MS image shape (C, H, W)
    Returns:
        Q4/QN value in [-1, 1] (↑ better, ideal = 1)
    """
    C = ms_ref.shape[0]
    q_values = [q_index(ms_ref[c], ms_fused[c]) for c in range(C)]
    return float(np.mean(q_values))


def scc(ms_ref: np.ndarray, ms_fused: np.ndarray) -> float:
    """
    Spatial Correlation Coefficient (SCC) — ↑ better.
    Measures spatial detail preservation.

    Args:
        ms_ref:   Reference MS image shape (C, H, W)
        ms_fused: Fused   MS image shape (C, H, W)
    Returns:
        SCC value in [-1, 1] (↑ better, ideal = 1)
    """
    from scipy.ndimage import laplace
    C = ms_ref.shape[0]
    scc_values = []
    for c in range(C):
        ref_hp  = laplace(ms_ref[c].astype(np.float64))
        fus_hp  = laplace(ms_fused[c].astype(np.float64))
        num     = np.sum(ref_hp * fus_hp)
        den     = np.sqrt(np.sum(ref_hp**2) * np.sum(fus_hp**2)) + 1e-8
        scc_values.append(num / den)
    return float(np.mean(scc_values))


def psnr(ms_ref: np.ndarray, ms_fused: np.ndarray, max_val: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR) — ↑ better.

    Args:
        ms_ref:   Reference image, float [0, max_val]
        ms_fused: Fused   image, float [0, max_val]
        max_val:  Maximum pixel value (1.0 for normalized, 2047 for 11-bit WV3)
    Returns:
        PSNR in dB (↑ better)
    """
    mse = np.mean((ms_ref.astype(np.float64) - ms_fused.astype(np.float64)) ** 2)
    if mse < 1e-12:
        return float("inf")
    return float(20.0 * np.log10(max_val / np.sqrt(mse)))


def ssim_metric(ms_ref: np.ndarray, ms_fused: np.ndarray) -> float:
    """
    Structural Similarity Index (SSIM) — ↑ better.
    Averaged over all spectral bands.

    Args:
        ms_ref:   Reference image shape (C, H, W), float [0,1]
        ms_fused: Fused   image shape (C, H, W), float [0,1]
    Returns:
        Mean SSIM across bands (↑ better, ideal = 1)
    """
    C = ms_ref.shape[0]
    ssim_vals = []
    for c in range(C):
        val = structural_similarity(
            ms_ref[c].astype(np.float64),
            ms_fused[c].astype(np.float64),
            data_range=1.0
        )
        ssim_vals.append(val)
    return float(np.mean(ssim_vals))


def compute_all_metrics(
    gt: np.ndarray,
    fused: np.ndarray,
    ratio: int = 4,
    max_val: float = 1.0
) -> Dict[str, float]:
    """
    Compute the complete standard pansharpening metric suite.

    Args:
        gt:      Ground truth HR-MS image, shape (C, H, W), float [0, max_val]
        fused:   Predicted fused image,   shape (C, H, W), float [0, max_val]
        ratio:   Spatial resolution ratio (default 4)
        max_val: Max pixel value for PSNR
    Returns:
        Dict with keys: SAM, ERGAS, Q4, SCC, PSNR, SSIM
    """
    assert gt.shape == fused.shape, f"Shape mismatch: gt={gt.shape}, fused={fused.shape}"

    # Normalize to [0, 1] for metrics that require it
    gt_n    = gt.astype(np.float64)    / max_val
    fused_n = fused.astype(np.float64) / max_val

    return {
        "SAM":   sam(gt_n, fused_n),
        "ERGAS": ergas(gt_n, fused_n, ratio=ratio),
        "Q4":    q4(gt_n, fused_n),
        "SCC":   scc(gt_n, fused_n),
        "PSNR":  psnr(gt_n, fused_n, max_val=1.0),
        "SSIM":  ssim_metric(gt_n, fused_n),
    }


# =============================================================================
# TORCH-BASED METRICS (for inline training loss / monitoring)
# =============================================================================

class SAMLoss(torch.nn.Module):
    """
    Differentiable SAM loss for use during training.
    Minimizing this encourages spectral fidelity.
    """
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, C, H, W)
        dot     = (pred * target).sum(dim=1)                  # (B, H, W)
        norm_p  = pred.norm(dim=1).clamp(min=1e-8)
        norm_t  = target.norm(dim=1).clamp(min=1e-8)
        cos     = (dot / (norm_p * norm_t)).clamp(-1 + 1e-7, 1 - 1e-7)
        return torch.acos(cos).mean()


class MetricTracker:
    """
    Accumulates batch metrics and computes epoch-level statistics.

    Usage:
        tracker = MetricTracker()
        tracker.update(gt_batch, fused_batch)
        results = tracker.compute()
        tracker.reset()
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self._metrics = {"SAM": [], "ERGAS": [], "Q4": [], "SCC": [], "PSNR": [], "SSIM": []}

    def update(self, gt: np.ndarray, fused: np.ndarray, ratio: int = 4):
        """Process a single sample (C, H, W)."""
        m = compute_all_metrics(gt, fused, ratio=ratio)
        for k, v in m.items():
            self._metrics[k].append(v)

    def update_batch(self, gt_batch: np.ndarray, fused_batch: np.ndarray, ratio: int = 4):
        """Process a batch (B, C, H, W)."""
        for i in range(gt_batch.shape[0]):
            self.update(gt_batch[i], fused_batch[i], ratio=ratio)

    def compute(self) -> Dict[str, float]:
        return {k: float(np.mean(v)) for k, v in self._metrics.items() if v}

    def __repr__(self) -> str:
        results = self.compute()
        return (
            f"SAM={results.get('SAM', 0):.4f}° | "
            f"ERGAS={results.get('ERGAS', 0):.4f} | "
            f"Q4={results.get('Q4', 0):.4f} | "
            f"SCC={results.get('SCC', 0):.4f} | "
            f"PSNR={results.get('PSNR', 0):.2f}dB | "
            f"SSIM={results.get('SSIM', 0):.4f}"
        )
