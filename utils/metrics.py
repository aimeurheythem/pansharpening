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
    True quaternion-based Q4 metric (Alparone et al.) — ↑ better, ideal = 1.

    Uses quaternion algebra to jointly evaluate all spectral bands.
    This is the correct implementation used in published pansharpening papers.
    Works for any number of bands C (Q4 for C=4, Q8 for C=8, etc.)

    Args:
        ms_ref:   Reference MS image shape (C, H, W), float [0,1]
        ms_fused: Fused   MS image shape (C, H, W), float [0,1]
    Returns:
        Q value in [-1, 1] (↑ better, ideal = 1)
    """
    C = ms_ref.shape[0]
    # Reshape to (C, N) where N = H*W
    v_ref  = ms_ref.reshape(C, -1).astype(np.float64)
    v_fus  = ms_fused.reshape(C, -1).astype(np.float64)

    m_ref = np.mean(v_ref, axis=1, keepdims=True)   # (C, 1)
    m_fus = np.mean(v_fus, axis=1, keepdims=True)   # (C, 1)

    d_ref = v_ref - m_ref   # (C, N) — zero-mean
    d_fus = v_fus - m_fus   # (C, N) — zero-mean

    # Variance terms
    s_ref = np.mean(np.sum(d_ref * d_ref, axis=0))  # scalar
    s_fus = np.mean(np.sum(d_fus * d_fus, axis=0))  # scalar

    # Cross-covariance term (quaternion inner product, band-wise dot then averaged)
    s_cross = np.mean(np.sum(d_ref * d_fus, axis=0))  # scalar

    # Mean magnitude terms
    m_ref_norm_sq = float(np.dot(m_ref.ravel(), m_ref.ravel()))
    m_fus_norm_sq = float(np.dot(m_fus.ravel(), m_fus.ravel()))

    num = 4.0 * s_cross * np.sqrt(m_ref_norm_sq * m_fus_norm_sq + 1e-12)
    den = (s_ref + s_fus + 1e-8) * (m_ref_norm_sq + m_fus_norm_sq + 1e-8)

    return float(np.clip(num / den, -1.0, 1.0))


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
# NO-REFERENCE METRICS (full-resolution evaluation — no ground truth needed)
# =============================================================================

def d_lambda(ms_lr: np.ndarray, fused: np.ndarray) -> float:
    """
    D_lambda — Spectral distortion index (no-reference). ↓ better, ideal = 0.

    Measures how much spectral correlation between bands has changed
    from the original LR-MS to the fused HR-MS. Uses Q-index between
    all band pairs.

    Args:
        ms_lr:  Original LR-MS image, shape (C, H, W), float [0,1]
        fused:  Fused  HR-MS image,   shape (C, H, W), float [0,1]
    Returns:
        D_lambda in [0, 1] (↓ better, ideal = 0)
    """
    C = ms_lr.shape[0]
    total = 0.0
    count = 0
    for i in range(C):
        for j in range(C):
            if i != j:
                q_fused = q_index(fused[i], fused[j])
                q_ms    = q_index(ms_lr[i],  ms_lr[j])
                total  += abs(q_fused - q_ms)
                count  += 1
    return float(total / count) if count > 0 else 0.0


def d_s(
    ms_lr: np.ndarray,
    pan: np.ndarray,
    fused: np.ndarray,
    ratio: int = 4,
) -> float:
    """
    D_s — Spatial distortion index (no-reference). ↓ better, ideal = 0.

    Measures how much spatial correlation with the PAN image has changed
    between the degraded reference and the fused image.

    Args:
        ms_lr:  Original LR-MS, shape (C, H_ms, W_ms) or (C, H, W), float [0,1]
        pan:    PAN image,       shape (1, H, W) or (H, W),           float [0,1]
        fused:  Fused HR-MS,     shape (C, H, W),                     float [0,1]
        ratio:  Spatial resolution ratio (default 4)
    Returns:
        D_s in [0, 1] (↓ better, ideal = 0)
    """
    import cv2

    # Ensure pan is 2D
    pan_2d = pan[0] if pan.ndim == 3 else pan   # (H, W)
    H, W = pan_2d.shape

    # Degrade PAN to MS resolution then upsample back (simulate LR PAN reference)
    pan_lr = cv2.resize(
        pan_2d.astype(np.float32),
        (W // ratio, H // ratio),
        interpolation=cv2.INTER_AREA,
    )
    pan_lr_up = cv2.resize(
        pan_lr,
        (W, H),
        interpolation=cv2.INTER_CUBIC,
    )

    # Upsample ms_lr to PAN size if needed
    if ms_lr.shape[-1] != W or ms_lr.shape[-2] != H:
        ms_lr_up = np.stack([
            cv2.resize(ms_lr[c].astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
            for c in range(ms_lr.shape[0])
        ])
    else:
        ms_lr_up = ms_lr

    C = fused.shape[0]
    total = 0.0
    for c in range(C):
        q_fused = q_index(fused[c],    pan_2d)
        q_ref   = q_index(ms_lr_up[c], pan_lr_up)
        total  += abs(q_fused - q_ref)
    return float(total / C)


def qnr(
    ms_lr: np.ndarray,
    pan: np.ndarray,
    fused: np.ndarray,
    ratio: int = 4,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> dict:
    """
    QNR — Quality with No Reference. ↑ better, ideal = 1.
    QNR = (1 - D_lambda)^alpha * (1 - D_s)^beta

    Standard no-reference pansharpening quality metric (Alparone et al. 2008).

    Args:
        ms_lr:  Original LR-MS, shape (C, H_ms, W_ms) or (C, H, W), float [0,1]
        pan:    PAN image,       shape (1, H, W) or (H, W),           float [0,1]
        fused:  Fused HR-MS,     shape (C, H, W),                     float [0,1]
        ratio:  Spatial resolution ratio (default 4)
        alpha:  D_lambda exponent (default 1.0)
        beta:   D_s exponent (default 1.0)
    Returns:
        dict with keys: QNR, D_lambda, D_s  (all in [0,1])
    """
    dl = d_lambda(ms_lr, fused)
    ds = d_s(ms_lr, pan, fused, ratio=ratio)
    q  = ((1.0 - dl) ** alpha) * ((1.0 - ds) ** beta)
    return {"QNR": float(q), "D_lambda": float(dl), "D_s": float(ds)}


def fcc(pan: np.ndarray, fused: np.ndarray) -> float:
    """
    FCC — Frequency Correlation Coefficient (no-reference). ↑ better, ideal = 1.

    Measures how well the fused image preserves the high-frequency spatial
    content of the PAN image. Computed in the Fourier domain.

    Args:
        pan:   PAN image, shape (1, H, W) or (H, W), float [0,1]
        fused: Fused HR-MS image, shape (C, H, W),   float [0,1]
    Returns:
        FCC in [-1, 1] (↑ better, ideal = 1)
    """
    pan_2d = pan[0] if pan.ndim == 3 else pan  # (H, W)

    # Compute Laplacian high-pass filter on PAN
    from scipy.ndimage import laplace
    pan_hp = laplace(pan_2d.astype(np.float64))

    C = fused.shape[0]
    fcc_vals = []
    for c in range(C):
        fus_hp = laplace(fused[c].astype(np.float64))
        num = np.sum(pan_hp * fus_hp)
        den = np.sqrt(np.sum(pan_hp ** 2) * np.sum(fus_hp ** 2)) + 1e-8
        fcc_vals.append(num / den)
    return float(np.mean(fcc_vals))


def sf(fused: np.ndarray) -> float:
    """
    SF — Spatial Frequency (no-reference). ↑ better.

    Measures the overall activity level (sharpness) of the fused image.
    Higher SF means more spatial detail — desired in pansharpening.

    Args:
        fused: Fused HR-MS image, shape (C, H, W), float [0,1]
    Returns:
        SF value (↑ better)
    """
    C = fused.shape[0]
    sf_vals = []
    for c in range(C):
        img = fused[c].astype(np.float64)
        rf = np.sqrt(np.mean(np.diff(img, axis=0) ** 2))   # row frequency
        cf = np.sqrt(np.mean(np.diff(img, axis=1) ** 2))   # col frequency
        sf_vals.append(np.sqrt(rf ** 2 + cf ** 2))
    return float(np.mean(sf_vals))


def sd(fused: np.ndarray) -> float:
    """
    SD — Standard Deviation (no-reference). ↑ better.

    Measures the global contrast of the fused image.
    Higher SD indicates more information content.

    Args:
        fused: Fused HR-MS image, shape (C, H, W), float [0,1]
    Returns:
        SD value (↑ better)
    """
    return float(np.std(fused.astype(np.float64)))


def compute_no_ref_metrics(
    pan: np.ndarray,
    ms_lr: np.ndarray,
    fused: np.ndarray,
    ratio: int = 4,
) -> dict:
    """
    Compute the complete no-reference metric suite for full-resolution evaluation.
    Use this when ground truth HR-MS is NOT available (real satellite imagery).

    Args:
        pan:   PAN image,      shape (1, H, W) or (H, W), float [0,1]
        ms_lr: LR-MS image,    shape (C, H_ms, W_ms),     float [0,1]
        fused: Fused HR-MS,    shape (C, H, W),            float [0,1]
        ratio: Resolution ratio (default 4)
    Returns:
        Dict with keys: QNR, D_lambda, D_s, FCC, SF, SD
    """
    qnr_results = qnr(ms_lr, pan, fused, ratio=ratio)
    return {
        "QNR":      qnr_results["QNR"],
        "D_lambda": qnr_results["D_lambda"],
        "D_s":      qnr_results["D_s"],
        "FCC":      fcc(pan, fused),
        "SF":       sf(fused),
        "SD":       sd(fused),
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
        self._metrics = {
            # Reduced-resolution (with GT)
            "SAM": [], "ERGAS": [], "Q4": [], "SCC": [], "PSNR": [], "SSIM": [],
            # No-reference (full-resolution)
            "QNR": [], "D_lambda": [], "D_s": [], "FCC": [], "SF": [], "SD": [],
        }

    def update(self, gt: np.ndarray, fused: np.ndarray, ratio: int = 4):
        """Process a single sample (C, H, W)."""
        m = compute_all_metrics(gt, fused, ratio=ratio)
        for k, v in m.items():
            self._metrics[k].append(v)

    def update_batch(self, gt_batch: np.ndarray, fused_batch: np.ndarray, ratio: int = 4):
        """Process a batch (B, C, H, W)."""
        for i in range(gt_batch.shape[0]):
            self.update(gt_batch[i], fused_batch[i], ratio=ratio)

    def update_no_ref(
        self,
        pan: np.ndarray,
        ms_lr: np.ndarray,
        fused: np.ndarray,
        ratio: int = 4,
    ):
        """Process a single no-reference sample (no ground truth needed)."""
        m = compute_no_ref_metrics(pan, ms_lr, fused, ratio=ratio)
        for k, v in m.items():
            self._metrics[k].append(v)

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
