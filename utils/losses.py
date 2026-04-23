"""
losses.py — Loss functions for pansharpening training

Includes:
  - HybridPanLoss:      Spectral (L1) + Spatial (SSIM) + SAM
  - WaveletLoss:        Multi-scale wavelet consistency (for Wav-CBT)
  - PerceptualLoss:     VGG feature-space loss (for ScaleFormer)
  - SpectralSpatialLoss: All-in-one configurable loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import pywt
import numpy as np


# =============================================================================
# BUILDING BLOCKS
# =============================================================================

class L1Loss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)


class SSIMLoss(nn.Module):
    """
    Differentiable SSIM loss.  1 - SSIM (minimized during training).
    Window: Gaussian kernel with sigma=1.5 over 11×11 neighborhood.
    """
    def __init__(self, window_size: int = 11, channel: int = 1):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.register_buffer("window", self._gaussian_window(window_size, 1.5))

    @staticmethod
    def _gaussian_window(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.outer(g)
        return window.unsqueeze(0).unsqueeze(0)   # (1,1,W,W)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pred.shape
        window = self.window.expand(C, 1, -1, -1).to(pred.device)

        mu1 = F.conv2d(pred,   window, padding=self.window_size//2, groups=C)
        mu2 = F.conv2d(target, window, padding=self.window_size//2, groups=C)

        mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2
        sigma1_sq = F.conv2d(pred*pred,     window, padding=self.window_size//2, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(target*target, window, padding=self.window_size//2, groups=C) - mu2_sq
        sigma12   = F.conv2d(pred*target,   window, padding=self.window_size//2, groups=C) - mu1_mu2

        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1.0 - ssim_map.mean()


class SAMLoss(nn.Module):
    """Differentiable Spectral Angle Mapper loss."""
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dot    = (pred * target).sum(dim=1)
        norm_p = pred.norm(dim=1).clamp(min=1e-8)
        norm_t = target.norm(dim=1).clamp(min=1e-8)
        cos    = (dot / (norm_p * norm_t)).clamp(-1 + 1e-7, 1 - 1e-7)
        return torch.acos(cos).mean()


# =============================================================================
# COMPOSITE LOSSES
# =============================================================================

class HybridPanLoss(nn.Module):
    """
    Standard hybrid loss for pansharpening (spectral + spatial + SAM).
    
    Default weights tuned for WV3 PanBench benchmark.
    Adjust 'sam_w' upwards for datasets with high spectral complexity (WV3).
    Adjust 'ssim_w' upwards for datasets with rich spatial texture (PanScale).

    Args:
        l1_w:   Weight for L1 spectral loss (default 1.0)
        ssim_w: Weight for SSIM spatial loss (default 0.5)
        sam_w:  Weight for SAM spectral angle loss (default 0.1)
    """
    def __init__(self, l1_w: float = 1.0, ssim_w: float = 0.5, sam_w: float = 0.1):
        super().__init__()
        self.l1_w, self.ssim_w, self.sam_w = l1_w, ssim_w, sam_w
        self.l1   = L1Loss()
        self.ssim = SSIMLoss()
        self.sam  = SAMLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        l1_loss   = self.l1(pred, target)
        ssim_loss = self.ssim(pred, target)
        sam_loss  = self.sam(pred, target)

        total = self.l1_w * l1_loss + self.ssim_w * ssim_loss + self.sam_w * sam_loss
        components = {
            "loss_l1":   l1_loss.item(),
            "loss_ssim": ssim_loss.item(),
            "loss_sam":  sam_loss.item(),
            "loss_total": total.item(),
        }
        return total, components


class WaveletLoss(nn.Module):
    """
    Multi-scale wavelet consistency loss for Wav-CBT.
    Decomposes both pred and target using DWT and computes L1 on subbands.

    Args:
        wavelet:   PyWavelets wavelet name ('haar', 'db2', 'sym4')
        levels:    Decomposition levels
        l1_w:      L1 pixel loss weight
        wav_w:     Wavelet subband loss weight
        ssim_w:    SSIM loss weight
    """
    def __init__(
        self,
        wavelet: str = "haar",
        levels: int = 3,
        l1_w: float = 1.0,
        wav_w: float = 0.2,
        ssim_w: float = 0.5
    ):
        super().__init__()
        self.wavelet = wavelet
        self.levels  = levels
        self.l1_w, self.wav_w, self.ssim_w = l1_w, wav_w, ssim_w
        self.l1   = L1Loss()
        self.ssim = SSIMLoss()

    def _dwt_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute wavelet subband L1 loss for a single band."""
        p_np = pred.detach().cpu().numpy()
        t_np = target.detach().cpu().numpy()
        total = 0.0
        for b in range(p_np.shape[0]):    # batch
            for c in range(p_np.shape[1]):  # channel
                coeffs_p = pywt.wavedec2(p_np[b, c], self.wavelet, level=self.levels)
                coeffs_t = pywt.wavedec2(t_np[b, c], self.wavelet, level=self.levels)
                for cp, ct in zip(coeffs_p[1:], coeffs_t[1:]):   # skip LL (already in L1)
                    for sp, st in zip(cp, ct):                    # LH, HL, HH subbands
                        total += np.mean(np.abs(sp - st))
        return torch.tensor(total / (p_np.shape[0] * p_np.shape[1]), device=pred.device)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        l1_loss  = self.l1(pred, target)
        wav_loss = self._dwt_loss(pred, target)
        ssim_loss= self.ssim(pred, target)

        total = self.l1_w * l1_loss + self.wav_w * wav_loss + self.ssim_w * ssim_loss
        components = {
            "loss_l1":      l1_loss.item(),
            "loss_wavelet": wav_loss.item(),
            "loss_ssim":    ssim_loss.item(),
            "loss_total":   total.item(),
        }
        return total, components


class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss for ScaleFormer (high-res cross-scale).
    Extracts features from VGG16 relu2_2 and relu3_3 layers.
    """
    def __init__(self, layers: list = ["relu2_2", "relu3_3"]):
        super().__init__()
        try:
            import torchvision.models as tvm
            vgg = tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT).features
            self.slice1 = nn.Sequential(*list(vgg)[:9])    # relu2_2
            self.slice2 = nn.Sequential(*list(vgg)[9:16])  # relu3_3
            for p in self.parameters():
                p.requires_grad = False
        except Exception:
            self.slice1 = None
            self.slice2 = None

    def _to_3ch(self, x: torch.Tensor) -> torch.Tensor:
        """Replicate first 3 channels if input has more/fewer."""
        if x.shape[1] == 3:
            return x
        if x.shape[1] > 3:
            return x[:, :3]
        return x.repeat(1, 3, 1, 1)[:, :3]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.slice1 is None:
            return torch.tensor(0.0, device=pred.device)
        p3 = self._to_3ch(pred)
        t3 = self._to_3ch(target)
        f1_p = self.slice1(p3);    f1_t = self.slice1(t3)
        f2_p = self.slice2(f1_p);  f2_t = self.slice2(f1_t)
        return F.l1_loss(f1_p, f1_t) + F.l1_loss(f2_p, f2_t)


# =============================================================================
# LOSS FACTORY
# =============================================================================

LOSS_REGISTRY = {
    "hybrid":    HybridPanLoss,
    "wavelet":   WaveletLoss,
}

def get_loss(name: str, **kwargs) -> nn.Module:
    """
    Factory function to instantiate loss by name.
    
    Usage:
        loss_fn = get_loss("hybrid", l1_w=1.0, ssim_w=0.5, sam_w=0.1)
        loss_fn = get_loss("wavelet", wavelet="haar", levels=3)
    """
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name](**kwargs)
