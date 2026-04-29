"""
losses.py — Loss functions for pansharpening training

Includes:
  - HybridPanLoss:   Spectral (L1) + Spatial (SSIM) + SAM
  - WaveletLoss:     Multi-scale wavelet consistency (for Wav-CBT)
  - PerceptualLoss:  VGG feature-space loss (for ScaleFormer)

FIXED:
  - WaveletLoss._dwt_loss previously used pywt + numpy, which:
      (a) detached gradients entirely — the wavelet term had ZERO gradient
      (b) ran on CPU per sample/channel — extreme throughput bottleneck
    Now uses a pure-PyTorch differentiable Haar DWT (GPU-compatible, fully
    differentiable, equivalent result to the analytical Haar transform).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# =============================================================================
# BUILDING BLOCKS
# =============================================================================

class L1Loss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)


class SSIMLoss(nn.Module):
    """
    Differentiable SSIM loss.  1 - SSIM (minimized during training).
    Window: Gaussian kernel with sigma=1.5 over 11x11 neighborhood.
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
# DIFFERENTIABLE HAAR DWT — replaces pywt (GPU-compatible, gradient-preserving)
# =============================================================================

class HaarDWT2D(nn.Module):
    """
    Differentiable 2D Haar Discrete Wavelet Transform using fixed-weight Conv2d.

    Unlike pywt-based approaches, this:
      - Runs entirely on GPU (no CPU/numpy round-trip)
      - Preserves the full gradient computation graph
      - Is analytically equivalent to the Haar DWT

    Returns 4 subbands: LL, LH, HL, HH — each (B, C, H/2, W/2)
    """
    def __init__(self):
        super().__init__()
        # LL: average both dims | LH: avg H, diff W
        # HL: diff H, avg W    | HH: diff both dims
        h = torch.tensor([[1.,  1.], [ 1.,  1.]]) / 2.0
        g = torch.tensor([[1., -1.], [ 1., -1.]]) / 2.0
        v = torch.tensor([[1.,  1.], [-1., -1.]]) / 2.0
        d = torch.tensor([[1., -1.], [-1.,  1.]]) / 2.0
        # Stack as a single (4, 1, 2, 2) filter bank
        filters = torch.stack([h, g, v, d], dim=0).unsqueeze(1)  # (4,1,2,2)
        self.register_buffer("filters", filters)

    def forward(self, x: torch.Tensor) -> tuple:
        B, C, H, W = x.shape
        x_flat = x.reshape(B * C, 1, H, W)
        out = F.conv2d(x_flat, self.filters, stride=2, padding=0)  # (B*C, 4, H/2, W/2)
        out = out.reshape(B, C, 4, H // 2, W // 2)
        LL, LH, HL, HH = out.unbind(dim=2)
        return LL, LH, HL, HH


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
    ) -> tuple:
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

    Decomposes pred and target using a differentiable Haar DWT and
    computes L1 on each high-frequency subband (LH, HL, HH).

    FIX: The original implementation used pywt + numpy which had two bugs:
      BUG 1 — pred.detach().cpu().numpy() cut the gradient graph entirely.
               The wavelet component contributed ZERO gradient during backprop,
               making it a monitoring value, not a training signal.
      BUG 2 — CPU round-trip per sample/channel was an extreme bottleneck.
    This version uses a fully differentiable GPU Haar DWT instead.

    Args:
        levels:   DWT decomposition levels (default 2)
        l1_w:     L1 pixel loss weight
        wav_w:    Wavelet subband loss weight
        ssim_w:   SSIM loss weight
    """
    def __init__(
        self,
        levels:  int   = 2,
        l1_w:    float = 1.0,
        wav_w:   float = 0.2,
        ssim_w:  float = 0.5,
    ):
        super().__init__()
        self.levels  = levels
        self.l1_w, self.wav_w, self.ssim_w = l1_w, wav_w, ssim_w
        self.l1   = L1Loss()
        self.ssim = SSIMLoss()
        self.dwt  = HaarDWT2D()

    def _multiscale_wavelet_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Multi-level Haar DWT loss on high-frequency subbands.

        Entirely differentiable — runs on GPU — gradients flow through
        every level of decomposition back to the network output.
        """
        loss = pred.new_zeros(1).squeeze()  # scalar zero, same device/dtype
        p, t = pred, target

        for level in range(self.levels):
            # Pad to even spatial dims if needed
            if p.shape[-2] % 2 != 0:
                p = F.pad(p, (0, 0, 0, 1))
                t = F.pad(t, (0, 0, 0, 1))
            if p.shape[-1] % 2 != 0:
                p = F.pad(p, (0, 1, 0, 0))
                t = F.pad(t, (0, 1, 0, 0))

            p_LL, p_LH, p_HL, p_HH = self.dwt(p)
            t_LL, t_LH, t_HL, t_HH = self.dwt(t)

            # High-frequency subband L1 (deeper levels get lower weight)
            level_weight = 1.0 / (2 ** level)
            loss = loss + level_weight * (
                F.l1_loss(p_LH, t_LH) +
                F.l1_loss(p_HL, t_HL) +
                F.l1_loss(p_HH, t_HH)
            )

            # Recurse on LL for next decomposition level
            p, t = p_LL, t_LL

        return loss

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple:
        l1_loss   = self.l1(pred, target)
        wav_loss  = self._multiscale_wavelet_loss(pred, target)
        ssim_loss = self.ssim(pred, target)

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
            self.slice1 = nn.Sequential(*list(vgg)[:9]) # relu2_2
            self.slice2 = nn.Sequential(*list(vgg)[9:16]) # relu3_3
            for p in self.parameters():
                p.requires_grad = False
        except Exception:
            self.slice1 = None
            self.slice2 = None

    def _to_3ch(self, x: torch.Tensor) -> torch.Tensor:
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
        f1_p = self.slice1(p3); f1_t = self.slice1(t3)
        f2_p = self.slice2(f1_p); f2_t = self.slice2(f1_t)
        return F.l1_loss(f1_p, f1_t) + F.l1_loss(f2_p, f2_t)


# =============================================================================
# GAN LOSSES — For Pan-Pix2Pix adversarial training
# =============================================================================

class GANLoss(nn.Module):
    """
    Vanilla GAN loss (LSGAN or BCE variant).

    LSGAN (least-squares GAN) uses MSE loss instead of BCE for more
    stable training and better gradient behavior. This is the standard
    choice for Pix2Pix-style conditional GANs.

    Args:
        gan_mode: "lsgan" (MSE, default) or "vanilla" (BCE)
        target_real: Label value for real samples (default 1.0)
        target_fake: Label value for fake samples (default 0.0)
    """
    def __init__(
        self,
        gan_mode: str = "lsgan",
        target_real: float = 1.0,
        target_fake: float = 0.0,
    ):
        super().__init__()
        self.gan_mode = gan_mode
        self.target_real = target_real
        self.target_fake = target_fake
        if gan_mode == "lsgan":
            self.loss_fn = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown gan_mode '{gan_mode}'. Use 'lsgan' or 'vanilla'.")

    def forward(
        self,
        prediction: torch.Tensor,
        is_real: bool,
    ) -> torch.Tensor:
        if self.gan_mode == "lsgan":
            target = torch.full_like(prediction, self.target_real if is_real else self.target_fake)
            return self.loss_fn(prediction, target)
        else:
            target = torch.full_like(
                prediction, 1.0 if is_real else 0.0,
            )
            return self.loss_fn(prediction, target)


class Pix2PixLoss(nn.Module):
    """
    Composite loss for Pan-Pix2Pix GAN training.

    Generator loss:
      G_loss = λ_L1 * L1(G(x), y) + λ_SSIM * SSIM(G(x), y) + λ_SAM * SAM(G(x), y)
               + λ_GAN * GAN_loss(D(G(x), x), real)

    Discriminator loss:
      D_loss = 0.5 * [GAN_loss(D(y, x), real) + GAN_loss(D(G(x), x), fake)]

    The L1 weight is typically much larger than the GAN weight (λ_L1=100, λ_GAN=1)
    following the original Pix2Pix paper, which found that L1 dominates for
    image quality while GAN adds high-frequency texture realism.

    Args:
        l1_w: Weight for L1 reconstruction loss (default 100.0 — Pix2Pix standard)
        ssim_w: Weight for SSIM structural loss
        sam_w: Weight for SAM spectral angle loss
        gan_w: Weight for adversarial GAN loss (default 1.0)
        gan_mode: "lsgan" or "vanilla"
    """
    def __init__(
        self,
        l1_w: float = 100.0,
        ssim_w: float = 10.0,
        sam_w: float = 5.0,
        gan_w: float = 1.0,
        gan_mode: str = "lsgan",
    ):
        super().__init__()
        self.l1_w = l1_w
        self.ssim_w = ssim_w
        self.sam_w = sam_w
        self.gan_w = gan_w
        self.gan_loss = GANLoss(gan_mode=gan_mode)
        self.l1 = L1Loss()
        self.ssim = SSIMLoss()
        self.sam = SAMLoss()

    def generator_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        d_pred_fake: torch.Tensor,
    ) -> tuple:
        l1_loss = self.l1(pred, gt)
        ssim_loss = self.ssim(pred, gt)
        sam_loss = self.sam(pred, gt)
        gan_loss = self.gan_loss(d_pred_fake, is_real=True)

        total = (
            self.l1_w * l1_loss
            + self.ssim_w * ssim_loss
            + self.sam_w * sam_loss
            + self.gan_w * gan_loss
        )
        components = {
            "G_loss_l1": l1_loss.item(),
            "G_loss_ssim": ssim_loss.item(),
            "G_loss_sam": sam_loss.item(),
            "G_loss_gan": gan_loss.item(),
            "G_loss_total": total.item(),
        }
        return total, components

    def discriminator_loss(
        self,
        d_pred_real: torch.Tensor,
        d_pred_fake: torch.Tensor,
    ) -> tuple:
        d_loss_real = self.gan_loss(d_pred_real, is_real=True)
        d_loss_fake = self.gan_loss(d_pred_fake, is_real=False)
        total = (d_loss_real + d_loss_fake) * 0.5
        components = {
            "D_loss_real": d_loss_real.item(),
            "D_loss_fake": d_loss_fake.item(),
            "D_loss_total": total.item(),
        }
        return total, components


# =============================================================================
# LOSS FACTORY
# =============================================================================

LOSS_REGISTRY = {
    "hybrid": HybridPanLoss,
    "wavelet": WaveletLoss,
    "pix2pix": Pix2PixLoss,
}

def get_loss(name: str, **kwargs) -> nn.Module:
    """
    Factory function to instantiate loss by name.

    Usage:
        loss_fn = get_loss("hybrid",  l1_w=1.0, ssim_w=0.5, sam_w=0.1)
        loss_fn = get_loss("wavelet", levels=2, wav_w=0.2)
    """
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name](**kwargs)