"""
psgan_losses.py — PSGAN-DWT Loss Functions

Contains three spectral regularizers from the paper:
  - SAM (Spectral Angle Mapper) loss
  - Perceptual loss (bottleneck features)
  - Gram matrix perceptual loss
  - Combined PSGAN-DWT loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PSGANGeneratorLoss(nn.Module):
    """
    Generator adversarial + L1 loss (Equation 2 from paper).

    L(G) = α * GAN_loss(D_fake, real) + β * L1(G(pan, lrms), hrms)
    """
    def __init__(self, alpha: float = 1.0, beta: float = 100.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gan_loss = nn.MSELoss()

    def forward(
        self,
        fake_pred: torch.Tensor,
        fake_img: torch.Tensor,
        real_img: torch.Tensor,
    ) -> torch.Tensor:
        adv_loss = self.gan_loss(fake_pred, torch.ones_like(fake_pred))
        l1_loss = F.l1_loss(fake_img, real_img)
        return self.alpha * adv_loss + self.beta * l1_loss


class PSGANDiscriminatorLoss(nn.Module):
    """
    Discriminator loss (Equation 3 from paper).

    L(D) = 0.5 * [(D_real - 1)^2 + D_fake^2]
    """
    def __init__(self):
        super().__init__()
        self.gan_loss = nn.MSELoss()

    def forward(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor,
    ) -> torch.Tensor:
        loss_real = self.gan_loss(real_pred, torch.ones_like(real_pred))
        loss_fake = self.gan_loss(fake_pred, torch.zeros_like(fake_pred))
        return 0.5 * (loss_real + loss_fake)


class PSGANSAMLoss(nn.Module):
    """
    SAM-based spectral regularizer (Equation 4).
    Minimizes spectral angle between generated and reference HRMS.
    """
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_norm = F.normalize(pred, dim=1, eps=1e-8)
        target_norm = F.normalize(target, dim=1, eps=1e-8)
        cos_sim = (pred_norm * target_norm).sum(dim=1)
        return (1.0 - cos_sim).mean()


class PSGANDualSAMLoss(nn.Module):
    """
    SAM on both full and downsampled resolution (Equation 5).
    Total_SAM = 0.5 * SAM(pred, gt) + 0.5 * SAM(pred_down, lrms)
    """
    def __init__(self, scale: int = 4):
        super().__init__()
        self.scale = scale

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        lrms: torch.Tensor,
    ) -> torch.Tensor:
        sam_full = PSGANSAMLoss()(pred, gt)
        # Downsample BOTH pred and lrms to the same reduced resolution
        # (lrms in data pipeline is already upsampled to PAN resolution)
        pred_down = F.interpolate(
            pred,
            scale_factor=1 / self.scale,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        lrms_down = F.interpolate(
            lrms,
            size=pred_down.shape[2:],
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        sam_down = PSGANSAMLoss()(pred_down, lrms_down)
        return 0.5 * sam_full + 0.5 * sam_down


class PSGANPerceptualLoss(nn.Module):
    """
    Perceptual loss: L2 distance between bottleneck features
    of generated and real HR-MS images passed through the generator.
    """
    def __init__(self, generator: nn.Module):
        super().__init__()
        self.generator = generator
        self._features_real = None
        self._features_fake = None
        self._hook_registered = False

    def _hook_fn(self, module, input, output):
        self._features_fake = output

    def register_hook(self, layer_idx: int = -1):
        if not self._hook_registered:
            self.generator.ms_encoder[layer_idx].register_forward_hook(self._hook_fn)
            self._hook_registered = True

    def forward(
        self,
        pred_hrms: torch.Tensor,
        real_hrms: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract bottleneck features from ms_encoder by partial forward pass.
        Compare features of pred vs real via MSE (L2 distance).
        """
        # Extract features for fake image
        def _encode(img):
            # Run through ms_encoder using HF subbands of the image itself
            from utils.losses import HaarDWT2D
            dwt = HaarDWT2D()
            _, LH, HL, HH = dwt(img)
            x = torch.cat([LH, HL, HH], dim=1)
            feats = []
            for layer in self.generator.ms_encoder:
                x = layer(x)
                feats.append(x)
            return feats[-1]  # bottleneck features

        feat_pred = _encode(pred_hrms)
        feat_real = _encode(real_hrms.detach())  # detach real to avoid unnecessary graph
        return F.mse_loss(feat_pred, feat_real)


class PSGANGramPerceptualLoss(nn.Module):
    """
    Gram matrix perceptual loss (Equations 7-9).
    Computes gram matrix of bottleneck features and minimizes distance.
    """
    @staticmethod
    def gram_matrix(features: torch.Tensor) -> torch.Tensor:
        B, C, H, W = features.shape
        f = features.view(B, C, -1)
        gram = torch.bmm(f, f.transpose(1, 2))
        return gram / (C * H * W)

    def forward(
        self,
        feat_pred: torch.Tensor,
        feat_real: torch.Tensor,
    ) -> torch.Tensor:
        G_pred = self.gram_matrix(feat_pred)
        G_real = self.gram_matrix(feat_real)
        return F.mse_loss(G_pred, G_real)


class PSGANGramReconLoss(nn.Module):
    """
    Gram matrix reconstruction loss (Equation 10).
    Applied directly on pixel values of generated and reference images.
    """
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, W = pred.shape
        p = pred.view(B, C, -1)
        t = target.view(B, C, -1)
        gram_pred = torch.bmm(p, p.transpose(1, 2)) / (C * H * W)
        gram_target = torch.bmm(t, t.transpose(1, 2)) / (C * H * W)
        return F.mse_loss(gram_pred, gram_target)


class PSGANDWTLoss(nn.Module):
    """
    Complete PSGAN-DWT training loss.

    Combines:
      - Generator adversarial + L1 loss    [weight: eta1]
      - Spectral regularizer               [weight: eta2]
        Regularizer = SAM + Perceptual + Gram Matrix (configurable)

    Args:
        alpha:          Adversarial weight in L(G) (default 1.0)
        beta:           L1 weight in L(G) (default 100.0)
        eta1:           L(G) weight in final loss (default 1.0)
        eta2:           Regularizer weight in final loss (default 0.1)
        regularizer:    Which regularizer to use:
                        'sam'        — SAM loss only
                        'sam_dual'   — Dual-resolution SAM
                        'perceptual' — Perceptual loss
                        'gram_perc'  — Gram matrix perceptual
                        'gram_recon' — Gram matrix reconstruction (BEST per paper)
                        'all'        — All regularizers combined
        generator:      Generator module (needed for perceptual losses)
    """
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 100.0,
        eta1: float = 1.0,
        eta2: float = 0.1,
        regularizer: str = "gram_recon",
        generator: nn.Module = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eta1 = eta1
        self.eta2 = eta2
        self.regularizer = regularizer
        self.generator = generator

        self.gen_loss = PSGANGeneratorLoss(alpha=alpha, beta=beta)
        self.disc_loss = PSGANDiscriminatorLoss()

    def _get_regularizer(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        lrms: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        reg = self.regularizer.lower()
        if reg == "sam":
            return PSGANSAMLoss()(pred, target)
        elif reg == "sam_dual":
            return PSGANDualSAMLoss()(pred, target, lrms)
        elif reg == "perceptual" and self.generator is not None:
            return PSGANPerceptualLoss(self.generator)(pred, target)
        elif reg == "gram_perc" and self.generator is not None:
            return PSGANGramPerceptualLoss()(pred, target)
        elif reg == "gram_recon":
            return PSGANGramReconLoss()(pred, target)
        elif reg == "all":
            sam = PSGANSAMLoss()(pred, target)
            gram = PSGANGramReconLoss()(pred, target)
            return sam + gram
        return torch.tensor(0.0, device=pred.device)

    def generator_loss(
        self,
        fake_pred: torch.Tensor,
        fake_img: torch.Tensor,
        real_img: torch.Tensor,
        lrms: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        loss_adv = self.gen_loss(fake_pred, fake_img, real_img)
        loss_l1 = F.l1_loss(fake_img, real_img)
        loss_reg = self._get_regularizer(fake_img, real_img, lrms)
        loss_total = self.eta1 * loss_adv + self.eta2 * loss_reg

        return loss_total, {
            "loss_adv": loss_adv.item(),
            "loss_l1": loss_l1.item(),
            "loss_reg": loss_reg.item(),
            "loss_total": loss_total.item(),
        }

    def discriminator_loss(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        loss = self.disc_loss(real_pred, fake_pred)
        return loss, {"loss_disc": loss.item()}