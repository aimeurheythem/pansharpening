"""
psgan_dwt.py — PSGAN-DWT: Two-Stream Wavelet U-Net GAN for Pansharpening

Based on:
  - PSGAN (Liu et al., ICIP 2018): Conditional GAN for pansharpening
  - DWT domain processing: separates spectral (LL) from spatial (LH, HL, HH)
  - Three spectral regularizers from Kantharia et al., arXiv 2024

Key innovation: generator operates in wavelet domain, reducing chromatic
drift and stabilizing adversarial training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class HaarIDWT2D(nn.Module):
    """
    Differentiable 2D Haar Inverse DWT.

    Input: LL, LH, HL, HH — each (B, C, H, W)
    Output: reconstructed image (B, C, H*2, W*2)

    Inverse of HaarDWT2D. Uses the exact transpose operation.
    """
    def forward(self, LL: torch.Tensor, LH: torch.Tensor, HL: torch.Tensor, HH: torch.Tensor) -> torch.Tensor:
        B, C, H, W = LL.shape

        top_left     = (LL + LH + HL + HH)
        top_right    = (LL - LH + HL - HH)
        bottom_left = (LL + LH - HL - HH)
        bottom_right = (LL - LH - HL + HH)

        out = torch.zeros(B, C, H * 2, W * 2, device=LL.device, dtype=LL.dtype)
        out[:, :, 0::2, 0::2] = top_left
        out[:, :, 0::2, 1::2] = top_right
        out[:, :, 1::2, 0::2] = bottom_left
        out[:, :, 1::2, 1::2] = bottom_right
        return out / 2.0


class UNetDown(nn.Module):
    """UNet downsampling block (Conv2d stride=2, BN, LeakyReLU)."""
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        normalize: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UNetUp(nn.Module):
    """UNet upsampling block using PixelShuffle."""
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
    ):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, 3, 1, 1, bias=False),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Conv2d(out_ch + skip_ch, out_ch, 3, 1, 1, bias=False)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class PSGANDWTGenerator(nn.Module):
    """
    Two-Stream Wavelet U-Net Generator.

    Architecture:
      1. DWT on both LRMS and HRPAN → 4 subbands each
      2. Two encoders process HF subbands (LH, HL, HH) separately
      3. Bottleneck fusion combines MS and PAN features
      4. Decoder predicts enriched HF subbands
      5. Weighted fusion: LL_ms + w*(HF* - HF_ms)
      6. IDWT → full resolution HRMS

    Args:
        ms_channels: Number of MS spectral bands (8 for WV3)
        pan_channels: Number of PAN channels (always 1)
        base_features: Base feature count (64)
        n_levels: Number of encoder levels (3)
        dropout: Dropout rate in bottleneck
    """
    def __init__(
        self,
        ms_channels: int = 8,
        pan_channels: int = 1,
        base_features: int = 64,
        n_levels: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ms_channels = ms_channels
        self.pan_channels = pan_channels
        self.n_levels = n_levels
        b = base_features

        from utils.losses import HaarDWT2D
        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D()

        self.fusion_w = nn.Parameter(torch.ones(1) * 0.5)

        ms_in_ch = ms_channels * 3
        pan_in_ch = pan_channels * 4

        self.ms_encoder = nn.ModuleList()
        self.pan_encoder = nn.ModuleList()

        ch = ms_in_ch
        for i in range(n_levels):
            out_ch = b * (2 ** i)
            self.ms_encoder.append(UNetDown(ch, out_ch, normalize=(i > 0), dropout=dropout))
            ch = out_ch

        ch = pan_in_ch
        for i in range(n_levels):
            out_ch = b * (2 ** i)
            self.pan_encoder.append(UNetDown(ch, out_ch, normalize=(i > 0), dropout=dropout))
            ch = out_ch

        bottleneck_ch = b * (2 ** (n_levels - 1)) * 2
        self.bottleneck_fuse = nn.Sequential(
            nn.Conv2d(bottleneck_ch, b * (2 ** (n_levels - 1)), 1, 1, 0, bias=False),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.ModuleList()
        for i in range(n_levels - 1, 0, -1):
            in_ch = b * (2 ** i)
            skip_ch = (b * (2 ** (i - 1))) * 2
            out_ch = b * (2 ** (i - 1))
            self.decoder.append(UNetUp(in_ch, skip_ch, out_ch))

        final_ch = b if n_levels > 1 else b * (2 ** (n_levels - 1))
        self.final_up = nn.Sequential(
            nn.Conv2d(final_ch, ms_channels * 4, 3, 1, 1, bias=False),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        ) if n_levels > 1 else nn.Identity()

        self.hf_predictor = nn.Conv2d(ms_channels, ms_channels * 3, 3, 1, 1, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def encode_ms(self, ms_hf: List[torch.Tensor]) -> List[torch.Tensor]:
        x = torch.cat(ms_hf, dim=1)
        feats = []
        for layer in self.ms_encoder:
            x = layer(x)
            feats.append(x)
        return feats

    def encode_pan(self, pan_subbands: List[torch.Tensor]) -> List[torch.Tensor]:
        x = torch.cat(pan_subbands, dim=1)
        feats = []
        for layer in self.pan_encoder:
            x = layer(x)
            feats.append(x)
        return feats

    def decode(
        self,
        bottleneck: torch.Tensor,
        ms_feats: List[torch.Tensor],
        pan_feats: List[torch.Tensor],
    ) -> torch.Tensor:
        x = bottleneck
        for i, layer in enumerate(self.decoder):
            feat_idx = len(ms_feats) - 1 - i
            ms_skip = ms_feats[feat_idx]
            pan_skip = pan_feats[feat_idx]
            skip = torch.cat([ms_skip, pan_skip], dim=1)
            x = layer(x, skip)

        if not isinstance(self.final_up, nn.Identity):
            x = self.final_up(x)

        x = self.hf_predictor(x)
        return x

    def forward(
        self,
        pan: torch.Tensor,
        lrms: torch.Tensor,
    ) -> torch.Tensor:
        LL_ms, LH_ms, HL_ms, HH_ms = self.dwt(lrms)
        LL_pan, LH_pan, HL_pan, HH_pan = self.dwt(pan)

        ms_feats = self.encode_ms([LH_ms, HL_ms, HH_ms])
        pan_feats = self.encode_pan([LL_pan, LH_pan, HL_pan, HH_pan])

        bottleneck = self.bottleneck_fuse(torch.cat([ms_feats[-1], pan_feats[-1]], dim=1))

        hf_pred = self.decode(bottleneck, ms_feats[:-1], pan_feats[:-1])
        LH_star, HL_star, HH_star = hf_pred.chunk(3, dim=1)

        w = torch.sigmoid(self.fusion_w)
        fused_LH = w * LH_star + (1 - w) * LH_ms
        fused_HL = w * HL_star + (1 - w) * HL_ms
        fused_HH = w * HH_star + (1 - w) * HH_ms

        hrms = self.idwt(LL_ms, fused_LH, fused_HL, fused_HH)
        return hrms.clamp(0.0, 1.0)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PSGANDWTModel(nn.Module):
    """
    Complete PSGAN-DWT model wrapper.

    Bundles Generator + Discriminator for training.
    .generator and .discriminator are accessible separately.
    .forward() calls generator only (for inference).
    """
    def __init__(
        self,
        ms_channels: int = 8,
        pan_channels: int = 1,
        base_features: int = 64,
        n_levels: int = 3,
        dropout: float = 0.0,
        n_disc_layers: int = 3,
    ):
        super().__init__()
        self.ms_channels = ms_channels
        self.generator = PSGANDWTGenerator(
            ms_channels=ms_channels,
            pan_channels=pan_channels,
            base_features=base_features,
            n_levels=n_levels,
            dropout=dropout,
        )
        from models.pan_pix2pix.pan_pix2pix import PanPix2PixDiscriminator
        self.discriminator = PanPix2PixDiscriminator(
            ms_channels=ms_channels,
            pan_channels=pan_channels,
            base_features=base_features,
            n_layers=n_disc_layers,
        )

    def forward(
        self,
        pan: torch.Tensor,
        lrms: torch.Tensor,
    ) -> torch.Tensor:
        return self.generator(pan, lrms)

    def count_parameters(self) -> int:
        g = self.generator.count_parameters()
        d = self.discriminator.count_parameters()
        return g + d