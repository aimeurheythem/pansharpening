"""
pan_pix2pix.py — Pan-Pix2Pix: Conditional GAN for Pansharpening

Based on Pix2Pix (Isola et al., CVPR 2017) — the canonical paired
image-to-image conditional GAN. Adapted for pansharpening:
  Condition: PAN + upsampled LR-MS (concatenated along channel axis)
  Target:    Ground-truth HR-MS
  Generator: UNet (encoder-decoder with skip connections)
  Discriminator: PatchGAN (classifies NxN patches as real/fake)

Architecture:
┌─────────────────────────────────────────────────────────────────────────┐
│ GENERATOR (UNet)                                                        │
│                                                                         │
│ Input: concat(PAN, LRMS) → (B, C+1, H, W)                             │
│                                                                         │
│ Encoder:                                                                │
│   E1: (C+1) → 64   ────────────────────────────────────┐              │
│   E2: 64 → 128     ────────────────────────────┐       │              │
│   E3: 128 → 256    ──────────────────────┐     │       │              │
│   E4: 256 → 512    ────────────────┐     │     │       │              │
│   E5: 256 → 512    ──────────┐     │     │     │       │              │
│   E6: 512 → 512    ────┐    │     │     │     │       │              │
│   E7: 512 → 512    ─┐ │   │     │     │     │       │              │
│   E8: 512 → 512     │ │  │    │     │     │     │       │              │
│                      ▼ │  │    │     │     │     │       │              │
│ Decoder:              │ │  │    │     │     │     │       │              │
│   D1: 512+512→512    ◄─┘ │  │    │     │     │     │       │              │
│   D2: 512+512→512    ◄───┘  │    │     │     │     │       │              │
│   D3: 512+512→512    ◄──────┘    │     │     │     │       │              │
│   D4: 512+512→256    ◄────────────┘     │     │     │       │              │
│   D5: 256+256→128    ◄───────────────────┘     │     │       │              │
│   D6: 128+128→64     ◄─────────────────────────────┘     │       │              │
│   D7: 64+64  → C     ◄───────────────────────────────────────┘       │              │
│   D8: final conv → C  ◄─────────────────────────────────────────────┘              │
│                                                                         │
│ Output: (B, C, H, W) + lrms (global spectral residual)                │
├─────────────────────────────────────────────────────────────────────────┤
│ DISCRIMINATOR (PatchGAN)                                                │
│                                                                         │
│ Input: concat(condition, target_or_fake) → (B, C+1+C, H, W)           │
│                                                                         │
│ C64 → C128 → C256 → C512 → 1×1 Conv → Sigmoid                        │
│ Each layer: Conv(stride=2) → BatchNorm → LeakyReLU(0.2)              │
│ Output: (B, 1, H', W') patch-wise real/fake probabilities             │
│                                                                         │
│ PatchGAN advantage:                                                     │
│ • Focuses on local texture realism (not global structure)              │
│ • Each output pixel sees a 70×70 receptive field in input             │
│ • Fewer parameters than full-image discriminator                       │
│ • Provides per-pixel feedback to generator                             │
└─────────────────────────────────────────────────────────────────────────┘

Training (handled in train_gan.py):
  G_loss = λ_L1 * L1(G(x), y) + λ_ssim * SSIM(G(x), y) + λ_sam * SAM(G(x), y)
           + GAN_loss(D(G(x), x), 1)        ← adversarial (generator tries to fool D)
  D_loss = 0.5 * [GAN_loss(D(y, x), 1)     ← real pairs
                  + GAN_loss(D(G(x), x), 0)] ← fake pairs

References:
• Image-to-Image Translation with Conditional Adversarial Networks
  (Isola et al., CVPR 2017)
• Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial
  Networks (Zhu et al., ICCV 2017) — CycleGAN baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# =============================================================================
# UNET GENERATOR BUILDING BLOCKS
# =============================================================================

class UNetDown(nn.Module):
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
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        dh, dw = skip.shape[2] - x.shape[2], skip.shape[3] - x.shape[3]
        x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return torch.cat([x, skip], dim=1)


# =============================================================================
# UNET GENERATOR
# =============================================================================

class PanPix2PixGenerator(nn.Module):
    r"""
    UNet Generator for pansharpening.

    Takes concatenated PAN + LR-MS as input and produces HR-MS output.
    The UNet architecture preserves spatial detail through skip connections
    while the bottleneck captures global context.

    The input is PAN (1ch) + LRMS (C ch) concatenated along channel axis.
    Total input channels: ms_channels + pan_channels.

    Channel progression follows the standard Pix2Pix pattern:
      Encoder:  b → 2b → 4b → 8b → 8b → ... (capped at 8b)
      Decoder:  mirrors encoder with skip connections (2× channels in)

    Args:
        ms_channels: Number of MS spectral bands
        pan_channels: Number of PAN channels (always 1)
        base_features: Base feature count (64 = standard Pix2Pix)
        n_encoder_layers: Number of encoder downsampling layers (8 = standard Pix2Pix)
            Each layer halves spatial resolution. Total downsample = 2^n_encoder_layers.
            Min input size = 2^n_encoder_layers (e.g. n=8 → 256×256).
        dropout: Dropout rate in bottleneck layers (0.0-0.5)
    """

    def __init__(
        self,
        ms_channels: int = 8,
        pan_channels: int = 1,
        base_features: int = 64,
        n_encoder_layers: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ms_channels = ms_channels
        self.n_encoder_layers = n_encoder_layers
        in_ch = ms_channels + pan_channels
        b = base_features
        max_ch = b * 8

        # Build encoder layers dynamically
        # Channel progression: in_ch → b → 2b → 4b → 8b → 8b → ...
        enc_channels = [b]  # first layer output
        ch = b
        for _ in range(1, n_encoder_layers):
            ch = min(ch * 2, max_ch)
            enc_channels.append(ch)

        self.encoder = nn.ModuleList()
        for i, out_ch in enumerate(enc_channels):
            in_c = in_ch if i == 0 else enc_channels[i - 1]
            # First and last encoder layers: no BatchNorm (standard Pix2Pix)
            norm = (i > 0) and (i < n_encoder_layers - 1)
            self.encoder.append(UNetDown(in_c, out_ch, normalize=norm))

        # Build decoder layers dynamically (n_encoder_layers - 1 upsampling layers)
        # Forward pass: after discarding bottleneck, skips = enc_channels[0..N-2]
        # Popped deepest-first: enc_channels[N-2], enc_channels[N-3], ..., enc_channels[0]
        # So decoder iterates skip indices: N-2, N-3, ..., 0
        n_decoder_layers = n_encoder_layers - 1
        self.decoder = nn.ModuleList()
        prev_out_ch = enc_channels[-1] # bottleneck output channels
        for i in range(n_encoder_layers - 2, -1, -1):
            skip_ch = enc_channels[i]
            out_ch = skip_ch
            # First 3 decoder layers (deepest) get dropout
            layer_idx = n_encoder_layers - 2 - i
            use_dropout = dropout > 0.0 and layer_idx < 3
            self.decoder.append(UNetUp(prev_out_ch, out_ch, dropout=dropout if use_dropout else 0.0))
            prev_out_ch = out_ch + skip_ch # output after cat = input to next layer

        # Final layer: upsample to original resolution + project to MS channels
        self.final = nn.Sequential(
            nn.ConvTranspose2d(prev_out_ch, ms_channels, 4, 2, 1),
            nn.Tanh(),
        )

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

    def forward(
        self,
        pan: torch.Tensor, # (B, 1, H, W)
        lrms: torch.Tensor, # (B, C, H, W)
    ) -> torch.Tensor: # (B, C, H, W)
        # Concatenate condition: PAN + LRMS along channel axis
        x = torch.cat([pan, lrms], dim=1) # (B, C+1, H, W)

        # Encoder — collect skip features
        skips = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)

        # Decoder with skip connections
        # Last encoder output = bottleneck = current x (NOT a skip)
        # Pop and discard it, then use remaining skips deepest-first
        skips.pop()
        for layer in self.decoder:
            skip = skips.pop()
            x = layer(x, skip)

        # Output: tanh produces [-1, 1], rescale to [0, 1]
        out = self.final(x)
        out = (out + 1.0) / 2.0 # [-1,1] → [0,1]

        # Global spectral residual
        out = out + lrms

        return out.clamp(0.0, 1.0)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# PATCHGAN DISCRIMINATOR
# =============================================================================

class PanPix2PixDiscriminator(nn.Module):
    r"""
    PatchGAN Discriminator for pansharpening.

    Classifies overlapping patches as real/fake instead of the full image.
    Each output pixel has a receptive field of approximately 70×70 pixels
    in the input image (for 4 conv layers with stride 2 and 4×4 kernels).

    Input: concat(condition, target_or_fake) → (B, 2*C+1, H, W)
    Output: (B, 1, H', W') — patch-wise real/fake probabilities

    The discriminator sees both the condition (PAN + LRMS) and the
    target (real or generated HR-MS) concatenated. This is the standard
    conditional GAN design: D(x, y) or D(x, G(x)).

    Args:
        ms_channels: Number of MS spectral bands
        pan_channels: Number of PAN channels (always 1)
        base_features: Base feature count (64 = standard Pix2Pix)
        n_layers: Number of intermediate conv layers (3 = standard 70×70 PatchGAN)
    """

    def __init__(
        self,
        ms_channels: int = 8,
        pan_channels: int = 1,
        base_features: int = 64,
        n_layers: int = 3,
    ):
        super().__init__()
        in_ch = (ms_channels * 2) + pan_channels
        b = base_features

        # First layer: no BatchNorm, stride 2
        prev_ch = b
        layers = [
            nn.Conv2d(in_ch, prev_ch, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Intermediate layers: BatchNorm + stride 2 (except last which is stride 1)
        for i in range(1, n_layers):
            curr_ch = b * min(2 ** i, 8)
            layers += [
                nn.Conv2d(prev_ch, curr_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(curr_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            prev_ch = curr_ch

        # Last layer: stride 1
        curr_ch = b * min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(prev_ch, curr_ch, 4, 1, 1, bias=False),
            nn.BatchNorm2d(curr_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(curr_ch, 1, 4, 1, 1),
        ]

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        pan: torch.Tensor,          # (B, 1, H, W) — condition
        lrms: torch.Tensor,         # (B, C, H, W) — condition (part of input pair)
        target_or_fake: torch.Tensor,  # (B, C, H, W) — real or generated HR-MS
    ) -> torch.Tensor:              # (B, 1, H', W') — patch logits
        x = torch.cat([pan, lrms, target_or_fake], dim=1)
        return self.model(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# PAN-PIX2PIX — WRAPPER (registered in model factory)
# =============================================================================

class PanPix2Pix(nn.Module):
    r"""
    Pan-Pix2Pix: Conditional GAN for Pansharpening.

    This wrapper bundles the Generator and Discriminator into a single
    nn.Module for the model factory. During training, train_gan.py
    accesses .generator and .discriminator separately for the alternating
    G/D optimization loop. During inference/evaluation, only the generator
    is used (forward() calls the generator only).

    Args:
        ms_channels: Number of MS spectral bands (4 for GF2/QB, 8 for WV3)
        pan_channels: Number of PAN channels (always 1)
        base_features: Base feature count for G and D (64 = standard)
        n_encoder_layers: Number of encoder downsampling layers (8 = standard)
        dropout: Dropout rate in generator bottleneck
        n_disc_layers: Number of intermediate discriminator layers (3 = 70×70 PatchGAN)
    """

    def __init__(
        self,
        ms_channels: int = 8,
        pan_channels: int = 1,
        base_features: int = 64,
        n_encoder_layers: int = 8,
        dropout: float = 0.0,
        n_disc_layers: int = 3,
    ):
        super().__init__()
        self.ms_channels = ms_channels
        self.generator = PanPix2PixGenerator(
            ms_channels=ms_channels,
            pan_channels=pan_channels,
            base_features=base_features,
            n_encoder_layers=n_encoder_layers,
            dropout=dropout,
        )
        self.discriminator = PanPix2PixDiscriminator(
            ms_channels=ms_channels,
            pan_channels=pan_channels,
            base_features=base_features,
            n_layers=n_disc_layers,
        )

    def forward(
        self,
        pan: torch.Tensor,   # (B, 1, H, W)
        lrms: torch.Tensor,  # (B, C, H, W)
    ) -> torch.Tensor:       # (B, C, H, W) — generator output only
        return self.generator(pan, lrms)

    def count_parameters(self) -> int:
        g = self.generator.count_parameters()
        d = self.discriminator.count_parameters()
        return g + d

    def __repr__(self) -> str:
        g = self.generator.count_parameters()
        d = self.discriminator.count_parameters()
        return (
            f"PanPix2Pix("
            f"ms_ch={self.ms_channels}, "
            f"G_params={g:,}, "
            f"D_params={d:,})"
        )
