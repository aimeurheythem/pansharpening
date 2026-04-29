"""
convnext_pan.py — ConvNeXt-PAN: Pure CNN Pansharpening Network

Architecture based on ConvNeXt (Meta AI, 2022) — the modern successor to ResNet.
No Transformer attention, no self-attention, no cross-attention whatsoever.
This is a dedicated standalone pure-CNN model as required.

Architecture:
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 1 — Dual ConvNeXt Encoder                                        │
│ PAN (B,1,H,W) → [Stem → ConvNeXtBlock×N] → pan_feat (B,D,H,W)       │
│ LRMS (B,C,H,W) → [Stem → ConvNeXtBlock×N] → ms_feat (B,D,H,W)      │
├─────────────────────────────────────────────────────────────────────────┤
│ Stage 2 — Cross-Modal CNN Fusion                                       │
│ Concat(pan_feat, ms_feat) → ConvNeXtBlock×K → fused (B,D,H,W)        │
│ Uses depthwise convolutions for efficient spatial mixing               │
├─────────────────────────────────────────────────────────────────────────┤
│ Stage 3 — Reconstruction Head                                          │
│ ConvNeXtBlock×2 → Conv(D→C) → output + lrms (global spectral residual)│
└─────────────────────────────────────────────────────────────────────────┘

Key ConvNeXt design choices (from Liu et al. 2022):
• 7×7 depthwise conv (large kernel = wide receptive field without attention)
• Inverted bottleneck (expand → DW → contract, like MobileNetV2)
• LayerNorm instead of BatchNorm (better training stability)
• GELU activation (smooth, non-dying gradients)
• Stochastic depth / drop path (regularization for deep networks)

References:
• A ConvNet for the 2020s (Liu et al., CVPR 2022)
• MobileNetV2 (Sandler et al., CVPR 2018) — inverted bottleneck
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# =============================================================================
# CONVNEXT PRIMITIVES
# =============================================================================

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        expansion: int = 4,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-6,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.dw_conv = nn.Conv2d(
            dim, dim, kernel_size, 1, padding, groups=dim, bias=True,
        )
        self.norm = LayerNorm2d(dim)
        mid = int(dim * expansion)
        self.pw_expand = nn.Linear(dim, mid, bias=True)
        self.act = nn.GELU()
        self.pw_contract = nn.Linear(mid, dim, bias=True)

        self.layer_scale = nn.Parameter(
            layer_scale_init * torch.ones(dim)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw_conv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pw_expand(x)
        x = self.act(x)
        x = self.pw_contract(x)
        x = self.layer_scale * x
        x = x.permute(0, 3, 1, 2)
        x = residual + self.drop_path(x)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        output = x.div(keep_prob) * random_tensor
        return output


# =============================================================================
# CONVNEXT ENCODER
# =============================================================================

class ConvNeXtEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        num_blocks: int,
        kernel_size: int = 7,
        expansion: int = 4,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, 4, 4, 0, bias=True),
            LayerNorm2d(embed_dim // 2),
            nn.Conv2d(embed_dim // 2, embed_dim, 1, 1, 0, bias=True),
            LayerNorm2d(embed_dim),
        )

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
        self.blocks = nn.Sequential(*[
            ConvNeXtBlock(
                dim=embed_dim,
                kernel_size=kernel_size,
                expansion=expansion,
                drop_path=dp_rates[i],
            )
            for i in range(num_blocks)
        ])
        self.norm = LayerNorm2d(embed_dim)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, 4, 4, 0, bias=True),
            LayerNorm2d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.upsample(x)
        return x


# =============================================================================
# CROSS-MODAL CNN FUSION
# =============================================================================

class CrossModalCNNFusion(nn.Module):
    def __init__(
        self,
        dim: int,
        num_blocks: int = 4,
        kernel_size: int = 7,
        expansion: int = 4,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.channel_merge = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, 1, 0, bias=True),
            LayerNorm2d(dim),
        )

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
        self.blocks = nn.Sequential(*[
            ConvNeXtBlock(
                dim=dim,
                kernel_size=kernel_size,
                expansion=expansion,
                drop_path=dp_rates[i],
            )
            for i in range(num_blocks)
        ])
        self.norm = LayerNorm2d(dim)

    def forward(
        self,
        pan_feat: torch.Tensor,
        ms_feat: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([pan_feat, ms_feat], dim=1)
        x = self.channel_merge(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x


# =============================================================================
# RECONSTRUCTION HEAD
# =============================================================================

class ReconstructionHead(nn.Module):
    def __init__(
        self,
        dim: int,
        out_channels: int,
        num_blocks: int = 2,
        kernel_size: int = 7,
        expansion: int = 4,
    ):
        super().__init__()
        self.blocks = nn.Sequential(*[
            ConvNeXtBlock(
                dim=dim,
                kernel_size=kernel_size,
                expansion=expansion,
                drop_path=0.0,
            )
            for _ in range(num_blocks)
        ])
        self.norm = LayerNorm2d(dim)
        self.proj = nn.Conv2d(dim, out_channels, 3, 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.norm(x)
        x = self.proj(x)
        return x


# =============================================================================
# CONVNEXT-PAN — MAIN MODEL
# =============================================================================

class ConvNeXtPan(nn.Module):
    r"""
    ConvNeXt-PAN: Pure CNN Pansharpening Network.

    A dedicated standalone CNN model based on ConvNeXt (Liu et al., CVPR 2022).
    No Transformer, no self-attention, no cross-attention — pure convolutions
    with large kernels for wide receptive fields.

    Architecture:
      1. Dual ConvNeXt Encoder (PAN + MS, with 4x downsample + upsample)
      2. Cross-Modal CNN Fusion (concatenate + ConvNeXt blocks)
      3. Reconstruction Head (ConvNeXt blocks + projection to MS channels)
      4. Global spectral residual (output + lrms)

    The 4x downsample in the encoder is critical:
    - Reduces memory by 16x (H/4 × W/4 vs H × W)
    - Increases effective receptive field without stacking many blocks
    - Allows 7×7 depthwise conv to cover 28×28 in original resolution
    - Decoder upsamples back to full resolution before fusion

    Args:
        ms_channels: Number of MS spectral bands (4 for GF2/QB, 8 for WV3)
        pan_channels: Number of PAN channels (always 1)
        embed_dim: Internal feature dimension D
        num_encoder_blocks: ConvNeXt blocks per encoder branch
        num_fusion_blocks: ConvNeXt blocks in cross-modal fusion stage
        num_head_blocks: ConvNeXt blocks in reconstruction head
        kernel_size: Depthwise conv kernel size (7 = ConvNeXt default)
        expansion: Inverted bottleneck expansion ratio
        drop_path_rate: Maximum stochastic depth rate (linearly increasing)
    """

    def __init__(
        self,
        ms_channels: int = 8,
        pan_channels: int = 1,
        embed_dim: int = 64,
        num_encoder_blocks: int = 6,
        num_fusion_blocks: int = 4,
        num_head_blocks: int = 2,
        kernel_size: int = 7,
        expansion: int = 4,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.ms_channels = ms_channels

        # Stage 1: Dual ConvNeXt Encoders
        # Each encoder: stem (4x downsample) → ConvNeXt blocks → upsample (4x)
        self.pan_encoder = ConvNeXtEncoder(
            in_channels=pan_channels,
            embed_dim=embed_dim,
            num_blocks=num_encoder_blocks,
            kernel_size=kernel_size,
            expansion=expansion,
            drop_path_rate=drop_path_rate,
        )
        self.ms_encoder = ConvNeXtEncoder(
            in_channels=ms_channels,
            embed_dim=embed_dim,
            num_blocks=num_encoder_blocks,
            kernel_size=kernel_size,
            expansion=expansion,
            drop_path_rate=drop_path_rate,
        )

        # Stage 2: Cross-Modal CNN Fusion
        self.fusion = CrossModalCNNFusion(
            dim=embed_dim,
            num_blocks=num_fusion_blocks,
            kernel_size=kernel_size,
            expansion=expansion,
            drop_path_rate=drop_path_rate,
        )

        # Stage 3: Reconstruction Head
        self.head = ReconstructionHead(
            dim=embed_dim,
            out_channels=ms_channels,
            num_blocks=num_head_blocks,
            kernel_size=kernel_size,
            expansion=expansion,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (LayerNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head.proj.weight)
        nn.init.zeros_(self.head.proj.bias)

    def forward(
        self,
        pan: torch.Tensor,   # (B, 1, H, W) — high-res panchromatic
        lrms: torch.Tensor,  # (B, C, H, W) — bicubic-upsampled MS to PAN resolution
    ) -> torch.Tensor:       # (B, C, H, W) — pansharpened MS image
        assert pan.shape[-2:] == lrms.shape[-2:], (
            f"PAN and LRMS spatial size mismatch: "
            f"PAN={pan.shape[-2:]}, LRMS={lrms.shape[-2:]}"
        )

        # Stage 1: Dual ConvNeXt Encoders (4x downsample, process, 4x upsample)
        pan_feat = self.pan_encoder(pan)    # (B, D, H, W)
        ms_feat = self.ms_encoder(lrms)     # (B, D, H, W)

        # Stage 2: Cross-Modal CNN Fusion
        fused = self.fusion(pan_feat, ms_feat)  # (B, D, H, W)

        # Stage 3: Reconstruction Head
        residual = self.head(fused)  # (B, C, H, W)

        # Global spectral residual: output + lrms
        output = residual + lrms

        return output.clamp(0.0, 1.0)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"ConvNeXtPan("
            f"ms_ch={self.ms_channels}, "
            f"params={self.count_parameters():,})"
        )
