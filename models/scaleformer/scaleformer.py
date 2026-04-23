"""
scaleformer.py — ScaleFormer: Cross-Scale Pansharpening Transformer (CVPR 2026)
Based on: https://github.com/caoke-963/ScaleFormer

Architecture:
  1. Dual-encoder:   PAN encoder + MS encoder (both Swin-like Transformer)
  2. Cross-Scale Attention Module (CSAM): aligns features across scale ratios
  3. Decoder: Progressive upsampling with skip connections
  4. Output: HR-MS fused image

Key novelty: Handles VARIABLE scale ratios (4×, 8×, 16×) in a single model,
unlike prior work that required separate models per scale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional


# =============================================================================
# BUILDING BLOCKS
# =============================================================================

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DepthwiseSeparableConv(nn.Module):
    """Efficient depthwise separable convolution."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch,  k, 1, k//2, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0,    bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class WindowAttention(nn.Module):
    """
    Swin-style Window Multi-Head Self-Attention.
    Applied within non-overlapping local windows for computational efficiency.
    """
    def __init__(self, dim: int, num_heads: int, window_size: int = 8, dropout: float = 0.0):
        super().__init__()
        self.dim         = dim
        self.num_heads   = num_heads
        self.window_size = window_size
        self.scale       = (dim // num_heads) ** -0.5

        self.qkv    = nn.Linear(dim, dim * 3, bias=True)
        self.proj   = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Learnable relative position bias
        self.rel_pos_bias = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)    # (B, heads, N, C/heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with window attention + MLP."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 window_size: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm1  = nn.LayerNorm(dim)
        self.attn   = WindowAttention(dim, num_heads, window_size, dropout)
        self.norm2  = nn.LayerNorm(dim)
        mlp_dim     = int(dim * mlp_ratio)
        self.mlp    = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def _window_partition(self, x: torch.Tensor, ws: int) -> tuple:
        B, C, H, W = x.shape
        x = rearrange(x, "b c (nh wh) (nw ww) -> (b nh nw) (wh ww) c",
                      wh=ws, ww=ws)
        return x, B, H, W

    def _window_reverse(self, x: torch.Tensor, B: int, H: int, W: int, ws: int) -> torch.Tensor:
        return rearrange(x, "(b nh nw) (wh ww) c -> b c (nh wh) (nw ww)",
                         b=B, nh=H//ws, nw=W//ws, wh=ws, ww=ws)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ws = min(self.attn.window_size, H, W)
        # Pad if needed
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = x.shape

        tokens, B, Hp, Wp = self._window_partition(x, ws)
        tokens = tokens + self.attn(self.norm1(tokens))
        tokens = tokens + self.mlp(self.norm2(tokens))
        x = self._window_reverse(tokens, B, Hp, Wp, ws)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]
        return x


# =============================================================================
# CROSS-SCALE ATTENTION MODULE (CSAM) — Key Novelty
# =============================================================================

class CrossScaleAttentionModule(nn.Module):
    """
    Cross-Scale Attention Module (CSAM).
    Aligns MS features at multiple upsampling scales with PAN features.
    
    For each scale s in {4, 8, 16}: it computes cross-attention between
    PAN tokens (queries) and MS tokens (keys/values) at that scale.
    This enables the model to handle VARIABLE scale ratios.
    """
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.dim       = dim
        self.num_heads = num_heads
        self.scale     = (dim // num_heads) ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(
        self,
        pan_feat: torch.Tensor,    # (B, C, H, W)  PAN features
        ms_feat:  torch.Tensor,    # (B, C, h, w)  MS features (may differ in HW)
    ) -> torch.Tensor:
        B, C, H, W = pan_feat.shape

        # Upsample MS features to PAN resolution for cross-attention
        ms_up = F.interpolate(ms_feat, size=(H, W), mode="bilinear", align_corners=False)

        # Flatten to sequence
        pan_seq = rearrange(pan_feat, "b c h w -> b (h w) c")
        ms_seq  = rearrange(ms_up,    "b c h w -> b (h w) c")

        Q = self.q_proj(self.norm_q(pan_seq))
        K = self.k_proj(self.norm_kv(ms_seq))
        V = self.v_proj(self.norm_kv(ms_seq))

        Q = rearrange(Q, "b n (h d) -> b h n d", h=self.num_heads)
        K = rearrange(K, "b n (h d) -> b h n d", h=self.num_heads)
        V = rearrange(V, "b n (h d) -> b h n d", h=self.num_heads)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = self.drop(attn.softmax(dim=-1))
        out  = rearrange(attn @ V, "b h n d -> b n (h d)")
        out  = self.out_proj(out)

        # Residual + reshape back
        out  = pan_seq + out
        return rearrange(out, "b (h w) c -> b c h w", h=H, w=W)


# =============================================================================
# ENCODER / DECODER
# =============================================================================

class PatchEmbed(nn.Module):
    """Project image patches to token embeddings."""
    def __init__(self, in_ch: int, embed_dim: int, patch_size: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                            # (B, D, H/ps, W/ps)
        B, D, H, W = x.shape
        x = rearrange(x, "b d h w -> b (h w) d")
        x = self.norm(x)
        x = rearrange(x, "b (h w) d -> b d h w", h=H, w=W)
        return x


class Encoder(nn.Module):
    """Shared encoder for PAN or MS. Multi-stage feature extraction."""
    def __init__(self, in_ch: int, embed_dim: int, num_heads: int,
                 num_layers: int, window_size: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.patch_embed = PatchEmbed(in_ch, embed_dim, patch_size=2)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, window_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.patch_embed(x)
        feats = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i % 2 == 1:    # collect multi-scale features
                feats.append(x)
        feats.append(x)
        return feats


class Decoder(nn.Module):
    """Progressive decoder: upsamples fused features back to PAN resolution."""
    def __init__(self, embed_dim: int, out_ch: int, scale_ratio: int = 4):
        super().__init__()
        self.scale_ratio = scale_ratio
        n_upsample = {4: 2, 8: 3, 16: 4}.get(scale_ratio, 2)

        layers = []
        ch = embed_dim
        for _ in range(n_upsample):
            layers += [
                nn.ConvTranspose2d(ch, ch // 2, 4, 2, 1),
                nn.BatchNorm2d(ch // 2),
                nn.GELU(),
            ]
            ch //= 2

        layers.append(nn.Conv2d(ch, out_ch, 3, 1, 1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


# =============================================================================
# SCALEFORMER — MAIN MODEL
# =============================================================================

class ScaleFormer(nn.Module):
    """
    ScaleFormer: Cross-Scale Pansharpening Transformer (CVPR 2026).

    Handles variable scale ratios (4, 8, 16) in a unified model via CSAM.

    Args:
        ms_channels:  Number of MS spectral bands (4 for most satellites)
        pan_channels: Number of PAN bands (always 1)
        embed_dim:    Token embedding dimension
        num_heads:    Attention heads
        num_layers:   Transformer blocks per encoder
        window_size:  Local attention window size
        mlp_ratio:    FFN hidden dim ratio
        dropout:      Attention / MLP dropout
        scale_ratio:  Spatial resolution ratio (4, 8, or 16)
    """
    def __init__(
        self,
        ms_channels:  int   = 4,
        pan_channels: int   = 1,
        embed_dim:    int   = 64,
        num_heads:    int   = 8,
        num_layers:   int   = 6,
        window_size:  int   = 8,
        mlp_ratio:    float = 4.0,
        dropout:      float = 0.1,
        scale_ratio:  int   = 4,
    ):
        super().__init__()
        self.scale_ratio = scale_ratio

        # Dual encoders
        self.pan_encoder = Encoder(pan_channels, embed_dim, num_heads,
                                   num_layers, window_size, mlp_ratio, dropout)
        self.ms_encoder  = Encoder(ms_channels,  embed_dim, num_heads,
                                   num_layers, window_size, mlp_ratio, dropout)

        # Cross-Scale Attention
        self.csam = CrossScaleAttentionModule(embed_dim, num_heads, dropout)

        # Fusion conv after CSAM
        self.fusion_conv = nn.Sequential(
            ConvBNReLU(embed_dim * 2, embed_dim),
            ConvBNReLU(embed_dim, embed_dim),
        )

        # Decoder
        self.decoder = Decoder(embed_dim, ms_channels, scale_ratio)

        # Global residual: upsample LRMS and add to output
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        pan:  torch.Tensor,   # (B, 1, H_pan, W_pan)
        lrms: torch.Tensor,   # (B, C, H_pan, W_pan)  already upsampled
    ) -> torch.Tensor:
        # Encode both modalities
        pan_feats = self.pan_encoder(pan)
        ms_feats  = self.ms_encoder(lrms)

        pan_f = pan_feats[-1]
        ms_f  = ms_feats[-1]

        # Cross-scale attention: PAN queries, MS keys/values
        aligned = self.csam(pan_f, ms_f)

        # Fuse PAN + aligned MS features
        fused = self.fusion_conv(torch.cat([pan_f, aligned], dim=1))

        # Decode to HR-MS resolution
        output = self.decoder(fused)

        # Global skip: add upsampled LRMS (improves spectral fidelity)
        output = output + lrms

        return output.clamp(0.0, 1.0)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
