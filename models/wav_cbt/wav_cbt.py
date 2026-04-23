"""
wav_cbt.py — Wav-CBT: Wavelet Cross-Band Transformer (European J. Remote Sens. 2025)
Based on: https://github.com/VisionVoyagerX/Wav-CBT

Architecture:
  1. Wavelet Decomposition: DWT on both PAN and MS → LL, LH, HL, HH subbands
  2. Cross-Band Transformer (CBT): attends across spectral bands within each subband
  3. Inverse DWT: reconstruct HR-MS from fused subbands
  4. Output: Spectrally-faithful, spatially-enhanced HR-MS image

Key novelty: Models inter-band spectral correlation EXPLICITLY via transformer
attention in the wavelet domain, avoiding spectral distortion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange


# =============================================================================
# DISCRETE WAVELET TRANSFORM (Haar, learnable-free, GPU-compatible)
# =============================================================================

class HaarDWT2D(nn.Module):
    """
    2D Haar Discrete Wavelet Transform (non-learnable, analytical).
    Returns 4 subbands: LL (low-low), LH, HL, HH.
    """
    def __init__(self):
        super().__init__()
        h = torch.tensor([[[1., 1.], [1., 1.]]]) / 2.0   # low-pass
        g = torch.tensor([[[1., -1.], [1., -1.]]]) / 2.0  # high-pass h
        v = torch.tensor([[[1., 1.], [-1., -1.]]]) / 2.0  # high-pass v
        d = torch.tensor([[[1., -1.], [-1., 1.]]]) / 2.0  # diagonal

        self.register_buffer("h_filter", h.unsqueeze(0))  # (1,1,2,2)
        self.register_buffer("g_filter", g.unsqueeze(0))
        self.register_buffer("v_filter", v.unsqueeze(0))
        self.register_buffer("d_filter", d.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:  x: (B, C, H, W)
        Returns: (LL, LH, HL, HH) each (B, C, H/2, W/2)
        """
        B, C, H, W = x.shape
        x_flat = x.reshape(B * C, 1, H, W)

        def _apply(filt):
            return F.conv2d(x_flat, filt.to(x.device), stride=2, padding=0
                           ).reshape(B, C, H // 2, W // 2)

        LL = _apply(self.h_filter)
        LH = _apply(self.g_filter)
        HL = _apply(self.v_filter)
        HH = _apply(self.d_filter)
        return LL, LH, HL, HH


class HaarIWDT2D(nn.Module):
    """2D Haar Inverse Wavelet Transform."""
    def forward(self, LL, LH, HL, HH) -> torch.Tensor:
        B, C, H, W = LL.shape
        out = torch.zeros(B, C, H * 2, W * 2, device=LL.device, dtype=LL.dtype)
        out[:, :, 0::2, 0::2] = (LL + LH + HL + HH)
        out[:, :, 0::2, 1::2] = (LL - LH + HL - HH)
        out[:, :, 1::2, 0::2] = (LL + LH - HL - HH)
        out[:, :, 1::2, 1::2] = (LL - LH - HL + HH)
        return out


# =============================================================================
# CROSS-BAND TRANSFORMER (CBT) — Key Novelty
# =============================================================================

class CrossBandAttention(nn.Module):
    """
    Cross-Band Attention: PAN subband attends to MS subbands.
    
    PAN (1 band) provides spatial detail queries.
    MS  (C bands) provides spectral context keys/values.
    
    This is applied independently for each wavelet subband (LH, HL, HH).
    """
    def __init__(self, in_ch: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (in_ch // num_heads) ** -0.5

        # PAN → queries
        self.q_conv = nn.Conv2d(1,     in_ch, 1)
        # MS  → keys/values
        self.k_conv = nn.Conv2d(in_ch, in_ch, 1)
        self.v_conv = nn.Conv2d(in_ch, in_ch, 1)
        # Output projection
        self.out_conv = nn.Conv2d(in_ch, in_ch, 1)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.GroupNorm(min(num_heads, in_ch), in_ch)

    def forward(
        self,
        pan_sub: torch.Tensor,   # (B, 1, H, W)   PAN subband
        ms_sub:  torch.Tensor,   # (B, C, H, W)   MS subband
    ) -> torch.Tensor:           # (B, C, H, W)
        B, C, H, W = ms_sub.shape

        # Project and reshape to (B*heads, N, d_head)
        Q = rearrange(self.q_conv(pan_sub), "b (h d) x y -> (b h) (x y) d",
                      h=self.num_heads)
        K = rearrange(self.k_conv(ms_sub),  "b (h d) x y -> (b h) (x y) d",
                      h=self.num_heads)
        V = rearrange(self.v_conv(ms_sub),  "b (h d) x y -> (b h) (x y) d",
                      h=self.num_heads)

        # Attention
        attn = torch.bmm(Q.expand_as(K), K.transpose(1, 2)) * self.scale
        attn = self.drop(attn.softmax(dim=-1))
        out  = torch.bmm(attn, V)                           # (B*heads, N, d_head)
        out  = rearrange(out, "(b h) (x y) d -> b (h d) x y",
                         b=B, h=self.num_heads, x=H, y=W)

        out = self.out_conv(out)
        return self.norm(ms_sub + out)                       # Residual + norm


class CBTBlock(nn.Module):
    """
    Cross-Band Transformer Block: attention + channel MLP.
    Applied independently per subband.
    """
    def __init__(self, in_ch: int, num_heads: int = 4, mlp_ratio: float = 2.0,
                 dropout: float = 0.0):
        super().__init__()
        self.cba = CrossBandAttention(in_ch, num_heads, dropout)
        mlp_ch   = int(in_ch * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch, mlp_ch, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(mlp_ch, in_ch, 1),
        )
        self.norm = nn.GroupNorm(min(4, in_ch), in_ch)

    def forward(self, pan_sub: torch.Tensor, ms_sub: torch.Tensor) -> torch.Tensor:
        x = self.cba(pan_sub, ms_sub)
        return self.norm(x + self.mlp(x))


# =============================================================================
# WAV-CBT — MAIN MODEL
# =============================================================================

class WavCBT(nn.Module):
    """
    Wav-CBT: Wavelet Cross-Band Transformer (European J. Remote Sens. 2025).
    
    Args:
        ms_channels:  Number of MS spectral bands
        pan_channels: Number of PAN channels (always 1)
        embed_dim:    Internal feature dimension
        num_heads:    Attention heads in CBT
        num_blocks:   Number of CBT blocks per subband branch
        mlp_ratio:    MLP expansion ratio
        dropout:      Dropout rate
    """
    def __init__(
        self,
        ms_channels:  int   = 4,
        pan_channels: int   = 1,
        embed_dim:    int   = 48,
        num_heads:    int   = 6,
        num_blocks:   int   = 4,
        mlp_ratio:    float = 2.0,
        dropout:      float = 0.0,
    ):
        super().__init__()
        self.ms_channels = ms_channels

        # DWT / IDWT
        self.dwt  = HaarDWT2D()
        self.idwt = HaarIWDT2D()

        # Stem convolutions: map raw subbands to embed_dim
        self.ms_stem  = nn.Conv2d(ms_channels,  embed_dim, 3, 1, 1)
        self.pan_stem = nn.Conv2d(pan_channels, embed_dim, 3, 1, 1)

        # 3 subband branches: LH, HL, HH (LL handled separately)
        def _make_branch():
            return nn.ModuleList([
                CBTBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(num_blocks)
            ])

        self.lh_branch = _make_branch()
        self.hl_branch = _make_branch()
        self.hh_branch = _make_branch()

        # LL branch: simple spectral-preserving conv (no spatial detail needed)
        self.ll_branch = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
        )

        # Subband output projections
        self.out_ll = nn.Conv2d(embed_dim, ms_channels, 1)
        self.out_lh = nn.Conv2d(embed_dim, ms_channels, 1)
        self.out_hl = nn.Conv2d(embed_dim, ms_channels, 1)
        self.out_hh = nn.Conv2d(embed_dim, ms_channels, 1)

        # Final refinement after IDWT
        self.refine = nn.Sequential(
            nn.Conv2d(ms_channels, ms_channels * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(ms_channels * 2, ms_channels, 3, 1, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        pan:  torch.Tensor,   # (B, 1, H, W)
        lrms: torch.Tensor,   # (B, C, H, W)  upsampled to PAN resolution
    ) -> torch.Tensor:
        # ── 1. Wavelet decomposition ──────────────────────────────────────────
        pan_LL, pan_LH, pan_HL, pan_HH = self.dwt(pan)
        ms_LL,  ms_LH,  ms_HL,  ms_HH  = self.dwt(lrms)

        # ── 2. Embed to feature space ─────────────────────────────────────────
        ms_LL_f  = self.ms_stem(ms_LL)
        ms_LH_f  = self.ms_stem(ms_LH)
        ms_HL_f  = self.ms_stem(ms_HL)
        ms_HH_f  = self.ms_stem(ms_HH)

        pan_LL_f = self.pan_stem(pan_LL)
        pan_LH_f = self.pan_stem(pan_LH)
        pan_HL_f = self.pan_stem(pan_HL)
        pan_HH_f = self.pan_stem(pan_HH)

        # ── 3. Process high-freq subbands with CBT ────────────────────────────
        # PAN_pan features act as 1-channel proxy for cross-band attention
        pan_proxy_lh = pan_LH_f[:, :1]   # (B, 1, H, W) — first channel
        pan_proxy_hl = pan_HL_f[:, :1]
        pan_proxy_hh = pan_HH_f[:, :1]

        x_lh = ms_LH_f
        for blk in self.lh_branch:
            x_lh = blk(pan_proxy_lh, x_lh)

        x_hl = ms_HL_f
        for blk in self.hl_branch:
            x_hl = blk(pan_proxy_hl, x_hl)

        x_hh = ms_HH_f
        for blk in self.hh_branch:
            x_hh = blk(pan_proxy_hh, x_hh)

        # LL: concatenate PAN_LL and MS_LL and fuse
        x_ll = self.ll_branch(torch.cat([pan_LL_f, ms_LL_f], dim=1))

        # ── 4. Project back to image space ────────────────────────────────────
        out_LL = self.out_ll(x_ll)
        out_LH = self.out_lh(x_lh)
        out_HL = self.out_hl(x_hl)
        out_HH = self.out_hh(x_hh)

        # ── 5. Inverse DWT ────────────────────────────────────────────────────
        reconstructed = self.idwt(out_LL, out_LH, out_HL, out_HH)

        # ── 6. Refinement + global residual (spectral preservation) ──────────
        output = self.refine(reconstructed) + lrms

        return output.clamp(0.0, 1.0)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
