"""
logger.py — Unified training logger (TensorBoard + WandB)

Handles:
    - Scalar metrics logging
    - Image grid logging (PAN, LRMS, Prediction, GT side-by-side)
    - Model architecture logging
    - Automatic experiment naming
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import torch
import torchvision.utils as vutils


class TrainingLogger:
    """
    Unified logger that writes to TensorBoard and optionally WandB.

    Args:
        log_dir:     Base log directory
        model_name:  Model name (used for subdirectory)
        use_wandb:   Enable WandB logging
        project:     WandB project name
        config:      Full config dict (logged as hyperparameters)
    """

    def __init__(
        self,
        log_dir: str,
        model_name: str,
        use_wandb: bool = False,
        project: str = "pansharpening_pro",
        config: Optional[dict] = None,
    ):
        self.model_name  = model_name
        self.use_wandb   = use_wandb
        self._step       = 0

        # ── TensorBoard ───────────────────────────────────────────────────────
        tb_dir = Path(log_dir) / "tensorboard" / model_name
        tb_dir.mkdir(parents=True, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(tb_dir))
        except ImportError:
            self.writer = None
            print("[Logger] TensorBoard not available")

        # ── WandB ─────────────────────────────────────────────────────────────
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=project,
                    name=f"{model_name}_{int(time.time())}",
                    config=config or {},
                    dir=str(Path(log_dir) / "wandb"),
                )
            except Exception as e:
                print(f"[Logger] WandB init failed: {e}")

    def log_scalars(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """Log scalar metrics to TensorBoard and WandB."""
        tag_prefix = f"{prefix}/" if prefix else ""
        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(f"{tag_prefix}{k}", v, step)
        if self.wandb_run:
            import wandb
            self.wandb_run.log(
                {f"{tag_prefix}{k}": v for k, v in metrics.items()},
                step=step
            )

    def log_images(
        self,
        pan:  torch.Tensor,    # (B, 1, H, W)
        lrms: torch.Tensor,    # (B, C, H, W)
        pred: torch.Tensor,    # (B, C, H, W)
        gt:   torch.Tensor,    # (B, C, H, W)
        step: int,
        n_samples: int = 4,
        rgb_bands: tuple = (0, 1, 2),
    ):
        """
        Log a grid of images: PAN | LRMS | Prediction | GT
        Uses the first 3 spectral bands as RGB for display.
        """
        if self.writer is None:
            return

        B = min(n_samples, pan.shape[0])

        def _to_rgb(x: torch.Tensor) -> torch.Tensor:
            """Extract RGB and normalize to [0,1]."""
            if x.shape[1] == 1:  # PAN
                return x[:B].expand(-1, 3, -1, -1).clamp(0, 1)
            bands = list(rgb_bands[:min(3, x.shape[1])])
            rgb = x[:B, bands].clamp(0, 1)
            if rgb.shape[1] < 3:
                rgb = rgb.expand(-1, 3, -1, -1)
            return rgb

        import torch.nn.functional as F
        h, w = gt.shape[-2], gt.shape[-1]

        pan_rgb  = F.interpolate(_to_rgb(pan),  size=(h, w), mode="bilinear", align_corners=False)
        lrms_rgb = _to_rgb(lrms)
        pred_rgb = _to_rgb(pred)
        gt_rgb   = _to_rgb(gt)

        # Concatenate horizontally: PAN | LRMS | PRED | GT
        grid_row = torch.cat([pan_rgb, lrms_rgb, pred_rgb, gt_rgb], dim=-1)
        grid = vutils.make_grid(grid_row, nrow=1, normalize=False, padding=2)
        self.writer.add_image("comparison/PAN-LRMS-Pred-GT", grid, step)

        # Difference map (prediction error)
        diff = (pred[:B] - gt[:B]).abs().mean(dim=1, keepdim=True)
        diff_grid = vutils.make_grid(diff.expand(-1, 3, -1, -1), nrow=B, normalize=True)
        self.writer.add_image("comparison/error_map", diff_grid, step)

    def log_histogram(self, name: str, values: torch.Tensor, step: int):
        """Log weight/gradient histograms."""
        if self.writer:
            self.writer.add_histogram(name, values, step)

    def log_model_graph(self, model: torch.nn.Module, sample_input: tuple):
        """Log model architecture graph."""
        if self.writer:
            try:
                self.writer.add_graph(model, sample_input)
            except Exception:
                pass  # Graph logging can fail for complex models

    def log_lr(self, lr: float, step: int):
        """Log learning rate."""
        self.log_scalars({"learning_rate": lr}, step, prefix="train")

    def close(self):
        if self.writer:
            self.writer.close()
        if self.wandb_run:
            self.wandb_run.finish()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
