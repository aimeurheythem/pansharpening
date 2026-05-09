"""
train.py — Main training entry point for pansharpening

Usage:
    python train.py --config configs/panfusionnet.yaml
    python train.py --config configs/scaleformer.yaml  --wandb
    python train.py --config configs/wav_cbt.yaml      --resume checkpoints/wav_cbt/last.pth

Supports:
    - Mixed precision (FP16) training via torch.cuda.amp
    - WandB + TensorBoard logging
    - Automatic checkpointing (best val + periodic)
    - Gradient accumulation (accum_steps in config)
    - Linear LR warmup + cosine decay (no external scheduler library needed)

FIXES applied vs original:
    FIX 1: validate() now clamps pred/gt to [0,1] before numpy conversion.
            Under FP16, model outputs can have tiny out-of-range values even
            after model-level .clamp() due to float precision — explicit clamp
            here is the safe guard for metric correctness.

    FIX 2: [ChatGPT ERROR 2 assessed as NOT a bug]
            lrms from PanBench HDF5 is already bicubic-upsampled to PAN
            resolution (documented in panbench.py: lrms key = upsampled).
            Both model forward() signatures confirm: lrms: (B,C,H_pan,W_pan).
            No upsampling needed here — added assertion in models instead.

    FIX 3: [ChatGPT ERROR 3 assessed as NOT a bug]
            HybridPanLoss already uses L1+SSIM+SAM with identical weights
            to what ChatGPT suggests. No change needed.

    FIX 4: Gradient accumulation (accum_steps) added. Effective batch size =
            batch_size * accum_steps without extra GPU memory.

    FIX 5: LR warmup added via cosine_warmup scheduler. Uses PyTorch
            LambdaLR — no external libraries required.

    FIX 6: LR now logged to TensorBoard every epoch.

    FIX 7 (ChatGPT missed): Train losses were only logged at val_interval
            (every N epochs), silently dropping all in-between epochs.
            Now logged every epoch independently of validation.

    FIX 8 (ChatGPT missed): WaveletLoss had zero gradients — fixed in losses.py.
"""

import os
import sys
import math
import time
import argparse
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from models.model_factory import get_model
from data.datasets.panbench import get_panbench_loaders
from utils.metrics import MetricTracker
from utils.losses import get_loss
from utils.logger import TrainingLogger

console = Console()


# =============================================================================
# SEEDING
# =============================================================================


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# SCHEDULER: Linear warmup + cosine decay (no external library needed)
# =============================================================================


def get_cosine_warmup_scheduler(
    optimizer,
    warmup_epochs: int,
    total_epochs: int,
    eta_min_ratio: float = 1e-3,
):
    """
    Linear LR warmup for `warmup_epochs`, then cosine annealing to
    lr * eta_min_ratio over the remaining epochs.

    Uses PyTorch's built-in LambdaLR — no `transformers` package needed.
    This is especially important for Transformer models where cold-starting
    at full LR causes unstable early training and divergence.

    Args:
        optimizer:      The optimizer to schedule
        warmup_epochs:  Number of linear warmup epochs
        total_epochs:   Total training epochs
        eta_min_ratio:  Final LR = base_lr * eta_min_ratio
    """

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            # Linear warmup: 1/W, 2/W, ..., W/W
            return float(epoch + 1) / float(max(1, warmup_epochs))
        # Cosine annealing from 1.0 down to eta_min_ratio
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# CHECKPOINT UTILITIES
# =============================================================================


def save_checkpoint(state: dict, path: Path, is_best: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    if is_best:
        best_path = path.parent / "best.pth"
        torch.save(state, best_path)
        console.print(f"  [bold green]✓ Best model saved → {best_path}[/bold green]")


def load_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None):
    console.print(f"  [cyan]Loading checkpoint: {path}[/cyan]")
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_metric = ckpt.get("best_metric", float("inf"))
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return start_epoch, best_metric


# =============================================================================
# VALIDATION
# FIX 1: Clamp pred and gt to [0,1] before numpy conversion.
#         Models already clamp in forward(), but FP16 autocast can produce
#         tiny out-of-range values due to float precision.  Explicit clamp
#         here ensures metrics (SAM, ERGAS, PSNR, SSIM) are always correct.
# =============================================================================


@torch.no_grad()
def validate(model, loader, device, cfg) -> dict:
    model.eval()
    tracker = MetricTracker()

    for batch in loader:
        pan = batch["pan"].to(device)
        lrms = batch["lrms"].to(device)
        gt = batch["gt"].to(device)

        with autocast(enabled=cfg.hardware.fp16):
            pred = model(pan, lrms)

        # FIX 1: Explicit clamp before numpy — guards against FP16 precision drift
        pred = pred.float().clamp(0.0, 1.0)
        gt = gt.float().clamp(0.0, 1.0)

        pred_np = pred.cpu().numpy()
        gt_np = gt.cpu().numpy()
        scale = cfg.dataset.get("scale_ratio", 4)
        tracker.update_batch(gt_np, pred_np, ratio=scale)

    return tracker.compute()


# =============================================================================
# TRAINING LOOP
# FIX 4: Gradient accumulation via accum_steps.
# =============================================================================


def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    scaler,
    device,
    cfg,
    epoch: int,
    writer: Optional[object] = None,
) -> dict:
    model.train()
    loss_components_sum = {}
    n_batches = len(loader)
    accum_steps = cfg.training.get("accum_steps", 1)  # FIX 4: gradient accumulation

    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(loader):
        pan = batch["pan"].to(device)
        lrms = batch["lrms"].to(device)
        gt = batch["gt"].to(device)

        with autocast(enabled=cfg.hardware.fp16):
            pred = model(pan, lrms)
            loss, components = loss_fn(pred, gt)

        # FIX 4: Scale loss by accumulation steps so gradients are consistent
        #         regardless of accum_steps value.
        loss_scaled = loss / accum_steps

        if cfg.hardware.fp16:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        # Optimizer step at the end of each accumulation window
        if (i + 1) % accum_steps == 0 or (i + 1) == n_batches:
            if cfg.hardware.fp16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        for k, v in components.items():
            loss_components_sum[k] = loss_components_sum.get(k, 0.0) + v

    result = {k: v / n_batches for k, v in loss_components_sum.items()}
    return result


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train pansharpening model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument(
        "--device", type=str, default=None, help="Override device (cuda/cpu)"
    )
    args = parser.parse_args()

    # ── Load config ────────────────────────────────────────────────────────────
    base_cfg = OmegaConf.load("configs/base.yaml")
    model_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, model_cfg)

    set_seed(cfg.project.seed)

    # ── Device setup ───────────────────────────────────────────────────────────
    device_str = args.device or cfg.hardware.device
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    console.print(
        f"\n[bold]PanSharpeningPro[/bold] | "
        f"Model: [cyan]{cfg.model.name}[/cyan] | "
        f"Device: [green]{device}[/green]\n"
    )

    # ── Data loaders ───────────────────────────────────────────────────────────
    console.print("[yellow]Loading datasets...[/yellow]")
    dataset_name = cfg.dataset.get("name", "panbench")

    if dataset_name == "panbench":
        loaders = get_panbench_loaders(
            h5_train=cfg.dataset.h5_train,
            h5_val=cfg.dataset.h5_val,
            h5_test=cfg.dataset.get("h5_test", None),
            satellite=cfg.dataset.get("satellites", ["wv3"])[0],
            batch_size=cfg.training.batch_size,
            num_workers=cfg.hardware.num_workers,
        )
    elif dataset_name == "panscale":
        from data.datasets.panscale import get_panscale_loaders

        loaders = get_panscale_loaders(
            root=cfg.dataset.root,
            batch_size=cfg.training.batch_size,
            patch_size=cfg.dataset.get("patch_size", 128),
            num_workers=cfg.hardware.num_workers,
            scale_ratio=cfg.dataset.get("scale_ratio", 4),
        )
    else:
        raise NotImplementedError(
            f"Dataset '{dataset_name}' loader not yet implemented. "
            f"Add it in data/datasets/ and register here."
        )

    accum_steps = cfg.training.get("accum_steps", 1)
    eff_batch = cfg.training.batch_size * accum_steps
    console.print(
        f"  Train: {len(loaders['train'].dataset):,} samples | "
        f"Val: {len(loaders['val'].dataset):,} samples\n"
        f"  Batch size: {cfg.training.batch_size} × {accum_steps} accum = "
        f"[bold]{eff_batch}[/bold] effective"
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)
    model_kwargs.pop("name")
    model = get_model(cfg.model.name, **model_kwargs).to(device)

    # ── Loss ───────────────────────────────────────────────────────────────────
    loss_cfg = OmegaConf.to_container(cfg.get("loss", {}), resolve=True)
    loss_fn = get_loss(
        "hybrid", **{k: v for k, v in loss_cfg.items() if k.endswith("_weight")}
    )

    # ── Optimizer ──────────────────────────────────────────────────────────────
    opt_cfg = cfg.optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg.lr,
        weight_decay=opt_cfg.weight_decay,
        betas=tuple(opt_cfg.betas),
    )

    # ── Scheduler (FIX 5: warmup + cosine) ────────────────────────────────────
    warmup_epochs = cfg.training.get("warmup_epochs", 10)
    scheduler = get_cosine_warmup_scheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=cfg.training.epochs,
        eta_min_ratio=cfg.scheduler.eta_min / opt_cfg.lr,  # ratio form
    )
    console.print(
        f"  Scheduler: {warmup_epochs} epoch warmup → cosine decay "
        f"(eta_min={cfg.scheduler.eta_min:.2e})"
    )

    # ── Mixed precision scaler ─────────────────────────────────────────────────
    scaler = GradScaler(enabled=cfg.hardware.fp16)

    # ── Logging ────────────────────────────────────────────────────────────────
    ckpt_dir = Path(cfg.paths.checkpoints) / cfg.model.name
    log_dir = Path(cfg.paths.logs) / "tensorboard" / cfg.model.name
    writer = SummaryWriter(log_dir=str(log_dir)) if cfg.logging.tensorboard else None

    # ── Resume ─────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_ergas = float("inf")
    patience_counter = 0

    if args.resume:
        start_epoch, best_ergas = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )

    # ── Training ───────────────────────────────────────────────────────────────
    console.print(
        f"\n[bold green]Starting training for {cfg.training.epochs} epochs...[/bold green]\n"
    )

    for epoch in range(start_epoch, cfg.training.epochs):
        t0 = time.time()

        # Train
        train_losses = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            loss_fn,
            scaler,
            device,
            cfg,
            epoch,
            writer,
        )

        # FIX 6 + FIX 7: Log LR and train losses EVERY epoch (not just at val_interval).
        # Original code silently dropped all epochs that weren't validation epochs.
        current_lr = optimizer.param_groups[0]["lr"]
        if writer:
            for k, v in train_losses.items():
                writer.add_scalar(f"train/{k}", v, epoch)
            writer.add_scalar("train/lr", current_lr, epoch)  # FIX 6: LR logging

        # Validate every val_interval epochs
        if (epoch + 1) % cfg.training.val_interval == 0:
            val_metrics = validate(model, loaders["val"], device, cfg)
            current_ergas = val_metrics.get("ERGAS", float("inf"))
            is_best = current_ergas < best_ergas

            if is_best:
                best_ergas = current_ergas
                patience_counter = 0
            else:
                patience_counter += 1

            if writer:
                for k, v in val_metrics.items():
                    writer.add_scalar(f"val/{k}", v, epoch)

            elapsed = time.time() - t0
            console.print(
                f"Epoch [{epoch + 1:4d}/{cfg.training.epochs}] "
                f"Loss={train_losses.get('loss_total', 0):.4f} | "
                f"LR={current_lr:.2e} | "
                f"SAM={val_metrics.get('SAM', 0):.3f}° | "
                f"ERGAS={val_metrics.get('ERGAS', 0):.4f} | "
                f"PSNR={val_metrics.get('PSNR', 0):.2f}dB | "
                f"SSIM={val_metrics.get('SSIM', 0):.4f} | "
                f"[{elapsed:.1f}s]"
                + (" [bold green]← BEST[/bold green]" if is_best else "")
            )

            if (epoch + 1) % cfg.training.save_interval == 0 or is_best:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_metric": best_ergas,
                        "val_metrics": val_metrics,
                        "cfg": OmegaConf.to_container(cfg),
                    },
                    ckpt_dir / f"epoch_{epoch + 1:04d}.pth",
                    is_best=is_best,
                )

            if patience_counter >= cfg.training.early_stopping_patience:
                console.print(f"\n[red]Early stopping at epoch {epoch + 1}[/red]")
                break

        scheduler.step()

    console.print(
        f"\n[bold green]✓ Training complete! Best ERGAS: {best_ergas:.4f}[/bold green]"
    )
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
