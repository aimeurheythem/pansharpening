"""
train_gan.py — GAN training entry point for Pan-Pix2Pix

Usage:
  python train_gan.py --config configs/pan_pix2pix.yaml
  python train_gan.py --config configs/pan_pix2pix.yaml --wandb
  python train_gan.py --config configs/pan_pix2pix.yaml --resume checkpoints/pan_pix2pix/last.pth

Supports:
  - Alternating Generator/Discriminator training (standard GAN)
  - Mixed precision (FP16) via torch.cuda.amp
  - WandB + TensorBoard logging
  - Automatic checkpointing (best val + periodic)
  - Gradient accumulation
  - Linear LR warmup + cosine decay
  - LSGAN loss (default) or vanilla BCE GAN loss

Training loop (per iteration):
  1. Generator forward: fake = G(pan, lrms)
  2. Discriminator on real: D_real = D(pan, lrms, gt)
  3. Discriminator on fake: D_fake = D(pan, lrms, fake.detach())
  4. Update D: D_loss = 0.5 * [LSGAN(D_real, 1) + LSGAN(D_fake, 0)]
  5. Generator adversarial: D_fake_for_G = D(pan, lrms, fake)
  6. Update G: G_loss = λ_L1*L1 + λ_SSIM*SSIM + λ_SAM*SAM + λ_GAN*LSGAN(D_fake_for_G, 1)

Note: ConvNeXt-PAN uses train.py (standard supervised), NOT this script.
      This script is ONLY for GAN-based models (pan_pix2pix).
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
from models.pan_pix2pix.pan_pix2pix import PanPix2Pix
from data.datasets.panbench import get_panbench_loaders
from utils.metrics import MetricTracker
from utils.losses import Pix2PixLoss

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
# SCHEDULER
# =============================================================================

def get_cosine_warmup_scheduler(
    optimizer,
    warmup_epochs: int,
    total_epochs: int,
    eta_min_ratio: float = 1e-3,
):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
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


def load_checkpoint(path: str, model, g_optimizer=None, d_optimizer=None,
                    g_scheduler=None, d_scheduler=None):
    console.print(f"  [cyan]Loading checkpoint: {path}[/cyan]")
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_metric = ckpt.get("best_metric", float("inf"))
    if g_optimizer and "g_optimizer" in ckpt:
        g_optimizer.load_state_dict(ckpt["g_optimizer"])
    if d_optimizer and "d_optimizer" in ckpt:
        d_optimizer.load_state_dict(ckpt["d_optimizer"])
    if g_scheduler and "g_scheduler" in ckpt:
        g_scheduler.load_state_dict(ckpt["g_scheduler"])
    if d_scheduler and "d_scheduler" in ckpt:
        d_scheduler.load_state_dict(ckpt["d_scheduler"])
    return start_epoch, best_metric


# =============================================================================
# VALIDATION
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

        pred = pred.float().clamp(0.0, 1.0)
        gt = gt.float().clamp(0.0, 1.0)

        pred_np = pred.cpu().numpy()
        gt_np = gt.cpu().numpy()
        scale = cfg.dataset.get("scale_ratio", 4)
        tracker.update_batch(gt_np, pred_np, ratio=scale)

    return tracker.compute()


# =============================================================================
# GAN TRAINING LOOP
# =============================================================================

def train_one_epoch(
    model: PanPix2Pix,
    loader,
    g_optimizer,
    d_optimizer,
    loss_fn: Pix2PixLoss,
    g_scaler: GradScaler,
    d_scaler: GradScaler,
    device,
    cfg,
    epoch: int,
    writer: Optional[object] = None,
) -> dict:
    model.train()
    loss_sum = {}
    n_batches = len(loader)
    accum_steps = cfg.training.get("accum_steps", 1)
    G = model.generator
    D = model.discriminator

    g_optimizer.zero_grad(set_to_none=True)
    d_optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(loader):
        pan = batch["pan"].to(device)
        lrms = batch["lrms"].to(device)
        gt = batch["gt"].to(device)

        with autocast(enabled=cfg.hardware.fp16):
            # ── 1. Generator forward ──────────────────────────────────────
            fake = G(pan, lrms)

            # ── 2. Discriminator on real ──────────────────────────────────
            D_real = D(pan, lrms, gt)

            # ── 3. Discriminator on fake (detach to prevent G gradient) ───
            D_fake = D(pan, lrms, fake.detach())

            # ── 4. Discriminator loss ─────────────────────────────────────
            d_loss, d_components = loss_fn.discriminator_loss(D_real, D_fake)

        d_loss_scaled = d_loss / accum_steps
        if cfg.hardware.fp16:
            d_scaler.scale(d_loss_scaled).backward()
        else:
            d_loss_scaled.backward()

        # ── 5. Discriminator step (at accumulation boundary) ──────────────
        if (i + 1) % accum_steps == 0 or (i + 1) == n_batches:
            if cfg.hardware.fp16:
                d_scaler.unscale_(d_optimizer)
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
            if cfg.hardware.fp16:
                d_scaler.step(d_optimizer)
                d_scaler.update()
            else:
                d_optimizer.step()
            d_optimizer.zero_grad(set_to_none=True)

        # ── 6. Generator step ─────────────────────────────────────────────
        with autocast(enabled=cfg.hardware.fp16):
            D_fake_for_G = D(pan, lrms, fake)
            g_loss, g_components = loss_fn.generator_loss(fake, gt, D_fake_for_G)

        g_loss_scaled = g_loss / accum_steps
        if cfg.hardware.fp16:
            g_scaler.scale(g_loss_scaled).backward()
        else:
            g_loss_scaled.backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == n_batches:
            if cfg.hardware.fp16:
                g_scaler.unscale_(g_optimizer)
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            if cfg.hardware.fp16:
                g_scaler.step(g_optimizer)
                g_scaler.update()
            else:
                g_optimizer.step()
            g_optimizer.zero_grad(set_to_none=True)

        # ── Accumulate losses ─────────────────────────────────────────────
        for k, v in d_components.items():
            loss_sum[k] = loss_sum.get(k, 0.0) + v
        for k, v in g_components.items():
            loss_sum[k] = loss_sum.get(k, 0.0) + v

    result = {k: v / n_batches for k, v in loss_sum.items()}
    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train GAN pansharpening model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--device", type=str, default=None, help="Override device")
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
        f"\n[bold]PanSharpeningPro — GAN Training[/bold] | "
        f"Model: [cyan]{cfg.model.name}[/cyan] | "
        f"Device: [green]{device}[/green]\n"
    )

    # ── Data loaders ───────────────────────────────────────────────────────────
    console.print("[yellow]Loading datasets...[/yellow]")
    dataset_name = cfg.dataset.get("name", "panbench")

    if dataset_name == "panbench":
        loaders = get_panbench_loaders(
            h5_train = cfg.dataset.h5_train,
            h5_val = cfg.dataset.h5_val,
            h5_test = cfg.dataset.get("h5_test", None),
            satellite = cfg.dataset.get("satellites", ["wv3"])[0],
            batch_size = cfg.training.batch_size,
            num_workers= cfg.hardware.num_workers,
        )
    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' not supported for GAN training.")

    accum_steps = cfg.training.get("accum_steps", 1)
    eff_batch = cfg.training.batch_size * accum_steps
    console.print(
        f" Train: {len(loaders['train'].dataset):,} samples | "
        f"Val: {len(loaders['val'].dataset):,} samples\n"
        f" Batch size: {cfg.training.batch_size} × {accum_steps} accum = "
        f"[bold]{eff_batch}[/bold] effective"
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)
    model_kwargs.pop("name")
    model = get_model(cfg.model.name, **model_kwargs).to(device)

    assert isinstance(model, PanPix2Pix), (
        f"train_gan.py requires a PanPix2Pix model, got {type(model).__name__}. "
        f"Use train.py for non-GAN models."
    )

    G = model.generator
    D = model.discriminator
    console.print(
        f"  Generator params: [green]{G.count_parameters():,}[/green] | "
        f"Discriminator params: [green]{D.count_parameters():,}[/green]"
    )

    # ── Loss ───────────────────────────────────────────────────────────────────
    loss_cfg = OmegaConf.to_container(cfg.get("loss", {}), resolve=True)
    loss_fn = Pix2PixLoss(
        l1_w=loss_cfg.get("l1_weight", 100.0),
        ssim_w=loss_cfg.get("ssim_weight", 10.0),
        sam_w=loss_cfg.get("sam_weight", 5.0),
        gan_w=loss_cfg.get("gan_weight", 1.0),
        gan_mode="lsgan",
    )

    # ── Optimizers (separate for G and D) ──────────────────────────────────────
    opt_cfg = cfg.optimizer
    g_lr = loss_cfg.get("g_lr", opt_cfg.lr)
    d_lr = loss_cfg.get("d_lr", opt_cfg.lr * 2.0)

    g_optimizer = torch.optim.AdamW(
        G.parameters(),
        lr=g_lr,
        weight_decay=opt_cfg.weight_decay,
        betas=tuple(opt_cfg.betas),
    )
    d_optimizer = torch.optim.AdamW(
        D.parameters(),
        lr=d_lr,
        weight_decay=opt_cfg.weight_decay,
        betas=tuple(opt_cfg.betas),
    )
    console.print(
        f"  G_lr: {g_lr:.2e} | D_lr: {d_lr:.2e} | "
        f"betas: {tuple(opt_cfg.betas)}"
    )

    # ── Schedulers ─────────────────────────────────────────────────────────────
    warmup_epochs = cfg.training.get("warmup_epochs", 10)
    g_scheduler = get_cosine_warmup_scheduler(
        g_optimizer, warmup_epochs, cfg.training.epochs,
        eta_min_ratio=cfg.scheduler.eta_min / g_lr,
    )
    d_scheduler = get_cosine_warmup_scheduler(
        d_optimizer, warmup_epochs, cfg.training.epochs,
        eta_min_ratio=cfg.scheduler.eta_min / d_lr,
    )

    # ── Mixed precision scalers (separate for G and D) ─────────────────────────
    g_scaler = GradScaler(enabled=cfg.hardware.fp16)
    d_scaler = GradScaler(enabled=cfg.hardware.fp16)

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
            args.resume, model, g_optimizer, d_optimizer, g_scheduler, d_scheduler
        )

    # ── Training ───────────────────────────────────────────────────────────────
    console.print(
        f"\n[bold green]Starting GAN training for {cfg.training.epochs} epochs...[/bold green]\n"
    )

    for epoch in range(start_epoch, cfg.training.epochs):
        t0 = time.time()

        train_losses = train_one_epoch(
            model, loaders["train"], g_optimizer, d_optimizer,
            loss_fn, g_scaler, d_scaler, device, cfg, epoch, writer
        )

        current_g_lr = g_optimizer.param_groups[0]["lr"]
        current_d_lr = d_optimizer.param_groups[0]["lr"]

        if writer:
            for k, v in train_losses.items():
                writer.add_scalar(f"train/{k}", v, epoch)
            writer.add_scalar("train/g_lr", current_g_lr, epoch)
            writer.add_scalar("train/d_lr", current_d_lr, epoch)

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
                f"Epoch [{epoch+1:4d}/{cfg.training.epochs}] "
                f"G_loss={train_losses.get('G_loss_total', 0):.4f} | "
                f"D_loss={train_losses.get('D_loss_total', 0):.4f} | "
                f"G_lr={current_g_lr:.2e} | "
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
                        "g_optimizer": g_optimizer.state_dict(),
                        "d_optimizer": d_optimizer.state_dict(),
                        "g_scheduler": g_scheduler.state_dict(),
                        "d_scheduler": d_scheduler.state_dict(),
                        "best_metric": best_ergas,
                        "val_metrics": val_metrics,
                        "cfg": OmegaConf.to_container(cfg),
                    },
                    ckpt_dir / f"epoch_{epoch+1:04d}.pth",
                    is_best=is_best,
                )

            if patience_counter >= cfg.training.early_stopping_patience:
                console.print(f"\n[red]Early stopping at epoch {epoch+1}[/red]")
                break

        g_scheduler.step()
        d_scheduler.step()

    console.print(
        f"\n[bold green]✓ GAN Training complete! Best ERGAS: {best_ergas:.4f}[/bold green]"
    )
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
