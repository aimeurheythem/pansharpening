"""
evaluate.py — Benchmark all trained models and produce a comparison table.

Usage:
    python evaluate.py --config configs/panbench.yaml --checkpoint checkpoints/wav_cbt/best.pth
    python evaluate.py --all --dataset wv3 --output results/

Produces:
    - Console table (SAM, ERGAS, Q4, SCC, PSNR, SSIM per model)
    - results/benchmark_wv3.csv
    - results/visual_comparison.png
"""

import argparse
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table

from models.model_factory import get_model, list_models
from data.datasets.panbench import get_panbench_loaders
from utils.metrics import MetricTracker, compute_all_metrics

console = Console()


@torch.no_grad()
def evaluate_model(model: nn.Module, loader, device: torch.device,
                    scale_ratio: int = 4) -> dict:
    """Run inference on test set and return all metrics."""
    model.eval()
    tracker = MetricTracker()

    for batch in loader:
        pan  = batch["pan"].to(device)
        lrms = batch["lrms"].to(device)
        gt   = batch["gt"].to(device)

        pred = model(pan, lrms).float()

        pred_np = pred.cpu().numpy()
        gt_np   = gt.float().cpu().numpy()
        tracker.update_batch(gt_np, pred_np, ratio=scale_ratio)

    return tracker.compute()


def print_results_table(results: dict):
    """Print a formatted Rich table of all model results."""
    table = Table(
        title="[bold]Pansharpening Benchmark Results[/bold]",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Model",  style="bold white", width=20)
    table.add_column("SAM↓",   style="yellow",     justify="right")
    table.add_column("ERGAS↓", style="yellow",     justify="right")
    table.add_column("Q4↑",    style="green",      justify="right")
    table.add_column("SCC↑",   style="green",      justify="right")
    table.add_column("PSNR↑",  style="green",      justify="right")
    table.add_column("SSIM↑",  style="green",      justify="right")

    for model_name, metrics in results.items():
        table.add_row(
            model_name,
            f"{metrics.get('SAM',   0):.4f}°",
            f"{metrics.get('ERGAS', 0):.4f}",
            f"{metrics.get('Q4',    0):.4f}",
            f"{metrics.get('SCC',   0):.4f}",
            f"{metrics.get('PSNR',  0):.2f} dB",
            f"{metrics.get('SSIM',  0):.4f}",
        )

    console.print(table)


def save_csv(results: dict, output_path: Path):
    """Save results to CSV for LaTeX/paper tables."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Model", "SAM", "ERGAS", "Q4", "SCC", "PSNR", "SSIM"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, metrics in results.items():
            writer.writerow({
                "Model": model_name,
                **{k: f"{v:.4f}" for k, v in metrics.items()},
            })
    console.print(f"[green]✓ CSV saved → {output_path}[/green]")


def main():
    parser = argparse.ArgumentParser(description="Evaluate pansharpening models")
    parser.add_argument("--config",     type=str, default="configs/panbench.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Single model checkpoint (.pth)")
    parser.add_argument("--model",      type=str, default=None,
                        help="Model name (overrides config)")
    parser.add_argument("--satellite",  type=str, default="wv3",
                        choices=["wv3", "gf2", "qb"])
    parser.add_argument("--output",     type=str, default="results/")
    parser.add_argument("--device",     type=str, default="cuda")
    args = parser.parse_args()

    base_cfg  = OmegaConf.load("configs/base.yaml")
    model_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, model_cfg)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    console.print(f"\n[bold]Evaluation[/bold] | Satellite: [cyan]{args.satellite}[/cyan] | Device: {device}\n")

    # ── Load test data ─────────────────────────────────────────────────────────
    loaders = get_panbench_loaders(
        h5_train  = cfg.dataset.h5_train,
        h5_val    = cfg.dataset.h5_val,
        h5_test   = cfg.dataset.get("h5_test", cfg.dataset.h5_val),
        satellite = args.satellite,
        batch_size= 1,
        num_workers=cfg.hardware.num_workers,
    )
    test_loader = loaders.get("test", loaders["val"])

    # ── Load model ─────────────────────────────────────────────────────────────
    model_name   = args.model or cfg.model.name
    model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)
    model_kwargs.pop("name")
    model = get_model(model_name, **model_kwargs).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        console.print(f"[cyan]Loaded checkpoint: {args.checkpoint}[/cyan]")
    else:
        console.print("[yellow]Warning: No checkpoint loaded, using random weights.[/yellow]")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    scale_ratio = cfg.dataset.get("scale_ratio", 4)
    metrics = evaluate_model(model, test_loader, device, scale_ratio=scale_ratio)
    results = {model_name: metrics}

    # ── Output ─────────────────────────────────────────────────────────────────
    print_results_table(results)

    out_dir = Path(args.output)
    save_csv(results, out_dir / f"benchmark_{args.satellite}.csv")
    console.print(f"\n[bold green]✓ Evaluation complete![/bold green]")


if __name__ == "__main__":
    main()
