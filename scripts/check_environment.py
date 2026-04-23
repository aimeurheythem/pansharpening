#!/usr/bin/env python3
"""
check_environment.py — Verify all dependencies are installed correctly.
Run this BEFORE training to catch issues early.

Usage:
    python scripts/check_environment.py
"""

import sys
from rich.console import Console
from rich.table import Table

console = Console()

CHECKS = [
    # (display_name, import_expr, min_version_optional)
    ("Python",         lambda: sys.version.split()[0],                     "3.9"),
    ("PyTorch",        lambda: __import__("torch").__version__,             "2.0"),
    ("CUDA Available", lambda: str(__import__("torch").cuda.is_available()), None),
    ("GPU Name",       lambda: __import__("torch").cuda.get_device_name(0)
                               if __import__("torch").cuda.is_available()
                               else "No GPU",                              None),
    ("torchvision",    lambda: __import__("torchvision").__version__,       "0.16"),
    ("numpy",          lambda: __import__("numpy").__version__,             "1.24"),
    ("scipy",          lambda: __import__("scipy").__version__,             "1.10"),
    ("h5py",           lambda: __import__("h5py").__version__,              "3.0"),
    ("rasterio",       lambda: __import__("rasterio").__version__,          "1.3"),
    ("opencv",         lambda: __import__("cv2").__version__,               "4.0"),
    ("scikit-image",   lambda: __import__("skimage").__version__,           "0.20"),
    ("einops",         lambda: __import__("einops").__version__,            "0.6"),
    ("timm",           lambda: __import__("timm").__version__,              "0.9"),
    ("PyWavelets",     lambda: __import__("pywt").__version__,              "1.4"),
    ("diffusers",      lambda: __import__("diffusers").__version__,         "0.20"),
    ("accelerate",     lambda: __import__("accelerate").__version__,        "0.20"),
    ("omegaconf",      lambda: __import__("omegaconf").__version__,         "2.3"),
    ("tensorboard",    lambda: __import__("tensorboard").__version__,       "2.10"),
    ("wandb",          lambda: __import__("wandb").__version__,             "0.15"),
    ("matplotlib",     lambda: __import__("matplotlib").__version__,        "3.7"),
    ("tqdm",           lambda: __import__("tqdm").__version__,              "4.60"),
    ("gdown",          lambda: __import__("gdown").__version__,             "4.0"),
    ("rich",           lambda: __import__("rich").__version__,              "13.0"),
]


def main():
    console.print("\n[bold cyan]╔══════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║   PanSharpeningPro — Env Check       ║[/bold cyan]")
    console.print("[bold cyan]╚══════════════════════════════════════╝[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold white")
    table.add_column("Package",  style="bold", width=20)
    table.add_column("Status",   width=6)
    table.add_column("Version",  width=15)
    table.add_column("Min Req",  width=10)

    passed = 0
    failed = 0

    for name, fn, min_ver in CHECKS:
        try:
            version = fn()
            status = "[green]  ✅[/green]"
            passed += 1
        except Exception as e:
            version = f"ERROR: {str(e)[:30]}"
            status = "[red]  ❌[/red]"
            failed += 1

        table.add_row(name, status, version, min_ver or "—")

    console.print(table)

    # GPU Memory
    try:
        import torch
        if torch.cuda.is_available():
            total_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
            alloc_mem  = torch.cuda.memory_allocated(0) / 1e9
            console.print(
                f"\n[bold]GPU Memory:[/bold] {total_mem:.1f} GB total | "
                f"{alloc_mem:.1f} GB allocated"
            )
            if total_mem < 8:
                console.print("[yellow]⚠ Warning: < 8GB GPU. Use batch_size=8 and fp16=true[/yellow]")
    except Exception:
        pass

    # H5 dataset check
    console.print("\n[bold]Dataset files:[/bold]")
    h5_files = [
        "data/h5/train_wv3.h5",
        "data/h5/valid_wv3.h5",
        "data/h5/train_gf2.h5",
        "data/h5/train_qb.h5",
    ]
    for f in h5_files:
        from pathlib import Path
        if Path(f).exists():
            size = Path(f).stat().st_size / 1e6
            console.print(f"  [green]✅[/green] {f} ({size:.1f} MB)")
        else:
            console.print(f"  [yellow]⚠[/yellow]  {f} — [yellow]NOT FOUND[/yellow] "
                          f"(download from PanBench Google Drive)")

    # Summary
    console.print(f"\n{'─'*40}")
    if failed == 0:
        console.print(f"[bold green]✓ All {passed} checks passed! Ready to train.[/bold green]")
    else:
        console.print(
            f"[bold yellow]{passed} passed[/bold yellow] | "
            f"[bold red]{failed} failed[/bold red]\n"
            f"Fix failed packages with: pip install -r requirements.txt"
        )
    console.print()


if __name__ == "__main__":
    main()
