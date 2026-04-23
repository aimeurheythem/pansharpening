#!/usr/bin/env python3
"""
download_datasets.py — Download all datasets from Professor Guellil's benchmark suite.

Usage:
    python scripts/download_datasets.py --dataset panbench
    python scripts/download_datasets.py --dataset all
    python scripts/download_datasets.py --list
"""

import argparse
import subprocess
import sys
from pathlib import Path
from rich.console import Console

console = Console()

DATASETS = {
    "panbench": {
        "description": "PanBench — Multi-satellite benchmark (MS:256×256, PAN:1024×1024)",
        "paper":       "Remote Sensing, 2024",
        "method":      "gdrive",
        "url":         "1fjwvRrCmExk02c5sxGXMoSvGdGL0FbYR",
        "dest":        "data/raw/PanBench/PanBench.zip",
        "note":        "Contains WV3, GF2, QB splits in HDF5 format",
    },
    "pansscale": {
        "description": "PanScale — Cross-scale multi-scene dataset",
        "paper":       "CVPR 2026",
        "method":      "github",
        "url":         "https://github.com/caoke-963/ScaleFormer",
        "dest":        "data/raw/PanScale/",
        "note":        "Clone the repo; dataset instructions in README",
    },
    "sentinel2_toolbox": {
        "description": "Sentinel-2 SR Toolbox — 10m/20m/60m bands",
        "paper":       "MDPI Remote Sens. 2025",
        "method":      "github",
        "url":         "https://github.com/matciotola/Sentinel2-SR-Toolbox",
        "dest":        "data/raw/Sentinel2/",
        "note":        "Clone repo; download Sentinel-2 data via ESA Copernicus Hub",
    },
    "sev2mod": {
        "description": "Sev2Mod — Geostationary + polar satellite dataset",
        "paper":       "European J. Remote Sens. 2025",
        "method":      "github",
        "url":         "https://github.com/VisionVoyagerX/Wav-CBT",
        "dest":        "data/raw/Sev2Mod/",
        "note":        "Dataset download instructions in repo README",
    },
    "lacas2k": {
        "description": "LACAS2K — Aerial & satellite VHR (~50cm)",
        "paper":       "IEEE DataPort 2025, DOI: 10.21227/6w87-gb75",
        "method":      "ieee",
        "url":         "https://ieee-dataport.org/documents/lacas2k-large-scale-aerial-and-satellite-image-dataset-image-super-resolution-vhr-remote",
        "dest":        "data/raw/LACAS2K/",
        "note":        "Requires free IEEE DataPort account to download",
    },
    "pantcr_gf2": {
        "description": "PanTCR-GF2 — GaoFen-2 cloudy images",
        "paper":       "AAAI 2026",
        "method":      "arxiv",
        "url":         "Contact authors via arXiv",
        "dest":        "data/raw/PanTCR/",
        "note":        "Email authors: code/data available upon request",
    },
}


def list_datasets():
    """Print all available datasets."""
    from rich.table import Table
    table = Table(title="[bold]Available Datasets[/bold]", show_header=True,
                  header_style="bold cyan")
    table.add_column("Name",        width=20)
    table.add_column("Description", width=45)
    table.add_column("Paper",       width=20)
    table.add_column("Method",      width=12)
    table.add_column("Note",        width=35)

    for name, info in DATASETS.items():
        method_style = {
            "gdrive": "[green]gdrive[/green]",
            "github": "[cyan]github[/cyan]",
            "ieee":   "[yellow]ieee[/yellow]",
            "arxiv":  "[red]contact[/red]",
        }.get(info["method"], info["method"])
        table.add_row(name, info["description"], info["paper"],
                      method_style, info["note"])

    console.print(table)


def download_panbench(dest: str):
    """Download PanBench from Google Drive using gdown."""
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    console.print("[cyan]Downloading PanBench from Google Drive...[/cyan]")
    console.print("[yellow]This file is ~3-5GB, may take several minutes.[/yellow]")

    try:
        import gdown
        gdown.download(
            id=DATASETS["panbench"]["url"],
            output=str(dest_path),
            quiet=False
        )
        console.print(f"[green]✓ Downloaded to {dest_path}[/green]")

        # Extract
        console.print("[cyan]Extracting...[/cyan]")
        import zipfile
        with zipfile.ZipFile(dest_path, "r") as z:
            z.extractall(dest_path.parent)
        console.print(f"[green]✓ Extracted to {dest_path.parent}[/green]")

    except ImportError:
        console.print("[red]gdown not installed. Run: pip install gdown[/red]")
    except Exception as e:
        console.print(f"[red]Download failed: {e}[/red]")
        console.print(
            "[yellow]Manual download:\n"
            "  https://drive.google.com/file/d/1fjwvRrCmExk02c5sxGXMoSvGdGL0FbYR/view[/yellow]"
        )


def clone_github_repo(url: str, dest: str):
    """Clone a GitHub repo."""
    dest_path = Path(dest)
    if dest_path.exists():
        console.print(f"[yellow]Already exists: {dest_path} — skipping clone[/yellow]")
        return
    dest_path.mkdir(parents=True, exist_ok=True)
    console.print(f"[cyan]Cloning {url}...[/cyan]")
    result = subprocess.run(["git", "clone", url, str(dest_path)], capture_output=True, text=True)
    if result.returncode == 0:
        console.print(f"[green]✓ Cloned to {dest_path}[/green]")
    else:
        console.print(f"[red]Clone failed: {result.stderr}[/red]")


def main():
    parser = argparse.ArgumentParser(description="Download pansharpening datasets")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name or 'all'")
    parser.add_argument("--list",    action="store_true",
                        help="List all datasets and their status")
    args = parser.parse_args()

    if args.list or args.dataset is None:
        list_datasets()
        return

    datasets_to_download = (
        list(DATASETS.keys()) if args.dataset == "all"
        else [args.dataset]
    )

    for name in datasets_to_download:
        if name not in DATASETS:
            console.print(f"[red]Unknown dataset '{name}'. Use --list to see options.[/red]")
            continue

        info = DATASETS[name]
        console.print(f"\n[bold]Processing: {name}[/bold] — {info['description']}")

        if info["method"] == "gdrive":
            download_panbench(info["dest"])
        elif info["method"] == "github":
            clone_github_repo(info["url"], info["dest"])
        elif info["method"] in ("ieee", "arxiv"):
            console.print(f"[yellow]⚠ Manual download required:[/yellow]")
            console.print(f"  URL:  {info['url']}")
            console.print(f"  Dest: {info['dest']}")
            console.print(f"  Note: {info['note']}")

    console.print("\n[bold green]✓ Done! Run python scripts/check_environment.py to verify.[/bold green]")


if __name__ == "__main__":
    main()
