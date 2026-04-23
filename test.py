"""
test.py — Run inference with a trained model on a single image or full test set.

Usage:
    # Single image inference
    python test.py --config configs/scaleformer.yaml \
                    --checkpoint checkpoints/scaleformer/best.pth \
                    --pan path/to/pan.tif \
                    --ms  path/to/ms.tif  \
                    --output outputs/result.tif

    # Full test set evaluation + save outputs
    python test.py --config configs/panbench.yaml \
                    --checkpoint checkpoints/wav_cbt/best.pth \
                    --save_images
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from rich.console import Console

from models.model_factory import get_model
from utils.metrics import compute_all_metrics

console = Console()


def load_geotiff(path: str) -> tuple[np.ndarray, object]:
    """Load a GeoTIFF. Returns (array CHW, rasterio profile)."""
    import rasterio
    with rasterio.open(path) as src:
        data    = src.read().astype(np.float32)
        profile = src.profile
    return data, profile


def save_geotiff(array: np.ndarray, path: str, profile: object):
    """Save a CHW numpy array as GeoTIFF preserving CRS/transform."""
    import rasterio
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    profile.update(count=array.shape[0], dtype="float32")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array)


@torch.no_grad()
def infer_single(
    model: torch.nn.Module,
    pan:   np.ndarray,       # (1, H, W) float32
    lrms:  np.ndarray,       # (C, H, W) float32 — already upsampled to PAN resolution
    device: torch.device,
    max_val: float = 2047.0,
) -> np.ndarray:
    """
    Run model on a single full image (no tiling).
    For very large images, use infer_tiled() instead.

    Returns: fused image (C, H, W) float32, same scale as input
    """
    model.eval()
    norm = max_val

    pan_t  = torch.from_numpy(pan  / norm).unsqueeze(0).to(device)   # (1,1,H,W)
    lrms_t = torch.from_numpy(lrms / norm).unsqueeze(0).to(device)   # (1,C,H,W)

    pred = model(pan_t, lrms_t)                                       # (1,C,H,W)
    fused = pred.squeeze(0).cpu().numpy() * norm                      # (C,H,W)
    return fused.astype(np.float32)


@torch.no_grad()
def infer_tiled(
    model:     torch.nn.Module,
    pan:       np.ndarray,
    lrms:      np.ndarray,
    tile_size: int = 512,
    overlap:   int = 64,
    device:    torch.device = torch.device("cpu"),
    max_val:   float = 2047.0,
) -> np.ndarray:
    """
    Tile-based inference for very large images (avoids OOM).
    Uses overlap-blend to remove tile boundary artifacts.
    """
    model.eval()
    C, H, W = lrms.shape
    output  = np.zeros((C, H, W), dtype=np.float64)
    weight  = np.zeros((1, H, W), dtype=np.float64)
    step    = tile_size - overlap

    for y in range(0, H, step):
        for x in range(0, W, step):
            y2 = min(y + tile_size, H)
            x2 = min(x + tile_size, W)
            y1 = max(0, y2 - tile_size)
            x1 = max(0, x2 - tile_size)

            pan_tile  = pan[:, y1:y2, x1:x2]
            lrms_tile = lrms[:, y1:y2, x1:x2]

            pred_tile = infer_single(model, pan_tile, lrms_tile, device, max_val)

            # Blend with Hann window to reduce boundary artifacts
            h_tile, w_tile = pred_tile.shape[-2], pred_tile.shape[-1]
            hann_h = np.hanning(h_tile).reshape(-1, 1)
            hann_w = np.hanning(w_tile).reshape(1, -1)
            w_tile_arr = (hann_h * hann_w)[np.newaxis]    # (1, h, w)

            output[:, y1:y2, x1:x2] += pred_tile * w_tile_arr
            weight[:,  y1:y2, x1:x2] += w_tile_arr

    weight = np.maximum(weight, 1e-8)
    return (output / weight).astype(np.float32)


@torch.no_grad()
def evaluate_test_set(
    model:   torch.nn.Module,
    loader,
    device:  torch.device,
    out_dir: Path,
    save_images: bool = False,
    scale_ratio: int = 4,
):
    """Evaluate on full test set, optionally saving all outputs."""
    from utils.metrics import MetricTracker
    import torchvision.utils as vutils

    model.eval()
    tracker = MetricTracker()

    for i, batch in enumerate(loader):
        pan  = batch["pan"].to(device)
        lrms = batch["lrms"].to(device)
        gt   = batch["gt"].to(device)

        pred = model(pan, lrms).float()

        tracker.update_batch(
            gt.cpu().numpy(),
            pred.cpu().numpy(),
            ratio=scale_ratio
        )

        if save_images:
            out_dir.mkdir(parents=True, exist_ok=True)
            # Save RGB visualization (first 3 bands)
            rgb  = pred[0, :3].clamp(0, 1).cpu()
            vutils.save_image(rgb, out_dir / f"pred_{i:04d}.png")
            gt_rgb = gt[0, :3].clamp(0, 1).cpu()
            vutils.save_image(gt_rgb, out_dir / f"gt_{i:04d}.png")

    results = tracker.compute()
    console.print(f"\n[bold]Test Results:[/bold]  {tracker}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Pansharpening inference")
    parser.add_argument("--config",      type=str, required=True)
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--pan",         type=str, default=None, help="PAN .tif for single-image mode")
    parser.add_argument("--ms",          type=str, default=None, help="MS .tif for single-image mode")
    parser.add_argument("--output",      type=str, default="outputs/result.tif")
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--tiled",       action="store_true", help="Use tile-based inference")
    parser.add_argument("--tile_size",   type=int, default=512)
    parser.add_argument("--device",      type=str, default="cuda")
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    base_cfg  = OmegaConf.load("configs/base.yaml")
    model_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, model_cfg)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Load model ────────────────────────────────────────────────────────────
    model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)
    model_kwargs.pop("name")
    model = get_model(cfg.model.name, **model_kwargs).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    console.print(f"[green]✓ Loaded {cfg.model.name} from {args.checkpoint}[/green]")

    # ── Single-image mode ─────────────────────────────────────────────────────
    if args.pan and args.ms:
        console.print("[cyan]Running single-image inference...[/cyan]")
        pan_arr,  pan_profile = load_geotiff(args.pan)
        ms_arr,   ms_profile  = load_geotiff(args.ms)

        # Upsample MS to PAN resolution
        import cv2
        _, H, W = pan_arr.shape
        ms_up = np.stack([
            cv2.resize(ms_arr[c], (W, H), interpolation=cv2.INTER_CUBIC)
            for c in range(ms_arr.shape[0])
        ])

        max_val = cfg.dataset.get("max_val", 2047.0)
        if args.tiled:
            fused = infer_tiled(model, pan_arr, ms_up, args.tile_size,
                                device=device, max_val=max_val)
        else:
            fused = infer_single(model, pan_arr, ms_up, device, max_val)

        save_geotiff(fused, args.output, pan_profile)
        console.print(f"[green]✓ Saved fused image → {args.output}[/green]")

    # ── Test set mode ─────────────────────────────────────────────────────────
    else:
        console.print("[cyan]Running test set evaluation...[/cyan]")
        from data.datasets.panbench import get_panbench_loaders

        loaders = get_panbench_loaders(
            h5_train   = cfg.dataset.h5_train,
            h5_val     = cfg.dataset.h5_val,
            h5_test    = cfg.dataset.get("h5_test", cfg.dataset.h5_val),
            satellite  = cfg.dataset.get("satellites", ["wv3"])[0],
            batch_size = 1,
            num_workers= 2,
        )
        out_dir = Path(cfg.paths.outputs) / cfg.model.name
        evaluate_test_set(
            model, loaders.get("test", loaders["val"]),
            device, out_dir,
            save_images=args.save_images,
            scale_ratio=cfg.dataset.get("scale_ratio", 4),
        )


if __name__ == "__main__":
    main()
