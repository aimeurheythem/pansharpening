#!/usr/bin/env python3
"""
setup_and_train.py — One command to rule them all.

Does EVERYTHING automatically:
    1. Installs all Python dependencies
    2. Downloads PanScale dataset from HuggingFace (kecao/PanScale)
    3. Inspects & validates the downloaded dataset structure
    4. Converts to HDF5 for fast training (or uses direct loading)
    5. Runs environment checks
    6. Starts training with PanFusionNet (hybrid CNN+Transformer)

Usage:
    python setup_and_train.py                      # Full pipeline
    python setup_and_train.py --download-only      # Just download the dataset
    python setup_and_train.py --skip-download      # Already downloaded, go to train
    python setup_and_train.py --model scaleformer  # Use a different model
    python setup_and_train.py --satellite wv3      # Choose satellite
    python setup_and_train.py --no-h5              # Skip HDF5 conversion (load directly)

Professor's commands (now automated):
    pip install -U huggingface_hub
    hf download kecao/PanScale --repo-type dataset --local-dir ./datasets/panscale
"""

import os
import sys
import subprocess
import argparse
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np


# =============================================================================
# COLORS — works without rich (before it's installed)
# =============================================================================


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BLUE = "\033[94m"


def ok(msg):
    print(f"{C.GREEN}{C.BOLD}  ✓  {C.RESET}{msg}")


def warn(msg):
    print(f"{C.YELLOW}{C.BOLD}  ⚠  {C.RESET}{msg}")


def err(msg):
    print(f"{C.RED}{C.BOLD}  ✗  {C.RESET}{msg}")


def info(msg):
    print(f"{C.CYAN}      {msg}{C.RESET}")


def header(msg):
    bar = "═" * (len(msg) + 4)
    print(f"\n{C.BLUE}{C.BOLD}╔{bar}╗")
    print(f"║  {msg}  ║")
    print(f"╚{bar}╝{C.RESET}\n")


# =============================================================================
# STEP 1 — INSTALL DEPENDENCIES
# =============================================================================

# Core packages required for training
REQUIREMENTS = [
    "torch>=2.0",
    "torchvision",
    "einops>=0.6",
    "omegaconf>=2.3",
    "h5py>=3.0",
    "numpy>=1.24",
    "scipy>=1.10",
    "scikit-image>=0.20",
    "opencv-python>=4.0",
    "rasterio>=1.3",
    "imageio",
    "tqdm>=4.60",
    "rich>=13.0",
    "tensorboard>=2.10",
    "huggingface_hub>=0.20",  # for dataset download
    "hf_transfer",  # faster HuggingFace downloads
    "timm>=0.9",
    "matplotlib>=3.7",
]


def install_dependencies():
    header("STEP 1 — Installing Dependencies")

    # Upgrade pip silently first
    print("  Upgrading pip...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        capture_output=True,
    )

    # Install huggingface_hub first (needed for download step)
    print("  Installing huggingface_hub (professor's command 1)...")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-U",
            "huggingface_hub",
            "hf_transfer",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        ok("huggingface_hub installed")
    else:
        warn(f"huggingface_hub install warning: {result.stderr[-200:]}")

    # Install all remaining requirements
    print("  Installing all project dependencies...")
    failed = []
    for pkg in REQUIREMENTS:
        pkg_name = pkg.split(">=")[0].split("[")[0]
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            ok(f"{pkg_name}")
        else:
            warn(f"{pkg_name} failed — will try to continue")
            failed.append(pkg_name)

    if failed:
        warn(f"Some packages failed to install: {failed}")
        warn("Training may still work if these are optional dependencies.")
    else:
        ok("All dependencies installed successfully")


# =============================================================================
# STEP 2 — DOWNLOAD PANSCALE DATASET
# =============================================================================


def download_panscale(local_dir: str = "./datasets/panscale"):
    header("STEP 2 — Downloading PanScale Dataset from HuggingFace")

    local_path = Path(local_dir)

    # Check if already downloaded (look for any image file)
    if local_path.exists():
        all_files = (
            list(local_path.rglob("*.tif"))
            + list(local_path.rglob("*.png"))
            + list(local_path.rglob("*.npy"))
        )
        if all_files:
            ok(
                f"Dataset already exists at {local_dir} ({len(all_files)} image files found)"
            )
            ok("Skipping download.")
            return local_dir

    local_path.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading kecao/PanScale → {local_dir}")
    print(
        f"  {C.YELLOW}(This may take several minutes depending on dataset size){C.RESET}"
    )
    print()

    # Enable fast HuggingFace transfers
    env = os.environ.copy()
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # Professor's exact command (adapted for cross-platform paths)
    cmd = [
        sys.executable,
        "-m",
        "huggingface_hub.commands.huggingface_cli",
        "download",
        "kecao/PanScale",
        "--repo-type",
        "dataset",
        "--local-dir",
        str(local_path),
    ]

    # Try CLI tool first (professor's: hf download ...)
    try:
        result = subprocess.run(
            [
                "huggingface-cli",
                "download",
                "kecao/PanScale",
                "--repo-type",
                "dataset",
                "--local-dir",
                str(local_path),
            ],
            env=env,
            text=True,
        )
        if result.returncode == 0:
            ok(f"Dataset downloaded to {local_dir}")
            return local_dir
    except FileNotFoundError:
        pass  # huggingface-cli not on PATH, try Python API

    # Fallback: use Python API directly
    print("  Using Python API for download (huggingface_hub)...")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="kecao/PanScale",
            repo_type="dataset",
            local_dir=str(local_path),
        )
        ok(f"Dataset downloaded to {local_dir}")
    except Exception as e:
        err(f"Download failed: {e}")
        print()
        print(f"  {C.YELLOW}Please run manually:{C.RESET}")
        print(f"    pip install -U huggingface_hub")
        print(
            f"    huggingface-cli download kecao/PanScale --repo-type dataset --local-dir {local_dir}"
        )
        sys.exit(1)

    return local_dir


# =============================================================================
# STEP 3 — INSPECT DATASET STRUCTURE
# =============================================================================


def inspect_dataset(local_dir: str) -> dict:
    header("STEP 3 — Inspecting Dataset Structure")

    root = Path(local_dir)
    info(f"Root: {root.resolve()}")

    # Print top-level structure
    print("\n  Directory tree (3 levels):")
    _print_tree(root, max_depth=3, prefix="    ")

    # Detect splits
    splits_found = []
    for split in [
        "train",
        "val",
        "test",
        "Train",
        "Val",
        "Test",
        "training",
        "validation",
    ]:
        if (root / split).is_dir():
            splits_found.append((root / split / "..").resolve().name)
            splits_found.append(split)

    splits_found = list(dict.fromkeys(splits_found))  # deduplicate
    actual_splits = [s for s in splits_found if (root / s).is_dir()]
    info(f"Splits found: {actual_splits}")

    # Count images
    tif_files = list(root.rglob("*.tif")) + list(root.rglob("*.tiff"))
    png_files = list(root.rglob("*.png"))
    npy_files = list(root.rglob("*.npy"))
    all_imgs = tif_files + png_files + npy_files

    info(
        f"Total image files: {len(all_imgs)} "
        f"({len(tif_files)} TIF, {len(png_files)} PNG, {len(npy_files)} NPY)"
    )

    # Detect number of MS bands from a sample image
    ms_bands = 4  # default
    ms_sample = _find_sample_image(root, ["MS", "ms", "LMS", "lms"])
    if ms_sample:
        try:
            import rasterio

            with rasterio.open(str(ms_sample)) as src:
                ms_bands = src.count
            info(f"MS bands detected: {ms_bands} (from {ms_sample.name})")
        except Exception:
            warn(f"Could not read band count from {ms_sample}")

    # Detect PAN/MS scale ratio from a sample pair
    scale_ratio = 4  # default
    pan_sample = _find_sample_image(root, ["PAN", "pan", "Pan"])
    if pan_sample and ms_sample:
        try:
            import rasterio

            with rasterio.open(str(pan_sample)) as src:
                pan_h = src.height
            with rasterio.open(str(ms_sample)) as src:
                ms_h = src.height
            if ms_h > 0:
                ratio = pan_h / ms_h
                if ratio in (2, 4, 8, 16):
                    scale_ratio = int(ratio)
            info(f"Scale ratio detected: {scale_ratio}× (PAN:{pan_h} / MS:{ms_h})")
        except Exception:
            warn("Could not detect scale ratio — using default 4×")

    config = {
        "root": str(root.resolve()),
        "splits": actual_splits,
        "ms_bands": ms_bands,
        "scale_ratio": scale_ratio,
        "n_images": len(all_imgs),
    }

    print()
    ok(f"Dataset inspection complete: {ms_bands}-band MS, scale={scale_ratio}×")
    return config


def _find_sample_image(root: Path, folder_names: list) -> Optional[Path]:
    """Find one sample image from a folder matching any of folder_names."""
    for pattern in folder_names:
        candidates = list(root.rglob(pattern))
        for c in candidates:
            if c.is_dir():
                imgs = list(c.glob("*.tif")) + list(c.glob("*.png"))
                if imgs:
                    return imgs[0]
    return None


from typing import Optional


def _print_tree(path: Path, max_depth: int = 3, prefix: str = "", depth: int = 0):
    if depth > max_depth:
        return
    try:
        items = sorted(path.iterdir())
    except PermissionError:
        return
    # Show dirs and count files
    dirs = [i for i in items if i.is_dir()]
    files = [i for i in items if i.is_file()]
    for d in dirs[:6]:  # limit to first 6 dirs
        print(f"{prefix}📁 {d.name}/")
        _print_tree(d, max_depth, prefix + "   ", depth + 1)
    if len(dirs) > 6:
        print(f"{prefix}   ... ({len(dirs) - 6} more folders)")
    if files:
        print(
            f"{prefix}📄 {len(files)} files "
            f"({', '.join(set(f.suffix for f in files[:5]))})"
        )


# =============================================================================
# STEP 4 — CONVERT TO HDF5 (optional but faster training)
# =============================================================================


def convert_to_hdf5(
    dataset_config: dict, h5_dir: str = "./data/h5", satellite_tag: str = "panscale"
) -> dict:
    header("STEP 4 — Converting Dataset to HDF5 (faster training)")

    import numpy as np
    import h5py
    from tqdm import tqdm

    h5_paths = {}
    root = dataset_config["root"]
    ms_bands = dataset_config["ms_bands"]
    scale_ratio = dataset_config["scale_ratio"]
    splits = dataset_config["splits"]

    # Map common split names to our standard names
    split_map = {
        "train": "train",
        "training": "train",
        "Train": "train",
        "val": "val",
        "valid": "val",
        "Val": "val",
        "validation": "val",
        "test": "test",
        "Test": "test",
    }

    Path(h5_dir).mkdir(parents=True, exist_ok=True)

    for raw_split in splits:
        std_split = split_map.get(raw_split, raw_split)
        h5_path = Path(h5_dir) / f"{std_split}_{satellite_tag}.h5"
        h5_paths[std_split] = str(h5_path)

        if h5_path.exists():
            ok(f"HDF5 already exists: {h5_path} — skipping conversion")
            continue

        print(f"\n  Converting '{raw_split}' split → {h5_path}")

        # Discover scenes for this split
        try:
            # Import here (avoid top-level import before sys.path setup)
            sys.path.insert(0, str(Path(__file__).parent))
            from data.datasets.panscale import discover_panscale_scenes

            scenes = discover_panscale_scenes(root, raw_split)
        except Exception as e:
            warn(f"Could not discover scenes for split '{raw_split}': {e}")
            continue

        if not scenes:
            warn(f"No scenes found for split '{raw_split}' — skipping")
            continue

        info(f"  Processing {len(scenes)} image triplets...")

        # Patch extraction settings
        pan_patch = 128
        ms_patch = pan_patch // scale_ratio
        stride = pan_patch // 2

        all_pan, all_ms, all_gt, all_lrms = [], [], [], []

        for scene in tqdm(scenes, desc=f"  {std_split}", ncols=70):
            try:
                pan_img = _load_and_check(scene["pan"], expected_ch=1)
                ms_img = _load_and_check(scene["ms"], expected_ch=None)
                gt_img = _load_and_check(scene["gt"], expected_ch=None)

                # Normalize to [0, 1]
                norm = 65535.0 if pan_img.max() > 1.0 else 1.0
                pan_img /= norm
                ms_img /= norm
                gt_img /= norm

                # Bicubic upsample MS → lrms at PAN resolution
                import torch.nn.functional as F_
                import torch

                H_pan, W_pan = pan_img.shape[-2], pan_img.shape[-1]
                lrms_img = (
                    F_.interpolate(
                        torch.from_numpy(ms_img).unsqueeze(0),
                        size=(H_pan, W_pan),
                        mode="bicubic",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .numpy()
                )

                # Extract patches
                pans, mss, gts, lrmss = _extract_patches(
                    pan_img,
                    ms_img,
                    gt_img,
                    lrms_img,
                    pan_patch,
                    ms_patch,
                    stride,
                    scale_ratio,
                )
                all_pan.append(pans)
                all_ms.append(mss)
                all_gt.append(gts)
                all_lrms.append(lrmss)

            except Exception as e:
                warn(f"Skipping {Path(scene['pan']).name}: {e}")
                continue

        if not all_pan:
            warn(f"No valid patches extracted for split '{std_split}'")
            continue

        # Concatenate and save
        pan_arr = np.concatenate(all_pan, axis=0).astype(np.float32)
        ms_arr = np.concatenate(all_ms, axis=0).astype(np.float32)
        gt_arr = np.concatenate(all_gt, axis=0).astype(np.float32)
        lrms_arr = np.concatenate(all_lrms, axis=0).astype(np.float32)

        print(
            f"    Patches: {pan_arr.shape[0]} | PAN:{pan_arr.shape[1:]} | GT:{gt_arr.shape[1:]}"
        )

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("pan", data=pan_arr, compression="gzip", chunks=True)
            f.create_dataset("ms", data=ms_arr, compression="gzip", chunks=True)
            f.create_dataset("gt", data=gt_arr, compression="gzip", chunks=True)
            f.create_dataset("lrms", data=lrms_arr, compression="gzip", chunks=True)
            f.attrs["satellite"] = satellite_tag
            f.attrs["split"] = std_split
            f.attrs["n_samples"] = pan_arr.shape[0]
            f.attrs["ms_bands"] = ms_arr.shape[1]
            f.attrs["scale_ratio"] = scale_ratio

        size_mb = h5_path.stat().st_size / 1e6
        ok(f"Saved {h5_path.name} ({size_mb:.1f} MB, {pan_arr.shape[0]:,} patches)")

    return h5_paths


def _load_and_check(path: str, expected_ch: Optional[int]) -> "np.ndarray":
    """Load image and validate channel count."""
    import numpy as np

    try:
        import rasterio

        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)
    except Exception:
        import imageio

        data = np.array(imageio.imread(path), dtype=np.float32)
        if data.ndim == 2:
            data = data[np.newaxis]
        elif data.ndim == 3:
            data = data.transpose(2, 0, 1)

    if expected_ch == 1 and data.shape[0] > 1:
        data = data[:1]  # keep only first channel
    return data


def _extract_patches(pan, ms, gt, lrms, pan_p, ms_p, stride, scale_ratio):
    """Extract patches from a single full-image triplet."""
    import numpy as np

    _, H, W = pan.shape
    pan_list, ms_list, gt_list, lrms_list = [], [], [], []

    for y in range(0, H - pan_p + 1, stride):
        for x in range(0, W - pan_p + 1, stride):
            ys = y // scale_ratio
            xs = x // scale_ratio

            pan_list.append(pan[:, y : y + pan_p, x : x + pan_p])
            lrms_list.append(lrms[:, y : y + pan_p, x : x + pan_p])
            gt_list.append(gt[:, y : y + pan_p, x : x + pan_p])
            ms_list.append(ms[:, ys : ys + ms_p, xs : xs + ms_p])

    if not pan_list:
        # Image too small for patches — use as-is (resize to patch size)
        import torch.nn.functional as F_
        import torch

        pan_r = (
            F_.interpolate(
                torch.from_numpy(pan).unsqueeze(0),
                (pan_p, pan_p),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .numpy()
        )
        lrms_r = (
            F_.interpolate(
                torch.from_numpy(lrms).unsqueeze(0),
                (pan_p, pan_p),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .numpy()
        )
        gt_r = (
            F_.interpolate(
                torch.from_numpy(gt).unsqueeze(0),
                (pan_p, pan_p),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .numpy()
        )
        ms_r = (
            F_.interpolate(
                torch.from_numpy(ms).unsqueeze(0),
                (ms_p, ms_p),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .numpy()
        )
        return (
            np.array([pan_r]),
            np.array([ms_r]),
            np.array([gt_r]),
            np.array([lrms_r]),
        )

    return (
        np.array(pan_list),
        np.array(ms_list),
        np.array(gt_list),
        np.array(lrms_list),
    )


# =============================================================================
# STEP 5 — ENVIRONMENT CHECK
# =============================================================================


def check_environment():
    header("STEP 5 — Environment Check")
    try:
        import torch

        ok(f"PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            ok(f"CUDA GPU: {gpu} ({mem:.1f} GB)")
        else:
            warn("No CUDA GPU found — training will run on CPU (very slow!)")
    except ImportError:
        err("PyTorch not found! Run setup again.")
        sys.exit(1)


# =============================================================================
# STEP 6 — GENERATE TRAINING CONFIG
# =============================================================================


def generate_config(
    dataset_config: dict,
    h5_paths: dict,
    model_name: str,
    satellite_tag: str,
    use_h5: bool,
) -> str:
    header("STEP 6 — Generating Training Configuration")

    ms_bands = dataset_config["ms_bands"]
    scale_ratio = dataset_config["scale_ratio"]
    config_path = f"configs/{model_name}_panscale.yaml"

    # Adjust model head/channel counts to actual MS band count
    if model_name == "panfusionnet":
        model_block = f"""model:
  name: "panfusionnet"
  ms_channels: {ms_bands}
  pan_channels: 1
  embed_dim: 64
  num_heads: 4
  num_cnn_blocks: 4
  num_attn_layers: 2
  token_size: 16
  dropout: 0.0"""

    elif model_name == "scaleformer":
        model_block = f"""model:
  name: "scaleformer"
  ms_channels: {ms_bands}
  pan_channels: 1
  embed_dim: 64
  num_heads: 8
  num_layers: 6
  window_size: 8
  mlp_ratio: 4.0
  dropout: 0.1
  scale_ratio: {scale_ratio}"""

    elif model_name == "wav_cbt":
        model_block = f"""model:
  name: "wav_cbt"
  ms_channels: {ms_bands}
  pan_channels: 1
  embed_dim: 48
  num_heads: 6
  num_blocks: 4
  mlp_ratio: 2.0
  dropout: 0.0"""

    else:
        model_block = f"""model:
  name: "{model_name}"
  ms_channels: {ms_bands}
  pan_channels: 1"""

    # Dataset block
    if use_h5 and h5_paths:
        train_h5 = h5_paths.get("train", "")
        val_h5 = h5_paths.get("val", h5_paths.get("train", ""))
        test_h5 = h5_paths.get("test", val_h5)
        dataset_block = f"""dataset:
  name: "panbench"
  h5_train: "{train_h5}"
  h5_val:   "{val_h5}"
  h5_test:  "{test_h5}"
  satellites:
    - {satellite_tag}
  scale_ratio: {scale_ratio}"""
    else:
        dataset_block = f"""dataset:
  name: "panscale"
  root: "{dataset_config["root"]}"
  scale_ratio: {scale_ratio}
  patch_size: 128"""

    config_content = f"""# =============================================================================
# AUTO-GENERATED by setup_and_train.py
# Model:   {model_name}
# Dataset: PanScale (kecao/PanScale)
# MS bands: {ms_bands} | Scale ratio: {scale_ratio}
# =============================================================================

defaults:
  - base

{model_block}

{dataset_block}

training:
  epochs: 200
  batch_size: 16
  accum_steps: 2           # effective batch = 16 x 2 = 32
  warmup_epochs: 10
  val_interval: 5
  save_interval: 10
  early_stopping_patience: 30

optimizer:
  name: "adamw"
  lr: 2.0e-4
  weight_decay: 1.0e-4
  betas: [0.9, 0.999]

scheduler:
  name: "cosine_warmup"
  eta_min: 1.0e-6

loss:
  l1_weight: 1.0
  ssim_weight: 0.5
  sam_weight: 0.1
"""

    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        f.write(config_content)

    ok(f"Config written → {config_path}")
    return config_path


# =============================================================================
# STEP 7 — UPDATE TRAIN.PY TO SUPPORT PANSCALE DIRECT LOADING
# =============================================================================


def patch_train_for_panscale():
    """
    Ensure train.py handles the 'panscale' dataset name.
    Adds a branch for get_panscale_loaders() alongside panbench.
    """
    train_path = Path("train.py")
    content = train_path.read_text()

    # Only patch if not already patched
    if "panscale" in content:
        return

    old_block = """    if dataset_name == "panbench":
        loaders = get_panbench_loaders(
            h5_train   = cfg.dataset.h5_train,
            h5_val     = cfg.dataset.h5_val,
            h5_test    = cfg.dataset.get("h5_test", None),
            satellite  = cfg.dataset.get("satellites", ["wv3"])[0],
            batch_size = cfg.training.batch_size,
            num_workers= cfg.hardware.num_workers,
        )
    else:
        raise NotImplementedError(
            f"Dataset \'{dataset_name}\' loader not yet implemented. "
            f"Add it in data/datasets/ and register here."
        )"""

    new_block = """    if dataset_name == "panbench":
        loaders = get_panbench_loaders(
            h5_train   = cfg.dataset.h5_train,
            h5_val     = cfg.dataset.h5_val,
            h5_test    = cfg.dataset.get("h5_test", None),
            satellite  = cfg.dataset.get("satellites", ["wv3"])[0],
            batch_size = cfg.training.batch_size,
            num_workers= cfg.hardware.num_workers,
        )
    elif dataset_name == "panscale":
        from data.datasets.panscale import get_panscale_loaders
        loaders = get_panscale_loaders(
            root        = cfg.dataset.root,
            batch_size  = cfg.training.batch_size,
            patch_size  = cfg.dataset.get("patch_size", 128),
            num_workers = cfg.hardware.num_workers,
            scale_ratio = cfg.dataset.get("scale_ratio", 4),
        )
    else:
        raise NotImplementedError(
            f"Dataset \'{dataset_name}\' loader not yet implemented. "
            f"Add it in data/datasets/ and register here."
        )"""

    if old_block in content:
        content = content.replace(old_block, new_block)
        train_path.write_text(content)
        ok("train.py patched to support PanScale direct loading")


# =============================================================================
# STEP 8 — LAUNCH TRAINING
# =============================================================================


def launch_training(
    config_path: str, resume: Optional[str] = None, wandb: bool = False
):
    header("STEP 8 — Launching Training 🚀")

    cmd = [sys.executable, "train.py", "--config", config_path]
    if resume:
        cmd += ["--resume", resume]
    if wandb:
        cmd += ["--wandb"]

    print(f"  Running: {' '.join(cmd)}\n")
    print(f"  {C.YELLOW}Training logs → logs/tensorboard/")
    print(f"  Checkpoints  → checkpoints/")
    print(f"  Monitor with: tensorboard --logdir logs/tensorboard{C.RESET}\n")

    # Run training (inherits stdout so progress is visible)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        err(f"Training exited with code {result.returncode}")
    else:
        ok("Training completed successfully!")


# =============================================================================
# MAIN
# =============================================================================


def parse_args():
    p = argparse.ArgumentParser(
        description="One-command setup + train for PanSharpeningPro",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_and_train.py                       # Full pipeline (recommended)
  python setup_and_train.py --download-only       # Only download dataset
  python setup_and_train.py --skip-download       # Already downloaded, run train
  python setup_and_train.py --no-h5               # Skip HDF5, load images directly
  python setup_and_train.py --model scaleformer   # Use ScaleFormer instead
  python setup_and_train.py --wandb               # Enable WandB logging
        """,
    )
    p.add_argument(
        "--dataset-dir",
        default="./datasets/panscale",
        help="Where to download PanScale dataset",
    )
    p.add_argument("--h5-dir", default="./data/h5", help="Where to save HDF5 files")
    p.add_argument(
        "--model",
        default="panfusionnet",
        choices=["panfusionnet", "scaleformer", "wav_cbt"],
        help="Which model to train",
    )
    p.add_argument(
        "--satellite", default="panscale", help="Satellite tag used for HDF5 filenames"
    )
    p.add_argument("--skip-install", action="store_true", help="Skip pip install step")
    p.add_argument(
        "--download-only", action="store_true", help="Only download dataset, then stop"
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download (dataset already present)",
    )
    p.add_argument(
        "--no-h5",
        action="store_true",
        help="Skip HDF5 conversion and load images directly",
    )
    p.add_argument("--resume", default=None, help="Resume training from checkpoint")
    p.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{C.BLUE}{C.BOLD}")
    print("  ╔═══════════════════════════════════════════════════╗")
    print("  ║     PanSharpeningPro — Full Setup & Train         ║")
    print("  ║     Dataset: kecao/PanScale (HuggingFace)         ║")
    print(f"  ║     Model:   {args.model:<38}║")
    print("  ╚═══════════════════════════════════════════════════╝")
    print(f"{C.RESET}")

    start_time = time.time()

    # Step 1: Install dependencies
    if not args.skip_install:
        install_dependencies()
    else:
        info("Skipping dependency installation (--skip-install)")

    # Step 2: Download dataset
    if not args.skip_download:
        download_panscale(args.dataset_dir)
    else:
        info(f"Skipping download — using existing dataset at {args.dataset_dir}")

    if args.download_only:
        ok("Dataset downloaded. Run again without --download-only to start training.")
        return

    # Step 3: Inspect dataset
    dataset_config = inspect_dataset(args.dataset_dir)

    # Step 4: Convert to HDF5 (or skip)
    h5_paths = {}
    if not args.no_h5:
        h5_paths = convert_to_hdf5(dataset_config, args.h5_dir, args.satellite)
    else:
        info("Skipping HDF5 conversion — images will be loaded directly")

    # Step 5: Environment check
    check_environment()

    # Step 6: Patch train.py for direct PanScale support
    patch_train_for_panscale()

    # Step 7: Generate training config
    use_h5 = (not args.no_h5) and bool(h5_paths)
    config_path = generate_config(
        dataset_config,
        h5_paths,
        model_name=args.model,
        satellite_tag=args.satellite,
        use_h5=use_h5,
    )

    # Summary before training
    elapsed = time.time() - start_time
    print(f"\n{'─' * 55}")
    ok(f"Setup complete in {elapsed:.0f}s")
    info(f"Config:  {config_path}")
    info(f"Model:   {args.model}")
    info(
        f"Dataset: PanScale ({dataset_config['ms_bands']}-band MS, "
        f"{dataset_config['scale_ratio']}× scale)"
    )
    if h5_paths:
        for split, path in h5_paths.items():
            info(f"  HDF5 {split}: {path}")
    print(f"{'─' * 55}\n")

    # Step 8: Launch training
    launch_training(config_path, resume=args.resume, wandb=args.wandb)


if __name__ == "__main__":
    main()
