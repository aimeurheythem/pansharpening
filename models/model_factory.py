"""
model_factory.py — Central registry for all pansharpening models.

Usage:
    from models.model_factory import get_model, list_models

    model = get_model("panfusionnet", ms_channels=8, embed_dim=64)
    model = get_model("scaleformer",  ms_channels=4, embed_dim=64)
    model = get_model("wav_cbt",      ms_channels=8)
"""

from models.scaleformer.scaleformer import ScaleFormer
from models.wav_cbt.wav_cbt import WavCBT
from models.panfusionnet.panfusionnet import PanFusionNet
from models.convnext_pan.convnext_pan import ConvNeXtPan
from models.pan_pix2pix.pan_pix2pix import PanPix2Pix
from models.psgan_dwt.psgan_dwt import PSGANDWTModel


# =============================================================================
# MODEL REGISTRY
# =============================================================================

_REGISTRY = {
    "scaleformer": ScaleFormer,
    "wav_cbt": WavCBT,
    "panfusionnet": PanFusionNet,
    "convnext_pan": ConvNeXtPan,
    "pan_pix2pix": PanPix2Pix,
    "psgan_dwt": PSGANDWTModel,
}


def get_model(name: str, **kwargs):
    """
    Instantiate a model by name with given kwargs.

    Args:
        name:     Model name (see list_models())
        **kwargs: Constructor arguments (ms_channels, embed_dim, etc.)

    Returns:
        nn.Module: Instantiated model (NOT moved to device — do that outside)

    Example:
        model = get_model("panfusionnet", ms_channels=8, embed_dim=64)
        model = model.cuda()
    """
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'.\n"
            f"Available models: {list(_REGISTRY.keys())}"
        )
    model = _REGISTRY[name](**kwargs)
    n_params = model.count_parameters()
    print(f"[ModelFactory] Loaded '{name}' — {n_params:,} trainable parameters")
    return model


def list_models() -> list:
    """Return list of available model names."""
    return sorted(_REGISTRY.keys())