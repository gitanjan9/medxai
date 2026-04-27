"""Grad-CAM wrapper for the serve layer.

Delegates to the existing GradCAMGenerator from src.train.explainability and
returns a base64-encoded PNG suitable for JSON responses.
"""
from __future__ import annotations

import base64
import io
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.common.logging import get_logger
from src.train.explainability import GradCAMGenerator

logger = get_logger("serve.explainability")

# EfficientNet-B3 (timm): last feature conv before classifier
_DEFAULT_TARGET_LAYERS: dict[str, str] = {
    "efficientnet_b3": "conv_head",
    "efficientnet_b4": "conv_head",
    "efficientnet_b0": "conv_head",
    "densenet121":     "features.denseblock4",
    "resnet50":        "layer4",
    "resnet18":        "layer4",
}


def build_gradcam(model: nn.Module, arch: str, device: torch.device) -> GradCAMGenerator:
    """Build a GradCAMGenerator for *arch*."""
    target_layer = _DEFAULT_TARGET_LAYERS.get(arch, "conv_head")
    logger.info("GradCAM target layer: %s (arch=%s)", target_layer, arch)
    return GradCAMGenerator(model=model, target_layer=target_layer, device=device)


def run_gradcam_base64(
    gradcam: GradCAMGenerator,
    tensor: torch.Tensor,        # (1, 1, H, W)
    target_class: int,
    output_size: tuple[int, int] = (320, 320),
) -> Optional[str]:
    """Return base64-encoded PNG of the Grad-CAM heatmap, or None on failure."""
    try:
        from PIL import Image as PILImage
    except ImportError:
        logger.warning("Pillow not available – cannot encode GradCAM to PNG")
        return None

    heatmap = gradcam.generate(tensor, target_class, output_size=output_size)
    if heatmap is None:
        return None

    # Normalise [0,1] → uint8 and encode as PNG
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    img = PILImage.fromarray(heatmap_uint8, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
