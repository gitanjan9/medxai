"""Model factory: creates and initialises MONAI/timm classifiers."""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from src.common.config import ModelConfig
from src.common.exceptions import ModelError

logger = logging.getLogger("medicalxai.model_factory")


# Public factory


def build_model(cfg: ModelConfig) -> nn.Module:
    """Instantiate and return the classifier described by *cfg*.

    Strategy:
    - ``densenet121`` → MONAI DenseNet121 with torchvision weight adaptation.
    - ``efficientnet_b0`` → timm EfficientNet-B0 (handles 1-channel pretrained init).
    - ``resnet50`` / ``resnet18`` → timm ResNet with 1-channel init.

    All models output raw logits of shape ``(B, num_classes)``.
    """
    arch = cfg.architecture
    logger.info(
        "Building model: arch=%s  pretrained=%s  in_ch=%d  num_classes=%d",
        arch, cfg.pretrained, cfg.in_channels, cfg.num_classes,
    )

    if arch == "densenet121":
        model = _build_densenet121(cfg)
    elif arch == "efficientnet_b0":
        model = _build_timm("efficientnet_b0", cfg)
    elif arch == "efficientnet_b3":
        model = _build_timm("efficientnet_b3", cfg)
    elif arch == "efficientnet_b4":
        model = _build_timm("efficientnet_b4", cfg)
    elif arch == "resnet50":
        model = _build_timm("resnet50", cfg)
    elif arch == "resnet18":
        model = _build_timm("resnet18", cfg)
    else:
        raise ModelError(f"Unknown architecture: '{arch}'")

    if cfg.freeze_backbone:
        _freeze_backbone(model, arch)

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters: total=%d  trainable=%d", n_params, n_train)

    return model

# Architecture builders

def _build_densenet121(cfg: ModelConfig) -> nn.Module:
    """MONAI DenseNet121 with optional pretrained weight adaptation."""
    try:
        from monai.networks.nets import DenseNet121
    except ImportError as exc:
        raise ModelError("MONAI is not installed. Run: pip install monai") from exc

    model = DenseNet121(
        spatial_dims=cfg.spatial_dims,
        in_channels=cfg.in_channels,
        out_channels=cfg.num_classes,
        dropout_prob=cfg.dropout_rate,
    )

    if cfg.pretrained:
        _adapt_densenet_pretrained(model, cfg.in_channels)

    return model


def _build_timm(model_name: str, cfg: ModelConfig) -> nn.Module:
    """Build a timm model with 1-channel pretrained weight adaptation."""
    try:
        import timm
    except ImportError as exc:
        raise ModelError("timm is not installed. Run: pip install timm") from exc

    model = timm.create_model(
        model_name,
        pretrained=cfg.pretrained,
        in_chans=cfg.in_channels,   # timm averages RGB channels into 1
        num_classes=cfg.num_classes,
        drop_rate=cfg.dropout_rate,
    )
    return model


# Pretrained weight helpers

def _adapt_densenet_pretrained(model: nn.Module, in_channels: int) -> None:
    """Load torchvision DenseNet-121 weights into a MONAI DenseNet-121.

    MONAI's DenseNet has the same topology as torchvision's but uses
    different state-dict key prefixes.  We remap the keys and adapt
    the first conv layer for *in_channels* ≠ 3 by channel-averaging.
    """
    try:
        import torchvision.models as tv_models
    except ImportError:
        logger.warning("torchvision not available – skipping pretrained init for DenseNet121.")
        return

    try:
        tv_weights = tv_models.densenet121(
            weights=tv_models.DenseNet121_Weights.DEFAULT
        ).state_dict()
    except Exception as exc:
        logger.warning("Failed to download torchvision weights: %s. Training from scratch.", exc)
        return

    # Key differences torchvision → MONAI:
    #   TV:    "features.denseblockN.denselayerM.norm1.weight"
    #   MONAI: "features.denseblockN.denselayerM.layers.norm1.weight"
    #   TV head:    "classifier.weight/bias"
    #   MONAI head: "class_layers.out.weight/bias"
    import re
    _DENSE_LAYER_RE = re.compile(
        r"(features\.denseblock\d+\.denselayer\d+)\.(.*)"
    )

    remapped: dict[str, torch.Tensor] = {}
    for k, v in tv_weights.items():
        if k.startswith("classifier."):
            continue  # skip head – num_classes may differ
        m = _DENSE_LAYER_RE.match(k)
        if m:
            # insert ".layers." between denselayer prefix and the rest
            new_key = f"{m.group(1)}.layers.{m.group(2)}"
        else:
            new_key = k
        remapped[new_key] = v

    # Adapt first conv: (64, 3, 7, 7) → (64, in_channels, 7, 7)
    if in_channels != 3 and "features.conv0.weight" in remapped:
        w = remapped["features.conv0.weight"]   # (64, 3, 7, 7)
        remapped["features.conv0.weight"] = w.mean(dim=1, keepdim=True).repeat(
            1, in_channels, 1, 1
        )

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    logger.info(
        "Loaded pretrained DenseNet-121 weights.  Missing keys: %d  Unexpected: %d",
        len(missing), len(unexpected),
    )


# Backbone freezing


def _freeze_backbone(model: nn.Module, arch: str) -> None:
    """Freeze all parameters except the final classification head."""
    frozen = 0
    for name, param in model.named_parameters():
        if _is_head_param(name, arch):
            param.requires_grad = True
        else:
            param.requires_grad = False
            frozen += 1
    logger.info("Froze %d backbone parameters (arch=%s).", frozen, arch)


def _is_head_param(name: str, arch: str) -> bool:
    """Return True if *name* belongs to the classification head."""
    head_prefixes = {
        "densenet121":    ("class_layers",),
        "efficientnet_b0": ("classifier",),
        "efficientnet_b3": ("classifier",),
        "efficientnet_b4": ("classifier",),
        "resnet50":  ("fc",),
        "resnet18":  ("fc",),
    }
    prefixes = head_prefixes.get(arch, ("classifier", "fc", "head"))
    return any(name.startswith(p) for p in prefixes)
