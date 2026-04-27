"""Model + artifact loading for the serve layer.

All loading is done once at startup via the lifespan; these helpers are
pure functions with no FastAPI dependencies.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from src.common.logging import get_logger

logger = get_logger("serve.model_loader")


def resolve_best_checkpoint(path: Path) -> Path:
    """Return the best `.pt` from a directory, or the file itself if given a file."""
    if path.is_file():
        return path
    pts = sorted(path.glob("*.pt"))
    if not pts:
        raise FileNotFoundError(f"No .pt checkpoints found in {path}")
    _re = re.compile(r"=(-?[\d.]+)\.pt$")

    def _score(f: Path) -> float:
        m = _re.search(f.name)
        return float(m.group(1)) if m else f.stat().st_mtime

    best = sorted(pts, key=_score, reverse=True)[0]
    logger.info("Resolved best checkpoint: %s", best.name)
    return best


def load_label_map(path: Path) -> dict:
    """Load label_map.json → {idx_to_str: {int: str}, num_classes: int}."""
    raw = json.loads(path.read_text())
    idx_to_str = {int(k): v for k, v in raw["idx_to_str"].items()}
    return {
        "idx_to_str": idx_to_str,
        "str_to_idx": raw["str_to_idx"],
        "num_classes": raw["num_classes"],
    }


def load_model(
    arch: str,
    num_classes: int,
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    """Build model architecture, load checkpoint weights, set to eval mode."""
    from src.common.config import ModelConfig
    from src.train.model_factory import build_model

    cfg = ModelConfig(
        architecture=arch,
        in_channels=1,
        num_classes=num_classes,
        pretrained=False,   # weights come from checkpoint
        dropout_rate=0.0,
    )
    model = build_model(cfg)

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Checkpoints may be raw state_dict or wrapped {"model_state_dict": ...}
    state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Checkpoint missing keys: %d", len(missing))
    if unexpected:
        logger.warning("Checkpoint unexpected keys: %d", len(unexpected))

    model.to(device).eval()
    logger.info(
        "Model loaded: arch=%s  classes=%d  device=%s  ckpt=%s",
        arch, num_classes, device, checkpoint_path.name,
    )
    return model
