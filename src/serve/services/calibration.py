"""Temperature-scaling calibration for the serve layer."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from src.common.logging import get_logger

logger = get_logger("serve.calibration")


def load_temperature(path: Path) -> float:
    """Read temperature scalar from calibration.json. Returns 1.0 if missing."""
    if not path.exists():
        logger.warning("Calibration file not found at %s – using T=1.0", path)
        return 1.0
    data = json.loads(path.read_text())
    T = float(data.get("temperature", 1.0))
    logger.info("Loaded calibration: T=%.4f from %s", T, path)
    return T


def apply_calibration(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide logits by temperature and softmax → calibrated probabilities.

    Args:
        logits: raw logits tensor of shape (num_classes,).
        temperature: scalar T > 0.

    Returns:
        Calibrated probability tensor of shape (num_classes,).
    """
    T = max(temperature, 1e-6)
    return F.softmax(logits / T, dim=0)
