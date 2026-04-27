"""Temperature-scaling calibration for post-hoc probability calibration.

Usage::

    python -m src.train.calibrate --config configs/train.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.common.logging import get_logger

logger = get_logger("calibrate")


# ---------------------------------------------------------------------------
# Temperature scaler
# ---------------------------------------------------------------------------


class TemperatureScaler(nn.Module):
    """Single-parameter temperature scaling module.

    Calibrates a classifier by dividing logits by a learned scalar T.
    T > 1  → softer probabilities (less extreme).
    T < 1  → sharper probabilities.
    """

    def __init__(self, init_T: float = 1.5) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_T)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.05)

    @property
    def T(self) -> float:
        return float(self.temperature.item())


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 50,
    lr: float = 0.01,
) -> TemperatureScaler:
    """Fit temperature T on held-out logits/labels using NLL (cross-entropy).

    Uses L-BFGS so it converges in very few steps regardless of dataset size.
    """
    scaler = TemperatureScaler()
    optimizer = torch.optim.LBFGS(
        [scaler.temperature], lr=lr, max_iter=max_iter
    )
    logits_t = torch.from_numpy(logits).float()
    labels_t = torch.from_numpy(labels).long()
    criterion = nn.CrossEntropyLoss()

    def _eval() -> torch.Tensor:
        optimizer.zero_grad()
        loss = criterion(scaler(logits_t), labels_t)
        loss.backward()
        return loss

    optimizer.step(_eval)
    logger.info("Temperature fitted: T=%.4f", scaler.T)
    return scaler


# ---------------------------------------------------------------------------
# Forward-pass collection
# ---------------------------------------------------------------------------


def collect_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return (logits, labels) as numpy arrays."""
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            lbs = batch["label"]
            all_logits.append(model(imgs).cpu().numpy())
            all_labels.append(lbs.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Artifact I/O
# ---------------------------------------------------------------------------


def save_calibration(
    scaler: TemperatureScaler,
    output_path: Path,
    meta: Optional[dict] = None,
) -> None:
    """Persist calibration artifact as JSON."""
    artifact = {
        "method": "temperature_scaling",
        "temperature": scaler.T,
        **(meta or {}),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(artifact, fh, indent=2)
    logger.info("Calibration saved → %s  (T=%.4f)", output_path, scaler.T)


def load_calibration(path: Path) -> dict:
    """Load calibration artifact from JSON. Returns dict with 'temperature' key."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration artifact not found: {path}")
    with open(path) as fh:
        return json.load(fh)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling and return calibrated softmax probabilities.

    Args:
        logits: Raw model logits, shape (N, C).
        temperature: Fitted T value.

    Returns:
        Calibrated probabilities, shape (N, C), rows sum to 1.
    """
    scaled = logits / max(float(temperature), 0.05)
    e = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def calibrate(
    config_path: str,
    output_path: Optional[str] = None,
    max_iter: int = 50,
    lr: float = 0.01,
) -> dict:
    """Load the best checkpoint, fit temperature on the val split, save artifact.

    Returns the saved calibration dict so callers can inspect T directly.
    """
    from src.common.config import TrainConfig
    from src.common.logging import setup_logging
    from src.common.schemas import LabelMap
    from src.common.utils import get_device
    from src.train.dataset import CXRDataset, load_and_prepare_dataframe
    from src.train.model_factory import build_model
    from src.train.transforms import build_val_transforms
    from src.train.evaluate import _resolve_checkpoint, _load_checkpoint

    setup_logging()
    cfg = TrainConfig.from_yaml(config_path)
    device = get_device()

    # Label map
    lm_path = cfg.data.label_mapping_path
    if lm_path and Path(lm_path).exists():
        label_map = LabelMap.load(lm_path)
    else:
        from src.train.dataset import build_label_map_from_csv
        label_map = build_label_map_from_csv(
            cfg.data.train_path, label_col=cfg.data.label_col
        )
    cfg.model.num_classes = label_map.num_classes

    # Val dataset
    val_csv = cfg.data.val_path or cfg.data.train_path
    df = load_and_prepare_dataframe(
        val_csv, cfg.data.label_col, label_map,
        merge_map_path=cfg.data.class_merge_map_path,
        tag="Calibration",
    )

    ds = CXRDataset(
        df=df,
        label_map=label_map,
        image_col=cfg.data.image_col,
        label_col=cfg.data.label_col,
        text_col=cfg.data.text_col,
        transforms=build_val_transforms(cfg.data.image_size),
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    # Model
    model = build_model(cfg.model).to(device)
    ckpt = _resolve_checkpoint(None, cfg)
    if ckpt:
        _load_checkpoint(model, ckpt, device)
    else:
        logger.warning("No checkpoint found – calibrating random weights.")

    # Collect logits
    logger.info("Collecting logits …")
    logits, labels = collect_logits(model, loader, device)

    # Override with config values if not passed explicitly
    max_iter = max_iter or cfg.calibration.max_iter
    lr = lr or cfg.calibration.lr

    scaler = fit_temperature(logits, labels, max_iter=max_iter, lr=lr)

    out_path = Path(output_path) if output_path else cfg.calibration.output_path
    save_calibration(
        scaler,
        out_path,
        meta={
            "num_samples": int(len(labels)),
            "num_classes": label_map.num_classes,
            "val_csv": str(val_csv),
        },
    )
    return load_calibration(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calibrate MedicalXAI model with temperature scaling"
    )
    p.add_argument("--config", required=True, help="Path to train.yaml")
    p.add_argument("--output", default=None, help="Output path for calibration JSON")
    p.add_argument("--max-iter", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.01)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = calibrate(
        config_path=args.config,
        output_path=args.output,
        max_iter=args.max_iter,
        lr=args.lr,
    )
    print(f"Done. Temperature = {result['temperature']:.4f}")
