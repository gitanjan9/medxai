"""Per-class threshold optimization for false-positive reduction.

For each class we find two thresholds:
  * ``high`` – minimum confidence to call "positive"  (PPV ≥ target)
  * ``low``  – maximum confidence to call "negative"  (recall ≥ target)
  * Between  → case sent to manual "review" band

Usage::

    python -m src.train.thresholds --config configs/train.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from src.common.logging import get_logger

logger = get_logger("thresholds")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class ReviewBand(str, Enum):
    NEGATIVE = "negative"
    REVIEW = "review"
    POSITIVE = "positive"


@dataclass
class ClassThreshold:
    class_idx: int
    class_name: str
    low: float    # cases below this → NEGATIVE
    high: float   # cases at or above this → POSITIVE
    ppv_at_high: float
    recall_at_low: float


# ---------------------------------------------------------------------------
# Per-class helpers
# ---------------------------------------------------------------------------


def _ppv_recall_at(
    probs_c: np.ndarray,
    true_binary: np.ndarray,
    threshold: float,
) -> tuple[float, float]:
    """(PPV, recall) for one class at *threshold*."""
    pred = probs_c >= threshold
    tp = (pred & true_binary).sum()
    fp = (pred & ~true_binary).sum()
    fn = (~pred & true_binary).sum()
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(ppv), float(recall)


def optimize_class_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    class_idx: int,
    class_name: str,
    target_ppv: float = 0.65,
    target_recall: float = 0.70,
    default_low: float = 0.25,
    default_high: float = 0.65,
    n_steps: int = 200,
) -> ClassThreshold:
    """Find optimal (low, high) pair for *class_idx*.

    Sweep strategy:
    - high: smallest threshold t s.t. PPV(t) >= target_ppv
    - low:  largest threshold  t s.t. recall(t) >= target_recall
    - Clamp low < high to guarantee a non-empty positive band.
    """
    true_binary = labels == class_idx
    probs_c = probs[:, class_idx]

    if true_binary.sum() == 0:
        logger.warning(
            "Class %d (%s) has no positive examples – using defaults.",
            class_idx, class_name,
        )
        return ClassThreshold(
            class_idx=class_idx,
            class_name=class_name,
            low=default_low,
            high=default_high,
            ppv_at_high=float("nan"),
            recall_at_low=float("nan"),
        )

    thresholds = np.linspace(0.01, 0.99, n_steps)

    # High threshold: smallest t where PPV >= target_ppv
    high, best_ppv = default_high, 0.0
    for t in reversed(thresholds):
        ppv, _ = _ppv_recall_at(probs_c, true_binary, t)
        if ppv >= target_ppv:
            high, best_ppv = float(t), ppv
            break

    # Low threshold: largest t where recall >= target_recall
    low, best_recall = default_low, 0.0
    for t in thresholds:
        _, recall = _ppv_recall_at(probs_c, true_binary, t)
        if recall >= target_recall:
            low, best_recall = float(t), recall

    # Guarantee low < high
    if low >= high:
        low = max(high * 0.5, 0.01)

    logger.debug(
        "Class %2d %-55s  low=%.3f (recall=%.3f)  high=%.3f (ppv=%.3f)",
        class_idx, class_name, low, best_recall, high, best_ppv,
    )
    return ClassThreshold(
        class_idx=class_idx,
        class_name=class_name,
        low=low,
        high=high,
        ppv_at_high=best_ppv,
        recall_at_low=best_recall,
    )


def optimize_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    target_ppv: float = 0.65,
    target_recall: float = 0.70,
    default_low: float = 0.25,
    default_high: float = 0.65,
) -> list[ClassThreshold]:
    """Optimize thresholds for all classes. Returns list indexed by class."""
    results = []
    for idx, name in enumerate(class_names):
        ct = optimize_class_threshold(
            probs, labels, idx, name,
            target_ppv=target_ppv,
            target_recall=target_recall,
            default_low=default_low,
            default_high=default_high,
        )
        results.append(ct)
    return results


# ---------------------------------------------------------------------------
# Decision function (reusable at serving time)
# ---------------------------------------------------------------------------


def apply_threshold(prob: float, low: float, high: float) -> ReviewBand:
    """Map a single class probability to a ReviewBand decision."""
    if prob >= high:
        return ReviewBand.POSITIVE
    if prob < low:
        return ReviewBand.NEGATIVE
    return ReviewBand.REVIEW


def apply_thresholds_batch(
    probs: np.ndarray,
    thresholds: list[dict],
) -> list[dict]:
    """Apply per-class thresholds to a batch of probability vectors.

    Args:
        probs:      Shape (N, C) probability matrix.
        thresholds: List of threshold dicts (from loaded JSON artifact).

    Returns:
        List of N dicts with keys: ``predicted_class``, ``predicted_label``,
        ``band``, ``confidence``, ``all_bands``.
    """
    results = []
    for row in probs:
        top_idx = int(np.argmax(row))
        top_conf = float(row[top_idx])
        th = thresholds[top_idx]
        band = apply_threshold(top_conf, th["low"], th["high"])
        all_bands = {
            thresholds[i]["class_name"]: apply_threshold(
                float(row[i]), thresholds[i]["low"], thresholds[i]["high"]
            ).value
            for i in range(len(thresholds))
        }
        results.append(
            {
                "predicted_class": top_idx,
                "predicted_label": th["class_name"],
                "band": band.value,
                "confidence": top_conf,
                "all_bands": all_bands,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Artifact I/O
# ---------------------------------------------------------------------------


def save_thresholds(
    thresholds: list[ClassThreshold],
    output_path: Path,
    meta: Optional[dict] = None,
) -> None:
    """Persist threshold artifact as JSON."""
    artifact = {
        "method": "ppv_recall_balanced",
        **(meta or {}),
        "thresholds": [asdict(ct) for ct in thresholds],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(artifact, fh, indent=2)
    logger.info("Thresholds saved → %s", output_path)


def load_thresholds(path: Path) -> dict:
    """Load threshold artifact from JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Threshold artifact not found: {path}")
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def tune_thresholds(
    config_path: str,
    output_path: Optional[str] = None,
    target_ppv: float = 0.65,
    target_recall: float = 0.70,
) -> dict:
    """Collect val-set probabilities, optimize per-class thresholds, save artifact."""
    from src.common.config import TrainConfig
    from src.common.logging import setup_logging
    from src.common.schemas import LabelMap
    from src.common.utils import get_device
    from src.train.calibrate import collect_logits, load_calibration, apply_temperature
    from src.train.dataset import CXRDataset, load_and_prepare_dataframe
    from src.train.model_factory import build_model
    from src.train.transforms import build_val_transforms
    from src.train.evaluate import _resolve_checkpoint, _load_checkpoint
    from torch.utils.data import DataLoader

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
        tag="Thresholds",
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

    # Model + checkpoint
    model = build_model(cfg.model).to(device)
    ckpt = _resolve_checkpoint(None, cfg)
    if ckpt:
        _load_checkpoint(model, ckpt, device)

    logits, labels = collect_logits(model, loader, device)

    # Apply calibration if available
    cal_path = cfg.calibration.output_path
    if Path(cal_path).exists():
        cal = load_calibration(cal_path)
        probs = apply_temperature(logits, cal["temperature"])
        logger.info("Using calibrated probabilities (T=%.4f)", cal["temperature"])
    else:
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        logger.info("No calibration found – using raw softmax")

    # Optimize
    thresholds = optimize_thresholds(
        probs, labels, label_map.class_names(),
        target_ppv=target_ppv,
        target_recall=target_recall,
        default_low=cfg.thresholds.default_low,
        default_high=cfg.thresholds.default_high,
    )

    out_path = Path(output_path) if output_path else cfg.thresholds.output_path
    save_thresholds(
        thresholds,
        out_path,
        meta={
            "target_ppv": target_ppv,
            "target_recall": target_recall,
            "num_samples": int(len(labels)),
            "val_csv": str(val_csv),
        },
    )
    return load_thresholds(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-class threshold tuning for false-positive reduction"
    )
    p.add_argument("--config", required=True, help="Path to train.yaml")
    p.add_argument("--output", default=None, help="Output path for thresholds JSON")
    p.add_argument("--target-ppv", type=float, default=0.65)
    p.add_argument("--target-recall", type=float, default=0.70)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = tune_thresholds(
        config_path=args.config,
        output_path=args.output,
        target_ppv=args.target_ppv,
        target_recall=args.target_recall,
    )
    print(f"Done. Thresholds written for {len(result['thresholds'])} classes.")
