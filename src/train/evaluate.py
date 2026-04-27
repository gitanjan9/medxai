"""Evaluation entry point – runs a full pass over a dataset and reports metrics.

Usage::

    python -m src.train.evaluate --config configs/train.yaml
    python -m src.train.evaluate --config configs/train.yaml \\
        --checkpoint artifacts/checkpoints/best.pt \\
        --data-csv /path/to/test.csv
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.common.config import TrainConfig
from src.common.logging import setup_logging, get_logger
from src.common.schemas import LabelMap
from src.common.utils import ensure_dir, get_device, set_reproducibility
from src.train.dataset import CXRDataset, build_label_map_from_csv, load_and_prepare_dataframe
from src.train.metrics import MetricAccumulator
from src.train.model_factory import build_model
from src.train.transforms import build_val_transforms

logger = get_logger("evaluate")


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------


def evaluate(
    config_path: str,
    checkpoint_path: Optional[str] = None,
    data_csv: Optional[str] = None,
    split_name: str = "test",
    output_dir: Optional[str] = None,
) -> dict[str, float]:
    """Load a checkpoint and evaluate it on a dataset split.

    Args:
        config_path: Path to ``train.yaml``.
        checkpoint_path: Path to a ``.pt`` checkpoint file.  If ``None``,
            falls back to the most recent checkpoint in the config's dir.
        data_csv: Override CSV path (uses config's ``test_path`` if None).
        split_name: Prefix for metric keys (e.g. ``"test"`` or ``"val"``).
        output_dir: Where to write ``metrics.json``.  Defaults to config
            ``output_dir``.

    Returns:
        Dict of metric name → value.
    """
    cfg = TrainConfig.from_yaml(config_path)

    log_dir = ensure_dir(cfg.experiment.output_dir / "logs")
    setup_logging(config_path=Path("configs/logging.yaml"), log_dir=log_dir)
    logger.info("Starting evaluation: split=%s", split_name)

    set_reproducibility(cfg.experiment.seed)
    device = get_device()

    # ---- Label map ----
    lm_path = cfg.data.label_mapping_path
    if lm_path and Path(lm_path).exists():
        label_map = LabelMap.load(Path(lm_path))
    else:
        label_map = build_label_map_from_csv(
            cfg.data.train_path, label_col=cfg.data.label_col
        )
    cfg.model.num_classes = label_map.num_classes

    # ---- Dataset ----
    csv_path = Path(data_csv) if data_csv else (
        cfg.data.test_path or cfg.data.val_path or cfg.data.train_path
    )
    if csv_path is None:
        raise ValueError(
            "No evaluation CSV specified. Pass --data-csv or set data.test_path in config."
        )

    df = load_and_prepare_dataframe(
        csv_path, cfg.data.label_col, label_map,
        merge_map_path=cfg.data.class_merge_map_path,
        tag="Evaluation",
    )
    tfm = build_val_transforms(cfg.data.image_size)
    dataset = CXRDataset(
        df=df,
        label_map=label_map,
        image_col=cfg.data.image_col,
        label_col=cfg.data.label_col,
        text_col=cfg.data.text_col,
        transforms=tfm,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size * 2,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and device.type == "cuda",
    )
    logger.info("Evaluating on %d samples from %s", len(dataset), csv_path)

    # ---- Model ----
    model = build_model(cfg.model).to(device)
    ckpt_path = _resolve_checkpoint(checkpoint_path, cfg)
    if ckpt_path:
        _load_checkpoint(model, ckpt_path, device)
    else:
        logger.warning("No checkpoint loaded – evaluating randomly initialised model.")

    # ---- Evaluation pass ----
    model.eval()
    accumulator = MetricAccumulator()
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            accumulator.update(logits, labels, loss.item())

    metrics = accumulator.compute(
        num_classes=label_map.num_classes,
        class_names=label_map.class_names(),
        prefix=split_name,
    )

    # ---- Report ----
    _print_metrics(metrics, split_name)

    # ---- Save ----
    out_dir = Path(output_dir) if output_dir else ensure_dir(cfg.experiment.output_dir)
    metrics_path = out_dir / f"{split_name}_metrics.json"
    _save_metrics(metrics, metrics_path)
    logger.info("Metrics written to %s", metrics_path)

    return metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_checkpoint(
    explicit: Optional[str], cfg: TrainConfig
) -> Optional[Path]:
    """Return explicit path, or the best `.pt` in the checkpoint dir by metric value."""
    if explicit:
        return Path(explicit)
    ckpt_dir = cfg.checkpoint.dir
    if not Path(ckpt_dir).exists():
        return None
    pts = list(Path(ckpt_dir).glob("*.pt"))
    if not pts:
        return None
    # Try to parse metric from filename, e.g. "epoch=016_val_auroc_macro=0.6279.pt"
    import re
    _metric_re = re.compile(r"=(-?[\d.]+)\.pt$")
    def _score(p: Path) -> float:
        m = _metric_re.search(p.name)
        return float(m.group(1)) if m else p.stat().st_mtime
    mode = cfg.checkpoint.mode  # "max" or "min"
    pts_sorted = sorted(pts, key=_score, reverse=(mode == "max"))
    return pts_sorted[0]


def _load_checkpoint(
    model: torch.nn.Module, path: Path, device: torch.device
) -> None:
    """Load model state dict from a checkpoint file."""
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(
        "Loaded checkpoint: %s  (missing=%d  unexpected=%d)",
        path.name, len(missing), len(unexpected),
    )


def _print_metrics(metrics: dict, split_name: str) -> None:
    """Log all scalar metrics in a readable table."""
    logger.info("=" * 55)
    logger.info("Evaluation results  [%s]", split_name.upper())
    logger.info("-" * 55)
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            logger.info("  %-45s %.4f", k, v)
    logger.info("=" * 55)


def _save_metrics(metrics: dict, path: Path) -> None:
    """Serialise metrics dict to JSON (skip non-serialisable types)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clean: dict = {}
    for k, v in metrics.items():
        if isinstance(v, float):
            clean[k] = round(v, 6)
        elif isinstance(v, list):
            clean[k] = v
    with open(path, "w") as fh:
        json.dump(clean, fh, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a MedicalXAI checkpoint")
    p.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to train.yaml",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .pt checkpoint (defaults to latest in checkpoint dir)",
    )
    p.add_argument(
        "--data-csv",
        type=str,
        default=None,
        help="Override evaluation CSV path",
    )
    p.add_argument(
        "--split-name",
        type=str,
        default="test",
        help="Prefix for metric keys in output (default: 'test')",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write metrics.json (default: experiment.output_dir)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        data_csv=args.data_csv,
        split_name=args.split_name,
        output_dir=args.output_dir,
    )
