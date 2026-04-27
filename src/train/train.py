"""Main training entry point.

Usage::

    python -m src.train.train --config configs/train.yaml
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

import mlflow

from src.common.config import TrainConfig
from src.common.logging import setup_logging, get_logger
from src.common.utils import (
    count_parameters,
    device_info,
    ensure_dir,
    get_device,
    save_json,
    set_reproducibility,
)
from src.train.dataset import (
    CXRDataset,
    MultiLabelCXRDataset,
    build_datasets,
    build_multilabel_datasets,
    build_label_map_from_csv,
)
from src.train.losses import build_criterion
from src.train.metrics import MetricAccumulator, MultiLabelMetricAccumulator
from src.train.mlflow_utils import (
    log_config,
    log_confusion_matrix,
    log_epoch_metrics,
    log_model_artifact,
    mlflow_run,
)
from src.train.model_factory import build_model
from src.train.transforms import build_train_transforms, build_val_transforms
from src.train.zoo_bootstrap import bootstrap_or_build

logger = get_logger("train")


# ---------------------------------------------------------------------------
# Training loop helpers
# ---------------------------------------------------------------------------


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    grad_clip: float,
    accumulator: MetricAccumulator,
) -> dict[str, float]:
    """Run one training epoch; return per-step mean loss dict."""
    model.train()
    accumulator.reset()
    step_losses: list[float] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        step_losses.append(loss.item())
        accumulator.update(logits.detach(), labels.detach(), loss.item())

    return {"train_loss": sum(step_losses) / max(len(step_losses), 1)}


@torch.no_grad()
def _validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    class_names: list[str],
    accumulator: MetricAccumulator,
) -> dict[str, float]:
    """Run validation; return full metric dict."""
    model.eval()
    accumulator.reset()

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        accumulator.update(logits, labels, loss.item())

    return accumulator.compute(
        num_classes=num_classes,
        class_names=class_names,
        prefix="val",
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


class CheckpointManager:
    """Maintains a top-k checkpoint queue, keeps the best *k* by metric."""

    def __init__(
        self,
        ckpt_dir: Path,
        save_top_k: int = 3,
        monitor: str = "val_auroc_macro",
        mode: str = "max",
    ) -> None:
        self.ckpt_dir = ensure_dir(ckpt_dir)
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.sign = 1.0 if mode == "max" else -1.0
        self._queue: list[tuple[float, Path]] = []  # (signed_score, path)

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict[str, float],
    ) -> Path:
        """Save checkpoint and prune the oldest if top-k is exceeded."""
        score = metrics.get(self.monitor, 0.0)
        fname = (
            f"epoch={epoch:03d}"
            f"_{self.monitor}={score:.4f}.pt"
        )
        path = self.ckpt_dir / fname

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
            },
            path,
        )
        logger.info("Saved checkpoint: %s", path.name)

        self._queue.append((self.sign * score, path))
        self._queue.sort(key=lambda x: x[0], reverse=True)

        while len(self._queue) > self.save_top_k:
            _, old_path = self._queue.pop()
            if old_path.exists():
                old_path.unlink()
                logger.debug("Pruned old checkpoint: %s", old_path.name)

        return path

    def best_path(self) -> Optional[Path]:
        return self._queue[0][1] if self._queue else None


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    def __init__(
        self,
        patience: int = 7,
        mode: str = "max",
        min_delta: float = 1e-4,
    ) -> None:
        self.patience = patience
        self.sign = 1.0 if mode == "max" else -1.0
        self.min_delta = min_delta
        self.best: float = -float("inf")
        self.counter: int = 0
        self.triggered: bool = False

    def step(self, value: float) -> bool:
        """Call once per epoch. Returns True when training should stop."""
        scaled = self.sign * value
        if scaled > self.best + self.min_delta:
            self.best = scaled
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ---------------------------------------------------------------------------
# Optimizer / scheduler builders
# ---------------------------------------------------------------------------


def _set_backbone_requires_grad(
    model: nn.Module,
    architecture: str,
    requires_grad: bool,
) -> None:
    """Freeze or unfreeze backbone, leaving classification head always trainable."""
    from src.train.model_factory import _is_head_param
    for name, param in model.named_parameters():
        if _is_head_param(name, architecture):
            param.requires_grad = True  # head always trainable
        else:
            param.requires_grad = requires_grad
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.debug(
        "_set_backbone_requires_grad(%s): trainable params = %d", requires_grad, trainable
    )


def _build_optimizer(
    cfg: TrainConfig, model: nn.Module
) -> torch.optim.Optimizer:
    """Build optimizer with optional differential LR (backbone vs head).

    When ``backbone_lr_scale < 1.0`` two parameter groups are created:
    - backbone: ``lr * backbone_lr_scale``
    - head:     ``lr`` (full learning rate)
    """
    from src.train.model_factory import _is_head_param
    arch = cfg.model.architecture
    scale = cfg.training.backbone_lr_scale
    lr = cfg.training.learning_rate

    backbone_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and not _is_head_param(n, arch)
    ]
    head_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and _is_head_param(n, arch)
    ]

    if backbone_params and scale != 1.0:
        param_groups = [
            {"params": backbone_params, "lr": lr * scale, "name": "backbone"},
            {"params": head_params,     "lr": lr,         "name": "head"},
        ]
        logger.info(
            "Differential LR: backbone=%.2e  head=%.2e", lr * scale, lr
        )
    else:
        param_groups = filter(lambda p: p.requires_grad, model.parameters())

    name = cfg.optimizer.name
    if name == "adamw":
        return torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=cfg.training.weight_decay,
            betas=tuple(cfg.optimizer.betas),
            eps=cfg.optimizer.eps,
        )
    elif name == "adam":
        return torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=cfg.training.weight_decay,
            betas=tuple(cfg.optimizer.betas),
            eps=cfg.optimizer.eps,
        )
    elif name == "sgd":
        return torch.optim.SGD(
            param_groups,
            lr=lr,
            weight_decay=cfg.training.weight_decay,
            momentum=cfg.optimizer.momentum,
            nesterov=True,
        )
    raise ValueError(f"Unknown optimizer: {name}")


def _build_scheduler(
    cfg: TrainConfig, optimizer: torch.optim.Optimizer
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    name = cfg.scheduler.name
    if name == "none":
        return None
    elif name == "cosine_annealing_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.scheduler.T_0,
            T_mult=cfg.scheduler.T_mult,
            eta_min=cfg.scheduler.eta_min,
        )
    elif name == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.scheduler.T_max,
            eta_min=cfg.scheduler.eta_min,
        )
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.scheduler.step_size,
            gamma=cfg.scheduler.gamma,
        )
    raise ValueError(f"Unknown scheduler: {name}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(config_path: str) -> None:
    """Load config, build all components, and run the training loop."""
    # ---- Config ----
    cfg = TrainConfig.from_yaml(config_path)

    # ---- Logging ----
    log_dir = ensure_dir(cfg.experiment.output_dir / "logs")
    setup_logging(
        config_path=Path("configs/logging.yaml"),
        log_dir=log_dir,
    )
    logger.info("Starting training: %s", cfg.experiment.name)
    logger.info("Config: %s", config_path)
    logger.info("Task: %s", cfg.data.task)

    # ---- Reproducibility ----
    set_reproducibility(cfg.experiment.seed)
    device = get_device()
    logger.info("Device info: %s", device_info())

    # ---- Zoo bootstrap ----
    bootstrap_or_build(
        bundle_root=cfg.bundle.bundle_root,
        task="cxr_classification",
    )

    is_multilabel = cfg.data.task == "multilabel"

    # ---- Transforms ----
    train_tfm = build_train_transforms(cfg.data.image_size)
    val_tfm = build_val_transforms(cfg.data.image_size)

    if is_multilabel:
        # ------------------------------------------------------------------
        # Multi-label path
        # ------------------------------------------------------------------
        from src.data.label_extractor import PATHOLOGY_CLASSES
        label_cols = cfg.data.label_cols or PATHOLOGY_CLASSES
        cfg.model.num_classes = len(label_cols)
        logger.info("Multilabel classes (%d): %s", len(label_cols), label_cols)

        train_ds_ml, val_ds_ml = build_multilabel_datasets(
            csv_path=cfg.data.train_path,
            label_cols=label_cols,
            val_csv_path=cfg.data.val_path,
            val_split=cfg.data.val_split,
            image_col=cfg.data.image_col,
            train_transforms=train_tfm,
            val_transforms=val_tfm,
            seed=cfg.experiment.seed,
        )
        logger.info("Train: %d samples | Val: %d samples", len(train_ds_ml), len(val_ds_ml))

        class_weights: Optional[torch.Tensor] = train_ds_ml.class_pos_weights().to(device)
        logger.info("Pos weights (first 5): %s", class_weights[:5].tolist())

        train_loader = DataLoader(
            train_ds_ml,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory and device.type == "cuda",
        )
        val_loader = DataLoader(
            val_ds_ml,
            batch_size=cfg.training.batch_size * 2,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory and device.type == "cuda",
        )

        train_acc = MultiLabelMetricAccumulator()
        val_acc = MultiLabelMetricAccumulator()
        class_names_for_metrics: Optional[list[str]] = label_cols

    else:
        # ------------------------------------------------------------------
        # Multi-class path (unchanged from v2)
        # ------------------------------------------------------------------
        from src.common.schemas import LabelMap
        from src.train.dataset import load_and_prepare_dataframe
        merge_map_path = cfg.data.class_merge_map_path
        label_map_path = cfg.data.label_mapping_path
        if label_map_path and Path(label_map_path).exists() and not merge_map_path:
            label_map = LabelMap.load(label_map_path)
            logger.info("Loaded cached label map: %d classes from %s", label_map.num_classes, label_map_path)
        else:
            label_source_df = load_and_prepare_dataframe(
                cfg.data.label_map_csv or cfg.data.train_path,
                cfg.data.label_col,
                merge_map_path=merge_map_path,
                tag="LabelMapSource",
            )
            label_map = LabelMap.from_labels(label_source_df[cfg.data.label_col].astype(str).tolist())
            if label_map_path:
                label_map.save(Path(label_map_path))
                logger.info("Saved merged label map (%d classes) → %s", label_map.num_classes, label_map_path)
            logger.info("Classes after merge: %s", label_map.class_names())

        cfg.model.num_classes = label_map.num_classes

        train_ds, val_ds = build_datasets(
            train_path=cfg.data.train_path,
            label_map=label_map,
            val_path=cfg.data.val_path,
            val_split=cfg.data.val_split,
            image_col=cfg.data.image_col,
            label_col=cfg.data.label_col,
            text_col=cfg.data.text_col,
            train_transforms=train_tfm,
            val_transforms=val_tfm,
            seed=cfg.experiment.seed,
            merge_map_path=merge_map_path,
        )
        logger.info("Train: %d samples | Val: %d samples", len(train_ds), len(val_ds))

        class_weights = None
        if cfg.training.class_balance_strategy in ("weighted_loss", "focal", "weighted_sampler"):
            class_weights = train_ds.compute_class_weights().to(device)
            logger.info("Class weights: %s", class_weights.tolist())

        use_sampler = cfg.training.class_balance_strategy == "weighted_sampler"
        if use_sampler:
            sampler = train_ds.make_weighted_sampler()
            train_loader = DataLoader(
                train_ds,
                batch_size=cfg.training.batch_size,
                sampler=sampler,
                num_workers=cfg.data.num_workers,
                pin_memory=cfg.data.pin_memory and device.type == "cuda",
            )
            logger.info("Using WeightedRandomSampler for training")
        else:
            train_loader = DataLoader(
                train_ds,
                batch_size=cfg.training.batch_size,
                shuffle=True,
                num_workers=cfg.data.num_workers,
                pin_memory=cfg.data.pin_memory and device.type == "cuda",
            )

        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.training.batch_size * 2,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory and device.type == "cuda",
        )

        train_acc = MetricAccumulator()
        val_acc = MetricAccumulator()
        class_names_for_metrics = label_map.class_names()

    # ---- Model ----
    model = build_model(cfg.model).to(device)
    logger.info("Model params: %s", count_parameters(model))

    # ---- Loss / optimiser / scheduler ----
    criterion = build_criterion(cfg.training, class_weights)
    optimizer = _build_optimizer(cfg, model)
    scheduler = _build_scheduler(cfg, optimizer)

    # ---- AMP ----
    scaler: Optional[torch.cuda.amp.GradScaler] = (
        torch.cuda.amp.GradScaler()
        if cfg.training.use_amp and device.type == "cuda"
        else None
    )
    logger.info("AMP enabled: %s", scaler is not None)

    # ---- Checkpoint manager ----
    ckpt_mgr = CheckpointManager(
        ckpt_dir=cfg.checkpoint.dir,
        save_top_k=cfg.checkpoint.save_top_k,
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
    )

    # ---- Early stopping ----
    stopper = EarlyStopping(
        patience=cfg.early_stopping.patience,
        mode=cfg.early_stopping.mode,
        min_delta=cfg.early_stopping.min_delta,
    ) if cfg.early_stopping.enabled else None

    # ---- Resolve dataset size and num_classes for MLflow logging ----
    n_train = len(train_loader.dataset)  # type: ignore[arg-type]
    n_val = len(val_loader.dataset)      # type: ignore[arg-type]
    num_classes = cfg.model.num_classes

    # ---- MLflow ----
    with mlflow_run(
        cfg.mlflow,
        tags={"experiment": cfg.experiment.name, "task": cfg.data.task},
    ) as run:
        log_config(cfg)
        mlflow.log_param("num_train_samples", n_train)
        mlflow.log_param("num_val_samples", n_val)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("task", cfg.data.task)
        mlflow.log_param("device", str(device))

        cm_dir = ensure_dir(cfg.experiment.output_dir / "confusion_matrices")
        best_metric: float = -float("inf") if cfg.checkpoint.mode == "max" else float("inf")

        # Two-phase training: freeze backbone for first freeze_epochs epochs
        freeze_epochs = cfg.training.freeze_epochs
        if freeze_epochs > 0:
            _set_backbone_requires_grad(model, cfg.model.architecture, requires_grad=False)
            logger.info("Backbone frozen for first %d epochs (head-only training)", freeze_epochs)

        for epoch in range(1, cfg.training.epochs + 1):
            t0 = time.perf_counter()

            # Unfreeze backbone after freeze_epochs
            if freeze_epochs > 0 and epoch == freeze_epochs + 1:
                _set_backbone_requires_grad(model, cfg.model.architecture, requires_grad=True)
                logger.info("Epoch %d: backbone unfrozen – full fine-tuning begins", epoch)

            # Train
            train_step_metrics = _train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                scaler, cfg.training.grad_clip_norm, train_acc,
            )

            # Val
            val_metrics = _validate(
                model, val_loader, criterion, device,
                num_classes, class_names_for_metrics, val_acc,
            )

            elapsed = time.perf_counter() - t0
            monitor_val = val_metrics.get(cfg.early_stopping.monitor, 0.0)

            logger.info(
                "Epoch %3d/%d | loss=%.4f | %s=%.4f | lr=%.2e | %.1fs",
                epoch, cfg.training.epochs,
                train_step_metrics["train_loss"],
                cfg.early_stopping.monitor, monitor_val,
                optimizer.param_groups[-1]["lr"],  # head LR (last group)
                elapsed,
            )

            # Log to MLflow
            combined = {**train_step_metrics, **val_metrics}
            log_epoch_metrics(combined, step=epoch)
            mlflow.log_metric("lr", optimizer.param_groups[-1]["lr"], step=epoch)  # head LR

            if "val_confusion_matrix" in val_metrics:
                log_confusion_matrix(val_metrics["val_confusion_matrix"], epoch, cm_dir)

            # Checkpoint
            ckpt_mgr.save(model, optimizer, epoch, val_metrics)

            # Scheduler step
            if scheduler is not None:
                if isinstance(
                    scheduler,
                    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
                ):
                    scheduler.step(epoch - 1)
                else:
                    scheduler.step()

            # Early stopping
            if stopper is not None and stopper.step(monitor_val):
                logger.info(
                    "Early stopping triggered at epoch %d (patience=%d).",
                    epoch, cfg.early_stopping.patience,
                )
                break

        # ---- Final: log best model ----
        best_path = ckpt_mgr.best_path()
        if best_path is not None and cfg.mlflow.log_model_artifact:
            mlflow.log_artifact(str(best_path), artifact_path="checkpoints")
            logger.info("Best checkpoint logged: %s", best_path.name)

        logger.info("Training complete. Run ID: %s", run.info.run_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MedicalXAI classifier")
    p.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to train.yaml config file",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(args.config)
