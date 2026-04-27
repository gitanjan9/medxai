"""MLflow helpers: run context, metric logging, and artifact upload."""
from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

import mlflow
import torch.nn as nn

from src.common.config import MLflowConfig, TrainConfig

logger = logging.getLogger("medicalxai.mlflow")


# ---------------------------------------------------------------------------
# Run context manager
# ---------------------------------------------------------------------------


@contextmanager
def mlflow_run(
    cfg: MLflowConfig,
    tags: Optional[dict[str, str]] = None,
) -> Generator[mlflow.ActiveRun, None, None]:
    """Context manager that starts (or resumes) an MLflow run.

    Sets the tracking URI and experiment before yielding the active run.
    Guarantees the run is properly ended even if an exception is raised.

    Usage::

        with mlflow_run(cfg.mlflow) as run:
            mlflow.log_metric("val_loss", 0.42, step=1)
    """
    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run(run_name=cfg.run_name, tags=tags or {}) as run:
        logger.info(
            "MLflow run started: id=%s  experiment=%s",
            run.info.run_id[:8],
            cfg.experiment_name,
        )
        yield run


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def log_config(cfg: TrainConfig) -> None:
    """Log all scalar config values as MLflow params."""
    flat = _flatten_dict(cfg.model_dump())
    for k, v in flat.items():
        if isinstance(v, (int, float, str, bool)):
            mlflow.log_param(k, v)


def log_epoch_metrics(
    metrics: dict[str, Any],
    step: int,
    exclude_keys: tuple[str, ...] = ("confusion_matrix",),
) -> None:
    """Log a metric dict for one epoch, skipping non-scalar entries."""
    scalar_metrics: dict[str, float] = {}
    for k, v in metrics.items():
        if any(ex in k for ex in exclude_keys):
            continue
        if isinstance(v, (int, float)) and not _is_nan_or_inf(v):
            scalar_metrics[k] = float(v)
    mlflow.log_metrics(scalar_metrics, step=step)


def log_confusion_matrix(cm: list[list[int]], epoch: int, artifact_dir: Path) -> None:
    """Persist confusion matrix JSON and upload as MLflow artifact."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"confusion_matrix_epoch_{epoch:03d}.json"
    with open(path, "w") as fh:
        json.dump(cm, fh, indent=2)
    mlflow.log_artifact(str(path), artifact_path="metrics")


def log_model_artifact(
    model: nn.Module,
    cfg: MLflowConfig,
    artifact_path: Optional[str] = None,
) -> None:
    """Log the PyTorch model as an MLflow artifact (state-dict only)."""
    import tempfile
    import torch

    with tempfile.TemporaryDirectory() as tmp:
        model_path = Path(tmp) / "model_state.pt"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(
            str(model_path),
            artifact_path=artifact_path or cfg.artifact_path,
        )
    logger.info("Logged model state dict to MLflow artifact store.")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _flatten_dict(d: dict, prefix: str = "") -> dict[str, Any]:
    """Recursively flatten a nested dict with dot-separated keys."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, full_key))
        else:
            out[full_key] = v
    return out


def _is_nan_or_inf(v: float) -> bool:
    import math
    return math.isnan(v) or math.isinf(v)
