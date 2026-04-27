"""Metric computation: AUROC, AUPRC, F1, precision, recall, specificity."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.common.exceptions import MetricComputationError

logger = logging.getLogger("medicalxai.metrics")


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    class_names: Optional[list[str]] = None,
    prefix: str = "val",
) -> dict[str, float]:
    """Compute a comprehensive set of classification metrics.

    Args:
        logits: Raw model outputs, shape ``(N, C)``.
        targets: Ground-truth integer labels, shape ``(N,)``.
        num_classes: Number of classes.
        class_names: Optional list of human-readable class names (length C).
        prefix: Metric key prefix (e.g. ``"val"`` or ``"test"``).

    Returns:
        Flat dict of ``{f"{prefix}_{metric_name}": value}``.
    """
    if logits.shape[0] == 0:
        raise MetricComputationError("Empty logits tensor – cannot compute metrics.")

    probs = torch.softmax(logits, dim=1).cpu().numpy()        # (N, C)
    preds = np.argmax(probs, axis=1)                          # (N,)
    y_true = targets.cpu().numpy()                            # (N,)

    results: dict[str, float] = {}

    # ---- AUROC ----
    try:
        present = np.unique(y_true)
        if num_classes == 2:
            auroc = roc_auc_score(y_true, probs[:, 1])
        elif len(present) < 2:
            auroc = float("nan")
        elif len(present) < num_classes:
            # Macro average only over classes present in y_true
            per_class = [
                roc_auc_score((y_true == c).astype(int), probs[:, c])
                for c in present
                if (y_true == c).sum() > 0 and (y_true != c).sum() > 0
            ]
            auroc = float(np.mean(per_class)) if per_class else float("nan")
        else:
            auroc = roc_auc_score(
                y_true, probs, multi_class="ovr", average="macro"
            )
        results[f"{prefix}_auroc_macro"] = float(auroc)
    except Exception as exc:
        logger.warning("AUROC computation failed: %s", exc)
        results[f"{prefix}_auroc_macro"] = float("nan")

    # ---- AUPRC (macro) ----
    try:
        y_bin = _binarize(y_true, num_classes)
        auprc = float(np.mean([
            average_precision_score(y_bin[:, c], probs[:, c])
            for c in range(num_classes)
            if y_bin[:, c].sum() > 0
        ]))
        results[f"{prefix}_auprc_macro"] = auprc
    except Exception as exc:
        logger.warning("AUPRC computation failed: %s", exc)
        results[f"{prefix}_auprc_macro"] = float("nan")

    # ---- Precision / Recall / F1 ----
    zero_div = 0.0
    results[f"{prefix}_precision_macro"] = float(
        precision_score(y_true, preds, average="macro", zero_division=zero_div)
    )
    results[f"{prefix}_recall_macro"] = float(
        recall_score(y_true, preds, average="macro", zero_division=zero_div)
    )
    results[f"{prefix}_f1_macro"] = float(
        f1_score(y_true, preds, average="macro", zero_division=zero_div)
    )
    results[f"{prefix}_f1_weighted"] = float(
        f1_score(y_true, preds, average="weighted", zero_division=zero_div)
    )

    # ---- Specificity (macro) ----
    results[f"{prefix}_specificity_macro"] = _specificity_macro(
        y_true, preds, num_classes
    )

    # ---- Accuracy ----
    results[f"{prefix}_accuracy"] = float(np.mean(preds == y_true))

    # ---- Confusion matrix (serialised as nested list via JSON later) ----
    cm = confusion_matrix(y_true, preds, labels=list(range(num_classes)))
    results[f"{prefix}_confusion_matrix"] = cm.tolist()  # type: ignore[assignment]

    return results


def compute_loss_avg(total_loss: float, n_batches: int) -> float:
    """Return mean loss over *n_batches*, guarding against zero division."""
    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _binarize(y: np.ndarray, num_classes: int) -> np.ndarray:
    """One-hot encode integer labels → binary matrix (N, C)."""
    out = np.zeros((len(y), num_classes), dtype=np.float32)
    for i, label in enumerate(y):
        out[i, label] = 1.0
    return out


def _specificity_macro(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> float:
    """Macro-averaged specificity = TN / (TN + FP), one-vs-rest."""
    specs = []
    for c in range(num_classes):
        yt_bin = (y_true == c).astype(int)
        yp_bin = (y_pred == c).astype(int)
        tn = int(np.sum((yt_bin == 0) & (yp_bin == 0)))
        fp = int(np.sum((yt_bin == 0) & (yp_bin == 1)))
        denom = tn + fp
        specs.append(tn / denom if denom > 0 else 0.0)
    return float(np.mean(specs))


# ---------------------------------------------------------------------------
# Epoch aggregator
# ---------------------------------------------------------------------------


class MetricAccumulator:
    """Accumulate per-batch tensors then compute epoch-level metrics."""

    def __init__(self) -> None:
        self._logits: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []
        self._loss_sum: float = 0.0
        self._n_batches: int = 0

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss: float,
    ) -> None:
        self._logits.append(logits.detach().cpu())
        self._targets.append(targets.detach().cpu())
        self._loss_sum += loss
        self._n_batches += 1

    def compute(
        self,
        num_classes: int,
        class_names: Optional[list[str]] = None,
        prefix: str = "val",
    ) -> dict[str, float]:
        all_logits = torch.cat(self._logits, dim=0)
        all_targets = torch.cat(self._targets, dim=0)
        metrics = compute_metrics(
            all_logits, all_targets, num_classes, class_names, prefix
        )
        metrics[f"{prefix}_loss"] = compute_loss_avg(self._loss_sum, self._n_batches)
        return metrics

    def reset(self) -> None:
        self._logits.clear()
        self._targets.clear()
        self._loss_sum = 0.0
        self._n_batches = 0


# ---------------------------------------------------------------------------
# Multi-label metrics
# ---------------------------------------------------------------------------


def compute_multilabel_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_names: Optional[list[str]] = None,
    threshold: float = 0.5,
    prefix: str = "val",
) -> dict[str, float]:
    """Compute multi-label classification metrics.

    Args:
        logits:      Raw model outputs, shape ``(N, C)``.
        targets:     Ground-truth binary float labels, shape ``(N, C)``.
        class_names: Optional list of class names (length C).
        threshold:   Decision threshold for binary predictions (default 0.5).
        prefix:      Metric key prefix.

    Returns:
        Flat dict of metrics including per-class and macro AUROC, mAP.
    """
    if logits.shape[0] == 0:
        raise MetricComputationError("Empty logits tensor – cannot compute metrics.")

    probs = torch.sigmoid(logits).cpu().numpy()    # (N, C)
    preds = (probs >= threshold).astype(int)       # (N, C)
    y_true = targets.cpu().numpy().astype(int)     # (N, C)
    num_classes = probs.shape[1]

    results: dict[str, float] = {}

    # ---- Per-class and macro AUROC ----
    per_class_auroc: list[float] = []
    for c in range(num_classes):
        if y_true[:, c].sum() == 0 or y_true[:, c].sum() == len(y_true):
            per_class_auroc.append(float("nan"))
            continue
        try:
            auc = float(roc_auc_score(y_true[:, c], probs[:, c]))
        except Exception:
            auc = float("nan")
        per_class_auroc.append(auc)
        if class_names:
            results[f"{prefix}_auroc_{class_names[c].replace(' ', '_').lower()}"] = auc

    valid_aucs = [v for v in per_class_auroc if not np.isnan(v)]
    results[f"{prefix}_auroc_macro"] = float(np.mean(valid_aucs)) if valid_aucs else float("nan")

    # ---- Per-class and mean Average Precision (mAP) ----
    per_class_ap: list[float] = []
    for c in range(num_classes):
        if y_true[:, c].sum() == 0:
            continue
        try:
            ap = float(average_precision_score(y_true[:, c], probs[:, c]))
        except Exception:
            ap = float("nan")
        per_class_ap.append(ap)

    valid_aps = [v for v in per_class_ap if not np.isnan(v)]
    results[f"{prefix}_map"] = float(np.mean(valid_aps)) if valid_aps else float("nan")

    # ---- Sample-level (micro) F1, Precision, Recall ----
    zero_div = 0.0
    results[f"{prefix}_f1_micro"] = float(
        f1_score(y_true, preds, average="micro", zero_division=zero_div)
    )
    results[f"{prefix}_f1_macro"] = float(
        f1_score(y_true, preds, average="macro", zero_division=zero_div)
    )
    results[f"{prefix}_precision_micro"] = float(
        precision_score(y_true, preds, average="micro", zero_division=zero_div)
    )
    results[f"{prefix}_recall_micro"] = float(
        recall_score(y_true, preds, average="micro", zero_division=zero_div)
    )

    return results


class MultiLabelMetricAccumulator:
    """Accumulate per-batch logits and multi-label targets, then compute epoch metrics."""

    def __init__(self) -> None:
        self._logits: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []
        self._loss_sum: float = 0.0
        self._n_batches: int = 0

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss: float,
    ) -> None:
        self._logits.append(logits.detach().cpu())
        self._targets.append(targets.detach().cpu())
        self._loss_sum += loss
        self._n_batches += 1

    def compute(
        self,
        num_classes: Optional[int] = None,  # accepted but unused (multiclass compat)
        class_names: Optional[list[str]] = None,
        threshold: float = 0.5,
        prefix: str = "val",
    ) -> dict[str, float]:
        all_logits = torch.cat(self._logits, dim=0)
        all_targets = torch.cat(self._targets, dim=0)
        metrics = compute_multilabel_metrics(all_logits, all_targets, class_names, threshold, prefix)
        metrics[f"{prefix}_loss"] = compute_loss_avg(self._loss_sum, self._n_batches)
        return metrics

    def reset(self) -> None:
        self._logits.clear()
        self._targets.clear()
        self._loss_sum = 0.0
        self._n_batches = 0
