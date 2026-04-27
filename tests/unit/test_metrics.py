"""Unit tests for src.train.metrics."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.train.metrics import (
    MetricAccumulator,
    _binarize,
    _specificity_macro,
    compute_metrics,
)


NUM_CLASSES = 4


def _random_logits(n: int = 50) -> torch.Tensor:
    return torch.randn(n, NUM_CLASSES)


def _random_targets(n: int = 50) -> torch.Tensor:
    return torch.randint(0, NUM_CLASSES, (n,))


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


def test_compute_metrics_keys():
    logits = _random_logits()
    targets = _random_targets()
    m = compute_metrics(logits, targets, NUM_CLASSES, prefix="val")
    expected_keys = [
        "val_auroc_macro",
        "val_auprc_macro",
        "val_precision_macro",
        "val_recall_macro",
        "val_f1_macro",
        "val_f1_weighted",
        "val_specificity_macro",
        "val_accuracy",
        "val_confusion_matrix",
    ]
    for k in expected_keys:
        assert k in m, f"Missing metric key: {k}"


def test_compute_metrics_ranges():
    logits = _random_logits()
    targets = _random_targets()
    m = compute_metrics(logits, targets, NUM_CLASSES, prefix="val")

    bounded = [
        "val_auroc_macro",
        "val_auprc_macro",
        "val_precision_macro",
        "val_recall_macro",
        "val_f1_macro",
        "val_specificity_macro",
        "val_accuracy",
    ]
    for k in bounded:
        v = m[k]
        if not np.isnan(v):
            assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"


def test_confusion_matrix_shape():
    logits = _random_logits()
    targets = _random_targets()
    m = compute_metrics(logits, targets, NUM_CLASSES, prefix="val")
    cm = m["val_confusion_matrix"]
    assert len(cm) == NUM_CLASSES
    assert all(len(row) == NUM_CLASSES for row in cm)


def test_compute_metrics_perfect():
    """Perfect predictions should give accuracy=1.0."""
    n = 40
    targets = torch.randint(0, NUM_CLASSES, (n,))
    # Build logits that argmax to targets
    logits = torch.full((n, NUM_CLASSES), -10.0)
    for i, t in enumerate(targets):
        logits[i, t] = 10.0
    m = compute_metrics(logits, targets, NUM_CLASSES, prefix="val")
    assert m["val_accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# MetricAccumulator
# ---------------------------------------------------------------------------


def test_accumulator_reset():
    acc = MetricAccumulator()
    acc.update(_random_logits(10), _random_targets(10), 0.5)
    acc.reset()
    acc.update(_random_logits(20), _random_targets(20), 0.3)
    m = acc.compute(NUM_CLASSES, prefix="val")
    assert "val_loss" in m


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_binarize():
    y = np.array([0, 1, 2, 3])
    out = _binarize(y, 4)
    assert out.shape == (4, 4)
    assert out[0, 0] == 1.0
    assert out[1, 1] == 1.0
    assert out.sum() == 4


def test_specificity_macro_perfect():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    spec = _specificity_macro(y_true, y_pred, 3)
    assert spec == pytest.approx(1.0)
