"""Unit tests for src/train/thresholds.py"""
import json
from pathlib import Path

import numpy as np
import pytest

from src.train.thresholds import (
    ClassThreshold,
    ReviewBand,
    apply_threshold,
    apply_thresholds_batch,
    load_thresholds,
    optimize_class_threshold,
    optimize_thresholds,
    save_thresholds,
)


# ---------------------------------------------------------------------------
# apply_threshold
# ---------------------------------------------------------------------------


def test_apply_threshold_positive():
    assert apply_threshold(0.90, low=0.30, high=0.70) == ReviewBand.POSITIVE


def test_apply_threshold_negative():
    assert apply_threshold(0.10, low=0.30, high=0.70) == ReviewBand.NEGATIVE


def test_apply_threshold_review():
    assert apply_threshold(0.50, low=0.30, high=0.70) == ReviewBand.REVIEW


def test_apply_threshold_boundary_high():
    # Exactly at high → POSITIVE
    assert apply_threshold(0.70, low=0.30, high=0.70) == ReviewBand.POSITIVE


def test_apply_threshold_boundary_low():
    # Exactly at low → REVIEW (not NEGATIVE)
    assert apply_threshold(0.30, low=0.30, high=0.70) == ReviewBand.REVIEW


# ---------------------------------------------------------------------------
# optimize_class_threshold
# ---------------------------------------------------------------------------


def _make_probs_labels(n=200, n_classes=4, class_idx=0, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, size=n)
    probs = rng.dirichlet(np.ones(n_classes), size=n).astype(np.float32)
    # Boost target class probability for positive examples
    mask = labels == class_idx
    probs[mask, class_idx] = rng.uniform(0.6, 0.95, size=mask.sum())
    probs[mask] /= probs[mask].sum(axis=1, keepdims=True)
    return probs, labels


def test_optimize_class_threshold_low_lt_high():
    probs, labels = _make_probs_labels()
    ct = optimize_class_threshold(probs, labels, 0, "class_0")
    assert ct.low < ct.high, f"Expected low < high, got low={ct.low} high={ct.high}"


def test_optimize_class_threshold_no_positives():
    probs, labels = _make_probs_labels()
    labels[:] = 1  # class 0 has 0 positives
    ct = optimize_class_threshold(probs, labels, 0, "empty_class",
                                  default_low=0.25, default_high=0.65)
    assert ct.low == 0.25
    assert ct.high == 0.65
    assert np.isnan(ct.ppv_at_high)


def test_optimize_class_threshold_range():
    probs, labels = _make_probs_labels()
    ct = optimize_class_threshold(probs, labels, 0, "class_0")
    assert 0.0 < ct.low <= 1.0
    assert 0.0 < ct.high <= 1.0


# ---------------------------------------------------------------------------
# optimize_thresholds
# ---------------------------------------------------------------------------


def test_optimize_thresholds_all_classes():
    n_classes = 4
    rng = np.random.default_rng(1)
    labels = rng.integers(0, n_classes, 300)
    probs = rng.dirichlet(np.ones(n_classes), 300).astype(np.float32)
    names = [f"cls_{i}" for i in range(n_classes)]
    thresholds = optimize_thresholds(probs, labels, names)
    assert len(thresholds) == n_classes
    for ct in thresholds:
        assert isinstance(ct, ClassThreshold)
        assert ct.low < ct.high


# ---------------------------------------------------------------------------
# apply_thresholds_batch
# ---------------------------------------------------------------------------


def test_apply_thresholds_batch_shape():
    n_classes = 3
    thresh_dicts = [
        {"class_idx": i, "class_name": f"c{i}", "low": 0.25, "high": 0.65,
         "ppv_at_high": 0.7, "recall_at_low": 0.8}
        for i in range(n_classes)
    ]
    rng = np.random.default_rng(2)
    probs = rng.dirichlet(np.ones(n_classes), 10).astype(np.float32)
    results = apply_thresholds_batch(probs, thresh_dicts)
    assert len(results) == 10
    for r in results:
        assert r["band"] in {b.value for b in ReviewBand}
        assert 0.0 <= r["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# save / load thresholds
# ---------------------------------------------------------------------------


def test_save_load_thresholds(tmp_path):
    ct = ClassThreshold(
        class_idx=0, class_name="foo", low=0.2, high=0.7,
        ppv_at_high=0.75, recall_at_low=0.80,
    )
    out = tmp_path / "thresholds.json"
    save_thresholds([ct], out, meta={"num_samples": 50})
    loaded = load_thresholds(out)
    assert loaded["thresholds"][0]["class_name"] == "foo"
    assert loaded["num_samples"] == 50


def test_load_thresholds_missing():
    with pytest.raises(FileNotFoundError):
        load_thresholds(Path("/nonexistent/thresholds.json"))
