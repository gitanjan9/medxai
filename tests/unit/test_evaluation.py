"""Unit tests for evaluation and calibration utilities."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.evaluate import (
    _auroc,
    _auprc,
    _sensitivity_specificity,
    expected_calibration_error,
    optimal_threshold_f1,
    optimal_threshold_youden,
)
from src.ml.calibration import ece_binary, fit_temperature
from src.inference.predict import classify, load_thresholds


# ── ECE ──────────────────────────────────────────────────────────────────────

def test_ece_perfect_calibration():
    """Perfectly calibrated model should have ECE ≈ 0."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 1000)
    y = rng.binomial(1, p)
    ece = ece_binary(y.astype(float), p)
    assert ece < 0.05, f"ECE={ece:.4f} expected < 0.05 for well-calibrated probs"


def test_ece_overconfident():
    """Constant high probability on 50% positives should have high ECE."""
    y = np.array([1, 0] * 500, dtype=float)
    p = np.full(1000, 0.9)
    ece = ece_binary(y, p)
    assert ece > 0.30


# ── AUROC ─────────────────────────────────────────────────────────────────────

def test_auroc_perfect():
    y = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert _auroc(y, scores) == pytest.approx(1.0)


def test_auroc_all_same_label_returns_nan():
    y = np.zeros(10, dtype=float)
    scores = np.random.rand(10)
    assert np.isnan(_auroc(y, scores))


def test_auprc_all_positive():
    y = np.ones(10, dtype=float)
    scores = np.random.rand(10)
    assert np.isnan(_auprc(y, scores))


# ── Thresholds ─────────────────────────────────────────────────────────────────

def test_youden_threshold_between_0_and_1():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 200).astype(float)
    s = rng.uniform(0, 1, 200)
    thr = optimal_threshold_youden(y, s)
    assert 0.0 <= thr <= 1.0


def test_f1_threshold_between_0_and_1():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 200).astype(float)
    s = rng.uniform(0, 1, 200)
    thr = optimal_threshold_f1(y, s)
    assert 0.0 <= thr <= 1.0


def test_sensitivity_specificity_perfect():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    sens, spec = _sensitivity_specificity(y_true, y_pred)
    assert sens == pytest.approx(1.0)
    assert spec == pytest.approx(1.0)


# ── Temperature calibration ───────────────────────────────────────────────────

def test_temperature_fit_returns_positive():
    rng = np.random.default_rng(7)
    logits = rng.normal(0, 2, (200, 5))
    targets = (rng.uniform(0, 1, (200, 5)) > 0.6).astype(float)
    T = fit_temperature(logits, targets)
    assert T > 0.0


def test_temperature_above_zero_clamp():
    """Very overconfident logits should not collapse temperature to 0."""
    logits = np.full((100, 3), 10.0)
    targets = np.zeros((100, 3))
    T = fit_temperature(logits, targets)
    assert T > 0.01


# ── inference classify ─────────────────────────────────────────────────────────

_CLASSES = ["Atelectasis", "Cardiomegaly", "Pneumothorax", "No Finding"]
_THRESHOLDS = {"Atelectasis": 0.70, "Cardiomegaly": 0.70, "Pneumothorax": 0.75}


def test_classify_no_raw_50_cutoff():
    """classify() must never use 0.50 as a threshold."""
    probs = np.array([0.55, 0.52, 0.58, 0.40])
    result = classify(probs, _CLASSES, _THRESHOLDS)
    assert result["status"] != "positive", (
        "Scores 0.52-0.58 are below all class thresholds — must not be positive."
    )
    assert all(f["score"] >= 0.60 for f in result["positive_findings"]), (
        "No finding should pass the positive threshold at 0.52-0.58."
    )


def test_classify_positive_when_above_threshold():
    """Pneumothorax at 0.80 >= 0.75 threshold → positive."""
    probs = np.array([0.30, 0.30, 0.80, 0.10])
    result = classify(probs, _CLASSES, _THRESHOLDS)
    assert result["status"] == "positive"
    assert result["top_label"] == "Pneumothorax"


def test_classify_review_required_between_bounds():
    """Atelectasis at 0.65: >= 0.60 but < 0.70 threshold → review_required."""
    probs = np.array([0.65, 0.25, 0.20, 0.30])
    result = classify(probs, _CLASSES, _THRESHOLDS)
    assert result["status"] == "review_required"
    assert any(f["name"] == "Atelectasis" for f in result["review_findings"])


def test_classify_likely_normal_when_all_low():
    """All scores around 0.50-0.58 → likely_normal_or_uncertain."""
    probs = np.array([0.52, 0.54, 0.50, 0.45])
    result = classify(probs, _CLASSES, _THRESHOLDS)
    assert result["status"] == "likely_normal_or_uncertain"


def test_classify_suppresses_below_review_min():
    """Scores below 0.60 must not appear in positive or review findings."""
    probs = np.array([0.55, 0.48, 0.59, 0.40])
    result = classify(probs, _CLASSES, _THRESHOLDS)
    all_finding_names = (
        [f["name"] for f in result["positive_findings"]]
        + [f["name"] for f in result["review_findings"]]
    )
    for name in all_finding_names:
        idx = _CLASSES.index(name)
        assert probs[idx] >= 0.60, (
            f"{name} at {probs[idx]:.2f} below REVIEW_MIN_SCORE appeared in findings."
        )
