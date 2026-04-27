"""Unit tests for src/train/review_logic.py"""
import json
from pathlib import Path

import numpy as np
import pytest

from src.train.review_logic import ReviewDecision, ReviewLogic
from src.train.thresholds import ReviewBand


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_thresholds(n_classes: int = 4, low: float = 0.25, high: float = 0.65):
    return [
        {"class_idx": i, "class_name": f"cls_{i}",
         "low": low, "high": high,
         "ppv_at_high": 0.7, "recall_at_low": 0.75}
        for i in range(n_classes)
    ]


def _uniform_probs(n_classes: int, peak_class: int, peak_val: float) -> np.ndarray:
    probs = np.ones(n_classes, dtype=np.float32) * (1.0 - peak_val) / (n_classes - 1)
    probs[peak_class] = peak_val
    return probs


# ---------------------------------------------------------------------------
# Band decisions
# ---------------------------------------------------------------------------


def test_decide_positive():
    logic = ReviewLogic(
        _make_thresholds(), confidence_threshold=0.5,
        require_explanation=False, flag_weak_calibration=False,
    )
    probs = _uniform_probs(4, peak_class=2, peak_val=0.80)
    d = logic.decide(0, probs)
    assert d.band == ReviewBand.POSITIVE
    assert not d.requires_review


def test_decide_negative():
    # Use low=0.10 so an argmax of ~0.15 can be NEGATIVE
    thresholds = _make_thresholds(n_classes=4, low=0.10, high=0.65)
    logic = ReviewLogic(
        thresholds, confidence_threshold=0.3,
        require_explanation=False, flag_weak_calibration=False,
    )
    # Class 0 is argmax at ~0.08/0.31 = 0.26... still tricky with 4 classes.
    # Instead: use near-uniform probs → argmax is ~0.27 < low=0.10 is impossible.
    # Solution: pass a single-class scenario where one class dominates at exactly below low.
    # low=0.10: prob must be in [0, 0.10) for NEGATIVE.
    # With 4 classes, argmax >= 0.25 always. So use low=0.30 and n_classes=10.
    thresholds10 = _make_thresholds(n_classes=10, low=0.30, high=0.65)
    logic10 = ReviewLogic(
        thresholds10, confidence_threshold=0.2,
        require_explanation=False, flag_weak_calibration=False,
    )
    # 10 near-equal classes → argmax ≈ 0.10 which is < low=0.30
    probs10 = np.ones(10, dtype=np.float32) / 10
    d = logic10.decide(0, probs10)
    assert d.band == ReviewBand.NEGATIVE


def test_decide_review_band():
    logic = ReviewLogic(
        _make_thresholds(), confidence_threshold=0.3,
        require_explanation=False, flag_weak_calibration=False,
    )
    probs = _uniform_probs(4, peak_class=1, peak_val=0.45)
    d = logic.decide(0, probs)
    assert d.band == ReviewBand.REVIEW
    assert d.requires_review
    assert d.in_review_band


# ---------------------------------------------------------------------------
# Weak calibration trigger
# ---------------------------------------------------------------------------


def test_weak_calibration_flag():
    logic = ReviewLogic(
        _make_thresholds(), confidence_threshold=0.80,
        require_explanation=False, flag_weak_calibration=True,
    )
    probs = _uniform_probs(4, peak_class=0, peak_val=0.70)
    d = logic.decide(0, probs)
    assert d.weak_calibration
    assert d.requires_review
    assert any("confidence" in r for r in d.reasons)


def test_weak_calibration_not_flagged_when_disabled():
    logic = ReviewLogic(
        _make_thresholds(), confidence_threshold=0.90,
        require_explanation=False, flag_weak_calibration=False,
    )
    probs = _uniform_probs(4, peak_class=2, peak_val=0.80)
    d = logic.decide(0, probs)
    assert not d.weak_calibration


# ---------------------------------------------------------------------------
# Grad-CAM failure trigger
# ---------------------------------------------------------------------------


def test_gradcam_failure_flag():
    logic = ReviewLogic(
        _make_thresholds(), confidence_threshold=0.3,
        require_explanation=False, flag_weak_calibration=False,
    )
    probs = _uniform_probs(4, peak_class=3, peak_val=0.80)
    d = logic.decide(0, probs, gradcam_failed=True)
    assert d.gradcam_failed
    assert any("grad-cam" in r for r in d.reasons)


# ---------------------------------------------------------------------------
# Explanation missing trigger
# ---------------------------------------------------------------------------


def test_explanation_path_none_triggers_review():
    logic = ReviewLogic(
        _make_thresholds(), confidence_threshold=0.3,
        require_explanation=True, flag_weak_calibration=False,
    )
    probs = _uniform_probs(4, peak_class=0, peak_val=0.80)
    d = logic.decide(0, probs, explanation_path=None)
    assert d.explanation_missing
    assert d.requires_review


def test_explanation_valid_json(tmp_path):
    logic = ReviewLogic(
        _make_thresholds(), confidence_threshold=0.3,
        require_explanation=True, flag_weak_calibration=False,
    )
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps({"sample_idx": 0, "confidence": 0.9}))
    probs = _uniform_probs(4, peak_class=0, peak_val=0.80)
    d = logic.decide(0, probs, explanation_path=str(meta_path))
    assert not d.explanation_missing


def test_explanation_empty_json_triggers_review(tmp_path):
    logic = ReviewLogic(
        _make_thresholds(), confidence_threshold=0.3,
        require_explanation=True, flag_weak_calibration=False,
    )
    meta_path = tmp_path / "empty.json"
    meta_path.write_text("{}")
    probs = _uniform_probs(4, peak_class=0, peak_val=0.80)
    d = logic.decide(0, probs, explanation_path=str(meta_path))
    assert d.explanation_missing


# ---------------------------------------------------------------------------
# Batch decide
# ---------------------------------------------------------------------------


def test_decide_batch_length():
    logic = ReviewLogic(
        _make_thresholds(), confidence_threshold=0.3,
        require_explanation=False, flag_weak_calibration=False,
    )
    rng = np.random.default_rng(7)
    probs = rng.dirichlet(np.ones(4), size=10).astype(np.float32)
    decisions = logic.decide_batch(probs)
    assert len(decisions) == 10
    assert all(isinstance(d, ReviewDecision) for d in decisions)
