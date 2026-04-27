"""Integration tests for artifact loading utilities (no server required)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.serve.services.calibration import load_temperature
from src.serve.services.model_loader import load_label_map, resolve_best_checkpoint
from src.serve.services.thresholds import apply_thresholds, load_thresholds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ARTIFACTS = Path("artifacts")
V2_ARTIFACTS = ARTIFACTS / "v2"


# ---------------------------------------------------------------------------
# Label map
# ---------------------------------------------------------------------------

def test_label_map_file_exists() -> None:
    assert (ARTIFACTS / "label_map.json").exists(), "artifacts/label_map.json missing"


def test_label_map_loads_correctly() -> None:
    lm = load_label_map(ARTIFACTS / "label_map.json")
    assert lm["num_classes"] == 5
    assert len(lm["idx_to_str"]) == 5
    assert "no_acute_cardiopulmonary" in lm["str_to_idx"]


def test_v2_label_map_file_exists() -> None:
    assert (V2_ARTIFACTS / "label_map.json").exists(), "artifacts/v2/label_map.json missing"


def test_v2_label_map_matches_main() -> None:
    lm_main = load_label_map(ARTIFACTS / "label_map.json")
    lm_v2   = load_label_map(V2_ARTIFACTS / "label_map.json")
    assert lm_main["str_to_idx"] == lm_v2["str_to_idx"]


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def test_calibration_file_exists() -> None:
    assert (ARTIFACTS / "calibration.json").exists(), "artifacts/calibration.json missing"


def test_calibration_temperature_positive() -> None:
    T = load_temperature(ARTIFACTS / "calibration.json")
    assert T > 0


def test_calibration_temperature_plausible() -> None:
    T = load_temperature(ARTIFACTS / "calibration.json")
    assert 0.5 < T < 5.0, f"Temperature {T} looks implausible"


def test_calibration_missing_file_returns_1() -> None:
    T = load_temperature(Path("/nonexistent/calibration.json"))
    assert T == 1.0


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

def test_thresholds_file_exists() -> None:
    assert (ARTIFACTS / "thresholds.json").exists(), "artifacts/thresholds.json missing"


def test_thresholds_load_five_classes() -> None:
    thresholds = load_thresholds(ARTIFACTS / "thresholds.json")
    assert len(thresholds) == 5


def test_thresholds_low_lt_high() -> None:
    thresholds = load_thresholds(ARTIFACTS / "thresholds.json")
    for t in thresholds:
        assert t["low"] < t["high"], f"low >= high for class {t['class_name']}"


def test_thresholds_decision_positive() -> None:
    thresholds = load_thresholds(ARTIFACTS / "thresholds.json")
    probs = torch.zeros(5)
    probs[0] = 0.99
    dec = apply_thresholds(probs, 0, thresholds)
    assert dec.decision == "positive"


def test_thresholds_decision_negative() -> None:
    thresholds = load_thresholds(ARTIFACTS / "thresholds.json")
    probs = torch.full((5,), 0.001)
    probs[0] = 0.001
    dec = apply_thresholds(probs, 0, thresholds)
    assert dec.decision == "negative"


def test_thresholds_decision_review() -> None:
    thresholds = load_thresholds(ARTIFACTS / "thresholds.json")
    entry = thresholds[0]
    mid = (entry["low"] + entry["high"]) / 2
    probs = torch.zeros(5)
    probs[0] = mid
    dec = apply_thresholds(probs, 0, thresholds)
    assert dec.decision == "review"


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def test_v2_checkpoint_dir_exists() -> None:
    assert (V2_ARTIFACTS / "checkpoints").exists()


def test_v2_best_checkpoint_resolves() -> None:
    best = resolve_best_checkpoint(V2_ARTIFACTS / "checkpoints")
    assert best.suffix == ".pt"
    assert best.exists()


def test_v2_best_checkpoint_is_epoch60() -> None:
    best = resolve_best_checkpoint(V2_ARTIFACTS / "checkpoints")
    assert "060" in best.name or "9472" in best.name
