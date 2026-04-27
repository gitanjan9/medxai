"""Unit tests for src/train/calibrate.py"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# TemperatureScaler
# ---------------------------------------------------------------------------


def test_temperature_scaler_forward():
    from src.train.calibrate import TemperatureScaler

    scaler = TemperatureScaler(init_T=2.0)
    logits = torch.randn(4, 8)
    out = scaler(logits)
    assert out.shape == logits.shape
    # scaled logits should be logits / 2.0
    assert torch.allclose(out, logits / 2.0, atol=1e-5)


def test_temperature_scaler_clamps_minimum():
    from src.train.calibrate import TemperatureScaler

    scaler = TemperatureScaler(init_T=0.001)
    with torch.no_grad():
        scaler.temperature.fill_(0.0)
    logits = torch.randn(2, 4)
    out = scaler(logits)  # should not divide by zero
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# fit_temperature
# ---------------------------------------------------------------------------


def test_fit_temperature_returns_scaler():
    from src.train.calibrate import fit_temperature

    np.random.seed(0)
    logits = np.random.randn(50, 4).astype(np.float32)
    labels = np.random.randint(0, 4, size=50)
    scaler = fit_temperature(logits, labels, max_iter=10, lr=0.1)
    assert 0.05 <= scaler.T <= 50.0, f"T={scaler.T} out of expected range"


def test_fit_temperature_reduces_nll():
    from src.train.calibrate import fit_temperature
    import torch.nn.functional as F

    np.random.seed(42)
    # Overconfident logits (large values)
    logits = np.random.randn(80, 3).astype(np.float32) * 5
    labels = np.argmax(logits, axis=1)  # perfect predictions but over-confident
    scaler = fit_temperature(logits, labels, max_iter=50, lr=0.05)
    # Temperature should be > 1 to soften over-confident predictions
    assert scaler.T > 1.0, f"Expected T > 1 for over-confident model, got {scaler.T}"


# ---------------------------------------------------------------------------
# apply_temperature
# ---------------------------------------------------------------------------


def test_apply_temperature_sums_to_one():
    from src.train.calibrate import apply_temperature

    logits = np.random.randn(10, 5).astype(np.float32)
    probs = apply_temperature(logits, temperature=1.5)
    assert probs.shape == (10, 5)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_apply_temperature_non_negative():
    from src.train.calibrate import apply_temperature

    logits = np.random.randn(10, 5).astype(np.float32)
    probs = apply_temperature(logits, temperature=2.0)
    assert (probs >= 0).all()


# ---------------------------------------------------------------------------
# save / load calibration
# ---------------------------------------------------------------------------


def test_save_load_calibration(tmp_path):
    from src.train.calibrate import TemperatureScaler, save_calibration, load_calibration

    scaler = TemperatureScaler(init_T=1.3)
    out = tmp_path / "calibration.json"
    save_calibration(scaler, out, meta={"num_samples": 100})

    loaded = load_calibration(out)
    assert loaded["method"] == "temperature_scaling"
    assert abs(loaded["temperature"] - scaler.T) < 1e-4
    assert loaded["num_samples"] == 100


def test_load_calibration_missing_file():
    from src.train.calibrate import load_calibration

    with pytest.raises(FileNotFoundError):
        load_calibration(Path("/nonexistent/calibration.json"))
