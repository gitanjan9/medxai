"""Unit tests for src/train/explainability.py"""
import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Minimal CNN with a named conv block for Grad-CAM targeting."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# _get_target_module
# ---------------------------------------------------------------------------


def test_get_target_module_valid():
    from src.train.explainability import _get_target_module

    model = _TinyModel()
    module = _get_target_module(model, "features")
    assert isinstance(module, nn.Sequential)


def test_get_target_module_invalid():
    from src.train.explainability import _get_target_module

    model = _TinyModel()
    with pytest.raises(AttributeError):
        _get_target_module(model, "nonexistent_layer")


# ---------------------------------------------------------------------------
# _resize_and_normalise
# ---------------------------------------------------------------------------


def test_resize_and_normalise_range():
    from src.train.explainability import _resize_and_normalise

    cam = np.random.rand(7, 7).astype(np.float32)
    out = _resize_and_normalise(cam, (224, 224))
    assert out.shape == (224, 224)
    assert 0.0 <= out.min() and out.max() <= 1.0 + 1e-5


def test_resize_and_normalise_uniform_input():
    from src.train.explainability import _resize_and_normalise

    cam = np.ones((4, 4), dtype=np.float32)
    out = _resize_and_normalise(cam, (16, 16))
    # Uniform input → all zeros (or all same normalised value)
    assert out.shape == (16, 16)


# ---------------------------------------------------------------------------
# GradCAMGenerator fallback
# ---------------------------------------------------------------------------


def test_gradcam_fallback_returns_heatmap():
    from src.train.explainability import GradCAMGenerator

    device = torch.device("cpu")
    model = _TinyModel(num_classes=4)
    gen = GradCAMGenerator(model, target_layer="features", device=device)

    image = torch.randn(1, 1, 32, 32)
    heatmap = gen.generate(image, target_class=0, output_size=(32, 32))

    assert heatmap is not None, "Expected a heatmap, got None"
    assert heatmap.shape == (32, 32)
    assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0 + 1e-5


def test_gradcam_fallback_different_target_classes():
    from src.train.explainability import GradCAMGenerator

    device = torch.device("cpu")
    model = _TinyModel(num_classes=4)
    gen = GradCAMGenerator(model, target_layer="features", device=device)
    image = torch.randn(1, 1, 32, 32)

    hm0 = gen.generate(image, target_class=0, output_size=(32, 32))
    hm1 = gen.generate(image, target_class=1, output_size=(32, 32))
    assert hm0 is not None and hm1 is not None


# ---------------------------------------------------------------------------
# save_heatmap
# ---------------------------------------------------------------------------


def test_save_heatmap(tmp_path):
    from src.train.explainability import save_heatmap

    heatmap = np.random.rand(224, 224).astype(np.float32)
    path = tmp_path / "test_heatmap.png"
    save_heatmap(heatmap, path)
    assert path.exists()
    assert path.stat().st_size > 0


# ---------------------------------------------------------------------------
# save_explanation_metadata
# ---------------------------------------------------------------------------


def test_save_explanation_metadata(tmp_path):
    from src.train.explainability import save_explanation_metadata

    meta = {"sample_idx": 5, "predicted_class": 2, "confidence": 0.87}
    path = tmp_path / "meta.json"
    save_explanation_metadata(meta, path)
    import json
    loaded = json.loads(path.read_text())
    assert loaded["sample_idx"] == 5
    assert loaded["confidence"] == pytest.approx(0.87)
