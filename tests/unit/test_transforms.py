"""Unit tests for src.train.transforms."""
from __future__ import annotations

import numpy as np
import pytest

from src.train.transforms import (
    build_inference_transforms,
    build_train_transforms,
    build_val_transforms,
)


def _random_image(h: int = 64, w: int = 64) -> np.ndarray:
    """Return a CHW float32 array in [0, 1]."""
    return np.random.rand(1, h, w).astype(np.float32)


@pytest.mark.parametrize("size", [(224, 224), (128, 128)])
def test_train_transforms_output_shape(size):
    img = _random_image()
    tfm = build_train_transforms(size)
    out = tfm(img)
    assert out.shape[-2] == size[0]
    assert out.shape[-1] == size[1]


@pytest.mark.parametrize("size", [(224, 224), (128, 128)])
def test_val_transforms_output_shape(size):
    img = _random_image()
    tfm = build_val_transforms(size)
    out = tfm(img)
    assert out.shape[-2] == size[0]
    assert out.shape[-1] == size[1]


def test_val_transforms_deterministic():
    """Same input should produce identical output for val transforms."""
    img = _random_image()
    tfm = build_val_transforms()
    out1 = tfm(img)
    out2 = tfm(img)
    np.testing.assert_array_equal(out1, out2)


def test_inference_transforms_same_as_val():
    img = _random_image()
    val_out = build_val_transforms()(img)
    inf_out = build_inference_transforms()(img)
    np.testing.assert_array_equal(val_out, inf_out)


def test_output_dtype_float32():
    img = _random_image()
    out = build_val_transforms()(img)
    import torch
    assert out.dtype == torch.float32
