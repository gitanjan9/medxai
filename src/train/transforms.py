"""MONAI-based transform pipelines for training, validation, and inference."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    EnsureType,
    RandAdjustContrast,
    RandAffine,
    RandFlip,
    RandShiftIntensity,
    RandRotate,
    RandZoom,
    Resize,
    ScaleIntensity,
)


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------


def build_train_transforms(image_size: Sequence[int] = (224, 224)) -> Callable:
    """Return the CXR-realistic augmentation pipeline for training.

    The pipeline expects a float32 numpy array of shape ``(1, H, W)``
    (already CHW, values in [0, 1]) and returns a float32 tensor.

    Augmentations (clinically grounded for chest X-rays):
    - Resize to *image_size*
    - Random horizontal flip (p=0.5)  – PA/AP projections can appear mirrored
    - Random rotation ±10°             – small patient positioning variation
    - Random zoom 0.95–1.05×           – subtle distance variation
    - Random affine shear (±0.05)      – mild projection distortion
    - Random intensity shift ±0.10     – exposure variation
    - Random contrast adjustment       – mA/kVp variation
    - Intensity re-normalise to [0, 1]
    Note: Gaussian noise removed – not a realistic CXR artefact.
    """
    h, w = image_size[0], image_size[1]
    return Compose(
        [
            EnsureChannelFirst(channel_dim=0),  # (1,H,W) already – no-op guard
            Resize(spatial_size=[h, w]),
            RandFlip(spatial_axis=1, prob=0.5),
            RandRotate(range_x=np.deg2rad(10), prob=0.5, keep_size=True),
            RandZoom(min_zoom=0.95, max_zoom=1.05, prob=0.4),
            RandAffine(
                prob=0.3,
                shear_range=[(-0.05, 0.05), (-0.05, 0.05)],
                padding_mode="zeros",
            ),
            RandShiftIntensity(offsets=0.10, prob=0.4),
            RandAdjustContrast(gamma=(0.85, 1.15), prob=0.35),
            ScaleIntensity(minv=0.0, maxv=1.0),
            EnsureType(dtype=np.float32),
        ]
    )


def build_val_transforms(image_size: Sequence[int] = (224, 224)) -> Callable:
    """Return the deterministic pipeline for validation and evaluation."""
    h, w = image_size[0], image_size[1]
    return Compose(
        [
            EnsureChannelFirst(channel_dim=0),
            Resize(spatial_size=[h, w]),
            ScaleIntensity(minv=0.0, maxv=1.0),
            EnsureType(dtype=np.float32),
        ]
    )


def build_inference_transforms(image_size: Sequence[int] = (224, 224)) -> Callable:
    """Return the deterministic pipeline for single-image inference.

    Identical to val transforms; kept as a separate entry-point for
    Day 2 TTA (test-time augmentation) extension.
    """
    return build_val_transforms(image_size)
