"""Image preprocessing: bytes → float32 tensor (1, 1, H, W).

Uses the exact same MONAI transform pipeline as training (build_val_transforms).
CLAHE is applied before the tensor pipeline to enhance local contrast in CXRs,
improving detection of subtle findings (pleural lines, early consolidations,
effusion margins).
"""
from __future__ import annotations

import io
import os

import cv2
import numpy as np
import torch
from PIL import Image

from src.train.transforms import build_val_transforms

# Cache transforms per image size to avoid rebuilding on each request
_TRANSFORM_CACHE: dict[tuple[int, int], object] = {}

# CLAHE can be disabled via env var if you want to A/B test
_CLAHE_ENABLED: bool = os.environ.get("MEDXAI_CLAHE", "true").lower() not in ("0", "false", "no")

# Standard CXR CLAHE settings (clip=2.0, 8×8 grid is radiologist-tuned default)
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _apply_clahe(arr: np.ndarray) -> np.ndarray:
    """Apply CLAHE to a float32 (H, W) array normalised to [0, 255].

    Returns float32 in same range.
    """
    uint8 = np.clip(arr, 0, 255).astype(np.uint8)
    enhanced = _CLAHE.apply(uint8)
    return enhanced.astype(np.float32)


def _get_transforms(image_size: tuple[int, int]):
    if image_size not in _TRANSFORM_CACHE:
        _TRANSFORM_CACHE[image_size] = build_val_transforms(list(image_size))
    return _TRANSFORM_CACHE[image_size]


def decode_image(data: bytes, image_size: tuple[int, int] = (320, 320)) -> torch.Tensor:
    """Decode raw image bytes → batched tensor of shape (1, 1, H, W).

    Steps:
    1. PIL decode → grayscale
    2. CLAHE contrast enhancement (improves subtle finding detection)
    3. numpy float32 (1, H, W)
    4. MONAI val transforms (resize, scale intensity)
    5. unsqueeze batch dim → (1, 1, H, W)
    """
    img = Image.open(io.BytesIO(data)).convert("L")
    arr = np.array(img, dtype=np.float32)              # (H, W)

    if _CLAHE_ENABLED:
        arr = _apply_clahe(arr)

    arr = arr[None, ...]                               # (1, H, W)

    tfm = _get_transforms(image_size)
    tensor = tfm(arr)                                  # (1, H, W) tensor

    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(np.array(tensor, dtype=np.float32))

    return tensor.unsqueeze(0)                         # (1, 1, H, W)
