"""Two-stage hierarchical classifier for CXR normal-finding classification.

Stage 1  (3-class):  normal_acute_absence | stable_no_change | specific_negative
Stage 2  (2-class):  no_acute_cardiopulmonary | no_acute_intrathoracic
         ↑ runs only when Stage 1 predicts ``normal_acute_absence``

Inference flow
--------------
probs1 = stage1_model(image)          # shape [B, 3]
if argmax(probs1) == normal_acute_absence:
    probs2 = stage2_model(image)      # shape [B, 2]
    final_class = stage2_decode(argmax(probs2))
else:
    final_class = stage1_decode(argmax(probs1))

CLI
---
    python -m src.train.hierarchical --stage 1 --config configs/stage1.yaml
    python -m src.train.hierarchical --stage 2 --config configs/stage2.yaml
    python -m src.train.hierarchical --infer   --config configs/inference.yaml \\
        --image path/to/image.jpg
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.common.logging import get_logger

logger = get_logger("hierarchical")

# Label used by Stage 1 that triggers Stage 2
_STAGE2_TRIGGER = "normal_acute_absence"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class HierarchicalResult:
    """Full inference result from the two-stage pipeline."""
    stage1_class: str
    stage1_probs: dict[str, float]
    stage2_class: Optional[str]
    stage2_probs: Optional[dict[str, float]]
    final_class: str
    top3: list[tuple[str, float]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "final_class": self.final_class,
            "stage1": {"class": self.stage1_class, "probs": self.stage1_probs},
            "stage2": (
                {"class": self.stage2_class, "probs": self.stage2_probs}
                if self.stage2_class is not None else None
            ),
            "top3": [{"class": c, "prob": round(p, 4)} for c, p in self.top3],
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class HierarchicalPipeline:
    """Wraps Stage-1 and Stage-2 models for end-to-end hierarchical inference.

    Args:
        stage1_model:   Trained 3-class PyTorch model (eval mode).
        stage1_classes: Class names in order matching stage1 output indices.
        stage2_model:   Trained 2-class PyTorch model (eval mode).
        stage2_classes: Class names in order matching stage2 output indices.
        temperature1:   Calibration temperature for Stage 1 (default 1.0).
        temperature2:   Calibration temperature for Stage 2 (default 1.0).
        device:         Torch device.
    """

    def __init__(
        self,
        stage1_model: torch.nn.Module,
        stage1_classes: list[str],
        stage2_model: torch.nn.Module,
        stage2_classes: list[str],
        temperature1: float = 1.0,
        temperature2: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.stage1_model = stage1_model.eval().to(self.device)
        self.stage2_model = stage2_model.eval().to(self.device)
        self.stage1_classes = stage1_classes
        self.stage2_classes = stage2_classes
        self.temperature1 = temperature1
        self.temperature2 = temperature2
        self._s2_trigger_idx = stage1_classes.index(_STAGE2_TRIGGER)

    @torch.no_grad()
    def predict_single(self, image: torch.Tensor) -> HierarchicalResult:
        """Run full pipeline on a single image tensor ``[1, C, H, W]``.

        Returns:
            :class:`HierarchicalResult` with stage1, optional stage2, and final decision.
        """
        image = image.to(self.device)
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Stage 1
        logits1 = self.stage1_model(image)
        probs1 = F.softmax(logits1 / self.temperature1, dim=1).squeeze(0).cpu()
        s1_idx = int(probs1.argmax())
        s1_class = self.stage1_classes[s1_idx]
        s1_probs = {c: round(float(probs1[i]), 4) for i, c in enumerate(self.stage1_classes)}

        # Build merged top-3 (Stage 1 probs as base)
        combined: dict[str, float] = dict(s1_probs)

        stage2_class: Optional[str] = None
        s2_probs: Optional[dict[str, float]] = None

        if s1_idx == self._s2_trigger_idx:
            # Stage 2
            logits2 = self.stage2_model(image)
            probs2 = F.softmax(logits2 / self.temperature2, dim=1).squeeze(0).cpu()
            s2_idx = int(probs2.argmax())
            stage2_class = self.stage2_classes[s2_idx]
            s2_probs = {c: round(float(probs2[i]), 4) for i, c in enumerate(self.stage2_classes)}
            final_class = stage2_class
            # Replace stage2-trigger entry in combined with stage2 breakdown
            trigger_prob = combined.pop(_STAGE2_TRIGGER)
            for c, p in s2_probs.items():
                combined[c] = round(float(p) * trigger_prob, 4)
        else:
            final_class = s1_class

        top3 = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:3]

        return HierarchicalResult(
            stage1_class=s1_class,
            stage1_probs=s1_probs,
            stage2_class=stage2_class,
            stage2_probs=s2_probs,
            final_class=final_class,
            top3=top3,
        )

    @torch.no_grad()
    def predict_batch(self, images: torch.Tensor) -> list[HierarchicalResult]:
        """Run pipeline over a batch ``[B, C, H, W]``.

        Each image is processed independently so Stage 2 only runs for relevant items.
        """
        return [self.predict_single(images[i]) for i in range(images.shape[0])]


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------


def load_pipeline(
    stage1_ckpt: Path,
    stage1_label_map: Path,
    stage2_ckpt: Path,
    stage2_label_map: Path,
    architecture: str = "densenet121",
    temperature1: float = 1.0,
    temperature2: float = 1.0,
    device: Optional[torch.device] = None,
) -> HierarchicalPipeline:
    """Build a :class:`HierarchicalPipeline` from checkpoint paths.

    Args:
        stage1_ckpt:        Path to Stage-1 ``.pt`` checkpoint.
        stage1_label_map:   Path to Stage-1 ``label_map.json``.
        stage2_ckpt:        Path to Stage-2 ``.pt`` checkpoint.
        stage2_label_map:   Path to Stage-2 ``label_map.json``.
        architecture:       Model architecture string (same for both stages).
        temperature1:       Calibration temperature for Stage 1.
        temperature2:       Calibration temperature for Stage 2.
        device:             Torch device.

    Returns:
        Ready-to-use :class:`HierarchicalPipeline`.
    """
    from src.common.config import ModelConfig
    from src.common.schemas import LabelMap
    from src.train.evaluate import _load_checkpoint
    from src.train.model_factory import build_model

    device = device or torch.device("cpu")

    def _load(ckpt: Path, lm: Path) -> tuple[torch.nn.Module, list[str]]:
        label_map = LabelMap.load(lm)
        cfg = ModelConfig(
            architecture=architecture,
            num_classes=label_map.num_classes,
            pretrained=False,
        )
        model = build_model(cfg)
        _load_checkpoint(model, ckpt, device)
        model.eval()
        return model, label_map.class_names()

    m1, c1 = _load(stage1_ckpt, stage1_label_map)
    m2, c2 = _load(stage2_ckpt, stage2_label_map)

    return HierarchicalPipeline(
        stage1_model=m1, stage1_classes=c1,
        stage2_model=m2, stage2_classes=c2,
        temperature1=temperature1, temperature2=temperature2,
        device=device,
    )


# ---------------------------------------------------------------------------
# Training entry point (delegates to train.py with overridden config)
# ---------------------------------------------------------------------------


def train_stage(stage: int, config_path: str) -> None:
    """Train a single stage by delegating to the main train loop.

    The config at ``config_path`` must already point to the correct merge map,
    label map, checkpoint dir, and num_classes for the given stage.
    """
    from src.train.train import train
    logger.info("Training Stage %d using config: %s", stage, config_path)
    train(config_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hierarchical CXR classifier")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--stage", type=int, choices=[1, 2],
                       help="Train stage 1 or 2")
    group.add_argument("--infer", action="store_true",
                       help="Run hierarchical inference on a single image")
    p.add_argument("--config", required=True, help="Config YAML path")
    p.add_argument("--image", type=str, default=None,
                   help="Path to image file (required with --infer)")
    p.add_argument("--stage1-ckpt", type=str, default=None)
    p.add_argument("--stage2-ckpt", type=str, default=None)
    p.add_argument("--stage1-lm",   type=str, default=None)
    p.add_argument("--stage2-lm",   type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.stage:
        train_stage(args.stage, args.config)

    elif args.infer:
        from src.common.utils import get_device
        from src.train.transforms import build_val_transforms

        if not args.image:
            raise ValueError("--image is required with --infer")

        device = get_device()
        pipeline = load_pipeline(
            stage1_ckpt=Path(args.stage1_ckpt),
            stage1_label_map=Path(args.stage1_lm),
            stage2_ckpt=Path(args.stage2_ckpt),
            stage2_label_map=Path(args.stage2_lm),
            device=device,
        )

        from PIL import Image
        import torchvision.transforms.functional as TF
        img = Image.open(args.image).convert("L")
        tfm = build_val_transforms([224, 224])
        tensor = tfm(img).unsqueeze(0)

        result = pipeline.predict_single(tensor)
        print(json.dumps(result.as_dict(), indent=2))
