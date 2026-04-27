"""Multi-model ensemble for CXR multi-label classification.

Loads N checkpoints (same or different architectures), averages their
calibrated probabilities, and detects strong disagreement between models.

Usage
-----
  from src.ml.ensemble import CxrEnsemble

  ens = CxrEnsemble.from_registry("models/registry.json", tag="production")
  probs, meta = ens.predict_bytes(image_bytes)

  if meta["disagreement"] == "high":
      # return review_required regardless of score
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.common.logging import get_logger

logger = get_logger("ml.ensemble")

_DISAGREE_THRESHOLD = 0.20   # max std across models to call "high disagreement"


@dataclass
class EnsembleMember:
    arch: str
    checkpoint: Path
    num_classes: int
    temperature: float = 1.0
    label_cols: list[str] = field(default_factory=list)

    _model: Optional[torch.nn.Module] = field(default=None, repr=False, init=False)

    def load(self, device: torch.device) -> None:
        from src.serve.services.model_loader import load_model
        self._model = load_model(self.arch, self.num_classes, self.checkpoint, device)
        self._model.eval()
        logger.info("Loaded member: arch=%s  ckpt=%s", self.arch, self.checkpoint.name)

    @torch.no_grad()
    def predict_tensor(self, tensor: torch.Tensor, device: torch.device) -> np.ndarray:
        """Return calibrated probabilities (C,) for a single image tensor (1,H,W)."""
        if self._model is None:
            raise RuntimeError("Call load() before predict_tensor().")
        x = tensor.unsqueeze(0).to(device)
        logits = self._model(x).squeeze(0).cpu().float().numpy()
        return 1.0 / (1.0 + np.exp(-logits / max(self.temperature, 0.01)))


@dataclass
class EnsemblePrediction:
    probs: np.ndarray           # (C,) averaged calibrated probabilities
    per_model_probs: np.ndarray # (N, C) individual model outputs
    disagreement: str           # "low" | "medium" | "high"
    disagreement_score: float   # max std over classes, averaged across top-5
    label_cols: list[str]


class CxrEnsemble:
    """Averages calibrated probabilities from N EfficientNet/DenseNet members."""

    def __init__(
        self,
        members: list[EnsembleMember],
        device: torch.device | None = None,
    ) -> None:
        self.members = members
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self._loaded = False

    def load(self) -> None:
        for m in self.members:
            m.load(self.device)
        self._loaded = True
        logger.info("Ensemble loaded: %d members on %s", len(self.members), self.device)

    @classmethod
    def from_registry(
        cls,
        registry_path: str | Path,
        tag: str = "production",
    ) -> "CxrEnsemble":
        """Build an ensemble from all entries in registry.json with matching tag."""
        with open(registry_path) as f:
            registry: list[dict] = json.load(f)

        members = []
        for entry in registry:
            if entry.get("tag") != tag:
                continue
            members.append(EnsembleMember(
                arch=entry["architecture"],
                checkpoint=Path(entry["checkpoint_path"]),
                num_classes=entry["num_classes"],
                temperature=entry.get("calibration", {}).get("temperature", 1.0),
                label_cols=entry.get("label_cols", []),
            ))
        if not members:
            raise ValueError(f"No registry entries with tag={tag!r}")
        return cls(members)

    def predict_tensor(self, tensor: torch.Tensor) -> EnsemblePrediction:
        """Average predictions across all members for a preprocessed tensor.

        Args:
            tensor: (1, H, W) float32 tensor (grayscale, normalised).
        """
        if not self._loaded:
            self.load()

        per_model = np.stack([
            m.predict_tensor(tensor, self.device) for m in self.members
        ])  # (N, C)

        avg_probs = per_model.mean(axis=0)  # (C,)

        top5_idx = np.argsort(avg_probs)[-5:]
        std_top5 = float(per_model[:, top5_idx].std(axis=0).mean())
        if std_top5 < 0.05:
            disagree = "low"
        elif std_top5 < _DISAGREE_THRESHOLD:
            disagree = "medium"
        else:
            disagree = "high"

        label_cols = self.members[0].label_cols or [str(i) for i in range(len(avg_probs))]

        return EnsemblePrediction(
            probs=avg_probs,
            per_model_probs=per_model,
            disagreement=disagree,
            disagreement_score=round(std_top5, 4),
            label_cols=label_cols,
        )

    def predict_bytes(self, image_bytes: bytes, image_size: int = 320) -> tuple[np.ndarray, dict]:
        """Convenience wrapper: decode bytes → tensor → predict.

        Returns (probs, meta) where meta contains disagreement info.
        """
        from src.serve.services.preprocessing import decode_image
        tensor = decode_image(image_bytes, image_size=(image_size, image_size))
        pred = self.predict_tensor(tensor)
        meta = {
            "disagreement": pred.disagreement,
            "disagreement_score": pred.disagreement_score,
            "n_models": len(self.members),
        }
        return pred.probs, meta
