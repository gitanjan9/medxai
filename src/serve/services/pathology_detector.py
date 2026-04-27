"""Multi-label chest pathology detection using torchxrayvision.

Uses a pretrained DenseNet-121 trained on MIMIC-CXR + CheXpert + NIH + PadChest
to predict 18 binary pathology scores from a chest X-ray image.

No training required — this is inference-only using a public pretrained model.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from PIL import Image

from src.common.logging import get_logger

logger = get_logger("serve.pathology_detector")

_XRV_MODEL_ID = "densenet121-res224-all"  # trained on MIMIC+CheXpert+NIH+PadChest

PATHOLOGY_NAMES: list[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
    "Infiltration",
    "Mass",
    "Nodule",
    "Emphysema",
]

_FINDING_THRESHOLD = 0.30   # score >= this → reported as present


@dataclass
class PathologyResult:
    scores: dict[str, float]              # all 18 pathology scores
    findings: list[str]                   # classes above threshold
    top_finding: str                      # highest-score pathology
    top_score: float
    no_finding_score: float               # model confidence of "No Finding"


class XrvPathologyDetector:
    """Thread-safe torchxrayvision-based 18-class pathology detector."""

    def __init__(self) -> None:
        self._model = None
        self._transforms = None
        self._device = torch.device("cpu")

    def load(self) -> None:
        import torchxrayvision as xrv
        from pathlib import Path
        logger.info("Loading torchxrayvision model: %s", _XRV_MODEL_ID)
        self._model = xrv.models.DenseNet(weights=_XRV_MODEL_ID)

        # Load fine-tuned weights produced by active_learning.py if available
        finetuned_path = Path("artifacts/txrv_finetuned.pt")
        if finetuned_path.exists():
            state = torch.load(finetuned_path, map_location="cpu")
            self._model.load_state_dict(state, strict=False)
            logger.info("Loaded fine-tuned weights from %s", finetuned_path)
        else:
            logger.info("No fine-tuned weights found — using pretrained weights only")

        self._model.to(self._device)
        self._model.eval()
        self._pathology_names = self._model.pathologies  # actual order from model
        logger.info(
            "torchxrayvision model loaded: %d pathologies", len(self._pathology_names)
        )

    def predict(self, image_bytes: bytes) -> PathologyResult:
        if self._model is None:
            raise RuntimeError("XrvPathologyDetector.load() was not called")

        import torchxrayvision as xrv
        import skimage.transform

        # Load as grayscale, resize to 224x224
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        img_np = np.array(image).astype(np.float32)

        # torchxrayvision expects values in [-1024, 1024]
        img_np = (img_np / 255.0) * 2048 - 1024

        # Resize to 224×224
        img_np = skimage.transform.resize(img_np, (224, 224), anti_aliasing=True)

        # Shape: [1, 1, 224, 224]
        tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float()
        tensor = tensor.to(self._device)

        with torch.no_grad():
            raw = self._model(tensor)  # shape: [1, num_pathologies]

        probs = torch.sigmoid(raw)[0].cpu().numpy()

        # Build score dict using model's own pathology name list
        scores: dict[str, float] = {}
        for name, score in zip(self._pathology_names, probs):
            if name is not None:
                scores[name] = round(float(score), 4)

        findings = [k for k, v in scores.items() if v >= _FINDING_THRESHOLD and k != "No Finding"]
        findings.sort(key=lambda k: scores[k], reverse=True)

        top = max(scores, key=lambda k: scores[k] if k != "No Finding" else -1)
        no_finding_score = scores.get("No Finding", 0.0)

        return PathologyResult(
            scores=scores,
            findings=findings,
            top_finding=top,
            top_score=round(scores[top], 4),
            no_finding_score=round(no_finding_score, 4),
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_detector: Optional[XrvPathologyDetector] = None


def init_pathology_detector() -> None:
    global _detector
    _detector = XrvPathologyDetector()
    _detector.load()


def get_pathology_detector() -> Optional[XrvPathologyDetector]:
    return _detector


def is_available() -> bool:
    return _detector is not None
