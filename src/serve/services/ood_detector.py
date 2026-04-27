"""Zero-shot OOD detector: is this image a chest X-ray?

Uses OpenAI CLIP (via HuggingFace transformers) to compute cosine similarity
between the image and a set of CXR vs non-CXR text prompts.

Loaded lazily and cached — only initialised when OOD checking is enabled.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image

from src.common.logging import get_logger

logger = get_logger("serve.ood_detector")

_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

_CXR_PROMPTS = [
    "a chest X-ray radiograph",
    "chest radiograph posteroanterior view",
    "frontal chest X-ray medical image",
    "CXR lung radiograph",
]

_NOT_CXR_PROMPTS = [
    "a natural photograph",
    "a portrait or selfie photo",
    "a CT scan cross section",
    "an MRI brain scan",
    "a document or text page",
    "an ultrasound image",
    "a drawing or illustration",
    "a food photograph",
]

_ALL_PROMPTS = _CXR_PROMPTS + _NOT_CXR_PROMPTS
_N_CXR = len(_CXR_PROMPTS)


@dataclass
class OodResult:
    decision: str         # "accept" | "review" | "reject"
    cxr_score: float      # sum of softmax weights on CXR prompts (0–1)
    reason: str           # "accepted" | "ood_in_review_band" | "non_cxr_rejected" | "check_failed"
    is_cxr: bool          # True for accept and review (inference still runs); False for reject


class ClipOodDetector:
    """Thread-safe CLIP-based OOD detector (loaded once at startup).

    Three-band decision:
      score >= accept_threshold  → accept   (clearly a CXR)
      score >= reject_threshold  → review   (uncertain; inference still runs + warning)
      score <  reject_threshold  → reject   (not a CXR; HTTP 422 is returned)
    """

    def __init__(self, accept_threshold: float = 0.45, reject_threshold: float = 0.20) -> None:
        self.accept_threshold = accept_threshold
        self.reject_threshold = reject_threshold
        self._model = None
        self._processor = None
        self._device = torch.device("cpu")  # CLIP runs on CPU always

    def load(self) -> None:
        from transformers import CLIPModel, CLIPProcessor
        logger.info("Loading CLIP OOD model: %s", _CLIP_MODEL_ID)
        self._processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_ID)
        self._model = CLIPModel.from_pretrained(_CLIP_MODEL_ID).to(self._device)
        self._model.eval()
        logger.info(
            "CLIP OOD model loaded (accept=%.2f  reject=%.2f)",
            self.accept_threshold, self.reject_threshold,
        )

    def check(self, image_bytes: bytes) -> OodResult:
        if self._model is None:
            raise RuntimeError("ClipOodDetector.load() was not called at startup")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self._processor(
            text=_ALL_PROMPTS,
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # logits_per_image shape: [1, num_texts]
        probs = outputs.logits_per_image[0].softmax(dim=0)
        cxr_score = float(probs[:_N_CXR].sum())

        if cxr_score >= self.accept_threshold:
            decision, reason, is_cxr = "accept", "accepted", True
        elif cxr_score >= self.reject_threshold:
            decision, reason, is_cxr = "review", "ood_in_review_band", True
        else:
            decision, reason, is_cxr = "reject", "non_cxr_rejected", False

        return OodResult(
            decision=decision,
            cxr_score=round(cxr_score, 3),
            reason=reason,
            is_cxr=is_cxr,
        )


# ---------------------------------------------------------------------------
# Module-level singleton (None when OOD is disabled)
# ---------------------------------------------------------------------------

_detector: Optional[ClipOodDetector] = None


def init_ood(accept_threshold: float = 0.45, reject_threshold: float = 0.20) -> None:
    global _detector
    _detector = ClipOodDetector(accept_threshold=accept_threshold, reject_threshold=reject_threshold)
    _detector.load()


def get_ood_detector() -> Optional[ClipOodDetector]:
    return _detector


def is_available() -> bool:
    return _detector is not None
