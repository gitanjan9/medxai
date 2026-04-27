"""Threshold loading and decision logic for the serve layer."""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import torch

from src.common.logging import get_logger

logger = get_logger("serve.thresholds")

_THRESHOLD_VERSION = "current"


class Decision(NamedTuple):
    decision: str              # "positive" | "review" | "negative"
    confidence_band: str       # "high" | "medium" | "low"
    review_reason: str = ""    # populated only when decision == "review"


def load_thresholds(path: Path) -> list[dict]:
    """Load thresholds.json → list of per-class threshold dicts."""
    if not path.exists():
        logger.warning("Thresholds file not found at %s – all decisions = review", path)
        return []
    data = json.loads(path.read_text())
    thresholds = data.get("thresholds", [])
    logger.info("Loaded %d class thresholds from %s", len(thresholds), path)
    return thresholds


def apply_thresholds(
    calibrated_probs: torch.Tensor,
    predicted_class_idx: int,
    thresholds: list[dict],
) -> Decision:
    """Map calibrated probability for the predicted class → (decision, confidence_band).

    Decision logic:
    - score >= high  → "positive"
    - score >= low   → "review"
    - score < low    → "negative"

    Confidence band (based on calibrated score magnitude):
    - score >= 0.85  → "high"
    - score >= 0.65  → "medium"
    - else           → "low"
    """
    score = float(calibrated_probs[predicted_class_idx])

    # Find matching threshold entry
    entry = next(
        (t for t in thresholds if t.get("class_idx") == predicted_class_idx),
        None,
    )

    review_reason = ""

    if entry is None:
        decision = "review"
        review_reason = "no_threshold_entry"
    elif score >= entry["high"]:
        decision = "positive"
    elif score >= entry["low"]:
        decision = "review"
        review_reason = "below_high_threshold"
    else:
        decision = "negative"

    if score >= 0.85:
        confidence_band = "high"
    elif score >= 0.65:
        confidence_band = "medium"
    else:
        confidence_band = "low"

    return Decision(decision=decision, confidence_band=confidence_band, review_reason=review_reason)
