"""Manual-review band logic.

A case is flagged for manual review when ANY of the following hold:

1. Predicted probability falls in the (low, high) threshold band.
2. Max calibrated confidence < ``confidence_threshold`` (weak prediction).
3. Grad-CAM failed or heatmap artifact is missing/empty.
4. Explanation metadata JSON is absent or malformed.

The ``ReviewDecision`` dataclass records which triggers fired, making the
output fully auditable and loadable from serving code on Day 3.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.common.logging import get_logger
from src.train.thresholds import ReviewBand, apply_threshold

logger = get_logger("review_logic")


# ---------------------------------------------------------------------------
# Decision dataclass
# ---------------------------------------------------------------------------


@dataclass
class ReviewDecision:
    sample_idx: int
    band: ReviewBand
    predicted_class: int
    predicted_label: str
    confidence: float
    # Trigger flags
    in_review_band: bool = False
    weak_calibration: bool = False
    gradcam_failed: bool = False
    explanation_missing: bool = False
    # Human-readable reasons list
    reasons: list[str] = field(default_factory=list)

    @property
    def requires_review(self) -> bool:
        return self.band == ReviewBand.REVIEW or bool(self.reasons)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


class ReviewLogic:
    """Stateless decision engine.  All parameters come from InferenceConfig.

    Intended to be instantiated once and called per sample / batch.
    """

    def __init__(
        self,
        thresholds: list[dict],           # loaded threshold artifact list
        confidence_threshold: float = 0.5,
        require_explanation: bool = True,
        flag_weak_calibration: bool = True,
    ) -> None:
        self._thresholds = thresholds
        self._confidence_threshold = confidence_threshold
        self._require_explanation = require_explanation
        self._flag_weak_calibration = flag_weak_calibration

    def decide(
        self,
        sample_idx: int,
        probs: np.ndarray,               # shape (C,)
        gradcam_failed: bool = False,
        explanation_path: Optional[str] = None,
    ) -> ReviewDecision:
        """Produce a ReviewDecision for one sample.

        Args:
            sample_idx:      Position in the dataset.
            probs:           Softmax (or calibrated) probability vector, shape (C,).
            gradcam_failed:  Whether Grad-CAM generation raised an exception.
            explanation_path: Path to the metadata JSON, or None if not generated.
        """
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        th = self._thresholds[pred_class]
        band = apply_threshold(confidence, th["low"], th["high"])
        pred_label = th["class_name"]

        reasons: list[str] = []
        in_review_band = band == ReviewBand.REVIEW
        weak_cal = False
        expl_missing = False

        if in_review_band:
            reasons.append(
                f"confidence {confidence:.3f} in review band "
                f"[{th['low']:.3f}, {th['high']:.3f})"
            )

        if self._flag_weak_calibration and confidence < self._confidence_threshold:
            weak_cal = True
            reasons.append(
                f"calibrated confidence {confidence:.3f} < "
                f"threshold {self._confidence_threshold:.3f}"
            )

        if gradcam_failed:
            reasons.append("grad-cam generation failed")

        if self._require_explanation:
            if explanation_path is None:
                expl_missing = True
                reasons.append("explanation artifact path is None")
            elif not Path(explanation_path).exists():
                expl_missing = True
                reasons.append(f"explanation file missing: {explanation_path}")
            else:
                # Validate the JSON is non-empty and well-formed
                try:
                    with open(explanation_path) as fh:
                        data = json.load(fh)
                    if not data:
                        expl_missing = True
                        reasons.append("explanation metadata JSON is empty")
                except Exception as exc:
                    expl_missing = True
                    reasons.append(f"explanation JSON invalid: {exc}")

        return ReviewDecision(
            sample_idx=sample_idx,
            band=band,
            predicted_class=pred_class,
            predicted_label=pred_label,
            confidence=confidence,
            in_review_band=in_review_band,
            weak_calibration=weak_cal,
            gradcam_failed=gradcam_failed,
            explanation_missing=expl_missing,
            reasons=reasons,
        )

    def decide_batch(
        self,
        probs: np.ndarray,               # shape (N, C)
        gradcam_failed_flags: Optional[list[bool]] = None,
        explanation_paths: Optional[list[Optional[str]]] = None,
    ) -> list[ReviewDecision]:
        """Vectorised wrapper over ``decide``."""
        n = probs.shape[0]
        gf = gradcam_failed_flags or [False] * n
        ep = explanation_paths or [None] * n
        return [
            self.decide(i, probs[i], gf[i], ep[i])
            for i in range(n)
        ]


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------


def build_review_logic_from_config(config_path: str) -> ReviewLogic:
    """Instantiate ReviewLogic from an inference.yaml config file."""
    from src.common.config import InferenceConfig
    from src.train.thresholds import load_thresholds

    cfg = InferenceConfig.from_yaml(config_path)
    th_path = cfg.thresholds_path
    if th_path is None or not Path(th_path).exists():
        raise FileNotFoundError(
            f"Threshold artifact not found at {th_path}. "
            "Run: python -m src.train.thresholds --config configs/train.yaml"
        )
    th_data = load_thresholds(th_path)
    return ReviewLogic(
        thresholds=th_data["thresholds"],
        confidence_threshold=cfg.review.confidence_threshold,
        require_explanation=cfg.review.require_explanation,
        flag_weak_calibration=cfg.review.flag_weak_calibration,
    )


# ---------------------------------------------------------------------------
# Batch review report
# ---------------------------------------------------------------------------


def run_review_pass(
    config_path: str,
    output_path: Optional[str] = None,
) -> list[ReviewDecision]:
    """Apply review logic to all explanation results from a prior run.

    Loads the explanations_summary.json produced by explainability.py and
    produces a per-sample ReviewDecision, writing a review_report.json.
    """
    from src.common.config import InferenceConfig
    from src.train.thresholds import load_thresholds

    cfg = InferenceConfig.from_yaml(config_path)
    logic = build_review_logic_from_config(config_path)

    summary_path = Path(cfg.explainability.output_dir) / "explanations_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Explanation summary not found: {summary_path}. "
            "Run explainability first."
        )
    with open(summary_path) as fh:
        summary = json.load(fh)

    decisions: list[ReviewDecision] = []
    for item in summary["results"]:
        # Reconstruct minimal probability vector
        # (confidence on predicted class only; full probs not stored in summary)
        n_classes = cfg.model.num_classes
        probs = np.zeros(n_classes, dtype=np.float32)
        pred = item["predicted_class"]
        probs[pred] = item["confidence"]
        # distribute remaining mass uniformly so sum = 1
        remainder = (1.0 - item["confidence"]) / max(n_classes - 1, 1)
        for c in range(n_classes):
            if c != pred:
                probs[c] = remainder

        decision = logic.decide(
            sample_idx=item["sample_idx"],
            probs=probs,
            gradcam_failed=item.get("gradcam_failed", False),
            explanation_path=item.get("metadata_path"),
        )
        decisions.append(decision)

    n_review = sum(d.requires_review for d in decisions)
    logger.info(
        "Review pass done: %d / %d cases flagged for review (%.1f%%)",
        n_review, len(decisions), 100 * n_review / max(len(decisions), 1),
    )

    out_path = Path(output_path) if output_path else (
        Path(cfg.explainability.output_dir) / "review_report.json"
    )
    report = {
        "total": len(decisions),
        "requires_review": n_review,
        "decisions": [asdict(d) for d in decisions],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    logger.info("Review report → %s", out_path)
    return decisions
