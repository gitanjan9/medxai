"""Production inference CLI for multi-label CXR classification.

Applies:
  1. Single-model or ensemble inference
  2. Temperature / isotonic calibration
  3. Class-specific thresholds from a validated thresholds.json
  4. Disagreement detection (ensemble mode)

Never uses a flat 0.50 cutoff — every class decision is driven by its
validated threshold from reports/thresholds.json.

Usage
-----
  python -m src.inference.predict \\
      --image   path/to/cxr.jpg \\
      --checkpoint artifacts/v3/checkpoints \\
      --label-map  artifacts/v3/label_map.json \\
      --thresholds reports/v3/thresholds.json \\
      --calibration artifacts/calibration.json

  # Ensemble mode:
  python -m src.inference.predict \\
      --image  path/to/cxr.jpg \\
      --registry models/registry.json --registry-tag production
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.common.logging import get_logger

logger = get_logger("inference.predict")

_REVIEW_MIN = 0.60


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_thresholds(path: str | Path) -> dict[str, float]:
    """Load class-specific positive thresholds from thresholds.json."""
    with open(path) as f:
        data = json.load(f)
    if "thresholds" in data:
        return {k: float(v) for k, v in data["thresholds"].items()}
    return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}


def classify(
    probs: np.ndarray,
    label_cols: list[str],
    thresholds: dict[str, float],
    default_threshold: float = 0.75,
) -> dict:
    """Apply class-specific thresholds to produce a structured prediction.

    Args:
        probs:           (C,) calibrated probabilities.
        label_cols:      Ordered list of C class names.
        thresholds:      Dict mapping class name → positive threshold.
        default_threshold: Used for classes not in thresholds dict.

    Returns:
        Dict with keys: positive_findings, review_findings, all_scores,
        top_label, top_score, status.
    """
    positive, review, all_scores = [], [], {}

    for i, cls in enumerate(label_cols):
        score = float(probs[i])
        thr = thresholds.get(cls, default_threshold)
        all_scores[cls] = round(score, 4)
        if score >= thr:
            positive.append({"name": cls, "score": round(score, 4), "threshold": thr})
        elif score >= _REVIEW_MIN:
            review.append({"name": cls, "score": round(score, 4), "threshold": thr})

    positive.sort(key=lambda x: x["score"], reverse=True)
    review.sort(key=lambda x: x["score"], reverse=True)

    non_nf = [(cls, float(probs[i]))
               for i, cls in enumerate(label_cols) if cls != "No Finding"]
    non_nf.sort(key=lambda x: x[1], reverse=True)
    top_score = non_nf[0][1] if non_nf else 0.0
    top_5 = [s for _, s in non_nf[:5]]
    spread = max(top_5) - min(top_5) if len(top_5) >= 2 else 1.0

    if positive:
        status = "positive"
        top_label = positive[0]["name"]
        top_score = positive[0]["score"]
    elif top_score < 0.70 or (spread < 0.10):
        status = "likely_normal_or_uncertain"
        top_label = "No Confident Finding"
    elif review:
        status = "review_required"
        top_label = review[0]["name"]
        top_score = review[0]["score"]
    else:
        status = "negative"
        top_label = "No Finding"

    return {
        "status": status,
        "top_label": top_label,
        "top_score": round(top_score, 4),
        "positive_findings": positive,
        "review_findings": review,
        "all_scores": all_scores,
        "disclaimer": (
            "AI finding is not a medical diagnosis. "
            "Scores below class threshold should not be treated as confirmed pathology."
        ),
    }


def predict_single(
    image_path: str | Path,
    checkpoint_dir: str | Path,
    label_map_json: str | Path,
    thresholds_json: str | Path,
    calibration_json: str | Path | None = None,
    image_size: int = 320,
    device_str: str = "auto",
) -> dict:
    """Run single-model inference on one image."""
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    with open(label_map_json) as f:
        lmap = json.load(f)
    label_cols: list[str] = lmap.get("classes", list(lmap.get("str_to_idx", {}).keys()))

    temperature = 1.0
    if calibration_json and Path(calibration_json).exists():
        with open(calibration_json) as f:
            cal = json.load(f)
        temperature = float(cal.get("temperature", 1.0))
        logger.info("Calibration temperature: T=%.4f", temperature)

    from src.serve.services.model_loader import load_model, resolve_best_checkpoint
    ckpt = resolve_best_checkpoint(Path(checkpoint_dir))
    model = load_model("efficientnet_b3", len(label_cols), ckpt, device)
    model.eval()

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    from src.serve.services.preprocessing import decode_image
    tensor = decode_image(image_bytes, image_size=(image_size, image_size))

    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(device)).squeeze(0).cpu().float().numpy()

    probs = _sigmoid(logits / max(temperature, 0.01))
    thresholds = load_thresholds(thresholds_json)
    result = classify(probs, label_cols, thresholds)
    result["model"] = "single"
    result["checkpoint"] = str(ckpt)
    return result


def predict_ensemble(
    image_path: str | Path,
    registry_path: str | Path,
    thresholds_json: str | Path,
    registry_tag: str = "production",
    image_size: int = 320,
) -> dict:
    """Run ensemble inference on one image."""
    from src.ml.ensemble import CxrEnsemble

    ens = CxrEnsemble.from_registry(registry_path, tag=registry_tag)
    ens.load()

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    probs, meta = ens.predict_bytes(image_bytes, image_size=image_size)
    label_cols = ens.members[0].label_cols
    thresholds = load_thresholds(thresholds_json)
    result = classify(probs, label_cols, thresholds)

    if meta["disagreement"] == "high":
        result["status"] = "review_required"
        result["review_reason"] = (
            f"Model disagreement detected (std={meta['disagreement_score']:.3f}). "
            "Manual review recommended."
        )

    result["model"] = "ensemble"
    result["ensemble_meta"] = meta
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CXR multi-label inference.")
    p.add_argument("--image",       required=True)
    p.add_argument("--thresholds",  required=True, help="thresholds.json from evaluate.py")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint",  help="Checkpoint dir (single-model mode)")
    group.add_argument("--registry",    help="models/registry.json (ensemble mode)")
    p.add_argument("--label-map",   help="Required for single-model mode")
    p.add_argument("--calibration", help="calibration.json (optional)")
    p.add_argument("--registry-tag", default="production")
    p.add_argument("--image-size",  type=int, default=320)
    p.add_argument("--device",      default="auto")
    args = p.parse_args()

    if args.checkpoint:
        if not args.label_map:
            p.error("--label-map is required for single-model mode")
        out = predict_single(
            image_path=args.image,
            checkpoint_dir=args.checkpoint,
            label_map_json=args.label_map,
            thresholds_json=args.thresholds,
            calibration_json=args.calibration,
            image_size=args.image_size,
            device_str=args.device,
        )
    else:
        out = predict_ensemble(
            image_path=args.image,
            registry_path=args.registry,
            thresholds_json=args.thresholds,
            registry_tag=args.registry_tag,
            image_size=args.image_size,
        )

    import sys
    json.dump(out, sys.stdout, indent=2)
    print()
