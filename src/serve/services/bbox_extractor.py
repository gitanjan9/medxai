"""Extract anatomical bounding boxes from GradCAM heatmaps.

Converts the continuous Grad-CAM activation map into:
- A tight bounding box around the high-activation region
- Normalised coordinates (0–1) for frontend overlays
- An anatomical region label based on position in a standard CXR layout

No additional model or training required — works directly on the existing
Grad-CAM output from run_gradcam_base64().
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Anatomical region heuristics (standard PA chest X-ray layout)
# Coordinates are (x1_norm, y1_norm, x2_norm, y2_norm) of each region
# ---------------------------------------------------------------------------

_REGIONS = {
    "right_upper_lobe":  (0.0,  0.0,  0.45, 0.45),
    "left_upper_lobe":   (0.55, 0.0,  1.0,  0.45),
    "right_mid_lobe":    (0.0,  0.30, 0.45, 0.65),
    "left_mid_lobe":     (0.55, 0.30, 1.0,  0.65),
    "right_lower_lobe":  (0.0,  0.55, 0.45, 1.0),
    "left_lower_lobe":   (0.55, 0.55, 1.0,  1.0),
    "mediastinum":       (0.35, 0.1,  0.65, 0.75),
    "cardiac_silhouette":(0.3,  0.3,  0.7,  0.85),
    "right_costophrenic":(0.0,  0.75, 0.45, 1.0),
    "left_costophrenic": (0.55, 0.75, 1.0,  1.0),
}


def _iou(box_a: tuple, box_b: tuple) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _label_region(x1_n: float, y1_n: float, x2_n: float, y2_n: float) -> str:
    """Return the anatomical region with highest IoU with the activation bbox."""
    pred_box = (x1_n, y1_n, x2_n, y2_n)
    best_region = "unspecified"
    best_iou = 0.0
    for name, ref_box in _REGIONS.items():
        score = _iou(pred_box, ref_box)
        if score > best_iou:
            best_iou = score
            best_region = name
    return best_region


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    x1_norm: float
    y1_norm: float
    x2_norm: float
    y2_norm: float
    width: int
    height: int
    area_fraction: float   # fraction of image covered by the bbox
    region: str            # anatomical region label
    region_description: str


_REGION_DESCRIPTIONS = {
    "right_upper_lobe":   "Right upper lobe — common site for TB, apical infiltrates",
    "left_upper_lobe":    "Left upper lobe — apical or perihilar opacity",
    "right_mid_lobe":     "Right middle lobe — lobar consolidation, atelectasis",
    "left_mid_lobe":      "Left mid zone — lingular segment opacity",
    "right_lower_lobe":   "Right lower lobe — pneumonia, effusion, atelectasis",
    "left_lower_lobe":    "Left lower lobe — basal consolidation, effusion",
    "mediastinum":        "Mediastinum — enlarged lymph nodes, cardiomegaly, mass",
    "cardiac_silhouette": "Cardiac silhouette — cardiomegaly, pericardial effusion",
    "right_costophrenic": "Right costophrenic angle — pleural effusion blunting",
    "left_costophrenic":  "Left costophrenic angle — pleural effusion blunting",
    "unspecified":        "Diffuse or unspecific activation across the image",
}


def extract_bbox_from_gradcam(
    gradcam_b64: str,
    threshold_fraction: float = 0.40,
) -> Optional[BoundingBox]:
    """Decode a base64 Grad-CAM PNG and extract a bounding box.

    Args:
        gradcam_b64: base64-encoded PNG of the Grad-CAM heatmap (H×W×3 or H×W).
        threshold_fraction: pixels above (max_activation * this) are included.

    Returns:
        BoundingBox or None if the heatmap has no clear activation.
    """
    try:
        raw = base64.b64decode(gradcam_b64)
        img = Image.open(io.BytesIO(raw)).convert("L")  # grayscale
        arr = np.array(img, dtype=np.float32)
    except Exception:
        return None

    h, w = arr.shape
    if arr.max() == 0:
        return None

    thresh = arr.max() * threshold_fraction
    mask = arr >= thresh

    rows_any = np.any(mask, axis=1)
    cols_any = np.any(mask, axis=0)
    if not rows_any.any() or not cols_any.any():
        return None

    y1 = int(np.where(rows_any)[0][0])
    y2 = int(np.where(rows_any)[0][-1])
    x1 = int(np.where(cols_any)[0][0])
    x2 = int(np.where(cols_any)[0][-1])

    x1_n = round(x1 / w, 3)
    y1_n = round(y1 / h, 3)
    x2_n = round(x2 / w, 3)
    y2_n = round(y2 / h, 3)

    region = _label_region(x1_n, y1_n, x2_n, y2_n)

    return BoundingBox(
        x1=x1, y1=y1, x2=x2, y2=y2,
        x1_norm=x1_n, y1_norm=y1_n, x2_norm=x2_n, y2_norm=y2_n,
        width=x2 - x1,
        height=y2 - y1,
        area_fraction=round(float(mask.sum()) / mask.size, 3),
        region=region,
        region_description=_REGION_DESCRIPTIONS.get(region, ""),
    )
