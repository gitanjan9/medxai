"""Radiological feature extractor for Pneumothorax / Emphysema / Bullous distinction.

Extracts three image-level signals from a CXR that distinguish these three
conditions — all of which the DenseNet confuses because they share "dark lung"
characteristics:

  PNEUMOTHORAX  – visceral pleural line present + absent vascular markings
                  in a peripheral wedge region
  EMPHYSEMA     – global hyperinflation (flat diaphragm) + globally reduced
                  but present vascular markings throughout both lungs
  BULLOUS       – focal avascular zone(s) within otherwise marked lung fields

Signals computed (all in [0, 1]):
  pleural_line_score    – confidence that a sharp peripheral pleural line exists
  vascular_density      – normalised vessel-marking density (high = normal lung)
  hyperinflation_score  – flat/depressed diaphragm → emphysema
  focal_avascular_score – small avascular patch within marked lung → bullae
  peripheral_dark_zone  – large peripheral avascular wedge → pneumothorax

Usage
-----
  from src.serve.services.pleural_analyzer import analyze_pleural_features
  feats = analyze_pleural_features(image_bytes)
  # feats.ptx_evidence, feats.emphysema_evidence, feats.bullous_evidence
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from src.common.logging import get_logger

logger = get_logger("serve.pleural_analyzer")


@dataclass
class PleuralFeatures:
    pleural_line_score: float     # 0-1, high → sharp peripheral line (PTX)
    vascular_density: float       # 0-1, high → dense vessel markings (normal/consolidation)
    hyperinflation_score: float   # 0-1, high → flat diaphragm (emphysema)
    focal_avascular_score: float  # 0-1, high → focal dark patch (bullous)
    peripheral_dark_zone: float   # 0-1, high → large peripheral darkness (PTX)

    @property
    def ptx_evidence(self) -> float:
        """Combined evidence score for pneumothorax."""
        return float(
            0.40 * self.pleural_line_score
            + 0.35 * self.peripheral_dark_zone
            + 0.25 * (1.0 - self.vascular_density)
        )

    @property
    def emphysema_evidence(self) -> float:
        """Combined evidence score for emphysema."""
        return float(
            0.55 * self.hyperinflation_score
            + 0.30 * (1.0 - self.vascular_density)
            + 0.15 * (1.0 - self.focal_avascular_score)
        )

    @property
    def bullous_evidence(self) -> float:
        """Combined evidence score for bullous disease."""
        return float(
            0.55 * self.focal_avascular_score
            + 0.30 * (1.0 - self.vascular_density)
            + 0.15 * (1.0 - self.peripheral_dark_zone)
        )

    @property
    def subtype(self) -> str:
        """Best-guess subtype label."""
        scores = {
            "Pneumothorax": self.ptx_evidence,
            "Emphysema":    self.emphysema_evidence,
            "Bullous":      self.bullous_evidence,
        }
        return max(scores, key=scores.get)

    def to_dict(self) -> dict:
        return {
            "pleural_line_score":    round(self.pleural_line_score, 3),
            "vascular_density":      round(self.vascular_density, 3),
            "hyperinflation_score":  round(self.hyperinflation_score, 3),
            "focal_avascular_score": round(self.focal_avascular_score, 3),
            "peripheral_dark_zone":  round(self.peripheral_dark_zone, 3),
            "ptx_evidence":          round(self.ptx_evidence, 3),
            "emphysema_evidence":    round(self.emphysema_evidence, 3),
            "bullous_evidence":      round(self.bullous_evidence, 3),
            "subtype":               self.subtype,
        }


def _decode_to_gray(image_bytes: bytes) -> "np.ndarray":
    try:
        import cv2
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("cv2.imdecode returned None")
        return img.astype(np.float32) / 255.0
    except ImportError:
        # Fallback: PIL
        import io
        from PIL import Image, ImageOps
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        return np.array(img, dtype=np.float32) / 255.0


def _pleural_line_score(gray: "np.ndarray") -> float:
    """Detect a sharp peripheral line along the lung apex.

    Pneumothorax shows a thin bright line (visceral pleura) separated from
    the chest wall by a dark avascular space.
    Uses Canny edges + Hough line probability in the upper-lateral quadrants.
    """
    try:
        import cv2
        H, W = gray.shape
        img8 = (gray * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(img8, (3, 3), 0)
        edges = cv2.Canny(blurred, 30, 90)

        # Focus on upper-lateral regions (apex of lungs)
        roi = np.zeros_like(edges)
        roi[:H // 2, :] = edges[:H // 2, :]
        lines = cv2.HoughLinesP(
            roi, 1, np.pi / 180,
            threshold=40,
            minLineLength=W // 5,
            maxLineGap=15,
        )
        if lines is None:
            return 0.0
        # Score by total length of long near-horizontal lines
        total_len = 0.0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 35 or angle > 145:  # near-horizontal
                total_len += np.hypot(x2 - x1, y2 - y1)
        return float(min(1.0, total_len / (W * 1.5)))
    except Exception as exc:
        logger.debug("pleural_line_score failed: %s", exc)
        return 0.0


def _vascular_density(gray: "np.ndarray") -> float:
    """Measure vascular marking density via Laplacian variance.

    Normal lung has high-frequency vessel markings → high Laplacian σ.
    Air-filled regions (PTX, bullae) have very low high-frequency content.
    """
    try:
        import cv2
        H, W = gray.shape
        img8 = (gray * 255).astype(np.uint8)
        # Restrict to lung fields: central 60% horizontally, 20-85% vertically
        lung_roi = img8[int(H * 0.20):int(H * 0.85), int(W * 0.20):int(W * 0.80)]
        lap = cv2.Laplacian(lung_roi.astype(np.float32), cv2.CV_32F)
        density = float(np.std(lap))
        return float(min(1.0, density / 18.0))   # empirically normalised
    except Exception as exc:
        logger.debug("vascular_density failed: %s", exc)
        return 0.5


def _hyperinflation_score(gray: "np.ndarray") -> float:
    """Estimate diaphragm flatness as a hyperinflation proxy.

    Normal diaphragm is a prominent dome; emphysema flattens it.
    Metric: vertical position of max row-mean brightness in lower lung.
    A lower (more caudal) peak means flatter diaphragm = hyperinflation.
    """
    try:
        H, W = gray.shape
        lower = gray[int(H * 0.55):int(H * 0.90), int(W * 0.10):int(W * 0.90)]
        row_means = lower.mean(axis=1)
        if len(row_means) == 0:
            return 0.3
        peak_rel = float(np.argmax(row_means)) / max(len(row_means) - 1, 1)
        # peak_rel near 0 → dome high up (normal)
        # peak_rel near 1 → dome pushed down (hyperinflated)
        return float(peak_rel)
    except Exception as exc:
        logger.debug("hyperinflation_score failed: %s", exc)
        return 0.3


def _focal_avascular_score(gray: "np.ndarray") -> float:
    """Detect focal avascular zones (bullae signature).

    Bullae are small, focal, well-defined very dark patches within
    otherwise normally-marked lung. Pneumothorax is a large wedge.
    Score = fraction of lung with abnormally low intensity
    weighted by patch-count (many small patches → bullae).
    """
    try:
        import cv2
        H, W = gray.shape
        img8 = (gray * 255).astype(np.uint8)
        lung = img8[int(H * 0.10):int(H * 0.85), int(W * 0.10):int(W * 0.90)]
        # Threshold for very dark regions (air pockets)
        dark_thr = int(lung.mean() * 0.50)
        _, dark_mask = cv2.threshold(lung, dark_thr, 255, cv2.THRESH_BINARY_INV)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask)
        lung_area = lung.shape[0] * lung.shape[1]
        small_patches = []
        for i in range(1, n_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if 200 < area < lung_area // 10:   # small to medium patch
                small_patches.append(area)
        patch_coverage = sum(small_patches) / lung_area if small_patches else 0.0
        patch_count_score = min(1.0, len(small_patches) / 5.0)
        return float(min(1.0, 0.5 * patch_coverage * 10 + 0.5 * patch_count_score))
    except Exception as exc:
        logger.debug("focal_avascular_score failed: %s", exc)
        return 0.0


def _peripheral_dark_zone(gray: "np.ndarray") -> float:
    """Detect a large peripheral dark zone (pneumothorax signature).

    PTX = large uniformly dark region at lung apex or lateral periphery
    with a sharp medial boundary (the pleural line).
    """
    try:
        H, W = gray.shape
        # Check: top 25% and lateral 15% strips
        apex  = gray[:H // 4, int(W * 0.15):int(W * 0.85)]
        lat_l = gray[int(H * 0.1):int(H * 0.7), :int(W * 0.12)]
        lat_r = gray[int(H * 0.1):int(H * 0.7), int(W * 0.88):]
        regions = [apex, lat_l, lat_r]
        scores = []
        for r in regions:
            if r.size == 0:
                continue
            # Very dark → air, not lung tissue
            dark_frac = float((r < 0.25).mean())
            scores.append(dark_frac)
        return float(min(1.0, max(scores) * 3.0)) if scores else 0.0
    except Exception as exc:
        logger.debug("peripheral_dark_zone failed: %s", exc)
        return 0.0


def analyze_pleural_features(image_bytes: bytes) -> PleuralFeatures:
    """Run all pleural feature extractors on raw image bytes.

    Safe: returns neutral 0.5 PleuralFeatures if any stage fails.
    """
    try:
        gray = _decode_to_gray(image_bytes)
        return PleuralFeatures(
            pleural_line_score=_pleural_line_score(gray),
            vascular_density=_vascular_density(gray),
            hyperinflation_score=_hyperinflation_score(gray),
            focal_avascular_score=_focal_avascular_score(gray),
            peripheral_dark_zone=_peripheral_dark_zone(gray),
        )
    except Exception as exc:
        logger.warning("analyze_pleural_features failed, returning neutral features: %s", exc)
        return PleuralFeatures(
            pleural_line_score=0.0,
            vascular_density=0.5,
            hyperinflation_score=0.5,
            focal_avascular_score=0.0,
            peripheral_dark_zone=0.0,
        )
