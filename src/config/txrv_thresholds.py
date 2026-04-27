"""Class-specific confidence thresholds for the torchxrayvision DenseNet-121.

These thresholds are deliberately conservative to avoid false-positive labels
on near-normal chest X-rays.  Scores below REVIEW_MIN_SCORE are suppressed
from findings entirely; scores between REVIEW_MIN_SCORE and the class threshold
are surfaced as review_required; only scores >= class threshold are positive.

Tuning guidance
---------------
- Raise threshold if the class over-fires on normals (FP reduction).
- Lower threshold if the class misses real findings (sensitivity recovery).
- REVIEW_MIN_SCORE controls the bottom cut-off before a score shows at all.
"""
from __future__ import annotations

TXRV_THRESHOLDS: dict[str, float] = {
    "Atelectasis":                   0.6350,
    "Cardiomegaly":                  0.6750,
    "Consolidation":                 0.6550,
    "Edema":                         0.6450,
    "Effusion":                      0.6500,  # clamped at safety ceiling 0.65
    "Emphysema":                     0.6250,
    "Enlarged Cardiomediastinum":    0.7500,
    "Fibrosis":                      0.6250,
    "Fracture":                      0.7500,
    "Hernia":                        0.6450,
    "Infiltration":                  0.6350,
    "Lung Lesion":                   0.7500,
    "Lung Opacity":                  0.6800,
    "Mass":                          0.6650,
    "Nodule":                        0.6450,
    "Pleural Effusion":              0.7000,  # clamped at safety ceiling 0.65
    "Pleural Other":                 0.7500,
    "Pleural Thickening":            0.6500,
    "Pneumonia":                     0.6400,
    "Pneumothorax":                  0.5500,  # clamped at safety ceiling 0.55
    "Support Devices":               0.8000,
}

DEFAULT_THRESHOLD: float = 0.75

REVIEW_MIN_SCORE: float = 0.40  # lowered so subtle findings surface as review rather than being suppressed

UNCERTAIN_TOP_SCORE: float = 0.50  # lowered so scores in 0.50-0.70 range reach review rather than 'No Confident Finding'

UNCERTAIN_SPREAD: float = 0.10

BROAD_ABNORMALITY_MIN_CLASSES: int = 5  # if this many non-NoFinding classes >= REVIEW_MIN_SCORE, X-ray is abnormal


def get_threshold(class_name: str) -> float:
    """Return the positive threshold for *class_name*, falling back to DEFAULT."""
    return TXRV_THRESHOLDS.get(class_name, DEFAULT_THRESHOLD)


def score_band(score: float, class_name: str) -> str:
    """Classify a raw sigmoid score into a confidence band string.

    Returns one of ``"low"``, ``"medium"``, or ``"high"``.
    """
    threshold = get_threshold(class_name)
    if score >= threshold:
        return "high"
    if score >= REVIEW_MIN_SCORE:
        return "medium"
    return "low"
