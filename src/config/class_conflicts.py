"""Clinical conflict rules and multi-head class groupings for CXR pathology.

Conflict rules encode pathophysiological mutual exclusivity:
- Emphysema causes hyperinflation (decreased density, large lung volumes)
- Fibrosis causes restriction (increased density, small lung volumes)
- These rarely co-exist acutely and the DenseNet confuses them.

Each conflict is (driver, target, penalty, driver_trigger_score):
  If driver_score >= driver_trigger_score:
    adjusted_score[target] = raw_score[target] - penalty * driver_score

Multi-head groups reflect the clinical diagnostic workflow:
  HEAD 1 – Interstitial / restrictive disease
  HEAD 2 – Obstructive / structural disease
  HEAD 3 – Acute opacity / consolidation / collapse
  HEAD 4 – Structural / mass lesions
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConflictRule:
    driver: str          # the class whose high score triggers the penalty
    target: str          # the class whose score gets penalised
    penalty: float       # subtracted from target_score: adj = raw - penalty * driver_score
    driver_trigger: float  # driver must be >= this to apply the rule


CONFLICT_RULES: list[ConflictRule] = [
    # Emphysema (hyperinflation) vs Fibrosis (restriction)
    ConflictRule("Emphysema",     "Fibrosis",          0.45, 0.55),
    # Bullous/emphysematous lung vs acute opacity
    ConflictRule("Emphysema",     "Consolidation",     0.25, 0.58),
    ConflictRule("Emphysema",     "Lung Opacity",      0.20, 0.55),
    # Pneumothorax (collapsed lung, air) vs fibrosis / consolidation
    ConflictRule("Pneumothorax",  "Fibrosis",          0.50, 0.62),
    ConflictRule("Pneumothorax",  "Consolidation",     0.30, 0.62),
    ConflictRule("Pneumothorax",  "Infiltration",      0.25, 0.62),
    # Atelectasis (collapse) vs emphysema (hyperinflation)
    ConflictRule("Atelectasis",   "Emphysema",         0.35, 0.65),
    # Pneumothorax (pleural air) should suppress Emphysema — PTX is a medical emergency
    # and must never be down-ranked by a co-firing Emphysema score
    ConflictRule("Pneumothorax",  "Emphysema",         0.45, 0.45),
    # Hyperinflated emphysematous lung creates false mass/nodule signals from bullous spaces
    ConflictRule("Emphysema",     "Mass",              0.30, 0.58),
    ConflictRule("Emphysema",     "Nodule",            0.25, 0.58),
    # Pleural thickening in emphysema context is often artefactual
    ConflictRule("Emphysema",     "Pleural Thickening",0.20, 0.58),
    # Fibrosis (chronic) vs acute edema
    ConflictRule("Fibrosis",      "Edema",             0.20, 0.65),
    # Cardiomegaly vs Pneumothorax (different distributions)
    ConflictRule("Cardiomegaly",  "Pneumothorax",      0.20, 0.70),
]


# ── Multi-head clinical groups ────────────────────────────────────────────────

CLINICAL_GROUPS: dict[str, list[str]] = {
    "Interstitial / Restrictive": [
        "Fibrosis",
        "Edema",
        "Infiltration",
        "Lung Opacity",
        "Pleural Thickening",
        "Pleural Other",
    ],
    "Obstructive / Structural": [
        "Emphysema",
        "Atelectasis",
        "Pneumothorax",
        "Effusion",
        "Pleural Effusion",
        "Enlarged Cardiomediastinum",
    ],
    "Acute Opacity / Consolidation": [
        "Pneumonia",
        "Consolidation",
        "Lung Lesion",
    ],
    "Mass / Nodular": [
        "Mass",
        "Nodule",
        "Hernia",
        "Fracture",
    ],
    "Cardiac": [
        "Cardiomegaly",
    ],
}

# Reverse lookup: class → group name
CLASS_TO_GROUP: dict[str, str] = {
    cls: grp for grp, classes in CLINICAL_GROUPS.items() for cls in classes
}
