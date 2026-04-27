"""Response schemas for all inference endpoints.

Schema design contract (for frontend developers):
─────────────────────────────────────────────────
  primary_prediction  → project-specific 5-class triage model output
  ood                 → CLIP-based CXR authenticity check (optional feature)
  pathologies         → torchxrayvision 18-class auxiliary findings (optional feature)
  localization        → approximate Grad-CAM attention region (explain only)
  warnings            → list of string codes; frontend should surface these
  status              → "ok" | "degraded"
                        "ok"       – all enabled features completed normally
                        "degraded" – primary ran but ≥1 feature warned or is in review band

OOD decision codes:
  "accept"  – score ≥ accept_threshold (image is clearly a CXR)
  "review"  – reject_threshold ≤ score < accept_threshold (uncertain; inference still ran)
  "reject"  – score < reject_threshold (not a CXR; HTTP 422 is returned instead)

OOD reason codes:
  "accepted"            "ood_in_review_band"
  "non_cxr_rejected"    "disabled"    "check_failed"

Pathology status codes:
  "ok"  "disabled"  "unavailable"  "error"

Localization type codes:
  "approximate_attention_region"  "not_generated"
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


# ── Probe / admin schemas ────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    ready: bool
    model_version: str
    details: dict[str, bool]


class AdminReloadResponse(BaseModel):
    reloaded: list[str]
    details: dict
    request_id: str


class ErrorResponse(BaseModel):
    detail: str
    request_id: Optional[str] = None


# ── Primary prediction ───────────────────────────────────────────────────────

class PrimaryPrediction(BaseModel):
    label: str
    raw_score: float
    calibrated_score: float
    all_scores: dict[str, float]
    decision: str           # "positive"|"review_required"|"review"|"negative"|"likely_normal_or_uncertain"
    confidence_band: str    # "high" | "medium" | "low"
    review_reason: str      # "" | "below_high_threshold" | "no_threshold_entry"
    threshold_version: str
    threshold: Optional[float] = None           # class-specific positive threshold
    positive_findings: list[PathologyFinding] = []  # scores >= class threshold
    review_findings: list[PathologyFinding] = []    # REVIEW_MIN_SCORE <= score < threshold
    clinical_groups: dict[str, list[dict]] = {}     # multi-head grouped findings
    conflict_log: list[str] = []                    # conflict penalties applied
    model_explanation: Optional[dict] = None        # full explanation report


# ── OOD section ─────────────────────────────────────────────────────────────

class OodSection(BaseModel):
    enabled: bool
    score: Optional[float] = None
    decision: Optional[str] = None  # "accept" | "review" | "reject" | None
    reason: str = ""


# ── Pathology section ────────────────────────────────────────────────────────

class PathologyFinding(BaseModel):
    name: str
    score: float


class PathologiesSection(BaseModel):
    enabled: bool
    status: str = "disabled"        # "ok" | "disabled" | "unavailable" | "error"
    top_finding: Optional[str] = None
    top_score: Optional[float] = None
    findings: list[PathologyFinding] = []


# ── Localization section ─────────────────────────────────────────────────────

class BboxCoords(BaseModel):
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
    area_fraction: float


class LocalizationSection(BaseModel):
    enabled: bool
    type: str = "not_generated"     # "approximate_attention_region" | "not_generated"
    bbox: Optional[BboxCoords] = None
    region: Optional[str] = None
    region_description: str = ""
    disclaimer: str = ""


# ── Unified inference response ───────────────────────────────────────────────

class InferenceResponse(BaseModel):
    """Single contract for both /v1/predict and /v1/explain.

    localization is always enabled=False for /v1/predict.
    gradcam_base64 / gradcam_available are only set by /v1/explain.
    """
    request_id: str
    model_version: str
    status: str                             # "ok" | "degraded"
    primary_prediction: Optional[PrimaryPrediction] = None
    ood: OodSection
    pathologies: PathologiesSection
    localization: LocalizationSection
    warnings: list[str] = []
    gradcam_base64: Optional[str] = None    # /v1/explain only
    gradcam_available: bool = False         # /v1/explain only
    db_id: Optional[str] = None             # DB record UUID for feedback


# ── Standalone pathology endpoint ────────────────────────────────────────────

class PathologyResponse(BaseModel):
    scores: dict[str, float]
    findings: list[PathologyFinding]
    top_finding: str
    top_score: float
    no_finding_score: float
    request_id: str
