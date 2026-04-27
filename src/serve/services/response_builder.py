"""Assemble InferenceResponse from raw inference outputs.

All business logic (calibration, thresholds, OOD, pathology) is resolved
before calling these builders; they only format and assemble data.
"""
from __future__ import annotations

from typing import Optional

import torch

from src.serve.schemas.response import (
    BboxCoords,
    InferenceResponse,
    LocalizationSection,
    OodSection,
    PathologiesSection,
    PathologyFinding,
    PrimaryPrediction,
)
from src.serve.services.bbox_extractor import BoundingBox
from src.serve.services.ood_detector import OodResult
from src.serve.services.pathology_detector import PathologyResult
from src.serve.services.thresholds import Decision

_LOCALIZATION_DISCLAIMER = (
    "Approximate attention region derived from Grad-CAM gradient visualization. "
    "Not a clinically validated lesion boundary."
)


def build_primary_prediction(
    predicted_class: str,
    predicted_idx: int,
    raw_probs: torch.Tensor,
    calibrated_probs: torch.Tensor,
    decision: Decision,
    label_map: dict,
) -> PrimaryPrediction:
    all_scores = {
        label_map["idx_to_str"][i]: round(float(calibrated_probs[i]), 4)
        for i in range(len(calibrated_probs))
    }
    return PrimaryPrediction(
        label=predicted_class,
        raw_score=round(float(raw_probs[predicted_idx]), 4),
        calibrated_score=round(float(calibrated_probs[predicted_idx]), 4),
        all_scores=all_scores,
        decision=decision.decision,
        confidence_band=decision.confidence_band,
        review_reason=decision.review_reason,
        threshold_version="current",
    )


def build_ood_section(
    ood_result: Optional[OodResult],
    enabled: bool,
) -> OodSection:
    if not enabled:
        return OodSection(enabled=False, reason="disabled")
    if ood_result is None:
        return OodSection(enabled=True, reason="check_failed")
    return OodSection(
        enabled=True,
        score=ood_result.cxr_score,
        decision=ood_result.decision,
        reason=ood_result.reason,
    )


def build_pathologies_section(
    patho_result: Optional[PathologyResult],
    enabled: bool,
    error: bool = False,
) -> PathologiesSection:
    if not enabled:
        return PathologiesSection(enabled=False, status="disabled")
    if patho_result is None:
        status = "error" if error else "unavailable"
        return PathologiesSection(enabled=True, status=status)
    findings = [
        PathologyFinding(name=f, score=round(patho_result.scores.get(f, 0.0), 4))
        for f in patho_result.findings
    ]
    return PathologiesSection(
        enabled=True,
        status="ok",
        top_finding=patho_result.top_finding,
        top_score=patho_result.top_score,
        findings=findings,
    )


def build_localization_section(
    bbox: Optional[BoundingBox],
    enabled: bool,
    failed: bool = False,
) -> LocalizationSection:
    if not enabled:
        return LocalizationSection(enabled=False, type="not_generated")
    if bbox is None:
        return LocalizationSection(enabled=True, type="not_generated")
    return LocalizationSection(
        enabled=True,
        type="approximate_attention_region",
        bbox=BboxCoords(
            x1=bbox.x1, y1=bbox.y1, x2=bbox.x2, y2=bbox.y2,
            x1_norm=bbox.x1_norm, y1_norm=bbox.y1_norm,
            x2_norm=bbox.x2_norm, y2_norm=bbox.y2_norm,
            width=bbox.width, height=bbox.height,
            area_fraction=bbox.area_fraction,
        ),
        region=bbox.region,
        region_description=bbox.region_description,
        disclaimer=_LOCALIZATION_DISCLAIMER,
    )


def build_inference_response(
    request_id: str,
    model_version: str,
    primary: Optional[PrimaryPrediction],
    ood: OodSection,
    pathologies: PathologiesSection,
    localization: LocalizationSection,
    warnings: list[str],
    gradcam_b64: Optional[str] = None,
) -> InferenceResponse:
    status = "ok" if not warnings else "degraded"
    return InferenceResponse(
        request_id=request_id,
        model_version=model_version,
        status=status,
        primary_prediction=primary,
        ood=ood,
        pathologies=pathologies,
        localization=localization,
        warnings=warnings,
        gradcam_base64=gradcam_b64,
        gradcam_available=gradcam_b64 is not None,
    )
