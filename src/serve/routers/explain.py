"""POST /v1/explain – Grad-CAM explainability endpoint."""
from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile

from src.common.logging import get_logger
from src.serve.auth.dependencies import get_current_user
from src.serve.dependencies import AppState, EnvConfig, get_app_state, get_env_config, get_request_id
from src.serve.schemas.response import InferenceResponse
from src.serve.services.prediction_store import save_prediction
from src.serve.services.audit import AuditEntry, log_audit
from src.serve.services.bbox_extractor import extract_bbox_from_gradcam
from src.serve.services.calibration import apply_calibration
from src.serve.services.explainability import run_gradcam_base64
from src.serve.services.inference import run_inference
from src.serve.services import ood_detector as _ood, pathology_detector as _patho
from src.serve.services.ood_detector import OodResult
from src.serve.services.preprocessing import decode_image
from src.serve.services.response_builder import (
    build_inference_response,
    build_localization_section,
    build_ood_section,
    build_pathologies_section,
    build_primary_prediction,
)
from src.serve.services.thresholds import apply_thresholds
from src.serve.services.txrv_primary_adapter import txrv_to_primary_prediction

logger = get_logger("serve.explain")

router = APIRouter(prefix="/v1", tags=["explainability"])

_MAX_FILE_BYTES = 20 * 1024 * 1024


@router.post("/explain", response_model=InferenceResponse)
async def explain(
    request: Request,
    file: UploadFile = File(...),
    target_class: Optional[int] = Query(default=None, description="Override predicted class index"),
    state: AppState = Depends(get_app_state),
    cfg: EnvConfig = Depends(get_env_config),
    request_id: str = Depends(get_request_id),
) -> InferenceResponse:
    """Return prediction + approximate Grad-CAM localization for the uploaded image.

    Localization type is always "approximate_attention_region" — it shows where
    the model focused, NOT a clinically validated lesion boundary.
    """
    txrv_primary = cfg.primary_model == "txrv"
    if not state.checkpoint_ok:
        raise HTTPException(status_code=503, detail="Model not ready")
    if state.gradcam is None and not txrv_primary:
        raise HTTPException(status_code=501, detail="Explainability not initialised")

    data = await file.read()
    if len(data) > _MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 20 MB)")

    logger.info("explain request_id=%s  bytes=%d", request_id, len(data))
    t0 = time.perf_counter()
    warnings: list[str] = []

    # ── OOD check (same three-band logic as /v1/predict) ───────────────────
    ood_result: Optional[OodResult] = None
    ood_enabled = _ood.is_available()
    if ood_enabled:
        try:
            ood_result = _ood.get_ood_detector().check(data)
            if ood_result.decision == "reject":
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Image rejected by OOD check (CXR score={ood_result.cxr_score:.2f}). "
                        f"Reason: {ood_result.reason}."
                    ),
                )
            if ood_result.decision == "review":
                warnings.append("ood_in_review_band")
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("OOD check failed request_id=%s: %s", request_id, exc)
            ood_result = None
            warnings.append("ood_check_failed")

    # ── Preprocess (always needed – EfficientNet GradCAM requires tensor) ──────
    try:
        tensor = decode_image(data, image_size=state.image_size)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {exc}") from exc

    patho_result = None
    patho_error = False

    if txrv_primary:
        # ── TXRv primary path ───────────────────────────────────────────────
        if not _patho.is_available():
            raise HTTPException(status_code=503, detail="TXRv model not loaded")
        try:
            patho_result = _patho.get_pathology_detector().predict(data)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"TXRv inference failed: {exc}") from exc
        from src.serve.services.pleural_analyzer import analyze_pleural_features
        pleural_feats = analyze_pleural_features(data)
        primary = txrv_to_primary_prediction(patho_result, pleural_feats)
        # GradCAM: run EfficientNet for attention map only (if checkpoint loaded)
        if state.model is not None:
            eff_logits, _ = run_inference(state.model, tensor, state.device)
            eff_probs = apply_calibration(eff_logits, state.temperature)
            cam_target = target_class if target_class is not None else int(eff_probs.argmax())
        else:
            cam_target = 0

    else:
        # ── EfficientNet primary path ───────────────────────────────────────
        logits, raw_probs = run_inference(state.model, tensor, state.device)
        calibrated_probs = apply_calibration(logits, state.temperature)
        predicted_idx = int(calibrated_probs.argmax())
        predicted_class = state.label_map["idx_to_str"][predicted_idx]
        cam_target = target_class if target_class is not None else predicted_idx
        decision = apply_thresholds(calibrated_probs, predicted_idx, state.thresholds)
        primary = build_primary_prediction(
            predicted_class=predicted_class,
            predicted_idx=predicted_idx,
            raw_probs=raw_probs,
            calibrated_probs=calibrated_probs,
            decision=decision,
            label_map=state.label_map,
        )
        if _patho.is_available():
            try:
                patho_result = _patho.get_pathology_detector().predict(data)
            except Exception as exc:
                logger.warning("Pathology detection failed request_id=%s: %s", request_id, exc)
                patho_error = True
                warnings.append("pathology_inference_failed")

    # ── Grad-CAM ─────────────────────────────────────────────────────────────
    gradcam_b64 = None
    if state.gradcam is not None:
        gradcam_b64 = run_gradcam_base64(
            state.gradcam, tensor, cam_target, output_size=state.image_size,
        )

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info(
        "explain request_id=%s  label=%s  gradcam=%s  ood=%s  mode=%s  ms=%.1f",
        request_id, primary.label,
        "ok" if gradcam_b64 else "failed",
        ood_result.decision if ood_result else "n/a",
        "txrv" if txrv_primary else "efficientnet",
        latency_ms,
    )

    # ── Localization ─────────────────────────────────────────────────────────
    bbox = None
    localization_failed = False
    if cfg.localization_enabled and gradcam_b64:
        try:
            bbox = extract_bbox_from_gradcam(gradcam_b64)
        except Exception as exc:
            logger.warning("BBox extraction failed request_id=%s: %s", request_id, exc)
            localization_failed = True
    if localization_failed or (cfg.localization_enabled and gradcam_b64 and bbox is None):
        warnings.append("localization_generation_failed")
    if cfg.localization_enabled and gradcam_b64:
        warnings.append("approximate_localization_only")

    # ── Build sections ──────────────────────────────────────────────────────
    ood_section = build_ood_section(ood_result, enabled=ood_enabled)
    patho_section = build_pathologies_section(
        patho_result,
        enabled=txrv_primary or _patho.is_available(),
        error=patho_error,
    )
    localization_section = build_localization_section(
        bbox=bbox,
        enabled=cfg.localization_enabled,
        failed=localization_failed,
    )

    # ── Audit ───────────────────────────────────────────────────────────────
    log_audit(
        AuditEntry(
            request_id=request_id,
            endpoint="/v1/explain",
            prediction=primary.label,
            calibrated_score=primary.calibrated_score,
            decision=primary.decision,
            latency_ms=latency_ms,
            model_version=state.model_version,
            environment=cfg.environment,
            confidence_band=primary.confidence_band,
            review_reason=primary.review_reason,
            threshold_version=primary.threshold_version,
            explanation_requested=True,
            ood_score=ood_result.cxr_score if ood_result else None,
            ood_decision=ood_result.decision if ood_result else "",
            ood_reason=ood_result.reason if ood_result else "",
            top_pathology=patho_result.top_finding if patho_result else "",
            top_pathology_score=patho_result.top_score if patho_result else None,
            localization_generated=bbox is not None,
            localization_region=bbox.region if bbox else "",
        ),
        cfg.audit_path,
    )

    response_data = build_inference_response(
        request_id=request_id,
        model_version=state.model_version,
        primary=primary,
        ood=ood_section,
        pathologies=patho_section,
        localization=localization_section,
        warnings=warnings,
        gradcam_b64=gradcam_b64,
    )

    # ── Persist to DB (same as /v1/predict) ─────────────────────────────────
    user_id: Optional[str] = None
    try:
        current_user = await get_current_user(request)
        user_id = current_user.user_id
    except Exception:
        pass

    db_id = save_prediction(
        request_id=request_id,
        user_id=user_id,
        image_data=data,
        model_version=state.model_version,
        primary_label=primary.label,
        confidence=primary.calibrated_score,
        decision=primary.decision,
        full_result=response_data.model_dump(),
    )
    response_data.db_id = db_id
    return response_data
