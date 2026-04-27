"""POST /v1/predict – single-image inference endpoint."""
from __future__ import annotations
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from src.common.logging import get_logger
from src.serve.auth.dependencies import get_current_user
from src.serve.dependencies import AppState, EnvConfig, get_app_state, get_env_config, get_request_id
from src.serve.services.prediction_store import image_sha256, save_prediction
from src.serve.schemas.response import InferenceResponse, LocalizationSection, OodSection
from src.serve.services.audit import AuditEntry, log_audit
from src.serve.services.calibration import apply_calibration
from src.serve.services.inference import run_inference
from src.serve.services import ood_detector as _ood, pathology_detector as _patho
from src.serve.services.ood_detector import OodResult
from src.serve.services.preprocessing import decode_image
from src.serve.services.response_builder import (
    build_inference_response,
    build_ood_section,
    build_pathologies_section,
    build_primary_prediction,
)
from src.serve.services.thresholds import apply_thresholds
from src.serve.services.txrv_primary_adapter import txrv_to_primary_prediction

logger = get_logger("serve.predict")

router = APIRouter(prefix="/v1", tags=["inference"])

_ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/tiff",
    "image/bmp", "application/octet-stream",
}
_MAX_FILE_BYTES = 20 * 1024 * 1024


@router.post("/predict", response_model=InferenceResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    state: AppState = Depends(get_app_state),
    cfg: EnvConfig = Depends(get_env_config),
    request_id: str = Depends(get_request_id),
) -> InferenceResponse:
    """Accept a chest X-ray image and return a calibrated, threshold-gated prediction.

    OOD behavior:
      accept  → inference runs normally
      review  → inference runs; warning "ood_in_review_band" added; status="degraded"
      reject  → HTTP 422 returned; no inference
    """
    if not state.checkpoint_ok:
        raise HTTPException(status_code=503, detail="Model not ready")

    if file.content_type and file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type: {file.content_type}",
        )
    data = await file.read()
    if len(data) > _MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 20 MB)")

    logger.info("predict request_id=%s  bytes=%d", request_id, len(data))
    t0 = time.perf_counter()
    warnings: list[str] = []

    # ── OOD check ──────────────────────────────────────────────────────────
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
                        f"Reason: {ood_result.reason}. "
                        f"Please upload a frontal chest radiograph."
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

    txrv_primary = cfg.primary_model == "txrv"
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

    else:
        # ── EfficientNet primary path ───────────────────────────────────────
        try:
            tensor = decode_image(data, image_size=state.image_size)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Cannot decode image: {exc}") from exc
        logits, raw_probs = run_inference(state.model, tensor, state.device)
        calibrated_probs = apply_calibration(logits, state.temperature)
        predicted_idx = int(calibrated_probs.argmax())
        predicted_class = state.label_map["idx_to_str"][predicted_idx]
        decision = apply_thresholds(calibrated_probs, predicted_idx, state.thresholds)
        primary = build_primary_prediction(
            predicted_class=predicted_class,
            predicted_idx=predicted_idx,
            raw_probs=raw_probs,
            calibrated_probs=calibrated_probs,
            decision=decision,
            label_map=state.label_map,
        )
        # Optional auxiliary pathology check
        if _patho.is_available():
            try:
                patho_result = _patho.get_pathology_detector().predict(data)
            except Exception as exc:
                logger.warning("Pathology detection failed request_id=%s: %s", request_id, exc)
                patho_error = True
                warnings.append("pathology_inference_failed")

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info(
        "predict request_id=%s  label=%s  score=%.3f  decision=%s  ood=%s  mode=%s  ms=%.1f",
        request_id, primary.label, primary.calibrated_score,
        primary.decision, ood_result.decision if ood_result else "n/a",
        "txrv" if txrv_primary else "efficientnet", latency_ms,
    )

    # ── Build sections ──────────────────────────────────────────────────────
    ood_section = build_ood_section(ood_result, enabled=ood_enabled)
    patho_section = build_pathologies_section(
        patho_result,
        enabled=txrv_primary or _patho.is_available(),
        error=patho_error,
    )

    # ── Audit ───────────────────────────────────────────────────────────────
    log_audit(
        AuditEntry(
            request_id=request_id,
            endpoint="/v1/predict",
            prediction=primary.label,
            calibrated_score=primary.calibrated_score,
            decision=primary.decision,
            latency_ms=latency_ms,
            model_version=state.model_version,
            environment=cfg.environment,
            confidence_band=primary.confidence_band,
            review_reason=primary.review_reason,
            threshold_version=primary.threshold_version,
            explanation_requested=False,
            ood_score=ood_result.cxr_score if ood_result else None,
            ood_decision=ood_result.decision if ood_result else "",
            ood_reason=ood_result.reason if ood_result else "",
            top_pathology=patho_result.top_finding if patho_result else "",
            top_pathology_score=patho_result.top_score if patho_result else None,
            localization_generated=False,
        ),
        cfg.audit_path,
    )

    # ── Persist prediction to DB (optional auth — anonymous if no token) ─────
    user_id: Optional[str] = None
    try:
        current_user = await get_current_user(request)
        user_id = current_user.user_id
    except Exception:
        pass  # unauthenticated — still save the record

    response_data = build_inference_response(
        request_id=request_id,
        model_version=state.model_version,
        primary=primary,
        ood=ood_section,
        pathologies=patho_section,
        localization=LocalizationSection(enabled=False, type="not_generated"),
        warnings=warnings,
    )

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

    # Cache image to disk so retraining can use it
    _cache_image(data)

    # Attach db_id so frontend can link the result to the DB record for feedback
    response_data.db_id = db_id
    return response_data


def _cache_image(data: bytes) -> None:
    """Save image under artifacts/images/<sha256>.jpg for retraining use."""
    try:
        img_dir = Path("artifacts/images")
        img_dir.mkdir(parents=True, exist_ok=True)
        img_hash = image_sha256(data)
        dest = img_dir / f"{img_hash}.jpg"
        if not dest.exists():
            dest.write_bytes(data)
    except Exception as exc:
        logger.warning("Image cache failed: %s", exc)


