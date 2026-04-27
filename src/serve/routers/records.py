"""Patient records and clinician feedback router — /v1/records/*"""
from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Cookie, Depends, HTTPException, status
from pydantic import BaseModel

from src.common.logging import get_logger
from src.serve.auth.dependencies import AdminRequired, AuthRequired, CurrentUser, get_current_user
from src.serve.services.prediction_store import (
    get_prediction,
    list_predictions,
    submit_feedback,
    update_patient_info,
)
from src.serve.services.retraining_service import (
    force_retrain,
    get_retraining_status,
    maybe_trigger_retraining,
)

router = APIRouter(prefix="/v1/records", tags=["records"])
logger = get_logger("serve.records")


# ── Schemas ───────────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    feedback: str           # "correct" | "wrong"
    true_label: Optional[str] = None

class PatientInfoRequest(BaseModel):
    patient_name: str = ""
    patient_id: str = ""
    patient_age: Optional[int] = None
    patient_gender: str = ""
    notes: str = ""

class PredictionOut(BaseModel):
    id: str
    request_id: str
    patient_name: str
    patient_id: str
    patient_age: Optional[int]
    patient_gender: str
    notes: str
    model_version: str
    primary_label: str
    confidence: float
    decision: str
    feedback: str
    true_label: Optional[str]
    created_at: str
    full_result: Optional[dict] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_out(r) -> PredictionOut:
    return PredictionOut(
        id=r.id,
        request_id=r.request_id,
        patient_name=r.patient_name,
        patient_id=r.patient_id,
        patient_age=r.patient_age,
        patient_gender=r.patient_gender,
        notes=r.notes,
        model_version=r.model_version,
        primary_label=r.primary_label,
        confidence=r.confidence,
        decision=r.decision,
        feedback=r.feedback,
        true_label=r.true_label,
        created_at=r.created_at.isoformat(),
        full_result=r.full_result if isinstance(r.full_result, dict) else None,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("", response_model=list[PredictionOut])
async def list_records(
    user: AuthRequired,
    limit: int = 50,
    offset: int = 0,
    all_users: bool = False,
) -> list[PredictionOut]:
    """List prediction records. Admins can pass all_users=true to see all."""
    uid = None if (all_users and user.is_admin) else user.user_id
    records = list_predictions(user_id=uid, limit=limit, offset=offset)
    return [_to_out(r) for r in records]


@router.get("/{prediction_id}", response_model=PredictionOut)
async def get_record(
    prediction_id: str,
    user: AuthRequired,
) -> PredictionOut:
    record = get_prediction(prediction_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    if not user.is_admin and record.user_id != user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return _to_out(record)


@router.patch("/{prediction_id}/patient", response_model=dict)
async def patch_patient_info(
    prediction_id: str,
    body: PatientInfoRequest,
    user: AuthRequired,
) -> dict:
    """Annotate patient metadata on an existing prediction."""
    record = get_prediction(prediction_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    if not user.is_admin and record.user_id != user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    ok = update_patient_info(
        prediction_id,
        patient_name=body.patient_name,
        patient_id=body.patient_id,
        patient_age=body.patient_age,
        patient_gender=body.patient_gender,
        notes=body.notes,
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Update failed")
    return {"message": "Patient info updated"}


@router.post("/{prediction_id}/feedback", response_model=dict)
async def post_feedback(
    prediction_id: str,
    body: FeedbackRequest,
    user: AuthRequired,
) -> dict:
    """Submit clinician feedback. Wrong predictions are queued for retraining."""
    if body.feedback not in ("correct", "wrong"):
        raise HTTPException(status_code=422, detail="feedback must be 'correct' or 'wrong'")
    if body.feedback == "wrong" and not body.true_label:
        raise HTTPException(status_code=422, detail="true_label is required when feedback is 'wrong'")

    record = get_prediction(prediction_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    if not user.is_admin and record.user_id != user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    ok = submit_feedback(prediction_id, body.feedback, body.true_label)
    if not ok:
        raise HTTPException(status_code=500, detail="Feedback submission failed")

    retraining_started = False
    if body.feedback == "wrong":
        logger.info(
            "Wrong prediction: id=%s predicted=%s true=%s user=%s",
            prediction_id, record.primary_label, body.true_label, user.user_id,
        )
        retraining_started = await maybe_trigger_retraining()

    return {
        "message": "Feedback saved",
        "retraining_triggered": retraining_started,
    }


# ── Admin retraining control ──────────────────────────────────────────────────

@router.post("/admin/retrain", response_model=dict)
async def trigger_retrain(admin: AdminRequired) -> dict:
    """Force retraining regardless of queue size (admin only)."""
    started = await force_retrain()
    return {"message": "Retraining started" if started else "Already running"}


@router.get("/admin/retrain/status", response_model=dict)
async def retrain_status(admin: AdminRequired) -> dict:
    return get_retraining_status()
