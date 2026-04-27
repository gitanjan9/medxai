"""Health and readiness endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Request

from src.serve.dependencies import get_app_state
from src.serve.schemas.response import HealthResponse, ReadyResponse

router = APIRouter(tags=["ops"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe – always 200 if the process is running."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse)
def ready(request: Request) -> ReadyResponse:
    """Readiness probe – 200 only when all artifacts loaded successfully."""
    state = get_app_state(request)
    details = {
        "checkpoint": state.checkpoint_ok,
        "calibration": state.calibration_ok,
        "thresholds": state.thresholds_ok,
        "label_map": state.label_map_ok,
    }
    is_ready = all(details.values())
    return ReadyResponse(
        ready=is_ready,
        model_version=state.model_version,
        details=details,
    )
