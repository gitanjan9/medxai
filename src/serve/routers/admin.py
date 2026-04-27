"""Admin endpoints for hot-reloading artifacts without restarting the server.

All endpoints require the ``X-Admin-Secret`` header when
``MEDXAI_ADMIN_SECRET`` env var is set.  If the env var is empty,
admin auth is disabled (suitable for local development only).

Endpoints
---------
POST /v1/admin/reload-thresholds   – reload thresholds.json
POST /v1/admin/reload-artifacts    – reload calibration + thresholds (no model reload)
GET  /v1/admin/audit               – return recent audit log entries
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from typing import Optional

from src.common.logging import get_logger
from src.serve.dependencies import AppState, EnvConfig, get_app_state, get_env_config, get_request_id
from src.serve.schemas.response import AdminReloadResponse
from src.serve.services.artifact_loader import load_calibration_only, load_thresholds_only
from src.serve.services.audit import read_recent_audit

logger = get_logger("serve.admin")

router = APIRouter(prefix="/v1/admin", tags=["admin"])


def _check_admin_secret(cfg: EnvConfig, secret: Optional[str]) -> None:
    """Raise 401 if admin auth is enabled and the header doesn't match."""
    if cfg.admin_secret and secret != cfg.admin_secret:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Secret")


@router.post("/reload-thresholds", response_model=AdminReloadResponse)
async def reload_thresholds(
    request: Request,
    x_admin_secret: Optional[str] = Header(default=None),
    state: AppState = Depends(get_app_state),
    cfg: EnvConfig = Depends(get_env_config),
    request_id: str = Depends(get_request_id),
) -> AdminReloadResponse:
    """Hot-reload ``thresholds.json`` without restarting the server.

    Acquires the reload lock to prevent concurrent reloads.
    """
    _check_admin_secret(cfg, x_admin_secret)

    logger.info("reload-thresholds requested request_id=%s", request_id)

    async with request.app.state.reload_lock:
        new_thresholds, ok = load_thresholds_only(cfg)
        state.thresholds = new_thresholds
        state.thresholds_ok = ok

    logger.info(
        "reload-thresholds complete request_id=%s  classes=%d  ok=%s",
        request_id, len(new_thresholds), ok,
    )
    return AdminReloadResponse(
        reloaded=["thresholds"],
        details={"thresholds_count": len(new_thresholds), "ok": ok},
        request_id=request_id,
    )


@router.post("/reload-artifacts", response_model=AdminReloadResponse)
async def reload_artifacts(
    request: Request,
    x_admin_secret: Optional[str] = Header(default=None),
    state: AppState = Depends(get_app_state),
    cfg: EnvConfig = Depends(get_env_config),
    request_id: str = Depends(get_request_id),
) -> AdminReloadResponse:
    """Hot-reload calibration + thresholds (soft artifacts).

    The model checkpoint is **not** reloaded here – use a rolling
    deployment for model version changes.
    """
    _check_admin_secret(cfg, x_admin_secret)

    logger.info("reload-artifacts requested request_id=%s", request_id)

    async with request.app.state.reload_lock:
        new_thresholds, thresh_ok = load_thresholds_only(cfg)
        new_T, calib_ok = load_calibration_only(cfg)
        state.thresholds = new_thresholds
        state.thresholds_ok = thresh_ok
        state.temperature = new_T
        state.calibration_ok = calib_ok

    logger.info(
        "reload-artifacts complete request_id=%s  T=%.4f  thresh_classes=%d",
        request_id, new_T, len(new_thresholds),
    )
    return AdminReloadResponse(
        reloaded=["calibration", "thresholds"],
        details={
            "temperature": round(new_T, 4),
            "thresholds_count": len(new_thresholds),
            "calibration_ok": calib_ok,
            "thresholds_ok": thresh_ok,
        },
        request_id=request_id,
    )


@router.get("/audit")
async def get_audit(
    request: Request,
    x_admin_secret: Optional[str] = Header(default=None),
    n: int = 100,
    cfg: EnvConfig = Depends(get_env_config),
) -> dict:
    """Return the last *n* audit log entries."""
    _check_admin_secret(cfg, x_admin_secret)
    records = read_recent_audit(cfg.audit_path, n=n)
    return {"count": len(records), "records": records}
