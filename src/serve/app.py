"""FastAPI application for the MedicalXAI v2 inference server.

Environment variables (all optional – fall back to defaults):
    MEDXAI_CHECKPOINT     path to checkpoint .pt or directory (picks best)
    MEDXAI_LABEL_MAP      path to label_map.json
    MEDXAI_CALIBRATION    path to calibration.json
    MEDXAI_THRESHOLDS     path to thresholds.json
    MEDXAI_ARCH           model architecture  (default: efficientnet_b3)
    MEDXAI_IMAGE_SIZE     square image size   (default: 320)
    MEDXAI_MODEL_VERSION  version tag         (default: v2-efficientnet-b3-320)
    MEDXAI_ADMIN_SECRET        admin header secret (default: empty = auth disabled)
    MEDXAI_AUDIT_PATH          audit JSONL path    (default: artifacts/audit.jsonl)
    MEDXAI_OOD_ENABLED         enable CLIP OOD check (default: false)
    MEDXAI_OOD_THRESHOLD       CLIP CXR score threshold (default: 0.45)
    MEDXAI_PATHOLOGY_ENABLED   enable torchxrayvision 18-class detector (default: false)

Local run::
    uvicorn src.serve.app:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

# ── Load .env FIRST — before any module-level os.getenv() calls elsewhere ──
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)   # existing shell exports take priority
except ImportError:
    pass

import asyncio
import os as _os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.common.logging import get_logger, setup_logging
from src.serve.dependencies import AppState
from src.serve.middleware.error_handler import register_exception_handlers
from src.serve.middleware.logging import StructuredLoggingMiddleware
from src.serve.middleware.request_id import RequestIDMiddleware
from src.serve.routers import admin, auth, chat, explain, health, pathologies, predict, records
from src.serve.auth.user_store import ensure_auth_schema
from src.serve.services.prediction_store import ensure_prediction_schema
from src.serve.services.artifact_loader import EnvConfig, load_all_artifacts
from src.serve.services.database import close_db, init_db
from src.serve.services import ood_detector as _ood
from src.serve.services import pathology_detector as _patho

setup_logging()   # initialise before any logger is used
logger = get_logger("serve.app")


# ---------------------------------------------------------------------------
# Lifespan – load all artifacts once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = EnvConfig.from_env()

    if cfg.device == "auto":
        device = torch.device(
            "mps"  if torch.backends.mps.is_available()  else
            "cuda" if torch.cuda.is_available()          else
            "cpu"
        )
    else:
        device = torch.device(cfg.device)
    logger.info("Serving on device=%s  version=%s", device, cfg.model_version)

    # Postgres (optional – degrades to JSONL if DATABASE_URL not set)
    init_db()
    ensure_auth_schema()
    ensure_prediction_schema()

    # OOD detector (optional)
    import os
    if os.environ.get("MEDXAI_OOD_ENABLED", "").lower() in ("1", "true", "yes"):
        _ood.init_ood(
            accept_threshold=cfg.ood_accept_threshold,
            reject_threshold=cfg.ood_reject_threshold,
        )
    else:
        logger.info("OOD detector disabled (set MEDXAI_OOD_ENABLED=true to enable)")

    # 18-class pathology detector – required when txrv is primary, optional otherwise
    txrv_primary = os.environ.get("MEDXAI_PRIMARY_MODEL", "efficientnet").lower() == "txrv"
    patho_requested = os.environ.get("MEDXAI_PATHOLOGY_ENABLED", "").lower() in ("1", "true", "yes")
    if txrv_primary or patho_requested:
        _patho.init_pathology_detector()
        if txrv_primary:
            logger.info("TXRv primary mode: torchxrayvision DenseNet-121 is the primary classifier")
    else:
        logger.info("Pathology detector disabled (set MEDXAI_PATHOLOGY_ENABLED=true to enable)")

    arts = load_all_artifacts(cfg, device)

    app.state.app = AppState(
        model=arts.model,
        label_map=arts.label_map,
        temperature=arts.temperature,
        thresholds=arts.thresholds,
        device=device,
        arch=cfg.arch,
        model_version=cfg.model_version,
        image_size=cfg.image_size,
        gradcam=arts.gradcam,
        checkpoint_ok=arts.checkpoint_ok,
        calibration_ok=arts.calibration_ok,
        thresholds_ok=arts.thresholds_ok,
        label_map_ok=arts.label_map_ok,
    )
    app.state.env_cfg = cfg
    app.state.reload_lock = asyncio.Lock()

    if arts.checkpoint_ok and arts.label_map_ok:
        logger.info(
            "Server ready: classes=%d  T=%.4f  thresholds=%d",
            arts.label_map.get("num_classes", 0),
            arts.temperature,
            len(arts.thresholds),
        )
    else:
        logger.warning("Server started with missing artifacts – /ready will return false")

    yield

    close_db()
    logger.info("Shutting down MedicalXAI serve.")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MedicalXAI CXR Inference API",
    description="Production-ready inference API for the v2 EfficientNet-B3 CXR classifier.",
    version="2.0.0",
    lifespan=lifespan,
)

# Middleware (order matters: RequestID first so all later middleware see the id)
app.add_middleware(StructuredLoggingMiddleware)
app.add_middleware(RequestIDMiddleware)
_CORS_ORIGINS = [
    o.strip()
    for o in _os.getenv(
        "CORS_ORIGINS",
        "http://localhost:4000,http://localhost:3000,http://localhost",
    ).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,           # required for HttpOnly cookie passthrough
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-Id"],
)

# Exception handlers
register_exception_handlers(app)

# Routers
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(explain.router)
app.include_router(pathologies.router)
app.include_router(records.router)
app.include_router(admin.router)
app.include_router(chat.router)
app.include_router(auth.router)
