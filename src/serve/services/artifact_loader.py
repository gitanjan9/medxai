"""Centralised artifact loading for the serve layer.

``EnvConfig`` reads all environment variables once.
``load_all_artifacts`` is called at startup (lifespan) and on admin reload.
Both paths share identical logic so reload is always consistent with startup.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from src.common.logging import get_logger
from src.serve.services.calibration import load_temperature
from src.serve.services.explainability import build_gradcam
from src.serve.services.model_loader import load_label_map, load_model, resolve_best_checkpoint
from src.serve.services.thresholds import load_thresholds

logger = get_logger("serve.artifact_loader")

_DEFAULTS: dict[str, str] = {
    "MEDXAI_CHECKPOINT":    "artifacts/v2/checkpoints",
    "MEDXAI_LABEL_MAP":     "artifacts/v2/label_map.json",
    "MEDXAI_CALIBRATION":   "artifacts/calibration.json",
    "MEDXAI_THRESHOLDS":    "artifacts/thresholds.json",
    "MEDXAI_ARCH":          "efficientnet_b3",
    "MEDXAI_IMAGE_SIZE":    "320",
    "MEDXAI_MODEL_VERSION": "v2-efficientnet-b3-320",
    "MEDXAI_ADMIN_SECRET":  "",          # empty → admin auth disabled
    "MEDXAI_AUDIT_PATH":    "artifacts/audit.jsonl",
    "MEDXAI_DEVICE":              "auto",  # auto | cpu | cuda | mps
    "MEDXAI_ENV":                 "dev",   # dev | prod | test
    "MEDXAI_OOD_ACCEPT_THRESHOLD": "0.45",
    "MEDXAI_OOD_REJECT_THRESHOLD": "0.20",
    "MEDXAI_LOCALIZATION_ENABLED": "true",  # false → explain skips bbox extraction
    "MEDXAI_PRIMARY_MODEL":        "efficientnet",  # efficientnet | txrv
}


def _env(key: str) -> str:
    return os.environ.get(key, _DEFAULTS[key])


@dataclass
class EnvConfig:
    """Snapshot of env-driven config, read once at startup."""
    checkpoint_path: Path
    label_map_path: Path
    calibration_path: Path
    thresholds_path: Path
    arch: str
    image_size: tuple[int, int]
    model_version: str
    admin_secret: str
    audit_path: Path
    device: str               # "auto" | "cpu" | "cuda" | "mps"
    environment: str          # "dev" | "prod" | "test"
    ood_accept_threshold: float
    ood_reject_threshold: float
    localization_enabled: bool
    primary_model: str        # "efficientnet" | "txrv"

    @classmethod
    def from_env(cls) -> "EnvConfig":
        sz = int(_env("MEDXAI_IMAGE_SIZE"))
        return cls(
            checkpoint_path=Path(_env("MEDXAI_CHECKPOINT")),
            label_map_path=Path(_env("MEDXAI_LABEL_MAP")),
            calibration_path=Path(_env("MEDXAI_CALIBRATION")),
            thresholds_path=Path(_env("MEDXAI_THRESHOLDS")),
            arch=_env("MEDXAI_ARCH"),
            image_size=(sz, sz),
            model_version=_env("MEDXAI_MODEL_VERSION"),
            admin_secret=_env("MEDXAI_ADMIN_SECRET"),
            audit_path=Path(_env("MEDXAI_AUDIT_PATH")),
            device=_env("MEDXAI_DEVICE"),
            environment=_env("MEDXAI_ENV"),
            ood_accept_threshold=float(_env("MEDXAI_OOD_ACCEPT_THRESHOLD")),
            ood_reject_threshold=float(_env("MEDXAI_OOD_REJECT_THRESHOLD")),
            localization_enabled=_env("MEDXAI_LOCALIZATION_ENABLED").lower() not in ("0", "false", "no"),
            primary_model=_env("MEDXAI_PRIMARY_MODEL").lower(),
        )


@dataclass
class LoadedArtifacts:
    """All runtime artifacts returned by load_all_artifacts."""
    model: Optional[nn.Module]
    label_map: dict
    temperature: float
    thresholds: list[dict]
    gradcam: object  # GradCAMGenerator | None
    # Readiness flags
    checkpoint_ok: bool
    calibration_ok: bool
    thresholds_ok: bool
    label_map_ok: bool


def load_all_artifacts(cfg: EnvConfig, device: torch.device) -> LoadedArtifacts:
    """Load model + all artifacts from disk.  Non-critical failures are logged,
    not raised — callers check readiness flags.
    """
    # --- label map (needed first to get num_classes) ---
    label_map: Optional[dict] = None
    lm_ok = False
    try:
        label_map = load_label_map(cfg.label_map_path)
        lm_ok = True
        logger.info("Label map loaded: %d classes", label_map["num_classes"])
    except Exception as exc:
        logger.error("label map load failed [%s]: %s", cfg.label_map_path, exc)

    # --- model checkpoint ---
    model: Optional[nn.Module] = None
    ckpt_ok = False
    num_classes = label_map["num_classes"] if label_map else 5
    try:
        best = resolve_best_checkpoint(cfg.checkpoint_path)
        model = load_model(cfg.arch, num_classes, best, device)
        ckpt_ok = True
    except Exception as exc:
        logger.error("checkpoint load failed [%s]: %s", cfg.checkpoint_path, exc)

    # In txrv-primary mode the EfficientNet is only used for GradCAM;
    # mark the server as ready even if the checkpoint is missing.
    if not ckpt_ok and getattr(cfg, "primary_model", "efficientnet") == "txrv":
        ckpt_ok = True
        logger.info(
            "TXRv primary mode: EfficientNet checkpoint optional (GradCAM only). "
            "Server marked ready."
        )

    # --- calibration ---
    temperature = 1.0
    calib_ok = False
    try:
        temperature = load_temperature(cfg.calibration_path)
        calib_ok = cfg.calibration_path.exists()
    except Exception as exc:
        logger.error("calibration load failed [%s]: %s", cfg.calibration_path, exc)

    # --- thresholds ---
    thresholds: list[dict] = []
    thresh_ok = False
    try:
        thresholds = load_thresholds(cfg.thresholds_path)
        thresh_ok = cfg.thresholds_path.exists()
    except Exception as exc:
        logger.error("thresholds load failed [%s]: %s", cfg.thresholds_path, exc)

    # --- GradCAM (non-critical) ---
    gradcam = None
    if model is not None:
        try:
            gradcam = build_gradcam(model, cfg.arch, device)
            logger.info("GradCAM ready")
        except Exception as exc:
            logger.warning("GradCAM init failed (explain disabled): %s", exc)

    return LoadedArtifacts(
        model=model,
        label_map=label_map or {},
        temperature=temperature,
        thresholds=thresholds,
        gradcam=gradcam,
        checkpoint_ok=ckpt_ok,
        calibration_ok=calib_ok,
        thresholds_ok=thresh_ok,
        label_map_ok=lm_ok,
    )


def load_thresholds_only(cfg: EnvConfig) -> tuple[list[dict], bool]:
    """Reload only thresholds.json.  Used by the hot-reload admin endpoint."""
    try:
        thresholds = load_thresholds(cfg.thresholds_path)
        ok = cfg.thresholds_path.exists()
        logger.info("Thresholds reloaded: %d classes", len(thresholds))
        return thresholds, ok
    except Exception as exc:
        logger.error("thresholds reload failed: %s", exc)
        return [], False


def load_calibration_only(cfg: EnvConfig) -> tuple[float, bool]:
    """Reload only calibration.json.  Used by the hot-reload admin endpoint."""
    try:
        T = load_temperature(cfg.calibration_path)
        ok = cfg.calibration_path.exists()
        logger.info("Calibration reloaded: T=%.4f", T)
        return T, ok
    except Exception as exc:
        logger.error("calibration reload failed: %s", exc)
        return 1.0, False
