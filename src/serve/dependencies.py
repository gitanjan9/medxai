"""FastAPI dependency injection for shared application state.

The AppState dataclass is populated once during the lifespan and stored
on ``app.state.app``.  Route handlers retrieve it via ``get_app_state``.
``EnvConfig`` is stored on ``app.state.env_cfg`` and injected via ``get_env_config``.
The ``asyncio.Lock`` at ``app.state.reload_lock`` prevents concurrent artifact reloads.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from fastapi import Request

from src.serve.services.explainability import GradCAMGenerator


@dataclass
class AppState:
    model: nn.Module
    label_map: dict
    temperature: float
    thresholds: list[dict]
    device: torch.device
    arch: str
    model_version: str
    image_size: tuple[int, int]
    gradcam: Optional[GradCAMGenerator] = field(default=None)

    # Readiness flags (set at load time)
    checkpoint_ok: bool = False
    calibration_ok: bool = False
    thresholds_ok: bool = False
    label_map_ok: bool = False


def get_app_state(request: Request) -> AppState:
    """FastAPI dependency – injects the shared AppState."""
    return request.app.state.app


def get_env_config(request: Request) -> "EnvConfig":
    """FastAPI dependency – injects the immutable EnvConfig."""
    return request.app.state.env_cfg


def get_request_id(request: Request) -> str:
    """FastAPI dependency – returns the request_id stamped by middleware."""
    return getattr(request.state, "request_id", "unknown")


# Re-export so routers only need to import from dependencies
from src.serve.services.artifact_loader import EnvConfig  # noqa: E402
