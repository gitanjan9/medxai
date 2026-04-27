"""Integration test fixtures.

Creates a minimal AppState with a tiny 2-layer model so tests run without
the 130 MB v2 checkpoint.  The fixture patches app.state.app after the
lifespan completes (inside the TestClient context manager).
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import numpy as np
import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from src.serve.dependencies import AppState
from src.serve.services.artifact_loader import EnvConfig


@pytest.fixture(scope="session", autouse=True)
def _block_db_in_tests():
    """Remove DATABASE_URL for the entire test session.

    Prevents integration tests from writing audit rows into the real Postgres
    database.  All audit writes fall back to the per-test JSONL in tmp_path.
    """
    original = os.environ.pop("DATABASE_URL", None)
    # Also reset the module-level pool in case it was already initialised
    from src.serve.services import database as _db
    _db._pool = None
    yield
    if original is not None:
        os.environ["DATABASE_URL"] = original


# ---------------------------------------------------------------------------
# Minimal stub model (5-class, accepts 1-ch 320×320)
# ---------------------------------------------------------------------------

class _StubModel(nn.Module):
    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(1, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(x).flatten(1))


_LABEL_MAP = {
    "idx_to_str": {
        0: "no_acute_cardiopulmonary",
        1: "no_acute_intrathoracic",
        2: "no_change_stable",
        3: "no_pneumonia",
        4: "no_pneumothorax",
    },
    "str_to_idx": {
        "no_acute_cardiopulmonary": 0,
        "no_acute_intrathoracic":   1,
        "no_change_stable":         2,
        "no_pneumonia":             3,
        "no_pneumothorax":          4,
    },
    "num_classes": 5,
}

_THRESHOLDS = [
    {"class_idx": i, "class_name": list(_LABEL_MAP["str_to_idx"])[i],
     "low": 0.2, "high": 0.7, "ppv_at_high": 1.0, "recall_at_low": 0.7}
    for i in range(5)
]


def _make_state() -> AppState:
    model = _StubModel(5).eval()
    return AppState(
        model=model,
        label_map=_LABEL_MAP,
        temperature=1.0,
        thresholds=_THRESHOLDS,
        device=torch.device("cpu"),
        arch="efficientnet_b3",
        model_version="test-stub",
        image_size=(320, 320),
        gradcam=None,
        checkpoint_ok=True,
        calibration_ok=True,
        thresholds_ok=True,
        label_map_ok=True,
    )


def _make_env_cfg(tmp_path) -> EnvConfig:
    import tempfile
    from pathlib import Path
    return EnvConfig(
        checkpoint_path=Path("artifacts/v2/checkpoints"),
        label_map_path=Path("artifacts/v2/label_map.json"),
        calibration_path=Path("artifacts/calibration.json"),
        thresholds_path=Path("artifacts/thresholds.json"),
        arch="efficientnet_b3",
        image_size=(320, 320),
        model_version="test-stub",
        admin_secret="",
        audit_path=Path(tmp_path) / "audit.jsonl",
        device="cpu",
        environment="test",
        ood_accept_threshold=0.45,
        ood_reject_threshold=0.20,
        localization_enabled=True,
    )


@pytest.fixture
def client(tmp_path) -> TestClient:
    """TestClient with stub model state (no disk I/O)."""
    from src.serve.app import app

    with TestClient(app, raise_server_exceptions=True) as c:
        # Override AFTER lifespan completes so real artifacts don't clobber stub
        app.state.app = _make_state()
        app.state.env_cfg = _make_env_cfg(tmp_path)
        if not hasattr(app.state, "reload_lock"):
            app.state.reload_lock = asyncio.Lock()
        yield c


@pytest.fixture
def tiny_jpeg() -> bytes:
    """Return a minimal valid JPEG (32×32 white image)."""
    from PIL import Image
    img = Image.fromarray(np.full((32, 32), 200, dtype=np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()
