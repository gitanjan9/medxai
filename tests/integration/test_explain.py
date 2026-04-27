"""Integration tests for POST /v1/explain."""
from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient


def test_explain_503_when_model_not_ready(client: TestClient, tiny_jpeg: bytes) -> None:
    from src.serve.app import app
    original = app.state.app.checkpoint_ok
    app.state.app.checkpoint_ok = False
    try:
        resp = client.post("/v1/explain", files={"file": ("t.jpg", tiny_jpeg, "image/jpeg")})
        assert resp.status_code == 503
    finally:
        app.state.app.checkpoint_ok = original


def test_explain_501_when_gradcam_none(client: TestClient, tiny_jpeg: bytes) -> None:
    """GradCAM is None in the stub state – should return 501."""
    resp = client.post("/v1/explain", files={"file": ("t.jpg", tiny_jpeg, "image/jpeg")})
    assert resp.status_code == 501


def test_explain_422_on_corrupt_file(client: TestClient) -> None:
    from src.serve.app import app
    from unittest.mock import MagicMock
    original_gradcam = app.state.app.gradcam
    app.state.app.gradcam = MagicMock()
    try:
        resp = client.post(
            "/v1/explain",
            files={"file": ("bad.jpg", b"not-an-image", "image/jpeg")},
        )
        assert resp.status_code == 422
    finally:
        app.state.app.gradcam = original_gradcam


def test_explain_response_schema_with_gradcam(client: TestClient, tiny_jpeg: bytes) -> None:
    """With a stub GradCAM generator that returns None, gradcam_available=False."""
    from src.serve.app import app
    from unittest.mock import MagicMock, patch

    mock_gradcam = MagicMock()
    original = app.state.app.gradcam
    app.state.app.gradcam = mock_gradcam

    with patch("src.serve.routers.explain.run_gradcam_base64", return_value=None):
        resp = client.post(
            "/v1/explain",
            files={"file": ("t.jpg", tiny_jpeg, "image/jpeg")},
        )
    app.state.app.gradcam = original

    assert resp.status_code == 200
    body = resp.json()
    assert "primary_prediction" in body
    assert "localization" in body
    assert "ood" in body
    assert "warnings" in body
    assert "request_id" in body
    assert body["gradcam_available"] is False
    assert body["primary_prediction"]["decision"] in {"positive", "review", "negative"}
    assert body["localization"]["type"] == "not_generated"


def test_explain_localization_marked_approximate(client: TestClient, tiny_jpeg: bytes) -> None:
    """When GradCAM returns data, localization type must be approximate_attention_region."""
    import base64
    from PIL import Image
    from src.serve.app import app
    from unittest.mock import MagicMock, patch

    app.state.app.gradcam = MagicMock()
    fake_heatmap = Image.fromarray(np.full((32, 32), 200, dtype=np.uint8), mode="L")
    buf = io.BytesIO()
    fake_heatmap.save(buf, format="PNG")
    fake_b64 = base64.b64encode(buf.getvalue()).decode()

    with patch("src.serve.routers.explain.run_gradcam_base64", return_value=fake_b64):
        resp = client.post("/v1/explain", files={"file": ("t.jpg", tiny_jpeg, "image/jpeg")})

    app.state.app.gradcam = None
    body = resp.json()
    loc = body["localization"]
    assert loc["enabled"] is True
    assert loc["type"] == "approximate_attention_region"
    assert "disclaimer" in loc
    assert loc["disclaimer"] != ""


def test_explain_request_id_echoed(client: TestClient, tiny_jpeg: bytes) -> None:
    from src.serve.app import app
    from unittest.mock import MagicMock, patch

    app.state.app.gradcam = MagicMock()
    with patch("src.serve.routers.explain.run_gradcam_base64", return_value=None):
        resp = client.post(
            "/v1/explain",
            files={"file": ("t.jpg", tiny_jpeg, "image/jpeg")},
            headers={"X-Request-ID": "expl-req-42"},
        )
    assert resp.json()["request_id"] == "expl-req-42"
    assert resp.headers.get("x-request-id") == "expl-req-42"


def test_explain_413_on_oversized_file(client: TestClient) -> None:
    from src.serve.app import app
    from unittest.mock import MagicMock
    app.state.app.gradcam = MagicMock()
    big = b"x" * (21 * 1024 * 1024)
    resp = client.post("/v1/explain", files={"file": ("big.jpg", big, "image/jpeg")})
    assert resp.status_code == 413
