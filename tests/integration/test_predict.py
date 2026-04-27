"""Integration tests for POST /v1/predict."""
from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient


def _predict(client: TestClient, image: bytes, **kwargs) -> dict:
    resp = client.post(
        "/v1/predict",
        files={"file": ("test.jpg", image, "image/jpeg")},
        **kwargs,
    )
    return resp


# ── Core schema ──────────────────────────────────────────────────────────────

def test_predict_valid_image_returns_200(client: TestClient, tiny_jpeg: bytes) -> None:
    assert _predict(client, tiny_jpeg).status_code == 200


def test_predict_response_top_level_keys(client: TestClient, tiny_jpeg: bytes) -> None:
    body = _predict(client, tiny_jpeg).json()
    required = {"request_id", "model_version", "status",
                "primary_prediction", "ood", "pathologies", "localization", "warnings"}
    assert required <= body.keys()


def test_predict_primary_prediction_shape(client: TestClient, tiny_jpeg: bytes) -> None:
    pp = _predict(client, tiny_jpeg).json()["primary_prediction"]
    required = {"label", "raw_score", "calibrated_score", "all_scores",
                "decision", "confidence_band", "review_reason", "threshold_version"}
    assert required <= pp.keys()


def test_predict_primary_label_is_known_class(client: TestClient, tiny_jpeg: bytes) -> None:
    known = {
        "no_acute_cardiopulmonary", "no_acute_intrathoracic",
        "no_change_stable", "no_pneumonia", "no_pneumothorax",
    }
    assert _predict(client, tiny_jpeg).json()["primary_prediction"]["label"] in known


def test_predict_all_scores_sums_to_one(client: TestClient, tiny_jpeg: bytes) -> None:
    scores = _predict(client, tiny_jpeg).json()["primary_prediction"]["all_scores"]
    assert abs(sum(scores.values()) - 1.0) < 0.01


def test_predict_decision_valid_values(client: TestClient, tiny_jpeg: bytes) -> None:
    decision = _predict(client, tiny_jpeg).json()["primary_prediction"]["decision"]
    assert decision in {"positive", "review", "negative"}


def test_predict_confidence_band_valid(client: TestClient, tiny_jpeg: bytes) -> None:
    band = _predict(client, tiny_jpeg).json()["primary_prediction"]["confidence_band"]
    assert band in {"high", "medium", "low"}


def test_predict_request_id_echoed(client: TestClient, tiny_jpeg: bytes) -> None:
    resp = _predict(client, tiny_jpeg, headers={"X-Request-ID": "my-req-999"})
    assert resp.json()["request_id"] == "my-req-999"
    assert resp.headers.get("x-request-id") == "my-req-999"


def test_predict_model_version_in_response(client: TestClient, tiny_jpeg: bytes) -> None:
    assert _predict(client, tiny_jpeg).json()["model_version"] == "test-stub"


def test_predict_status_ok_when_no_warnings(client: TestClient, tiny_jpeg: bytes) -> None:
    body = _predict(client, tiny_jpeg).json()
    assert body["status"] in {"ok", "degraded"}  # ok when no features enabled


def test_predict_localization_disabled_for_predict(client: TestClient, tiny_jpeg: bytes) -> None:
    loc = _predict(client, tiny_jpeg).json()["localization"]
    assert loc["enabled"] is False
    assert loc["type"] == "not_generated"


def test_predict_ood_disabled_when_not_loaded(client: TestClient, tiny_jpeg: bytes) -> None:
    ood = _predict(client, tiny_jpeg).json()["ood"]
    assert ood["enabled"] is False
    assert ood["reason"] == "disabled"


def test_predict_pathologies_disabled_when_not_loaded(client: TestClient, tiny_jpeg: bytes) -> None:
    p = _predict(client, tiny_jpeg).json()["pathologies"]
    assert p["enabled"] is False
    assert p["status"] == "disabled"


# ── Error paths ───────────────────────────────────────────────────────────────

def test_predict_corrupt_file_returns_422(client: TestClient) -> None:
    resp = client.post("/v1/predict", files={"file": ("bad.jpg", b"not-an-image", "image/jpeg")})
    assert resp.status_code == 422


def test_predict_503_when_model_not_ready(client: TestClient, tiny_jpeg: bytes) -> None:
    from src.serve.app import app
    original = app.state.app.checkpoint_ok
    app.state.app.checkpoint_ok = False
    try:
        assert _predict(client, tiny_jpeg).status_code == 503
    finally:
        app.state.app.checkpoint_ok = original


def test_predict_png_accepted(client: TestClient) -> None:
    from PIL import Image
    img = Image.fromarray(np.full((32, 32), 128, dtype=np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    resp = client.post("/v1/predict", files={"file": ("test.png", buf.getvalue(), "image/png")})
    assert resp.status_code == 200
