"""Integration tests for POST /v1/admin/reload-thresholds and /reload-artifacts."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# reload-thresholds
# ---------------------------------------------------------------------------

def test_reload_thresholds_returns_200(client: TestClient) -> None:
    resp = client.post("/v1/admin/reload-thresholds")
    assert resp.status_code == 200


def test_reload_thresholds_response_schema(client: TestClient) -> None:
    body = client.post("/v1/admin/reload-thresholds").json()
    assert "reloaded" in body
    assert "details" in body
    assert "request_id" in body
    assert "thresholds" in body["reloaded"]


def test_reload_thresholds_updates_state(client: TestClient) -> None:
    from src.serve.app import app
    original_count = len(app.state.app.thresholds)
    resp = client.post("/v1/admin/reload-thresholds")
    assert resp.status_code == 200
    # After reload from real file, count should still be 5
    assert len(app.state.app.thresholds) == 5


def test_reload_thresholds_echoes_request_id(client: TestClient) -> None:
    resp = client.post(
        "/v1/admin/reload-thresholds",
        headers={"X-Request-ID": "admin-reload-1"},
    )
    assert resp.json()["request_id"] == "admin-reload-1"


def test_reload_thresholds_requires_secret_when_set(client: TestClient) -> None:
    from src.serve.app import app
    original = app.state.env_cfg.admin_secret
    app.state.env_cfg.admin_secret = "supersecret"
    try:
        resp = client.post("/v1/admin/reload-thresholds")
        assert resp.status_code == 401
    finally:
        app.state.env_cfg.admin_secret = original


def test_reload_thresholds_passes_with_correct_secret(client: TestClient) -> None:
    from src.serve.app import app
    app.state.env_cfg.admin_secret = "supersecret"
    try:
        resp = client.post(
            "/v1/admin/reload-thresholds",
            headers={"X-Admin-Secret": "supersecret"},
        )
        assert resp.status_code == 200
    finally:
        app.state.env_cfg.admin_secret = ""


# ---------------------------------------------------------------------------
# reload-artifacts
# ---------------------------------------------------------------------------

def test_reload_artifacts_returns_200(client: TestClient) -> None:
    resp = client.post("/v1/admin/reload-artifacts")
    assert resp.status_code == 200


def test_reload_artifacts_response_schema(client: TestClient) -> None:
    body = client.post("/v1/admin/reload-artifacts").json()
    assert "reloaded" in body
    assert "calibration" in body["reloaded"]
    assert "thresholds" in body["reloaded"]
    assert "details" in body
    assert "temperature" in body["details"]


def test_reload_artifacts_updates_temperature(client: TestClient) -> None:
    from src.serve.app import app
    resp = client.post("/v1/admin/reload-artifacts")
    assert resp.status_code == 200
    # Temperature from real calibration.json (T≈1.352)
    assert 0.5 < app.state.app.temperature < 5.0


# ---------------------------------------------------------------------------
# audit endpoint
# ---------------------------------------------------------------------------

def test_audit_returns_200(client: TestClient) -> None:
    resp = client.get("/v1/admin/audit")
    assert resp.status_code == 200


def test_audit_response_schema(client: TestClient) -> None:
    body = client.get("/v1/admin/audit").json()
    assert "count" in body
    assert "records" in body
    assert isinstance(body["records"], list)
