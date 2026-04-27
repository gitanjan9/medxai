"""Integration tests for GET /health and GET /ready."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_has_no_x_request_id_by_default(client: TestClient) -> None:
    resp = client.get("/health")
    assert "x-request-id" in resp.headers


def test_health_echoes_caller_request_id(client: TestClient) -> None:
    resp = client.get("/health", headers={"X-Request-ID": "test-123"})
    assert resp.headers.get("x-request-id") == "test-123"


def test_ready_returns_true_with_full_state(client: TestClient) -> None:
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is True
    assert body["model_version"] == "test-stub"
    assert body["details"]["checkpoint"] is True
    assert body["details"]["calibration"] is True
    assert body["details"]["thresholds"] is True
    assert body["details"]["label_map"] is True


def test_ready_returns_false_when_checkpoint_missing(client: TestClient) -> None:
    from src.serve.app import app
    original = app.state.app.checkpoint_ok
    app.state.app.checkpoint_ok = False
    try:
        resp = client.get("/ready")
        body = resp.json()
        assert body["ready"] is False
        assert body["details"]["checkpoint"] is False
    finally:
        app.state.app.checkpoint_ok = original
