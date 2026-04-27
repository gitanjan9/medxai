"""Integration tests for GET /ready – detailed readiness probe."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_ready_200_when_all_ok(client: TestClient) -> None:
    resp = client.get("/ready")
    assert resp.status_code == 200


def test_ready_body_schema(client: TestClient) -> None:
    body = client.get("/ready").json()
    assert "ready" in body
    assert "model_version" in body
    assert "details" in body


def test_ready_details_keys(client: TestClient) -> None:
    details = client.get("/ready").json()["details"]
    assert "checkpoint" in details
    assert "calibration" in details
    assert "thresholds" in details
    assert "label_map" in details


def test_ready_true_with_full_stub_state(client: TestClient) -> None:
    assert client.get("/ready").json()["ready"] is True


def test_ready_false_when_label_map_missing(client: TestClient) -> None:
    from src.serve.app import app
    original = app.state.app.label_map_ok
    app.state.app.label_map_ok = False
    try:
        body = client.get("/ready").json()
        assert body["ready"] is False
        assert body["details"]["label_map"] is False
    finally:
        app.state.app.label_map_ok = original


def test_ready_false_when_calibration_missing(client: TestClient) -> None:
    from src.serve.app import app
    original = app.state.app.calibration_ok
    app.state.app.calibration_ok = False
    try:
        assert client.get("/ready").json()["ready"] is False
    finally:
        app.state.app.calibration_ok = original


def test_ready_false_when_thresholds_missing(client: TestClient) -> None:
    from src.serve.app import app
    original = app.state.app.thresholds_ok
    app.state.app.thresholds_ok = False
    try:
        assert client.get("/ready").json()["ready"] is False
    finally:
        app.state.app.thresholds_ok = original


def test_ready_model_version_matches_state(client: TestClient) -> None:
    body = client.get("/ready").json()
    assert body["model_version"] == "test-stub"


def test_ready_has_request_id_header(client: TestClient) -> None:
    resp = client.get("/ready")
    assert "x-request-id" in resp.headers
