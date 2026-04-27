"""Production hardening tests.

Covers:
  1. OOD accept path (score >= accept_threshold)
  2. OOD review path (reject_threshold <= score < accept_threshold)
  3. OOD reject path (score < reject_threshold)
  4. Pathologies separate from primary_prediction
  5. Localization marked as approximate_attention_region with disclaimer
  6. Pathology failure does not crash primary prediction
  7. Localization failure does not crash primary prediction
  8. Audit row includes new OOD / pathology / localization fields
  9. Response schema valid when all optional features are disabled
 10. Warnings list is correct per scenario
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.serve.services.ood_detector import OodResult
from src.serve.services.pathology_detector import PathologyResult


# ── helpers ─────────────────────────────────────────────────────────────────

def _fake_gradcam_b64() -> str:
    img = Image.fromarray(np.full((32, 32), 200, dtype=np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _fake_patho_result() -> PathologyResult:
    return PathologyResult(
        scores={"Atelectasis": 0.65, "No Finding": 0.10, "Pneumonia": 0.35},
        findings=["Atelectasis", "Pneumonia"],
        top_finding="Atelectasis",
        top_score=0.65,
        no_finding_score=0.10,
    )


def _ood(decision: str, score: float, reason: str) -> OodResult:
    return OodResult(
        decision=decision,
        cxr_score=score,
        reason=reason,
        is_cxr=(decision != "reject"),
    )


# ── 1. OOD accept path ───────────────────────────────────────────────────────

def test_ood_accept_path(client: TestClient, tiny_jpeg: bytes) -> None:
    """When OOD returns accept, primary runs normally, no ood_in_review_band warning."""
    with patch("src.serve.routers.predict._ood") as mock_ood:
        mock_ood.is_available.return_value = True
        mock_ood.get_ood_detector.return_value.check.return_value = _ood("accept", 0.95, "accepted")

        resp = client.post("/v1/predict", files={"file": ("t.jpg", tiny_jpeg, "image/jpeg")})

    assert resp.status_code == 200
    body = resp.json()
    assert body["ood"]["enabled"] is True
    assert body["ood"]["decision"] == "accept"
    assert body["ood"]["reason"] == "accepted"
    assert body["ood"]["score"] == pytest.approx(0.95)
    assert body["primary_prediction"] is not None
    assert "ood_in_review_band" not in body["warnings"]
    assert body["status"] == "ok"


# ── 2. OOD review path ───────────────────────────────────────────────────────

def test_ood_review_path_runs_inference(client: TestClient, tiny_jpeg: bytes) -> None:
    """When OOD is in review band, inference still runs and warning is added."""
    with patch("src.serve.routers.predict._ood") as mock_ood:
        mock_ood.is_available.return_value = True
        mock_ood.get_ood_detector.return_value.check.return_value = _ood("review", 0.30, "ood_in_review_band")

        resp = client.post("/v1/predict", files={"file": ("t.jpg", tiny_jpeg, "image/jpeg")})

    assert resp.status_code == 200
    body = resp.json()
    assert body["ood"]["decision"] == "review"
    assert body["ood"]["reason"] == "ood_in_review_band"
    assert body["primary_prediction"] is not None  # inference still ran
    assert "ood_in_review_band" in body["warnings"]
    assert body["status"] == "degraded"


# ── 3. OOD reject path ───────────────────────────────────────────────────────

def test_ood_reject_path_returns_422(client: TestClient, tiny_jpeg: bytes) -> None:
    """When OOD rejects the image, HTTP 422 is returned and no inference runs."""
    with patch("src.serve.routers.predict._ood") as mock_ood:
        mock_ood.is_available.return_value = True
        mock_ood.get_ood_detector.return_value.check.return_value = _ood("reject", 0.05, "non_cxr_rejected")

        resp = client.post("/v1/predict", files={"file": ("t.jpg", tiny_jpeg, "image/jpeg")})

    assert resp.status_code == 422
    assert "non_cxr_rejected" in resp.json()["detail"]


# ── 4. Pathologies separate from primary_prediction ─────────────────────────

def test_pathologies_separate_from_primary(client: TestClient, tiny_jpeg: bytes) -> None:
    """pathologies section must be separate from primary_prediction."""
    with patch("src.serve.routers.predict._patho") as mock_patho:
        mock_patho.is_available.return_value = True
        mock_patho.get_pathology_detector.return_value.predict.return_value = _fake_patho_result()

        resp = client.post("/v1/predict", files={"file": ("t.jpg", tiny_jpeg, "image/jpeg")})

    body = resp.json()
    assert "primary_prediction" in body
    assert "pathologies" in body
    pp = body["primary_prediction"]
    pt = body["pathologies"]
    # They must be distinct sections
    assert "label" in pp
    assert "findings" in pt
    assert "label" not in pt
    assert pt["top_finding"] == "Atelectasis"
    assert pt["status"] == "ok"
    assert any(f["name"] == "Atelectasis" for f in pt["findings"])


# ── 5. Localization is marked approximate ────────────────────────────────────

def test_localization_type_is_approximate(client: TestClient, tiny_jpeg: bytes) -> None:
    """Localization type must be 'approximate_attention_region' with a disclaimer."""
    from src.serve.app import app
    app.state.app.gradcam = MagicMock()

    with patch("src.serve.routers.explain.run_gradcam_base64", return_value=_fake_gradcam_b64()):
        resp = client.post("/v1/explain", files={"file": ("t.jpg", tiny_jpeg, "image/jpeg")})

    app.state.app.gradcam = None
    loc = resp.json()["localization"]
    assert loc["type"] == "approximate_attention_region"
    assert "approximate" in loc["disclaimer"].lower() or "not a" in loc["disclaimer"].lower()
    assert loc["bbox"] is not None


def test_localization_has_bbox_coords_schema(client: TestClient, tiny_jpeg: bytes) -> None:
    from src.serve.app import app
    app.state.app.gradcam = MagicMock()

    with patch("src.serve.routers.explain.run_gradcam_base64", return_value=_fake_gradcam_b64()):
        resp = client.post("/v1/explain", files={"file": ("t.jpg", tiny_jpeg, "image/jpeg")})

    app.state.app.gradcam = None
    bbox = resp.json()["localization"]["bbox"]
    for k in ("x1", "y1", "x2", "y2", "x1_norm", "y1_norm", "x2_norm", "y2_norm",
              "width", "height", "area_fraction"):
        assert k in bbox, f"Missing bbox key: {k}"


# ── 6. Pathology failure does not crash primary prediction ───────────────────

def test_pathology_failure_does_not_crash(client: TestClient, tiny_jpeg: bytes) -> None:
    with patch("src.serve.routers.predict._patho") as mock_patho:
        mock_patho.is_available.return_value = True
        mock_patho.get_pathology_detector.return_value.predict.side_effect = RuntimeError("GPU OOM")

        resp = client.post("/v1/predict", files={"file": ("t.jpg", tiny_jpeg, "image/jpeg")})

    assert resp.status_code == 200
    body = resp.json()
    assert body["primary_prediction"] is not None
    assert body["pathologies"]["status"] == "error"
    assert "pathology_inference_failed" in body["warnings"]
    assert body["status"] == "degraded"


# ── 7. Localization failure does not crash primary prediction ────────────────

def test_localization_failure_does_not_crash(client: TestClient, tiny_jpeg: bytes) -> None:
    from src.serve.app import app
    app.state.app.gradcam = MagicMock()

    with patch("src.serve.routers.explain.run_gradcam_base64", return_value=_fake_gradcam_b64()), \
         patch("src.serve.routers.explain.extract_bbox_from_gradcam", side_effect=ValueError("decode error")):
        resp = client.post("/v1/explain", files={"file": ("t.jpg", tiny_jpeg, "image/jpeg")})

    app.state.app.gradcam = None
    assert resp.status_code == 200
    body = resp.json()
    assert body["primary_prediction"] is not None
    assert "localization_generation_failed" in body["warnings"]
    assert body["localization"]["type"] == "not_generated"


# ── 8. Audit fields include OOD/pathology/localization ──────────────────────

def test_audit_row_includes_new_fields(tmp_path: Path, tiny_jpeg: bytes) -> None:
    from src.serve.services.audit import AuditEntry, log_audit
    audit_path = tmp_path / "audit.jsonl"

    entry = AuditEntry(
        request_id="test-hardening-001",
        endpoint="/v1/predict",
        prediction="no_pneumonia",
        calibrated_score=0.72,
        decision="positive",
        latency_ms=42.0,
        model_version="v2-efficientnet-b3-320",
        ood_score=0.91,
        ood_decision="accept",
        ood_reason="accepted",
        top_pathology="Atelectasis",
        top_pathology_score=0.65,
        localization_generated=False,
        localization_region="",
    )
    log_audit(entry, audit_path)

    records = json.loads(audit_path.read_text().strip())
    assert records["ood_score"] == pytest.approx(0.91)
    assert records["ood_decision"] == "accept"
    assert records["ood_reason"] == "accepted"
    assert records["top_pathology"] == "Atelectasis"
    assert records["top_pathology_score"] == pytest.approx(0.65)
    assert records["localization_generated"] is False


# ── 9. Response schema valid when all features disabled ──────────────────────

def test_response_valid_all_features_disabled(client: TestClient, tiny_jpeg: bytes) -> None:
    """All optional features off → response still has all required top-level keys."""
    resp = client.post("/v1/predict", files={"file": ("t.jpg", tiny_jpeg, "image/jpeg")})
    body = resp.json()

    assert body["ood"]["enabled"] is False
    assert body["pathologies"]["enabled"] is False
    assert body["localization"]["enabled"] is False
    assert body["primary_prediction"] is not None
    assert isinstance(body["warnings"], list)
    assert body["status"] in {"ok", "degraded"}


# ── 10. OOD check failure degrades gracefully (no crash) ────────────────────

def test_ood_check_failure_degrades_gracefully(client: TestClient, tiny_jpeg: bytes) -> None:
    """If OOD model itself throws, primary inference still runs with ood_check_failed warning."""
    with patch("src.serve.routers.predict._ood") as mock_ood:
        mock_ood.is_available.return_value = True
        mock_ood.get_ood_detector.return_value.check.side_effect = RuntimeError("CLIP model crash")

        resp = client.post("/v1/predict", files={"file": ("t.jpg", tiny_jpeg, "image/jpeg")})

    assert resp.status_code == 200
    body = resp.json()
    assert body["primary_prediction"] is not None
    assert "ood_check_failed" in body["warnings"]
    assert body["ood"]["reason"] == "check_failed"
    assert body["status"] == "degraded"
