"""Structured audit logging for inference events.

Primary backend: **Postgres** (when ``DATABASE_URL`` is set).
Fallback:        **JSONL file** at ``MEDXAI_AUDIT_PATH`` (default: artifacts/audit.jsonl).

Each record captures only non-PHI operational fields:
  request_id, endpoint, prediction, calibrated_score, decision, latency_ms, model_version.

No image bytes, filenames, or patient identifiers are ever written.
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.common.logging import get_logger

logger = get_logger("serve.audit")

_write_lock = threading.Lock()


@dataclass
class AuditEntry:
    request_id: str
    endpoint: str               # "/v1/predict" | "/v1/explain"
    prediction: str
    calibrated_score: float
    decision: str               # "positive" | "review" | "negative"
    latency_ms: float
    model_version: str
    environment: str = "dev"    # "dev" | "test" | "prod"
    confidence_band: str = ""   # "high" | "medium" | "low"
    review_reason: str = ""     # "below_high_threshold" | "no_threshold_entry" | ""
    threshold_version: str = "current"
    explanation_requested: bool = False
    status_code: int = 200
    error_message: str = ""
    ood_score: Optional[float] = None
    ood_decision: str = ""      # "accept" | "review" | "reject" | ""
    ood_reason: str = ""        # "accepted" | "ood_in_review_band" | ... | ""
    top_pathology: str = ""
    top_pathology_score: Optional[float] = None
    localization_generated: bool = False
    localization_region: str = ""
    timestamp: str = ""         # filled automatically if empty

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_audit(entry: AuditEntry, audit_path: Path) -> None:
    """Write *entry* to Postgres (preferred) or JSONL fallback.

    Never raises – audit failures are logged as warnings so they never
    interrupt the inference path.
    """
    from src.serve.services import database
    if database.is_available():
        _write_postgres(entry)
    else:
        _write_jsonl(entry, audit_path)


def read_recent_audit(audit_path: Path, n: int = 100) -> list[dict]:
    """Return the last *n* records from Postgres or the JSONL file."""
    from src.serve.services import database
    if database.is_available():
        return _read_postgres(n)
    return _read_jsonl(audit_path, n)


# ---------------------------------------------------------------------------
# Postgres backend
# ---------------------------------------------------------------------------

_INSERT_SQL = """
INSERT INTO inference_audit
    (request_id, endpoint, prediction, calibrated_score,
     decision, latency_ms, model_version,
     environment, confidence_band, review_reason, threshold_version,
     explanation_requested, status_code, error_message,
     ood_score, ood_decision, ood_reason,
     top_pathology, top_pathology_score,
     localization_generated, localization_region,
     created_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s)
"""

_SELECT_SQL = """
SELECT request_id, endpoint, prediction, calibrated_score,
       decision, latency_ms, model_version,
       environment, confidence_band, review_reason, threshold_version,
       explanation_requested, status_code, error_message,
       ood_score, ood_decision, ood_reason,
       top_pathology, top_pathology_score,
       localization_generated, localization_region,
       to_char(created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS timestamp
FROM   inference_audit
ORDER  BY created_at DESC
LIMIT  %s
"""


def _write_postgres(entry: AuditEntry) -> None:
    from src.serve.services.database import get_conn
    try:
        ts = datetime.now(timezone.utc) if not entry.timestamp else \
             datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00"))
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(_INSERT_SQL, (
                    entry.request_id, entry.endpoint, entry.prediction,
                    entry.calibrated_score, entry.decision,
                    entry.latency_ms, entry.model_version,
                    entry.environment, entry.confidence_band, entry.review_reason,
                    entry.threshold_version, entry.explanation_requested,
                    entry.status_code, entry.error_message,
                    entry.ood_score, entry.ood_decision, entry.ood_reason,
                    entry.top_pathology, entry.top_pathology_score,
                    entry.localization_generated, entry.localization_region,
                    ts,
                ))
    except Exception as exc:
        logger.warning("Postgres audit write failed request_id=%s: %s", entry.request_id, exc)


def _read_postgres(n: int) -> list[dict]:
    from src.serve.services.database import get_conn
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(_SELECT_SQL, (n,))
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as exc:
        logger.warning("Postgres audit read failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# JSONL fallback backend
# ---------------------------------------------------------------------------

def _write_jsonl(entry: AuditEntry, audit_path: Path) -> None:
    try:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(entry), ensure_ascii=False) + "\n"
        with _write_lock:
            with audit_path.open("a", encoding="utf-8") as fh:
                fh.write(line)
    except Exception as exc:
        logger.warning("JSONL audit write failed request_id=%s: %s", entry.request_id, exc)


def _read_jsonl(audit_path: Path, n: int) -> list[dict]:
    if not audit_path.exists():
        return []
    try:
        with audit_path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except Exception as exc:
        logger.warning("JSONL audit read failed: %s", exc)
        return []
    records = []
    for line in lines[-n:]:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records
