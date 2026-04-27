"""Persistence layer for predictions, patient records, and retraining queue.

Schema
------
predictions
    Stores every inference result with optional patient metadata and
    clinician feedback (correct / wrong + true_label).

retraining_queue
    Staging table for wrong predictions ready for model fine-tuning.
    Each row maps to one labelled training sample.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from src.common.logging import get_logger
from src.serve.services.database import get_conn, is_available

logger = get_logger("serve.prediction_store")

# ── Schema SQL ────────────────────────────────────────────────────────────────

PREDICTION_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID        REFERENCES users(id) ON DELETE SET NULL,
    request_id      TEXT        NOT NULL UNIQUE,
    patient_name    TEXT        NOT NULL DEFAULT '',
    patient_id      TEXT        NOT NULL DEFAULT '',
    patient_age     INTEGER,
    patient_gender  TEXT        NOT NULL DEFAULT '',
    notes           TEXT        NOT NULL DEFAULT '',
    image_hash      TEXT        NOT NULL,
    model_version   TEXT        NOT NULL,
    primary_label   TEXT        NOT NULL,
    confidence      FLOAT       NOT NULL,
    decision        TEXT        NOT NULL,
    full_result     JSONB       NOT NULL DEFAULT '{}',
    feedback        TEXT        CHECK (feedback IN ('correct','wrong','pending')) DEFAULT 'pending',
    true_label      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS retraining_queue (
    id                  UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id       UUID    NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    image_hash          TEXT    NOT NULL,
    true_label          TEXT    NOT NULL,
    used_in_training    BOOLEAN NOT NULL DEFAULT FALSE,
    training_run_id     TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_user_id    ON predictions (user_id);
CREATE INDEX IF NOT EXISTS idx_predictions_request_id ON predictions (request_id);
CREATE INDEX IF NOT EXISTS idx_predictions_feedback   ON predictions (feedback);
CREATE INDEX IF NOT EXISTS idx_predictions_created    ON predictions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_retrain_used           ON retraining_queue (used_in_training);
"""


def ensure_prediction_schema() -> None:
    if not is_available():
        return
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(PREDICTION_SCHEMA_SQL)
        logger.info("Prediction schema verified / created")
    except Exception as exc:
        logger.error("Failed to create prediction schema: %s", exc)


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class PredictionRecord:
    id: str
    user_id: Optional[str]
    request_id: str
    patient_name: str
    patient_id: str
    patient_age: Optional[int]
    patient_gender: str
    notes: str
    image_hash: str
    model_version: str
    primary_label: str
    confidence: float
    decision: str
    full_result: dict
    feedback: str        # pending | correct | wrong
    true_label: Optional[str]
    created_at: datetime


@dataclass
class RetrainingItem:
    id: str
    prediction_id: str
    image_hash: str
    true_label: str
    created_at: datetime


# ── Write helpers ─────────────────────────────────────────────────────────────

def image_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_prediction(
    *,
    request_id: str,
    user_id: Optional[str],
    image_data: bytes,
    model_version: str,
    primary_label: str,
    confidence: float,
    decision: str,
    full_result: dict,
    patient_name: str = "",
    patient_id: str = "",
    patient_age: Optional[int] = None,
    patient_gender: str = "",
    notes: str = "",
) -> Optional[str]:
    """Insert a prediction row; returns the new UUID or None on failure."""
    if not is_available():
        return None
    img_hash = image_sha256(image_data)
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO predictions
                        (request_id, user_id, patient_name, patient_id, patient_age,
                         patient_gender, notes, image_hash, model_version,
                         primary_label, confidence, decision, full_result)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (request_id) DO NOTHING
                    RETURNING id
                    """,
                    (
                        request_id, user_id, patient_name, patient_id, patient_age,
                        patient_gender, notes, img_hash, model_version,
                        primary_label, confidence, decision,
                        json.dumps(full_result),
                    ),
                )
                row = cur.fetchone()
        return str(row[0]) if row else None
    except Exception as exc:
        logger.error("save_prediction failed: %s", exc)
        return None


def update_patient_info(
    prediction_id: str,
    patient_name: str = "",
    patient_id: str = "",
    patient_age: Optional[int] = None,
    patient_gender: str = "",
    notes: str = "",
) -> bool:
    """Clinician can annotate patient metadata after initial prediction."""
    if not is_available():
        return False
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE predictions
                    SET patient_name=%s, patient_id=%s, patient_age=%s,
                        patient_gender=%s, notes=%s, updated_at=NOW()
                    WHERE id=%s
                    """,
                    (patient_name, patient_id, patient_age, patient_gender, notes, prediction_id),
                )
        return True
    except Exception as exc:
        logger.error("update_patient_info failed: %s", exc)
        return False


def submit_feedback(
    prediction_id: str,
    feedback: str,          # "correct" | "wrong"
    true_label: Optional[str] = None,
) -> bool:
    """Record clinician's verdict. If wrong, queues for retraining."""
    if not is_available():
        return False
    if feedback not in ("correct", "wrong"):
        return False
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE predictions
                    SET feedback=%s, true_label=%s, updated_at=NOW()
                    WHERE id=%s
                    RETURNING image_hash
                    """,
                    (feedback, true_label, prediction_id),
                )
                row = cur.fetchone()
                if not row:
                    return False
                image_hash = row[0]

                # Enqueue for retraining if wrong + corrected label provided
                if feedback == "wrong" and true_label:
                    cur.execute(
                        """
                        INSERT INTO retraining_queue (prediction_id, image_hash, true_label)
                        VALUES (%s, %s, %s)
                        ON CONFLICT DO NOTHING
                        """,
                        (prediction_id, image_hash, true_label),
                    )
                    logger.info(
                        "Queued prediction %s for retraining (true_label=%s)",
                        prediction_id, true_label,
                    )
        return True
    except Exception as exc:
        logger.error("submit_feedback failed: %s", exc)
        return False


# ── Read helpers ──────────────────────────────────────────────────────────────

def _row_to_record(row: tuple) -> PredictionRecord:
    return PredictionRecord(
        id=str(row[0]), user_id=str(row[1]) if row[1] else None,
        request_id=row[2], patient_name=row[3], patient_id=row[4],
        patient_age=row[5], patient_gender=row[6], notes=row[7],
        image_hash=row[8], model_version=row[9], primary_label=row[10],
        confidence=float(row[11]), decision=row[12],
        full_result=row[13] if isinstance(row[13], dict) else {},
        feedback=row[14] or "pending", true_label=row[15],
        created_at=row[16],
    )


_SELECT = """
    SELECT id, user_id, request_id, patient_name, patient_id,
           patient_age, patient_gender, notes, image_hash, model_version,
           primary_label, confidence, decision, full_result,
           feedback, true_label, created_at
    FROM predictions
"""


def list_predictions(
    user_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[PredictionRecord]:
    if not is_available():
        return []
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                if user_id:
                    cur.execute(
                        _SELECT + " WHERE user_id=%s ORDER BY created_at DESC LIMIT %s OFFSET %s",
                        (user_id, limit, offset),
                    )
                else:
                    cur.execute(
                        _SELECT + " ORDER BY created_at DESC LIMIT %s OFFSET %s",
                        (limit, offset),
                    )
                return [_row_to_record(r) for r in cur.fetchall()]
    except Exception as exc:
        logger.error("list_predictions failed: %s", exc)
        return []


def get_prediction(prediction_id: str) -> Optional[PredictionRecord]:
    if not is_available():
        return None
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(_SELECT + " WHERE id=%s", (prediction_id,))
                row = cur.fetchone()
        return _row_to_record(row) if row else None
    except Exception as exc:
        logger.error("get_prediction failed: %s", exc)
        return None


def get_retraining_queue(limit: int = 200) -> list[RetrainingItem]:
    """Returns unprocessed wrong predictions ready for training."""
    if not is_available():
        return []
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, prediction_id, image_hash, true_label, created_at
                    FROM retraining_queue
                    WHERE used_in_training = FALSE
                    ORDER BY created_at ASC LIMIT %s
                    """,
                    (limit,),
                )
                return [
                    RetrainingItem(
                        id=str(r[0]), prediction_id=str(r[1]),
                        image_hash=r[2], true_label=r[3], created_at=r[4],
                    )
                    for r in cur.fetchall()
                ]
    except Exception as exc:
        logger.error("get_retraining_queue failed: %s", exc)
        return []


def mark_retrained(queue_ids: list[str], training_run_id: str) -> None:
    """Mark samples as consumed in a training run."""
    if not is_available() or not queue_ids:
        return
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE retraining_queue
                    SET used_in_training=TRUE, training_run_id=%s
                    WHERE id = ANY(%s::uuid[])
                    """,
                    (training_run_id, queue_ids),
                )
    except Exception as exc:
        logger.error("mark_retrained failed: %s", exc)


def retraining_queue_size() -> int:
    if not is_available():
        return 0
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM retraining_queue WHERE used_in_training = FALSE"
                )
                return cur.fetchone()[0]
    except Exception:
        return 0
