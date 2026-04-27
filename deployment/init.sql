-- ──────────────────────────────────────────────────────────────────────────────
-- MedicalXAI – Postgres schema
-- Runs automatically when the postgres container starts for the first time
-- (mounted via docker-compose volumes: ./deployment/init.sql:/docker-entrypoint-initdb.d/init.sql)
-- ──────────────────────────────────────────────────────────────────────────────

-- Audit log table: one row per inference request
CREATE TABLE IF NOT EXISTS inference_audit (
    id                    BIGSERIAL PRIMARY KEY,
    request_id            TEXT        NOT NULL,
    endpoint              TEXT        NOT NULL,     -- /v1/predict | /v1/explain
    prediction            TEXT        NOT NULL,
    calibrated_score      REAL        NOT NULL,
    decision              TEXT        NOT NULL,     -- positive | review | negative
    latency_ms            REAL        NOT NULL,
    model_version         TEXT        NOT NULL,
    environment           TEXT        NOT NULL DEFAULT 'dev', -- dev | prod | test
    confidence_band       TEXT        NOT NULL DEFAULT '',    -- high | medium | low
    review_reason         TEXT        NOT NULL DEFAULT '',    -- below_high_threshold | no_threshold_entry
    threshold_version     TEXT        NOT NULL DEFAULT 'current',
    explanation_requested BOOLEAN     NOT NULL DEFAULT FALSE,
    status_code           INTEGER     NOT NULL DEFAULT 200,
    error_message         TEXT        NOT NULL DEFAULT '',
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for fast lookup by request_id or time range
CREATE INDEX IF NOT EXISTS idx_audit_request_id  ON inference_audit (request_id);
CREATE INDEX IF NOT EXISTS idx_audit_created_at  ON inference_audit (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_decision     ON inference_audit (decision);
CREATE INDEX IF NOT EXISTS idx_audit_model        ON inference_audit (model_version);
