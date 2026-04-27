"""Postgres connection pool for the serve layer.

Uses ``psycopg2`` (sync, thread-safe) so audit writes don't require an
event loop and can be called from anywhere – including background threads.

The pool is created once at API startup (``init_db``) and closed on shutdown
(``close_db``).  All other modules call ``get_conn()`` as a context manager.

If ``DATABASE_URL`` is not set the module degrades gracefully: every call
is a no-op and audit falls back to the flat JSONL file.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator, Optional

from src.common.logging import get_logger

logger = get_logger("serve.database")

# Module-level pool (None when DB is disabled)
_pool = None


def _database_url() -> Optional[str]:
    return os.environ.get("DATABASE_URL") or None


def init_db() -> bool:
    """Create the connection pool.  Returns True if Postgres is available."""
    global _pool
    url = _database_url()
    if not url:
        logger.info("DATABASE_URL not set – Postgres audit disabled (JSONL fallback active)")
        return False
    try:
        from psycopg2 import pool as pg_pool
        _pool = pg_pool.ThreadedConnectionPool(minconn=1, maxconn=10, dsn=url)
        logger.info("Postgres pool initialised: %s", _sanitise_url(url))
        _ensure_schema()
        return True
    except Exception as exc:
        logger.error("Postgres init failed – JSONL fallback active: %s", exc)
        _pool = None
        return False


def close_db() -> None:
    """Close all pool connections.  Called in the lifespan shutdown."""
    global _pool
    if _pool is not None:
        try:
            _pool.closeall()
            logger.info("Postgres pool closed")
        except Exception as exc:
            logger.warning("Error closing Postgres pool: %s", exc)
        _pool = None


def is_available() -> bool:
    return _pool is not None


@contextmanager
def get_conn() -> Generator:
    """Context manager that yields a psycopg2 connection from the pool.

    Commits on clean exit, rolls back on exception.
    Must only be called when ``is_available()`` is True.
    """
    conn = _pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


def _ensure_schema() -> None:
    """Create or migrate the audit table (idempotent)."""
    create = """
    CREATE TABLE IF NOT EXISTS inference_audit (
        id                    BIGSERIAL PRIMARY KEY,
        request_id            TEXT        NOT NULL,
        endpoint              TEXT        NOT NULL,
        prediction            TEXT        NOT NULL,
        calibrated_score      REAL        NOT NULL,
        decision              TEXT        NOT NULL,
        latency_ms            REAL        NOT NULL,
        model_version         TEXT        NOT NULL,
        environment           TEXT        NOT NULL DEFAULT 'dev',
        confidence_band       TEXT        NOT NULL DEFAULT '',
        review_reason         TEXT        NOT NULL DEFAULT '',
        threshold_version     TEXT        NOT NULL DEFAULT 'current',
        explanation_requested BOOLEAN     NOT NULL DEFAULT FALSE,
        status_code           INTEGER     NOT NULL DEFAULT 200,
        error_message         TEXT        NOT NULL DEFAULT '',
        created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """
    # Migrate existing tables that were created before new columns existed
    migrate = """
    ALTER TABLE inference_audit ADD COLUMN IF NOT EXISTS environment           TEXT    NOT NULL DEFAULT 'dev';
    ALTER TABLE inference_audit ADD COLUMN IF NOT EXISTS confidence_band       TEXT    NOT NULL DEFAULT '';
    ALTER TABLE inference_audit ADD COLUMN IF NOT EXISTS review_reason         TEXT    NOT NULL DEFAULT '';
    ALTER TABLE inference_audit ADD COLUMN IF NOT EXISTS threshold_version     TEXT    NOT NULL DEFAULT 'current';
    ALTER TABLE inference_audit ADD COLUMN IF NOT EXISTS explanation_requested BOOLEAN NOT NULL DEFAULT FALSE;
    ALTER TABLE inference_audit ADD COLUMN IF NOT EXISTS status_code           INTEGER NOT NULL DEFAULT 200;
    ALTER TABLE inference_audit ADD COLUMN IF NOT EXISTS error_message         TEXT    NOT NULL DEFAULT '';
    ALTER TABLE inference_audit ADD COLUMN IF NOT EXISTS ood_score             REAL;
    ALTER TABLE inference_audit ADD COLUMN IF NOT EXISTS ood_decision          TEXT    NOT NULL DEFAULT '';
    ALTER TABLE inference_audit ADD COLUMN IF NOT EXISTS ood_reason            TEXT    NOT NULL DEFAULT '';
    ALTER TABLE inference_audit ADD COLUMN IF NOT EXISTS top_pathology         TEXT    NOT NULL DEFAULT '';
    ALTER TABLE inference_audit ADD COLUMN IF NOT EXISTS top_pathology_score   REAL;
    ALTER TABLE inference_audit ADD COLUMN IF NOT EXISTS localization_generated BOOLEAN NOT NULL DEFAULT FALSE;
    ALTER TABLE inference_audit ADD COLUMN IF NOT EXISTS localization_region   TEXT    NOT NULL DEFAULT '';
    """
    indexes = """
    CREATE INDEX IF NOT EXISTS idx_audit_request_id  ON inference_audit (request_id);
    CREATE INDEX IF NOT EXISTS idx_audit_created_at  ON inference_audit (created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_audit_decision    ON inference_audit (decision);
    CREATE INDEX IF NOT EXISTS idx_audit_model       ON inference_audit (model_version);
    CREATE INDEX IF NOT EXISTS idx_audit_environment ON inference_audit (environment);
    CREATE INDEX IF NOT EXISTS idx_audit_review      ON inference_audit (review_reason) WHERE review_reason <> '';
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(create)
            cur.execute(migrate)
            cur.execute(indexes)
    logger.info("Postgres schema verified / created")


def _sanitise_url(url: str) -> str:
    """Mask password in URL for safe logging."""
    import re
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", url)
