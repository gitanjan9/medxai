"""User CRUD against Postgres (psycopg2 pool).

All queries go through the shared connection pool in database.py.
Falls back gracefully when Postgres is unavailable.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Optional

from src.common.logging import get_logger
from src.serve.auth.password import hash_password, verify_password
from src.serve.services.database import get_conn, is_available

logger = get_logger("serve.auth.user_store")

VALID_ROLES = {"user", "clinician", "admin"}


@dataclass
class User:
    id: str
    email: str
    name: str
    role: str
    is_active: bool


# ── Schema creation (called from database._ensure_schema) ────────────────────

AUTH_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    email       TEXT        NOT NULL UNIQUE,
    name        TEXT        NOT NULL DEFAULT '',
    password    TEXT        NOT NULL,
    role        TEXT        NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'clinician', 'admin')),
    is_active   BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sessions (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID        NOT NULL UNIQUE DEFAULT gen_random_uuid(),
    user_id         UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    refresh_token   TEXT        NOT NULL UNIQUE,
    user_agent      TEXT        NOT NULL DEFAULT '',
    ip_address      TEXT        NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMPTZ NOT NULL,
    revoked         BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_users_email          ON users (email);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id     ON sessions (user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_token       ON sessions (refresh_token);
CREATE INDEX IF NOT EXISTS idx_sessions_expires     ON sessions (expires_at);
"""


def ensure_auth_schema() -> None:
    if not is_available():
        return
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                # 1. Create tables and base indexes
                cur.execute(AUTH_SCHEMA_SQL)
                # 2. Add session_id column if missing (migration for pre-existing tables)
                cur.execute("""
                    ALTER TABLE sessions
                    ADD COLUMN IF NOT EXISTS session_id UUID UNIQUE DEFAULT gen_random_uuid()
                """)
                # 3. Create index on session_id (safe now that column exists)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sessions_session_id
                    ON sessions (session_id)
                """)
        logger.info("Auth schema verified / created")
        _seed_default_admin()
    except Exception as exc:
        logger.error("Failed to create auth schema: %s", exc)


def _seed_default_admin() -> None:
    """Create a default admin account if the users table is empty."""
    import os
    admin_email    = os.getenv("MEDXAI_ADMIN_EMAIL",    "admin@medicalxai.local")
    admin_password = os.getenv("MEDXAI_ADMIN_PASSWORD", "admin123")
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM users")
                count = cur.fetchone()[0]
        if count == 0:
            create_user(admin_email, admin_password, "Admin", "admin")
            logger.info("Seeded default admin user: %s", admin_email)
    except Exception as exc:
        logger.warning("Could not seed default admin: %s", exc)


# ── User CRUD ─────────────────────────────────────────────────────────────────

def get_user_by_email(email: str) -> Optional[User]:
    if not is_available():
        return None
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, email, name, role, is_active FROM users WHERE email = %s",
                (email.lower(),),
            )
            row = cur.fetchone()
    if not row:
        return None
    return User(id=str(row[0]), email=row[1], name=row[2], role=row[3], is_active=row[4])


def get_user_by_id(user_id: str) -> Optional[User]:
    if not is_available():
        return None
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, email, name, role, is_active FROM users WHERE id = %s",
                (user_id,),
            )
            row = cur.fetchone()
    if not row:
        return None
    return User(id=str(row[0]), email=row[1], name=row[2], role=row[3], is_active=row[4])


def authenticate_user(email: str, password: str) -> Optional[User]:
    """Return User if credentials are correct, None otherwise."""
    if not is_available():
        return _dev_authenticate(email, password)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, email, name, role, is_active, password FROM users WHERE email = %s",
                (email.lower(),),
            )
            row = cur.fetchone()
    if not row:
        return None
    user_id, db_email, name, role, is_active, hashed = row
    if not is_active:
        return None
    if not verify_password(password, hashed):
        return None
    return User(id=str(user_id), email=db_email, name=name, role=role, is_active=is_active)


def create_user(email: str, password: str, name: str = "", role: str = "user") -> User:
    if role not in VALID_ROLES:
        role = "user"
    hashed = hash_password(password)
    user_id = str(uuid.uuid4())
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO users (id, email, name, password, role)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (email) DO NOTHING
                   RETURNING id, email, name, role, is_active""",
                (user_id, email.lower(), name, hashed, role),
            )
            row = cur.fetchone()
    if not row:
        raise ValueError(f"Email already registered: {email}")
    return User(id=str(row[0]), email=row[1], name=row[2], role=row[3], is_active=row[4])


def list_users() -> list[User]:
    if not is_available():
        return []
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, email, name, role, is_active FROM users ORDER BY created_at DESC")
            rows = cur.fetchall()
    return [User(id=str(r[0]), email=r[1], name=r[2], role=r[3], is_active=r[4]) for r in rows]


def deactivate_user(user_id: str) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET is_active = FALSE WHERE id = %s", (user_id,))


# ── Session store ─────────────────────────────────────────────────────────────

def create_session(
    user_id: str,
    session_id: str,
    refresh_token: str,
    expires_at,
    user_agent: str = "",
    ip_address: str = "",
) -> None:
    """Persist a new browser session with its opaque session_id and refresh JWT."""
    if not is_available():
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO sessions
                       (session_id, user_id, refresh_token, user_agent, ip_address, expires_at)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (session_id, user_id, refresh_token, user_agent, ip_address, expires_at),
            )


def get_user_by_session_id(session_id: str) -> Optional[User]:
    """Look up the active user for an opaque session_id cookie."""
    if not is_available():
        return None
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT u.id, u.email, u.name, u.role, u.is_active
                   FROM sessions s
                   JOIN users u ON u.id = s.user_id
                   WHERE s.session_id = %s
                     AND s.revoked   = FALSE
                     AND s.expires_at > NOW()
                     AND u.is_active  = TRUE""",
                (session_id,),
            )
            row = cur.fetchone()
    if not row:
        return None
    return User(id=str(row[0]), email=row[1], name=row[2], role=row[3], is_active=row[4])


def get_refresh_token_by_session_id(session_id: str) -> Optional[str]:
    """Return the refresh JWT associated with a session_id."""
    if not is_available():
        return None
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT refresh_token FROM sessions
                   WHERE session_id = %s AND revoked = FALSE AND expires_at > NOW()""",
                (session_id,),
            )
            row = cur.fetchone()
    return row[0] if row else None


def revoke_session(session_id: str) -> None:
    """Revoke a session by its opaque session_id (called on logout)."""
    if not is_available():
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET revoked = TRUE WHERE session_id = %s",
                (session_id,),
            )


def rotate_session_refresh_token(
    session_id: str,
    new_refresh_token: str,
    new_expires_at,
) -> None:
    """Replace the refresh JWT in the session row (token rotation)."""
    if not is_available():
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE sessions
                   SET refresh_token = %s, expires_at = %s
                   WHERE session_id = %s AND revoked = FALSE""",
                (new_refresh_token, new_expires_at, session_id),
            )


# ── Legacy helpers (kept for backward compat) ─────────────────────────────────

def save_refresh_token(
    user_id: str,
    refresh_token: str,
    expires_at,
    user_agent: str = "",
    ip_address: str = "",
) -> None:
    if not is_available():
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO sessions (user_id, refresh_token, user_agent, ip_address, expires_at)
                   VALUES (%s, %s, %s, %s, %s)""",
                (user_id, refresh_token, user_agent, ip_address, expires_at),
            )


def revoke_refresh_token(refresh_token: str) -> None:
    if not is_available():
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET revoked = TRUE WHERE refresh_token = %s",
                (refresh_token,),
            )


def is_refresh_token_valid(refresh_token: str) -> bool:
    if not is_available():
        return True  # dev mode: skip DB check
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT 1 FROM sessions
                   WHERE refresh_token = %s AND revoked = FALSE AND expires_at > NOW()""",
                (refresh_token,),
            )
            return cur.fetchone() is not None


# ── Dev-mode fallback (no Postgres) ──────────────────────────────────────────
# Hardcoded admin account used ONLY when DATABASE_URL is not set.
_DEV_ADMIN_EMAIL = "admin@medicalxai.local"
_DEV_ADMIN_PASSWORD = "admin123"


def _dev_authenticate(email: str, password: str) -> Optional[User]:
    if email.lower() == _DEV_ADMIN_EMAIL and password == _DEV_ADMIN_PASSWORD:
        logger.warning("DEV MODE: using hardcoded admin account — set DATABASE_URL in production")
        return User(
            id="00000000-0000-0000-0000-000000000001",
            email=_DEV_ADMIN_EMAIL,
            name="Dev Admin",
            role="admin",
            is_active=True,
        )
    return None
