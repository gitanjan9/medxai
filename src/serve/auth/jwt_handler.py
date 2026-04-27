"""JWT token creation and verification.

Token storage strategy (secure):
  access_token  — returned in response body, stored in JS memory only.
                  Never in cookie or localStorage. Lost on page refresh;
                  silently renewed via the refresh cookie.
  refresh_token — HttpOnly Secure cookie, path=/v1/auth.
                  Never readable by JavaScript.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt

JWT_SECRET: str = os.getenv("JWT_SECRET_KEY", "CHANGE_ME_use_openssl_rand_hex_32")
ALGORITHM: str = "HS256"
ACCESS_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
REFRESH_EXPIRE_DAYS: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

COOKIE_SECURE: bool = os.getenv("COOKIE_SECURE", "true").lower() in ("1", "true", "yes")
COOKIE_SAMESITE: str = os.getenv("COOKIE_SAMESITE", "lax")


def create_access_token(user_id: str, email: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_EXPIRE_MINUTES)
    payload = {"sub": user_id, "email": email, "role": role, "exp": expire, "type": "access"}
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_EXPIRE_DAYS)
    payload = {"sub": user_id, "exp": expire, "type": "refresh"}
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """Raises JWTError on invalid/expired token."""
    return jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])


def set_session_cookie(response, session_id: str) -> None:
    """Set the opaque session_id as an HttpOnly cookie (entire site, full lifetime).

    This is the primary session credential sent with every request.
    Value is an opaque UUID — no user data encoded inside.
    """
    response.set_cookie(
        "session_id",
        session_id,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        max_age=REFRESH_EXPIRE_DAYS * 86400,   # same lifetime as the DB session
        path="/",                              # available to all routes
    )


def set_refresh_cookie(response, refresh_token: str) -> None:
    """Set the refresh JWT in an HttpOnly cookie (auth routes only)."""
    response.set_cookie(
        "refresh_token",
        refresh_token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        max_age=REFRESH_EXPIRE_DAYS * 86400,
        path="/v1/auth",
    )


def clear_auth_cookies(response) -> None:
    response.delete_cookie("session_id",    path="/")
    response.delete_cookie("refresh_token", path="/v1/auth")
