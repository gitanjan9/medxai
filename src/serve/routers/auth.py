"""Authentication router — /v1/auth/*

Endpoints:
  POST /v1/auth/login      — issue tokens in HttpOnly cookies
  POST /v1/auth/logout     — clear cookies + revoke refresh token
  POST /v1/auth/refresh    — rotate access token using refresh token
  GET  /v1/auth/me         — return current user (requires valid access_token)
  POST /v1/auth/register   — create user (admin only in production)
  GET  /v1/auth/users      — list all users (admin only)
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from jose import JWTError

from src.common.logging import get_logger
from src.serve.auth.dependencies import AdminRequired, AuthRequired, CurrentUser, get_current_user
from src.serve.auth.jwt_handler import (
    REFRESH_EXPIRE_DAYS,
    clear_auth_cookies,
    create_access_token,
    create_refresh_token,
    decode_token,
    set_refresh_cookie,
    set_session_cookie,
)
from src.serve.auth.user_store import (
    authenticate_user,
    create_session,
    create_user,
    get_user_by_session_id,
    get_refresh_token_by_session_id,
    is_refresh_token_valid,
    list_users,
    revoke_session,
    rotate_session_refresh_token,
)
from src.serve.schemas.auth_schemas import (
    AuthResponse,
    LoginRequest,
    RegisterRequest,
    UserOut,
)

router = APIRouter(prefix="/v1/auth", tags=["auth"])
logger = get_logger("serve.auth")

_OPEN_REGISTRATION = os.getenv("MEDXAI_OPEN_REGISTRATION", "false").lower() in ("1", "true", "yes")


# ── Login ─────────────────────────────────────────────────────────────────────

@router.post("/login", response_model=AuthResponse)
async def login(
    credentials: LoginRequest,
    request: Request,
    response: Response,
) -> AuthResponse:
    user = authenticate_user(credentials.email, credentials.password)
    if not user:
        logger.warning("Failed login for email=%s ip=%s", credentials.email, request.client.host if request.client else "?")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    access_token  = create_access_token(user.id, user.email, user.role)
    refresh_token = create_refresh_token(user.id)
    session_id    = str(uuid.uuid4())
    expires_at    = datetime.now(timezone.utc) + timedelta(days=REFRESH_EXPIRE_DAYS)

    create_session(
        user_id=user.id,
        session_id=session_id,
        refresh_token=refresh_token,
        expires_at=expires_at,
        user_agent=request.headers.get("user-agent", ""),
        ip_address=request.client.host if request.client else "",
    )

    # session_id  → opaque HttpOnly cookie, path=/ (sent on every request)
    set_session_cookie(response, session_id)
    # refresh_token → HttpOnly cookie, path=/v1/auth (for token renewal only)
    set_refresh_cookie(response, refresh_token)
    logger.info("Login success: user_id=%s role=%s session=%s", user.id, user.role, session_id)
    # access_token → response body (JS stores in memory only)
    return AuthResponse(
        user=UserOut(id=user.id, email=user.email, name=user.name, role=user.role, is_active=user.is_active),
        access_token=access_token,
    )


# ── Logout ────────────────────────────────────────────────────────────────────

@router.post("/logout")
async def logout(
    response: Response,
    session_id: Annotated[str | None, Cookie()] = None,
) -> dict:
    if session_id:
        revoke_session(session_id)
    clear_auth_cookies(response)
    return {"message": "Logged out"}


# ── Refresh ───────────────────────────────────────────────────────────────────

@router.post("/refresh", response_model=AuthResponse)
async def refresh_access_token(
    response: Response,
    session_id: Annotated[str | None, Cookie()] = None,
) -> AuthResponse:
    """Use the opaque session_id cookie to issue a new access token.
    Also rotates the underlying refresh JWT for forward secrecy.
    """
    if not session_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No session cookie")

    # Look up session in DB — validates expiry + not revoked + user active
    user = get_user_by_session_id(session_id)
    if not user:
        clear_auth_cookies(response)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired or invalid")

    # Rotate the refresh JWT stored in the session row
    new_access  = create_access_token(user.id, user.email, user.role)
    new_refresh = create_refresh_token(user.id)
    new_expires = datetime.now(timezone.utc) + timedelta(days=REFRESH_EXPIRE_DAYS)
    rotate_session_refresh_token(session_id, new_refresh, new_expires)
    set_refresh_cookie(response, new_refresh)

    return AuthResponse(
        user=UserOut(id=user.id, email=user.email, name=user.name, role=user.role, is_active=user.is_active),
        access_token=new_access,
    )


# ── Current user ──────────────────────────────────────────────────────────────

@router.get("/me", response_model=AuthResponse)
async def me(user: AuthRequired, request: Request) -> AuthResponse:
    # Re-issue a fresh access token on /me so the client can store it in memory
    new_access = create_access_token(user.user_id, user.email, user.role)
    return AuthResponse(
        user=UserOut(id=user.user_id, email=user.email, name="", role=user.role, is_active=True),
        access_token=new_access,
    )


# ── Register ──────────────────────────────────────────────────────────────────

@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterRequest,
    request: Request,
) -> AuthResponse:
    # Optionally resolve current user from Bearer header (may be None if unauthenticated)
    current_user: CurrentUser | None = None
    try:
        current_user = await get_current_user(request)
    except HTTPException:
        pass
    # Allow open registration only if explicitly enabled; otherwise admin-only
    if not _OPEN_REGISTRATION:
        if current_user is None or not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Registration is restricted. Contact an administrator.",
            )
    try:
        user = create_user(
            email=body.email,
            password=body.password,
            name=body.name,
            role=body.role if (current_user and current_user.is_admin) else "user",
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))

    logger.info("New user registered: email=%s role=%s", user.email, user.role)
    return AuthResponse(
        user=UserOut(id=user.id, email=user.email, name=user.name, role=user.role, is_active=user.is_active),
        access_token="",   # no auto-login on register
        message="User created",
    )


# ── Admin: list users ─────────────────────────────────────────────────────────

@router.get("/users", response_model=list[UserOut])
async def list_all_users(admin: AdminRequired) -> list[UserOut]:
    users = list_users()
    return [UserOut(id=u.id, email=u.email, name=u.name, role=u.role, is_active=u.is_active) for u in users]
