"""FastAPI dependency injection for authentication and authorization.

Access token arrives in the Authorization header as:
    Authorization: Bearer <token>

The token is stored in JavaScript memory only (never in cookie or
localStorage), so it is immune to XSS cookie theft and CSRF attacks.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from jose import JWTError

from src.serve.auth.jwt_handler import decode_token


class CurrentUser:
    def __init__(self, user_id: str, email: str, role: str):
        self.user_id = user_id
        self.email = email
        self.role = role

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def is_clinician(self) -> bool:
        return self.role in ("admin", "clinician")


async def get_current_user(request: Request) -> CurrentUser:
    # Extract Bearer token from Authorization header (JS memory)
    auth_header = request.headers.get("Authorization", "")
    token: str | None = None
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip() or None

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated — missing Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = decode_token(token)
        if payload.get("type") != "access":
            raise ValueError("wrong token type")
        return CurrentUser(
            user_id=payload["sub"],
            email=payload["email"],
            role=payload.get("role", "user"),
        )
    except (JWTError, KeyError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired or invalid",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def require_admin(user: Annotated[CurrentUser, Depends(get_current_user)]) -> CurrentUser:
    if not user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return user


async def require_clinician(user: Annotated[CurrentUser, Depends(get_current_user)]) -> CurrentUser:
    if not user.is_clinician:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Clinician access required")
    return user


# Type aliases for use in route signatures
AuthRequired = Annotated[CurrentUser, Depends(get_current_user)]
AdminRequired = Annotated[CurrentUser, Depends(require_admin)]
ClinicianRequired = Annotated[CurrentUser, Depends(require_clinician)]
