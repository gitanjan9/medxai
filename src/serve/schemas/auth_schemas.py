"""Pydantic schemas for authentication endpoints."""
from __future__ import annotations

from pydantic import BaseModel, EmailStr, field_validator


class LoginRequest(BaseModel):
    email: str          # plain str — validated by DB lookup, not format check
    password: str

    @field_validator("password")
    @classmethod
    def password_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Password must not be empty")
        return v


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: str = ""
    role: str = "user"

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v

    @field_validator("role")
    @classmethod
    def valid_role(cls, v: str) -> str:
        if v not in ("user", "clinician", "admin"):
            return "user"
        return v


class UserOut(BaseModel):
    id: str
    email: str
    name: str
    role: str
    is_active: bool


class AuthResponse(BaseModel):
    user: UserOut
    access_token: str         # store in JS memory — never in localStorage/cookie
    message: str = "OK"


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v
