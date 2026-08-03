"""Pydantic schemas for authentication endpoints."""
from __future__ import annotations

import re
from pydantic import BaseModel, field_validator


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
    email: str
    password: str
    name: str = ""
    role: str = "user"

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        # Basic email validation without email-validator dependency
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, v):
            raise ValueError("Invalid email format")
        return v.lower()

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
