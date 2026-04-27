"""Centralised exception handlers – registered on the FastAPI app at startup.

Provides:
- HTTPException handler   → clean JSON body with request_id
- Unhandled exception handler → 500, no stack trace in response, full trace in log
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.common.logging import get_logger

logger = get_logger("serve.error_handler")


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all exception handlers to *app*."""

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        rid = _request_id(request)
        logger.warning(
            "HTTP %d request_id=%s path=%s: %s",
            exc.status_code, rid, request.url.path, exc.detail,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "request_id": rid},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        rid = _request_id(request)
        logger.exception(
            "Unhandled error request_id=%s path=%s: %s",
            rid, request.url.path, exc,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": rid},
        )
