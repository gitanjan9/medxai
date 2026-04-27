"""Structured request/response logging middleware.

Logs one JSON line per request containing:
  method, path, status_code, latency_ms, request_id

Never logs query strings, headers, or body content to avoid PHI leakage.
"""
from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.common.logging import get_logger

logger = get_logger("serve.access")


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request as a structured record after the response is sent."""

    async def dispatch(self, request: Request, call_next) -> Response:
        t0 = time.perf_counter()
        response: Response = await call_next(request)
        latency_ms = (time.perf_counter() - t0) * 1000

        request_id = getattr(request.state, "request_id", "unknown")
        logger.info(
            "access method=%s path=%s status=%d latency_ms=%.1f request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
            request_id,
        )
        return response
