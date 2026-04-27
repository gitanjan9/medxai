"""Gunicorn configuration for production.

Usage:
    gunicorn src.serve.app:app \
        -c deployment/gunicorn.conf.py \
        -k uvicorn.workers.UvicornWorker

Tune WEB_CONCURRENCY via environment variable (default: 2).
Rule of thumb: 2 × CPU cores for I/O-bound, 1 × for GPU/model-heavy.
"""
import multiprocessing
import os

# ── Workers ───────────────────────────────────────────────────────────────────
workers = int(os.getenv("WEB_CONCURRENCY", max(2, multiprocessing.cpu_count())))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 120          # seconds; long for model inference
keepalive = 5

# ── Binding ───────────────────────────────────────────────────────────────────
bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"

# ── Logging ───────────────────────────────────────────────────────────────────
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")
accesslog = "-"        # stdout
errorlog  = "-"        # stderr
access_log_format = '%(h)s "%(r)s" %(s)s %(b)s %(D)sµs'

# ── Process ───────────────────────────────────────────────────────────────────
preload_app = True     # load model once, fork workers (saves ~600 MB × N workers)
max_requests = 500     # recycle workers to prevent memory leaks
max_requests_jitter = 50
graceful_timeout = 30
