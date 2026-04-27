"""Structured logging setup for the MedicalXAI pipeline."""
from __future__ import annotations

import logging
import logging.config
import logging.handlers
from pathlib import Path
from typing import Optional

import yaml


_ROOT_LOGGER_NAME = "medicalxai"


def setup_logging(
    config_path: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
) -> None:
    """Configure logging from a YAML config file or sensible defaults.

    Args:
        config_path: Path to a logging.yaml file.
        log_dir: Override for the file handler log directory.
        level: Fallback level when no YAML config is found.
    """
    if config_path is not None and Path(config_path).exists():
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh)

        if log_dir is not None:
            _patch_log_dir(cfg, log_dir)

        _ensure_log_dirs(cfg)
        logging.config.dictConfig(cfg)
    else:
        _setup_default(level, log_dir)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'medicalxai' hierarchy."""
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_log_dir(cfg: dict, log_dir: Path) -> None:
    """Replace the file handler's filename with one under log_dir."""
    for handler in cfg.get("handlers", {}).values():
        if handler.get("class") in (
            "logging.FileHandler",
            "logging.handlers.RotatingFileHandler",
        ):
            original = Path(handler.get("filename", "train.log")).name
            handler["filename"] = str(log_dir / original)


def _ensure_log_dirs(cfg: dict) -> None:
    """Create parent directories for all file-based handlers."""
    for handler in cfg.get("handlers", {}).values():
        filename = handler.get("filename")
        if filename:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)


def _setup_default(level: int, log_dir: Optional[Path]) -> None:
    """Apply a minimal logging configuration without a YAML file."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(level)

    if not root.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        root.addHandler(ch)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_dir / "train.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        fh.setFormatter(formatter)
        root.addHandler(fh)
