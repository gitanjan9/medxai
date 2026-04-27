"""General-purpose utilities shared across the pipeline."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_reproducibility(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible runs.

    Args:
        seed: Integer seed value.
        deterministic: If True, enable CUDA deterministic mode (slower).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def get_device() -> torch.device:
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_info() -> dict[str, Any]:
    """Return a dict with device metadata for logging."""
    info: dict[str, Any] = {"device": str(get_device())}
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2
        )
    return info


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create path (and parents) if it does not exist, then return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def file_md5(path: Union[str, Path]) -> str:
    """Compute the MD5 hex digest of a file."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def save_json(obj: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Serialize *obj* to JSON at *path*, creating parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        json.dump(obj, fh, indent=indent, default=str)


def load_json(path: Union[str, Path]) -> Any:
    """Load and return a JSON file."""
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


class Timer:
    """Context manager that records elapsed wall-clock time."""

    def __init__(self) -> None:
        self._start: Optional[float] = None
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        assert self._start is not None
        self.elapsed = time.perf_counter() - self._start

    def __str__(self) -> str:
        return f"{self.elapsed:.3f}s"


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
