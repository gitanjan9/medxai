"""MONAI Bundle structure validator and artifact organiser."""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from src.common.config import BundleConfig
from src.common.exceptions import BundleValidationError

logger = logging.getLogger("medicalxai.bundle")

# Required bundle layout


_REQUIRED_DIRS = ["configs", "models", "docs"]
_REQUIRED_FILES = [
    "configs/metadata.json",
]
_OPTIONAL_FILES = [
    "configs/train.yaml",
    "configs/inference.yaml",
    "docs/README.md",
]


# Validation



def validate_bundle_structure(bundle_root: Path) -> dict[str, bool]:
    """Check that *bundle_root* conforms to MONAI Bundle conventions.

    Returns a status dict; raises :class:`BundleValidationError` if any
    required component is missing.
    """
    bundle_root = Path(bundle_root)
    status: dict[str, bool] = {}
    errors: list[str] = []

    for d in _REQUIRED_DIRS:
        exists = (bundle_root / d).is_dir()
        status[f"dir:{d}"] = exists
        if not exists:
            errors.append(f"Missing required directory: {bundle_root / d}")

    for f in _REQUIRED_FILES:
        exists = (bundle_root / f).is_file()
        status[f"file:{f}"] = exists
        if not exists:
            errors.append(f"Missing required file: {bundle_root / f}")

    for f in _OPTIONAL_FILES:
        status[f"optional:{f}"] = (bundle_root / f).is_file()

    if errors:
        raise BundleValidationError(
            "Bundle validation failed:\n" + "\n".join(f"  • {e}" for e in errors)
        )

    logger.info("Bundle structure validated OK: %s", bundle_root)
    return status


def report_bundle_status(bundle_root: Path) -> None:
    """Print a human-readable bundle status table to the logger."""
    try:
        status = validate_bundle_structure(bundle_root)
        ok = sum(1 for v in status.values() if v)
        total = len(status)
        logger.info("Bundle status: %d/%d components present", ok, total)
        for key, present in status.items():
            icon = "✓" if present else "✗"
            logger.info("  %s  %s", icon, key)
    except BundleValidationError as exc:
        logger.error("Bundle validation FAILED:\n%s", exc)



# Artifact export



def export_checkpoint_to_bundle(
    checkpoint_path: Path,
    bundle_root: Path,
    model_name: str = "model.pt",
) -> Path:
    """Copy a checkpoint file into the bundle's ``models/`` directory.

    Args:
        checkpoint_path: Source ``.pt`` or ``.pth`` file.
        bundle_root: Root of the classifier bundle.
        model_name: Destination filename inside ``models/``.

    Returns:
        Path to the copied model file.
    """
    dest_dir = bundle_root / "models"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / model_name

    shutil.copy2(checkpoint_path, dest)
    logger.info("Exported checkpoint → %s", dest)
    return dest


def write_bundle_metadata(
    bundle_root: Path,
    cfg: BundleConfig,
    extra: Optional[dict] = None,
) -> None:
    """Write / update ``configs/metadata.json`` with runtime info."""
    meta_path = bundle_root / "configs" / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict = {}
    if meta_path.exists():
        with open(meta_path) as fh:
            existing = json.load(fh)

    existing.update({"name": cfg.name, "version": cfg.version})
    if extra:
        existing.update(extra)

    with open(meta_path, "w") as fh:
        json.dump(existing, fh, indent=2)
    logger.info("Updated bundle metadata: %s", meta_path)
