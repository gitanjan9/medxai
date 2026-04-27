"""Bundle artifact export utilities.

Copies all production artifacts (checkpoint, calibration, thresholds,
label map, inference config) into the MONAI Bundle directory so that
Day 3 serving code can load them with a single bundle_root path.

Usage::

    python -m src.train.artifacts --config configs/inference.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from src.common.logging import get_logger

logger = get_logger("artifacts")

# Canonical bundle artifact paths (relative to bundle_root)
BUNDLE_ARTIFACTS = {
    "label_map":   "configs/label_map.json",
    "calibration": "configs/calibration.json",
    "thresholds":  "configs/thresholds.json",
    "inference":   "configs/inference.yaml",
    "model":       "models/best_checkpoint.pt",
}


# ---------------------------------------------------------------------------
# Core export
# ---------------------------------------------------------------------------


def export_bundle_artifacts(
    config_path: str,
    bundle_root: Optional[str] = None,
) -> dict[str, str]:
    """Copy all production artifacts into the bundle directory.

    Returns a dict of artifact_name → final_path.
    """
    from src.common.config import InferenceConfig, TrainConfig
    from src.train.evaluate import _resolve_checkpoint

    inf_cfg = InferenceConfig.from_yaml(config_path)
    bundle_path = Path(bundle_root) if bundle_root else inf_cfg.bundle.bundle_root
    bundle_path.mkdir(parents=True, exist_ok=True)

    exported: dict[str, str] = {}

    # ---- label map ----
    _copy_artifact(
        inf_cfg.label_map_path,
        bundle_path / BUNDLE_ARTIFACTS["label_map"],
        "label_map",
        exported,
    )

    # ---- calibration ----
    if inf_cfg.calibration_path:
        _copy_artifact(
            inf_cfg.calibration_path,
            bundle_path / BUNDLE_ARTIFACTS["calibration"],
            "calibration",
            exported,
        )

    # ---- thresholds ----
    if inf_cfg.thresholds_path:
        _copy_artifact(
            inf_cfg.thresholds_path,
            bundle_path / BUNDLE_ARTIFACTS["thresholds"],
            "thresholds",
            exported,
        )

    # ---- inference config ----
    _copy_artifact(
        Path(config_path),
        bundle_path / BUNDLE_ARTIFACTS["inference"],
        "inference",
        exported,
    )

    # ---- best checkpoint ----
    try:
        import yaml
        with open("configs/train.yaml") as fh:
            raw = yaml.safe_load(fh)
        from src.common.config import TrainConfig
        train_cfg = TrainConfig.model_validate(raw)
        ckpt = _resolve_checkpoint(None, train_cfg)
        if ckpt and Path(ckpt).exists():
            dest = bundle_path / BUNDLE_ARTIFACTS["model"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ckpt, dest)
            exported["model"] = str(dest)
            logger.info("Checkpoint → %s", dest)
        else:
            logger.warning("No checkpoint found; model artifact not exported.")
    except Exception as exc:
        logger.warning("Could not export checkpoint: %s", exc)

    # ---- write manifest ----
    manifest = {
        "bundle_name": inf_cfg.bundle.name,
        "version": inf_cfg.bundle.version,
        "artifacts": exported,
    }
    manifest_path = bundle_path / "configs" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Bundle manifest → %s", manifest_path)
    logger.info("Bundle export complete: %d artifacts", len(exported))
    return exported


def _copy_artifact(
    src: Path,
    dst: Path,
    name: str,
    exported: dict,
) -> None:
    """Copy src to dst, creating parent dirs. Logs and records result."""
    src = Path(src)
    if not src.exists():
        logger.warning("Artifact '%s' not found at %s – skipping.", name, src)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    exported[name] = str(dst)
    logger.info("%s → %s", name, dst)


# ---------------------------------------------------------------------------
# Bundle structure validation
# ---------------------------------------------------------------------------


REQUIRED_BUNDLE_FILES = [
    "configs/metadata.json",
    "configs/inference.yaml",
    "configs/label_map.json",
    "models/best_checkpoint.pt",
]


def validate_bundle(bundle_root: str) -> tuple[bool, list[str]]:
    """Check that required bundle files exist.

    Returns (is_valid, list_of_missing_files).
    """
    root = Path(bundle_root)
    missing = [
        f for f in REQUIRED_BUNDLE_FILES
        if not (root / f).exists()
    ]
    is_valid = len(missing) == 0
    if is_valid:
        logger.info("Bundle at %s is valid.", root)
    else:
        logger.warning(
            "Bundle at %s is incomplete. Missing: %s", root, missing
        )
    return is_valid, missing


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export bundle artifacts for Day 3 serving"
    )
    p.add_argument("--config", required=True, help="Path to inference.yaml")
    p.add_argument("--bundle-root", default=None, help="Override bundle root path")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    exported = export_bundle_artifacts(
        config_path=args.config,
        bundle_root=args.bundle_root,
    )
    print(f"\nExported {len(exported)} artifacts:")
    for name, path in exported.items():
        print(f"  {name:<15} {path}")
