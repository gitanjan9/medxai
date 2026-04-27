"""MONAI Model Zoo / Bundle bootstrap utilities.

Day 1 intent:
- Check whether a suitable CXR classifier exists in the MONAI Model Zoo.
- If found, download and adapt it via the MONAI Bundle API.
- If not found, log a notice and fall back to the local factory build.

These helpers are thin wrappers; the heavy lifting stays in
:mod:`src.train.model_factory`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("medicalxai.zoo")

# Bundles that are suitable CXR classifiers (update as the Zoo grows).
_KNOWN_CXR_BUNDLES: dict[str, str] = {
    # "bundle_name": "zoo_source_url_or_id"
    # As of MONAI 1.3, no publicly released CXR classifier bundle exists
    # in the official Zoo; this dict is intentionally empty and serves as
    # the extension point for Day 2+.
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_zoo_bundle(task: str = "cxr_classification") -> Optional[str]:
    """Search the local registry for a matching MONAI Zoo bundle name.

    Args:
        task: Task descriptor used to filter known bundles.

    Returns:
        Bundle name string, or ``None`` if no match found.
    """
    for name in _KNOWN_CXR_BUNDLES:
        if task.lower() in name.lower():
            return name
    return None


def try_download_zoo_bundle(
    bundle_name: str,
    target_dir: Path,
    version: Optional[str] = None,
) -> Optional[Path]:
    """Attempt to download a bundle from the MONAI Model Zoo.

    Args:
        bundle_name: Registered bundle name.
        target_dir: Local directory to download into.
        version: Optional pinned version string.

    Returns:
        Path to the downloaded bundle root, or ``None`` on failure.
    """
    try:
        from monai.bundle import download  # type: ignore[attr-defined]
    except ImportError:
        logger.warning("monai.bundle.download not available in this MONAI version.")
        return None

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        download(name=bundle_name, bundle_dir=str(target_dir), version=version)
        bundle_path = target_dir / bundle_name
        logger.info("Downloaded Zoo bundle '%s' → %s", bundle_name, bundle_path)
        return bundle_path
    except Exception as exc:
        logger.warning("Failed to download bundle '%s': %s", bundle_name, exc)
        return None


def bootstrap_or_build(
    bundle_root: Path,
    task: str = "cxr_classification",
) -> bool:
    """Try to populate *bundle_root* from the Zoo; return True on success.

    Used at the start of training to attempt a Zoo-based starting point
    before falling back to local model construction.

    Args:
        bundle_root: Directory that should contain the bundle configs.
        task: Task hint for Zoo search.

    Returns:
        ``True`` if a Zoo bundle was installed, ``False`` otherwise.
    """
    bundle_name = find_zoo_bundle(task)
    if bundle_name is None:
        logger.info(
            "No MONAI Zoo bundle found for task='%s'. "
            "Using local bundle-compatible build.",
            task,
        )
        return False

    result = try_download_zoo_bundle(bundle_name, bundle_root.parent)
    if result is None:
        logger.info("Zoo download failed. Falling back to local build.")
        return False

    logger.info("Zoo bootstrap successful: %s", result)
    return True
