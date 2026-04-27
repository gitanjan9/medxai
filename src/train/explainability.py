"""Grad-CAM explainability for the MedicalXAI classifier.

Uses Captum's LayerGradCam when available, with a pure-PyTorch fallback.
Generates per-sample heatmaps + metadata JSON for the val split.

Usage::

    python -m src.train.explainability --config configs/inference.yaml
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.common.logging import get_logger

logger = get_logger("explainability")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ExplanationResult:
    sample_idx: int
    predicted_class: int
    predicted_label: str
    confidence: float
    heatmap_path: Optional[str]   # None if save_heatmaps=False
    metadata_path: str
    gradcam_failed: bool
    failure_reason: Optional[str]


# ---------------------------------------------------------------------------
# Captum / fallback Grad-CAM
# ---------------------------------------------------------------------------


def _get_target_module(model: nn.Module, layer_name: str) -> nn.Module:
    """Resolve a dot-separated layer path to the actual nn.Module."""
    parts = layer_name.split(".")
    module = model
    for part in parts:
        module = getattr(module, part)
    return module


class GradCAMGenerator:
    """Wraps Captum LayerGradCam with a pure-PyTorch fallback.

    The fallback computes gradient-weighted average pooling of the
    target layer activations, equivalent to classic Grad-CAM.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: str,
        device: torch.device,
    ) -> None:
        self.model = model
        self.device = device
        self._layer_name = target_layer
        self._target_module = _get_target_module(model, target_layer)
        self._use_captum = self._try_import_captum()
        if self._use_captum:
            from captum.attr import LayerGradCam
            self._gradcam = LayerGradCam(model, self._target_module)
        else:
            logger.warning(
                "captum not installed – using pure-PyTorch Grad-CAM fallback."
            )
            self._gradcam = None
        # Hooks for fallback
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

    @staticmethod
    def _try_import_captum() -> bool:
        try:
            import captum  # noqa: F401
            return True
        except ImportError:
            return False

    def _register_fallback_hooks(self):
        def _fwd_hook(module, inp, out):
            self._activations = out.detach()

        def _bwd_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        h1 = self._target_module.register_forward_hook(_fwd_hook)
        h2 = self._target_module.register_full_backward_hook(_bwd_hook)
        return h1, h2

    def generate(
        self,
        image: torch.Tensor,   # shape (1, C, H, W)
        target_class: int,
        output_size: tuple[int, int] = (224, 224),
    ) -> Optional[np.ndarray]:
        """Return normalised Grad-CAM heatmap (H, W) in [0, 1], or None on error."""
        image = image.to(self.device).requires_grad_(True)
        self.model.eval()

        try:
            if self._use_captum:
                return self._generate_captum(image, target_class, output_size)
            return self._generate_fallback(image, target_class, output_size)
        except Exception as exc:
            logger.warning("Grad-CAM failed for class %d: %s", target_class, exc)
            return None

    def _generate_captum(
        self,
        image: torch.Tensor,
        target_class: int,
        output_size: tuple[int, int],
    ) -> np.ndarray:
        attr = self._gradcam.attribute(image, target=target_class)
        heatmap = attr.squeeze().cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        return _resize_and_normalise(heatmap, output_size)

    def _generate_fallback(
        self,
        image: torch.Tensor,
        target_class: int,
        output_size: tuple[int, int],
    ) -> np.ndarray:
        h1, h2 = self._register_fallback_hooks()
        try:
            self.model.zero_grad()
            logits = self.model(image)
            score = logits[0, target_class]
            score.backward()
        finally:
            h1.remove()
            h2.remove()

        acts = self._activations    # (1, C_feat, H_f, W_f)
        grads = self._gradients     # (1, C_feat, H_f, W_f)
        weights = grads.mean(dim=(2, 3), keepdim=True)       # (1, C_feat, 1, 1)
        cam = (weights * acts).sum(dim=1).squeeze().cpu().numpy()  # (H_f, W_f)
        cam = np.maximum(cam, 0)
        return _resize_and_normalise(cam, output_size)


def _resize_and_normalise(
    cam: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Bilinear resize + min-max normalise to [0, 1]."""
    import torch.nn.functional as F
    t = torch.from_numpy(cam).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    t = F.interpolate(t, size=output_size, mode="bilinear", align_corners=False)
    cam_np = t.squeeze().numpy()
    mn, mx = cam_np.min(), cam_np.max()
    if mx - mn > 1e-8:
        cam_np = (cam_np - mn) / (mx - mn)
    return cam_np


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------


def save_heatmap(heatmap: np.ndarray, path: Path) -> None:
    """Save heatmap as a grayscale PNG."""
    try:
        from PIL import Image
        img = Image.fromarray((heatmap * 255).astype(np.uint8), mode="L")
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
    except Exception as exc:
        logger.warning("Could not save heatmap to %s: %s", path, exc)


def save_explanation_metadata(meta: dict, path: Path) -> None:
    """Save per-sample explanation metadata as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2)


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------


def generate_explanations(
    config_path: str,
    checkpoint_path: Optional[str] = None,
) -> list[ExplanationResult]:
    """Run Grad-CAM over the val split and save heatmaps + metadata.

    Returns a list of ExplanationResult for downstream review logic.
    """
    from src.common.config import InferenceConfig
    from src.common.logging import setup_logging
    from src.common.schemas import LabelMap
    from src.common.utils import get_device
    from src.train.calibrate import load_calibration, apply_temperature
    from src.train.dataset import CXRDataset, load_and_prepare_dataframe
    from src.train.model_factory import build_model
    from src.train.transforms import build_val_transforms
    from src.train.evaluate import _resolve_checkpoint, _load_checkpoint
    from torch.utils.data import DataLoader

    setup_logging()
    cfg = InferenceConfig.from_yaml(config_path)
    device = get_device()

    # Label map
    label_map = LabelMap.load(cfg.label_map_path)
    cfg.model.num_classes = label_map.num_classes

    # Dataset
    val_csv = cfg.data.val_path or cfg.data.test_path or cfg.data.train_path
    df = load_and_prepare_dataframe(
        val_csv, cfg.data.label_col, label_map,
        merge_map_path=cfg.data.class_merge_map_path,
        tag="Explainability",
    )
    if cfg.explainability.max_samples:
        df = df.head(cfg.explainability.max_samples)
    logger.info("Generating explanations for %d samples", len(df))

    ds = CXRDataset(
        df=df,
        label_map=label_map,
        image_col=cfg.data.image_col,
        label_col=cfg.data.label_col,
        text_col=cfg.data.text_col,
        transforms=build_val_transforms(cfg.data.image_size),
    )

    # Model
    model = build_model(cfg.model).to(device)
    if checkpoint_path:
        ckpt = Path(checkpoint_path)
    else:
        from src.common.config import TrainConfig
        import yaml
        # Build a minimal TrainConfig-like object to reuse _resolve_checkpoint
        with open("configs/train.yaml") as fh:
            raw = yaml.safe_load(fh)
        train_cfg = TrainConfig.model_validate(raw)
        ckpt = _resolve_checkpoint(None, train_cfg)
    if ckpt and Path(ckpt).exists():
        _load_checkpoint(model, Path(ckpt), device)

    # Calibration
    cal: Optional[dict] = None
    if cfg.calibration_path and Path(cfg.calibration_path).exists():
        cal = load_calibration(cfg.calibration_path)
        logger.info("Using calibration T=%.4f", cal["temperature"])

    # Grad-CAM generator
    generator = GradCAMGenerator(
        model=model,
        target_layer=cfg.explainability.target_layer,
        device=device,
    )

    out_dir = Path(cfg.explainability.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[ExplanationResult] = []

    for idx in range(len(ds)):
        sample = ds[idx]
        image = sample["image"].unsqueeze(0)   # (1, C, H, W)
        true_label = int(sample["label"])

        # Get predicted class
        with torch.no_grad():
            logits = model(image.to(device)).cpu().numpy()
        if cal:
            probs = apply_temperature(logits, cal["temperature"])
        else:
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = e / e.sum(axis=1, keepdims=True)

        pred_class = int(np.argmax(probs[0]))
        confidence = float(probs[0, pred_class])
        pred_label = label_map.decode(pred_class)

        # Generate Grad-CAM for the predicted class
        heatmap = generator.generate(
            image,
            target_class=pred_class,
            output_size=tuple(cfg.data.image_size),
        )

        failed = heatmap is None
        failure_reason = "gradcam_returned_none" if failed else None

        # Save heatmap
        heatmap_path: Optional[str] = None
        if not failed and cfg.explainability.save_heatmaps:
            hp = out_dir / f"sample_{idx:05d}_class{pred_class}.png"
            save_heatmap(heatmap, hp)
            heatmap_path = str(hp)

        # Save metadata
        meta = {
            "sample_idx": idx,
            "true_class": true_label,
            "true_label": label_map.decode(true_label),
            "predicted_class": pred_class,
            "predicted_label": pred_label,
            "confidence": confidence,
            "gradcam_failed": failed,
            "failure_reason": failure_reason,
            "heatmap_path": heatmap_path,
        }
        meta_path = out_dir / f"sample_{idx:05d}_meta.json"
        save_explanation_metadata(meta, meta_path)

        results.append(
            ExplanationResult(
                sample_idx=idx,
                predicted_class=pred_class,
                predicted_label=pred_label,
                confidence=confidence,
                heatmap_path=heatmap_path,
                metadata_path=str(meta_path),
                gradcam_failed=failed,
                failure_reason=failure_reason,
            )
        )

        if (idx + 1) % 50 == 0:
            logger.info("  %d / %d samples processed", idx + 1, len(ds))

    # Write summary
    summary_path = out_dir / "explanations_summary.json"
    summary = {
        "total": len(results),
        "failed": sum(r.gradcam_failed for r in results),
        "results": [asdict(r) for r in results],
    }
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(
        "Explanations done. %d/%d failed. Summary → %s",
        summary["failed"], len(results), summary_path,
    )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Generate Grad-CAM explanations")
    p.add_argument("--config", required=True, help="Path to inference.yaml")
    p.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_explanations(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
    )
