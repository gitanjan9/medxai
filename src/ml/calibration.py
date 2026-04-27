"""Post-hoc calibration for multi-label CXR classifiers.

Implements:
  - Temperature scaling (single scalar, minimises NLL on val set)
  - Per-class isotonic regression (sklearn)
  - Expected calibration error (ECE) for binary outputs
  - Reliability diagram per class

Usage
-----
  from src.ml.calibration import calibrate, load_calibrator, apply_calibration

  cal = calibrate(logits_val, targets_val, method="temperature", out_dir="artifacts/cal")
  probs_cal = apply_calibration(logits_test, cal)

Or CLI:
  python -m src.ml.calibration \\
      --logits  artifacts/val_logits.npy \\
      --targets artifacts/val_targets.npy \\
      --out     artifacts/calibration_v3 \\
      --method  temperature
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn

from src.common.logging import get_logger

logger = get_logger("ml.calibration")


# ── ECE ──────────────────────────────────────────────────────────────────────

def ece_binary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error for a single binary column."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(ece)


def ece_multilabel(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict[str, float]:
    """Per-class + macro ECE for (N, C) arrays."""
    C = y_true.shape[1]
    per_class = {str(i): ece_binary(y_true[:, i], y_prob[:, i], n_bins) for i in range(C)}
    per_class["macro"] = float(np.mean(list(per_class.values())))
    return per_class


# ── Temperature scaling ───────────────────────────────────────────────────────

class _TemperatureModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.01)


def fit_temperature(
    logits: np.ndarray,
    targets: np.ndarray,
    lr: float = 0.01,
    max_iter: int = 200,
) -> float:
    """Fit a single temperature scalar on (N, C) logits/targets.

    Returns the fitted temperature T.
    Calibrated probabilities: sigmoid(logits / T).
    """
    logits_t = torch.tensor(logits, dtype=torch.float32)
    targets_t = torch.tensor(targets, dtype=torch.float32)

    model = _TemperatureModel()
    optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=max_iter)
    criterion = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(model(logits_t), targets_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    T = float(model.temperature.item())
    logger.info("Temperature scaling fitted: T=%.4f", T)
    return T


# ── Isotonic calibration ──────────────────────────────────────────────────────

def fit_isotonic(
    probs: np.ndarray,
    targets: np.ndarray,
) -> list[Any]:
    """Fit per-class isotonic regression calibrators.

    Args:
        probs:   (N, C) sigmoid probabilities.
        targets: (N, C) binary targets.

    Returns:
        List of C fitted ``IsotonicRegression`` objects.
    """
    from sklearn.isotonic import IsotonicRegression
    C = probs.shape[1]
    calibrators = []
    for i in range(C):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(probs[:, i], targets[:, i])
        calibrators.append(ir)
    logger.info("Isotonic regression fitted for %d classes.", C)
    return calibrators


def apply_isotonic(probs: np.ndarray, calibrators: list[Any]) -> np.ndarray:
    out = np.empty_like(probs)
    for i, cal in enumerate(calibrators):
        out[:, i] = cal.predict(probs[:, i])
    return out


# ── Reliability diagram ────────────────────────────────────────────────────────

def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob_raw: np.ndarray,
    y_prob_cal: np.ndarray,
    class_names: list[str],
    out_path: str | Path,
) -> None:
    """Plot before/after reliability diagrams for up to 9 classes."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = min(9, len(class_names))
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
        axes = np.array(axes).ravel()
        bins = np.linspace(0, 1, 11)

        for idx in range(n):
            ax = axes[idx]
            for probs, label, ls in [
                (y_prob_raw[:, idx], "Raw", "--"),
                (y_prob_cal[:, idx], "Calibrated", "-"),
            ]:
                avg_pred, avg_true = [], []
                for lo, hi in zip(bins[:-1], bins[1:]):
                    mask = (probs >= lo) & (probs < hi)
                    if mask.sum() > 0:
                        avg_pred.append(float(probs[mask].mean()))
                        avg_true.append(float(y_true[mask, idx].mean()))
                ax.plot(avg_pred, avg_true, ls + "o", lw=1.2, markersize=4, label=label)
            ax.plot([0, 1], [0, 1], "k--", lw=0.7)
            ax.set_title(class_names[idx], fontsize=9)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.legend(fontsize=7)

        for ax in axes[n:]:
            ax.set_visible(False)
        fig.suptitle("Reliability Diagrams", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        logger.info("Reliability diagram saved to %s", out_path)
    except Exception as exc:
        logger.warning("Could not plot reliability diagram: %s", exc)


# ── Top-level calibrate / persist / load ─────────────────────────────────────

def calibrate(
    logits_val: np.ndarray,
    targets_val: np.ndarray,
    method: Literal["temperature", "isotonic"] = "temperature",
    out_dir: str | Path | None = None,
    class_names: list[str] | None = None,
) -> dict:
    """Fit calibration on validation logits/targets and optionally save.

    Returns a calibration artifact dict compatible with ``apply_calibration``.
    """
    out_dir = Path(out_dir) if out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    probs_raw = 1.0 / (1.0 + np.exp(-logits_val))

    if method == "temperature":
        T = fit_temperature(logits_val, targets_val)
        probs_cal = 1.0 / (1.0 + np.exp(-logits_val / T))
        artifact = {"method": "temperature", "temperature": T}
    elif method == "isotonic":
        calibrators = fit_isotonic(probs_raw, targets_val)
        probs_cal = apply_isotonic(probs_raw, calibrators)
        try:
            import joblib
            if out_dir:
                joblib.dump(calibrators, out_dir / "isotonic_calibrators.pkl")
        except ImportError:
            logger.warning("joblib not installed – isotonic calibrators not persisted.")
        artifact = {"method": "isotonic", "calibrators": calibrators}
    else:
        raise ValueError(f"Unknown method: {method!r}.  Choose 'temperature' or 'isotonic'.")

    ece_before = ece_multilabel(targets_val, probs_raw)
    ece_after = ece_multilabel(targets_val, probs_cal)
    artifact["ece_before"] = ece_before
    artifact["ece_after"] = ece_after
    logger.info(
        "ECE before=%.4f  after=%.4f  (macro)",
        ece_before["macro"], ece_after["macro"],
    )

    if out_dir:
        saveable = {k: v for k, v in artifact.items() if k != "calibrators"}
        with open(out_dir / "calibration_meta.json", "w") as f:
            json.dump(saveable, f, indent=2)
        if method == "temperature":
            cal_compat = {"temperature": T}
            with open(out_dir / "calibration.json", "w") as f:
                json.dump(cal_compat, f, indent=2)

        if class_names:
            plot_reliability_diagram(
                targets_val, probs_raw, probs_cal,
                class_names, out_dir / "reliability_diagram.png",
            )

    return artifact


def apply_calibration(logits: np.ndarray, artifact: dict) -> np.ndarray:
    """Return calibrated probabilities given raw logits and a calibration artifact."""
    probs_raw = 1.0 / (1.0 + np.exp(-logits))
    if artifact["method"] == "temperature":
        T = artifact["temperature"]
        return 1.0 / (1.0 + np.exp(-logits / T))
    elif artifact["method"] == "isotonic":
        return apply_isotonic(probs_raw, artifact["calibrators"])
    raise ValueError(f"Unknown calibration method: {artifact['method']!r}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Calibrate CXR classifier probabilities.")
    p.add_argument("--logits",  required=True, help=".npy file of val logits (N, C)")
    p.add_argument("--targets", required=True, help=".npy file of val targets (N, C)")
    p.add_argument("--out",     default="artifacts/calibration_v3", help="Output dir")
    p.add_argument("--method",  default="temperature", choices=["temperature", "isotonic"])
    p.add_argument("--classes", nargs="*", help="Optional class names list")
    args = p.parse_args()

    logits_val = np.load(args.logits)
    targets_val = np.load(args.targets)
    calibrate(
        logits_val, targets_val,
        method=args.method,
        out_dir=args.out,
        class_names=args.classes,
    )
