"""Comprehensive multi-label evaluation for CXR classifiers.

Computes per-class AUROC, AUPRC, sensitivity, specificity, F1, precision,
recall, and calibration error.  Finds optimal decision thresholds (Youden-J
and F1-maximising) from the validation/test split.  Saves metric tables,
JSON thresholds, ROC/PR curve PNGs, and a reliability diagram.

Usage
-----
  python -m src.evaluation.evaluate \\
      --checkpoint artifacts/v3/checkpoints \\
      --data      artifacts/v3_disease_labels.csv \\
      --label-map artifacts/v3/label_map.json \\
      --out       reports/v3

Or import directly:
  from src.evaluation.evaluate import run_evaluation
  report = run_evaluation(checkpoint_dir, data_csv, label_map_json, out_dir)
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.common.logging import get_logger
from src.train.dataset import MultiLabelCXRDataset
from src.train.transforms import build_val_transforms

logger = get_logger("evaluation.evaluate")

warnings.filterwarnings("ignore", category=UserWarning)

_REPORT_DIRS = ("roc_curves", "pr_curves")


# ── metric helpers ────────────────────────────────────────────────────────────

def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _sensitivity_specificity(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return sens, spec


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Compute binary ECE for one class."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        avg_conf = float(y_prob[mask].mean())
        avg_acc = float(y_true[mask].mean())
        ece += (mask.sum() / n) * abs(avg_conf - avg_acc)
    return ece


def optimal_threshold_youden(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Threshold that maximises sensitivity + specificity − 1 (Youden-J)."""
    from sklearn.metrics import roc_curve
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def optimal_threshold_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Threshold that maximises F1 on the precision-recall curve."""
    from sklearn.metrics import precision_recall_curve
    if y_true.sum() == 0:
        return 0.5
    prec, rec, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    idx = int(np.argmax(f1[:-1]))
    return float(thresholds[idx])


# ── inference pass ───────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (logits, targets) arrays shaped (N, C)."""
    all_logits, all_targets = [], []
    model.eval()
    for batch in loader:
        imgs = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = model(imgs)
        all_logits.append(logits.cpu().float())
        all_targets.append(labels.cpu().float())
    return (
        torch.cat(all_logits).numpy(),
        torch.cat(all_targets).numpy(),
    )


# ── plotting ──────────────────────────────────────────────────────────────────

def _plot_roc(y_true: np.ndarray, y_score: np.ndarray, cls: str, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = _auroc(y_true, y_score)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, lw=1.5, label=f"AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(f"ROC – {cls}"); ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"{cls.replace(' ', '_')}_roc.png", dpi=100)
        plt.close(fig)
    except Exception as exc:
        logger.debug("ROC plot failed for %s: %s", cls, exc)


def _plot_pr(y_true: np.ndarray, y_score: np.ndarray, cls: str, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap = _auprc(y_true, y_score)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(rec, prec, lw=1.5, label=f"AP={ap:.3f}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title(f"PR – {cls}"); ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"{cls.replace(' ', '_')}_pr.png", dpi=100)
        plt.close(fig)
    except Exception as exc:
        logger.debug("PR plot failed for %s: %s", cls, exc)


def _plot_reliability(y_true: np.ndarray, y_prob: np.ndarray, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        avg_pred, avg_true = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (y_prob >= lo) & (y_prob < hi)
            if mask.sum() == 0:
                avg_pred.append(float("nan"))
                avg_true.append(float("nan"))
            else:
                avg_pred.append(float(y_prob[mask].mean()))
                avg_true.append(float(y_true[mask].mean()))
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect")
        ax.plot(avg_pred, avg_true, "o-", lw=1.5, label="Model")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction positive")
        ax.set_title("Reliability diagram (macro-avg)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "reliability_diagram.png", dpi=100)
        plt.close(fig)
    except Exception as exc:
        logger.debug("Reliability plot failed: %s", exc)


# ── main evaluation logic ────────────────────────────────────────────────────

def run_evaluation(
    checkpoint_dir: str | Path,
    data_csv: str | Path,
    label_map_json: str | Path,
    out_dir: str | Path,
    image_size: int = 320,
    batch_size: int = 32,
    device_str: str = "auto",
    image_col: str = "image",
) -> dict:
    """Run full evaluation and return a summary dict.

    Saves to *out_dir*:
      metrics.csv, thresholds.json, roc_curves/*.png, pr_curves/*.png,
      reliability_diagram.png.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in _REPORT_DIRS:
        (out_dir / d).mkdir(exist_ok=True)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    with open(label_map_json) as f:
        lmap = json.load(f)
    label_cols: list[str] = lmap.get("classes", list(lmap.get("str_to_idx", {}).keys()))

    from src.serve.services.model_loader import load_model, resolve_best_checkpoint
    ckpt = resolve_best_checkpoint(Path(checkpoint_dir))
    model = load_model("efficientnet_b3", len(label_cols), ckpt, device)
    model.eval()

    transforms = build_val_transforms(image_size=(image_size, image_size))
    df = pd.read_csv(data_csv)
    ds = MultiLabelCXRDataset(df, image_col=image_col, label_cols=label_cols,
                               transform=transforms)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=False)

    logger.info("Running inference on %d samples (%d classes)…", len(ds), len(label_cols))
    logits, targets = collect_predictions(model, loader, device)
    probs = 1.0 / (1.0 + np.exp(-logits))   # sigmoid

    rows = []
    thresholds_youden: dict[str, float] = {}
    thresholds_f1: dict[str, float] = {}

    for i, cls in enumerate(label_cols):
        y_true = targets[:, i]
        y_prob = probs[:, i]
        prevalence = float(y_true.mean())

        auc = _auroc(y_true, y_prob)
        ap = _auprc(y_true, y_prob)
        ece = expected_calibration_error(y_true, y_prob)
        thr_j = optimal_threshold_youden(y_true, y_prob)
        thr_f1 = optimal_threshold_f1(y_true, y_prob)
        y_pred_j = (y_prob >= thr_j).astype(int)
        y_pred_f1 = (y_prob >= thr_f1).astype(int)
        sens_j, spec_j = _sensitivity_specificity(y_true, y_pred_j)
        f1_j = f1_score(y_true, y_pred_j, zero_division=0)
        prec_j = precision_score(y_true, y_pred_j, zero_division=0)
        rec_j = recall_score(y_true, y_pred_j, zero_division=0)

        thresholds_youden[cls] = round(thr_j, 4)
        thresholds_f1[cls] = round(thr_f1, 4)

        rows.append({
            "class": cls,
            "prevalence": round(prevalence, 4),
            "auroc": round(auc, 4) if not np.isnan(auc) else None,
            "auprc": round(ap, 4) if not np.isnan(ap) else None,
            "ece": round(ece, 4),
            "threshold_youden": round(thr_j, 4),
            "threshold_f1": round(thr_f1, 4),
            "sensitivity_youden": round(sens_j, 4) if not np.isnan(sens_j) else None,
            "specificity_youden": round(spec_j, 4) if not np.isnan(spec_j) else None,
            "f1_youden": round(f1_j, 4),
            "precision_youden": round(prec_j, 4),
            "recall_youden": round(rec_j, 4),
        })

        _plot_roc(y_true, y_prob, cls, out_dir / "roc_curves")
        _plot_pr(y_true, y_prob, cls, out_dir / "pr_curves")
        logger.info("  %-28s AUROC=%.3f  AUPRC=%.3f  ECE=%.3f  thr_j=%.3f",
                    cls, auc if not np.isnan(auc) else -1,
                    ap if not np.isnan(ap) else -1, ece, thr_j)

    _plot_reliability(targets.ravel(), probs.ravel(), out_dir)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)

    thresholds_out = {
        "source": "youden_j",
        "note": "Use threshold_f1 for higher sensitivity; use threshold_youden for balanced sens/spec.",
        "thresholds": thresholds_youden,
        "thresholds_f1": thresholds_f1,
    }
    with open(out_dir / "thresholds.json", "w") as f:
        json.dump(thresholds_out, f, indent=2)

    valid_aurocs = [r["auroc"] for r in rows if r["auroc"] is not None]
    macro_auroc = float(np.mean(valid_aurocs)) if valid_aurocs else float("nan")
    summary = {
        "n_samples": len(ds),
        "n_classes": len(label_cols),
        "macro_auroc": round(macro_auroc, 4),
        "macro_auprc": round(float(np.nanmean([r["auprc"] for r in rows if r["auprc"] is not None])), 4),
        "macro_ece": round(float(np.mean([r["ece"] for r in rows])), 4),
        "out_dir": str(out_dir),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Evaluation complete → macro_auroc=%.4f  macro_auprc=%.4f  macro_ece=%.4f",
        summary["macro_auroc"], summary["macro_auprc"], summary["macro_ece"],
    )
    logger.info("Saved: %s/metrics.csv  thresholds.json  summary.json", out_dir)
    return summary


# ── CLI entry point ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate multi-label CXR classifier.")
    p.add_argument("--checkpoint", required=True, help="Checkpoint dir or .pt file")
    p.add_argument("--data",       required=True, help="CSV with images + multi-hot labels")
    p.add_argument("--label-map",  required=True, help="JSON label map with 'classes' list")
    p.add_argument("--out",        default="reports/evaluation", help="Output directory")
    p.add_argument("--image-size", type=int, default=320)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device",     default="auto")
    p.add_argument("--image-col",  default="image")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_evaluation(
        checkpoint_dir=args.checkpoint,
        data_csv=args.data,
        label_map_json=args.label_map,
        out_dir=args.out,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device_str=args.device,
        image_col=args.image_col,
    )
