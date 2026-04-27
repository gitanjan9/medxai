"""Per-class threshold calibration for the torchxrayvision DenseNet-121.

Runs the txrv model over a labelled CSV, computes precision/recall/F1 at
every candidate threshold, then writes optimal per-class thresholds to
src/config/txrv_thresholds.py so they are picked up at the next server start.

Usage
-----
    # Calibrate on v3 labels (recommended)
    .venv/bin/python -m src.train.calibrate_txrv \
        --csv artifacts/v3_disease_labels.csv \
        --out src/config/txrv_thresholds.py

    # Calibrate on a separate held-out val split
    .venv/bin/python -m src.train.calibrate_txrv \
        --csv artifacts/val_split.csv \
        --out src/config/txrv_thresholds.py \
        --metric f1            # or 'ppv' for high-precision mode

Label CSV format
----------------
    The CSV must have one column per txrv pathology class (0/1 binary) OR
    a single column named 'label' / 'finding' with the class name as a string.
    The --label-col flag selects the label column when using string labels.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── txrv class list (same order as model output) ─────────────────────────────
TXRV_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural Thickening", "Pneumonia", "Pneumothorax",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "Pleural Effusion", "Pleural Other", "Support Devices", "No Finding",
]

# These thresholds should never be raised above these safety ceilings
# (critical findings must remain sensitive)
_SAFETY_CEILING: dict[str, float] = {
    "Pneumothorax": 0.55,
    "Pneumonia":    0.65,
    "Edema":        0.65,
    "Effusion":     0.65,
    "Pleural Effusion": 0.65,
}
_SAFETY_FLOOR: dict[str, float] = {
    "Pneumothorax": 0.35,   # never go below this even if data says so
    "Pneumonia":    0.40,
}

# Current defaults (fallback when no data exists for a class)
_CURRENT_THRESHOLDS: dict[str, float] = {
    "Atelectasis":        0.70,
    "Cardiomegaly":       0.70,
    "Consolidation":      0.72,
    "Edema":              0.72,
    "Effusion":           0.70,
    "Pleural Effusion":   0.70,
    "Emphysema":          0.63,
    "Fibrosis":           0.63,
    "Hernia":             0.75,
    "Infiltration":       0.70,
    "Mass":               0.70,
    "Nodule":             0.70,
    "Pleural Thickening": 0.65,
    "Pneumonia":          0.68,
    "Pneumothorax":       0.50,
    "Enlarged Cardiomediastinum": 0.75,
    "Fracture":           0.75,
    "Lung Lesion":        0.75,
    "Lung Opacity":       0.68,
    "Pleural Other":      0.75,
    "Support Devices":    0.80,
}


# ── Inference ─────────────────────────────────────────────────────────────────

def _load_image_bytes(item) -> bytes:
    """Accept a file path OR an inline bytes/byte-string-literal from the CSV."""
    import io as _io
    # Already bytes
    if isinstance(item, (bytes, bytearray)):
        return bytes(item)
    s = str(item)
    # Python bytes literal: b'\xff\xd8...'
    if s.startswith("b'") or s.startswith('b"'):
        try:
            return ast.literal_eval(s)
        except Exception:
            pass
    # File path
    p = Path(s)
    if p.exists():
        return p.read_bytes()
    raise ValueError(f"Cannot load image from: {s[:60]!r}")


def run_txrv_inference(image_items: list) -> np.ndarray:
    """Run txrv on every image (path or inline bytes) → (N, C) score matrix."""
    try:
        import torchxrayvision as xrv
        import torch
        import skimage.transform
        from PIL import Image
        import cv2
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}")

    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    all_scores: list[np.ndarray] = []
    n = len(image_items)
    for i, item in enumerate(image_items):
        if i % 200 == 0:
            print(f"  Inference {i}/{n} …", end="\r", flush=True)
        try:
            img_bytes = _load_image_bytes(item)
            from PIL import Image as _PIL
            img = _PIL.open(__import__("io").BytesIO(img_bytes)).convert("L")
            arr = np.array(img, dtype=np.uint8)
            arr = clahe.apply(arr).astype(np.float32)
            # txrv expects [-1024, 1024]
            arr = (arr / 255.0) * 2048 - 1024
            arr = skimage.transform.resize(arr, (224, 224), anti_aliasing=True)
            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.no_grad():
                out = torch.sigmoid(model(tensor)).cpu().numpy()[0]
            all_scores.append(out)
        except Exception as exc:
            print(f"\nWARN row {i}: {exc}")
            all_scores.append(np.zeros(len(model.pathologies)))
    print()
    return np.array(all_scores)   # (N, C)


# ── Threshold search ──────────────────────────────────────────────────────────

def best_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    candidates: np.ndarray,
    metric: str = "f1",
) -> tuple[float, float, dict]:
    """Return (best_threshold, best_metric_value, stats_dict)."""
    best_thr = 0.5
    best_val = 0.0
    best_stats: dict = {}

    for thr in candidates:
        preds = (scores >= thr).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())

        ppv   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1    = 2 * ppv * recall / (ppv + recall) if (ppv + recall) > 0 else 0.0

        val = {"f1": f1, "ppv": ppv, "recall": recall}.get(metric, f1)
        if val > best_val:
            best_val = val
            best_thr = float(thr)
            best_stats = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                          "ppv": ppv, "recall": recall, "f1": f1}

    return best_thr, best_val, best_stats


# ── Output writer ─────────────────────────────────────────────────────────────

def write_thresholds(thresholds: dict[str, float], out_path: Path) -> None:
    """Overwrite the TXRV_THRESHOLDS dict in txrv_thresholds.py."""
    content = out_path.read_text()

    # Build new dict literal
    lines = ["TXRV_THRESHOLDS: dict[str, float] = {\n"]
    for cls, thr in sorted(thresholds.items()):
        safety = ""
        if cls in _SAFETY_CEILING and thr >= _SAFETY_CEILING[cls]:
            safety = f"  # clamped at safety ceiling {_SAFETY_CEILING[cls]}"
        lines.append(f'    "{cls}":{" " * max(1, 30 - len(cls))}{thr:.4f},{safety}\n')
    lines.append("}\n")

    new_dict = "".join(lines)

    # Replace the existing dict
    pattern = r"TXRV_THRESHOLDS: dict\[str, float\] = \{[^}]+\}"
    import re as _re
    new_content = _re.sub(pattern, new_dict.rstrip(), content, flags=_re.DOTALL)

    out_path.write_text(new_content)
    print(f"✓ Thresholds written to {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def calibrate_txrv(
    csv_path: Path,
    out_path: Path,
    image_dir: Path | None,
    image_col: str,
    metric: str,
    min_positives: int,
) -> dict[str, float]:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Resolve image items (file path or inline bytes literal)
    if image_dir:
        paths = [image_dir / str(r[image_col]) for _, r in df.iterrows()]
    else:
        paths = list(df[image_col])

    # Check which classes exist as binary columns in the CSV
    available = [c for c in TXRV_CLASSES if c in df.columns]
    if not available:
        sys.exit(
            "No txrv class columns found in CSV. "
            "Columns must be named exactly as txrv class names (e.g. 'Pneumothorax')."
        )
    print(f"Found {len(available)} label columns: {available}")

    # Run inference
    print("Running txrv inference …")
    scores_matrix = run_txrv_inference(paths)  # (N, all_txrv_classes)

    # Build class→index map from model
    try:
        import torchxrayvision as xrv
        model_classes = xrv.models.DenseNet(weights="densenet121-res224-all").pathologies
    except Exception:
        model_classes = TXRV_CLASSES

    cls_to_idx = {c: i for i, c in enumerate(model_classes)}

    candidates = np.linspace(0.20, 0.90, 141)
    new_thresholds = dict(_CURRENT_THRESHOLDS)  # start from current

    print(f"\n{'Class':<30} {'N+':>5} {'Old thr':>8} {'New thr':>8} "
          f"{'F1':>6} {'PPV':>6} {'Recall':>6}")
    print("-" * 80)

    for cls in available:
        idx = cls_to_idx.get(cls)
        if idx is None or idx >= scores_matrix.shape[1]:
            continue
        labels = df[cls].fillna(0).astype(int).values
        n_pos = int(labels.sum())
        if n_pos < min_positives:
            print(f"  SKIP {cls}: only {n_pos} positives (< {min_positives})")
            continue

        cls_scores = scores_matrix[:, idx]
        thr, val, stats = best_threshold(cls_scores, labels, candidates, metric)

        # Apply safety constraints
        floor = _SAFETY_FLOOR.get(cls, 0.25)
        ceiling = _SAFETY_CEILING.get(cls, 0.90)
        thr = max(floor, min(ceiling, thr))

        old = _CURRENT_THRESHOLDS.get(cls, 0.75)
        new_thresholds[cls] = round(thr, 4)

        print(f"  {cls:<28} {n_pos:>5} {old:>8.3f} {thr:>8.3f} "
              f"{stats.get('f1', 0):>6.3f} {stats.get('ppv', 0):>6.3f} "
              f"{stats.get('recall', 0):>6.3f}")

    write_thresholds(new_thresholds, out_path)
    return new_thresholds


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate txrv per-class thresholds")
    p.add_argument("--csv",   required=True, help="Labelled CSV path")
    p.add_argument("--out",   default="src/config/txrv_thresholds.py",
                   help="Output Python file to update (default: src/config/txrv_thresholds.py)")
    p.add_argument("--image-dir", default=None,
                   help="Directory containing images (if image_col is a relative filename)")
    p.add_argument("--image-col", default="image",
                   help="CSV column with image path/bytes (default: image)")
    p.add_argument("--metric", default="f1", choices=["f1", "ppv", "recall"],
                   help="Optimisation metric per class (default: f1)")
    p.add_argument("--min-positives", type=int, default=20,
                   help="Skip classes with fewer positives than this (default: 20)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    calibrate_txrv(
        csv_path=Path(args.csv),
        out_path=Path(args.out),
        image_dir=Path(args.image_dir) if args.image_dir else None,
        image_col=args.image_col,
        metric=args.metric,
        min_positives=args.min_positives,
    )
