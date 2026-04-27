"""Active learning pipeline: pull clinician corrections from DB → fine-tune txrv.

Every time a doctor clicks "Wrong → <correct label>", the case lands in the
retraining_queue table.  This script:

  1. Pulls all queued cases (prediction_id, image_hash, true_label)
  2. Loads the original images from artifacts/images/<hash>.jpg
  3. Builds a fine-tuning dataset (txrv multi-label format)
  4. Fine-tunes the last two dense blocks of txrv DenseNet-121
  5. Saves the updated weights to artifacts/txrv_finetuned.pt
  6. Marks processed queue rows as done

Usage
-----
    # Run after accumulating corrections
    .venv/bin/python -m src.train.active_learning

    # Dry-run — show what would be trained, do nothing
    .venv/bin/python -m src.train.active_learning --dry-run

    # Fine-tune on at least 10 corrections (default 5)
    .venv/bin/python -m src.train.active_learning --min-samples 10

Requirements
------------
    pip install torchxrayvision opencv-python-headless
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.common.logging import get_logger
from src.serve.services.database import init_db
from src.serve.services.prediction_store import get_conn, is_available

logger = get_logger("active_learning")

# ── txrv class list (must match model.pathologies order) ─────────────────────
TXRV_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural Thickening", "Pneumonia", "Pneumothorax",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "Pleural Effusion", "Pleural Other", "Support Devices", "No Finding",
]

# Map v3 / doctor labels → txrv class names (case-insensitive fuzzy match)
_LABEL_ALIASES: dict[str, str] = {
    "pneumothorax":  "Pneumothorax",
    "ptx":           "Pneumothorax",
    "pneumonia":     "Pneumonia",
    "consolidation": "Consolidation",
    "effusion":      "Effusion",
    "pleural effusion": "Pleural Effusion",
    "atelectasis":   "Atelectasis",
    "cardiomegaly":  "Cardiomegaly",
    "edema":         "Edema",
    "emphysema":     "Emphysema",
    "fibrosis":      "Fibrosis",
    "mass":          "Mass",
    "nodule":        "Nodule",
    "infiltration":  "Infiltration",
    "no finding":    "No Finding",
    "normal":        "No Finding",
    "lung opacity":  "Lung Opacity",
    "fracture":      "Fracture",
}


def _normalise_label(label: str) -> Optional[str]:
    key = label.strip().lower()
    return _LABEL_ALIASES.get(key) or next(
        (c for c in TXRV_CLASSES if c.lower() == key), None
    )


# ── DB helpers ────────────────────────────────────────────────────────────────

def fetch_queue() -> list[dict]:
    """Return all unprocessed retraining_queue rows."""
    if not is_available():
        logger.error("DB not available")
        return []
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT rq.id, rq.prediction_id, rq.image_hash, rq.true_label
                    FROM   retraining_queue rq
                    WHERE  rq.used_in_training = FALSE
                    ORDER  BY rq.created_at
                    """
                )
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as exc:
        logger.error("fetch_queue failed: %s", exc)
        return []


def fetch_image_bytes(image_hash: str) -> Optional[bytes]:
    """Read image from artifacts/images/<hash>.jpg."""
    for ext in (".jpg", ".jpeg", ".png"):
        p = Path("artifacts/images") / f"{image_hash}{ext}"
        if p.exists():
            return p.read_bytes()
    return None


def mark_processed(queue_ids: list[str]) -> None:
    if not queue_ids or not is_available():
        return
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE retraining_queue SET used_in_training=TRUE "
                    "WHERE id = ANY(%s::uuid[])",
                    (queue_ids,),
                )
        logger.info("Marked %d queue rows as used_in_training", len(queue_ids))
    except Exception as exc:
        logger.error("mark_processed failed: %s", exc)


# ── Dataset ───────────────────────────────────────────────────────────────────

class CorrectionDataset(Dataset):
    def __init__(self, samples: list[dict], class_to_idx: dict[str, int]) -> None:
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.n_classes = len(class_to_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        from PIL import Image
        import cv2
        row = self.samples[idx]
        # Load image
        img_bytes = row["image_bytes"]
        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        arr = np.array(img, dtype=np.uint8)
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        arr = clahe.apply(arr).astype(np.float32) / 255.0
        # Resize to 224×224 (txrv native)
        import cv2 as _cv2
        arr = _cv2.resize(arr, (224, 224))
        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, 224, 224)

        # One-hot label
        label = torch.zeros(self.n_classes)
        cls_name = _normalise_label(row["true_label"] or "")
        if cls_name and cls_name in self.class_to_idx:
            label[self.class_to_idx[cls_name]] = 1.0

        return tensor, label


# ── Fine-tuning ───────────────────────────────────────────────────────────────

def finetune(
    samples: list[dict],
    out_path: Path,
    epochs: int = 5,
    lr: float = 1e-4,
    batch_size: int = 8,
) -> None:
    try:
        import torchxrayvision as xrv
    except ImportError:
        sys.exit("torchxrayvision not installed: pip install torchxrayvision")

    model = xrv.models.DenseNet(weights="densenet121-res224-all")

    # Load previously fine-tuned weights if they exist
    if out_path.exists():
        state = torch.load(out_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        logger.info("Loaded existing fine-tuned weights from %s", out_path)

    device = torch.device(
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()          else "cpu"
    )
    model = model.to(device)

    # Freeze all layers except the last dense block + classifier
    for name, param in model.named_parameters():
        param.requires_grad = "denseblock4" in name or "norm5" in name or "classifier" in name

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable params: %d / %d", trainable,
                sum(p.numel() for p in model.parameters()))

    class_to_idx = {c: i for i, c in enumerate(model.pathologies)}
    dataset = CorrectionDataset(samples, class_to_idx)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / max(len(loader), 1)
        logger.info("Epoch %d/%d — loss=%.4f", epoch, epochs, avg)
        print(f"  Epoch {epoch}/{epochs}  loss={avg:.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    logger.info("Fine-tuned weights saved → %s", out_path)
    print(f"\n✓ Fine-tuned weights saved → {out_path}")
    print("  Restart the server to load them.")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(min_samples: int = 5, dry_run: bool = False) -> None:
    init_db()
    rows = fetch_queue()
    if not rows:
        print("No pending corrections in retraining_queue.")
        return

    print(f"Found {len(rows)} correction(s) in queue:")
    valid: list[dict] = []
    for r in rows:
        label = _normalise_label(r.get("true_label") or "")
        img_bytes = fetch_image_bytes(r["image_hash"])
        status = "✓" if (label and img_bytes) else "✗"
        reason = "" if label else f"unmapped label '{r['true_label']}'" if not label else "image missing"
        print(f"  {status} {r['true_label']!r:20s} → {label or 'SKIP ' + reason}")
        if label and img_bytes:
            valid.append({**r, "true_label": label, "image_bytes": img_bytes})

    print(f"\n{len(valid)} usable sample(s) (need ≥ {min_samples})")

    if dry_run:
        print("Dry-run — no changes made.")
        return

    if len(valid) < min_samples:
        print(f"Not enough samples yet. Collect at least {min_samples} corrections and re-run.")
        return

    out_path = Path("artifacts/txrv_finetuned.pt")
    print(f"\nFine-tuning txrv on {len(valid)} samples → {out_path}")
    finetune(valid, out_path)

    mark_processed([str(r["id"]) for r in valid])


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Active learning: fine-tune txrv on clinician corrections")
    p.add_argument("--min-samples", type=int, default=5,
                   help="Minimum corrections needed before fine-tuning starts (default: 5)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would happen without actually fine-tuning")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    run(min_samples=args.min_samples, dry_run=args.dry_run)
