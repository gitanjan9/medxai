"""Diagnostic script: label map, output shape, pred distribution, confusion matrix, top-k."""
import sys
sys.path.insert(0, ".")

import torch
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.metrics import confusion_matrix

from src.common.config import TrainConfig
from src.common.schemas import LabelMap
from src.common.utils import get_device
from src.train.dataset import CXRDataset, _filter_known_labels, _read_csv
from src.train.model_factory import build_model
from src.train.transforms import build_val_transforms
from src.train.evaluate import _resolve_checkpoint, _load_checkpoint
from torch.utils.data import DataLoader

cfg = TrainConfig.from_yaml("configs/train.yaml")
label_map = LabelMap.load(Path("artifacts/label_map.json"))
device = get_device()

# ── 1. Label mapping ──────────────────────────────────────────────────────────
from src.train.dataset import apply_class_merge
_merge = cfg.data.class_merge_map_path

def _load(path):
    df = _read_csv(path)
    if _merge and Path(_merge).exists():
        df = apply_class_merge(df, "impression", _merge)
    return _filter_known_labels(df, "impression", label_map)

train_df = _load(cfg.data.train_path)
val_df   = _load(cfg.data.val_path)
train_targets = [label_map.encode(l) for l in train_df["impression"]]
val_targets   = [label_map.encode(l) for l in val_df["impression"]]

print("=== 1. Label mapping ===")
print(f"  num_classes  : {label_map.num_classes}")
print(f"  train range  : {min(train_targets)} – {max(train_targets)}")
print(f"  val range    : {min(val_targets)} – {max(val_targets)}")
print(f"  Shared map   : YES (artifacts/label_map.json)")

# ── 2. Model output shape ─────────────────────────────────────────────────────
model = build_model(cfg.model).to(device)
with torch.no_grad():
    out = model(torch.randn(2, 1, 224, 224).to(device))
print(f"\n=== 2. Output shape ===")
print(f"  out.shape = {tuple(out.shape)}  (must be [2, 16])")

# ── 3. Loss ───────────────────────────────────────────────────────────────────
print(f"\n=== 3. Loss ===")
print(f"  WeightedCrossEntropyLoss over raw logits — correct")
print(f"  class_balance_strategy = {cfg.training.class_balance_strategy}")

# ── 4. Prediction distribution ────────────────────────────────────────────────
ckpt = _resolve_checkpoint(None, cfg)
_load_checkpoint(model, ckpt, device)

val_ds = CXRDataset(
    df=val_df.reset_index(drop=True),
    label_map=label_map,
    image_col=cfg.data.image_col,
    label_col=cfg.data.label_col,
    text_col=cfg.data.text_col,
    transforms=build_val_transforms(cfg.data.image_size),
)
loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

all_preds, all_true, logits_list = [], [], []
model.eval()
with torch.no_grad():
    for batch in loader:
        logits = model(batch["image"].to(device)).cpu()
        all_preds.extend(logits.argmax(dim=1).tolist())
        all_true.extend(batch["label"].tolist())
        logits_list.append(logits)

all_logits = torch.cat(logits_list)
pred_counts = Counter(all_preds)
true_counts = Counter(all_true)

print(f"\n=== 4. Prediction vs True distribution ===")
print(f"  {'Cls':<4} {'True':>5} {'Pred':>5}  Name")
print(f"  {'-'*75}")
for i in range(label_map.num_classes):
    flag = " <-- NEVER PREDICTED" if pred_counts.get(i, 0) == 0 else ""
    print(f"  {i:<4} {true_counts.get(i,0):>5} {pred_counts.get(i,0):>5}  {label_map.decode(i)[:45]}{flag}")

# ── 5. Per-class accuracy (confusion matrix diagonal) ────────────────────────
cm = confusion_matrix(all_true, all_preds, labels=list(range(16)))
print(f"\n=== 5. Per-class accuracy ===")
print(f"  {'Cls':<4} {'Corr/Tot':>9} {'Acc%':>6}  Name")
print(f"  {'-'*70}")
for i in range(16):
    t = true_counts.get(i, 0)
    c = int(cm[i, i]) if t > 0 else 0
    pct = 100 * c / t if t > 0 else 0.0
    print(f"  {i:<4} {c:>3}/{t:<4}  {pct:>5.1f}%  {label_map.decode(i)[:45]}")

# ── 6. Top-k accuracy ─────────────────────────────────────────────────────────
tgt = torch.tensor(all_true)
top1 = (all_logits.argmax(1) == tgt).float().mean().item()
top3 = all_logits.topk(3, 1).indices.eq(tgt.view(-1, 1)).any(1).float().mean().item()
top5 = all_logits.topk(5, 1).indices.eq(tgt.view(-1, 1)).any(1).float().mean().item()

print(f"\n=== 6. Top-k accuracy ===")
print(f"  top-1 : {top1:.3f}")
print(f"  top-3 : {top3:.3f}")
print(f"  top-5 : {top5:.3f}")

if top3 > top1 * 1.5:
    print("  → Signal present but class boundaries weak (classes too similar)")
elif len([v for v in pred_counts.values() if v == 0]) > 8:
    print("  → Model collapsed: predicting <8 classes; imbalance is dominant cause")
else:
    print("  → Top-k gap small: model is mostly underfitting")
