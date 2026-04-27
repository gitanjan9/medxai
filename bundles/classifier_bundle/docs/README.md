# mimic_cxr_classifier Bundle — v0.2.0

## Overview

MONAI Bundle-compatible chest X-ray classifier trained on MIMIC-CXR with
post-hoc calibration, per-class threshold tuning, and Grad-CAM explainability.

| Property | Value |
|---|---|
| Task | Multi-class classification |
| Input | Grayscale CXR, 1×224×224, float32 [0,1] |
| Output | Calibrated softmax probabilities, 16 classes |
| Architecture | DenseNet-121 (MONAI) |
| Calibration | Temperature scaling |
| Thresholds | Per-class PPV/recall-balanced low/high bands |
| Explainability | Grad-CAM (Captum / pure-PyTorch fallback) |
| Dataset | MIMIC-CXR (PhysioNet) |

## Bundle structure

```
classifier_bundle/
├── configs/
│   ├── metadata.json        # MONAI Bundle schema metadata
│   ├── train.yaml           # Bundle-native training config
│   ├── inference.yaml       # Inference + explainability config (Day 2)
│   ├── label_map.json       # 16-class label map  ← exported by artifacts.py
│   ├── calibration.json     # Temperature T        ← exported by artifacts.py
│   └── thresholds.json      # Per-class low/high   ← exported by artifacts.py
├── models/
│   └── best_checkpoint.pt   # Best val-AUROC checkpoint ← exported by artifacts.py
└── docs/
    └── README.md
```

## Day 2 CLI workflow

```bash
# 1. Train (Day 1)
.venv/bin/python -m src.train.train --config configs/train.yaml

# 2. Evaluate best checkpoint
.venv/bin/python -m src.train.evaluate --config configs/train.yaml

# 3. Calibrate (temperature scaling on val split)
.venv/bin/python -m src.train.calibrate --config configs/train.yaml

# 4. Tune per-class thresholds
.venv/bin/python -m src.train.thresholds --config configs/train.yaml

# 5. Generate Grad-CAM explanations
.venv/bin/python -m src.train.explainability --config configs/inference.yaml

# 6. Export all artifacts into bundle
.venv/bin/python -m src.train.artifacts --config configs/inference.yaml
```

## Artifact schema

### `calibration.json`
```json
{
  "method": "temperature_scaling",
  "temperature": 1.23,
  "num_samples": 292,
  "num_classes": 16
}
```

### `thresholds.json`
```json
{
  "method": "ppv_recall_balanced",
  "target_ppv": 0.65,
  "target_recall": 0.70,
  "thresholds": [
    { "class_idx": 0, "class_name": "...", "low": 0.28, "high": 0.62,
      "ppv_at_high": 0.71, "recall_at_low": 0.73 },
    ...
  ]
}
```

### Decision bands
| Band | Condition | Action |
|---|---|---|
| **positive** | `prob ≥ high` | Report finding |
| **review** | `low ≤ prob < high` | Send to radiologist |
| **negative** | `prob < low` | Clear |

## Loading bundle artifacts in Day 3

```python
import json
from pathlib import Path
from src.common.schemas import LabelMap
from src.train.calibrate import apply_temperature
from src.train.thresholds import load_thresholds, apply_thresholds_batch

bundle_root = Path("bundles/classifier_bundle")
label_map   = LabelMap.load(bundle_root / "configs/label_map.json")
cal         = json.loads((bundle_root / "configs/calibration.json").read_text())
th_data     = load_thresholds(bundle_root / "configs/thresholds.json")

# At inference time:
probs     = apply_temperature(logits, cal["temperature"])
decisions = apply_thresholds_batch(probs, th_data["thresholds"])
```

## Day 3 roadmap

- ONNX export + TorchScript tracing
- FastAPI inference endpoint
- Triton model repository layout
- Batch inference pipeline
