---
title: MedXAI API
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# MedicalXAI – Day 1: Training Pipeline

Production-ready chest X-ray classification system built on **MONAI Bundle-compatible** design.

| | |
|---|---|
| **Task** | 17-class CXR impression classification |
| **Dataset** | MIMIC-CXR (PhysioNet) |
| **Model** | DenseNet-121 (MONAI) / EfficientNet-B0 (timm) |
| **Tracking** | MLflow |
| **Day** | 1 – Training + Evaluation pipeline |

---

## Architecture summary

```
Input (grayscale CXR, 1×224×224)
        ↓
  MONAI Transforms (augmentation / normalisation)
        ↓
  DenseNet-121 backbone  (pretrained, adapted for 1-channel)
        ↓
  Dropout + Linear head (17-class)
        ↓
  Weighted Cross-Entropy / Focal Loss
        ↓
  AdamW + CosineAnnealingWarmRestarts
        ↓
  MLflow tracking + MONAI Bundle-compatible artefacts
```

---

## Repository structure

```
medicalxai/
├── README.md
├── pyproject.toml
├── .env.example
├── Makefile
├── Dockerfile
├── requirements/
│   ├── base.txt
│   ├── train.txt
│   └── dev.txt
├── configs/
│   ├── metadata.json        # MONAI Bundle metadata
│   ├── train.yaml           # Main training config
│   ├── model.yaml           # Per-architecture presets
│   └── logging.yaml         # Python logging config
├── bundles/
│   └── classifier_bundle/
│       ├── configs/
│       │   ├── metadata.json
│       │   └── train.yaml   # Bundle-native config stub
│       ├── models/          # Exported .pt lives here
│       └── docs/
│           └── README.md
├── scripts/
│   ├── train_local.sh
│   ├── eval_local.sh
│   └── verify_bundle.sh
├── src/
│   ├── common/
│   │   ├── config.py        # Pydantic v2 config models
│   │   ├── logging.py       # Structured logging setup
│   │   ├── utils.py         # Seeds, device, paths, timers
│   │   ├── schemas.py       # LabelMap + DataSample schemas
│   │   └── exceptions.py    # Custom exception hierarchy
│   └── train/
│       ├── dataset.py       # CXRDataset (bytes + paths)
│       ├── transforms.py    # MONAI train / val pipelines
│       ├── model_factory.py # DenseNet121 / EfficientNet factory
│       ├── losses.py        # WeightedCE + FocalLoss
│       ├── metrics.py       # AUROC, AUPRC, F1, specificity, CM
│       ├── mlflow_utils.py  # MLflow tracking + logging helpers
│       ├── bundle_utils.py  # Bundle structure validator
│       ├── zoo_bootstrap.py # MONAI Zoo lookup (extension point)
│       ├── train.py         # Main training loop (CLI)
│       └── evaluate.py      # Evaluation loop (CLI)
└── tests/
    └── unit/
        ├── test_dataset.py
        ├── test_transforms.py
        └── test_metrics.py
```

---

## Quick start

### 1. Clone and install

```bash
cd medicalxai
cp .env.example .env           # fill in your CSV paths
make install                   # or: pip install -r requirements/train.txt && pip install -e .
```

### 2. Configure paths

Edit **`.env`** (or `configs/train.yaml`) to point at your data:

```env
MEDAI_TRAIN_CSV=/path/to/mimic_cxr_processed.csv
```

Or edit `configs/train.yaml` directly:

```yaml
data:
  train_path: /path/to/mimic_cxr_processed.csv
  val_path:   null      # null = auto 15% split
```

The dataset supports **two image sources** (auto-detected per row):
- `image` column → Python bytes literal (e.g. from `mimic_cxr_processed.csv`)
- `image_path` column → file-system path to a JPEG/PNG

### 3. Train

```bash
# via Makefile
make train

# or directly
python -m src.train.train --config configs/train.yaml
```

### 4. Evaluate

```bash
# latest checkpoint, val split
make eval

# explicit checkpoint + test CSV
python -m src.train.evaluate \
    --config configs/train.yaml \
    --checkpoint artifacts/checkpoints/best.pt \
    --data-csv /path/to/test.csv \
    --split-name test
```

### 5. MLflow UI

```bash
make mlflow-ui          # opens at http://localhost:5000
```

### 6. Verify bundle structure

```bash
make verify-bundle
```

---

## Key configuration knobs (`configs/train.yaml`)

| Key | Default | Notes |
|---|---|---|
| `model.architecture` | `densenet121` | `efficientnet_b0` \| `resnet50` also supported |
| `model.pretrained` | `true` | Adapts first conv for 1-channel input |
| `training.class_balance_strategy` | `weighted_loss` | `focal` \| `weighted_sampler` |
| `training.use_amp` | `true` | FP16 on CUDA only |
| `early_stopping.patience` | `7` | Monitors `val_auroc_macro` |
| `training.batch_size` | `32` | Reduce if OOM |

---

## Docker

```bash
make docker-build
make docker-train   # needs MEDAI_TRAIN_CSV env var set
```

---

## Tests

```bash
make test           # with coverage
make test-fast      # fast, no coverage
```

---

## Day 2 roadmap

- [ ] Calibration (temperature scaling, Platt)
- [ ] Per-class threshold optimisation
- [ ] Grad-CAM / SHAP visualisation
- [ ] ONNX export via `monai.bundle`
- [ ] REST inference API (FastAPI)
- [ ] Triton Inference Server config

---

## Dataset notes

The `mimic_cxr_processed.csv` dataset contains **15,317 chest X-ray records** with:
- `image` – JPEG image data as a Python bytes literal string
- `findings` – free-text radiology findings
- `impression` – 17-class diagnostic impression label

`mimic_cxr_train.csv` / `mimic_cxr_val.csv` contain text-only splits
(`findings` + `impression`) which can be used with an upcoming NLP/multimodal
pipeline (Day 3+).

---

## Production Architecture

```
medicalxai/
├── frontend/                   # React + Vite SPA
│   └── src/
│       ├── components/         # UI components (ImageViewer, ChatPanel, etc.)
│       ├── context/
│       │   └── AuthContext.tsx # JWT session state
│       └── services/
│           └── api.ts          # Centralised fetch layer (auto token refresh)
├── src/serve/                  # FastAPI backend
│   ├── auth/
│   │   ├── jwt_handler.py      # Token creation / HttpOnly cookies
│   │   ├── password.py         # bcrypt hashing
│   │   ├── dependencies.py     # FastAPI deps (AuthRequired, AdminRequired)
│   │   └── user_store.py       # Postgres user / session CRUD
│   ├── routers/
│   │   ├── auth.py             # /v1/auth/* (login, logout, refresh, me)
│   │   ├── predict.py          # /v1/predict
│   │   ├── explain.py          # /v1/explain
│   │   └── chat.py             # /v1/chat
│   └── services/
│       ├── txrv_primary_adapter.py
│       ├── explanation_engine.py
│       └── chat_service.py
└── deployment/
    ├── Dockerfile.api           # FastAPI + Gunicorn production image
    ├── Dockerfile.frontend      # React → nginx multi-stage image
    ├── docker-compose.yml       # frontend + api + postgres + redis
    ├── nginx.conf               # Reverse proxy + security headers
    ├── gunicorn.conf.py         # Worker / timeout / preload config
    └── init.sql                 # DB schema bootstrap
```

### Security model

| Layer | Implementation |
|---|---|
| Auth tokens | JWT (HS256) in **HttpOnly cookies** — never in JS |
| Password storage | bcrypt via `passlib` |
| Session rotation | Refresh token rotated on each `/auth/refresh` call |
| Rate limiting | `slowapi` (pluggable; add `@limiter.limit("5/minute")` to any route) |
| Secure headers | nginx: `X-Frame-Options`, `CSP`, `X-Content-Type-Options` |
| File uploads | MIME check + 50 MB cap |
| RBAC | `user` / `clinician` / `admin` roles checked via FastAPI `Depends` |

---

## Local Development

### Prerequisites
- Python 3.10+, Node 20+, Docker Desktop

### Backend

```bash
# 1. Create virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies (includes new auth packages)
pip install -e ".[dev]"

# 3. Set environment (minimum required)
export DATABASE_URL=postgresql://medxai:medxai123@localhost:5432/medxai
export JWT_SECRET_KEY=$(openssl rand -hex 32)
export COOKIE_SECURE=false   # http in local dev

# 4. Start (no GPU needed — MPS / CPU auto-detected)
uvicorn src.serve.app:app --reload --port 8000
```

> **Dev mode** (no Postgres): when `DATABASE_URL` is unset the backend uses a  
> hardcoded dev account — `admin@medicalxai.local` / `admin123`.  
> Never expose this in production.

### Frontend

```bash
cd frontend
npm install
npm run dev        # http://localhost:4000 (proxies /v1/* → :8000)
```

---

## Docker (Full Stack)

```bash
# 1. Copy and fill in secrets
cp .env.example .env
# Edit .env: set JWT_SECRET_KEY, POSTGRES_PASSWORD

# 2. Build and start all services
docker compose -f deployment/docker-compose.yml up --build -d

# Services:
#   http://localhost      → React frontend (nginx)
#   http://localhost:8000 → FastAPI backend
#   postgres:5432         → Postgres (internal)
#   redis:6379            → Redis (internal)
```

---

## Render.com Deployment

```bash
# 1. Push to GitHub
git push origin main

# 2. Go to Render Dashboard → New → Blueprint
# 3. Connect your repo — Render auto-detects render.yaml
# 4. Set secret env vars in dashboard:
#      JWT_SECRET_KEY  → openssl rand -hex 32
#      POSTGRES_PASSWORD → strong random password

# Render will create:
#   - medicalxai-api     (Docker web service)
#   - medicalxai-frontend (Static site)
#   - medicalxai-postgres (Managed Postgres)
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `JWT_SECRET_KEY` | **Yes** | `openssl rand -hex 32` |
| `DATABASE_URL` | Yes (prod) | Postgres connection string |
| `REDIS_URL` | No | Rate limiting / session cache |
| `POSTGRES_PASSWORD` | Yes (Docker) | DB password |
| `COOKIE_SECURE` | No | `false` for local http dev |
| `MEDXAI_OPEN_REGISTRATION` | No | `true` = public signup |
| `OPENAI_API_KEY` | No | Chat GPT fallback |
| `WEB_CONCURRENCY` | No | Gunicorn workers (default 2) |

See `.env.example` for the full list.
