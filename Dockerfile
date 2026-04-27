# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements/train.txt requirements/train.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements/train.txt

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="MedicalXAI Trainer"
LABEL org.opencontainers.image.version="0.1.0"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project source
COPY src/       src/
COPY configs/   configs/
COPY bundles/   bundles/
COPY pyproject.toml .

RUN pip install --no-cache-dir -e . --no-deps

# Default: run training
CMD ["python", "-m", "src.train.train", "--config", "configs/train.yaml"]
