.PHONY: help install install-dev train eval test lint fmt clean docker-build docker-train mlflow-ui serve serve-prod

PYTHON  := python3
CONFIG  := configs/train.yaml

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ── Installation ────────────────────────────────────────────────────────────

install:  ## Install all dependencies from pyproject.toml
	pip install --upgrade pip
	pip install -e .

install-dev:  ## Install with dev extras
	pip install --upgrade pip
	pip install -e '.[dev]'

# ── Training & Evaluation ────────────────────────────────────────────────────

train:  ## Run training (python -m src.train.train)
	$(PYTHON) -m src.train.train --config $(CONFIG)

eval:  ## Evaluate latest checkpoint
	$(PYTHON) -m src.train.evaluate --config $(CONFIG)

eval-test:  ## Evaluate on explicit test CSV
	$(PYTHON) -m src.train.evaluate --config $(CONFIG) \
	  --data-csv $(TEST_CSV) --split-name test

# ── Tests ────────────────────────────────────────────────────────────────────

test:  ## Run unit tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-fast:  ## Run tests without coverage (fast)
	pytest tests/ -v -x

# ── Code Quality ─────────────────────────────────────────────────────────────

lint:  ## Lint with ruff
	ruff check src/ tests/

fmt:  ## Format with black + isort
	black src/ tests/
	isort src/ tests/

typecheck:  ## Run mypy type checker
	mypy src/ --ignore-missing-imports

# ── Serve ────────────────────────────────────────────────────────────────────

serve:  ## Start the FastAPI inference API (loads .env automatically)
	@export $$(grep -v '^\#' .env | grep -v '^$$' | xargs) && \
	.venv/bin/uvicorn src.serve.app:app --host 0.0.0.0 --port 8000 --reload

serve-prod:  ## Start with Gunicorn 4 workers (forces CPU on macOS to avoid MPS fork crash)
	@export $$(grep -v '^\#' .env | grep -v '^$$' | xargs) && \
	OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES \
	MEDXAI_DEVICE=cpu \
	.venv/bin/gunicorn src.serve.app:app \
	  -k uvicorn.workers.UvicornWorker \
	  --workers 4 --bind 0.0.0.0:8000 \
	  --timeout 120 --access-logfile -

# ── MLflow ───────────────────────────────────────────────────────────────────

mlflow-ui:  ## Launch MLflow tracking UI
	mlflow ui --backend-store-uri mlruns --port 5000

# ── Docker ───────────────────────────────────────────────────────────────────

docker-build:  ## Build the training Docker image
	docker build -t medicalxai:latest .

docker-train:  ## Run training inside Docker
	docker run --rm --gpus all \
	  -v $(PWD)/artifacts:/app/artifacts \
	  -v $(PWD)/mlruns:/app/mlruns \
	  -e MEDAI_TRAIN_CSV=$(MEDAI_TRAIN_CSV) \
	  medicalxai:latest

# ── Housekeeping ─────────────────────────────────────────────────────────────

clean:  ## Remove build and cache artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
