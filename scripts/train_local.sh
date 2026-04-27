#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Train locally from the project root.
# Usage:
#   bash scripts/train_local.sh
#   bash scripts/train_local.sh --config configs/train.yaml
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Load .env if present
if [ -f ".env" ]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs)
  echo "[info] Loaded .env"
fi

CONFIG="${1:---config configs/train.yaml}"
EXTRA_ARGS="${@:2}"

echo "[info] Starting training: $CONFIG $EXTRA_ARGS"
python -m src.train.train $CONFIG $EXTRA_ARGS
