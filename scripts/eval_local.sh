#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Evaluate the latest checkpoint locally.
# Usage:
#   bash scripts/eval_local.sh
#   bash scripts/eval_local.sh --checkpoint artifacts/checkpoints/best.pt
#   bash scripts/eval_local.sh --data-csv /path/to/test.csv --split-name test
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

if [ -f ".env" ]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs)
fi

CONFIG="configs/train.yaml"

echo "[info] Running evaluation with config: $CONFIG"
python -m src.train.evaluate --config "$CONFIG" "$@"
