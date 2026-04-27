#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

[ -f .env ] && source .env

VENV_PYTHON="${VENV_PYTHON:-.venv/bin/python}"
CONFIG="${1:-configs/inference.yaml}"

echo "[generate_explanations] config=$CONFIG"
"$VENV_PYTHON" -m src.train.explainability \
    --config "$CONFIG"
echo "[generate_explanations] Done → $(grep output_dir configs/inference.yaml | awk '{print $2}')"
