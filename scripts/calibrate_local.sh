#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

[ -f .env ] && source .env

VENV_PYTHON="${VENV_PYTHON:-.venv/bin/python}"
CONFIG="${1:-configs/train.yaml}"
OUTPUT="${2:-artifacts/calibration.json}"

echo "[calibrate_local] config=$CONFIG  output=$OUTPUT"
"$VENV_PYTHON" -m src.train.calibrate \
    --config "$CONFIG" \
    --output "$OUTPUT"
echo "[calibrate_local] Done → $OUTPUT"
