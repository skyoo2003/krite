#!/usr/bin/env bash
# Lints and tests the training code (training/). Tests use a tiny random encoder; no downloads.
set -euo pipefail
cd "$(dirname "$0")/../training"
uv run --quiet ruff check .
uv run --quiet ruff format --check .
uv run --quiet pytest -q -m "not network"
echo "train ok"
