#!/usr/bin/env bash
# Builds the Python extension (python/, PyPI `krite`) into its venv and tests it against a tiny random
# model the tests write; no weights or network needed.
set -euo pipefail
cd "$(dirname "$0")/../python"
uv sync --quiet
uv run --quiet maturin develop --uv --release --quiet
uv run --quiet ruff check .
uv run --quiet ruff format --check .
uv run --quiet pytest -q
echo "python ok"
