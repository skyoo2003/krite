#!/usr/bin/env bash
# Lints and tests the benchmark harness (benchmarks/), including fake-engine end-to-end runs.
set -euo pipefail
cd "$(dirname "$0")/../benchmarks"
uv run --quiet ruff check .
uv run --quiet ruff format --check .
uv run --quiet pytest -q
echo "bench ok"
