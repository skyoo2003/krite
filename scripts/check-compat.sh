#!/usr/bin/env bash
# Jev wire conformance: runs the official TypeSafe SDK suite (compat/jev) against the weight-free
# `hash_server` example. Against a real model instead, from compat/jev:
#   KRITE_BASE_URL=http://127.0.0.1:8110 uv run pytest -q
set -euo pipefail
cd "$(dirname "$0")/.."
port="${PORT:-8199}"

cargo build --quiet -p krite-server --example hash_server
target/debug/examples/hash_server "$port" &
pid=$!
trap 'kill "$pid" 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null && break
  kill -0 "$pid" 2>/dev/null || { echo "hash_server exited" >&2; exit 1; }
  sleep 0.5
done
curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null || { echo "hash_server did not start" >&2; exit 1; }

cd compat/jev
uv run --quiet ruff check .
uv run --quiet ruff format --check .
KRITE_BASE_URL="http://127.0.0.1:$port" uv run --quiet pytest -q
echo "compat ok"
