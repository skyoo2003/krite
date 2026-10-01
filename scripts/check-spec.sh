#!/usr/bin/env bash
# Validates Protocol v1 schemas and example fixtures.
set -euo pipefail
cd "$(dirname "$0")/../docs/protocol"
cjs() { uvx --quiet check-jsonschema "$@"; }

cjs --check-metaschema request.schema.json response.schema.json error.schema.json
cjs --schemafile request.schema.json examples/*.request.json
cjs --schemafile response.schema.json examples/*.response.json
cjs --schemafile error.schema.json examples/*.error.json

# Negative fixtures must fail schema validation itself; a tool or load error is not a rejection.
for f in examples/invalid/*.request.json; do
  if out=$(cjs --schemafile request.schema.json "$f" 2>&1); then
    echo "FAIL: $f should be rejected" >&2; exit 1
  fi
  if ! grep -q "Schema validation errors were encountered" <<<"$out"; then
    echo "FAIL: $f did not produce a schema validation error:" >&2
    echo "$out" >&2; exit 1
  fi
done
# Limits are server config, not protocol: the schema must accept requests above the defaults (64 questions, 255 options).
big=$(mktemp)
trap 'rm -f "$big"' EXIT
python3 -c 'import json; c = {f"o{i}": None for i in range(256)}; print(json.dumps({"state": "s", "questions": {f"q{i}": {"type": "choice", "instructions": "i", "criteria": c} for i in range(65)}}))' > "$big"
cjs --schemafile request.schema.json "$big"

echo "spec ok"
