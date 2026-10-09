#!/usr/bin/env bash
# Uploads krite-0.15b-v1 (model directory, model card, license) to the Hugging Face Hub.
# Log in first: uvx --from huggingface_hub hf auth login. Usage: scripts/publish-model.sh [repo id]
set -euo pipefail
cd "$(dirname "$0")/.."
model=${KRITE_MODEL_DIR:-training/ckpt/late8-v2/candle}
repo=${1:-skyoo2003/krite-0.15b-v1}
if ! grep -q '"model_id": "krite-0.15b-v1"' "$model/krite.json"; then
  echo "FAIL: $model is not krite-0.15b-v1" >&2; exit 1
fi
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
cp "$model"/{model.safetensors,config.json,tokenizer.json,krite.json,probe.json} "$stage"/
cp docs/models/krite-0.15b-v1.md "$stage/README.md"
cp LICENSE "$stage/LICENSE"
uvx --from huggingface_hub hf upload "$repo" "$stage" . --repo-type model --commit-message "krite-0.15b-v1"
echo "model ok: https://huggingface.co/$repo"
