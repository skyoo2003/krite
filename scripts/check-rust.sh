#!/usr/bin/env bash
# Formats, lints, and tests the Rust workspace. Tests that need model weights are #[ignore]d;
# run them with: cargo test -p krite-candle --release -- --ignored
set -euo pipefail
cd "$(dirname "$0")/.."
cargo fmt --all --check
cargo clippy --workspace --all-targets --quiet -- -D warnings
cargo test --workspace --quiet
echo "rust ok"
