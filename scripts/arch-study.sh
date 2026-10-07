#!/usr/bin/env bash
# Architecture study (docs/architecture-study.md): trains every arm, then measures each engine alone in
# the baseline order (fresh start, cold, warm, sustained, quality, invariance, interference), then
# renders the tables and the pre-registered verdict. Keep the Mac on AC with nothing else running.
#
#   scripts/arch-study.sh                          # everything (~12 h)
#   ARMS= ENGINES=arch-d2 scripts/arch-study.sh    # resume or rerun a subset
#   REPEAT="arch-b arch-late8 arch-b arch-late8"   # afterwards, latency only, in this order
#   ARMS="late8-broad late8-broad-s14" ENGINES="arch-late8-broad arch-late8-broad-s14" LATENCY=0 VERDICT=0 \
#     DEST=../docs/training-data.md REPORT="krite arch-late8 arch-late8-s14 arch-late8-broad arch-late8-broad-s14" \
#     scripts/arch-study.sh                        # a release-recipe stage (docs/training-data.md)
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -n "${CAFFEINATED:-}" ]] || exec env CAFFEINATED=1 caffeinate -i "$0" "$@"

ARMS="${ARMS-b d1 d2 d4 d2-emb}"
ENGINES="${ENGINES-arch-b arch-d1 arch-d2 arch-d4 arch-d2-emb arch-d2-nocache}"
COOL="${COOL:-600}"  # seconds idle between heavy steps (fanless: heat carries over)
DEST="${DEST-../docs/architecture-study.md}"  # relative to benchmarks/
VERDICT="${VERDICT-1}"  # 0: skip the architecture verdict
LATENCY="${LATENCY-1}"  # 0: skip the cold/warm cells (recipe arms share late8's architecture)
OUT=results/arch     # relative to benchmarks/
INVARIANCE_SUITES=agnews-choice,banking77-choice,massive-choice-en,massive-choice-ko
CACHE_SUITES=agnews-choice,boolq-noul,massive-choice-ko,amazon-score-ja

kb() { (cd benchmarks && uv run --quiet krite-bench --out "$OUT" "$@"); }
study() { (cd training && uv run --quiet python -m krite_train.study "$@" --out "../benchmarks/$OUT"); }
port() { python3 -c 'import sys, tomllib; print(tomllib.load(open("benchmarks/baselines/engines.toml", "rb"))[sys.argv[1]]["port"])' "$1"; }
stop() { lsof -ti "tcp:$(port "$1")" -sTCP:LISTEN | xargs kill 2>/dev/null || true; sleep 5; }

mkdir -p training/ckpt/logs
for arm in $ARMS; do
  if [[ -f "training/ckpt/$arm/train_meta.json" ]]; then echo "skip training $arm (checkpoint exists)"; continue; fi
  echo "== train $arm"
  (cd training && uv run --quiet python -m krite_train.train --arm "$arm") 2>&1 | tee "training/ckpt/logs/$arm.log"
  sleep "$COOL"
done

[[ -n "$ENGINES" ]] || exit 0
kb env
for engine in $ENGINES; do
  echo "== measure $engine"
  stop "$engine"
  kb latency --engines "$engine" --cache startup  # leaves the engine running
  if [[ "$LATENCY" == 1 ]]; then
    kb latency --engines "$engine" --cells aux --cache cold
    kb latency --engines "$engine" --cells aux --cache warm
  fi
  if [[ "$engine" == *-nocache ]]; then
    kb quality --engine "$engine" --suites "$CACHE_SUITES"
  else
    if [[ "$engine" == arch-b || "$engine" == arch-d2 || "$engine" == arch-late4 || "$engine" == krite || "$engine" == krite-v1 || "$engine" == krite-base ]]; then
      kb latency --engines "$engine" --mode sustained
    fi
    kb quality --engine "$engine" --memory
    kb invariance --engine "$engine" --suites "$INVARIANCE_SUITES"
    kb interference --engine "$engine"
  fi
  stop "$engine"
  sleep "$COOL"
done

# Latency-only repeats bound run-to-run variation; the last row per engine is what the verdict reads.
for engine in ${REPEAT:-}; do
  echo "== repeat latency $engine"
  stop "$engine"
  kb latency --engines "$engine" --cache startup
  kb latency --engines "$engine" --cells aux --cache cold
  kb latency --engines "$engine" --cells aux --cache warm
  stop "$engine"
  sleep "$COOL"
done

for engine in $ENGINES; do
  if [[ "$engine" == *-nocache ]]; then study compare --a "${engine%-nocache}" --b "$engine"; fi
done
REPORT="${REPORT-arch-b arch-d1 arch-d2 arch-d4 arch-d2-emb arch-d2-nocache arch-d1-pool arch-d1-lr arch-d1-set arch-late4 arch-late4-nocache arch-late8 arch-late4-s14 arch-b-s14 arch-late8-nocache arch-late6 arch-late8-s14 krite krite-nocache}"
kb report --engines "${REPORT// /,}" --dest "$DEST"
if [[ "$VERDICT" == 1 ]]; then study verdict | tee "benchmarks/$OUT/verdict.json"; fi
