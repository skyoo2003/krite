# Krite Benchmark Spec

Rules for measuring Krite and baseline engines under identical conditions. The primary measurement platform is Apple Silicon (MacBook Air 13, M4, 16GB, fanless). Terms follow [ARCHITECTURE.md](../ARCHITECTURE.md).

## 1. Environment record

Attach this record to every result.

- Device model, chip, memory, macOS version
- Power adapter connected or not, Low Power Mode on or off, approximate ambient temperature
- `pmset -g therm` output
- backend: `candle-metal` | `candle-cpu` | `torch-mps` | `torch-cpu` | `onnxruntime`
- precision: `fp32` | `fp16` | `bf16` | `int8`
- Model id, git commit, measurement time (ISO 8601)

## 2. Latency layers

Always report the three layers separately. Never report one layer's number as another layer's.

| Layer | Start | End | Includes |
|---|---|---|---|
| `model` | tokenized tensor input | logits | device synchronization (Metal command buffer completion) |
| `runtime` | parsed request struct | answers | tokenization, cache lookup, calibration |
| `http` | client sends request | client receives response | loopback, keep-alive, JSON serialization/deserialization |

Metal executes asynchronously. Timing the `model` layer without device synchronization measures only kernel submission (latency theater). Always synchronize before recording time.

## 3. Cache states

| `cache_state` | Definition |
|---|---|
| `startup` | First call after process load. Reported once, separately. |
| `cold` | Weights loaded and warmup done, but state cache miss. Flush the cache or use a unique state on every iteration. |
| `warm` | Same state requested again; cache hit. |

Repeating the same state in a cold measurement produces cache hits, which measures warm, not cold.

## 4. Modes

- **burst**: 20 warmup runs, then 200 measured runs. Report p50/p95/p99. When comparing several engines, interleave per call (as cbjev does) so thermal drift does not land on one engine only.
- **sustained**: 10 minutes of continuous requests. Report a p50 curve in 1-minute windows and the p50 of the last minute. The goal is to expose thermal throttling on the fanless device.

## 5. Matrix

| Axis | Values |
|---|---|
| state length | 64, 512, 2048 tokens |
| questions | 1, 4, 16, 64 |
| options K | 2, 4, 8 |

**Primary cell: 512 tokens / 1 question / K = 4**, reported both cold and warm. Other documents refer to this section instead of redefining the primary cell.

Auxiliary cells for cbjev comparison: state ~500 tokens × questions {1, 10, 30}.

## 6. Throughput

`decisions_per_sec` = answered questions / wall-clock time. Default concurrency is 1. Do not report tokens/sec.

## 7. Invariance metrics

- **full-permutation flip rate**: All permutations when K ≤ 5; 24 random permutations with seed 13 when K > 5. The fraction of (question, permutation) pairs whose argmax differs from the original order. Primary metric.
- **reverse flip rate**: Only the reversed order. Auxiliary metric for cbjev comparison.
- **max prob deviation**: Maximum absolute difference in the same candidate's probability across permutations.
- **question interference**: Compare a question asked alone vs. inside the full question set. Report the argmax change rate and max prob deviation.
- Tolerance: 1e-5 for fp32, 1e-3 for fp16/bf16 (absolute). Report anything above it as an invariant violation.

## 8. Quality metrics

| Type | Metrics |
|---|---|
| choice | accuracy, macro-F1 |
| noul | accuracy, AUROC, Brier |
| score | MAE, QWK, RPS (discrete CRPS) |

## 9. Calibration metrics

- ECE: top-1 probability, 15 equal-width bins
- adaptive ECE: 15 equal-mass bins
- NLL
- multi-class Brier

Report overall and per calibration bucket (see Calibration in the ARCHITECTURE.md Glossary). The split used to fit the calibrator must be separate from the evaluation split. Re-measure every metric for each quantized variant.

## 10. Selectivity

Report the risk-coverage curve and AURC, sorted by top-1 probability.

## 11. Data rules

- Case sampling seed is 13. Use 400 cases per suite where possible.
- Keep a separate source-held-out suite.
- Record dataset name, version, and hash.
- Include non-English suites, and report quality and calibration metrics per language as well as overall.
- Jev outputs are an evaluation reference only and are never used for training.
- The concrete dataset list is decided in M1 (TBD).

## 12. Result record

Results are JSONL, one line per cell.

| Field | Description |
|---|---|
| `engine` | Engine under test (`krite`, `cbjev`, `laya`, etc.) |
| `model` | Model id |
| `backend` | backend value from §1 |
| `precision` | precision value from §1 |
| `layer` | `model` \| `runtime` \| `http` |
| `cache_state` | `startup` \| `cold` \| `warm` |
| `mode` | `burst` \| `sustained` |
| `state_tokens` | state length |
| `questions` | number of questions |
| `options` | number of options K |
| `n` | number of measured runs |
| `p50_ms`, `p95_ms`, `p99_ms` | latency percentiles |
| `decisions_per_sec` | §6 |
| `env_ref` | path to the §1 environment record |
| `commit` | git commit |
| `timestamp` | ISO 8601 |

## 13. Reporting rules

Never mix numbers quoted from READMEs and numbers measured directly in the same column. When showing both, use separate columns and link the source of every quoted number.
