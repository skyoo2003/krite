# Krite Benchmark Spec

Rules for measuring Krite and baseline engines under identical conditions. The primary measurement platform is Apple Silicon (MacBook Air 13, M4, 16GB, fanless). Terms follow [ARCHITECTURE.md](../ARCHITECTURE.md).

## 1. Environment record

Attach this record to every result.

- Device model, chip, memory, macOS version
- Power adapter connected or not, Low Power Mode on or off, approximate ambient temperature
- `pmset -g therm` output
- backend: `candle-metal` | `candle-cpu` | `torch-mps` | `torch-cpu` | `mlx` | `onnxruntime`
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

- **burst**: 20 warmup runs, then 200 measured runs. Report p50/p95/p99. When comparing several engines, interleave per call (as cbjev does) so thermal drift does not land on one engine only. When the engines do not fit in memory together (memory contention distorts latency more than drift does), measure them one at a time instead, each with the same sequence on an otherwise idle machine: fresh start (`startup`), cold cells, warm cells, then sustained.
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

`decisions_per_sec` = answered questions / wall-clock time, where the time includes every measured call, failed calls too (a timeout must lower throughput, not vanish from it). Default concurrency is 1. Do not report tokens/sec.

## 7. Invariance metrics

- **full-permutation flip rate**: All K! − 1 non-identity permutations when K ≤ 5; 24 random permutations with seed 13 (identity excluded) when K > 5. The fraction of (question, permutation) pairs whose argmax differs from the original order. Primary metric.
- **reverse flip rate**: Only the reversed order. Auxiliary metric for cbjev comparison. When the reversed order is not in the permutation set (K > 5), it is called once more and counted only here.
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
- Suites below are built by `krite-bench data` (seed 13) from pinned dataset revisions. `benchmarks/data/manifest.json` records each source file's sha256 and each case file's sha256; `krite-bench data --verify` rebuilds and compares. Dataset text is never committed; only ids and hashes are.
- Instructions and criteria are English in every suite (`instructions_lang: en`); only the state is in the suite language.
- Choice distractors are seeded samples of the same dataset's other labels, and the gold option's position is randomized.
- Invariance and interference use a seeded 100-case subset per suite (call budget on the M4 Air), not 400.

| Suite | Source (mirror, pinned revision) | Type, K | Languages | Upstream license |
|---|---|---|---|---|
| `agnews-choice` | `fancyzhx/ag_news` test | choice, 4 | en | unspecified; eval-only |
| `banking77-choice` | `mteb/banking77` test | choice, 12 (gold + 11 distractors) | en | CC-BY-4.0 |
| `massive-choice-<lang>` | `mteb/amazon_massive_intent` test | choice, 8 (gold + 7 distractors) | en, ko, ja, zh-CN, de, es, fr, ar, hi, ru | CC-BY-4.0 |
| `boolq-noul` | `google/boolq` validation | noul; state `{passage, question}` | en | CC-BY-SA-3.0 |
| `xnli-noul-<lang>` | `facebook/xnli` test | noul (entailment = true, contradiction = false, neutral dropped); state `{premise, hypothesis}` | en, de, es, fr, ar, hi, ru, zh | CC-BY-NC-4.0; eval-only |
| `sst5-score` | `SetFit/sst5` test | score, 5 | en | unspecified; eval-only |
| `amazon-score-<lang>` | `mteb/amazon_reviews_multi` test | score, 5 (stars 1–5) | en, de, es, fr, ja, zh | Amazon research terms; eval-only |

Korean appears only in `massive-choice-ko`: XNLI and the Amazon corpus have no Korean split.

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

Optional latency fields: `truncated` (n reduced to 50 because 220 calls would exceed 15 minutes), `interleaved_with` (engines interleaved per call), `error_rate`, `therm_before`/`therm_after` (`pmset -g therm`), and for sustained mode `window_p50_ms` (per 60 s window) and `last_minute_p50_ms`. Startup rows add `probe`: the harness times `serve.sh` from process launch until its readiness probe (a one-question noul request, polled every 0.2 s) returns 200, which approximates §3 `startup` with a smaller request than the primary cell. `p50_ms` and the other percentiles are `null` when no measured call succeeded.

Other result files share `engine`, `model`, `backend`, `precision`, `commit`, and `timestamp`:

| File | One line per | Fields |
|---|---|---|
| `quality.jsonl` | engine × suite | `suite`, `lang`, `type`, `n`, `error_rate`, `not_supported_rate`, `n_scored`, `max_sum_dev`, §8 metrics, `raw` block of §9–10 metrics over all cases (`ece`, `adaptive_ece`, `nll`, `brier`, `aurc`) |
| `calibration.jsonl` | engine × suite | `temperature` and `n_fit_pooled` per bucket, `raw_eval_half` / `scaled_eval_half` blocks. One temperature per engine model and calibration bucket, fit on the pooled calibration halves of every suite in the bucket (split keyed on case index), evaluated on each suite's other half. Engines with a model per dataset (`request_model`) get one calibrator per dataset. Rebuilt from the `pred-*` files by `krite-bench calibrate` |
| `invariance.jsonl` | engine × suite | `cases`, `calls`, `error_rate`, `full_permutation_flip_rate`, `reverse_flip_rate`, `max_prob_dev`, `mean_prob_dev` |
| `interference.jsonl` | engine × filler count | `suite`, `fillers`, `cases`, `error_rate`, `argmax_change_rate`, `max_prob_dev` |
| `memory.jsonl` | engine | `pid`, `peak_phys_footprint_bytes` (macOS `footprint`, includes GPU-backed unified memory), `peak_rss_bytes` |
| `status.jsonl` | engine × event | `status` (`not_runnable` \| `not_supported` \| `error`), `reason`, optional `cell`/`cache_state` |
| `pred-<engine>-<suite>.jsonl` | case | `case_id`, `http_ms`, `error`, `probs` (request candidate order), `gold` (index) |

A call counts as an error (kept in the `error_rate` denominator) when the response is not schema-valid, does not answer every question id, answers with a different question type, omits a candidate's probability, or returns a `choice` that is not an offered candidate with the top probability (ties within 1e-6 accepted).

## 13. Reporting rules

Never mix numbers quoted from READMEs and numbers measured directly in the same column. When showing both, use separate columns and link the source of every quoted number.
