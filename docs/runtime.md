# Krite Runtime

The Rust runtime serves [Protocol v1](protocol/v1.md) on `POST /v1/systemone`. It encodes each state once, independently of any question, keeps the resulting state memory in a cross-request cache, and scores every candidate with the late-interaction decision tower that the [architecture study](architecture-study.md) accepted (`late8`). Terms follow [ARCHITECTURE.md](../ARCHITECTURE.md).

The model behind it is `krite-0.15b-v0`: the study's `late8` checkpoint (mmBERT-small, 32,000 training examples, seed 13) with per-bucket temperatures. It is a pre-release model; quality numbers are the study's, not a release claim. Its successor `krite-0.15b-v1` (`late8-broad`: the same architecture on the broad training mixture, [training-data.md](training-data.md)) passes every release gate except accuracy (shortfall 0.783, down from 1.416); serve it with `--model training/ckpt/late8-broad/candle`. The measurements below are v0's.

## Crates

| Crate | Owns |
|---|---|
| `krite-core` | Protocol v1 request/response/error types, validation the JSON Schemas cannot express, server limits, canonical state string. No ML dependencies. |
| `krite-runtime` | `Backend` trait, the request pipeline, the state cache, `temperature[type][bucket]` calibration. |
| `krite-candle` | `CandleBackend`: the late-interaction decision tower on the mmBERT-small (ModernBERT) encoder, Candle Metal or CPU, and the candidate cache. |
| `krite-server` | axum router for `POST /v1/systemone`: body limit, errors, headers, timing. |
| `krite-cli` | The `krite` binary: `krite serve`, `krite bench-encoder`. |

## Build and serve

```bash
(cd training && uv run python -m krite_train.export --ckpt ckpt/late8)   # model directory, once
cargo build --release -p krite-cli
target/release/krite serve --model training/ckpt/late8/candle [--port 8110] [--device auto|cpu] \
    [--state-cache-mb 1024] [--candidate-cache-mb 64] [--raw]
target/release/krite bench-encoder --model training/ckpt/late8/candle [--tokens 64,512,2048] [--warmup 20] [--n 200]
scripts/check-rust.sh                                                   # fmt, clippy -D warnings, tests
KRITE_MODEL=$PWD/training/ckpt/late8/candle cargo test -p krite-candle --release -- --ignored
```

The server binds `127.0.0.1` only and has no authentication. `--device auto` uses Metal when it is available. The server compiles its Metal pipelines with one warmup request before it listens. A cache budget of 0 turns that cache off. `--raw` ignores the shipped temperatures and serves uncalibrated probabilities; it exists to fit them.

## Model directory

`python -m krite_train.export` (in `training/`) turns a late-interaction checkpoint into a directory that the runtime loads; nothing is downloaded at runtime.

| File | Content |
|---|---|
| `model.safetensors` | The checkpoint's weights, fp32, torch key names (`encoder.*`, `type_emb.weight`, `scorer.*`). |
| `config.json`, `tokenizer.json` | The base model's files, `jhu-clsp/mmBERT-small` at `abc32620dd4f6ab06f5fbe905dc25f310618e09f` (training only fine-tunes weights). |
| `krite.json` | Model id, number of interaction layers, candidate token limits, per-bucket temperatures, and the checkpoint's provenance (training-set hash, seed). |
| `probe.json` | torch CPU energies on synthetic text; the ignored Rust tests check token ids and energies against it. |

Temperatures come from a calibration run: `--calibration ../benchmarks/results/arch/calibration.jsonl --engine krite-raw` ships the temperatures that `krite-bench quality` fitted on the uncalibrated engine `krite-raw` (`krite serve --raw`). The exporter refuses an engine whose start command lacks `--raw`: a fit on already-calibrated probabilities is a correction near 1.0, and shipping it would silently drop the calibration. Without `--calibration`, every temperature is 1.0. The exporter writes into `<out>.staging` and swaps it in only after every file is written, so a failed export leaves the previous directory intact.

The encoder code is adapted from candle-transformers 0.11.0 (MIT OR Apache-2.0). It uses Candle's fused attention kernel on Metal for the encoder, splits fused projections at load (per-head q/k/v weights, GeGLU gate and up halves) because Candle's strided Metal kernels are an order of magnitude slower than its contiguous ones, and builds the local-attention band mask in the weight dtype. Precision is fp32.

## Request path

1. Parse and validate (`krite-core`). A `model` other than the loaded model id returns `unknown_model`.
2. Canonicalize the state (string as is; object as its RFC 8785 JCS string, so `{"x":1}` and `{"x":1.0}` are the same state) and tokenize it with `<bos>`/`<eos>`.
3. Look up the state cache. On a miss, run the lower 14 encoder layers on the state and keep its rotated keys and values in each of the top 8 layers (the state memory).
4. Build one input per candidate: `<bos> instructions \n name[: desc] <eos>`, at most 32 tokens, with up to 14 kept for the criterion. Instructions are tokenized once per question, so tokenization work is linear in the request body (a 256 KiB instruction with 255 criteria tokenizes in under 50 ms). Look up each candidate's lower-layer states in the candidate cache; misses run the lower 14 layers together. In the top 8 layers every candidate attends to the state memory and to its own tokens, never to other candidates, at positions after the state; a masked mean plus a question-type embedding goes to the scorer. Question ids never reach the model.
5. Softmax per question with the bucket temperature from `krite.json`, then build Choice / Score / Noul answers. Choice ties go to the smallest name in codepoint order.

**State cache.** The key is `sha256(model id, tokenizer version, canonical state)`; the tokenizer version is a hash of `tokenizer.json`, so a model or tokenizer change invalidates every entry. The cache is bounded by bytes (`--state-cache-mb`, default 1024 MiB; a 512-token state takes 12 MiB, so the default holds about 85 of them) and evicts the least recently used state. Concurrency is one request at a time (a single lock around the runtime).

**Candidate cache.** Lower-layer states depend only on a candidate's token ids, so they are cached under `sha256(ids)` within `--candidate-cache-mb` (default 64 MiB; a 32-token candidate takes 48 KiB). Each cached entry owns its storage, copied out of the batch it was computed in, so the budget bounds the memory it holds. The cache changes no output: with both caches off, every probability is identical (below).

**Limits** (Protocol v1 §4): 64 questions, 255 criteria per question, 4 MiB body, 8192 encoder tokens per state, `<bos>` and `<eos>` included (so up to 8190 content tokens), and 65,536 tokens of candidate input per request. Candidates run sorted by length in batches of at most 16,384 padded tokens, in the lower and the top layers (and, in the top layers, at most 256 MiB of attention scores), so a request of many short candidates and one long one does not pad every candidate to the longest: such a request (5,721 candidates, 52 questions) peaks at 4.9 GiB physical footprint, against 9.7 GiB with one batch.

**Headers.** Every response carries `x-krite-request-id` and `x-typesafe-request-id` (same value). A 200 also carries `x-krite-state-cache: hit|miss` and `x-krite-timing: tokenize=…;encode=…;decide=…;calibrate=…` in milliseconds.

## Measured

MacBook Air 13 (Apple M4, 16 GB, fanless), AC power, Candle Metal, fp32, 2026-10-05. Measured with the baseline harness under the same procedure as the baselines ([baselines.md](baselines.md)): engine alone on an otherwise idle machine; fresh start, cold cells, warm cells, 10 sustained minutes, quality, invariance, interference; then the engine with both caches off. The encoder comparison ran after the server stopped. Raw rows are in [`benchmarks/results/arch/`](../benchmarks/results/arch) (engines `krite` and `krite-nocache`, next to the torch reference `arch-late8`). Every measured call succeeded (`error_rate` 0).

### Same model as the torch reference

| Check | Result |
|---|---|
| Candidate and state token ids vs. `training/` (probe) | identical |
| Energies vs. torch CPU (probe, 7 candidates, 492-token state) | max \|Δ\| 8.6e-6 |
| Probabilities vs. `arch-late8` (torch MPS), agnews, boolq, massive-ko, amazon-ja | max \|Δ\| 6.6e-6 |
| Accuracy / F1 / QWK, all 28 suites | identical to `arch-late8`: in-domain accuracy 0.877, held-out 0.708, mean QWK 0.264 |

### Encoder, model layer (one forward pass, device synchronized, 200 runs)

| State tokens | Candle Metal p50 / p95 | torch MPS p50 / p95 | Candle / torch |
|---|---|---|---|
| 64 | 10.8 / 11.2 ms | 9.3 / 10.1 ms | 1.2× |
| 512 | 74.8 / 76.4 ms | 39.1 / 40.7 ms | 1.9× |
| 2048 | 613 / 703 ms | 252 / 265 ms | 2.4× |

The torch column is the earlier measurement of the same backbone shape (transformers, `sdpa` attention); fine-tuning does not change its cost.

### Request latency, 512-token state, K = 4 (http layer; runtime layer in parentheses)

| Cell | p50 | p95 | Decisions/s | torch `arch-late8` p50 |
|---|---|---|---|---|
| startup (launch to first 200 on the ping probe) | 524 ms | — | — | 12.4 s |
| cold, 1 question | 91.3 ms (90.9) | 94.6 ms | 10.9 | |
| cold, 10 questions | 127.2 ms (126.8) | 133.9 ms | 78.0 | |
| cold, 30 questions | 227.9 ms (227.3) | 242.9 ms | 130 | |
| warm, 1 question | 7.8 ms (7.6) | 8.1 ms | 128 | 6.8 ms (6.2) |
| warm, 10 questions | 46.4 ms (45.8) | 49.2 ms | 215 | 28.2 ms |
| warm, 30 questions | 163.9 ms (162.6) | 206.8 ms | 180 | 87.9 ms |
| warm, 1 question, sustained 10 min | 8.1 ms (last minute 7.9) | 18.8 ms | 88.6 | |

Sustained minutes alternate between about 8 ms and 13–17 ms per-minute p50 with no thermal warning recorded; the burst cells did not show this. Per-candidate work is slower than torch's: Candle's Metal path still has strided copies and broadcasts that grow with the number of candidates, so the gap widens with more questions. With both caches off (`krite-nocache`), warm equals cold: 125.6 ms for one question.

Peak memory during the quality run: 4.17 GiB physical footprint, 0.94 GiB RSS.

### Invariants through HTTP

| Check | Result |
|---|---|
| Option-order flips, all permutations (agnews, banking77, massive-en, massive-ko; 100 cases each) | 0; max probability deviation 0 |
| Question interference (agnews; alone vs. 3 and 15 fillers) | argmax change 0; max probability deviation 5.3e-7 |
| Cache hit vs. miss (`krite` vs. `krite-nocache`, 4 suites) | max probability deviation 0 |

### Calibration (one temperature per bucket, fitted on half of each suite, scored on the other half)

| Bucket | Suites | Temperature | ECE raw → scaled |
|---|---|---|---|
| `choice/4` | 1 | 1.36 | 0.152 → 0.171 |
| `choice/5-8` | 10 | 1.60 | 0.059 → 0.041 |
| `choice/9+` | 1 | 1.78 | 0.046 → 0.048 |
| `noul` | 9 | 1.84 | 0.134 → 0.071 |
| `score/5` | 7 | 2.24 | 0.064 → 0.051 |
| mean of 28 suites | | | 0.087 → 0.058 |

The runtime applies the shipped temperatures exactly: served probabilities equal the temperature-scaled raw predictions within 3e-16. The bucket temperatures are pooled over all suites in the bucket, so single-suite buckets (`choice/4` is agnews only, held out from training) can get worse.

### Against the targets (measured, not gated at this stage)

| Target | Best baseline | Krite runtime | Note |
|---|---|---|---|
| Cold, primary cell ≤ 210 ms | Laya 210 ms | 91.3 ms | Dominated by the encoder (75 ms). |
| Warm, primary cell ≤ 10 ms | Kev 52 ms | 7.8 ms | 7.8× faster than the joint encoder (`arch-b`, 61.3 ms). |
| 512/30Q warm ≥ 175 decisions/s | Kev 35.1 | 180 | Narrow margin; torch runs the same model at 341. |
| Option permutation flip rate 0 | — | 0 | |
| ECE after calibration ≤ 0.071 | Kev 0.071 | 0.058 (mean of suites) | Study model; the release model is calibrated again. |
