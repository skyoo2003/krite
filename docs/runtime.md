# Krite Runtime

The Rust runtime serves [Protocol v1](protocol/v1.md) on `POST /v1/systemone`, plus `GET /v1/models` for the Jev SDK. It encodes each state once, independently of any question, keeps the resulting state memory in a cross-request cache, and scores every candidate with the late-interaction decision tower that the [architecture study](architecture-study.md) accepted (`late8`). Terms follow [ARCHITECTURE.md](../ARCHITECTURE.md).

The model behind it is `krite-0.15b-v0`: the study's `late8` checkpoint (mmBERT-small, 32,000 training examples, seed 13) with per-bucket temperatures. It is a pre-release model; quality numbers are the study's, not a release claim. Its successor `krite-0.15b-v1` (`late8-broad`: the same architecture on the broad training mixture, [training-data.md](training-data.md)) passes every release gate except accuracy (macro accuracy 0.783 vs. 0.795, macro QWK 0.273 vs. 0.357) and ships as a pre-release with that gap stated; serve it with `--model training/ckpt/late8-broad/candle`. The optimized runtime is measured on v1 first; the tables after it are v0's on the earlier binary.

## Crates

| Crate | Owns |
|---|---|
| `krite-core` | Protocol v1 request/response/error types, validation the JSON Schemas cannot express, server limits, canonical state string. No ML dependencies. |
| `krite-runtime` | `Backend` trait, the request pipeline, the state cache, `temperature[type][bucket]` calibration. |
| `krite-candle` | `CandleBackend`: the late-interaction decision tower on the mmBERT-small (ModernBERT) encoder, Candle Metal or CPU, and the candidate cache. |
| `krite-server` | axum router for `POST /v1/systemone` and `GET /v1/models`: body limit, errors, headers, timing. |
| `krite-cli` | The `krite` binary: `krite serve`, `krite bench-encoder`, `krite bench-decide`. |
| `python/` (`krite-py`) | The PyPI package `krite`: the same runtime in process from Python (PyO3). Never published to crates.io. |

## Build and serve

Install from the registries: `cargo install krite-cli` and `hf download skyoo2003/krite-0.15b-v1 --local-dir krite-0.15b-v1`, then `krite serve --model krite-0.15b-v1` ([model card](https://huggingface.co/skyoo2003/krite-0.15b-v1)). Building from source:

```bash
(cd training && uv run python -m krite_train.export --ckpt ckpt/late8)   # model directory, once
cargo build --release -p krite-cli
target/release/krite serve --model training/ckpt/late8/candle [--port 8110] [--device auto|cpu] \
    [--state-cache-mb 1024] [--candidate-cache-mb 64] [--raw]
target/release/krite bench-encoder --model training/ckpt/late8/candle [--tokens 64,512,2048] [--warmup 20] [--n 200]
target/release/krite bench-decide --model training/ckpt/late8/candle [--state-tokens 512] [--questions 1,10,30] [--label step]
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
| `krite.json` | Model id, number of interaction layers, candidate token limits, per-bucket temperatures, export date (`release_date`, `YYYY-MM-DD`), and the checkpoint's provenance (training-set hash, seed). |
| `probe.json` | torch CPU energies on synthetic text; the ignored Rust tests check token ids and energies against it. |

Temperatures come from a calibration run: `--calibration ../benchmarks/results/arch/calibration.jsonl --engine krite-raw` ships the temperatures that `krite-bench quality` fitted on the uncalibrated engine `krite-raw` (`krite serve --raw`). The exporter refuses an engine whose start command lacks `--raw`: a fit on already-calibrated probabilities is a correction near 1.0, and shipping it would silently drop the calibration. Without `--calibration`, every temperature is 1.0. The exporter writes into `<out>.staging` and swaps it in only after every file is written, so a failed export leaves the previous directory intact.

The encoder code is adapted from candle-transformers 0.11.0 (MIT OR Apache-2.0). It uses Candle's fused attention kernel on Metal for the encoder, splits fused projections at load (per-head q/k/v weights, GeGLU gate and up halves) because Candle's strided Metal kernels are an order of magnitude slower than its contiguous ones, and builds the local-attention band mask in the weight dtype. Precision is fp32.

## Request path

1. Parse and validate (`krite-core`). A `model` other than the loaded model id or `jev-latest` returns `unknown_model`.
2. Canonicalize the state (string as is; object or array as its RFC 8785 JCS string, so `{"x":1}` and `{"x":1.0}` are the same state) and tokenize it with `<bos>`/`<eos>`.
3. Look up the state cache. On a miss, run the lower 14 encoder layers on the state and keep its rotated keys and values in each of the top 8 layers (the state memory).
4. Build one input per candidate: `<bos> instructions \n name[: desc] <eos>` (JSON instructions and descriptions as their JCS string; absent instructions as nothing), at most 32 tokens, with up to 14 kept for the criterion. Instructions are tokenized once per question, so tokenization work is linear in the request body (a 256 KiB instruction with 255 criteria tokenizes in under 50 ms). Look up each candidate's lower-layer states in the candidate cache; misses run the lower 14 layers together. In the top 8 layers every candidate attends to the state memory and to its own tokens, never to other candidates, at positions after the state; a masked mean plus a question-type embedding goes to the scorer. Question ids never reach the model.
5. Softmax per question with the bucket temperature from `krite.json`, then build Choice / Score / Noul answers. Choice ties go to the smallest name in codepoint order.

**State cache.** The key is `sha256(model id, tokenizer version, canonical state)`; the tokenizer version is a hash of `tokenizer.json`, so a model or tokenizer change invalidates every entry. The cache is bounded by bytes (`--state-cache-mb`, default 1024 MiB; a 512-token state takes 12 MiB, so the default holds about 85 of them) and evicts the least recently used state. Concurrency is one request at a time (a single lock around the runtime).

**Candidate cache.** Lower-layer states depend only on a candidate's token ids, so they are cached under `sha256(ids)` within `--candidate-cache-mb` (default 64 MiB; a 32-token candidate takes 48 KiB). Each cached entry owns its storage, copied out of the batch it was computed in, so the budget bounds the memory it holds. The cache changes no output: with both caches off, every probability is identical (below).

**Limits** (Protocol v1 §4): 64 questions, 255 criteria per question, 4 MiB body, 8192 encoder tokens per state, `<bos>` and `<eos>` included (so up to 8190 content tokens), and 65,536 tokens of candidate input per request. Candidates run sorted by length in batches of at most 16,384 padded tokens, in the lower and the top layers (and, in the top layers, at most 256 MiB of attention scores), so a request of many short candidates and one long one does not pad every candidate to the longest: such a request (5,721 candidates, 52 questions) peaks at 4.9 GiB physical footprint, against 9.7 GiB with one batch.

**Headers.** Every response carries `x-krite-request-id` and `x-typesafe-request-id` (same value). A 200 from `/v1/systemone` also carries `x-krite-state-cache: hit|miss` and `x-krite-timing: tokenize=…;encode=…;decide=…;calibrate=…` in milliseconds.

## Jev compatibility

The official TypeSafe Python SDK (`typesafe-sdk`, MIT) works against `krite serve` with only `base_url` changed. The SDK requires an API key client-side; any value works, since the server has no authentication.

```python
from typesafe_sdk import Choice, TypeSafeClient

with TypeSafeClient(api_key="any", base_url="http://127.0.0.1:8110") as client:
    r = client.system_one(state={"document": "I was charged twice."},
                          questions={"category": Choice(criteria={"billing": None, "technical": None})})
```

- The SDK sends `model: "jev-latest"` unless told otherwise; the server resolves that alias to the loaded model and answers with the real id.
- `GET /v1/models` lists the loaded model and `jev-latest` (`{"models": [{"name", "description", "release_date"}]}`; `release_date` is empty for exports that predate the field).
- Score answers key `probabilities` and `legend` by level index, which the SDK reads as integer keys.
- Wire differences that remain are listed in [Protocol v1 §7](protocol/v1.md#7-differences-from-jev).

`compat/jev` is the conformance suite: it drives the server through the SDK (pinned to 0.7.2, sync and async clients) and checks shapes, keys, sums, the alias, `GET /v1/models`, SDK-legal request forms (array or object state, absent or JSON instructions, JSON criteria), state key-order invariance, and 422 errors. It never checks answer values, so it runs against any model. `scripts/check-compat.sh` runs it against the weight-free `hash_server` example (CI: the `compat` job in `.github/workflows/ci.yml`); against a real model, start `krite serve` and run `KRITE_BASE_URL=http://127.0.0.1:8110 uv run pytest -q` in `compat/jev`. `krite-0.15b-v1` passes it.

## Python

`pip install krite` (wheels for macOS arm64 and Linux x86_64, CPython 3.11 or later) runs the runtime in process, built by the same `serving_runtime` as `krite serve`:

```python
import krite

k = krite.Krite("krite-0.15b-v1", device="auto", state_cache_mb=1024, candidate_cache_mb=64, raw=False)
r = k.decide({"state": "I was charged twice.", "questions": {"refund": {"type": "noul"}}})
```

`decide` takes a dict, a JSON string, or JSON bytes and returns what `POST /v1/systemone` would answer for the same body: the same JSON (serialized by the same code, then parsed into a dict), the same `jev-latest` alias, and the same errors, raised as `krite.KriteError` with the response's `status`, `type`, `message`, and `param`. A missing model directory raises `RuntimeError`. `device="auto"` uses Metal when available, `device="cpu"` forces the CPU. Calls on one `Krite` run one at a time behind one lock, like the server's, and release the GIL. `scripts/check-python.sh` builds the extension and tests it on the CPU against a tiny random model that the tests write (CI: the `binding` job).

## Optimization rules

Written down before any runtime optimization step was measured. Each step rewrites tensor layouts so that Candle runs its contiguous Metal kernels instead of the strided ones; it must not change what the model computes. The rules decide each step; the thresholds do not change after a result is seen.

- **R1 parity (each step).** `scripts/check-rust.sh` passes (CPU unit tests, including the late-attention reference), and `KRITE_MODEL=$PWD/training/ckpt/late8-broad/candle cargo test -p krite-candle --release -- --ignored` passes unchanged: torch probe max \|Δ energy\| ≤ 1e-4; warm, reversed, crowded, and uncached ≤ 1e-5.
- **R2 speed (each step).** `krite bench-decide` (512 state tokens, 1/10/30 questions, K = 4, n = 50) runs right before and right after the step, on AC power with nothing else running. The step is kept only if its target p50 drops by at least 3% and no other p50 (`encode_ms`, decide at 1, 10, and 30 questions) rises by more than 3%. Otherwise it is reverted and recorded as rejected. Targets: attention masks and head layout, `encode_ms`; late attention and stacked projections, decide at 30 questions. Rows go to [`benchmarks/results/arch/decide.jsonl`](../benchmarks/results/arch/decide.jsonl); the first row is the baseline.
- **R3 final (whole binary).** Engine `krite-v1-fast` serves the same `krite-0.15b-v1` export from the optimized binary. `study compare --a krite-v1 --b krite-v1-fast --tol 1e-4` passes on the cache suites, and `study release --engine krite-v1-fast --raw krite-v1-raw --nocache krite-v1-fast-nocache` passes every latency and invariant row. Accuracy and QWK are reported, not gated: the weights do not change. Burst and sustained runs are both reported (sustained is the primary cell, 10 minutes).
- **Precision stays fp32.** Measured on the 30-question shapes, an f16 gemm is only 1.3–1.7× faster than f32, the strided non-matmul ops that dominate do not get faster, and lower precision would put the 1e-5 invariance gates and the fitted temperatures at risk. INT8 is not adopted for the same reason; Candle's INT8 path (GGML `q8_0` `QMatMul`) also targets single-row LLM matmuls, not batched encoder matmuls.

| Step | Target p50 before → after | Other p50s | Result |
|---|---|---|---|
| Attention masks expanded once per encode or batch, not per layer | `encode_ms` 76.9 → 61.4 ms (−20.2%) | within ±3% | kept |
| Heads folded into sdpa's batch; per-head output projection | `encode_ms` 61.4 → 57.9 ms (−5.7%) | decide 30 questions −4.9% | kept |
| Late attention with contiguous kernels only | decide 30 questions 130.1 → 103.0 ms (−20.8%) | 1 question −11.5%, 10 questions −18.2% | kept |
| q/k/v stacked into one batched matmul | decide 30 questions 103.2 → 103.0 ms (−0.2%) | `encode_ms` +2.2% | rejected |
| MLP gate and up stacked into one batched matmul | decide 30 questions 103.0 → 102.6 ms (−0.4%) | within ±3% | rejected |

Each row compares the binary before and after the step, run back to back; R1 passed for every step.

## Measured

### Optimized runtime (`krite-0.15b-v1`, 2026-10-08)

The same machine and procedure as below. Engine `krite-v1-fast` is the `krite-v1` export served by the optimized binary; the `krite-v1` column is the earlier binary on 2026-10-07. Raw rows: `latency.jsonl`, `release-fast.json`, and `decide.jsonl` in [`benchmarks/results/arch/`](../benchmarks/results/arch). Every measured call succeeded.

| Cell, 512-token state, K = 4 (http; runtime in parentheses) | `krite-v1` p50 | `krite-v1-fast` p50 | p95 | Decisions/s |
|---|---|---|---|---|
| cold, 1 question | 90.1 ms (89.9) | 68.5 ms (68.2) | 69.3 ms | 14.6 |
| cold, 10 questions | 126.8 ms (126.5) | 96.9 ms (96.5) | 97.2 ms | 103 |
| cold, 30 questions | 224.9 ms (224.5) | 170.7 ms (170.2) | 171.3 ms | 176 |
| warm, 1 question | 7.85 ms (7.61) | 6.85 ms (6.62) | 6.93 ms | 146 |
| warm, 10 questions | 44.1 ms (43.8) | 34.7 ms (34.5) | 35.0 ms | 288 |
| warm, 30 questions | 141.7 ms (141.4) | 108.5 ms (108.1) | 111.4 ms | 276 |
| warm, 1 question, sustained 10 min | 8.12 ms (last minute 8.50) | 7.54 ms (last minute 7.72) | 8.01 ms | 134 |
| both caches off, warm, 1 question | 147.3 ms | 77.2 ms | 78.8 ms | 13.0 |

- **Parity (R3).** Probabilities vs. `krite-v1` on the four cache suites: max \|Δ\| 3.3e-6 (limit 1e-4). Cache on vs. off: 0.
- **Release gate.** Every latency and invariant row passes: warm 6.85 ms (≤ 10), cold 68.5 ms (≤ 210), 276.6 decisions/s at 30 questions (≥ 175), option-order flips 0 with max deviation 0, interference 1.6e-7, cache on/off 0. Accuracy (0.783) and QWK (0.273) are unchanged, since the weights are.
- **Sustained.** Per-minute p50 6.8–7.7 ms (`krite-v1`: 7.8–8.5 ms), no thermal warning recorded.
- **Model layer** (`krite bench-decide`, cooled down): `encode` at 512 tokens 76.8 → 56.8 ms; warm scoring 6.7 → 5.7 ms (1 question), 42.0 → 32.9 ms (10), 137.6 → 103.0 ms (30). The row labeled `final` was taken right after the 3-hour run on the fanless machine (30 questions 126.1 ms) and shows the thermal effect; `final-cool` is the same binary after 10 idle minutes.
- **Encoder** (`krite bench-encoder`, same session, earlier binary → optimized): 64 tokens 10.8 → 10.0 ms, 512 tokens 72.5 → 53.6 ms, 2048 tokens 603 → 286 ms (torch MPS: 9.3, 39.1, 252 ms). At 512 tokens Candle is now 1.4× torch's time, down from 1.9×.
- **Startup** (launch to first 200) varies with the file cache from 0.5 to 2.7 s for both binaries (cell 2.7 s, `krite-v1` 1.6 s); launched back to back, both take 0.49 s.
- **Memory.** One max-length request (8000-token state, 30 questions) peaks at 4.3 GiB physical footprint instead of 11.6 GiB and takes 2.7 s instead of 9.8 s; the 8190-token encoder forward peaks the same (3.9 vs. 4.0 GiB) and runs 8.1 → 2.2 s. Over the whole quality run the peak footprint rose from 4.46 to 6.14 GiB (RSS 1.28 → 1.23 GiB); the cause was not isolated.

### Earlier runtime (`krite-0.15b-v0`, 2026-10-05)

MacBook Air 13 (Apple M4, 16 GB, fanless), AC power, Candle Metal, fp32, 2026-10-05. Measured with the baseline harness under the same procedure as the baselines ([baselines.md](baselines.md)): engine alone on an otherwise idle machine; fresh start, cold cells, warm cells, 10 sustained minutes, quality, invariance, interference; then the engine with both caches off. The encoder comparison ran after the server stopped. Raw rows are in [`benchmarks/results/arch/`](../benchmarks/results/arch) (engines `krite` and `krite-nocache`, next to the torch reference `arch-late8`). Every measured call succeeded (`error_rate` 0).

#### Same model as the torch reference

| Check | Result |
|---|---|
| Candidate and state token ids vs. `training/` (probe) | identical |
| Energies vs. torch CPU (probe, 7 candidates, 492-token state) | max \|Δ\| 8.6e-6 |
| Probabilities vs. `arch-late8` (torch MPS), agnews, boolq, massive-ko, amazon-ja | max \|Δ\| 6.6e-6 |
| Accuracy / F1 / QWK, all 28 suites | identical to `arch-late8`: in-domain accuracy 0.877, held-out 0.708, mean QWK 0.264 |

#### Encoder, model layer (one forward pass, device synchronized, 200 runs)

| State tokens | Candle Metal p50 / p95 | torch MPS p50 / p95 | Candle / torch |
|---|---|---|---|
| 64 | 10.8 / 11.2 ms | 9.3 / 10.1 ms | 1.2× |
| 512 | 74.8 / 76.4 ms | 39.1 / 40.7 ms | 1.9× |
| 2048 | 613 / 703 ms | 252 / 265 ms | 2.4× |

The torch column is the earlier measurement of the same backbone shape (transformers, `sdpa` attention); fine-tuning does not change its cost.

#### Request latency, 512-token state, K = 4 (http layer; runtime layer in parentheses)

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

#### Invariants through HTTP

| Check | Result |
|---|---|
| Option-order flips, all permutations (agnews, banking77, massive-en, massive-ko; 100 cases each) | 0; max probability deviation 0 |
| Question interference (agnews; alone vs. 3 and 15 fillers) | argmax change 0; max probability deviation 5.3e-7 |
| Cache hit vs. miss (`krite` vs. `krite-nocache`, 4 suites) | max probability deviation 0 |

#### Calibration (one temperature per bucket, fitted on half of each suite, scored on the other half)

| Bucket | Suites | Temperature | ECE raw → scaled |
|---|---|---|---|
| `choice/4` | 1 | 1.36 | 0.152 → 0.171 |
| `choice/5-8` | 10 | 1.60 | 0.059 → 0.041 |
| `choice/9+` | 1 | 1.78 | 0.046 → 0.048 |
| `noul` | 9 | 1.84 | 0.134 → 0.071 |
| `score/5` | 7 | 2.24 | 0.064 → 0.051 |
| mean of 28 suites | | | 0.087 → 0.058 |

The runtime applies the shipped temperatures exactly: served probabilities equal the temperature-scaled raw predictions within 3e-16. The bucket temperatures are pooled over all suites in the bucket, so single-suite buckets (`choice/4` is agnews only, held out from training) can get worse.

#### Against the targets (measured, not gated at this stage)

| Target | Best baseline | Krite runtime | Note |
|---|---|---|---|
| Cold, primary cell ≤ 210 ms | Laya 210 ms | 91.3 ms | Dominated by the encoder (75 ms). |
| Warm, primary cell ≤ 10 ms | Kev 52 ms | 7.8 ms | 7.8× faster than the joint encoder (`arch-b`, 61.3 ms). |
| 512/30Q warm ≥ 175 decisions/s | Kev 35.1 | 180 | Narrow margin; torch runs the same model at 341. |
| Option permutation flip rate 0 | — | 0 | |
| ECE after calibration ≤ 0.071 | Kev 0.071 | 0.058 (mean of suites) | Study model; the release model is calibrated again. |
