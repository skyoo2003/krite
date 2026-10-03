# Krite Runtime

The Rust runtime serves [Protocol v1](protocol/v1.md) on `POST /v1/systemone`. It encodes each state once, independently of any question, and keeps the encoder output in a cross-request cache. Terms follow [ARCHITECTURE.md](../ARCHITECTURE.md).

> **Temporary head.** The model behind this runtime is `krite-0.15b-v0`: the mmBERT-small encoder plus a placeholder scoring head, not the Krite decision tower. Its answers are well-formed, order-invariant, and question-isolated, but their quality is meaningless, so no quality or calibration numbers are published for it. Warm and multi-question latencies below measure the runtime floor (tokenization, cache, one device round trip), not the cost of a trained decision tower.

## Crates

| Crate | Owns |
|---|---|
| `krite-core` | Protocol v1 request/response/error types, validation the JSON Schemas cannot express, server limits, canonical state string. No ML dependencies. |
| `krite-runtime` | `Backend` trait, the request pipeline, the state cache, `temperature[type][bucket]` calibration. |
| `krite-candle` | `CandleBackend`: mmBERT-small (ModernBERT) encoder on Candle Metal or CPU, the temporary head. |
| `krite-server` | axum router for `POST /v1/systemone`: body limit, errors, headers, timing. |
| `krite-cli` | The `krite` binary: `krite serve`, `krite bench-encoder`. |

## Build and serve

```bash
cargo build --release -p krite-cli
target/release/krite serve [--port 8110] [--device auto|cpu] [--state-cache-mb 1024]
target/release/krite bench-encoder [--tokens 64,512,2048] [--warmup 20] [--n 200] [--device auto|cpu]
scripts/check-rust.sh                                  # fmt, clippy -D warnings, tests
cargo test -p krite-candle --release -- --ignored     # tests that need the weights
```

The server binds `127.0.0.1` only and has no authentication. `--device auto` uses Metal when it is available. The server compiles its Metal pipelines with one warmup request before it listens.

## Weights

Weights resolve from the local Hugging Face cache (`$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`). Nothing is downloaded at runtime; a missing file fails startup with the command to fetch it.

| File | Repo | Revision |
|---|---|---|
| `config.json`, `tokenizer.json` | `jhu-clsp/mmBERT-small` | `abc32620dd4f6ab06f5fbe905dc25f310618e09f` |
| `model.safetensors` | `jhu-clsp/mmBERT-small` | `461475a70b192efcbb760df5541d3825b07c4d5b` (`refs/pr/12`, safetensors conversion of the same weights) |

```bash
uvx --from huggingface_hub hf download jhu-clsp/mmBERT-small config.json tokenizer.json --revision abc32620dd4f6ab06f5fbe905dc25f310618e09f
uvx --from huggingface_hub hf download jhu-clsp/mmBERT-small model.safetensors --revision 461475a70b192efcbb760df5541d3825b07c4d5b
```

The encoder code is adapted from candle-transformers 0.11.0 (MIT OR Apache-2.0). It uses Candle's fused attention kernel on Metal, runs the global-attention layers without a mask (one unpadded sequence per call), and builds the local-attention band mask in the weight dtype. Precision is fp32.

## Request path

1. Parse and validate (`krite-core`). A `model` other than `krite-0.15b-v0` returns `unknown_model`.
2. Canonicalize the state (string as is; object as its RFC 8785 JCS string, so `{"x":1}` and `{"x":1.0}` are the same state) and tokenize it with `<bos>`/`<eos>`.
3. Look up the state cache. On a miss, run the encoder and insert the per-token hidden states.
4. Build one text per candidate (`instructions`, the candidate name, and its description if any), tokenize them, and score all candidates of the request in one backend call. Question ids never reach the model.
5. Softmax per question with the bucket temperature (1.0 for every bucket in `v0`), then build Choice / Score / Noul answers. Choice ties go to the smallest name in codepoint order.

**State cache.** The key is `sha256(model id, tokenizer version, canonical state)`; the tokenizer version is a hash of `tokenizer.json`, so a model or tokenizer change invalidates every entry. The cache is bounded by bytes (`--state-cache-mb`, default 1024 MiB; a 512-token state takes 0.75 MiB) and evicts the least recently used state. Concurrency is one request at a time (a single lock around the runtime).

**Limits** (Protocol v1 §4): 64 questions, 255 criteria per question, 4 MiB body, 8192 encoder tokens per state, `<bos>` and `<eos>` included (so up to 8190 content tokens), and 65,536 tokens of candidate text (instructions plus criterion) per request. The last limit bounds scoring memory, which grows with candidate tokens; the body limit alone allows requests that would need over 100 GiB. Candidates are mean-pooled per segment, so scoring memory is linear in candidate tokens (about 100 MB at the limit).

**Headers.** Every response carries `x-krite-request-id` and `x-typesafe-request-id` (same value). A 200 also carries `x-krite-state-cache: hit|miss` and `x-krite-timing: tokenize=…;encode=…;decide=…;calibrate=…` in milliseconds.

## Measured

MacBook Air 13 (Apple M4, 16 GB, fanless), AC power, Candle Metal, fp32, 2026-10-03. Measured with the baseline harness under the same procedure as the baselines ([baselines.md](baselines.md)): engine alone on an otherwise idle machine; fresh start, cold cells, warm cells, 10 sustained minutes, then invariance and interference; the encoder comparison ran after the server stopped and the machine idled for 10 minutes. Raw rows are in [`benchmarks/results/krite/`](../benchmarks/results/krite). Every measured call succeeded (`error_rate` 0).

### Encoder, model layer (one forward pass, device synchronized, 200 runs)

| State tokens | Candle Metal p50 / p95 | torch MPS p50 / p95 | Candle / torch | max \|Δ\| vs torch |
|---|---|---|---|---|
| 64 | 11.6 / 11.7 ms | 9.3 / 10.1 ms | 1.2× | 1.1e-5 |
| 512 | 91.0 / 91.7 ms | 39.1 / 40.7 ms | 2.3× | 1.4e-5 |
| 2048 | 670 / 672 ms | 252 / 265 ms | 2.7× | 6.4e-6 |

The torch side is the same mmBERT-small backbone (transformers, `sdpa` attention) on identical token ids. Candle's gap is not attention: with attention removed, the Candle encoder still takes about 70 ms at 512 tokens, in fp32 and in fp16.

### Request latency, 512-token state, K = 4 (http layer; runtime layer in parentheses)

| Cell | p50 | p95 | Decisions/s |
|---|---|---|---|
| startup (launch to first 200 on the ping probe) | 1161 ms | — | — |
| cold, 1 question | 97.6 ms (97.3) | 100.8 ms | 10.2 |
| cold, 10 questions | 98.1 ms (97.8) | 98.8 ms | 102 |
| cold, 30 questions | 99.3 ms (99.0) | 100.0 ms | 302 |
| warm, 1 question | 1.10 ms (0.86) | 1.14 ms | 900 |
| warm, 10 questions | 2.15 ms (1.90) | 2.22 ms | 4668 |
| warm, 30 questions | 3.41 ms (3.13) | 3.73 ms | 8771 |
| warm, 1 question, sustained 10 min | 1.09 ms (last minute 1.09) | 1.12 ms | 914 |

### Invariants through HTTP

| Check | Result |
|---|---|
| Option-order flips, all permutations (agnews, banking77, massive-en, massive-ko; 100 cases each) | 0; max probability deviation 0 |
| Question interference (agnews; alone vs. 3 and 15 fillers) | argmax change 0; max probability deviation 8.8e-8 |

### Against the targets (measured, not gated at this stage)

| Target | Best baseline | Krite runtime | Note |
|---|---|---|---|
| Cold, primary cell ≤ 210 ms | Laya 210 ms | 97.6 ms | Dominated by the encoder (91 ms); the decision tower will add to it. |
| Warm, primary cell ≤ 10 ms | Kev 52 ms | 1.10 ms | Temporary head: runtime floor only. |
| 512/30Q warm ≥ 175 decisions/s | Kev 35.1 | 8771 | Temporary head: runtime floor only. |

Read these numbers with the temporary-head caveat at the top: the cold column already includes the full encoder, but the warm and throughput columns will move once the decision tower replaces the head.
