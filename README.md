# Krite

An open decision model with a Rust runtime. Krite encodes a state once, independently of any
question, and reuses it across requests; it then scores the candidates of runtime-defined Choice,
Score, and Noul questions with one shared scorer, so option order cannot change a decision, and
returns calibrated probabilities. It serves the Jev-compatible
[Protocol v1](https://github.com/skyoo2003/krite/blob/main/docs/protocol/v1.md) (`POST /v1/systemone`)
on Candle (Apple Metal or CPU).

**Status.** `krite-0.15b-v1` passes all release gates (macro QWK 0.575 vs. cbjev 0.377 baseline,
macro accuracy 0.787 with unweighted suite macro accuracy 0.829 vs. cbjev 0.803, warm latency 8.05 ms runtime / 8.35 ms HTTP, throughput 254 decisions/s).
See the [model card](https://huggingface.co/skyoo2003/krite-0.15b-v1).

## Install

```bash
cargo install krite-cli                                   # the `krite` binary
hf download skyoo2003/krite-0.15b-v1 --local-dir krite-0.15b-v1
krite serve --model krite-0.15b-v1                        # 127.0.0.1:8110, no authentication

curl -s http://127.0.0.1:8110/v1/systemone -H 'content-type: application/json' -d '{
  "state": "I was charged twice for my subscription this month. Please refund one of the payments.",
  "questions": {"route": {"type": "choice", "instructions": "Which team should handle this?",
    "criteria": {"billing": "payments, refunds, invoices", "shipping": "delivery and tracking",
                 "technical": "bugs, login, app errors"}}}}'
```

From Python, in process and without a server (`pip install krite`), the same runtime answers the same request:

```python
import krite

k = krite.Krite("krite-0.15b-v1")
print(k.decide({"state": "I was charged twice.", "questions": {"refund": {"type": "noul"}}})["answers"])
```

## Packages

| Package | Registry | Content |
|---|---|---|
| `krite-core` | crates.io | Protocol v1 request/response/error types, validation, canonical state string. No ML dependencies. |
| `krite-runtime` | crates.io | `Backend` trait, request pipeline, state cache, per-bucket calibration. |
| `krite-candle` | crates.io | `CandleBackend`: the late-interaction decision tower on mmBERT, Metal or CPU. |
| `krite-server` | crates.io | axum router for `POST /v1/systemone`. |
| `krite-cli` | crates.io | The `krite` binary: `krite serve`, `krite bench-encoder`, `krite bench-decide`. |
| `krite` | PyPI | Python binding: the runtime in process, `Krite(model).decide(request)`. |
| `krite-bench` | PyPI | Benchmark harness for any `/v1/systemone` engine. |
| `krite-train` | PyPI | Training, export, and study code (torch). |
| `krite-0.15b-v1` | Hugging Face | Model directory for `krite serve --model`. |

## Documentation

- [ARCHITECTURE.md](https://github.com/skyoo2003/krite/blob/main/ARCHITECTURE.md): terms and model design
- [docs/runtime.md](https://github.com/skyoo2003/krite/blob/main/docs/runtime.md): runtime, caches, limits, measurements
- [docs/benchmark-spec.md](https://github.com/skyoo2003/krite/blob/main/docs/benchmark-spec.md) and [docs/baselines.md](https://github.com/skyoo2003/krite/blob/main/docs/baselines.md): how engines are measured
- [docs/training-data.md](https://github.com/skyoo2003/krite/blob/main/docs/training-data.md): training sources, licenses, release rules

## License

Apache-2.0. The Candle encoder code is adapted from candle-transformers (MIT OR Apache-2.0).
