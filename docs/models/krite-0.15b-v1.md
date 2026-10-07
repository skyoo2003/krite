---
license: apache-2.0
base_model: jhu-clsp/mmBERT-small
language: [en, ko, ja, zh, de, es, fr, ar, hi, ru]
pipeline_tag: zero-shot-classification
tags: [krite, decision-model, calibration, late-interaction, candle]
datasets: [mteb/banking77, clinc/clinc_oos, mteb/amazon_massive_intent, fancyzhx/dbpedia_14, google/boolq, stanfordnlp/snli, google/civil_comments, Davlan/sib200, nyu-mll/multi_nli, google-research-datasets/paws-x, google-research-datasets/go_emotions]
---

# krite-0.15b-v1

**Pre-release. This model passes every Krite release gate except accuracy: macro accuracy 0.783
against a target of 0.795, and macro QWK 0.273 against 0.357 (target: the better of Laya and cbjev,
minus 0.02). The largest per-suite gaps are agnews (0.28), sst5 (0.22), and boolq (0.14).**

Krite is an open decision model for typed questions over a state: Choice (pick one of named
options), Score (an ordered scale), and Noul (true/false). It returns a calibrated probability per
candidate and never generates text. It is served by the Rust runtime
([github.com/skyoo2003/krite](https://github.com/skyoo2003/krite)) over Protocol v1
(`POST /v1/systemone`).

## Model

- **Encoder**: mmBERT-small (22 layers, hidden size 384), fine-tuned. ≈ 140M parameters.
- **Late interaction (`late8`)**: the lower 14 layers encode the state once, without seeing any
  question; the runtime caches the state's keys and values across requests. Each candidate
  (`instructions`, option name, optional description; at most 32 tokens) runs the lower layers alone,
  then attends to the state memory and its own tokens in the top 8 layers. Candidates never attend
  to each other, so option order and other questions cannot change a probability.
- **Scorer**: masked mean + question-type embedding → MLP energy; softmax per question.
- **Calibration**: one temperature per bucket (below).

Details: [ARCHITECTURE.md](https://github.com/skyoo2003/krite/blob/main/ARCHITECTURE.md),
[docs/runtime.md](https://github.com/skyoo2003/krite/blob/main/docs/runtime.md).

## Use

This is not a `transformers` model: the weights use the Krite runtime's key names.

```bash
cargo install krite-cli
hf download skyoo2003/krite-0.15b-v1 --local-dir krite-0.15b-v1
krite serve --model krite-0.15b-v1        # 127.0.0.1:8110; Metal when available, else CPU

curl -s http://127.0.0.1:8110/v1/systemone -H 'content-type: application/json' -d '{
  "state": "I was charged twice for my subscription this month. Please refund one of the payments.",
  "questions": {"route": {"type": "choice", "instructions": "Which team should handle this?",
    "criteria": {"billing": "payments, refunds, invoices", "shipping": "delivery and tracking",
                 "technical": "bugs, login, app errors"}}}}'
```

## Files

| File | Content |
|---|---|
| `model.safetensors` | weights, fp32 |
| `config.json`, `tokenizer.json` | `jhu-clsp/mmBERT-small` at `abc32620dd4f6ab06f5fbe905dc25f310618e09f` |
| `krite.json` | model id, interaction layers, candidate token limits, per-bucket temperatures, training-set hash and seed |
| `probe.json` | torch CPU energies on synthetic text; the runtime's weight tests compare against it |

## Release gate

Measured on a MacBook Air 13 (Apple M4, 16 GB, fanless), AC power, Candle Metal, fp32, HTTP layer,
512-token state, 4 options per question. Latency numbers are not comparable with GPU results.

| Gate | Value | Limit | Pass |
|---|---|---|---|
| Accuracy (macro over 5 choice and noul datasets) | 0.783 | ≥ 0.795 | no |
| QWK (macro over 2 score datasets) | 0.273 | ≥ 0.357 | no |
| ECE after calibration (mean over 28 suites) | 0.057 | ≤ 0.071 | yes |
| Warm latency, 1 question, p50 | 7.85 ms | ≤ 10 ms | yes |
| Cold latency, 1 question, p50 | 90.1 ms | ≤ 210 ms | yes |
| Throughput, 30 questions, warm | 211.7 decisions/s | ≥ 175 | yes |
| Option-order flip rate (all permutations) | 0 | 0 | yes |
| Option-order max probability deviation | 0 | ≤ 1e-5 | yes |
| Question interference max deviation | 1.7e-7 | ≤ 1e-5 | yes |
| Cache on vs. off max deviation | 0 | ≤ 1e-5 | yes |

Per-suite results and baselines (Laya, Kev, SemIf, cbjev):
[docs/training-data.md](https://github.com/skyoo2003/krite/blob/main/docs/training-data.md),
[docs/baselines.md](https://github.com/skyoo2003/krite/blob/main/docs/baselines.md).

## Calibration

Temperatures were fitted on the uncalibrated model's probabilities on half of each evaluation suite
and scored on the other half.

| Bucket | Temperature |
|---|---|
| `choice/4` | 2.60 |
| `choice/5-8` | 1.36 |
| `choice/9+` | 1.31 |
| `noul` | 1.47 |
| `score/5` | 2.49 |

## Training data

The `broad` mixture: permissively licensed train splits, each pinned to a Hugging Face revision.
Rows whose state appears in any evaluation suite are dropped. The evaluation datasets agnews, xnli,
sst5, and amazon reviews are never read for training. One epoch, cross-entropy, seed 13; training
set sha256 `572b832568c39c9dd81c2e3af91c496fda9384ba3534a428cbdc9b1a8c9cf76a`.

| Source | Repo @ revision | License |
|---|---|---|
| banking77 | `mteb/banking77` @ `18072d26` | CC-BY-4.0 |
| clinc | `clinc/clinc_oos` @ `155b9c71` | CC-BY-3.0 |
| massive | `mteb/amazon_massive_intent` @ `940fd47a` | CC-BY-4.0 |
| dbpedia | `fancyzhx/dbpedia_14` @ `9abd46cf` | CC-BY-SA-3.0 |
| boolq | `google/boolq` @ `35b264d0` | CC-BY-SA-3.0 |
| snli | `stanfordnlp/snli` @ `cdb5c3d5` | CC-BY-SA-4.0 |
| civil | `google/civil_comments` @ `f2970eb3` | CC0-1.0 |
| sib200 | `Davlan/sib200` @ `38977a66` | CC-BY-SA-4.0 |
| mnli | `nyu-mll/multi_nli` @ `da70db2a` | OANC, CC-BY-3.0, CC-BY-SA-3.0 (per genre) |
| pawsx | `google-research-datasets/paws-x` @ `4cd8187c` | PAWS-X terms (free use for any purpose) |
| goemotions | `google-research-datasets/go_emotions` @ `add49224` | Apache-2.0 |

Views (re-asking a source as another question type) and augmentations (state formatting,
irrelevant candidates) are described in
[docs/training-data.md](https://github.com/skyoo2003/krite/blob/main/docs/training-data.md).

## Known limitations

- Accuracy is below the release target (see the first paragraph). Topic classification of news,
  reading-comprehension yes/no questions, and fine-grained sentiment scales are the weakest suites.
- `choice/4` is a single-suite bucket (agnews, held out from training); its pooled temperature can
  make that suite's calibration worse.
- On Candle, per-candidate work grows faster than on torch, so warm latency with many questions is
  higher than the same model on torch MPS.
- fp32 only; no quantized variants.
- Instructions in training are English; states are multilingual (10 languages above).

## License

The weights are released under Apache-2.0. The base model, mmBERT-small, is MIT. Training sources
and their licenses are listed above; several are CC-BY-SA.
