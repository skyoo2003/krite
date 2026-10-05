# Krite Architecture

## 1. Overview

Krite is an open decision model that judges without generating text. It encodes the state once, independently of any question, and caches it across requests. It then scores the candidates of runtime-defined Choice/Score/Noul questions with one shared scorer. The output is a calibrated probability distribution over the candidates, and the wire format is Jev-compatible ([Protocol v1](docs/protocol/v1.md)). The model is multilingual: state, instructions, and criteria may be in any supported language.

```text
State ──► State Encoder ──► State Memory (H_state) ──┐   cache key:
          (never sees questions)                     │   (model_id, tokenizer_version,
                                                     │    sha256(canonical_state_bytes))
Question q + Criterion c_i ──► Decision Tower ◄──────┘
                               (cross-attention to H_state; no attention between questions)
                                     │
                          shared energy head: e_i = f(H_state, q, c_i)
                                     │
                                  softmax
                                     │
                                calibration
                                     │
                        Choice / Score / Noul answer
```

**Decision tower.** The [architecture study](docs/architecture-study.md) accepted late interaction inside the encoder (`late8`) over the joint Laya-style encoder. Each candidate is the text `"{instructions}\n{name}[: {desc}]"`, tokenized with bos/eos and cut to 32 tokens. It runs the lower encoder layers (0–13 of mmBERT-small's 22) alone. In the top 8 layers its tokens attend to their own tokens and to the state's keys and values at that layer, with the pretrained attention weights and RoPE positions continuing after the state. A masked mean over the candidate's tokens plus a question-type embedding goes to an MLP scorer that gives `e_i`. State Memory is therefore the state's rotated keys and values in the top 8 layers. The state is still encoded once and never sees questions. A candidate's lower-layer states depend only on its text, so they can be cached as well. Candidates never attend to each other, so isolation and option-order invariance hold by construction.

## 2. Glossary

In this document and everywhere under `docs/`, these terms have only the meanings defined here.

- **State**: The input being judged. A string or a JSON object. An object is treated as its RFC 8785 (JCS) canonical string. The state is encoded independently of questions.
- **State Memory** (`H_state`): The per-token hidden states of `Encoder(state)`. It never sees questions. It is cached across requests under the key `(model_id, tokenizer_version, sha256(canonical_state_bytes))`.
- **Question**: `{type, instructions, criteria}`, identified by an id within a request. Questions never attend to each other (isolation).
- **Criterion**: One candidate of a question: a Choice option, a Score level, or Noul's true/false. It has a name and an optional description.
- **Decision**: Krite's output for one Question: a calibrated probability distribution over the candidate set, plus the values derived from it (`choice`, `score`, `noul`).
- **Energy**: The raw scalar of candidate i, `e_i = f(H_state, q, c_i)`. The same `f` applies to every candidate.
- **Probability**: Each element of `p = Calibrate(softmax(e))`. It is the calibrated estimate of "this candidate is correct". The response fields `probabilities` and `noul` carry these values.
- **Confidence**: A summary of how certain the whole distribution is. It is not a probability. Definition: `confidence = 1 − H(p) / ln K` (K ≥ 2; 1.0 when K = 1). It is based on normalized entropy, so its range is [0, 1]. It is returned for Choice and Score only, never for Noul. Jev's confidence definition is not public, so numeric equivalence with Jev is not guaranteed.
- **Calibration**: A post-hoc transform that matches raw distributions to observed correctness frequencies. v1 uses `temperature[type][cardinality_bucket]`. Each model version (including each quantized variant) has its own calibrator. Choice/Score with K = 1 skip calibration (probability 1.0).

  | # | Bucket | # | Bucket |
  |---|---|---|---|
  | 1 | `choice/2` | 7 | `score/2` |
  | 2 | `choice/3` | 8 | `score/3` |
  | 3 | `choice/4` | 9 | `score/4` |
  | 4 | `choice/5-8` | 10 | `score/5` |
  | 5 | `choice/9+` | 11 | `score/6+` |
  | 6 | `noul` | | |

- **Cold / Warm**: Cold is a state cache miss (the encoder runs). Warm is a cache hit (only the decision tower runs). Same definition as in [benchmark-spec](docs/benchmark-spec.md).

## 3. Primitives → internal representation

Internally, all three primitives are "a distribution over a runtime-defined candidate set". Only Score adds ordinal meaning.

| Type | Candidate set | Output |
|---|---|---|
| choice | keys of the criteria object (K ≥ 1; server default limit 255) | `choice` = argmax name, `probabilities`, `confidence` |
| noul | `{true, false}` (K = 2) | `noul` = P(true) |
| score | levels 0..K−1 in criteria array order (K ≥ 1; server default limit 255) | `score` = Σ i·p_i, `legend` `{"0": level_0, …}`, `probabilities`, `confidence` |

Argmax ties are broken by **ascending Unicode codepoint order of the names**, not by criteria input order. This keeps the result order-invariant.

## 4. Architectural Invariants

Tolerance is 1e-5 for fp32 and 1e-3 for fp16/bf16 (absolute). Tests are written with the runtime and model code.

| ID | Invariant | How to verify |
|---|---|---|
| I1 | **Permutation equivariance**: Reordering Choice candidates does not change any candidate's probability. Trivial for Noul (fixed candidates). Excluded for Score (order is meaningful). | Compare per-candidate probabilities across permutations; max absolute difference ≤ tolerance |
| I2 | **Question isolation**: Adding, removing, or reordering questions in a request does not change the output of any other question. | Compare a question asked alone vs. inside the full question set |
| I3 | **State independence**: `H_state` does not depend on the question set, so cache hit and cache miss give the same output. | Run the same request cold and warm; compare outputs |
| I4 | **Question id irrelevance**: Question id strings are not model inputs. Decisions use only instructions and criteria. | Compare outputs of requests that differ only in question ids |
| I5 | **Criterion names are used**: Option names and descriptions are model inputs (names carry meaning). Position is not an input. | I1 confirms position independence; a tokenizer input dump confirms no position tokens |

The only things the output must be independent of are **candidate position** and **question id**. Candidate names do affect the output.

## 5. Versioning

Three independent axes:

- **Protocol**: major version in the URL path (`/v1`). Backward-compatible field additions only revise the documentation. Breaking changes go to `/v2`.
- **Runtime**: crate/package semver `0.x`.
- **Model**: `krite-<size>-v<N>[+<quant>]` (e.g. `krite-0.15b-v1`, `krite-0.15b-v1+int8`). One unit bundles weights, calibrator, and tokenizer. The response `model` field returns the exact id.

The state cache key includes the model id and tokenizer version, so replacing the model invalidates the cache automatically.

## 6. Non-goals

Text generation, Jev behavioral equivalence, and CUDA optimization are not goals. Scope and rationale follow the PRD.
