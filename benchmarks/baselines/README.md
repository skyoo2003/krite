# Baseline engines

Every engine runs as its own process in its own environment and is measured only through
`POST /v1/systemone` (Protocol v1). The harness (`krite-bench`) contains no ML code, so engine
dependency pins never conflict, and `krite serve` will be measured the same way.

```bash
benchmarks/baselines/setup.sh <engine>     # isolated env under .venvs/<engine> (gitignored)
benchmarks/baselines/serve.sh <engine>     # start on its port; waits for the first 200; prints the PID
```

| Engine | Model | How it is served | Port | Device | License |
|---|---|---|---|---|---|
| `laya` | `convaiinnovations/laya` (421M) | `shims/laya_shim.py` over `laya.Agent.predict` | 8101 | torch MPS | Apache-2.0 |
| `laya-ml` | `convaiinnovations/laya-multilingual` (322M) | same shim, same env as `laya`; non-English suites | 8106 | torch MPS | Apache-2.0 |
| `kev` | `jaredpalmer/kev-0.8b` | Kev's own server (`python -m kev.serve`) | 8102 | MLX | Apache-2.0 |
| `semif` | `Qwen/Qwen3-0.6B` | `shims/semif_shim.py` over SemIf's shared-prefix scorer | 8103 | torch MPS, bf16 | MIT |
| `cbjev` | `0010101010-1/cbjev` | its own server (`cbjev-serve`), black box | 8000 | `CBJEV_DEVICE=mps` | **GPL-3.0** |
| `classifier` | `jhu-clsp/mmBERT-small` + per-dataset head | `shims/classifier_shim.py`; trained by `classifier_train.py` | 8105 | torch MPS | MIT (base) |
| `fake`, `fake-biased` | none | `shims/fake_shim.py` (tests) | 8199, 8198 | — | — |

Exact versions are in `engines.toml` (git commits, Hugging Face revisions) and in each
`.venvs/<engine>/INSTALLED.json` (full `pip freeze`), which the environment record copies.

## Rules

- **cbjev is GPL-3.0; Krite is Apache-2.0.** cbjev is installed into its own environment and used
  only as a black-box server process. Never open, read, copy, or vendor its source, including
  when it fails. If it cannot run, record `not_runnable` in `results/baselines/status.jsonl` and cite its
  README numbers in the quoted table only.
- Shims add nothing to the response body beyond Protocol v1; extras go in headers
  (`x-bench-device`). Engines' own probabilities are used as-is, never recomputed.
- Kev and cbjev answer in a Jev-style dialect (score probabilities keyed by level index,
  nonzero `output_tokens`, missing `latency_ms`, extra fields). `dialect = "jev"` in
  `engines.toml` maps it onto Protocol v1 before schema validation (`client.from_jev`).
- The classifier sees each dataset's train split (minus any text that appears in a test case).
  It is a fixed-label upper reference, not a peer of the zero-shot engines.

## Deviations from upstream defaults

- **SemIf**: its MLX backend accepts only Qwen3.5 checkpoints, and its torch loader requires CUDA.
  The shim loads Qwen3-0.6B on MPS with SemIf's settings (bf16, no remote code) and calls
  SemIf's device-agnostic `shared.score_shared` unchanged for every request, one question or many,
  so single-question and multi-question answers come from one scorer (states with no shared token
  prefix fall back to `direct.score` for all questions). SemIf supports
  2–16 options; other counts return `too_many_options`.
- **cbjev**: routes between English and multilingual weights itself, so there is no `cbjev-ml`.
