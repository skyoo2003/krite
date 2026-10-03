# Baselines

Five baseline engines measured on one MacBook Air 13 (Apple M4, 16 GB, fanless) under
[benchmark-spec](benchmark-spec.md), with the harness in [`benchmarks/`](../benchmarks) and the
engines described in [`benchmarks/baselines/`](../benchmarks/baselines/README.md). Raw results are
in [`benchmarks/results/baselines/`](../benchmarks/results/baselines). Every engine is measured through
`POST /v1/systemone` only, so `krite serve` will be measured the same way.

Read the tables with these caveats:

- Laya's English suites use `convaiinnovations/laya`; its non-English suites use
  `convaiinnovations/laya-multilingual`. cbjev routes between its own English and multilingual
  weights. Kev and SemIf are single multilingual LLMs.
- † The classifier is mmBERT-small fine-tuned on each dataset's train split. It is a
  fixed-label upper reference, not a peer: every other engine is zero-shot on these suites.
- Instructions and criteria are English in every suite; only the state is in the suite language.
- Model-layer latency is not measurable for black-box engines; only `runtime` (engine-reported)
  and `http` layers appear (spec §2). cbjev reports no runtime latency.

- Latency ran on AC, one engine at a time on an otherwise idle machine, with the same sequence
  for every engine: fresh start (startup), cold cells, warm cells, then 10 sustained minutes.
  Running all four engines at once inflated cold latency up to 2.5× (memory contention on 16 GB),
  so per-call interleaving was dropped.
- The fanless M4 Air slows down under continuous load without any `pmset -g therm` warning. Cold
  cells run first on a rested machine and warm cells after them, so Laya's and cbjev's warm
  1-question p50 is above their cold p50. A check with six different fixed states and with
  unique-vs-repeated states showed no state dependence (Laya 306–318 ms either way), so this is
  heat, not a measurement artifact. Compare engines within one cache state, not cold vs. warm
  for engines without a state cache.
- Startup is `serve.sh` wall time from process launch to the first 200 on a one-question noul
  ping probe (polled every 0.2 s), not a primary-cell call.

## Findings

1. **Quality.** Kev-0.8B is the strongest zero-shot engine, most clearly outside English: mean
   accuracy on non-English choice suites is 0.81 (Laya 0.60, cbjev 0.62, SemIf 0.49), and it is the
   only zero-shot engine with useful score predictions (mean QWK 0.80 English, 0.77 non-English).
   Of the encoders under 0.45B, cbjev matches or beats Laya on most suites; both are strong on
   English topic and yes/no suites. The fixed-label classifier† is best on most MASSIVE and Amazon suites,
   as expected for a model trained on each dataset's train split, but not on boolq or agnews.
2. **Score-level names matter.** On `amazon-score-*`, Laya, cbjev, and SemIf put 89–98% of their
   argmax on the lowest one or two of the five levels (`1 star` … `5 stars`) while gold labels are
   uniform, so QWK is about 0. The same engines and the same response mapping work on `sst5-score`,
   so this is sensitivity to criterion names, not a harness error.
3. **Calibration.** Raw ECE (mean over suites): Laya 0.257, cbjev 0.201, SemIf 0.477, Kev 0.087.
   One temperature per engine and calibration bucket, shared by every suite in the bucket, brings
   Kev, cbjev, and Laya to 0.071–0.079; SemIf stays at 0.127. Raw probabilities from these engines are not usable as thresholds without a fitted
   calibrator.
4. **Option order.** No engine is permutation-invariant. Full-permutation flip rate on 8-option
   MASSIVE suites: Laya 0.20–0.25, SemIf 0.49–0.57, cbjev 0.08–0.15, Kev 0.06–0.09. cbjev's
   reverse-order flip rate is exactly 0 on every suite while its full-permutation flip rate is
   not, so a reverse-only metric understates order dependence.
5. **Question interference.** Adding questions to a request changes other answers' probabilities:
   max deviation cbjev 0.68, SemIf 0.15, Kev 0.03, Laya 0.002 (agnews target question with 15
   filler questions; cbjev's argmax changed in 6% of cases). SemIf scores the alone and the
   multi-question request with the same shared-prefix scorer, so its deviation is batching, not a
   scorer switch; it is identical with 3 and 15 fillers.
6. **Latency.** Primary cell (512 tokens, 1 question, 4 options), cold p50: Laya 210 ms, cbjev
   334 ms, Kev 367 ms, SemIf 739 ms. Kev is the only engine with a state cache: warm p50 52 ms
   burst and 54 ms over 10 sustained minutes. Laya's latency grows linearly with the number of
   questions (30 questions warm: 8.9 s); the engines that share one state pass grow sublinearly
   (Kev 855 ms, cbjev 1.9 s, SemIf 4.3 s). Measured alone, every engine's last-minute p50 is within
   1% of its 10-minute p50.
7. **Memory.** Peak phys_footprint is 3.4–4.3 GB for Laya, Kev, and cbjev and 11.0 GB for SemIf
   (5.1 GB before the shim switched every request to SemIf's shared-prefix scorer); RSS under-reports it by 2–30× because MPS and
   MLX allocations live in unified memory.

## Derived Krite targets

Measured on the same MacBook Air M4, on AC, one engine at a time. Krite is measured with the same
sequence (fresh start, cold, warm, sustained).

| Metric | Best baseline | Krite target |
|---|---|---|
| Cold p50, 512/1/4 | 210 ms (Laya) | ≤ 210 ms |
| Warm p50, 512/1/4 | 52 ms (Kev) | ≤ 10 ms (≥ 5× faster) |
| Warm decisions/sec, 512/30/4 | 35.1 (Kev) | ≥ 175 (≥ 5×) |
| Full-permutation flip rate | 0.003 (cbjev, agnews) | 0 within numeric tolerance |
| Question interference, max prob deviation | 0.002 (Laya) | 0 within numeric tolerance |
| Temperature-scaled ECE, mean over suites | 0.071 (Kev) | ≤ 0.071 |
| Accuracy per suite | best zero-shot engine ≤ 0.45B (Laya or cbjev) | no more than 2 points below |

## Measured

<!-- GENERATED:START -->

### Quality

| Suite | Metric | Laya | Kev-0.8B | SemIf/Qwen3-0.6B | cbjev | Classifier† |
|---|---|---|---|---|---|---|
| agnews-choice | accuracy / macro_f1 | 0.945 / 0.946 | 0.870 / 0.867 | 0.740 / 0.743 | 0.940 / 0.942 | 0.925 / 0.925 |
| banking77-choice | accuracy / macro_f1 | 0.812 / 0.800 | 0.945 / 0.940 | 0.580 / 0.533 | 0.855 / 0.846 | 0.978 / 0.977 |
| boolq-noul | accuracy / auroc | 0.845 / 0.918 | 0.775 / 0.870 | 0.580 / 0.610 | 0.823 / 0.922 | 0.705 / 0.765 |
| sst5-score | mae / qwk | 1.074 / 0.402 | 0.585 / 0.782 | 1.288 / 0.042 | 0.628 / 0.748 | 0.568 / 0.778 |
| massive-choice-en | accuracy / macro_f1 | 0.752 / 0.751 | 0.890 / 0.906 | 0.552 / 0.521 | 0.748 / 0.736 | 0.973 / 0.943 |
| massive-choice-ko | accuracy / macro_f1 | 0.615 / 0.577 | 0.855 / 0.826 | 0.482 / 0.407 | 0.613 / 0.586 | 0.757 / 0.680 |
| massive-choice-ja | accuracy / macro_f1 | 0.667 / 0.656 | 0.897 / 0.903 | 0.512 / 0.474 | 0.700 / 0.690 | 0.905 / 0.864 |
| massive-choice-zh-CN | accuracy / macro_f1 | 0.650 / 0.568 | 0.890 / 0.871 | 0.588 / 0.552 | 0.698 / 0.646 | 0.922 / 0.905 |
| massive-choice-de | accuracy / macro_f1 | 0.565 / 0.536 | 0.805 / 0.798 | 0.460 / 0.409 | 0.525 / 0.534 | 0.870 / 0.815 |
| massive-choice-es | accuracy / macro_f1 | 0.585 / 0.559 | 0.807 / 0.816 | 0.517 / 0.465 | 0.625 / 0.615 | 0.912 / 0.912 |
| massive-choice-fr | accuracy / macro_f1 | 0.642 / 0.613 | 0.833 / 0.847 | 0.560 / 0.465 | 0.652 / 0.676 | 0.920 / 0.887 |
| massive-choice-ar | accuracy / macro_f1 | 0.527 / 0.458 | 0.725 / 0.653 | 0.355 / 0.315 | 0.565 / 0.523 | 0.745 / 0.630 |
| massive-choice-hi | accuracy / macro_f1 | 0.512 / 0.418 | 0.625 / 0.531 | 0.400 / 0.294 | 0.585 / 0.497 | 0.770 / 0.598 |
| massive-choice-ru | accuracy / macro_f1 | 0.627 / 0.605 | 0.858 / 0.826 | 0.522 / 0.500 | 0.640 / 0.611 | 0.900 / 0.881 |
| xnli-noul-en | accuracy / auroc | 0.627 / 0.622 | 0.940 / 0.988 | 0.510 / 0.674 | 0.922 / 0.996 | 0.938 / 0.974 |
| xnli-noul-de | accuracy / auroc | 0.568 / 0.698 | 0.823 / 0.926 | 0.512 / 0.612 | 0.807 / 0.949 | 0.825 / 0.925 |
| xnli-noul-es | accuracy / auroc | 0.608 / 0.672 | 0.897 / 0.958 | 0.460 / 0.656 | 0.823 / 0.944 | 0.855 / 0.932 |
| xnli-noul-fr | accuracy / auroc | 0.578 / 0.623 | 0.880 / 0.954 | 0.505 / 0.614 | 0.830 / 0.948 | 0.880 / 0.944 |
| xnli-noul-ar | accuracy / auroc | 0.552 / 0.618 | 0.823 / 0.920 | 0.497 / 0.576 | 0.792 / 0.899 | 0.755 / 0.902 |
| xnli-noul-hi | accuracy / auroc | 0.540 / 0.602 | 0.655 / 0.779 | 0.542 / 0.505 | 0.760 / 0.910 | 0.700 / 0.871 |
| xnli-noul-ru | accuracy / auroc | 0.590 / 0.650 | 0.882 / 0.949 | 0.495 / 0.617 | 0.815 / 0.925 | 0.772 / 0.923 |
| xnli-noul-zh | accuracy / auroc | 0.575 / 0.627 | 0.865 / 0.940 | 0.475 / 0.582 | 0.818 / 0.921 | 0.765 / 0.919 |
| amazon-score-en | mae / qwk | 1.513 / 0.097 | 0.603 / 0.824 | 1.806 / 0.016 | 1.429 / 0.033 | 0.493 / 0.832 |
| amazon-score-de | mae / qwk | 1.596 / 0.046 | 0.672 / 0.819 | 1.665 / -0.112 | 1.784 / 0.008 | 0.515 / 0.848 |
| amazon-score-es | mae / qwk | 1.640 / 0.058 | 0.750 / 0.775 | 1.523 / 0.082 | 1.815 / 0.003 | 0.587 / 0.835 |
| amazon-score-fr | mae / qwk | 1.633 / 0.038 | 0.708 / 0.779 | 1.484 / 0.066 | 1.852 / -0.005 | 0.505 / 0.872 |
| amazon-score-ja | mae / qwk | 1.786 / 0.051 | 0.762 / 0.729 | 1.719 / -0.037 | 2.003 / -0.001 | 0.619 / 0.789 |
| amazon-score-zh | mae / qwk | 1.736 / 0.037 | 0.735 / 0.748 | 1.652 / -0.032 | 1.961 / -0.003 | 0.612 / 0.778 |

### Calibration: ECE raw → temperature-scaled

Evaluation half of each suite. One temperature per engine model and calibration bucket, fit on the pooled other halves of every suite in that bucket (benchmark-spec §9); the classifier has one model, and so one calibrator, per dataset.

| Suite | Laya | Kev-0.8B | SemIf/Qwen3-0.6B | cbjev | Classifier† |
|---|---|---|---|---|---|
| agnews-choice | 0.021 → 0.015 | 0.090 → 0.066 | 0.254 → 0.105 | 0.040 → 0.033 | 0.045 → 0.028 |
| banking77-choice | 0.103 → 0.087 | 0.069 → 0.038 | 0.287 → 0.109 | 0.095 → 0.083 | 0.018 → 0.023 |
| boolq-noul | 0.069 → 0.292 | 0.081 → 0.074 | 0.410 → 0.086 | 0.050 → 0.045 | 0.065 → 0.060 |
| sst5-score | 0.385 → 0.079 | 0.076 → 0.172 | 0.729 → 0.048 | 0.114 → 0.289 | 0.064 → 0.073 |
| massive-choice-en | 0.047 → 0.054 | 0.101 → 0.048 | 0.346 → 0.113 | 0.187 → 0.124 | 0.027 → 0.021 |
| massive-choice-ko | 0.100 → 0.118 | 0.075 → 0.053 | 0.396 → 0.154 | 0.101 → 0.087 | 0.105 → 0.105 |
| massive-choice-ja | 0.138 → 0.081 | 0.099 → 0.060 | 0.398 → 0.186 | 0.109 → 0.091 | 0.038 → 0.074 |
| massive-choice-zh-CN | 0.128 → 0.097 | 0.085 → 0.059 | 0.348 → 0.175 | 0.077 → 0.086 | 0.051 → 0.053 |
| massive-choice-de | 0.189 → 0.109 | 0.085 → 0.069 | 0.452 → 0.131 | 0.064 → 0.089 | 0.037 → 0.034 |
| massive-choice-es | 0.130 → 0.099 | 0.068 → 0.054 | 0.364 → 0.148 | 0.141 → 0.124 | 0.042 → 0.032 |
| massive-choice-fr | 0.129 → 0.083 | 0.073 → 0.052 | 0.321 → 0.141 | 0.100 → 0.103 | 0.032 → 0.055 |
| massive-choice-ar | 0.166 → 0.107 | 0.105 → 0.083 | 0.472 → 0.158 | 0.092 → 0.115 | 0.074 → 0.057 |
| massive-choice-hi | 0.179 → 0.080 | 0.095 → 0.080 | 0.430 → 0.109 | 0.091 → 0.125 | 0.086 → 0.068 |
| massive-choice-ru | 0.134 → 0.121 | 0.088 → 0.045 | 0.355 → 0.146 | 0.083 → 0.064 | 0.039 → 0.048 |
| xnli-noul-en | 0.345 → 0.109 | 0.096 → 0.074 | 0.465 → 0.143 | 0.037 → 0.048 | 0.047 → 0.110 |
| xnli-noul-de | 0.392 → 0.085 | 0.065 → 0.068 | 0.475 → 0.142 | 0.104 → 0.078 | 0.099 → 0.069 |
| xnli-noul-es | 0.394 → 0.016 | 0.077 → 0.077 | 0.505 → 0.174 | 0.124 → 0.084 | 0.080 → 0.073 |
| xnli-noul-fr | 0.378 → 0.046 | 0.072 → 0.052 | 0.515 → 0.174 | 0.074 → 0.042 | 0.074 → 0.095 |
| xnli-noul-ar | 0.353 → 0.050 | 0.054 → 0.040 | 0.560 → 0.229 | 0.072 → 0.048 | 0.144 → 0.053 |
| xnli-noul-hi | 0.430 → 0.031 | 0.057 → 0.082 | 0.480 → 0.145 | 0.079 → 0.025 | 0.163 → 0.073 |
| xnli-noul-ru | 0.350 → 0.073 | 0.080 → 0.056 | 0.520 → 0.191 | 0.125 → 0.083 | 0.145 → 0.090 |
| xnli-noul-zh | 0.337 → 0.085 | 0.043 → 0.042 | 0.565 → 0.236 | 0.080 → 0.043 | 0.176 → 0.092 |
| amazon-score-en | 0.475 → 0.133 | 0.121 → 0.084 | 0.698 → 0.084 | 0.376 → 0.005 | 0.050 → 0.050 |
| amazon-score-de | 0.377 → 0.012 | 0.141 → 0.103 | 0.617 → 0.044 | 0.612 → 0.021 | 0.069 → 0.072 |
| amazon-score-es | 0.298 → 0.037 | 0.119 → 0.090 | 0.549 → 0.024 | 0.620 → 0.005 | 0.077 → 0.086 |
| amazon-score-fr | 0.351 → 0.022 | 0.081 → 0.089 | 0.572 → 0.043 | 0.637 → 0.013 | 0.136 → 0.137 |
| amazon-score-ja | 0.358 → 0.014 | 0.100 → 0.073 | 0.616 → 0.057 | 0.632 → 0.029 | 0.082 → 0.080 |
| amazon-score-zh | 0.430 → 0.089 | 0.125 → 0.111 | 0.649 → 0.064 | 0.699 → 0.066 | 0.102 → 0.097 |
| **mean over suites** | 0.257 → 0.079 | 0.087 → 0.071 | 0.477 → 0.127 | 0.201 → 0.073 | 0.077 → 0.068 |

### Calibration: NLL and Brier, raw → scaled

| Suite | Laya | Kev-0.8B | SemIf/Qwen3-0.6B | cbjev | Classifier† |
|---|---|---|---|---|---|
| agnews-choice | 0.188 → 0.187 · B 0.094 → 0.094 | 0.373 → 0.347 · B 0.200 → 0.188 | 2.148 → 0.781 · B 0.517 → 0.421 | 0.213 → 0.205 · B 0.098 → 0.097 | 0.236 → 0.216 · B 0.107 → 0.106 |
| banking77-choice | 1.474 → 0.953 · B 0.291 → 0.284 | 0.254 → 0.236 · B 0.107 → 0.098 | 2.681 → 1.520 · B 0.677 → 0.578 | 0.603 → 0.591 · B 0.239 → 0.240 | 0.106 → 0.113 · B 0.038 → 0.038 |
| boolq-noul | 0.350 → 0.590 · B 0.220 → 0.402 | 0.468 → 0.489 · B 0.297 → 0.303 | 6.006 → 0.692 · B 0.820 → 0.497 | 0.367 → 0.362 · B 0.235 → 0.229 | 0.557 → 0.556 · B 0.374 → 0.374 |
| sst5-score | 2.160 → 1.515 · B 0.980 → 0.763 | 1.059 → 1.129 · B 0.610 → 0.644 | 6.140 → 1.583 · B 1.464 → 0.789 | 1.088 → 1.549 · B 0.622 → 0.776 | 1.001 → 1.008 · B 0.570 → 0.571 |
| massive-choice-en | 0.676 → 0.674 · B 0.304 → 0.303 | 0.439 → 0.405 · B 0.210 → 0.199 | 3.148 → 1.414 · B 0.770 → 0.597 | 0.803 → 0.737 · B 0.355 → 0.329 | 0.141 → 0.136 · B 0.057 → 0.056 |
| massive-choice-ko | 1.222 → 1.159 · B 0.526 → 0.525 | 0.486 → 0.463 · B 0.230 → 0.227 | 3.529 → 1.561 · B 0.873 → 0.645 | 1.136 → 1.136 · B 0.527 → 0.528 | 0.746 → 0.715 · B 0.328 → 0.317 |
| massive-choice-ja | 1.173 → 1.035 · B 0.466 → 0.448 | 0.348 → 0.318 · B 0.160 → 0.153 | 3.098 → 1.384 · B 0.832 → 0.587 | 0.902 → 0.891 · B 0.412 → 0.411 | 0.255 → 0.266 · B 0.121 → 0.123 |
| massive-choice-zh-CN | 1.224 → 1.102 · B 0.497 → 0.493 | 0.408 → 0.375 · B 0.191 → 0.181 | 3.095 → 1.367 · B 0.759 → 0.572 | 1.003 → 0.995 · B 0.471 → 0.468 | 0.258 → 0.262 · B 0.120 → 0.123 |
| massive-choice-de | 1.969 → 1.562 · B 0.675 → 0.633 | 0.611 → 0.582 · B 0.292 → 0.286 | 4.117 → 1.674 · B 0.983 → 0.701 | 1.349 → 1.362 · B 0.610 → 0.617 | 0.368 → 0.371 · B 0.182 → 0.182 |
| massive-choice-es | 1.238 → 1.174 · B 0.539 → 0.532 | 0.616 → 0.599 · B 0.293 → 0.291 | 3.164 → 1.399 · B 0.775 → 0.588 | 1.125 → 1.098 · B 0.505 → 0.491 | 0.292 → 0.289 · B 0.150 → 0.148 |
| massive-choice-fr | 1.396 → 1.205 · B 0.515 → 0.500 | 0.615 → 0.600 · B 0.287 → 0.285 | 2.851 → 1.347 · B 0.728 → 0.564 | 1.085 → 1.080 · B 0.470 → 0.467 | 0.229 → 0.241 · B 0.121 → 0.122 |
| massive-choice-ar | 1.469 → 1.304 · B 0.617 → 0.594 | 0.837 → 0.843 · B 0.361 → 0.355 | 4.114 → 1.753 · B 1.010 → 0.731 | 1.198 → 1.201 · B 0.540 → 0.539 | 0.866 → 0.803 · B 0.349 → 0.337 |
| massive-choice-hi | 1.563 → 1.313 · B 0.640 → 0.589 | 1.090 → 1.112 · B 0.474 → 0.480 | 4.104 → 1.713 · B 0.949 → 0.710 | 1.117 → 1.111 · B 0.528 → 0.528 | 0.848 → 0.789 · B 0.340 → 0.333 |
| massive-choice-ru | 1.352 → 1.187 · B 0.533 → 0.516 | 0.511 → 0.473 · B 0.232 → 0.224 | 3.163 → 1.429 · B 0.795 → 0.606 | 1.025 → 1.008 · B 0.471 → 0.468 | 0.314 → 0.317 · B 0.146 → 0.148 |
| xnli-noul-en | 2.681 → 0.708 · B 0.711 → 0.507 | 0.176 → 0.155 · B 0.088 → 0.081 | 6.343 → 0.723 · B 0.930 → 0.528 | 0.189 → 0.200 · B 0.110 → 0.113 | 0.240 → 0.288 · B 0.119 → 0.152 |
| xnli-noul-de | 1.709 → 0.678 · B 0.778 → 0.485 | 0.433 → 0.440 · B 0.280 → 0.283 | 6.540 → 0.732 · B 0.950 → 0.536 | 0.436 → 0.406 · B 0.285 → 0.266 | 0.523 → 0.442 · B 0.306 → 0.284 |
| xnli-noul-es | 1.887 → 0.683 · B 0.789 → 0.490 | 0.273 → 0.262 · B 0.165 → 0.163 | 6.976 → 0.752 · B 1.010 → 0.555 | 0.473 → 0.426 · B 0.302 → 0.279 | 0.400 → 0.379 · B 0.245 → 0.236 |
| xnli-noul-fr | 2.008 → 0.680 · B 0.770 → 0.487 | 0.314 → 0.309 · B 0.193 → 0.193 | 6.407 → 0.746 · B 1.030 → 0.551 | 0.431 → 0.401 · B 0.267 → 0.255 | 0.343 → 0.352 · B 0.200 → 0.212 |
| xnli-noul-ar | 1.778 → 0.681 · B 0.740 → 0.487 | 0.359 → 0.349 · B 0.223 → 0.220 | 7.824 → 0.794 · B 1.120 → 0.596 | 0.464 → 0.437 · B 0.278 → 0.273 | 0.639 → 0.491 · B 0.364 → 0.322 |
| xnli-noul-hi | 2.211 → 0.691 · B 0.840 → 0.497 | 0.611 → 0.618 · B 0.427 → 0.432 | 6.580 → 0.737 · B 0.960 → 0.541 | 0.503 → 0.486 · B 0.328 → 0.320 | 0.784 → 0.594 · B 0.445 → 0.399 |
| xnli-noul-ru | 1.597 → 0.672 · B 0.693 → 0.479 | 0.350 → 0.341 · B 0.216 → 0.212 | 7.387 → 0.768 · B 1.040 → 0.570 | 0.541 → 0.485 · B 0.342 → 0.320 | 0.729 → 0.517 · B 0.376 → 0.335 |
| xnli-noul-zh | 1.642 → 0.670 · B 0.682 → 0.476 | 0.325 → 0.316 · B 0.197 → 0.193 | 7.993 → 0.799 · B 1.130 → 0.601 | 0.463 → 0.433 · B 0.287 → 0.276 | 0.766 → 0.508 · B 0.381 → 0.329 |
| amazon-score-en | 3.273 → 1.625 · B 1.129 → 0.802 | 1.002 → 0.950 · B 0.530 → 0.512 | 5.048 → 1.601 · B 1.405 → 0.796 | 1.990 → 1.586 · B 0.944 → 0.790 | 0.839 → 0.839 · B 0.473 → 0.472 |
| amazon-score-de | 5.217 → 1.634 · B 1.119 → 0.810 | 1.084 → 1.033 · B 0.578 → 0.561 | 4.443 → 1.624 · B 1.335 → 0.806 | 3.665 → 1.617 · B 1.306 → 0.803 | 0.889 → 0.890 · B 0.488 → 0.488 |
| amazon-score-es | 4.783 → 1.606 · B 1.046 → 0.798 | 1.169 → 1.136 · B 0.626 → 0.618 | 3.716 → 1.612 · B 1.245 → 0.801 | 3.626 → 1.602 · B 1.312 → 0.797 | 0.983 → 0.983 · B 0.548 → 0.548 |
| amazon-score-fr | 5.185 → 1.630 · B 1.106 → 0.808 | 1.129 → 1.084 · B 0.608 → 0.596 | 3.775 → 1.620 · B 1.264 → 0.804 | 3.825 → 1.618 · B 1.360 → 0.804 | 0.861 → 0.862 · B 0.501 → 0.501 |
| amazon-score-ja | 5.095 → 1.619 · B 1.081 → 0.804 | 1.264 → 1.247 · B 0.657 → 0.649 | 4.307 → 1.603 · B 1.331 → 0.797 | 3.785 → 1.611 · B 1.332 → 0.801 | 1.095 → 1.094 · B 0.572 → 0.572 |
| amazon-score-zh | 5.422 → 1.636 · B 1.138 → 0.811 | 1.228 → 1.208 · B 0.646 → 0.640 | 5.026 → 1.632 · B 1.378 → 0.809 | 4.170 → 1.626 · B 1.432 → 0.807 | 1.060 → 1.060 · B 0.579 → 0.579 |

### Option-order invariance (100 cases per suite)

| Engine | Suite | Full-perm flip | Reverse flip | Max prob dev | Errors |
|---|---|---|---|---|---|
| Laya | agnews-choice | 0.014 | 0.010 | 0.3156 | 0% |
| Laya | massive-choice-en | 0.197 | 0.280 | 0.9129 | 0% |
| Laya | massive-choice-ko | 0.246 | 0.290 | 0.9937 | 0% |
| Kev-0.8B | agnews-choice | 0.008 | 0.010 | 0.2135 | 0% |
| Kev-0.8B | massive-choice-en | 0.060 | 0.050 | 0.3560 | 0% |
| Kev-0.8B | massive-choice-ko | 0.087 | 0.090 | 0.6073 | 0% |
| SemIf/Qwen3-0.6B | agnews-choice | 0.204 | 0.240 | 1.0000 | 0% |
| SemIf/Qwen3-0.6B | massive-choice-en | 0.487 | 0.670 | 1.0000 | 0% |
| SemIf/Qwen3-0.6B | massive-choice-ko | 0.571 | 0.690 | 0.9999 | 0% |
| cbjev | agnews-choice | 0.003 | 0.000 | 0.1179 | 0% |
| cbjev | massive-choice-en | 0.084 | 0.000 | 0.5224 | 0% |
| cbjev | massive-choice-ko | 0.155 | 0.000 | 0.4264 | 0% |

### Question interference (agnews-choice, 100 cases; alone vs. with fillers)

| Engine | Fillers | Argmax change | Max prob dev |
|---|---|---|---|
| Laya | 3 | 0.000 | 0.0000 |
| Laya | 15 | 0.000 | 0.0017 |
| Kev-0.8B | 3 | 0.000 | 0.0252 |
| Kev-0.8B | 15 | 0.010 | 0.0341 |
| SemIf/Qwen3-0.6B | 3 | 0.000 | 0.1548 |
| SemIf/Qwen3-0.6B | 15 | 0.000 | 0.1548 |
| cbjev | 3 | 0.020 | 0.6744 |
| cbjev | 15 | 0.060 | 0.6757 |

### Latency (MacBook Air M4, concurrency 1)

| Engine | Mode | Cache | State/Q/K | Layer | n | p50 ms | p95 ms | p99 ms | dec/s | Errors | Power | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cbjev | burst | cold | 512/1/4 | http | 200 | 334.0 | 359.1 | 362.9 | 3.0 | 0% | AC |  |
| Kev-0.8B | burst | cold | 512/1/4 | http | 200 | 367.1 | 398.4 | 402.0 | 2.7 | 0% | AC |  |
| Kev-0.8B | burst | cold | 512/1/4 | runtime | 200 | 365.5 | 396.9 | 400.4 | 2.7 | 0% | AC |  |
| Laya | burst | cold | 512/1/4 | http | 200 | 210.4 | 211.3 | 212.0 | 4.7 | 0% | AC |  |
| Laya | burst | cold | 512/1/4 | runtime | 200 | 210.1 | 210.9 | 211.7 | 4.7 | 0% | AC |  |
| SemIf/Qwen3-0.6B | burst | cold | 512/1/4 | http | 200 | 738.6 | 773.3 | 799.8 | 1.4 | 0% | AC |  |
| SemIf/Qwen3-0.6B | burst | cold | 512/1/4 | runtime | 200 | 738.1 | 772.8 | 799.2 | 1.4 | 0% | AC |  |
| cbjev | burst | cold | 512/10/4 | http | 200 | 869.2 | 892.0 | 915.1 | 11.5 | 0% | AC |  |
| Kev-0.8B | burst | cold | 512/10/4 | http | 200 | 682.2 | 693.1 | 700.6 | 14.7 | 0% | AC |  |
| Kev-0.8B | burst | cold | 512/10/4 | runtime | 200 | 679.2 | 690.0 | 697.1 | 14.7 | 0% | AC |  |
| Laya | burst | cold | 512/10/4 | http | 200 | 2396.7 | 2578.3 | 2602.6 | 4.3 | 0% | AC |  |
| Laya | burst | cold | 512/10/4 | runtime | 200 | 2396.0 | 2577.5 | 2602.0 | 4.3 | 0% | AC |  |
| SemIf/Qwen3-0.6B | burst | cold | 512/10/4 | http | 200 | 1880.4 | 1904.2 | 1918.6 | 5.3 | 0% | AC |  |
| SemIf/Qwen3-0.6B | burst | cold | 512/10/4 | runtime | 200 | 1879.5 | 1903.4 | 1917.7 | 5.3 | 0% | AC |  |
| cbjev | burst | cold | 512/30/4 | http | 200 | 1932.1 | 1969.5 | 2008.8 | 15.5 | 0% | AC |  |
| Kev-0.8B | burst | cold | 512/30/4 | http | 200 | 1243.5 | 1251.7 | 1255.9 | 24.2 | 0% | AC |  |
| Kev-0.8B | burst | cold | 512/30/4 | runtime | 200 | 1238.4 | 1246.7 | 1284.2 | 24.2 | 0% | AC |  |
| Laya | burst | cold | 512/30/4 | http | 50 | 8419.5 | 8638.8 | 8655.9 | 3.6 | 0% | AC | truncated n |
| Laya | burst | cold | 512/30/4 | runtime | 50 | 8418.7 | 8638.0 | 8655.0 | 3.6 | 0% | AC | truncated n |
| SemIf/Qwen3-0.6B | burst | cold | 512/30/4 | http | 50 | 4002.6 | 4267.4 | 4498.7 | 7.4 | 0% | AC | truncated n |
| SemIf/Qwen3-0.6B | burst | cold | 512/30/4 | runtime | 50 | 4001.7 | 4266.4 | 4496.0 | 7.4 | 0% | AC | truncated n |
| cbjev | burst | startup | ping probe | http | 1 | 7971.3 | 7971.3 | 7971.3 | 0.1 | 0% | AC |  |
| Kev-0.8B | burst | startup | ping probe | http | 1 | 6271.1 | 6271.1 | 6271.1 | 0.2 | 0% | AC |  |
| Laya | burst | startup | ping probe | http | 1 | 4407.5 | 4407.5 | 4407.5 | 0.2 | 0% | AC |  |
| SemIf/Qwen3-0.6B | burst | startup | ping probe | http | 1 | 5135.0 | 5135.0 | 5135.0 | 0.2 | 0% | AC |  |
| cbjev | burst | warm | 512/1/4 | http | 200 | 386.9 | 397.0 | 410.5 | 2.6 | 0% | AC |  |
| Kev-0.8B | burst | warm | 512/1/4 | http | 200 | 52.1 | 53.5 | 54.1 | 19.2 | 0% | AC |  |
| Kev-0.8B | burst | warm | 512/1/4 | runtime | 200 | 50.8 | 52.2 | 52.8 | 19.2 | 0% | AC |  |
| Laya | burst | warm | 512/1/4 | http | 200 | 320.0 | 326.2 | 334.0 | 3.1 | 0% | AC |  |
| Laya | burst | warm | 512/1/4 | runtime | 200 | 319.6 | 325.5 | 333.6 | 3.1 | 0% | AC |  |
| SemIf/Qwen3-0.6B | burst | warm | 512/1/4 | http | 200 | 677.8 | 706.3 | 757.5 | 1.5 | 0% | AC |  |
| SemIf/Qwen3-0.6B | burst | warm | 512/1/4 | runtime | 200 | 677.3 | 705.0 | 757.0 | 1.5 | 0% | AC |  |
| cbjev | burst | warm | 512/10/4 | http | 200 | 849.5 | 867.9 | 873.5 | 11.8 | 0% | AC |  |
| Kev-0.8B | burst | warm | 512/10/4 | http | 200 | 292.2 | 295.0 | 296.9 | 34.2 | 0% | AC |  |
| Kev-0.8B | burst | warm | 512/10/4 | runtime | 200 | 289.5 | 292.3 | 293.8 | 34.2 | 0% | AC |  |
| Laya | burst | warm | 512/10/4 | http | 200 | 2999.1 | 3241.6 | 4625.8 | 3.3 | 0% | AC |  |
| Laya | burst | warm | 512/10/4 | runtime | 200 | 2998.3 | 3240.8 | 4623.4 | 3.3 | 0% | AC |  |
| SemIf/Qwen3-0.6B | burst | warm | 512/10/4 | http | 200 | 1722.4 | 1872.2 | 2099.1 | 5.7 | 0% | AC |  |
| SemIf/Qwen3-0.6B | burst | warm | 512/10/4 | runtime | 200 | 1721.5 | 1871.4 | 2098.3 | 5.7 | 0% | AC |  |
| cbjev | burst | warm | 512/30/4 | http | 200 | 1936.0 | 1973.6 | 2070.2 | 15.5 | 0% | AC |  |
| Kev-0.8B | burst | warm | 512/30/4 | http | 200 | 855.3 | 861.6 | 863.3 | 35.1 | 0% | AC |  |
| Kev-0.8B | burst | warm | 512/30/4 | runtime | 200 | 850.2 | 856.4 | 858.2 | 35.1 | 0% | AC |  |
| Laya | burst | warm | 512/30/4 | http | 50 | 8865.6 | 9800.1 | 11353.6 | 3.3 | 0% | AC | truncated n |
| Laya | burst | warm | 512/30/4 | runtime | 50 | 8864.8 | 9799.3 | 11352.6 | 3.3 | 0% | AC | truncated n |
| SemIf/Qwen3-0.6B | burst | warm | 512/30/4 | http | 50 | 4265.4 | 4389.6 | 4443.6 | 7.0 | 0% | AC | truncated n |
| SemIf/Qwen3-0.6B | burst | warm | 512/30/4 | runtime | 50 | 4264.5 | 4388.8 | 4442.8 | 7.0 | 0% | AC | truncated n |
| cbjev | sustained | warm | 512/1/4 | http | 1558 | 384.0 | 396.1 | 409.8 | 2.6 | 0% | AC | last-minute p50 381.8 |
| Kev-0.8B | sustained | warm | 512/1/4 | http | 10951 | 54.4 | 57.0 | 58.7 | 18.4 | 0% | AC | last-minute p50 54.2 |
| Laya | sustained | warm | 512/1/4 | http | 1819 | 329.7 | 335.6 | 339.3 | 3.0 | 0% | AC | last-minute p50 332.0 |
| SemIf/Qwen3-0.6B | sustained | warm | 512/1/4 | http | 807 | 738.7 | 778.3 | 882.2 | 1.3 | 0% | AC | last-minute p50 732.7 |

### Peak memory during the quality run

| Engine | Model | phys_footprint GiB | RSS GiB |
|---|---|---|---|
| Laya | convaiinnovations/laya | 3.43 | 0.74 |
| Kev-0.8B | jaredpalmer/kev-0.8b | 3.55 | 0.12 |
| cbjev | 0010101010-1/cbjev | 4.27 | 0.29 |
| SemIf/Qwen3-0.6B | Qwen/Qwen3-0.6B | 11.00 | 4.91 |
| Laya (multilingual) | convaiinnovations/laya-multilingual | 1.97 | 0.77 |
| Classifier† | jhu-clsp/mmBERT-small | 5.53 | 2.14 |

<!-- GENERATED:END -->

## Quoted (not measured here)

Numbers from each project's README on other hardware. They are not comparable with the measured
tables above and are never mixed into them (spec §13).

| Engine | Claim | Hardware | Source |
|---|---|---|---|
| cbjev | 1 question: 3.0 ms; 10 questions / 500-token state: 11.4 ms; 30 questions: 31.4 ms | RTX 4090 | [cbjev BENCHMARKS](https://github.com/tomek7667/cbjev/blob/master/BENCHMARKS.md) |
| cbjev | Option-reversal flip rate: Jev 13%, Laya 7.8%, cbjev 0.2% | — | [cbjev README](https://github.com/tomek7667/cbjev) |
| Laya | 1 question p50 32.8 ms | Tesla T4 | [Laya README](https://github.com/he-jev/laya) |
| Laya | ECE 0.466 raw → 0.081 after temperature refit | — | [Laya README](https://github.com/he-jev/laya) |
| Kev-0.8B | 5 questions: 149 ms new text, 28 ms cached text | Apple M5, 32 GB | [Kev README](https://github.com/jaredpalmer/kev) |
| Kev-4B | 721 ms new text, 136 ms cached | Apple M5 | [Kev README](https://github.com/jaredpalmer/kev) |
| Kev-4B | 6 questions: 18.1 ms (short text), 89.4 ms (2.2k tokens) | H100 | [Kev README](https://github.com/jaredpalmer/kev) |

## Protocol compatibility findings

- Kev and cbjev, both described as Jev wire-compatible, key score `probabilities` by level index
  (`"0"`, `"1"`, …) instead of level name. Krite Protocol v1 uses level names. If the official
  SDK confirms the index form, v1 should adopt it.
- Kev returns nonzero `usage.output_tokens`; Krite fixes it at 0.
- cbjev omits `latency_ms`, adds a top-level `routing` field, and adds `confidence` to noul answers.
- The harness maps this dialect onto Protocol v1 before validation (`client.from_jev`); see
  [protocol open items](protocol/v1.md#8-open-items-confirm-with-the-official-jev-sdk).
