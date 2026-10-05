# Architecture Study

Does the Krite decision tower (state encoded once, candidates scored independently against it) keep
the quality of a joint encoder that reads question, options, and state together, while answering
repeated questions on the same state much faster? This study trains both designs and ten tower
variants from the same initialization on the same data, measures them with the baseline harness, and
applies a decision rule fixed before the runs. Terms follow [ARCHITECTURE.md](../ARCHITECTURE.md).

## Arms

Every arm fine-tunes the full mmBERT-small encoder (`jhu-clsp/mmBERT-small` @ `abc32620`), in fp32,
on the same examples in the same order with the same hyperparameters (except `arch-d1-lr`). Code: [`training/`](../training).

| Engine | Design | Per request |
|---|---|---|
| `arch-b` | **Joint** (Laya-style, adapted from Laya's Apache-2.0 head) | One encoder pass per question over `<bos> "<type> question: <instructions>" <sep> (<mask> option)×K <sep> state <eos>`, two transformer layers, an MLP scorer on each `<mask>` position. No state cache is possible, and option order is an input. |
| `arch-d1`, `arch-d2`, `arch-d4` | **Tower**, shared option encoder | State: `<bos> state <eos>` through the encoder once, cached as `H_state`. Each candidate `<bos> instructions \n name[: description] <eos>` goes through the same encoder, then 1, 2, or 4 tower layers (self-attention within the candidate, cross-attention to `H_state`, feed-forward), mean pooling, and an MLP scorer. |
| `arch-d2-emb` | **Tower**, embedding-only option encoder | As `arch-d2`, but candidates use only the encoder's token embeddings, so a warm request runs no encoder layer at all. |
| `arch-d2-nocache` | `arch-d2` with the state cache off | Same checkpoint; every request re-encodes the state. Checks that a cache hit and a miss give the same output. |
| `arch-d1-pool` | `arch-d1` + pooled state in the scorer (follow-up) | The scorer reads `[c, s, c ⊙ s]`: the pooled candidate `c`, a projection `s` of the mean of `H_state`, and their product. |
| `arch-d1-set` | `arch-d1` + set attention (follow-up) | One attention layer between the pooled candidates of a question (no positions, so still permutation-equivariant). |
| `arch-d1-lr` | `arch-d1`, longer (follow-up) | New-module learning rate 1e-3 instead of 3e-4 and 2 epochs: twice the training compute of every other arm. |
| `arch-late4` | **Late interaction** (follow-up) | No tower layers. The candidate runs encoder layers 0–17 alone; in layers 18–21 its tokens attend to its own tokens and to the state's hidden states at that layer, with the pretrained attention weights and RoPE positions continuing after the state. The state side never sees candidates, so its four layer inputs are the cached memory. Mean pooling and an MLP scorer. |
| `arch-late8` | Late interaction, deeper (second round) | As `arch-late4`, with the candidate attending to the state in layers 14–21. |
| `arch-late4-s14`, `arch-b-s14` | Second seed (second round) | `arch-late4` and `arch-b` trained with seed 14: a different sample of the same sources, batch order, and initialization of new modules. Not part of the decision rule. |
| `arch-late4-nocache` | Cache off (second round) | `arch-late4` with both the state cache and the candidate cache turned off. |
| `arch-late6` | Late interaction, 6 layers (third round) | As `arch-late4`, with the candidate attending to the state in layers 16–21. |
| `arch-late8-s14` | Second seed (third round) | `arch-late8` trained with seed 14. Not part of the decision rule. |
| `arch-late8-nocache` | Cache off (third round) | `arch-late8` with both caches turned off. |

A softmax over each question's candidates gives the probabilities. Training loss is cross-entropy.

Two other candidate designs are not trained here. Decoder logit scoring is represented by
SemIf/Qwen3-0.6B and a packed encoder by cbjev, both measured zero-shot in [baselines.md](baselines.md).

## Training

One mixture of permissively licensed train splits, built by `krite_train.data` (seed 13, 32,000
examples, sha256 `50496145fdff…` for every arm). Rows whose state appears in any evaluation suite are
dropped first. Training instructions are hand-written templates that never equal a suite's
instructions, so every suite is zero-shot in wording.

| Source | Repo @ revision | Type | n | Leaked rows dropped | License |
|---|---|---|---|---|---|
| banking77 | `mteb/banking77` @ `18072d26` | choice | 4,000 | 0 | CC-BY-4.0 |
| clinc | `clinc/clinc_oos` @ `155b9c71` (`plus`, `oos` dropped) | choice | 4,000 | 2 | CC-BY-3.0 |
| massive | `mteb/amazon_massive_intent` @ `940fd47a` (10 suite languages, 800 each) | choice | 8,000 | 539 | CC-BY-4.0 |
| dbpedia | `fancyzhx/dbpedia_14` @ `9abd46cf` | choice | 4,000 | 0 | CC-BY-SA-3.0 |
| boolq | `google/boolq` @ `35b264d0` | noul | 4,000 | 0 | CC-BY-SA-3.0 |
| snli | `stanfordnlp/snli` @ `cdb5c3d5` (neutral dropped) | noul | 4,000 | 0 | CC-BY-SA-4.0 |
| civil | `google/civil_comments` @ `f2970eb3` (toxicity in 5 levels, 800 each) | score, K = 5 | 4,000 | 1 | CC0-1.0 |

Choice questions get 2–12 candidates (gold plus seeded distractors, shuffled); batches group one
candidate count each so tensor shapes stay fixed on MPS. One epoch, batch 16 (1,994 steps), AdamW
(encoder lr 5e-5, new modules 3e-4, weight decay 0.01, 6% warmup, linear decay, clip 1.0), fp32,
gradient checkpointing on the encoder for every arm (without it the MPS allocator peaked at 11.5 GiB
and the 16 GB machine swapped). States are cut to 254 tokens in training only.

| Arm | Wall time | Peak MPS memory | Final training loss (last 94 steps) |
|---|---|---|---|
| `b` | 114 min | 5.1 GiB | 0.43 |
| `d1` | 101 min | 5.1 GiB | 1.07 |
| `d2` | 100 min | 6.1 GiB | 1.10 |
| `d4` | 106 min | 6.2 GiB | 1.35 |
| `d2-emb` | 73 min | 6.1 GiB | 1.42 |
| `d1-pool` | 116 min | 5.1 GiB | 0.44 |
| `d1-set` | 101 min | 5.2 GiB | 0.90 |
| `d1-lr` (2 epochs) | 205 min | 5.1 GiB | 1.16 |
| `late4` | 98 min | 6.1 GiB | 0.44 |
| `late8` | 127 min | 6.1 GiB | 0.44 |
| `late4-s14` | 115 min | 6.1 GiB | 0.50 |
| `b-s14` | 102 min | 5.1 GiB | 0.39 |
| `late8-s14` | 120 min | 6.1 GiB | 0.37 |
| `late6` | 108 min | 6.1 GiB | 0.45 |

A model that ignores the state and learns only candidate priors reaches a loss of about 1.53 on this
mixture. Every first-round tower arm, `d1-set`, and `d1-lr` stayed on that plateau for the first
~800 steps or more (0.70 for `b` by step 300); `d1-pool` left it at step ~350 and `late4` at ~300.
Small (std 0.02) instead of N(0, 1) initialization of the tower's type and position embeddings made
no difference in a 300-step probe (loss 1.46 vs. 1.49).

## Decision rule

Fixed before training. Suites are grouped into **in-domain** (banking77, massive, boolq: their train
splits are in the mixture, their test cases are not) and **held-out** (agnews, xnli, sst5, amazon:
never trained on). D-best is the tower arm with the highest held-out mean accuracy; within 0.005 of
it, the one with the lower warm p50 wins.

| Rule | Passes when |
|---|---|
| Q1 accuracy | `arch-b` − D-best ≤ 0.02 mean accuracy (choice and noul suites), in-domain and held-out |
| Q2 score | `arch-b` − D-best ≤ 0.05 mean QWK over the score suites |
| L1 latency | D-best warm runtime p50 × 5 ≤ `arch-b` warm runtime p50 (512 tokens, 1 question, K = 4) |
| I1 option order | D-best full-permutation flip rate 0, max probability deviation ≤ 1e-5 |
| I2 isolation | D-best question-interference max deviation ≤ 1e-5 |
| I3 cache | `arch-d2` vs `arch-d2-nocache` max probability difference ≤ 1e-5 |

D is accepted when every rule passes. If Q1 misses by at most 0.01, `arch-b` and D-best are retrained
once with seed 14 and the rule is applied to the two-seed mean.

## Results

<!-- GENERATED:START -->

### Quality

| Suite | Metric | arch-b | arch-d1 | arch-d2 | arch-d4 | arch-d2-emb | arch-d2-nocache | arch-d1-pool | arch-d1-lr | arch-d1-set | arch-late4 | arch-late4-nocache | arch-late8 | arch-late4-s14 | arch-b-s14 | arch-late8-nocache | arch-late6 | arch-late8-s14 | krite | krite-nocache |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agnews-choice | accuracy / macro_f1 | 0.623 / 0.592 | 0.318 / 0.210 | 0.307 / 0.154 | 0.220 / 0.090 | 0.220 / 0.090 | 0.307 / 0.154 | 0.443 / 0.397 | 0.297 / 0.178 | 0.365 / 0.296 | 0.652 / 0.647 | 0.652 / 0.647 | 0.625 / 0.621 | 0.517 / 0.503 | 0.557 / 0.531 | 0.625 / 0.621 | 0.647 / 0.650 | 0.412 / 0.379 | 0.625 / 0.621 | 0.625 / 0.621 |
| banking77-choice | accuracy / macro_f1 | 0.902 / 0.901 | 0.330 / 0.282 | 0.330 / 0.305 | 0.083 / 0.031 | 0.070 / 0.026 | n/a (not measured) | 0.877 / 0.866 | 0.280 / 0.269 | 0.407 / 0.362 | 0.907 / 0.902 | n/a (not measured) | 0.902 / 0.900 | 0.800 / 0.789 | 0.902 / 0.904 | n/a (not measured) | 0.895 / 0.888 | 0.910 / 0.907 | 0.902 / 0.900 | n/a (not measured) |
| boolq-noul | accuracy / auroc | 0.603 / 0.648 | 0.580 / 0.510 | 0.580 / 0.485 | 0.580 / 0.555 | 0.580 / 0.497 | 0.580 / 0.485 | 0.583 / 0.605 | 0.580 / 0.516 | 0.580 / 0.487 | 0.598 / 0.629 | 0.598 / 0.629 | 0.630 / 0.678 | 0.580 / 0.568 | 0.600 / 0.633 | 0.630 / 0.678 | 0.580 / 0.644 | 0.605 / 0.637 | 0.630 / 0.678 | 0.630 / 0.678 |
| sst5-score | mae / qwk | 0.881 / 0.520 | 1.141 / -0.046 | 1.155 / -0.016 | 1.133 / -0.038 | 1.133 / -0.051 | n/a (not measured) | 1.038 / 0.319 | 1.150 / 0.068 | 1.140 / 0.014 | 0.983 / 0.397 | n/a (not measured) | 0.881 / 0.464 | 1.139 / 0.121 | 0.876 / 0.534 | n/a (not measured) | 0.926 / 0.462 | 0.919 / 0.439 | 0.881 / 0.464 | n/a (not measured) |
| massive-choice-en | accuracy / macro_f1 | 0.915 / 0.878 | 0.453 / 0.351 | 0.443 / 0.358 | 0.350 / 0.114 | 0.263 / 0.086 | n/a (not measured) | 0.900 / 0.862 | 0.362 / 0.248 | 0.575 / 0.407 | 0.922 / 0.880 | n/a (not measured) | 0.907 / 0.880 | 0.880 / 0.865 | 0.920 / 0.926 | n/a (not measured) | 0.922 / 0.892 | 0.917 / 0.883 | 0.907 / 0.880 | n/a (not measured) |
| massive-choice-ko | accuracy / macro_f1 | 0.897 / 0.859 | 0.420 / 0.259 | 0.340 / 0.199 | 0.323 / 0.108 | 0.253 / 0.082 | 0.340 / 0.199 | 0.885 / 0.852 | 0.347 / 0.186 | 0.583 / 0.446 | 0.905 / 0.889 | 0.905 / 0.889 | 0.887 / 0.847 | 0.845 / 0.802 | 0.875 / 0.877 | 0.887 / 0.847 | 0.892 / 0.890 | 0.882 / 0.850 | 0.887 / 0.847 | 0.887 / 0.847 |
| massive-choice-ja | accuracy / macro_f1 | 0.920 / 0.888 | 0.468 / 0.356 | 0.425 / 0.334 | 0.328 / 0.107 | 0.268 / 0.092 | n/a (not measured) | 0.870 / 0.829 | 0.407 / 0.267 | 0.618 / 0.454 | 0.940 / 0.932 | n/a (not measured) | 0.907 / 0.907 | 0.860 / 0.827 | 0.927 / 0.942 | n/a (not measured) | 0.917 / 0.903 | 0.892 / 0.877 | 0.907 / 0.907 | n/a (not measured) |
| massive-choice-zh-CN | accuracy / macro_f1 | 0.932 / 0.905 | 0.468 / 0.338 | 0.378 / 0.280 | 0.323 / 0.128 | 0.235 / 0.095 | n/a (not measured) | 0.907 / 0.866 | 0.388 / 0.213 | 0.605 / 0.453 | 0.945 / 0.947 | n/a (not measured) | 0.920 / 0.909 | 0.877 / 0.888 | 0.943 / 0.921 | n/a (not measured) | 0.927 / 0.907 | 0.932 / 0.932 | 0.920 / 0.909 | n/a (not measured) |
| massive-choice-de | accuracy / macro_f1 | 0.890 / 0.861 | 0.407 / 0.261 | 0.350 / 0.211 | 0.328 / 0.123 | 0.258 / 0.092 | n/a (not measured) | 0.880 / 0.837 | 0.372 / 0.213 | 0.578 / 0.423 | 0.900 / 0.863 | n/a (not measured) | 0.902 / 0.872 | 0.833 / 0.785 | 0.875 / 0.827 | n/a (not measured) | 0.895 / 0.842 | 0.895 / 0.844 | 0.902 / 0.872 | n/a (not measured) |
| massive-choice-es | accuracy / macro_f1 | 0.915 / 0.914 | 0.480 / 0.308 | 0.453 / 0.341 | 0.357 / 0.121 | 0.275 / 0.107 | n/a (not measured) | 0.895 / 0.892 | 0.415 / 0.248 | 0.632 / 0.477 | 0.920 / 0.908 | n/a (not measured) | 0.912 / 0.882 | 0.858 / 0.841 | 0.900 / 0.890 | n/a (not measured) | 0.912 / 0.902 | 0.907 / 0.897 | 0.912 / 0.882 | n/a (not measured) |
| massive-choice-fr | accuracy / macro_f1 | 0.940 / 0.929 | 0.472 / 0.348 | 0.372 / 0.297 | 0.352 / 0.116 | 0.235 / 0.083 | n/a (not measured) | 0.900 / 0.851 | 0.362 / 0.235 | 0.598 / 0.456 | 0.925 / 0.916 | n/a (not measured) | 0.935 / 0.921 | 0.880 / 0.870 | 0.930 / 0.934 | n/a (not measured) | 0.922 / 0.915 | 0.912 / 0.897 | 0.935 / 0.921 | n/a (not measured) |
| massive-choice-ar | accuracy / macro_f1 | 0.863 / 0.788 | 0.455 / 0.310 | 0.318 / 0.252 | 0.323 / 0.118 | 0.212 / 0.083 | n/a (not measured) | 0.807 / 0.762 | 0.328 / 0.202 | 0.535 / 0.385 | 0.873 / 0.822 | n/a (not measured) | 0.855 / 0.797 | 0.800 / 0.735 | 0.843 / 0.769 | n/a (not measured) | 0.840 / 0.782 | 0.875 / 0.798 | 0.855 / 0.797 | n/a (not measured) |
| massive-choice-hi | accuracy / macro_f1 | 0.865 / 0.750 | 0.370 / 0.209 | 0.320 / 0.199 | 0.273 / 0.092 | 0.240 / 0.099 | n/a (not measured) | 0.802 / 0.683 | 0.360 / 0.185 | 0.552 / 0.354 | 0.870 / 0.784 | n/a (not measured) | 0.860 / 0.731 | 0.755 / 0.595 | 0.865 / 0.772 | n/a (not measured) | 0.840 / 0.742 | 0.865 / 0.759 | 0.860 / 0.731 | n/a (not measured) |
| massive-choice-ru | accuracy / macro_f1 | 0.910 / 0.864 | 0.443 / 0.289 | 0.362 / 0.233 | 0.318 / 0.104 | 0.225 / 0.068 | n/a (not measured) | 0.880 / 0.824 | 0.380 / 0.217 | 0.580 / 0.464 | 0.917 / 0.878 | n/a (not measured) | 0.902 / 0.852 | 0.870 / 0.828 | 0.910 / 0.855 | n/a (not measured) | 0.882 / 0.858 | 0.887 / 0.811 | 0.902 / 0.852 | n/a (not measured) |
| xnli-noul-en | accuracy / auroc | 0.775 / 0.855 | 0.510 / 0.550 | 0.510 / 0.527 | 0.510 / 0.546 | 0.510 / 0.538 | n/a (not measured) | 0.688 / 0.786 | 0.510 / 0.433 | 0.532 / 0.534 | 0.700 / 0.819 | n/a (not measured) | 0.755 / 0.831 | 0.515 / 0.524 | 0.748 / 0.858 | n/a (not measured) | 0.728 / 0.817 | 0.728 / 0.823 | 0.755 / 0.831 | n/a (not measured) |
| xnli-noul-de | accuracy / auroc | 0.748 / 0.809 | 0.510 / 0.462 | 0.510 / 0.471 | 0.510 / 0.477 | 0.510 / 0.470 | n/a (not measured) | 0.650 / 0.759 | 0.510 / 0.501 | 0.512 / 0.537 | 0.703 / 0.788 | n/a (not measured) | 0.730 / 0.826 | 0.552 / 0.568 | 0.733 / 0.811 | n/a (not measured) | 0.713 / 0.787 | 0.723 / 0.806 | 0.730 / 0.826 | n/a (not measured) |
| xnli-noul-es | accuracy / auroc | 0.787 / 0.843 | 0.460 / 0.487 | 0.460 / 0.471 | 0.460 / 0.528 | 0.460 / 0.512 | n/a (not measured) | 0.647 / 0.767 | 0.460 / 0.527 | 0.475 / 0.510 | 0.730 / 0.808 | n/a (not measured) | 0.738 / 0.823 | 0.507 / 0.537 | 0.750 / 0.817 | n/a (not measured) | 0.708 / 0.815 | 0.743 / 0.814 | 0.738 / 0.823 | n/a (not measured) |
| xnli-noul-fr | accuracy / auroc | 0.728 / 0.792 | 0.505 / 0.467 | 0.505 / 0.497 | 0.505 / 0.570 | 0.505 / 0.524 | n/a (not measured) | 0.665 / 0.745 | 0.505 / 0.523 | 0.500 / 0.512 | 0.718 / 0.788 | n/a (not measured) | 0.765 / 0.827 | 0.545 / 0.591 | 0.738 / 0.789 | n/a (not measured) | 0.730 / 0.805 | 0.757 / 0.822 | 0.765 / 0.827 | n/a (not measured) |
| xnli-noul-ar | accuracy / auroc | 0.720 / 0.792 | 0.497 / 0.520 | 0.497 / 0.476 | 0.497 / 0.496 | 0.497 / 0.488 | n/a (not measured) | 0.637 / 0.762 | 0.497 / 0.492 | 0.497 / 0.526 | 0.670 / 0.747 | n/a (not measured) | 0.728 / 0.799 | 0.550 / 0.570 | 0.725 / 0.789 | n/a (not measured) | 0.677 / 0.765 | 0.713 / 0.793 | 0.728 / 0.799 | n/a (not measured) |
| xnli-noul-hi | accuracy / auroc | 0.680 / 0.739 | 0.542 / 0.535 | 0.542 / 0.517 | 0.542 / 0.499 | 0.542 / 0.449 | n/a (not measured) | 0.657 / 0.717 | 0.542 / 0.494 | 0.535 / 0.502 | 0.640 / 0.718 | n/a (not measured) | 0.642 / 0.739 | 0.517 / 0.551 | 0.667 / 0.730 | n/a (not measured) | 0.682 / 0.762 | 0.690 / 0.770 | 0.642 / 0.739 | n/a (not measured) |
| xnli-noul-ru | accuracy / auroc | 0.733 / 0.807 | 0.495 / 0.494 | 0.495 / 0.521 | 0.495 / 0.524 | 0.495 / 0.482 | n/a (not measured) | 0.637 / 0.762 | 0.495 / 0.487 | 0.532 / 0.577 | 0.703 / 0.744 | n/a (not measured) | 0.690 / 0.771 | 0.470 / 0.521 | 0.698 / 0.777 | n/a (not measured) | 0.690 / 0.768 | 0.688 / 0.763 | 0.690 / 0.771 | n/a (not measured) |
| xnli-noul-zh | accuracy / auroc | 0.740 / 0.813 | 0.475 / 0.459 | 0.475 / 0.470 | 0.475 / 0.452 | 0.475 / 0.490 | n/a (not measured) | 0.650 / 0.732 | 0.475 / 0.551 | 0.505 / 0.573 | 0.698 / 0.750 | n/a (not measured) | 0.703 / 0.780 | 0.547 / 0.547 | 0.700 / 0.790 | n/a (not measured) | 0.705 / 0.775 | 0.695 / 0.774 | 0.703 / 0.780 | n/a (not measured) |
| amazon-score-en | mae / qwk | 1.294 / -0.081 | 1.169 / -0.006 | 1.180 / -0.008 | 1.168 / 0.000 | 1.168 / 0.000 | n/a (not measured) | 1.202 / -0.173 | 1.185 / 0.045 | 1.179 / 0.000 | 1.288 / 0.014 | n/a (not measured) | 1.112 / 0.390 | 1.188 / 0.013 | 1.282 / 0.088 | n/a (not measured) | 1.168 / 0.136 | 1.167 / 0.150 | 1.112 / 0.390 | n/a (not measured) |
| amazon-score-de | mae / qwk | 1.295 / -0.167 | 1.157 / 0.010 | 1.164 / 0.002 | 1.156 / 0.000 | 1.156 / 0.000 | n/a (not measured) | 1.185 / -0.073 | 1.164 / 0.129 | 1.165 / 0.000 | 1.268 / -0.147 | n/a (not measured) | 1.130 / 0.308 | 1.190 / 0.053 | 1.294 / -0.037 | n/a (not measured) | 1.172 / 0.081 | 1.201 / 0.001 | 1.130 / 0.308 | n/a (not measured) |
| amazon-score-es | mae / qwk | 1.345 / -0.151 | 1.218 / -0.003 | 1.225 / -0.016 | 1.218 / 0.000 | 1.218 / 0.000 | n/a (not measured) | 1.245 / -0.145 | 1.238 / 0.031 | 1.231 / -0.002 | 1.300 / -0.131 | n/a (not measured) | 1.195 / 0.248 | 1.259 / 0.006 | 1.344 / -0.047 | n/a (not measured) | 1.214 / 0.098 | 1.266 / 0.047 | 1.195 / 0.248 | n/a (not measured) |
| amazon-score-fr | mae / qwk | 1.337 / -0.143 | 1.188 / -0.009 | 1.197 / -0.018 | 1.190 / 0.000 | 1.188 / 0.000 | n/a (not measured) | 1.218 / -0.223 | 1.204 / 0.040 | 1.195 / 0.000 | 1.307 / -0.181 | n/a (not measured) | 1.177 / 0.202 | 1.237 / -0.059 | 1.344 / -0.014 | n/a (not measured) | 1.196 / 0.163 | 1.230 / -0.008 | 1.177 / 0.202 | n/a (not measured) |
| amazon-score-ja | mae / qwk | 1.339 / -0.071 | 1.189 / 0.005 | 1.198 / 0.002 | 1.188 / 0.000 | 1.188 / 0.000 | 1.198 / 0.002 | 1.218 / -0.038 | 1.196 / 0.086 | 1.194 / 0.004 | 1.339 / -0.089 | 1.339 / -0.089 | 1.191 / 0.090 | 1.195 / 0.007 | 1.391 / -0.018 | 1.191 / 0.090 | 1.210 / 0.097 | 1.197 / -0.048 | 1.191 / 0.090 | 1.191 / 0.090 |
| amazon-score-zh | mae / qwk | 1.307 / -0.154 | 1.154 / 0.003 | 1.162 / 0.000 | 1.153 / 0.000 | 1.153 / 0.000 | n/a (not measured) | 1.183 / -0.125 | 1.158 / 0.140 | 1.168 / -0.003 | 1.281 / -0.123 | n/a (not measured) | 1.150 / 0.142 | 1.182 / 0.040 | 1.368 / -0.032 | n/a (not measured) | 1.158 / 0.134 | 1.195 / -0.059 | 1.150 / 0.142 | n/a (not measured) |

### Calibration: ECE raw → temperature-scaled

Evaluation half of each suite. One temperature per engine model and calibration bucket, fit on the pooled other halves of every suite in that bucket (benchmark-spec §9); the classifier has one model, and so one calibrator, per dataset.

| Suite | arch-b | arch-d1 | arch-d2 | arch-d4 | arch-d2-emb | arch-d2-nocache | arch-d1-pool | arch-d1-lr | arch-d1-set | arch-late4 | arch-late4-nocache | arch-late8 | arch-late4-s14 | arch-b-s14 | arch-late8-nocache | arch-late6 | arch-late8-s14 | krite | krite-nocache |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agnews-choice | 0.103 → 0.102 | 0.366 → 0.077 | 0.109 → 0.017 | 0.014 → 0.005 | 0.008 → 0.005 | 0.109 → 0.017 | 0.245 → 0.073 | 0.090 → 0.031 | 0.170 → 0.046 | 0.090 → 0.065 | 0.090 → 0.065 | 0.152 → 0.171 | 0.136 → 0.124 | 0.205 → 0.090 | 0.152 → 0.171 | 0.102 → 0.054 | 0.256 → 0.105 | 0.152 → 0.171 | 0.152 → 0.171 |
| banking77-choice | 0.058 → 0.065 | 0.120 → 0.098 | 0.087 → 0.072 | 0.101 → 0.102 | 0.033 → 0.014 | n/a (not measured) | 0.034 → 0.053 | 0.086 → 0.098 | 0.108 → 0.077 | 0.045 → 0.031 | n/a (not measured) | 0.046 → 0.048 | 0.082 → 0.066 | 0.047 → 0.047 | n/a (not measured) | 0.039 → 0.071 | 0.052 → 0.047 | 0.046 → 0.048 | n/a (not measured) |
| boolq-noul | 0.089 → 0.104 | 0.053 → 0.028 | 0.080 → 0.022 | 0.027 → 0.055 | 0.049 → 0.076 | 0.080 → 0.024 | 0.070 → 0.096 | 0.040 → 0.018 | 0.061 → 0.065 | 0.070 → 0.041 | 0.070 → 0.039 | 0.070 → 0.054 | 0.035 → 0.074 | 0.104 → 0.085 | 0.070 → 0.078 | 0.093 → 0.061 | 0.104 → 0.093 | 0.070 → 0.054 | 0.070 → 0.078 |
| sst5-score | 0.066 → 0.088 | 0.103 → 0.004 | 0.119 → 0.029 | 0.071 → 0.075 | 0.023 → 0.018 | n/a (not measured) | 0.105 → 0.057 | 0.232 → 0.068 | 0.052 → 0.031 | 0.164 → 0.036 | n/a (not measured) | 0.176 → 0.044 | 0.043 → 0.067 | 0.099 → 0.142 | n/a (not measured) | 0.110 → 0.047 | 0.109 → 0.056 | 0.176 → 0.044 | n/a (not measured) |
| massive-choice-en | 0.066 → 0.032 | 0.114 → 0.090 | 0.099 → 0.083 | 0.096 → 0.077 | 0.093 → 0.073 | n/a (not measured) | 0.053 → 0.041 | 0.075 → 0.070 | 0.124 → 0.083 | 0.048 → 0.040 | n/a (not measured) | 0.056 → 0.031 | 0.064 → 0.038 | 0.070 → 0.061 | n/a (not measured) | 0.049 → 0.042 | 0.063 → 0.048 | 0.056 → 0.031 | n/a (not measured) |
| massive-choice-ko | 0.046 → 0.034 | 0.070 → 0.050 | 0.128 → 0.096 | 0.098 → 0.092 | 0.058 → 0.079 | 0.128 → 0.094 | 0.067 → 0.032 | 0.107 → 0.085 | 0.128 → 0.097 | 0.062 → 0.049 | 0.062 → 0.045 | 0.071 → 0.052 | 0.060 → 0.053 | 0.052 → 0.039 | 0.071 → 0.042 | 0.035 → 0.049 | 0.056 → 0.046 | 0.071 → 0.052 | 0.071 → 0.042 |
| massive-choice-ja | 0.028 → 0.046 | 0.141 → 0.115 | 0.103 → 0.102 | 0.079 → 0.081 | 0.095 → 0.094 | n/a (not measured) | 0.060 → 0.058 | 0.069 → 0.070 | 0.076 → 0.065 | 0.027 → 0.050 | n/a (not measured) | 0.043 → 0.048 | 0.069 → 0.057 | 0.052 → 0.042 | n/a (not measured) | 0.041 → 0.044 | 0.058 → 0.038 | 0.043 → 0.048 | n/a (not measured) |
| massive-choice-zh-CN | 0.042 → 0.041 | 0.114 → 0.089 | 0.130 → 0.093 | 0.096 → 0.057 | 0.039 → 0.044 | n/a (not measured) | 0.034 → 0.051 | 0.096 → 0.090 | 0.068 → 0.099 | 0.031 → 0.057 | n/a (not measured) | 0.042 → 0.040 | 0.055 → 0.050 | 0.037 → 0.042 | n/a (not measured) | 0.037 → 0.050 | 0.032 → 0.037 | 0.042 → 0.040 | n/a (not measured) |
| massive-choice-de | 0.049 → 0.052 | 0.128 → 0.052 | 0.093 → 0.102 | 0.106 → 0.094 | 0.087 → 0.052 | n/a (not measured) | 0.058 → 0.041 | 0.070 → 0.051 | 0.119 → 0.064 | 0.074 → 0.056 | n/a (not measured) | 0.049 → 0.045 | 0.089 → 0.056 | 0.076 → 0.039 | n/a (not measured) | 0.064 → 0.048 | 0.051 → 0.042 | 0.049 → 0.045 | n/a (not measured) |
| massive-choice-es | 0.062 → 0.029 | 0.118 → 0.087 | 0.084 → 0.065 | 0.083 → 0.095 | 0.107 → 0.115 | n/a (not measured) | 0.052 → 0.045 | 0.064 → 0.057 | 0.082 → 0.091 | 0.057 → 0.047 | n/a (not measured) | 0.064 → 0.039 | 0.088 → 0.052 | 0.055 → 0.038 | n/a (not measured) | 0.056 → 0.066 | 0.053 → 0.034 | 0.064 → 0.039 | n/a (not measured) |
| massive-choice-fr | 0.050 → 0.041 | 0.152 → 0.091 | 0.096 → 0.069 | 0.130 → 0.090 | 0.079 → 0.084 | n/a (not measured) | 0.082 → 0.049 | 0.097 → 0.068 | 0.046 → 0.070 | 0.046 → 0.032 | n/a (not measured) | 0.039 → 0.037 | 0.058 → 0.055 | 0.051 → 0.055 | n/a (not measured) | 0.061 → 0.057 | 0.044 → 0.040 | 0.039 → 0.037 | n/a (not measured) |
| massive-choice-ar | 0.081 → 0.087 | 0.086 → 0.083 | 0.115 → 0.078 | 0.107 → 0.114 | 0.054 → 0.077 | n/a (not measured) | 0.089 → 0.046 | 0.102 → 0.080 | 0.075 → 0.057 | 0.089 → 0.048 | n/a (not measured) | 0.078 → 0.024 | 0.064 → 0.057 | 0.061 → 0.058 | n/a (not measured) | 0.050 → 0.061 | 0.046 → 0.037 | 0.078 → 0.024 | n/a (not measured) |
| massive-choice-hi | 0.050 → 0.056 | 0.139 → 0.080 | 0.115 → 0.082 | 0.097 → 0.044 | 0.074 → 0.047 | n/a (not measured) | 0.080 → 0.056 | 0.095 → 0.080 | 0.120 → 0.089 | 0.051 → 0.053 | n/a (not measured) | 0.080 → 0.054 | 0.090 → 0.090 | 0.063 → 0.043 | n/a (not measured) | 0.079 → 0.056 | 0.041 → 0.065 | 0.080 → 0.054 | n/a (not measured) |
| massive-choice-ru | 0.051 → 0.039 | 0.086 → 0.046 | 0.074 → 0.036 | 0.062 → 0.081 | 0.059 → 0.088 | n/a (not measured) | 0.072 → 0.062 | 0.124 → 0.114 | 0.095 → 0.082 | 0.059 → 0.045 | n/a (not measured) | 0.069 → 0.036 | 0.057 → 0.053 | 0.050 → 0.044 | n/a (not measured) | 0.080 → 0.046 | 0.064 → 0.070 | 0.069 → 0.036 | n/a (not measured) |
| xnli-noul-en | 0.113 → 0.081 | 0.019 → 0.026 | 0.029 → 0.033 | 0.021 → 0.029 | 0.010 → 0.020 | n/a (not measured) | 0.190 → 0.044 | 0.034 → 0.034 | 0.103 → 0.033 | 0.162 → 0.089 | n/a (not measured) | 0.166 → 0.090 | 0.176 → 0.035 | 0.090 → 0.043 | n/a (not measured) | 0.107 → 0.047 | 0.144 → 0.070 | 0.166 → 0.090 | n/a (not measured) |
| xnli-noul-de | 0.093 → 0.048 | 0.008 → 0.016 | 0.018 → 0.022 | 0.010 → 0.018 | 0.022 → 0.010 | n/a (not measured) | 0.199 → 0.060 | 0.024 → 0.024 | 0.132 → 0.009 | 0.127 → 0.058 | n/a (not measured) | 0.119 → 0.054 | 0.165 → 0.045 | 0.096 → 0.077 | n/a (not measured) | 0.108 → 0.062 | 0.120 → 0.058 | 0.119 → 0.054 | n/a (not measured) |
| xnli-noul-es | 0.127 → 0.081 | 0.022 → 0.014 | 0.012 → 0.008 | 0.019 → 0.011 | 0.050 → 0.020 | n/a (not measured) | 0.180 → 0.050 | 0.006 → 0.006 | 0.162 → 0.006 | 0.140 → 0.043 | n/a (not measured) | 0.146 → 0.045 | 0.244 → 0.037 | 0.107 → 0.101 | n/a (not measured) | 0.065 → 0.071 | 0.125 → 0.084 | 0.146 → 0.045 | n/a (not measured) |
| xnli-noul-fr | 0.125 → 0.066 | 0.032 → 0.024 | 0.021 → 0.017 | 0.029 → 0.022 | 0.062 → 0.031 | n/a (not measured) | 0.175 → 0.052 | 0.016 → 0.016 | 0.164 → 0.040 | 0.144 → 0.066 | n/a (not measured) | 0.108 → 0.056 | 0.171 → 0.011 | 0.118 → 0.090 | n/a (not measured) | 0.095 → 0.075 | 0.115 → 0.066 | 0.108 → 0.056 | n/a (not measured) |
| xnli-noul-ar | 0.136 → 0.063 | 0.078 → 0.069 | 0.069 → 0.063 | 0.076 → 0.067 | 0.104 → 0.075 | n/a (not measured) | 0.233 → 0.074 | 0.061 → 0.061 | 0.206 → 0.084 | 0.162 → 0.060 | n/a (not measured) | 0.108 → 0.070 | 0.187 → 0.018 | 0.081 → 0.075 | n/a (not measured) | 0.092 → 0.038 | 0.142 → 0.122 | 0.108 → 0.070 | n/a (not measured) |
| xnli-noul-hi | 0.115 → 0.047 | 0.002 → 0.011 | 0.012 → 0.017 | 0.005 → 0.013 | 0.022 → 0.006 | n/a (not measured) | 0.186 → 0.056 | 0.019 → 0.019 | 0.130 → 0.016 | 0.153 → 0.068 | n/a (not measured) | 0.166 → 0.098 | 0.117 → 0.040 | 0.103 → 0.060 | n/a (not measured) | 0.104 → 0.034 | 0.084 → 0.068 | 0.166 → 0.098 | n/a (not measured) |
| xnli-noul-ru | 0.133 → 0.071 | 0.038 → 0.029 | 0.029 → 0.023 | 0.034 → 0.026 | 0.065 → 0.035 | n/a (not measured) | 0.252 → 0.095 | 0.021 → 0.021 | 0.163 → 0.028 | 0.154 → 0.068 | n/a (not measured) | 0.179 → 0.087 | 0.252 → 0.115 | 0.113 → 0.081 | n/a (not measured) | 0.134 → 0.073 | 0.176 → 0.108 | 0.179 → 0.087 | n/a (not measured) |
| xnli-noul-zh | 0.078 → 0.066 | 0.081 → 0.074 | 0.073 → 0.068 | 0.079 → 0.071 | 0.109 → 0.080 | n/a (not measured) | 0.241 → 0.055 | 0.066 → 0.066 | 0.181 → 0.055 | 0.150 → 0.065 | n/a (not measured) | 0.147 → 0.087 | 0.209 → 0.114 | 0.096 → 0.081 | n/a (not measured) | 0.062 → 0.057 | 0.105 → 0.057 | 0.147 → 0.087 | n/a (not measured) |
| amazon-score-en | 0.121 → 0.006 | 0.067 → 0.056 | 0.061 → 0.041 | 0.024 → 0.025 | 0.019 → 0.018 | n/a (not measured) | 0.074 → 0.058 | 0.018 → 0.000 | 0.057 → 0.031 | 0.097 → 0.020 | n/a (not measured) | 0.103 → 0.145 | 0.008 → 0.016 | 0.158 → 0.028 | n/a (not measured) | 0.015 → 0.042 | 0.034 → 0.055 | 0.103 → 0.145 | n/a (not measured) |
| amazon-score-de | 0.159 → 0.056 | 0.018 → 0.006 | 0.020 → 0.001 | 0.011 → 0.010 | 0.039 → 0.038 | n/a (not measured) | 0.062 → 0.044 | 0.019 → 0.044 | 0.028 → 0.001 | 0.115 → 0.029 | n/a (not measured) | 0.046 → 0.072 | 0.006 → 0.031 | 0.224 → 0.027 | n/a (not measured) | 0.021 → 0.021 | 0.057 → 0.013 | 0.046 → 0.072 | n/a (not measured) |
| amazon-score-es | 0.194 → 0.021 | 0.007 → 0.019 | 0.000 → 0.019 | 0.004 → 0.005 | 0.007 → 0.008 | n/a (not measured) | 0.049 → 0.033 | 0.003 → 0.019 | 0.006 → 0.034 | 0.079 → 0.011 | n/a (not measured) | 0.026 → 0.039 | 0.039 → 0.014 | 0.191 → 0.025 | n/a (not measured) | 0.028 → 0.019 | 0.056 → 0.013 | 0.026 → 0.039 | n/a (not measured) |
| amazon-score-fr | 0.187 → 0.065 | 0.008 → 0.004 | 0.004 → 0.024 | 0.006 → 0.005 | 0.021 → 0.022 | n/a (not measured) | 0.019 → 0.003 | 0.001 → 0.019 | 0.008 → 0.019 | 0.116 → 0.034 | n/a (not measured) | 0.033 → 0.021 | 0.042 → 0.019 | 0.210 → 0.037 | n/a (not measured) | 0.013 → 0.028 | 0.050 → 0.013 | 0.033 → 0.021 | n/a (not measured) |
| amazon-score-ja | 0.154 → 0.038 | 0.007 → 0.004 | 0.010 → 0.009 | 0.001 → 0.000 | 0.051 → 0.052 | 0.010 → 0.009 | 0.006 → 0.009 | 0.016 → 0.035 | 0.021 → 0.004 | 0.105 → 0.010 | 0.105 → 0.010 | 0.029 → 0.023 | 0.007 → 0.031 | 0.210 → 0.028 | 0.029 → 0.024 | 0.038 → 0.060 | 0.059 → 0.017 | 0.029 → 0.023 | 0.029 → 0.024 |
| amazon-score-zh | 0.140 → 0.045 | 0.048 → 0.036 | 0.034 → 0.016 | 0.022 → 0.020 | 0.018 → 0.018 | n/a (not measured) | 0.044 → 0.031 | 0.027 → 0.002 | 0.055 → 0.026 | 0.097 → 0.014 | n/a (not measured) | 0.034 → 0.010 | 0.038 → 0.014 | 0.222 → 0.064 | n/a (not measured) | 0.049 → 0.036 | 0.062 → 0.021 | 0.034 → 0.010 | n/a (not measured) |
| **mean over suites** | 0.097 → 0.056 | 0.079 → 0.049 | 0.065 → 0.047 | 0.054 → 0.049 | 0.052 → 0.046 | 0.082 → 0.036 | 0.108 → 0.051 | 0.060 → 0.048 | 0.098 → 0.050 | 0.097 → 0.046 | 0.082 → 0.040 | 0.087 → 0.058 | 0.095 → 0.051 | 0.109 → 0.059 | 0.081 → 0.079 | 0.065 → 0.050 | 0.084 → 0.054 | 0.087 → 0.058 | 0.081 → 0.079 |

### Calibration: NLL and Brier, raw → scaled

| Suite | arch-b | arch-d1 | arch-d2 | arch-d4 | arch-d2-emb | arch-d2-nocache | arch-d1-pool | arch-d1-lr | arch-d1-set | arch-late4 | arch-late4-nocache | arch-late8 | arch-late4-s14 | arch-b-s14 | arch-late8-nocache | arch-late6 | arch-late8-s14 | krite | krite-nocache |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agnews-choice | 1.060 → 1.013 · B 0.537 → 0.535 | 1.703 → 1.352 · B 0.928 → 0.734 | 1.413 → 1.385 · B 0.767 → 0.749 | 1.385 → 1.386 · B 0.749 → 0.750 | 1.386 → 1.386 · B 0.750 → 0.750 | 1.413 → 1.385 · B 0.767 → 0.749 | 1.349 → 1.221 · B 0.743 → 0.668 | 1.416 → 1.383 · B 0.766 → 0.748 | 1.384 → 1.325 · B 0.764 → 0.722 | 1.004 → 0.975 · B 0.535 → 0.528 | 1.004 → 0.975 · B 0.535 → 0.528 | 0.988 → 0.979 · B 0.548 → 0.552 | 1.106 → 1.075 · B 0.626 → 0.610 | 1.246 → 1.060 · B 0.640 → 0.590 | 0.988 → 0.979 · B 0.548 → 0.552 | 0.948 → 0.937 · B 0.504 → 0.501 | 1.285 → 1.168 · B 0.740 → 0.664 | 0.988 → 0.979 · B 0.548 → 0.552 | 0.988 → 0.979 · B 0.548 → 0.552 |
| banking77-choice | 0.312 → 0.339 · B 0.151 → 0.156 | 1.886 → 1.846 · B 0.770 → 0.772 | 1.811 → 1.788 · B 0.765 → 0.763 | 2.307 → 2.309 · B 0.883 → 0.885 | 2.495 → 2.485 · B 0.919 → 0.917 | n/a | 0.425 → 0.418 · B 0.187 → 0.188 | 1.883 → 1.879 · B 0.806 → 0.803 | 1.644 → 1.627 · B 0.715 → 0.712 | 0.309 → 0.276 · B 0.130 → 0.127 | n/a | 0.365 → 0.324 · B 0.153 → 0.154 | 0.646 → 0.638 · B 0.308 → 0.307 | 0.386 → 0.399 · B 0.172 → 0.173 | n/a | 0.296 → 0.290 · B 0.123 → 0.130 | 0.291 → 0.283 · B 0.125 → 0.127 | 0.365 → 0.324 · B 0.153 → 0.154 | n/a |
| boolq-noul | 0.654 → 0.639 · B 0.462 → 0.450 | 0.676 → 0.675 · B 0.483 → 0.482 | 0.693 → 0.678 · B 0.499 → 0.485 | 0.658 → 0.670 · B 0.468 → 0.478 | 0.681 → 0.689 · B 0.488 → 0.495 | 0.693 → 0.678 · B 0.499 → 0.485 | 0.664 → 0.673 · B 0.472 → 0.480 | 0.680 → 0.678 · B 0.486 → 0.484 | 0.689 → 0.685 · B 0.495 → 0.492 | 0.667 → 0.661 · B 0.474 → 0.468 | 0.667 → 0.660 · B 0.474 → 0.468 | 0.621 → 0.631 · B 0.435 → 0.441 | 0.670 → 0.687 · B 0.478 → 0.493 | 0.655 → 0.653 · B 0.465 → 0.462 | 0.621 → 0.625 · B 0.435 → 0.436 | 0.637 → 0.637 · B 0.451 → 0.448 | 0.653 → 0.660 · B 0.462 → 0.467 | 0.621 → 0.631 · B 0.435 → 0.441 | 0.621 → 0.625 · B 0.435 → 0.436 |
| sst5-score | 1.398 → 1.547 · B 0.731 → 0.775 | 1.641 → 1.609 · B 0.815 → 0.800 | 1.666 → 1.610 · B 0.823 → 0.800 | 1.609 → 1.609 · B 0.800 → 0.800 | 1.603 → 1.600 · B 0.797 → 0.796 | n/a | 1.534 → 1.559 · B 0.775 → 0.780 | 1.854 → 1.626 · B 0.921 → 0.807 | 1.614 → 1.609 · B 0.802 → 0.800 | 1.674 → 1.601 · B 0.827 → 0.797 | n/a | 1.580 → 1.507 · B 0.796 → 0.759 | 1.614 → 1.599 · B 0.793 → 0.796 | 1.442 → 1.532 · B 0.736 → 0.768 | n/a | 1.573 → 1.538 · B 0.790 → 0.772 | 1.480 → 1.500 · B 0.762 → 0.757 | 1.580 → 1.507 · B 0.796 → 0.759 | n/a |
| massive-choice-en | 0.304 → 0.279 · B 0.131 → 0.128 | 1.472 → 1.433 · B 0.685 → 0.676 | 1.423 → 1.408 · B 0.677 → 0.673 | 1.606 → 1.590 · B 0.755 → 0.741 | 1.973 → 1.928 · B 0.849 → 0.840 | n/a | 0.413 → 0.361 · B 0.180 → 0.173 | 1.602 → 1.587 · B 0.713 → 0.710 | 1.184 → 1.161 · B 0.562 → 0.555 | 0.247 → 0.224 · B 0.129 → 0.122 | n/a | 0.320 → 0.269 · B 0.150 → 0.142 | 0.341 → 0.335 · B 0.175 → 0.174 | 0.302 → 0.271 · B 0.156 → 0.148 | n/a | 0.258 → 0.242 · B 0.128 → 0.129 | 0.294 → 0.273 · B 0.154 → 0.150 | 0.320 → 0.269 · B 0.150 → 0.142 | n/a |
| massive-choice-ko | 0.308 → 0.291 · B 0.135 → 0.132 | 1.570 → 1.529 · B 0.706 → 0.696 | 1.732 → 1.698 · B 0.766 → 0.757 | 1.807 → 1.728 · B 0.774 → 0.764 | 1.964 → 1.935 · B 0.844 → 0.833 | 1.732 → 1.698 · B 0.766 → 0.757 | 0.374 → 0.340 · B 0.171 → 0.165 | 1.704 → 1.675 · B 0.753 → 0.745 | 1.098 → 1.096 · B 0.511 → 0.511 | 0.354 → 0.305 · B 0.152 → 0.148 | 0.354 → 0.303 · B 0.152 → 0.148 | 0.405 → 0.329 · B 0.177 → 0.162 | 0.446 → 0.431 · B 0.202 → 0.202 | 0.371 → 0.347 · B 0.167 → 0.161 | 0.405 → 0.328 · B 0.177 → 0.162 | 0.343 → 0.311 · B 0.145 → 0.145 | 0.335 → 0.317 · B 0.150 → 0.148 | 0.405 → 0.329 · B 0.177 → 0.162 | 0.405 → 0.328 · B 0.177 → 0.162 |
| massive-choice-ja | 0.202 → 0.202 · B 0.094 → 0.096 | 1.454 → 1.433 · B 0.688 → 0.674 | 1.474 → 1.474 · B 0.686 → 0.686 | 1.640 → 1.627 · B 0.728 → 0.729 | 1.965 → 1.926 · B 0.847 → 0.840 | n/a | 0.287 → 0.277 · B 0.149 → 0.144 | 1.494 → 1.496 · B 0.691 → 0.691 | 1.051 → 1.051 · B 0.533 → 0.528 | 0.139 → 0.154 · B 0.068 → 0.074 | n/a | 0.228 → 0.215 · B 0.105 → 0.111 | 0.373 → 0.364 · B 0.204 → 0.201 | 0.222 → 0.221 · B 0.119 → 0.116 | n/a | 0.279 → 0.257 · B 0.118 → 0.119 | 0.267 → 0.254 · B 0.140 → 0.136 | 0.228 → 0.215 · B 0.105 → 0.111 | n/a |
| massive-choice-zh-CN | 0.188 → 0.200 · B 0.106 → 0.105 | 1.450 → 1.434 · B 0.695 → 0.684 | 1.568 → 1.550 · B 0.737 → 0.730 | 1.783 → 1.711 · B 0.809 → 0.788 | 1.956 → 1.909 · B 0.844 → 0.833 | n/a | 0.201 → 0.217 · B 0.115 → 0.117 | 1.490 → 1.488 · B 0.698 → 0.694 | 1.056 → 1.057 · B 0.524 → 0.522 | 0.189 → 0.179 · B 0.086 → 0.087 | n/a | 0.236 → 0.226 · B 0.112 → 0.112 | 0.324 → 0.326 · B 0.165 → 0.166 | 0.216 → 0.221 · B 0.102 → 0.104 | n/a | 0.198 → 0.204 · B 0.107 → 0.109 | 0.191 → 0.199 · B 0.094 → 0.097 | 0.236 → 0.226 · B 0.112 → 0.112 | n/a |
| massive-choice-de | 0.308 → 0.305 · B 0.160 → 0.157 | 1.588 → 1.541 · B 0.733 → 0.713 | 1.657 → 1.633 · B 0.746 → 0.740 | 1.640 → 1.623 · B 0.748 → 0.742 | 1.972 → 1.948 · B 0.846 → 0.837 | n/a | 0.316 → 0.315 · B 0.177 → 0.173 | 1.614 → 1.602 · B 0.717 → 0.715 | 1.099 → 1.098 · B 0.543 → 0.539 | 0.341 → 0.296 · B 0.174 → 0.165 | n/a | 0.372 → 0.331 · B 0.168 → 0.166 | 0.564 → 0.546 · B 0.273 → 0.271 | 0.427 → 0.392 · B 0.204 → 0.199 | n/a | 0.333 → 0.313 · B 0.164 → 0.163 | 0.353 → 0.343 · B 0.178 → 0.177 | 0.372 → 0.331 · B 0.168 → 0.166 | n/a |
| massive-choice-es | 0.297 → 0.282 · B 0.146 → 0.143 | 1.439 → 1.406 · B 0.682 → 0.666 | 1.470 → 1.456 · B 0.677 → 0.674 | 1.623 → 1.592 · B 0.746 → 0.734 | 1.957 → 1.924 · B 0.846 → 0.843 | n/a | 0.266 → 0.266 · B 0.156 → 0.152 | 1.500 → 1.493 · B 0.690 → 0.687 | 1.085 → 1.074 · B 0.502 → 0.504 | 0.300 → 0.265 · B 0.145 → 0.139 | n/a | 0.405 → 0.326 · B 0.175 → 0.165 | 0.480 → 0.462 · B 0.230 → 0.227 | 0.247 → 0.241 · B 0.124 → 0.122 | n/a | 0.285 → 0.264 · B 0.132 → 0.130 | 0.352 → 0.326 · B 0.163 → 0.160 | 0.405 → 0.326 · B 0.175 → 0.165 | n/a |
| massive-choice-fr | 0.195 → 0.202 · B 0.099 → 0.101 | 1.429 → 1.408 · B 0.701 → 0.682 | 1.521 → 1.510 · B 0.717 → 0.710 | 1.705 → 1.634 · B 0.766 → 0.747 | 1.956 → 1.913 · B 0.845 → 0.838 | n/a | 0.363 → 0.331 · B 0.169 → 0.165 | 1.487 → 1.491 · B 0.691 → 0.691 | 1.063 → 1.064 · B 0.514 → 0.513 | 0.238 → 0.225 · B 0.118 → 0.113 | n/a | 0.271 → 0.246 · B 0.116 → 0.118 | 0.419 → 0.407 · B 0.209 → 0.207 | 0.270 → 0.254 · B 0.126 → 0.126 | n/a | 0.308 → 0.277 · B 0.138 → 0.131 | 0.265 → 0.254 · B 0.140 → 0.137 | 0.271 → 0.246 · B 0.116 → 0.118 | n/a |
| massive-choice-ar | 0.537 → 0.498 · B 0.231 → 0.229 | 1.554 → 1.520 · B 0.699 → 0.693 | 1.730 → 1.704 · B 0.774 → 0.767 | 1.702 → 1.687 · B 0.754 → 0.756 | 1.981 → 1.969 · B 0.850 → 0.849 | n/a | 0.596 → 0.534 · B 0.263 → 0.248 | 1.671 → 1.658 · B 0.755 → 0.751 | 1.153 → 1.145 · B 0.544 → 0.542 | 0.495 → 0.444 · B 0.213 → 0.208 | n/a | 0.563 → 0.471 · B 0.234 → 0.223 | 0.619 → 0.606 · B 0.295 → 0.294 | 0.467 → 0.444 · B 0.227 → 0.221 | n/a | 0.436 → 0.412 · B 0.200 → 0.201 | 0.392 → 0.377 · B 0.188 → 0.183 | 0.563 → 0.471 · B 0.234 → 0.223 | n/a |
| massive-choice-hi | 0.343 → 0.341 · B 0.170 → 0.166 | 1.809 → 1.727 · B 0.786 → 0.763 | 1.817 → 1.781 · B 0.805 → 0.795 | 1.847 → 1.800 · B 0.815 → 0.801 | 2.002 → 2.013 · B 0.855 → 0.858 | n/a | 0.528 → 0.488 · B 0.237 → 0.230 | 1.686 → 1.676 · B 0.740 → 0.739 | 1.218 → 1.207 · B 0.590 → 0.585 | 0.329 → 0.310 · B 0.160 → 0.155 | n/a | 0.477 → 0.403 · B 0.226 → 0.208 | 0.673 → 0.649 · B 0.346 → 0.339 | 0.461 → 0.433 · B 0.199 → 0.197 | n/a | 0.473 → 0.433 · B 0.218 → 0.210 | 0.361 → 0.351 · B 0.184 → 0.179 | 0.477 → 0.403 · B 0.226 → 0.208 | n/a |
| massive-choice-ru | 0.315 → 0.302 · B 0.153 → 0.152 | 1.541 → 1.508 · B 0.699 → 0.693 | 1.697 → 1.669 · B 0.766 → 0.759 | 1.791 → 1.756 · B 0.789 → 0.782 | 1.961 → 1.923 · B 0.847 → 0.843 | n/a | 0.359 → 0.334 · B 0.189 → 0.180 | 1.699 → 1.672 · B 0.778 → 0.768 | 1.185 → 1.172 · B 0.572 → 0.568 | 0.354 → 0.306 · B 0.156 → 0.153 | n/a | 0.394 → 0.327 · B 0.171 → 0.158 | 0.399 → 0.392 · B 0.206 → 0.203 | 0.290 → 0.279 · B 0.147 → 0.145 | n/a | 0.354 → 0.321 · B 0.191 → 0.177 | 0.366 → 0.336 · B 0.178 → 0.168 | 0.394 → 0.327 · B 0.171 → 0.158 | n/a |
| xnli-noul-en | 0.549 → 0.505 · B 0.357 → 0.336 | 0.691 → 0.692 · B 0.498 → 0.499 | 0.692 → 0.693 · B 0.499 → 0.500 | 0.691 → 0.692 · B 0.498 → 0.499 | 0.690 → 0.691 · B 0.497 → 0.498 | n/a | 0.776 → 0.603 · B 0.482 → 0.415 | 0.693 → 0.693 · B 0.500 → 0.500 | 0.714 → 0.688 · B 0.518 → 0.495 | 0.660 → 0.564 · B 0.423 → 0.381 | n/a | 0.653 → 0.539 · B 0.390 → 0.357 | 0.785 → 0.683 · B 0.554 → 0.490 | 0.518 → 0.507 · B 0.339 → 0.334 | n/a | 0.605 → 0.562 · B 0.390 → 0.376 | 0.609 → 0.544 · B 0.399 → 0.367 | 0.653 → 0.539 · B 0.390 → 0.357 | n/a |
| xnli-noul-de | 0.539 → 0.515 · B 0.353 → 0.341 | 0.693 → 0.693 · B 0.499 → 0.500 | 0.693 → 0.693 · B 0.500 → 0.500 | 0.693 → 0.693 · B 0.500 → 0.500 | 0.693 → 0.692 · B 0.500 → 0.499 | n/a | 0.754 → 0.610 · B 0.490 → 0.422 | 0.693 → 0.693 · B 0.500 → 0.500 | 0.728 → 0.692 · B 0.532 → 0.499 | 0.629 → 0.565 · B 0.407 → 0.380 | n/a | 0.600 → 0.534 · B 0.378 → 0.354 | 0.796 → 0.689 · B 0.566 → 0.496 | 0.551 → 0.542 · B 0.367 → 0.362 | n/a | 0.621 → 0.582 · B 0.398 → 0.388 | 0.559 → 0.526 · B 0.369 → 0.351 | 0.600 → 0.534 · B 0.378 → 0.354 | n/a |
| xnli-noul-es | 0.560 → 0.516 · B 0.370 → 0.345 | 0.694 → 0.694 · B 0.501 → 0.500 | 0.694 → 0.693 · B 0.501 → 0.500 | 0.693 → 0.693 · B 0.500 → 0.500 | 0.698 → 0.694 · B 0.505 → 0.501 | n/a | 0.816 → 0.620 · B 0.496 → 0.430 | 0.693 → 0.693 · B 0.500 → 0.500 | 0.742 → 0.693 · B 0.545 → 0.500 | 0.641 → 0.574 · B 0.418 → 0.389 | n/a | 0.630 → 0.543 · B 0.398 → 0.364 | 0.863 → 0.698 · B 0.617 → 0.504 | 0.549 → 0.540 · B 0.369 → 0.364 | n/a | 0.510 → 0.499 · B 0.339 → 0.332 | 0.600 → 0.549 · B 0.396 → 0.370 | 0.630 → 0.543 · B 0.398 → 0.364 | n/a |
| xnli-noul-fr | 0.611 → 0.560 · B 0.405 → 0.380 | 0.695 → 0.694 · B 0.502 → 0.501 | 0.694 → 0.693 · B 0.501 → 0.500 | 0.693 → 0.693 · B 0.500 → 0.500 | 0.700 → 0.694 · B 0.507 → 0.501 | n/a | 0.799 → 0.624 · B 0.501 → 0.434 | 0.693 → 0.693 · B 0.500 → 0.500 | 0.742 → 0.694 · B 0.545 → 0.501 | 0.675 → 0.592 · B 0.441 → 0.405 | n/a | 0.562 → 0.522 · B 0.362 → 0.347 | 0.798 → 0.687 · B 0.567 → 0.494 | 0.625 → 0.603 · B 0.423 → 0.412 | n/a | 0.614 → 0.576 · B 0.402 → 0.388 | 0.623 → 0.568 · B 0.408 → 0.385 | 0.562 → 0.522 · B 0.362 → 0.347 | n/a |
| xnli-noul-ar | 0.641 → 0.583 · B 0.426 → 0.397 | 0.698 → 0.695 · B 0.505 → 0.502 | 0.695 → 0.694 · B 0.502 → 0.501 | 0.697 → 0.695 · B 0.504 → 0.502 | 0.708 → 0.697 · B 0.515 → 0.504 | n/a | 0.907 → 0.666 · B 0.582 → 0.473 | 0.693 → 0.693 · B 0.500 → 0.500 | 0.765 → 0.698 · B 0.568 → 0.505 | 0.717 → 0.628 · B 0.479 → 0.436 | n/a | 0.633 → 0.580 · B 0.413 → 0.393 | 0.800 → 0.691 · B 0.568 → 0.498 | 0.636 → 0.617 · B 0.420 → 0.415 | n/a | 0.637 → 0.609 · B 0.434 → 0.420 | 0.670 → 0.611 · B 0.454 → 0.424 | 0.633 → 0.580 · B 0.413 → 0.393 | n/a |
| xnli-noul-hi | 0.635 → 0.590 · B 0.429 → 0.406 | 0.692 → 0.692 · B 0.499 → 0.499 | 0.693 → 0.693 · B 0.499 → 0.500 | 0.691 → 0.692 · B 0.498 → 0.499 | 0.693 → 0.692 · B 0.500 → 0.499 | n/a | 0.784 → 0.623 · B 0.491 → 0.433 | 0.693 → 0.693 · B 0.500 → 0.500 | 0.723 → 0.692 · B 0.529 → 0.499 | 0.700 → 0.627 · B 0.484 → 0.441 | n/a | 0.644 → 0.588 · B 0.446 → 0.408 | 0.690 → 0.677 · B 0.494 → 0.484 | 0.628 → 0.617 · B 0.442 → 0.433 | n/a | 0.624 → 0.600 · B 0.428 → 0.415 | 0.625 → 0.606 · B 0.432 → 0.419 | 0.644 → 0.588 · B 0.446 → 0.408 | n/a |
| xnli-noul-ru | 0.632 → 0.574 · B 0.415 → 0.387 | 0.695 → 0.694 · B 0.502 → 0.501 | 0.694 → 0.694 · B 0.501 → 0.500 | 0.694 → 0.694 · B 0.501 → 0.500 | 0.701 → 0.695 · B 0.508 → 0.502 | n/a | 0.889 → 0.663 · B 0.585 → 0.472 | 0.693 → 0.693 · B 0.500 → 0.500 | 0.744 → 0.694 · B 0.547 → 0.501 | 0.710 → 0.623 · B 0.469 → 0.431 | n/a | 0.764 → 0.639 · B 0.491 → 0.442 | 0.818 → 0.696 · B 0.602 → 0.502 | 0.645 → 0.622 · B 0.447 → 0.433 | n/a | 0.687 → 0.637 · B 0.473 → 0.446 | 0.753 → 0.659 · B 0.513 → 0.464 | 0.764 → 0.639 · B 0.491 → 0.442 | n/a |
| xnli-noul-zh | 0.528 → 0.497 · B 0.320 → 0.316 | 0.698 → 0.696 · B 0.505 → 0.502 | 0.696 → 0.694 · B 0.503 → 0.501 | 0.698 → 0.695 · B 0.505 → 0.502 | 0.709 → 0.697 · B 0.516 → 0.504 | n/a | 0.916 → 0.661 · B 0.568 → 0.467 | 0.693 → 0.693 · B 0.500 → 0.500 | 0.759 → 0.696 · B 0.562 → 0.503 | 0.703 → 0.614 · B 0.456 → 0.421 | n/a | 0.714 → 0.598 · B 0.432 → 0.403 | 0.887 → 0.700 · B 0.631 → 0.507 | 0.586 → 0.570 · B 0.388 → 0.383 | n/a | 0.615 → 0.587 · B 0.394 → 0.390 | 0.628 → 0.595 · B 0.424 → 0.407 | 0.714 → 0.598 · B 0.432 → 0.403 | n/a |
| amazon-score-en | 1.659 → 1.609 · B 0.821 → 0.800 | 1.614 → 1.610 · B 0.802 → 0.800 | 1.618 → 1.610 · B 0.804 → 0.800 | 1.610 → 1.609 · B 0.800 → 0.800 | 1.609 → 1.609 · B 0.800 → 0.800 | n/a | 1.630 → 1.615 · B 0.808 → 0.802 | 1.597 → 1.606 · B 0.795 → 0.799 | 1.617 → 1.610 · B 0.803 → 0.800 | 1.654 → 1.608 · B 0.818 → 0.800 | n/a | 1.542 → 1.570 · B 0.767 → 0.783 | 1.609 → 1.606 · B 0.800 → 0.799 | 1.714 → 1.598 · B 0.841 → 0.795 | n/a | 1.573 → 1.585 · B 0.785 → 0.790 | 1.586 → 1.578 · B 0.786 → 0.788 | 1.542 → 1.570 · B 0.767 → 0.783 | n/a |
| amazon-score-de | 1.716 → 1.621 · B 0.848 → 0.805 | 1.610 → 1.609 · B 0.800 → 0.800 | 1.612 → 1.609 · B 0.801 → 0.800 | 1.611 → 1.610 · B 0.801 → 0.800 | 1.609 → 1.609 · B 0.800 → 0.800 | n/a | 1.623 → 1.613 · B 0.806 → 0.801 | 1.598 → 1.606 · B 0.795 → 0.799 | 1.614 → 1.610 · B 0.802 → 0.800 | 1.702 → 1.612 · B 0.838 → 0.801 | n/a | 1.578 → 1.587 · B 0.786 → 0.791 | 1.624 → 1.609 · B 0.803 → 0.800 | 1.782 → 1.611 · B 0.877 → 0.801 | n/a | 1.616 → 1.608 · B 0.803 → 0.799 | 1.701 → 1.618 · B 0.819 → 0.802 | 1.578 → 1.587 · B 0.786 → 0.791 | n/a |
| amazon-score-es | 1.684 → 1.616 · B 0.838 → 0.803 | 1.611 → 1.609 · B 0.801 → 0.800 | 1.614 → 1.610 · B 0.802 → 0.800 | 1.609 → 1.609 · B 0.800 → 0.800 | 1.609 → 1.609 · B 0.800 → 0.800 | n/a | 1.621 → 1.612 · B 0.805 → 0.801 | 1.607 → 1.608 · B 0.799 → 0.799 | 1.611 → 1.609 · B 0.801 → 0.800 | 1.634 → 1.609 · B 0.811 → 0.800 | n/a | 1.581 → 1.588 · B 0.788 → 0.791 | 1.648 → 1.615 · B 0.812 → 0.802 | 1.737 → 1.607 · B 0.857 → 0.799 | n/a | 1.587 → 1.593 · B 0.791 → 0.793 | 1.730 → 1.630 · B 0.826 → 0.806 | 1.581 → 1.588 · B 0.788 → 0.791 | n/a |
| amazon-score-fr | 1.729 → 1.622 · B 0.853 → 0.805 | 1.609 → 1.609 · B 0.800 → 0.800 | 1.605 → 1.609 · B 0.798 → 0.800 | 1.610 → 1.609 · B 0.800 → 0.800 | 1.609 → 1.609 · B 0.800 → 0.800 | n/a | 1.617 → 1.611 · B 0.804 → 0.801 | 1.614 → 1.609 · B 0.801 → 0.800 | 1.604 → 1.609 · B 0.798 → 0.800 | 1.701 → 1.612 · B 0.836 → 0.801 | n/a | 1.605 → 1.598 · B 0.798 → 0.796 | 1.656 → 1.617 · B 0.814 → 0.803 | 1.836 → 1.620 · B 0.890 → 0.804 | n/a | 1.578 → 1.588 · B 0.788 → 0.792 | 1.729 → 1.630 · B 0.827 → 0.807 | 1.605 → 1.598 · B 0.798 → 0.796 | n/a |
| amazon-score-ja | 1.719 → 1.620 · B 0.847 → 0.804 | 1.611 → 1.609 · B 0.800 → 0.800 | 1.609 → 1.609 · B 0.800 → 0.800 | 1.610 → 1.609 · B 0.800 → 0.800 | 1.610 → 1.610 · B 0.800 → 0.800 | 1.609 → 1.609 · B 0.800 → 0.800 | 1.615 → 1.611 · B 0.802 → 0.800 | 1.608 → 1.608 · B 0.799 → 0.799 | 1.607 → 1.609 · B 0.799 → 0.800 | 1.696 → 1.611 · B 0.833 → 0.801 | 1.696 → 1.611 · B 0.833 → 0.801 | 1.641 → 1.614 · B 0.813 → 0.802 | 1.630 → 1.610 · B 0.806 → 0.800 | 1.817 → 1.618 · B 0.881 → 0.803 | 1.641 → 1.612 · B 0.813 → 0.801 | 1.593 → 1.596 · B 0.793 → 0.795 | 1.717 → 1.624 · B 0.825 → 0.805 | 1.641 → 1.614 · B 0.813 → 0.802 | 1.641 → 1.612 · B 0.813 → 0.801 |
| amazon-score-zh | 1.683 → 1.616 · B 0.834 → 0.803 | 1.615 → 1.610 · B 0.802 → 0.800 | 1.614 → 1.610 · B 0.802 → 0.800 | 1.610 → 1.609 · B 0.800 → 0.800 | 1.610 → 1.610 · B 0.800 → 0.800 | n/a | 1.621 → 1.613 · B 0.805 → 0.801 | 1.603 → 1.607 · B 0.798 → 0.799 | 1.617 → 1.610 · B 0.803 → 0.800 | 1.675 → 1.610 · B 0.827 → 0.800 | n/a | 1.633 → 1.613 · B 0.809 → 0.802 | 1.633 → 1.611 · B 0.806 → 0.800 | 1.778 → 1.617 · B 0.875 → 0.803 | n/a | 1.596 → 1.598 · B 0.795 → 0.795 | 1.678 → 1.611 · B 0.816 → 0.801 | 1.633 → 1.613 · B 0.809 → 0.802 | n/a |

### Option-order invariance (100 cases per suite)

| Engine | Suite | Full-perm flip | Reverse flip | Max prob dev | Errors |
|---|---|---|---|---|---|
| arch-b | agnews-choice | 0.125 | 0.130 | 0.6371 | 0% |
| arch-b | banking77-choice | 0.059 | 0.080 | 0.6867 | 0% |
| arch-b | massive-choice-en | 0.025 | 0.040 | 0.6132 | 0% |
| arch-b | massive-choice-ko | 0.038 | 0.060 | 0.7813 | 0% |
| arch-d1 | agnews-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-d1 | banking77-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-d1 | massive-choice-en | 0.000 | 0.000 | 0.0000 | 0% |
| arch-d1 | massive-choice-ko | 0.000 | 0.000 | 0.0000 | 0% |
| arch-d2 | agnews-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-d2 | banking77-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-d2 | massive-choice-en | 0.000 | 0.000 | 0.0000 | 0% |
| arch-d2 | massive-choice-ko | 0.000 | 0.000 | 0.0000 | 0% |
| arch-d4 | agnews-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-d4 | banking77-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-d4 | massive-choice-en | 0.000 | 0.000 | 0.0000 | 0% |
| arch-d4 | massive-choice-ko | 0.000 | 0.000 | 0.0000 | 0% |
| arch-d2-emb | agnews-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-d2-emb | banking77-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-d2-emb | massive-choice-en | 0.000 | 0.000 | 0.0000 | 0% |
| arch-d2-emb | massive-choice-ko | 0.000 | 0.000 | 0.0000 | 0% |
| arch-d1-pool | agnews-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-d1-pool | banking77-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-d1-pool | massive-choice-en | 0.000 | 0.000 | 0.0000 | 0% |
| arch-d1-pool | massive-choice-ko | 0.000 | 0.000 | 0.0000 | 0% |
| arch-d1-lr | agnews-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-d1-lr | banking77-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-d1-lr | massive-choice-en | 0.000 | 0.000 | 0.0000 | 0% |
| arch-d1-lr | massive-choice-ko | 0.000 | 0.000 | 0.0000 | 0% |
| arch-d1-set | agnews-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-d1-set | banking77-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-d1-set | massive-choice-en | 0.000 | 0.000 | 0.0000 | 0% |
| arch-d1-set | massive-choice-ko | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late4 | agnews-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late4 | banking77-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late4 | massive-choice-en | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late4 | massive-choice-ko | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8 | agnews-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8 | banking77-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8 | massive-choice-en | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8 | massive-choice-ko | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late4-s14 | agnews-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late4-s14 | banking77-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late4-s14 | massive-choice-en | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late4-s14 | massive-choice-ko | 0.000 | 0.000 | 0.0000 | 0% |
| arch-b-s14 | agnews-choice | 0.131 | 0.160 | 0.6280 | 0% |
| arch-b-s14 | banking77-choice | 0.038 | 0.030 | 0.5793 | 0% |
| arch-b-s14 | massive-choice-en | 0.014 | 0.020 | 0.5203 | 0% |
| arch-b-s14 | massive-choice-ko | 0.020 | 0.000 | 0.7784 | 0% |
| arch-late6 | agnews-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late6 | banking77-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late6 | massive-choice-en | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late6 | massive-choice-ko | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-s14 | agnews-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-s14 | banking77-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-s14 | massive-choice-en | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-s14 | massive-choice-ko | 0.000 | 0.000 | 0.0000 | 0% |
| krite | agnews-choice | 0.000 | 0.000 | 0.0000 | 0% |
| krite | banking77-choice | 0.000 | 0.000 | 0.0000 | 0% |
| krite | massive-choice-en | 0.000 | 0.000 | 0.0000 | 0% |
| krite | massive-choice-ko | 0.000 | 0.000 | 0.0000 | 0% |

### Question interference (agnews-choice, 100 cases; alone vs. with fillers)

| Engine | Fillers | Argmax change | Max prob dev |
|---|---|---|---|
| arch-b | 3 | 0.000 | 0.0000 |
| arch-b | 15 | 0.000 | 0.0000 |
| arch-d1 | 3 | 0.000 | 0.0000 |
| arch-d1 | 15 | 0.000 | 0.0000 |
| arch-d2 | 3 | 0.000 | 0.0000 |
| arch-d2 | 15 | 0.000 | 0.0000 |
| arch-d4 | 3 | 0.000 | 0.0000 |
| arch-d4 | 15 | 0.000 | 0.0000 |
| arch-d2-emb | 3 | 0.000 | 0.0000 |
| arch-d2-emb | 15 | 0.000 | 0.0000 |
| arch-d1-pool | 3 | 0.000 | 0.0000 |
| arch-d1-pool | 15 | 0.000 | 0.0000 |
| arch-d1-lr | 3 | 0.000 | 0.0000 |
| arch-d1-lr | 15 | 0.000 | 0.0000 |
| arch-d1-set | 3 | 0.000 | 0.0000 |
| arch-d1-set | 15 | 0.000 | 0.0000 |
| arch-late4 | 3 | 0.000 | 0.0000 |
| arch-late4 | 15 | 0.000 | 0.0000 |
| arch-late8 | 3 | 0.000 | 0.0000 |
| arch-late8 | 15 | 0.000 | 0.0000 |
| arch-late4-s14 | 3 | 0.000 | 0.0000 |
| arch-late4-s14 | 15 | 0.000 | 0.0000 |
| arch-b-s14 | 3 | 0.000 | 0.0000 |
| arch-b-s14 | 15 | 0.000 | 0.0000 |
| arch-late6 | 3 | 0.000 | 0.0000 |
| arch-late6 | 15 | 0.000 | 0.0000 |
| arch-late8-s14 | 3 | 0.000 | 0.0000 |
| arch-late8-s14 | 15 | 0.000 | 0.0000 |
| krite | 3 | 0.000 | 0.0000 |
| krite | 15 | 0.000 | 0.0000 |

### Latency (MacBook Air M4, concurrency 1)

| Engine | Mode | Cache | State/Q/K | Layer | n | p50 ms | p95 ms | p99 ms | dec/s | Errors | Power | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| arch-b | burst | cold | 512/1/4 | http | 200 | 44.3 | 49.3 | 50.1 | 22.0 | 0% | AC |  |
| arch-b | burst | cold | 512/1/4 | runtime | 200 | 43.6 | 46.7 | 47.1 | 22.0 | 0% | AC |  |
| arch-b-s14 | burst | cold | 512/1/4 | http | 200 | 44.3 | 49.2 | 54.2 | 22.1 | 0% | AC |  |
| arch-b-s14 | burst | cold | 512/1/4 | runtime | 200 | 43.7 | 47.4 | 52.6 | 22.1 | 0% | AC |  |
| arch-d1 | burst | cold | 512/1/4 | http | 200 | 51.9 | 52.3 | 62.6 | 19.1 | 0% | AC |  |
| arch-d1 | burst | cold | 512/1/4 | runtime | 200 | 51.3 | 51.7 | 60.4 | 19.1 | 0% | AC |  |
| arch-d1-lr | burst | cold | 512/1/4 | http | 200 | 52.1 | 56.0 | 57.7 | 19.1 | 0% | AC |  |
| arch-d1-lr | burst | cold | 512/1/4 | runtime | 200 | 51.5 | 55.0 | 56.3 | 19.1 | 0% | AC |  |
| arch-d1-pool | burst | cold | 512/1/4 | http | 200 | 52.5 | 58.8 | 60.4 | 18.7 | 0% | AC |  |
| arch-d1-pool | burst | cold | 512/1/4 | runtime | 200 | 51.8 | 57.3 | 58.6 | 18.7 | 0% | AC |  |
| arch-d1-set | burst | cold | 512/1/4 | http | 200 | 52.1 | 54.8 | 61.6 | 19.1 | 0% | AC |  |
| arch-d1-set | burst | cold | 512/1/4 | runtime | 200 | 51.5 | 53.8 | 60.1 | 19.1 | 0% | AC |  |
| arch-d2 | burst | cold | 512/1/4 | http | 200 | 53.1 | 53.4 | 55.7 | 18.8 | 0% | AC |  |
| arch-d2 | burst | cold | 512/1/4 | runtime | 200 | 52.5 | 52.8 | 54.8 | 18.8 | 0% | AC |  |
| arch-d2-emb | burst | cold | 512/1/4 | http | 200 | 48.5 | 49.3 | 49.6 | 21.0 | 0% | AC |  |
| arch-d2-emb | burst | cold | 512/1/4 | runtime | 200 | 46.4 | 47.2 | 47.3 | 21.0 | 0% | AC |  |
| arch-d2-nocache | burst | cold | 512/1/4 | http | 200 | 53.1 | 53.4 | 59.8 | 18.8 | 0% | AC |  |
| arch-d2-nocache | burst | cold | 512/1/4 | runtime | 200 | 52.5 | 52.8 | 58.5 | 18.8 | 0% | AC |  |
| arch-d4 | burst | cold | 512/1/4 | http | 200 | 61.1 | 63.1 | 63.7 | 17.0 | 0% | AC |  |
| arch-d4 | burst | cold | 512/1/4 | runtime | 200 | 59.8 | 61.6 | 62.4 | 17.0 | 0% | AC |  |
| arch-late4 | burst | cold | 512/1/4 | http | 200 | 49.6 | 60.3 | 62.9 | 19.8 | 0% | AC |  |
| arch-late4 | burst | cold | 512/1/4 | runtime | 200 | 49.0 | 57.8 | 59.7 | 19.8 | 0% | AC |  |
| arch-late4-nocache | burst | cold | 512/1/4 | http | 200 | 58.2 | 58.5 | 58.6 | 17.2 | 0% | AC |  |
| arch-late4-nocache | burst | cold | 512/1/4 | runtime | 200 | 57.6 | 57.9 | 58.0 | 17.2 | 0% | AC |  |
| arch-late4-s14 | burst | cold | 512/1/4 | http | 200 | 49.5 | 49.7 | 50.3 | 20.2 | 0% | AC |  |
| arch-late4-s14 | burst | cold | 512/1/4 | runtime | 200 | 48.9 | 49.1 | 49.6 | 20.2 | 0% | AC |  |
| arch-late6 | burst | cold | 512/1/4 | http | 200 | 55.1 | 55.6 | 56.3 | 18.1 | 0% | AC |  |
| arch-late6 | burst | cold | 512/1/4 | runtime | 200 | 54.4 | 54.9 | 55.4 | 18.1 | 0% | AC |  |
| arch-late8 | burst | cold | 512/1/4 | http | 200 | 56.1 | 56.8 | 76.1 | 17.7 | 0% | AC |  |
| arch-late8 | burst | cold | 512/1/4 | runtime | 200 | 55.4 | 56.0 | 75.3 | 17.7 | 0% | AC |  |
| arch-late8-nocache | burst | cold | 512/1/4 | http | 200 | 63.4 | 73.0 | 76.8 | 15.5 | 0% | AC |  |
| arch-late8-nocache | burst | cold | 512/1/4 | runtime | 200 | 62.7 | 71.7 | 75.4 | 15.5 | 0% | AC |  |
| arch-late8-s14 | burst | cold | 512/1/4 | http | 200 | 60.9 | 64.9 | 79.7 | 16.2 | 0% | AC |  |
| arch-late8-s14 | burst | cold | 512/1/4 | runtime | 200 | 60.2 | 64.1 | 78.9 | 16.2 | 0% | AC |  |
| krite | burst | cold | 512/1/4 | http | 200 | 91.3 | 94.6 | 96.9 | 10.9 | 0% | AC |  |
| krite | burst | cold | 512/1/4 | runtime | 200 | 90.9 | 94.0 | 95.9 | 10.9 | 0% | AC |  |
| krite-nocache | burst | cold | 512/1/4 | http | 200 | 102.0 | 103.8 | 104.4 | 9.8 | 0% | AC |  |
| krite-nocache | burst | cold | 512/1/4 | runtime | 200 | 100.9 | 102.6 | 103.3 | 9.8 | 0% | AC |  |
| arch-b | burst | cold | 512/10/4 | http | 200 | 402.7 | 417.9 | 422.4 | 24.7 | 0% | AC |  |
| arch-b | burst | cold | 512/10/4 | runtime | 200 | 401.7 | 416.9 | 421.3 | 24.7 | 0% | AC |  |
| arch-b-s14 | burst | cold | 512/10/4 | http | 200 | 415.0 | 442.5 | 456.3 | 24.0 | 0% | AC |  |
| arch-b-s14 | burst | cold | 512/10/4 | runtime | 200 | 414.2 | 441.1 | 455.1 | 24.0 | 0% | AC |  |
| arch-d1 | burst | cold | 512/10/4 | http | 200 | 98.3 | 98.6 | 100.0 | 101.7 | 0% | AC |  |
| arch-d1 | burst | cold | 512/10/4 | runtime | 200 | 97.6 | 97.9 | 99.0 | 101.7 | 0% | AC |  |
| arch-d1-lr | burst | cold | 512/10/4 | http | 200 | 98.3 | 98.5 | 99.6 | 101.8 | 0% | AC |  |
| arch-d1-lr | burst | cold | 512/10/4 | runtime | 200 | 97.6 | 97.8 | 98.8 | 101.8 | 0% | AC |  |
| arch-d1-pool | burst | cold | 512/10/4 | http | 200 | 98.7 | 107.4 | 111.5 | 100.2 | 0% | AC |  |
| arch-d1-pool | burst | cold | 512/10/4 | runtime | 200 | 97.9 | 105.4 | 109.5 | 100.2 | 0% | AC |  |
| arch-d1-set | burst | cold | 512/10/4 | http | 200 | 98.4 | 98.7 | 100.3 | 101.6 | 0% | AC |  |
| arch-d1-set | burst | cold | 512/10/4 | runtime | 200 | 97.7 | 98.0 | 99.3 | 101.6 | 0% | AC |  |
| arch-d2 | burst | cold | 512/10/4 | http | 200 | 108.2 | 109.1 | 114.3 | 92.2 | 0% | AC |  |
| arch-d2 | burst | cold | 512/10/4 | runtime | 200 | 107.5 | 108.2 | 112.9 | 92.2 | 0% | AC |  |
| arch-d2-emb | burst | cold | 512/10/4 | http | 200 | 65.1 | 65.3 | 65.4 | 153.6 | 0% | AC |  |
| arch-d2-emb | burst | cold | 512/10/4 | runtime | 200 | 64.5 | 64.6 | 64.7 | 153.6 | 0% | AC |  |
| arch-d2-nocache | burst | cold | 512/10/4 | http | 200 | 108.5 | 110.8 | 112.7 | 92.0 | 0% | AC |  |
| arch-d2-nocache | burst | cold | 512/10/4 | runtime | 200 | 107.8 | 110.1 | 112.0 | 92.0 | 0% | AC |  |
| arch-d4 | burst | cold | 512/10/4 | http | 200 | 129.0 | 130.9 | 132.2 | 77.5 | 0% | AC |  |
| arch-d4 | burst | cold | 512/10/4 | runtime | 200 | 128.3 | 130.1 | 131.5 | 77.5 | 0% | AC |  |
| arch-late4 | burst | cold | 512/10/4 | http | 200 | 80.5 | 88.7 | 94.9 | 122.8 | 0% | AC |  |
| arch-late4 | burst | cold | 512/10/4 | runtime | 200 | 79.9 | 86.8 | 91.6 | 122.8 | 0% | AC |  |
| arch-late4-nocache | burst | cold | 512/10/4 | http | 200 | 117.2 | 119.5 | 123.4 | 85.1 | 0% | AC |  |
| arch-late4-nocache | burst | cold | 512/10/4 | runtime | 200 | 116.5 | 118.8 | 122.4 | 85.1 | 0% | AC |  |
| arch-late4-s14 | burst | cold | 512/10/4 | http | 200 | 80.5 | 82.4 | 83.8 | 123.9 | 0% | AC |  |
| arch-late4-s14 | burst | cold | 512/10/4 | runtime | 200 | 79.8 | 81.8 | 83.1 | 123.9 | 0% | AC |  |
| arch-late6 | burst | cold | 512/10/4 | http | 200 | 100.5 | 117.2 | 118.8 | 97.7 | 0% | AC |  |
| arch-late6 | burst | cold | 512/10/4 | runtime | 200 | 99.7 | 113.4 | 115.1 | 97.7 | 0% | AC |  |
| arch-late8 | burst | cold | 512/10/4 | http | 200 | 76.2 | 77.2 | 82.8 | 130.9 | 0% | AC |  |
| arch-late8 | burst | cold | 512/10/4 | runtime | 200 | 75.4 | 76.3 | 81.5 | 130.9 | 0% | AC |  |
| arch-late8-nocache | burst | cold | 512/10/4 | http | 200 | 105.1 | 105.6 | 126.3 | 94.7 | 0% | AC |  |
| arch-late8-nocache | burst | cold | 512/10/4 | runtime | 200 | 104.3 | 104.7 | 124.5 | 94.7 | 0% | AC |  |
| arch-late8-s14 | burst | cold | 512/10/4 | http | 200 | 120.7 | 140.2 | 142.4 | 80.7 | 0% | AC |  |
| arch-late8-s14 | burst | cold | 512/10/4 | runtime | 200 | 120.0 | 137.2 | 139.2 | 80.7 | 0% | AC |  |
| krite | burst | cold | 512/10/4 | http | 200 | 127.2 | 133.9 | 137.1 | 78.0 | 0% | AC |  |
| krite | burst | cold | 512/10/4 | runtime | 200 | 126.8 | 132.9 | 135.5 | 78.0 | 0% | AC |  |
| krite-nocache | burst | cold | 512/10/4 | http | 200 | 196.9 | 199.2 | 200.2 | 50.8 | 0% | AC |  |
| krite-nocache | burst | cold | 512/10/4 | runtime | 200 | 195.6 | 197.8 | 198.8 | 50.8 | 0% | AC |  |
| arch-b | burst | cold | 512/30/4 | http | 200 | 1737.1 | 1818.4 | 1941.6 | 17.3 | 0% | AC |  |
| arch-b | burst | cold | 512/30/4 | runtime | 200 | 1735.7 | 1816.9 | 1940.0 | 17.3 | 0% | AC |  |
| arch-b-s14 | burst | cold | 512/30/4 | http | 200 | 1977.7 | 2693.2 | 3215.4 | 14.6 | 0% | AC |  |
| arch-b-s14 | burst | cold | 512/30/4 | runtime | 200 | 1976.0 | 2689.8 | 3213.5 | 14.6 | 0% | AC |  |
| arch-d1 | burst | cold | 512/30/4 | http | 200 | 213.0 | 218.3 | 228.1 | 140.4 | 0% | AC |  |
| arch-d1 | burst | cold | 512/30/4 | runtime | 200 | 212.2 | 217.5 | 225.2 | 140.4 | 0% | AC |  |
| arch-d1-lr | burst | cold | 512/30/4 | http | 200 | 213.5 | 215.4 | 220.3 | 140.3 | 0% | AC |  |
| arch-d1-lr | burst | cold | 512/30/4 | runtime | 200 | 212.8 | 214.5 | 219.5 | 140.3 | 0% | AC |  |
| arch-d1-pool | burst | cold | 512/30/4 | http | 200 | 214.4 | 218.5 | 219.9 | 139.7 | 0% | AC |  |
| arch-d1-pool | burst | cold | 512/30/4 | runtime | 200 | 213.5 | 217.6 | 218.6 | 139.7 | 0% | AC |  |
| arch-d1-set | burst | cold | 512/30/4 | http | 200 | 213.9 | 214.4 | 218.8 | 140.2 | 0% | AC |  |
| arch-d1-set | burst | cold | 512/30/4 | runtime | 200 | 213.2 | 213.6 | 217.9 | 140.2 | 0% | AC |  |
| arch-d2 | burst | cold | 512/30/4 | http | 200 | 241.6 | 255.6 | 257.9 | 123.3 | 0% | AC |  |
| arch-d2 | burst | cold | 512/30/4 | runtime | 200 | 240.7 | 253.6 | 254.7 | 123.3 | 0% | AC |  |
| arch-d2-emb | burst | cold | 512/30/4 | http | 200 | 110.8 | 113.7 | 114.7 | 270.1 | 0% | AC |  |
| arch-d2-emb | burst | cold | 512/30/4 | runtime | 200 | 110.1 | 113.0 | 113.9 | 270.1 | 0% | AC |  |
| arch-d2-nocache | burst | cold | 512/30/4 | http | 200 | 241.5 | 242.1 | 247.0 | 124.1 | 0% | AC |  |
| arch-d2-nocache | burst | cold | 512/30/4 | runtime | 200 | 240.8 | 241.4 | 246.2 | 124.1 | 0% | AC |  |
| arch-d4 | burst | cold | 512/30/4 | http | 200 | 298.9 | 308.1 | 309.4 | 99.8 | 0% | AC |  |
| arch-d4 | burst | cold | 512/30/4 | runtime | 200 | 298.1 | 307.3 | 308.6 | 99.8 | 0% | AC |  |
| arch-late4 | burst | cold | 512/30/4 | http | 200 | 153.4 | 168.9 | 170.3 | 190.4 | 0% | AC |  |
| arch-late4 | burst | cold | 512/30/4 | runtime | 200 | 152.6 | 166.8 | 167.9 | 190.4 | 0% | AC |  |
| arch-late4-nocache | burst | cold | 512/30/4 | http | 200 | 261.7 | 283.0 | 291.6 | 113.9 | 0% | AC |  |
| arch-late4-nocache | burst | cold | 512/30/4 | runtime | 200 | 260.9 | 280.6 | 288.8 | 113.9 | 0% | AC |  |
| arch-late4-s14 | burst | cold | 512/30/4 | http | 200 | 153.3 | 153.7 | 153.9 | 195.6 | 0% | AC |  |
| arch-late4-s14 | burst | cold | 512/30/4 | runtime | 200 | 152.6 | 152.9 | 153.2 | 195.6 | 0% | AC |  |
| arch-late6 | burst | cold | 512/30/4 | http | 200 | 206.6 | 214.0 | 225.8 | 144.6 | 0% | AC |  |
| arch-late6 | burst | cold | 512/30/4 | runtime | 200 | 205.6 | 213.0 | 223.0 | 144.6 | 0% | AC |  |
| arch-late8 | burst | cold | 512/30/4 | http | 200 | 135.3 | 139.1 | 142.5 | 221.1 | 0% | AC |  |
| arch-late8 | burst | cold | 512/30/4 | runtime | 200 | 134.5 | 138.2 | 141.6 | 221.1 | 0% | AC |  |
| arch-late8-nocache | burst | cold | 512/30/4 | http | 200 | 220.4 | 224.6 | 252.3 | 135.6 | 0% | AC |  |
| arch-late8-nocache | burst | cold | 512/30/4 | runtime | 200 | 219.4 | 223.7 | 250.8 | 135.6 | 0% | AC |  |
| arch-late8-s14 | burst | cold | 512/30/4 | http | 200 | 260.0 | 267.5 | 280.1 | 114.9 | 0% | AC |  |
| arch-late8-s14 | burst | cold | 512/30/4 | runtime | 200 | 259.1 | 266.6 | 277.2 | 114.9 | 0% | AC |  |
| krite | burst | cold | 512/30/4 | http | 200 | 227.9 | 242.9 | 244.4 | 129.9 | 0% | AC |  |
| krite | burst | cold | 512/30/4 | runtime | 200 | 227.3 | 241.3 | 242.8 | 129.9 | 0% | AC |  |
| krite-nocache | burst | cold | 512/30/4 | http | 200 | 483.4 | 580.2 | 657.4 | 59.4 | 0% | AC |  |
| krite-nocache | burst | cold | 512/30/4 | runtime | 200 | 481.9 | 579.4 | 655.5 | 59.4 | 0% | AC |  |
| arch-b | burst | startup | ping probe | http | 1 | 11727.9 | 11727.9 | 11727.9 | 0.1 | 0% | AC |  |
| arch-b-s14 | burst | startup | ping probe | http | 1 | 11281.6 | 11281.6 | 11281.6 | 0.1 | 0% | AC |  |
| arch-d1 | burst | startup | ping probe | http | 1 | 11322.0 | 11322.0 | 11322.0 | 0.1 | 0% | AC |  |
| arch-d1-lr | burst | startup | ping probe | http | 1 | 11977.0 | 11977.0 | 11977.0 | 0.1 | 0% | AC |  |
| arch-d1-pool | burst | startup | ping probe | http | 1 | 13111.3 | 13111.3 | 13111.3 | 0.1 | 0% | AC |  |
| arch-d1-set | burst | startup | ping probe | http | 1 | 12000.7 | 12000.7 | 12000.7 | 0.1 | 0% | AC |  |
| arch-d2 | burst | startup | ping probe | http | 1 | 10702.5 | 10702.5 | 10702.5 | 0.1 | 0% | AC |  |
| arch-d2-emb | burst | startup | ping probe | http | 1 | 11772.1 | 11772.1 | 11772.1 | 0.1 | 0% | AC |  |
| arch-d2-nocache | burst | startup | ping probe | http | 1 | 10965.8 | 10965.8 | 10965.8 | 0.1 | 0% | AC |  |
| arch-d4 | burst | startup | ping probe | http | 1 | 11331.9 | 11331.9 | 11331.9 | 0.1 | 0% | AC |  |
| arch-late4 | burst | startup | ping probe | http | 1 | 12673.7 | 12673.7 | 12673.7 | 0.1 | 0% | AC |  |
| arch-late4-nocache | burst | startup | ping probe | http | 1 | 10647.2 | 10647.2 | 10647.2 | 0.1 | 0% | AC |  |
| arch-late4-s14 | burst | startup | ping probe | http | 1 | 11737.6 | 11737.6 | 11737.6 | 0.1 | 0% | AC |  |
| arch-late6 | burst | startup | ping probe | http | 1 | 10448.4 | 10448.4 | 10448.4 | 0.1 | 0% | AC |  |
| arch-late8 | burst | startup | ping probe | http | 1 | 12424.2 | 12424.2 | 12424.2 | 0.1 | 0% | AC |  |
| arch-late8-nocache | burst | startup | ping probe | http | 1 | 11084.6 | 11084.6 | 11084.6 | 0.1 | 0% | AC |  |
| arch-late8-s14 | burst | startup | ping probe | http | 1 | 11687.5 | 11687.5 | 11687.5 | 0.1 | 0% | AC |  |
| krite | burst | startup | ping probe | http | 1 | 523.6 | 523.6 | 523.6 | 1.9 | 0% | AC |  |
| krite-nocache | burst | startup | ping probe | http | 1 | 1383.5 | 1383.5 | 1383.5 | 0.7 | 0% | AC |  |
| arch-b | burst | warm | 512/1/4 | http | 200 | 61.3 | 64.8 | 66.6 | 16.2 | 0% | AC |  |
| arch-b | burst | warm | 512/1/4 | runtime | 200 | 58.8 | 61.1 | 62.6 | 16.2 | 0% | AC |  |
| arch-b-s14 | burst | warm | 512/1/4 | http | 200 | 71.3 | 81.4 | 83.9 | 13.9 | 0% | AC |  |
| arch-b-s14 | burst | warm | 512/1/4 | runtime | 200 | 68.2 | 78.5 | 80.7 | 13.9 | 0% | AC |  |
| arch-d1 | burst | warm | 512/1/4 | http | 200 | 14.5 | 15.2 | 15.4 | 69.2 | 0% | AC |  |
| arch-d1 | burst | warm | 512/1/4 | runtime | 200 | 13.9 | 14.5 | 14.8 | 69.2 | 0% | AC |  |
| arch-d1-lr | burst | warm | 512/1/4 | http | 200 | 14.5 | 15.3 | 15.5 | 68.8 | 0% | AC |  |
| arch-d1-lr | burst | warm | 512/1/4 | runtime | 200 | 13.9 | 14.6 | 14.9 | 68.8 | 0% | AC |  |
| arch-d1-pool | burst | warm | 512/1/4 | http | 200 | 14.6 | 15.4 | 15.8 | 68.5 | 0% | AC |  |
| arch-d1-pool | burst | warm | 512/1/4 | runtime | 200 | 14.0 | 14.8 | 15.2 | 68.5 | 0% | AC |  |
| arch-d1-set | burst | warm | 512/1/4 | http | 200 | 14.6 | 15.4 | 15.7 | 68.6 | 0% | AC |  |
| arch-d1-set | burst | warm | 512/1/4 | runtime | 200 | 13.9 | 14.8 | 14.9 | 68.6 | 0% | AC |  |
| arch-d2 | burst | warm | 512/1/4 | http | 200 | 14.9 | 15.1 | 15.5 | 66.9 | 0% | AC |  |
| arch-d2 | burst | warm | 512/1/4 | runtime | 200 | 14.3 | 14.5 | 14.9 | 66.9 | 0% | AC |  |
| arch-d2-emb | burst | warm | 512/1/4 | http | 200 | 5.3 | 9.6 | 11.5 | 168.4 | 0% | AC |  |
| arch-d2-emb | burst | warm | 512/1/4 | runtime | 200 | 4.7 | 8.3 | 10.3 | 168.4 | 0% | AC |  |
| arch-d2-nocache | burst | warm | 512/1/4 | http | 200 | 52.6 | 53.5 | 60.5 | 18.9 | 0% | AC |  |
| arch-d2-nocache | burst | warm | 512/1/4 | runtime | 200 | 51.9 | 52.8 | 58.6 | 18.9 | 0% | AC |  |
| arch-d4 | burst | warm | 512/1/4 | http | 200 | 17.0 | 17.9 | 18.7 | 58.1 | 0% | AC |  |
| arch-d4 | burst | warm | 512/1/4 | runtime | 200 | 16.4 | 17.3 | 18.0 | 58.1 | 0% | AC |  |
| arch-late4 | burst | warm | 512/1/4 | http | 200 | 7.5 | 7.6 | 7.7 | 134.0 | 0% | AC |  |
| arch-late4 | burst | warm | 512/1/4 | runtime | 200 | 6.9 | 7.0 | 7.1 | 134.0 | 0% | AC |  |
| arch-late4-nocache | burst | warm | 512/1/4 | http | 200 | 58.7 | 59.0 | 60.6 | 17.0 | 0% | AC |  |
| arch-late4-nocache | burst | warm | 512/1/4 | runtime | 200 | 58.1 | 58.4 | 59.8 | 17.0 | 0% | AC |  |
| arch-late4-s14 | burst | warm | 512/1/4 | http | 200 | 7.5 | 7.6 | 7.8 | 133.8 | 0% | AC |  |
| arch-late4-s14 | burst | warm | 512/1/4 | runtime | 200 | 6.9 | 7.0 | 7.2 | 133.8 | 0% | AC |  |
| arch-late6 | burst | warm | 512/1/4 | http | 200 | 9.6 | 10.8 | 10.9 | 102.2 | 0% | AC |  |
| arch-late6 | burst | warm | 512/1/4 | runtime | 200 | 9.0 | 10.2 | 10.3 | 102.2 | 0% | AC |  |
| arch-late8 | burst | warm | 512/1/4 | http | 200 | 6.8 | 7.3 | 7.6 | 143.6 | 0% | AC |  |
| arch-late8 | burst | warm | 512/1/4 | runtime | 200 | 6.2 | 6.7 | 6.9 | 143.6 | 0% | AC |  |
| arch-late8-nocache | burst | warm | 512/1/4 | http | 200 | 63.5 | 64.3 | 65.0 | 15.7 | 0% | AC |  |
| arch-late8-nocache | burst | warm | 512/1/4 | runtime | 200 | 62.9 | 63.6 | 64.0 | 15.7 | 0% | AC |  |
| arch-late8-s14 | burst | warm | 512/1/4 | http | 200 | 12.8 | 13.4 | 13.6 | 78.1 | 0% | AC |  |
| arch-late8-s14 | burst | warm | 512/1/4 | runtime | 200 | 12.2 | 12.8 | 12.9 | 78.1 | 0% | AC |  |
| krite | burst | warm | 512/1/4 | http | 200 | 7.8 | 8.1 | 8.5 | 128.0 | 0% | AC |  |
| krite | burst | warm | 512/1/4 | runtime | 200 | 7.6 | 7.8 | 8.3 | 128.0 | 0% | AC |  |
| krite-nocache | burst | warm | 512/1/4 | http | 200 | 125.6 | 127.9 | 131.7 | 8.0 | 0% | AC |  |
| krite-nocache | burst | warm | 512/1/4 | runtime | 200 | 124.3 | 126.6 | 130.3 | 8.0 | 0% | AC |  |
| arch-b | burst | warm | 512/10/4 | http | 200 | 589.9 | 607.6 | 651.3 | 16.9 | 0% | AC |  |
| arch-b | burst | warm | 512/10/4 | runtime | 200 | 588.6 | 606.0 | 649.9 | 16.9 | 0% | AC |  |
| arch-b-s14 | burst | warm | 512/10/4 | http | 200 | 733.6 | 1048.7 | 1308.7 | 12.9 | 0% | AC |  |
| arch-b-s14 | burst | warm | 512/10/4 | runtime | 200 | 730.4 | 1047.2 | 1303.7 | 12.9 | 0% | AC |  |
| arch-d1 | burst | warm | 512/10/4 | http | 200 | 61.4 | 61.6 | 61.8 | 162.9 | 0% | AC |  |
| arch-d1 | burst | warm | 512/10/4 | runtime | 200 | 60.7 | 60.9 | 61.0 | 162.9 | 0% | AC |  |
| arch-d1-lr | burst | warm | 512/10/4 | http | 200 | 61.6 | 61.8 | 62.1 | 162.4 | 0% | AC |  |
| arch-d1-lr | burst | warm | 512/10/4 | runtime | 200 | 60.9 | 61.1 | 61.2 | 162.4 | 0% | AC |  |
| arch-d1-pool | burst | warm | 512/10/4 | http | 200 | 63.2 | 65.4 | 66.0 | 158.1 | 0% | AC |  |
| arch-d1-pool | burst | warm | 512/10/4 | runtime | 200 | 62.5 | 64.7 | 65.3 | 158.1 | 0% | AC |  |
| arch-d1-set | burst | warm | 512/10/4 | http | 200 | 61.7 | 63.7 | 64.6 | 161.4 | 0% | AC |  |
| arch-d1-set | burst | warm | 512/10/4 | runtime | 200 | 61.1 | 63.1 | 63.9 | 161.4 | 0% | AC |  |
| arch-d2 | burst | warm | 512/10/4 | http | 200 | 73.0 | 74.0 | 74.6 | 137.4 | 0% | AC |  |
| arch-d2 | burst | warm | 512/10/4 | runtime | 200 | 72.3 | 73.3 | 73.9 | 137.4 | 0% | AC |  |
| arch-d2-emb | burst | warm | 512/10/4 | http | 200 | 27.8 | 27.9 | 27.9 | 360.1 | 0% | AC |  |
| arch-d2-emb | burst | warm | 512/10/4 | runtime | 200 | 27.1 | 27.3 | 27.3 | 360.1 | 0% | AC |  |
| arch-d2-nocache | burst | warm | 512/10/4 | http | 200 | 109.0 | 111.8 | 112.7 | 91.3 | 0% | AC |  |
| arch-d2-nocache | burst | warm | 512/10/4 | runtime | 200 | 108.3 | 111.2 | 112.0 | 91.3 | 0% | AC |  |
| arch-d4 | burst | warm | 512/10/4 | http | 200 | 95.9 | 105.2 | 106.0 | 102.8 | 0% | AC |  |
| arch-d4 | burst | warm | 512/10/4 | runtime | 200 | 95.1 | 104.5 | 105.3 | 102.8 | 0% | AC |  |
| arch-late4 | burst | warm | 512/10/4 | http | 200 | 39.3 | 39.5 | 39.8 | 254.5 | 0% | AC |  |
| arch-late4 | burst | warm | 512/10/4 | runtime | 200 | 38.6 | 38.8 | 39.1 | 254.5 | 0% | AC |  |
| arch-late4-nocache | burst | warm | 512/10/4 | http | 200 | 117.3 | 120.3 | 120.7 | 84.9 | 0% | AC |  |
| arch-late4-nocache | burst | warm | 512/10/4 | runtime | 200 | 116.6 | 119.6 | 120.1 | 84.9 | 0% | AC |  |
| arch-late4-s14 | burst | warm | 512/10/4 | http | 200 | 39.3 | 39.4 | 39.5 | 254.6 | 0% | AC |  |
| arch-late4-s14 | burst | warm | 512/10/4 | runtime | 200 | 38.6 | 38.8 | 38.8 | 254.6 | 0% | AC |  |
| arch-late6 | burst | warm | 512/10/4 | http | 200 | 56.0 | 57.6 | 58.9 | 177.7 | 0% | AC |  |
| arch-late6 | burst | warm | 512/10/4 | runtime | 200 | 55.3 | 56.4 | 57.4 | 177.7 | 0% | AC |  |
| arch-late8 | burst | warm | 512/10/4 | http | 200 | 28.2 | 28.5 | 28.8 | 354.6 | 0% | AC |  |
| arch-late8 | burst | warm | 512/10/4 | runtime | 200 | 27.5 | 27.7 | 28.0 | 354.6 | 0% | AC |  |
| arch-late8-nocache | burst | warm | 512/10/4 | http | 200 | 104.7 | 105.1 | 110.7 | 95.4 | 0% | AC |  |
| arch-late8-nocache | burst | warm | 512/10/4 | runtime | 200 | 103.9 | 104.3 | 109.6 | 95.4 | 0% | AC |  |
| arch-late8-s14 | burst | warm | 512/10/4 | http | 200 | 72.7 | 73.1 | 73.6 | 137.5 | 0% | AC |  |
| arch-late8-s14 | burst | warm | 512/10/4 | runtime | 200 | 72.0 | 72.3 | 72.7 | 137.5 | 0% | AC |  |
| krite | burst | warm | 512/10/4 | http | 200 | 46.4 | 49.2 | 50.0 | 214.9 | 0% | AC |  |
| krite | burst | warm | 512/10/4 | runtime | 200 | 45.8 | 48.2 | 48.7 | 214.9 | 0% | AC |  |
| krite-nocache | burst | warm | 512/10/4 | http | 200 | 336.5 | 411.2 | 467.8 | 29.8 | 0% | AC |  |
| krite-nocache | burst | warm | 512/10/4 | runtime | 200 | 335.4 | 409.4 | 466.6 | 29.8 | 0% | AC |  |
| arch-b | burst | warm | 512/30/4 | http | 200 | 1834.0 | 1885.6 | 1976.9 | 16.3 | 0% | AC |  |
| arch-b | burst | warm | 512/30/4 | runtime | 200 | 1832.6 | 1884.1 | 1975.3 | 16.3 | 0% | AC |  |
| arch-b-s14 | burst | warm | 512/30/4 | http | 200 | 2405.8 | 3658.1 | 4063.8 | 11.5 | 0% | AC |  |
| arch-b-s14 | burst | warm | 512/30/4 | runtime | 200 | 2404.4 | 3656.4 | 4061.3 | 11.5 | 0% | AC |  |
| arch-d1 | burst | warm | 512/30/4 | http | 200 | 184.4 | 202.9 | 213.6 | 160.2 | 0% | AC |  |
| arch-d1 | burst | warm | 512/30/4 | runtime | 200 | 183.7 | 202.2 | 212.6 | 160.2 | 0% | AC |  |
| arch-d1-lr | burst | warm | 512/30/4 | http | 200 | 180.9 | 185.2 | 188.1 | 166.5 | 0% | AC |  |
| arch-d1-lr | burst | warm | 512/30/4 | runtime | 200 | 180.2 | 184.3 | 187.0 | 166.5 | 0% | AC |  |
| arch-d1-pool | burst | warm | 512/30/4 | http | 200 | 221.7 | 262.7 | 307.8 | 135.4 | 0% | AC |  |
| arch-d1-pool | burst | warm | 512/30/4 | runtime | 200 | 220.1 | 259.8 | 305.2 | 135.4 | 0% | AC |  |
| arch-d1-set | burst | warm | 512/30/4 | http | 200 | 181.8 | 187.9 | 191.6 | 165.4 | 0% | AC |  |
| arch-d1-set | burst | warm | 512/30/4 | runtime | 200 | 181.0 | 187.1 | 190.8 | 165.4 | 0% | AC |  |
| arch-d2 | burst | warm | 512/30/4 | http | 200 | 243.4 | 268.4 | 294.3 | 123.5 | 0% | AC |  |
| arch-d2 | burst | warm | 512/30/4 | runtime | 200 | 242.6 | 267.5 | 293.5 | 123.5 | 0% | AC |  |
| arch-d2-emb | burst | warm | 512/30/4 | http | 200 | 75.6 | 75.9 | 76.3 | 396.4 | 0% | AC |  |
| arch-d2-emb | burst | warm | 512/30/4 | runtime | 200 | 75.0 | 75.2 | 75.4 | 396.4 | 0% | AC |  |
| arch-d2-nocache | burst | warm | 512/30/4 | http | 200 | 280.1 | 305.0 | 306.3 | 107.7 | 0% | AC |  |
| arch-d2-nocache | burst | warm | 512/30/4 | runtime | 200 | 279.2 | 304.2 | 305.5 | 107.7 | 0% | AC |  |
| arch-d4 | burst | warm | 512/30/4 | http | 200 | 363.1 | 374.3 | 398.7 | 83.9 | 0% | AC |  |
| arch-d4 | burst | warm | 512/30/4 | runtime | 200 | 362.3 | 373.5 | 397.9 | 83.9 | 0% | AC |  |
| arch-late4 | burst | warm | 512/30/4 | http | 200 | 112.6 | 112.8 | 113.0 | 266.5 | 0% | AC |  |
| arch-late4 | burst | warm | 512/30/4 | runtime | 200 | 111.9 | 112.1 | 112.1 | 266.5 | 0% | AC |  |
| arch-late4-nocache | burst | warm | 512/30/4 | http | 200 | 297.6 | 321.1 | 322.2 | 100.9 | 0% | AC |  |
| arch-late4-nocache | burst | warm | 512/30/4 | runtime | 200 | 296.8 | 320.3 | 321.4 | 100.9 | 0% | AC |  |
| arch-late4-s14 | burst | warm | 512/30/4 | http | 200 | 112.7 | 114.5 | 116.8 | 265.8 | 0% | AC |  |
| arch-late4-s14 | burst | warm | 512/30/4 | runtime | 200 | 111.9 | 113.8 | 116.1 | 265.8 | 0% | AC |  |
| arch-late6 | burst | warm | 512/30/4 | http | 200 | 162.8 | 169.8 | 174.7 | 182.1 | 0% | AC |  |
| arch-late6 | burst | warm | 512/30/4 | runtime | 200 | 161.9 | 167.5 | 172.4 | 182.1 | 0% | AC |  |
| arch-late8 | burst | warm | 512/30/4 | http | 200 | 87.9 | 89.3 | 91.7 | 340.6 | 0% | AC |  |
| arch-late8 | burst | warm | 512/30/4 | runtime | 200 | 87.1 | 88.2 | 90.8 | 340.6 | 0% | AC |  |
| arch-late8-nocache | burst | warm | 512/30/4 | http | 200 | 244.8 | 265.2 | 268.7 | 122.4 | 0% | AC |  |
| arch-late8-nocache | burst | warm | 512/30/4 | runtime | 200 | 243.7 | 264.1 | 267.6 | 122.4 | 0% | AC |  |
| arch-late8-s14 | burst | warm | 512/30/4 | http | 200 | 222.2 | 245.4 | 248.3 | 131.6 | 0% | AC |  |
| arch-late8-s14 | burst | warm | 512/30/4 | runtime | 200 | 221.3 | 244.2 | 247.3 | 131.6 | 0% | AC |  |
| krite | burst | warm | 512/30/4 | http | 200 | 163.9 | 206.8 | 225.9 | 179.7 | 0% | AC |  |
| krite | burst | warm | 512/30/4 | runtime | 200 | 162.6 | 206.0 | 225.1 | 179.7 | 0% | AC |  |
| krite-nocache | burst | warm | 512/30/4 | http | 200 | 597.1 | 721.6 | 881.3 | 48.9 | 0% | AC |  |
| krite-nocache | burst | warm | 512/30/4 | runtime | 200 | 595.8 | 720.1 | 879.8 | 48.9 | 0% | AC |  |
| arch-b | sustained | warm | 512/1/4 | http | 9302 | 63.7 | 66.1 | 67.8 | 15.7 | 0% | AC | last-minute p50 64.0 |
| arch-d2 | sustained | warm | 512/1/4 | http | 34343 | 17.0 | 19.4 | 20.3 | 57.8 | 0% | AC | last-minute p50 17.6 |
| arch-late4 | sustained | warm | 512/1/4 | http | 79279 | 7.3 | 8.2 | 9.1 | 134.5 | 0% | AC | last-minute p50 7.9 |
| krite | sustained | warm | 512/1/4 | http | 52185 | 8.1 | 18.8 | 20.4 | 88.6 | 0% | AC | last-minute p50 7.9 |

### Peak memory during the quality run

| Engine | Model | phys_footprint GiB | RSS GiB |
|---|---|---|---|
| arch-b | arch-b | 2.73 | 1.27 |
| arch-d1 | arch-d1 | 2.41 | 1.74 |
| arch-d2 | arch-d2 | 2.42 | 1.76 |
| arch-d4 | arch-d4 | 3.42 | 1.77 |
| arch-d2-emb | arch-d2-emb | 2.35 | 1.70 |
| arch-d1-pool | arch-d1-pool | 2.42 | 1.75 |
| arch-d1-set | arch-d1-set | 2.40 | 1.73 |
| arch-late4 | arch-late4 | 4.31 | 1.71 |
| arch-d1-lr | arch-d1-lr | 2.31 | 1.63 |
| arch-late8 | arch-late8 | 5.03 | 1.41 |
| arch-late4-s14 | arch-late4-s14 | 4.31 | 1.70 |
| arch-b-s14 | arch-b-s14 | 2.63 | 0.43 |
| arch-late6 | arch-late6 | 5.19 | 1.75 |
| arch-late8-s14 | arch-late8-s14 | 5.53 | 1.86 |
| krite | krite-0.15b-v0 | 4.17 | 0.94 |

<!-- GENERATED:END -->

## Verdict, first round

**D is rejected in this form.** Produced by `python -m krite_train.study verdict` over the first four
tower arms. D-best is `arch-d1`.

| Rule | Value | Pass |
|---|---|---|
| Q1 in-domain accuracy gap (`arch-b` 0.879 − `arch-d1` 0.445) | 0.434 | no |
| Q1 held-out accuracy gap (`arch-b` 0.726 − `arch-d1` 0.479) | 0.247 | no |
| Q2 score QWK gap (`arch-b` −0.035, `arch-d1` −0.006) | −0.029 | yes |
| L1 `arch-d1` warm p50 × 5 / `arch-b` warm p50 (13.9 ms × 5 / 58.7 ms) | 1.19 | no |
| I1 full-permutation flip rate / max deviation | 0 / 4.4e-16 | yes |
| I2 interference max deviation | 8.7e-8 | yes |
| I3 cache on/off max deviation | 0 | yes |

Q1 misses by far more than 0.01, so the seed-14 re-run does not apply.

## Findings, first round

1. **The towers barely learned to read the state.** Noul answers are constant: across xnli-noul-en
   and boolq-noul, the tower arms' P(true) has a standard deviation of 0.002–0.06 (0.13–0.37 for
   `arch-b`), so their noul accuracy is the majority-class rate. Choice accuracy is above chance but
   about half of `arch-b` or less (massive 0.21–0.48 on 8 candidates vs. 0.86–0.94). Training loss stayed on
   the candidate-prior plateau (≈ 1.53) for ~800 of 1,994 steps. On a synthetic task (pick the
   candidate whose token is in the state) the same tower code reaches near-zero loss within 200 steps,
   so this is an optimization failure of the design at this data scale, not a wiring bug.
2. **More tower depth hurt.** `d1` > `d2` > `d4` in both training loss and accuracy, and the
   embedding-only option encoder (`d2-emb`) is the worst (banking77 0.070, below the 12-option chance
   of 0.083). Deeper randomly initialized stacks between the pretrained encoder and the scorer slow
   learning further at one epoch.
3. **The latency side of D holds.** Warm, 512 tokens: `arch-d1` 13.9 ms at 1 question vs. `arch-b`
   58.7 ms (4.2×, short of the 5× rule), and 184 ms vs. 1,784 ms at 30 questions (9.7×, http).
   `arch-d2-emb` answers 1 question in 4.7 ms and 30 in 76 ms: running the encoder over every
   candidate is the cost that decides warm latency. Sustained 10 minutes: `arch-d2` 17.0 ms,
   `arch-b` 63.7 ms, no drift.
4. **The invariants hold by construction.** Every tower arm has zero option-order flips
   (`arch-b`: 0.03–0.13 full-permutation flip rate), interference ≤ 1e-7, and identical output with
   the state cache on and off.
5. **Score transfer failed for every arm**, `arch-b` included (QWK ≈ 0 on sst5 and amazon): toxicity
   levels from civil_comments do not teach sentiment or star levels. Q2 passes only because both
   sides are at zero; it says nothing about D.

## Follow-up round

Four arms, one change each against `arch-d1`, aimed at finding 1, measured the same way and judged
by the same rule over all eight tower arms
([`verdict.json`](../benchmarks/results/arch/verdict.json)).

| Arm | In-domain acc | Held-out acc | Warm p50, 1 / 30 questions (runtime) |
|---|---|---|---|
| `arch-b` | 0.879 | 0.726 | 58.7 / 1,783 ms |
| `arch-d1` | 0.445 | 0.479 | 13.9 / 184 ms |
| `arch-d1-pool` | 0.849 | 0.631 | 14.0 / 222 ms |
| `arch-d1-set` | 0.570 | 0.495 | 13.9 / 182 ms |
| `arch-d1-lr` | 0.382 | 0.477 | 13.9 / 181 ms |
| **`arch-late4`** | **0.885** | **0.690** | **18.9 / 239 ms** |

**Verdict: D is still not accepted, but `arch-late4` misses narrowly.** D-best is `arch-late4`.

| Rule | Value | Pass |
|---|---|---|
| Q1 in-domain accuracy gap (`arch-b` 0.879 − `arch-late4` 0.885) | −0.006 | yes |
| Q1 held-out accuracy gap (`arch-b` 0.726 − `arch-late4` 0.690) | 0.036 | no |
| Q2 score QWK gap | 0.002 | yes |
| L1 `arch-late4` warm p50 × 5 / `arch-b` warm p50 (18.9 ms × 5 / 58.7 ms) | 1.61 | no |
| I1 full-permutation flip rate / max deviation | 0 / 4.4e-16 | yes |
| I2 interference max deviation | 1.1e-6 | yes |
| I3 cache on/off max deviation (`arch-d2`) | 0 | yes |

Q1 misses by 0.016, outside the 0.01 window for a seed-14 re-run.

1. **A direct path from the state to the score is what was missing.** Both arms that give the scorer
   such a path left the prior plateau within ~350 steps and reached `arch-b`'s training loss
   (0.44 vs. 0.43). Set attention and twice the training (`d1-lr`) did not leave it sooner, and
   `d1-lr` ends below `d1` in accuracy, so the first round's failure was structural, not a budget
   or learning-rate problem.
2. **`arch-late4` matches `arch-b` on choice and loses on NLI.** It is at or above `arch-b` on every
   in-domain suite and on held-out agnews (0.652 vs. 0.623), and 1–8 points below on every
   xnli-noul language (mean −4.4). The whole held-out gap comes from premise/hypothesis inference,
   where reading the question and the state in one sequence helps most. `arch-d1-pool` is close
   in-domain (0.849) but weaker held-out (0.631), so the pooled state alone transfers less.
3. **Warm latency is set by the lower layers on the candidate side.** `arch-late4` runs all 22
   encoder layers over every candidate on every request, so it is 3.1× faster than `arch-b` at 1
   question and 7.5× at 30. Layers 0–17 of a candidate do not depend on the state, so they can be
   cached per instructions-and-option text the way the state is cached per state; a warm request
   would then run only the top four layers on the candidate side.
4. **The invariants still hold.** `arch-late4` and `arch-d1-pool` have zero option-order flips and
   interference ≤ 1.1e-6. The cache on/off check was run on `arch-d2`; `arch-late4` uses the same
   shim cache path but was not compared directly.

## Second round

Three follow-ups on `arch-late4`: a cache for the candidate side, a deeper arm, and a second seed. The
shim now caches each candidate's states after layers 0–17 by its token ids, next to the state cache,
which now holds the state's rotated keys and values for the top layers instead of their inputs. The
trained `late4` weights are unchanged; its predictions with the new cache path are identical to the
first measurement on agnews-choice and boolq-noul (max |Δp| 0).

| Arm | In-domain acc | Held-out acc | Score QWK | Warm p50, 1 / 10 / 30 questions (runtime) |
|---|---|---|---|---|
| `arch-b` | 0.879 | 0.726 | −0.035 | 58.7 / 581 / 1,783 ms |
| `arch-b-s14` | 0.874 | 0.702 | 0.068 | 68.2 / 730 / 2,404 ms |
| `arch-late4` (cached) | 0.885 | 0.690 | −0.037 | 6.9 / 39 / 112 ms |
| `arch-late4-s14` | 0.820 | 0.525 | 0.026 | 6.9 / 39 / 112 ms |
| `arch-late8` | 0.877 | 0.708 | 0.264 | 12.8 / 72 / 212 ms |

**Verdict: D is still not accepted.** D-best is now `arch-late8` (highest held-out accuracy). It
passes every quality and invariance rule and misses L1 by 9%.

| Rule | `arch-late8` | Pass | `arch-late4` (cached), for comparison | Pass |
|---|---|---|---|---|
| Q1 in-domain accuracy gap | 0.003 | yes | −0.006 | yes |
| Q1 held-out accuracy gap | 0.018 | yes | 0.036 | no |
| Q2 score QWK gap | −0.299 | yes | 0.002 | yes |
| L1 warm p50 × 5 / `arch-b` warm p50 | 1.09 | no | 0.58 | yes |
| I1 flip rate / max deviation | 0 / 3.3e-16 | yes | 0 / 0 | yes |
| I2 interference max deviation | 6.7e-7 | yes | ≤ 1.1e-6 | yes |
| I3 cache on/off max deviation | not run on `late8` | — | 8.5e-7 | yes |

`verdict.json` reports I3 on `arch-d2` for `arch-late8` because only `arch-late4` has a no-cache
twin; both late arms run the same cache code.

1. **The candidate cache moves `late4` well inside L1.** Warm p50 drops from 18.9 to 6.9 ms at one
   question (8.5× faster than `arch-b`) and from 239 to 112 ms at 30 (16×). Cached and uncached
   predictions differ by at most 8.5e-7.
2. **Depth closes most of the NLI gap.** `arch-late8` is within 1.8 points of `arch-b` held-out;
   on xnli-noul it is above `arch-b` for ar and fr, within 2 points for en and de, and 3.7–4.9 points
   below for es, hi, ru, and zh; agnews is level (0.625 vs. 0.623). It is
   also the only arm with a clearly positive score QWK (0.26 vs. about 0 for every other arm; no arm
   learned the score suites well). The cost is latency: twice the work of `late4` on the candidate
   side, 12.8 ms against an L1 limit of 11.7 ms.
3. **`late4` is not stable across seeds; `b` is.** `arch-late4-s14` falls to 0.525 held-out, with
   every xnli-noul language near chance (0.47–0.55) and in-domain 6.5 points lower. `arch-b-s14`
   moves 2.4 points held-out and 0.5 in-domain. Its training loss ended higher (0.50 vs. 0.44), so
   the seed-14 run of `late4` learned less, not differently. The first-round `late4` result is
   therefore an optimistic draw, and `late8` has one seed only.
4. **`arch-b-s14` was slower (68 vs. 59 ms) on identical compute.** This is run-to-run variation on
   a fanless machine, about 15%, which is larger than `late8`'s L1 miss.

## Third round

Three follow-ups on `arch-late8`: its own cache on/off check and a second seed, a 6-layer arm, and
back-to-back latency repeats of `arch-b` and `arch-late8` (b, late8, b, late8, latency only). Then
one serving change: with a single state, the shim puts all candidates of a request in one packed row
with a block-diagonal mask, so the state's keys and values are no longer copied once per candidate.
This changes no weights and no result: `arch-late8` predictions before and after differ by at most
1.9e-6 over all 22,400 cases. `arch-late8` and `arch-late8-nocache` were measured again with it.

| Arm | In-domain acc | Held-out acc | Score QWK | Warm p50, 1 / 10 / 30 questions (runtime) |
|---|---|---|---|---|
| `arch-b` | 0.879 | 0.726 | −0.035 | 58.7–60.2 / 581 / 1,783 ms |
| `arch-b-s14` | 0.874 | 0.702 | 0.068 | — |
| `arch-late6` (per-candidate rows) | 0.869 | 0.698 | 0.167 | 9.0 ms (1 question) |
| `arch-late8` (per-candidate rows) | 0.877 | 0.708 | 0.264 | 12.2–12.8 / 72 / 212–224 ms |
| **`arch-late8` (packed row)** | **0.877** | **0.708** | **0.264** | **6.2 / 28 / 87 ms** |
| `arch-late8-s14` | 0.874 | 0.683 | 0.075 | 12.2 ms (per-candidate rows) |

The three `arch-b` repeats span 58.7–60.2 ms at one question (2.6%) and the three `arch-late8`
per-candidate repeats 12.2–12.8 ms, so the second round's 15% `arch-b-s14` gap was a one-off.

**Verdict: D is accepted** ([`verdict.json`](../benchmarks/results/arch/verdict.json)). D-best is
`arch-late8`, and every pre-registered rule passes.

| Rule | Value | Pass |
|---|---|---|
| Q1 in-domain accuracy gap (`arch-b` 0.879 − `arch-late8` 0.877) | 0.003 | yes |
| Q1 held-out accuracy gap (`arch-b` 0.726 − `arch-late8` 0.708) | 0.018 | yes |
| Q2 score QWK gap | −0.299 | yes |
| L1 `arch-late8` warm p50 × 5 / `arch-b` warm p50 (6.19 ms × 5 / 58.85 ms) | 0.53 | yes |
| I1 full-permutation flip rate / max deviation | 0 / 2.3e-6 | yes |
| I2 interference max deviation | 6.7e-7 | yes |
| I3 cache on/off max deviation (`arch-late8`) | 1.4e-6 | yes |

1. **The result holds on the second seed.** Seed 14 against seed 14, `arch-late8` is 0.001 below
   `arch-b-s14` in-domain and 0.019 below held-out, both inside Q1. Its xnli-noul accuracy per
   language moves at most 5 points between seeds; the held-out drop comes from agnews (0.412 vs.
   0.625), the suite on which `arch-b` also moved most between seeds (0.623 vs. 0.557). This is
   unlike `arch-late4`, whose second seed lost NLI entirely.
2. **Packing, not fewer layers, decided L1.** Per-candidate rows copied the state's keys and values
   for every candidate in each of the eight late layers; this memory traffic was most of the warm
   cost. Packed, `arch-late8` is 9.5× faster than `arch-b` at one question and 20× at 30, and faster
   than `arch-late6` was with per-candidate rows.
3. **Six layers sit between four and eight.** `arch-late6` is 1.0 point below `arch-late8`
   held-out and 0.8 in-domain, outside the 0.005 tie band, so `arch-late8` stays D-best.
4. **Caches are exact within tolerance.** `arch-late8` with and without its state and candidate
   caches differs by at most 1.4e-6.

## Next

- Done: `late8` runs in the Candle runtime (engines `krite` and `krite-nocache` in the tables above)
  with per-bucket temperatures; it matches `arch-late8` within 6.6e-6 per probability and on every
  suite's accuracy. Numbers, including where Candle is still slower than torch, are in
  [runtime.md](runtime.md#measured).
- The release recipe and its training data: [training-data.md](training-data.md).

## Caveats

- Seeds: two for `b`, `late4`, and `late8`, one for every other arm. The first-round gaps (25–43
  points) are far above seed noise; the later differences (1–4 points) are not, and `late4` moved
  16.5 points held-out between its two seeds.
- The decision rule reads the seed-13 runs; the seed-14 runs are a check, not part of the rule.
- Score suites are weak for every arm (QWK ≤ 0.27), so Q2 says little.
- torch on MPS, not the Candle runtime; absolute latencies are not `krite serve` numbers, only the
  B-vs-D ratio carries over.
- Training cut states to 254 tokens; evaluation did not.
- Every suite is zero-shot in instruction wording, so absolute accuracies are lower than with
  suite-specific instructions.
