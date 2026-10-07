# Training Data and Release Recipe

The [architecture study](architecture-study.md) fixed the model: `late8`, late interaction in the top
8 of mmBERT-small's 22 layers. Its checkpoint ships as `krite-0.15b-v0` ([runtime.md](runtime.md)) and
already meets the latency, invariance, and calibration targets. It does not meet the accuracy target
on 11 of 28 suites. This document is the recipe search for the release model: which training data,
which loss, and how many epochs. Each choice is decided by a rule fixed before its training runs.
Terms follow [ARCHITECTURE.md](../ARCHITECTURE.md).

Two mixtures are built by `krite_train.data`. `study` is the architecture study's mixture, frozen
(32,000 examples, sha256 `50496145fdff…` at seed 13) so that every study arm stays reproducible.
`broad` adds sources, question-type views, and augmentations. Dataset text never leaves the Hugging
Face cache; only counts, hashes, revisions, and licenses are recorded (`train_meta.json`, this
document).

## Targets

The accuracy target per suite is the best zero-shot engine under 0.45B parameters (Laya or cbjev,
[baselines.md](baselines.md)) minus 0.02, on accuracy for choice and noul suites and QWK for score
suites. Laya ran the non-English suites on its multilingual weights. The **shortfall** of an engine is
the sum over all 28 suites of how far it falls below the target (0 where it meets it).

| Suite | Metric | Laya | cbjev | Target | `krite-0.15b-v0` | Shortfall |
|---|---|---|---|---|---|---|
| agnews-choice | accuracy | 0.945 | 0.940 | 0.925 | 0.625 | 0.300 |
| banking77-choice | accuracy | 0.812 | 0.855 | 0.835 | 0.902 | 0 |
| boolq-noul | accuracy | 0.845 | 0.823 | 0.825 | 0.630 | 0.195 |
| sst5-score | QWK | 0.402 | 0.748 | 0.728 | 0.464 | 0.264 |
| massive-choice-en | accuracy | 0.752 | 0.748 | 0.732 | 0.907 | 0 |
| massive-choice-ko | accuracy | 0.615 | 0.613 | 0.595 | 0.887 | 0 |
| massive-choice-ja | accuracy | 0.667 | 0.700 | 0.680 | 0.907 | 0 |
| massive-choice-zh-CN | accuracy | 0.650 | 0.698 | 0.677 | 0.920 | 0 |
| massive-choice-de | accuracy | 0.565 | 0.525 | 0.545 | 0.902 | 0 |
| massive-choice-es | accuracy | 0.585 | 0.625 | 0.605 | 0.912 | 0 |
| massive-choice-fr | accuracy | 0.642 | 0.652 | 0.632 | 0.935 | 0 |
| massive-choice-ar | accuracy | 0.527 | 0.565 | 0.545 | 0.855 | 0 |
| massive-choice-hi | accuracy | 0.512 | 0.585 | 0.565 | 0.860 | 0 |
| massive-choice-ru | accuracy | 0.627 | 0.640 | 0.620 | 0.902 | 0 |
| xnli-noul-en | accuracy | 0.627 | 0.922 | 0.902 | 0.755 | 0.147 |
| xnli-noul-de | accuracy | 0.568 | 0.807 | 0.787 | 0.730 | 0.057 |
| xnli-noul-es | accuracy | 0.608 | 0.823 | 0.802 | 0.738 | 0.065 |
| xnli-noul-fr | accuracy | 0.578 | 0.830 | 0.810 | 0.765 | 0.045 |
| xnli-noul-ar | accuracy | 0.552 | 0.792 | 0.772 | 0.728 | 0.045 |
| xnli-noul-hi | accuracy | 0.540 | 0.760 | 0.740 | 0.642 | 0.098 |
| xnli-noul-ru | accuracy | 0.590 | 0.815 | 0.795 | 0.690 | 0.105 |
| xnli-noul-zh | accuracy | 0.575 | 0.818 | 0.797 | 0.703 | 0.095 |
| amazon-score-en | QWK | 0.097 | 0.033 | 0.077 | 0.390 | 0 |
| amazon-score-de | QWK | 0.046 | 0.008 | 0.026 | 0.308 | 0 |
| amazon-score-es | QWK | 0.058 | 0.003 | 0.038 | 0.248 | 0 |
| amazon-score-fr | QWK | 0.038 | -0.005 | 0.018 | 0.202 | 0 |
| amazon-score-ja | QWK | 0.051 | -0.001 | 0.031 | 0.090 | 0 |
| amazon-score-zh | QWK | 0.037 | -0.003 | 0.017 | 0.142 | 0 |

Shortfall: `krite-0.15b-v0` (= `arch-late8`, seed 13) 1.416; `arch-late8-s14` (seed 14) 1.902. The
two seeds of the same recipe differ by 0.49, which is why every rule below reads two seeds.

### Revised release target

Set after the recipe search and the mmBERT-base feasibility check, which left the per-suite target
out of reach. The per-suite target takes the best engine **per suite**, a model that does not exist:
neither of its sources meets it (shortfall: Laya 2.249, cbjev 0.194). The release gate therefore
compares Krite with whole engines. For each engine, the per-dataset means of the suite metric are
averaged: accuracy over the 5 choice and noul datasets, QWK over the 2 score datasets. The target per
metric is the better of Laya and cbjev minus 0.02 (`krite_train.study.reference`). The stage rules
above keep the per-suite shortfall: they decided the completed stages, and it stays the per-suite
diagnostic.

| Engine | Accuracy (macro) | QWK (macro) |
|---|---|---|
| Laya | 0.759 | 0.228 |
| cbjev | 0.815 | 0.377 |
| **Target** (cbjev − 0.02) | **0.795** | **0.357** |

## Pre-registered rules

Fixed before the first training run of the recipe search; implemented in
[`krite_train.study`](../training/krite_train/study.py) (`stage`, `release`).

**Stage rule.** A recipe change is adopted when all of these hold, comparing the new recipe's two
seeds (13, 14) with the current recipe's two seeds:

| Rule | Passes when |
|---|---|
| shortfall gain | mean shortfall (current) − mean shortfall (new) ≥ 0.10 |
| seed agreement | the new recipe's shortfall is lower than the current one's at seed 13 and at seed 14 |
| in-domain drop | mean in-domain accuracy (banking77, massive, boolq) drops by at most 0.01 |
| invariants | every new engine: full-permutation flip rate 0, invariance and interference max deviation ≤ 1e-5 |

**Order.** Greedy, fixed now: (A) `broad` vs `study` mixture → (B1) adding the Brier term vs the A
winner → (B2) adding the ordinal term vs the winner so far → (C) two epochs vs the winner so far. A
rejected change leaves the previous winner in place. The release checkpoint is the final winner's
**seed-13** run; the better of the two seeds is not picked.

**Release gate.** Measured on the release engine in the standard order (fresh start, cold, warm,
sustained, quality, invariance, interference), after refitting temperatures on its uncalibrated twin:

| Gate | Passes when |
|---|---|
| accuracy | macro accuracy and macro QWK at or above the revised target (see "Revised release target"; originally shortfall 0) |
| ECE | mean over suites of the temperature-scaled ECE on each suite's evaluation half ≤ 0.071 |
| warm latency | HTTP p50, burst, 512-token state, 1 question, 4 options ≤ 10 ms |
| cold latency | same cell, state cache miss ≤ 210 ms |
| throughput | 30 questions / warm HTTP p50 at 512 tokens, 30 questions, 4 options ≥ 175 decisions/s |
| option order | full-permutation flip rate 0 and max deviation ≤ 1e-5 |
| isolation | question-interference max deviation ≤ 1e-5 |
| cache | release engine vs. the same model with both caches off: max probability difference ≤ 1e-5 |

If the accuracy gate fails, the per-suite shortfall is reported and the next step is a decision
outside this recipe: a larger encoder (mmBERT-base, ~0.4B) or a revised accuracy target.

Applied to today's `krite-0.15b-v0`, the gate passes every row except accuracy: ECE 0.058, warm
7.8 ms, cold 91 ms, 183 decisions/s, flip rate 0, isolation 5.3e-7, cache 0.

## Encoder

The recipe search ended with every gate passing except accuracy (see Results). The rules above send
the next step outside the recipe, and this section is that step: the final recipe (`late8`, `broad`,
cross-entropy, one epoch, same learning rates and batch) on a larger encoder. Fixed before its pilot
ran.

| | mmBERT-small | mmBERT-base |
|---|---|---|
| Repo @ revision | `jhu-clsp/mmBERT-small` @ `abc32620` | `jhu-clsp/mmBERT-base` @ `c5955035` |
| Hidden size, heads (head dim) | 384, 6 (64) | 768, 12 (64) |
| Layers, intermediate size | 22, 1,152 | 22, 1,152 |
| Parameters (embedding / rest) | ≈ 98M / 42M | ≈ 197M / 110M |
| Tokenizer | identical `tokenizer.json` | identical `tokenizer.json` |
| License | MIT | MIT |

Layer count, attention pattern, RoPE, and tokenizer are shared, so `late8` keeps its split (14 lower,
8 late layers), training data tokenizes identically, and the Candle runtime reads every size from
`config.json`. Per token, base costs about 2.6× small's non-embedding compute.

**Feasibility.** Latency depends on shapes and kernels, not on trained weights, so a 1% pilot of
`base8-broad` (`krite_train.train --scale 0.01`) is exported, served by Candle as `krite-base-pilot`,
and timed in the release gate's latency cells before any full training run. Every row must hold, or
the encoder stage stops before training:

| Check | Passes when |
|---|---|
| Candle numerics | the ignored `krite-candle` weight tests pass on the pilot export |
| warm latency | HTTP p50, burst, 512-token state, 1 question, 4 options ≤ 10 ms |
| cold latency | same cell, state cache miss ≤ 210 ms |
| throughput | 30 questions / warm HTTP p50 at 512 tokens, 30 questions, 4 options ≥ 175 decisions/s |
| training memory | pilot MPS peak ≤ 10 GiB; above it, the pilot reruns with two-step gradient accumulation (same examples per optimizer step) and must then fit |

`krite_train.study feasibility` applies the three latency rows.

**Encoder stage (E).** The stage rule above, unchanged: `broad` on small (`arch-late8-broad`, `-s14`)
vs `broad` on base (`arch-base8-broad`, `-s14`). Nothing is retuned after a seed's result is seen.

**Release.** If E adopts base, its seed-13 run ships as `krite-0.3b-v0` and goes through the same
release gate. If E rejects base or the gate fails, `krite-0.15b-v1` stays the best model, and the next
step is a revised accuracy target.

## Sources

Every source is pinned to a Hugging Face revision. Rows whose state appears in any evaluation suite
are dropped first (leakage guard). The evaluation datasets agnews, xnli, sst5, and amazon are never
read for training.

| Source | Repo @ revision | Type (views) | n | Leaked rows dropped | License | Mixtures |
|---|---|---|---|---|---|---|
| banking77 | `mteb/banking77` @ `18072d26` | choice (noul) | 4,000 | 0 | CC-BY-4.0 | study, broad |
| clinc | `clinc/clinc_oos` @ `155b9c71` (`plus`, `oos` dropped) | choice (noul) | 4,000 | 2 | CC-BY-3.0 | study, broad |
| massive | `mteb/amazon_massive_intent` @ `940fd47a` (10 suite languages) | choice (noul) | 8,000 | 539 | CC-BY-4.0 | study, broad |
| dbpedia | `fancyzhx/dbpedia_14` @ `9abd46cf` | choice (noul) | 4,000 | 0 | CC-BY-SA-3.0 | study, broad |
| boolq | `google/boolq` @ `35b264d0` | noul | 4,000 (broad: 8,000) | 0 | CC-BY-SA-3.0 | study, broad |
| snli | `stanfordnlp/snli` @ `cdb5c3d5` (neutral dropped for noul) | noul (3-way choice) | 4,000 | 0 | CC-BY-SA-4.0 | study, broad |
| civil | `google/civil_comments` @ `f2970eb3` (toxicity in 5 levels) | score, K = 5 (noul) | 4,000 | 1 | CC0-1.0 | study, broad |
| sib200 | `Davlan/sib200` @ `38977a66` (10 suite languages, 7 topics) | choice (noul) | 7,000 | 0 | CC-BY-SA-4.0 | broad |
| mnli | `nyu-mll/multi_nli` @ `da70db2a` (neutral dropped for noul) | noul (3-way choice) | 6,000 | 0 | OANC, CC-BY-3.0, CC-BY-SA-3.0 (per genre) | broad |
| pawsx | `google-research-datasets/paws-x` @ `4cd8187c` (de en es fr ja ko zh) | noul | 7,000 | 0 | PAWS-X terms (free use for any purpose) | broad |
| goemotions | `google-research-datasets/go_emotions` @ `add49224` (`simplified`) | score, K = 3 | 6,000 | 0 | Apache-2.0 | broad |

GoEmotions labels map to a sentiment level with the dataset authors' grouping: negative (anger,
annoyance, disappointment, disapproval, disgust, embarrassment, fear, grief, nervousness, remorse,
sadness), neutral, positive (admiration, amusement, approval, caring, desire, excitement, gratitude,
joy, love, optimism, pride, relief). Rows with an ambiguous emotion (confusion, curiosity,
realization, surprise) or with labels of two polarities are dropped. The levels are named
`negative / neutral / positive` or `unhappy / neutral / happy`, one set per example.

agnews, xnli, and sst5 stay held out by dataset, but SIB-200 (topic), MultiNLI (entailment), and
GoEmotions (sentiment) make them less zero-shot in task than they were for the study.

## Views and augmentations (`broad` only)

**Views** re-ask a source's labeled rows as another question type:

| Source | View | Count | Construction |
|---|---|---|---|
| banking77, clinc, dbpedia, sib200 | noul | 1,000 each | "Is the user's intent {label}?" / "Is this text about {label}?": the gold label or a random other label, each with probability 0.5 |
| massive | noul | 2,000 | as above |
| civil | noul | 1,000 | "Is this comment toxic?": half from levels ≥ 2 (toxicity ≥ 0.4, true), half below |
| snli, mnli | 3-way choice | 2,000 / 3,000 | criteria `entailment`, `neutral`, `contradiction`, each with a description |

**Augmentations**, applied per example with a seeded generator:

- State formatting, probability 0.3: a text state becomes a JSON object under one of six keys (`text`,
  `message`, `input`, `content`, `observation`, `utterance`); a JSON state has its keys renamed to
  synonyms with probability 0.5 each (`passage` → `context`/`document`, `question` →
  `query`/`user_question`, …); then, with probability 0.5, one or two unrelated fields (`turn`,
  `channel`, `step`) are added.
- Irrelevant candidates, probability 0.2, choice questions of intent and topic sources with K ≥ 3: up
  to two distractors are replaced with labels from the other domain (an intent label in a topic
  question and the reverse). K and the gold label are unchanged.

Not used, with the reason:

- Option permutation, question-id randomization, JSON key reordering: no-ops for this model. Option
  order cannot change a probability (I1), ids are not inputs, and JSON states are canonicalized with
  sorted keys before tokenization.
- Paraphrase and multilingual instructions: they need a generator model; every suite's instructions
  are English.

## Losses

| Recipe | Loss |
|---|---|
| CE | cross-entropy over each question's candidates |
| + Brier | + Σ_k (p_k − y_k)², weight 1.0 |
| + ordinal | + on score questions only: Σ_k (P(level ≤ k) − Y(level ≤ k))² / (K − 1), weight 1.0 |

The weights are fixed now and are not tuned on results.

## Not used

| Source | Reason |
|---|---|
| xnli train | CC-BY-NC-4.0 |
| agnews, sst5, amazon reviews | evaluation-only |
| yahoo_answers_topics, SetFit/20_newsgroups, MoritzLaurer multilingual NLI | no verifiable license |
| stanfordnlp/sst2, stanfordnlp/imdb, Yelp/yelp_review_full | license unknown or proprietary terms |
| tyqiangz/multilingual-sentiments | aggregate of upstreams with unknown licenses, likely including Amazon reviews |
| cardiffnlp/tweet_sentiment_multilingual | no license tag |
| Kev-0.8B outputs (knowledge distillation) | Kev trained on ag_news, sst5, amazon_reviews_multi, boolq, and multi_nli; its labels would carry four evaluation sources into training |
| Jev outputs | terms of service |

## Results

<!-- GENERATED:START -->

### Quality

| Suite | Metric | krite | arch-late8-broad | arch-late8-broad-s14 | arch-late8-brier | arch-late8-brier-s14 | arch-late8-ord | arch-late8-ord-s14 | arch-late8-e2 | arch-late8-e2-s14 | krite-v1 | krite-v1-raw | krite-v1-nocache |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agnews-choice | accuracy / macro_f1 | 0.625 / 0.621 | 0.642 / 0.597 | 0.682 / 0.625 | 0.660 / 0.620 | 0.545 / 0.537 | 0.700 / 0.674 | 0.705 / 0.698 | 0.660 / 0.607 | 0.635 / 0.568 | 0.642 / 0.597 | 0.642 / 0.597 | 0.642 / 0.597 |
| banking77-choice | accuracy / macro_f1 | 0.902 / 0.900 | 0.880 / 0.880 | 0.912 / 0.909 | 0.873 / 0.868 | 0.718 / 0.713 | 0.880 / 0.875 | 0.905 / 0.905 | 0.922 / 0.919 | 0.932 / 0.927 | 0.880 / 0.880 | 0.880 / 0.880 | n/a (not measured) |
| boolq-noul | accuracy / auroc | 0.630 / 0.678 | 0.688 / 0.787 | 0.675 / 0.739 | 0.703 / 0.771 | 0.580 / 0.584 | 0.710 / 0.768 | 0.637 / 0.701 | 0.738 / 0.825 | 0.740 / 0.816 | 0.688 / 0.787 | 0.688 / 0.787 | 0.688 / 0.787 |
| sst5-score | mae / qwk | 0.881 / 0.464 | 0.875 / 0.509 | 0.833 / 0.543 | 0.837 / 0.558 | 1.044 / 0.299 | 0.800 / 0.601 | 0.826 / 0.576 | 0.838 / 0.515 | 0.882 / 0.484 | 0.913 / 0.509 | 0.875 / 0.509 | n/a (not measured) |
| massive-choice-en | accuracy / macro_f1 | 0.907 / 0.880 | 0.920 / 0.886 | 0.927 / 0.894 | 0.910 / 0.877 | 0.790 / 0.747 | 0.912 / 0.873 | 0.917 / 0.873 | 0.925 / 0.895 | 0.925 / 0.895 | 0.920 / 0.886 | 0.920 / 0.886 | n/a (not measured) |
| massive-choice-ko | accuracy / macro_f1 | 0.887 / 0.847 | 0.890 / 0.884 | 0.880 / 0.836 | 0.877 / 0.857 | 0.728 / 0.671 | 0.887 / 0.870 | 0.885 / 0.874 | 0.900 / 0.878 | 0.920 / 0.901 | 0.890 / 0.884 | 0.890 / 0.884 | 0.890 / 0.884 |
| massive-choice-ja | accuracy / macro_f1 | 0.907 / 0.907 | 0.920 / 0.914 | 0.943 / 0.910 | 0.912 / 0.881 | 0.792 / 0.744 | 0.902 / 0.897 | 0.920 / 0.896 | 0.930 / 0.911 | 0.922 / 0.899 | 0.920 / 0.914 | 0.920 / 0.914 | n/a (not measured) |
| massive-choice-zh-CN | accuracy / macro_f1 | 0.920 / 0.909 | 0.915 / 0.887 | 0.935 / 0.933 | 0.900 / 0.886 | 0.807 / 0.752 | 0.907 / 0.872 | 0.925 / 0.905 | 0.927 / 0.937 | 0.948 / 0.955 | 0.915 / 0.887 | 0.915 / 0.887 | n/a (not measured) |
| massive-choice-de | accuracy / macro_f1 | 0.902 / 0.872 | 0.892 / 0.863 | 0.892 / 0.849 | 0.890 / 0.853 | 0.738 / 0.683 | 0.895 / 0.845 | 0.885 / 0.847 | 0.900 / 0.856 | 0.915 / 0.880 | 0.892 / 0.863 | 0.892 / 0.863 | n/a (not measured) |
| massive-choice-es | accuracy / macro_f1 | 0.912 / 0.882 | 0.902 / 0.886 | 0.895 / 0.885 | 0.907 / 0.881 | 0.780 / 0.719 | 0.887 / 0.875 | 0.900 / 0.900 | 0.920 / 0.903 | 0.932 / 0.939 | 0.902 / 0.886 | 0.902 / 0.886 | n/a (not measured) |
| massive-choice-fr | accuracy / macro_f1 | 0.935 / 0.921 | 0.927 / 0.901 | 0.930 / 0.919 | 0.935 / 0.926 | 0.835 / 0.784 | 0.912 / 0.889 | 0.912 / 0.902 | 0.935 / 0.916 | 0.917 / 0.922 | 0.927 / 0.901 | 0.927 / 0.901 | n/a (not measured) |
| massive-choice-ar | accuracy / macro_f1 | 0.855 / 0.797 | 0.840 / 0.749 | 0.855 / 0.765 | 0.833 / 0.769 | 0.703 / 0.599 | 0.855 / 0.792 | 0.833 / 0.746 | 0.863 / 0.824 | 0.845 / 0.763 | 0.840 / 0.749 | 0.840 / 0.749 | n/a (not measured) |
| massive-choice-hi | accuracy / macro_f1 | 0.860 / 0.731 | 0.863 / 0.741 | 0.848 / 0.750 | 0.865 / 0.750 | 0.652 / 0.531 | 0.845 / 0.737 | 0.873 / 0.787 | 0.873 / 0.764 | 0.877 / 0.759 | 0.863 / 0.741 | 0.863 / 0.741 | n/a (not measured) |
| massive-choice-ru | accuracy / macro_f1 | 0.902 / 0.852 | 0.887 / 0.849 | 0.920 / 0.886 | 0.897 / 0.871 | 0.762 / 0.696 | 0.892 / 0.824 | 0.892 / 0.844 | 0.905 / 0.836 | 0.912 / 0.878 | 0.887 / 0.849 | 0.887 / 0.849 | n/a (not measured) |
| xnli-noul-en | accuracy / auroc | 0.755 / 0.831 | 0.910 / 0.954 | 0.897 / 0.954 | 0.880 / 0.947 | 0.490 / 0.512 | 0.890 / 0.954 | 0.895 / 0.957 | 0.873 / 0.946 | 0.890 / 0.957 | 0.910 / 0.954 | 0.910 / 0.954 | n/a (not measured) |
| xnli-noul-de | accuracy / auroc | 0.730 / 0.826 | 0.812 / 0.896 | 0.818 / 0.897 | 0.812 / 0.895 | 0.530 / 0.562 | 0.770 / 0.897 | 0.775 / 0.884 | 0.792 / 0.899 | 0.823 / 0.891 | 0.812 / 0.896 | 0.812 / 0.896 | n/a (not measured) |
| xnli-noul-es | accuracy / auroc | 0.738 / 0.823 | 0.845 / 0.917 | 0.840 / 0.923 | 0.818 / 0.912 | 0.560 / 0.577 | 0.818 / 0.904 | 0.828 / 0.912 | 0.818 / 0.912 | 0.830 / 0.915 | 0.845 / 0.917 | 0.845 / 0.917 | n/a (not measured) |
| xnli-noul-fr | accuracy / auroc | 0.765 / 0.827 | 0.823 / 0.925 | 0.823 / 0.924 | 0.807 / 0.917 | 0.552 / 0.560 | 0.820 / 0.915 | 0.853 / 0.921 | 0.815 / 0.919 | 0.777 / 0.880 | 0.823 / 0.925 | 0.823 / 0.925 | n/a (not measured) |
| xnli-noul-ar | accuracy / auroc | 0.728 / 0.799 | 0.755 / 0.856 | 0.782 / 0.869 | 0.755 / 0.870 | 0.505 / 0.517 | 0.760 / 0.866 | 0.762 / 0.868 | 0.752 / 0.871 | 0.748 / 0.848 | 0.755 / 0.856 | 0.755 / 0.856 | n/a (not measured) |
| xnli-noul-hi | accuracy / auroc | 0.642 / 0.739 | 0.715 / 0.848 | 0.718 / 0.843 | 0.695 / 0.842 | 0.532 / 0.547 | 0.693 / 0.828 | 0.720 / 0.832 | 0.715 / 0.851 | 0.695 / 0.828 | 0.715 / 0.848 | 0.715 / 0.848 | n/a (not measured) |
| xnli-noul-ru | accuracy / auroc | 0.690 / 0.771 | 0.805 / 0.905 | 0.810 / 0.905 | 0.777 / 0.896 | 0.530 / 0.548 | 0.800 / 0.895 | 0.805 / 0.900 | 0.782 / 0.897 | 0.800 / 0.896 | 0.805 / 0.905 | 0.805 / 0.905 | n/a (not measured) |
| xnli-noul-zh | accuracy / auroc | 0.703 / 0.780 | 0.828 / 0.910 | 0.835 / 0.895 | 0.802 / 0.895 | 0.560 / 0.569 | 0.800 / 0.894 | 0.792 / 0.877 | 0.810 / 0.901 | 0.823 / 0.891 | 0.828 / 0.910 | 0.828 / 0.910 | n/a (not measured) |
| amazon-score-en | mae / qwk | 1.112 / 0.390 | 1.139 / 0.142 | 1.095 / 0.357 | 1.095 / 0.313 | 1.299 / 0.019 | 1.094 / 0.362 | 1.110 / 0.386 | 1.154 / 0.140 | 1.192 / 0.246 | 1.155 / 0.142 | 1.139 / 0.142 | n/a (not measured) |
| amazon-score-de | mae / qwk | 1.130 / 0.308 | 1.182 / 0.039 | 1.136 / 0.106 | 1.145 / 0.197 | 1.326 / 0.033 | 1.204 / 0.238 | 1.136 / 0.256 | 1.159 / 0.067 | 1.232 / -0.046 | 1.164 / 0.039 | 1.182 / 0.039 | n/a (not measured) |
| amazon-score-es | mae / qwk | 1.195 / 0.248 | 1.223 / 0.022 | 1.188 / 0.164 | 1.199 / 0.274 | 1.330 / -0.007 | 1.260 / 0.226 | 1.200 / 0.260 | 1.226 / 0.004 | 1.314 / -0.058 | 1.217 / 0.022 | 1.223 / 0.022 | n/a (not measured) |
| amazon-score-fr | mae / qwk | 1.177 / 0.202 | 1.216 / 0.051 | 1.162 / 0.160 | 1.188 / 0.152 | 1.301 / 0.017 | 1.250 / 0.199 | 1.195 / 0.176 | 1.217 / 0.012 | 1.260 / -0.007 | 1.198 / 0.051 | 1.216 / 0.051 | n/a (not measured) |
| amazon-score-ja | mae / qwk | 1.191 / 0.090 | 1.240 / 0.015 | 1.170 / 0.035 | 1.166 / 0.125 | 1.432 / -0.007 | 1.194 / 0.125 | 1.176 / 0.185 | 1.236 / 0.010 | 1.219 / 0.093 | 1.208 / 0.015 | 1.240 / 0.015 | 1.208 / 0.015 |
| amazon-score-zh | mae / qwk | 1.150 / 0.142 | 1.225 / -0.053 | 1.140 / 0.052 | 1.134 / 0.161 | 1.402 / -0.003 | 1.190 / 0.173 | 1.136 / 0.232 | 1.209 / -0.017 | 1.215 / 0.067 | 1.181 / -0.053 | 1.225 / -0.053 | n/a (not measured) |

### Calibration: ECE raw → temperature-scaled

Evaluation half of each suite. One temperature per engine model and calibration bucket, fit on the pooled other halves of every suite in that bucket (benchmark-spec §9); the classifier has one model, and so one calibrator, per dataset.

| Suite | krite | arch-late8-broad | arch-late8-broad-s14 | arch-late8-brier | arch-late8-brier-s14 | arch-late8-ord | arch-late8-ord-s14 | arch-late8-e2 | arch-late8-e2-s14 | krite-v1 | krite-v1-raw | krite-v1-nocache |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agnews-choice | 0.152 → 0.171 | 0.207 → 0.085 | 0.183 → 0.085 | 0.226 → 0.100 | 0.219 → 0.122 | 0.170 → 0.066 | 0.130 → 0.087 | 0.217 → 0.063 | 0.258 → 0.092 | 0.085 → 0.085 | 0.207 → 0.085 | 0.085 → 0.085 |
| banking77-choice | 0.046 → 0.048 | 0.073 → 0.048 | 0.045 → 0.037 | 0.053 → 0.053 | 0.102 → 0.096 | 0.060 → 0.060 | 0.030 → 0.067 | 0.042 → 0.045 | 0.057 → 0.042 | 0.048 → 0.048 | 0.073 → 0.048 | n/a (not measured) |
| boolq-noul | 0.070 → 0.054 | 0.075 → 0.107 | 0.063 → 0.039 | 0.073 → 0.115 | 0.053 → 0.037 | 0.050 → 0.097 | 0.085 → 0.080 | 0.050 → 0.096 | 0.077 → 0.066 | 0.107 → 0.107 | 0.075 → 0.107 | 0.107 → 0.076 |
| sst5-score | 0.176 → 0.044 | 0.176 → 0.056 | 0.140 → 0.075 | 0.113 → 0.063 | 0.118 → 0.062 | 0.125 → 0.035 | 0.176 → 0.131 | 0.289 → 0.065 | 0.339 → 0.066 | 0.056 → 0.056 | 0.176 → 0.056 | n/a (not measured) |
| massive-choice-en | 0.056 → 0.031 | 0.061 → 0.035 | 0.052 → 0.048 | 0.059 → 0.058 | 0.075 → 0.062 | 0.060 → 0.044 | 0.048 → 0.039 | 0.053 → 0.031 | 0.050 → 0.039 | 0.035 → 0.035 | 0.061 → 0.035 | n/a (not measured) |
| massive-choice-ko | 0.071 → 0.052 | 0.067 → 0.045 | 0.041 → 0.055 | 0.054 → 0.048 | 0.101 → 0.099 | 0.050 → 0.051 | 0.048 → 0.065 | 0.068 → 0.036 | 0.039 → 0.052 | 0.045 → 0.045 | 0.067 → 0.045 | 0.045 → 0.041 |
| massive-choice-ja | 0.043 → 0.048 | 0.032 → 0.042 | 0.035 → 0.059 | 0.060 → 0.041 | 0.071 → 0.067 | 0.050 → 0.049 | 0.054 → 0.049 | 0.045 → 0.031 | 0.065 → 0.049 | 0.042 → 0.042 | 0.032 → 0.042 | n/a (not measured) |
| massive-choice-zh-CN | 0.042 → 0.040 | 0.041 → 0.053 | 0.052 → 0.053 | 0.051 → 0.049 | 0.049 → 0.037 | 0.052 → 0.044 | 0.027 → 0.045 | 0.040 → 0.051 | 0.043 → 0.037 | 0.053 → 0.053 | 0.041 → 0.053 | n/a (not measured) |
| massive-choice-de | 0.049 → 0.045 | 0.075 → 0.070 | 0.056 → 0.055 | 0.057 → 0.064 | 0.090 → 0.076 | 0.042 → 0.040 | 0.050 → 0.044 | 0.075 → 0.040 | 0.057 → 0.045 | 0.070 → 0.070 | 0.075 → 0.070 | n/a (not measured) |
| massive-choice-es | 0.064 → 0.039 | 0.049 → 0.032 | 0.063 → 0.035 | 0.065 → 0.063 | 0.062 → 0.090 | 0.052 → 0.058 | 0.062 → 0.042 | 0.064 → 0.037 | 0.064 → 0.045 | 0.032 → 0.032 | 0.049 → 0.032 | n/a (not measured) |
| massive-choice-fr | 0.039 → 0.037 | 0.039 → 0.039 | 0.040 → 0.034 | 0.042 → 0.024 | 0.058 → 0.083 | 0.065 → 0.058 | 0.086 → 0.064 | 0.073 → 0.042 | 0.084 → 0.067 | 0.039 → 0.039 | 0.039 → 0.039 | n/a (not measured) |
| massive-choice-ar | 0.078 → 0.024 | 0.097 → 0.082 | 0.063 → 0.054 | 0.057 → 0.063 | 0.069 → 0.079 | 0.063 → 0.066 | 0.086 → 0.090 | 0.107 → 0.074 | 0.094 → 0.076 | 0.082 → 0.082 | 0.097 → 0.082 | n/a (not measured) |
| massive-choice-hi | 0.080 → 0.054 | 0.046 → 0.043 | 0.047 → 0.031 | 0.062 → 0.058 | 0.094 → 0.081 | 0.082 → 0.083 | 0.060 → 0.055 | 0.089 → 0.056 | 0.092 → 0.054 | 0.043 → 0.043 | 0.046 → 0.043 | n/a (not measured) |
| massive-choice-ru | 0.069 → 0.036 | 0.083 → 0.060 | 0.052 → 0.052 | 0.058 → 0.047 | 0.073 → 0.080 | 0.053 → 0.046 | 0.036 → 0.024 | 0.067 → 0.066 | 0.089 → 0.043 | 0.060 → 0.060 | 0.083 → 0.060 | n/a (not measured) |
| xnli-noul-en | 0.166 → 0.090 | 0.058 → 0.075 | 0.052 → 0.062 | 0.048 → 0.045 | 0.048 → 0.045 | 0.044 → 0.058 | 0.039 → 0.042 | 0.090 → 0.059 | 0.080 → 0.065 | 0.075 → 0.075 | 0.058 → 0.075 | n/a (not measured) |
| xnli-noul-de | 0.119 → 0.054 | 0.137 → 0.104 | 0.106 → 0.073 | 0.110 → 0.067 | 0.030 → 0.031 | 0.102 → 0.057 | 0.113 → 0.067 | 0.155 → 0.086 | 0.140 → 0.063 | 0.104 → 0.104 | 0.137 → 0.104 | n/a (not measured) |
| xnli-noul-es | 0.146 → 0.045 | 0.103 → 0.066 | 0.069 → 0.051 | 0.107 → 0.066 | 0.045 → 0.028 | 0.142 → 0.058 | 0.105 → 0.064 | 0.164 → 0.094 | 0.130 → 0.058 | 0.066 → 0.066 | 0.103 → 0.066 | n/a (not measured) |
| xnli-noul-fr | 0.108 → 0.056 | 0.080 → 0.046 | 0.076 → 0.049 | 0.074 → 0.064 | 0.057 → 0.058 | 0.114 → 0.069 | 0.076 → 0.073 | 0.116 → 0.056 | 0.093 → 0.048 | 0.046 → 0.046 | 0.080 → 0.046 | n/a (not measured) |
| xnli-noul-ar | 0.108 → 0.070 | 0.105 → 0.052 | 0.061 → 0.064 | 0.099 → 0.083 | 0.032 → 0.037 | 0.126 → 0.084 | 0.069 → 0.061 | 0.152 → 0.071 | 0.165 → 0.107 | 0.052 → 0.052 | 0.105 → 0.052 | n/a (not measured) |
| xnli-noul-hi | 0.166 → 0.098 | 0.099 → 0.062 | 0.098 → 0.052 | 0.156 → 0.101 | 0.040 → 0.082 | 0.120 → 0.088 | 0.093 → 0.044 | 0.160 → 0.066 | 0.155 → 0.053 | 0.062 → 0.062 | 0.099 → 0.062 | n/a (not measured) |
| xnli-noul-ru | 0.179 → 0.087 | 0.083 → 0.038 | 0.069 → 0.051 | 0.102 → 0.073 | 0.019 → 0.037 | 0.119 → 0.091 | 0.066 → 0.045 | 0.166 → 0.097 | 0.156 → 0.089 | 0.038 → 0.038 | 0.083 → 0.038 | n/a (not measured) |
| xnli-noul-zh | 0.147 → 0.087 | 0.096 → 0.076 | 0.103 → 0.091 | 0.112 → 0.072 | 0.037 → 0.031 | 0.118 → 0.077 | 0.101 → 0.053 | 0.147 → 0.074 | 0.131 → 0.050 | 0.076 → 0.076 | 0.096 → 0.076 | n/a (not measured) |
| amazon-score-en | 0.103 → 0.145 | 0.083 → 0.056 | 0.068 → 0.089 | 0.044 → 0.051 | 0.310 → 0.036 | 0.097 → 0.085 | 0.075 → 0.099 | 0.128 → 0.033 | 0.070 → 0.070 | 0.056 → 0.056 | 0.083 → 0.056 | n/a (not measured) |
| amazon-score-de | 0.046 → 0.072 | 0.116 → 0.047 | 0.046 → 0.031 | 0.115 → 0.081 | 0.327 → 0.003 | 0.083 → 0.054 | 0.032 → 0.048 | 0.158 → 0.041 | 0.091 → 0.023 | 0.047 → 0.047 | 0.116 → 0.047 | n/a (not measured) |
| amazon-score-es | 0.026 → 0.039 | 0.096 → 0.021 | 0.035 → 0.018 | 0.058 → 0.036 | 0.297 → 0.022 | 0.061 → 0.027 | 0.049 → 0.065 | 0.150 → 0.024 | 0.085 → 0.013 | 0.021 → 0.021 | 0.096 → 0.021 | n/a (not measured) |
| amazon-score-fr | 0.033 → 0.021 | 0.087 → 0.074 | 0.041 → 0.037 | 0.047 → 0.029 | 0.301 → 0.008 | 0.065 → 0.078 | 0.019 → 0.005 | 0.164 → 0.047 | 0.093 → 0.024 | 0.074 → 0.074 | 0.087 → 0.074 | n/a (not measured) |
| amazon-score-ja | 0.029 → 0.023 | 0.104 → 0.025 | 0.032 → 0.032 | 0.057 → 0.032 | 0.345 → 0.013 | 0.048 → 0.026 | 0.055 → 0.067 | 0.175 → 0.036 | 0.089 → 0.039 | 0.025 → 0.025 | 0.104 → 0.025 | 0.025 → 0.012 |
| amazon-score-zh | 0.034 → 0.010 | 0.113 → 0.059 | 0.045 → 0.036 | 0.042 → 0.014 | 0.376 → 0.039 | 0.057 → 0.033 | 0.044 → 0.056 | 0.200 → 0.034 | 0.092 → 0.020 | 0.059 → 0.059 | 0.113 → 0.059 | n/a (not measured) |
| **mean over suites** | 0.087 → 0.058 | 0.089 → 0.057 | 0.066 → 0.052 | 0.079 → 0.059 | 0.125 → 0.055 | 0.081 → 0.060 | 0.068 → 0.060 | 0.122 → 0.055 | 0.106 → 0.053 | 0.057 → 0.057 | 0.089 → 0.057 | 0.065 → 0.053 |

### Calibration: NLL and Brier, raw → scaled

| Suite | krite | arch-late8-broad | arch-late8-broad-s14 | arch-late8-brier | arch-late8-brier-s14 | arch-late8-ord | arch-late8-ord-s14 | arch-late8-e2 | arch-late8-e2-s14 | krite-v1 | krite-v1-raw | krite-v1-nocache |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agnews-choice | 0.988 → 0.979 · B 0.548 → 0.552 | 1.317 → 0.923 · B 0.558 → 0.494 | 1.081 → 0.873 · B 0.515 → 0.475 | 1.265 → 0.914 · B 0.580 → 0.499 | 1.123 → 0.971 · B 0.621 → 0.554 | 1.067 → 0.858 · B 0.501 → 0.460 | 0.828 → 0.761 · B 0.457 → 0.435 | 1.395 → 0.905 · B 0.548 → 0.477 | 1.546 → 0.931 · B 0.621 → 0.501 | 0.923 → 0.923 · B 0.494 → 0.494 | 1.317 → 0.923 · B 0.558 → 0.494 | 0.923 → 0.923 · B 0.494 → 0.494 |
| banking77-choice | 0.365 → 0.324 · B 0.153 → 0.154 | 0.404 → 0.379 · B 0.196 → 0.190 | 0.277 → 0.272 · B 0.126 → 0.128 | 0.375 → 0.375 · B 0.191 → 0.191 | 0.830 → 0.823 · B 0.364 → 0.374 | 0.371 → 0.360 · B 0.182 → 0.180 | 0.287 → 0.282 · B 0.126 → 0.129 | 0.342 → 0.260 · B 0.120 → 0.116 | 0.275 → 0.219 · B 0.116 → 0.109 | 0.379 → 0.379 · B 0.190 → 0.190 | 0.404 → 0.379 · B 0.196 → 0.190 | n/a |
| boolq-noul | 0.621 → 0.631 · B 0.435 → 0.441 | 0.549 → 0.566 · B 0.372 → 0.382 | 0.593 → 0.592 · B 0.407 → 0.407 | 0.536 → 0.559 · B 0.359 → 0.376 | 0.670 → 0.670 · B 0.478 → 0.477 | 0.547 → 0.577 · B 0.368 → 0.390 | 0.592 → 0.599 · B 0.412 → 0.415 | 0.512 → 0.538 · B 0.337 → 0.355 | 0.521 → 0.524 · B 0.338 → 0.345 | 0.566 → 0.566 · B 0.382 → 0.382 | 0.549 → 0.566 · B 0.372 → 0.382 | 0.566 → 0.548 · B 0.382 → 0.371 |
| sst5-score | 1.580 → 1.507 · B 0.796 → 0.759 | 1.553 → 1.443 · B 0.775 → 0.735 | 1.469 → 1.428 · B 0.769 → 0.743 | 1.424 → 1.384 · B 0.729 → 0.715 | 1.558 → 1.598 · B 0.789 → 0.796 | 1.399 → 1.366 · B 0.726 → 0.707 | 1.518 → 1.433 · B 0.771 → 0.738 | 1.695 → 1.409 · B 0.847 → 0.728 | 1.833 → 1.452 · B 0.906 → 0.744 | 1.443 → 1.443 · B 0.735 → 0.735 | 1.553 → 1.443 · B 0.775 → 0.735 | n/a |
| massive-choice-en | 0.320 → 0.269 · B 0.150 → 0.142 | 0.288 → 0.261 · B 0.134 → 0.130 | 0.220 → 0.212 · B 0.117 → 0.116 | 0.277 → 0.275 · B 0.146 → 0.146 | 0.559 → 0.557 · B 0.277 → 0.276 | 0.335 → 0.306 · B 0.150 → 0.147 | 0.290 → 0.273 · B 0.134 → 0.132 | 0.329 → 0.231 · B 0.114 → 0.106 | 0.306 → 0.215 · B 0.119 → 0.112 | 0.261 → 0.261 · B 0.130 → 0.130 | 0.288 → 0.261 · B 0.134 → 0.130 | n/a |
| massive-choice-ko | 0.405 → 0.329 · B 0.177 → 0.162 | 0.380 → 0.352 · B 0.169 → 0.167 | 0.393 → 0.377 · B 0.163 → 0.163 | 0.364 → 0.355 · B 0.173 → 0.171 | 0.769 → 0.765 · B 0.362 → 0.364 | 0.326 → 0.314 · B 0.153 → 0.150 | 0.346 → 0.336 · B 0.155 → 0.155 | 0.379 → 0.275 · B 0.148 → 0.137 | 0.369 → 0.285 · B 0.113 → 0.123 | 0.352 → 0.352 · B 0.167 → 0.167 | 0.380 → 0.352 · B 0.169 → 0.167 | 0.352 → 0.351 · B 0.167 → 0.167 |
| massive-choice-ja | 0.228 → 0.215 · B 0.105 → 0.111 | 0.235 → 0.224 · B 0.109 → 0.111 | 0.173 → 0.182 · B 0.081 → 0.086 | 0.281 → 0.274 · B 0.151 → 0.148 | 0.575 → 0.571 · B 0.279 → 0.279 | 0.205 → 0.210 · B 0.127 → 0.124 | 0.217 → 0.216 · B 0.108 → 0.110 | 0.216 → 0.177 · B 0.087 → 0.086 | 0.314 → 0.213 · B 0.138 → 0.118 | 0.224 → 0.224 · B 0.111 → 0.111 | 0.235 → 0.224 · B 0.109 → 0.111 | n/a |
| massive-choice-zh-CN | 0.236 → 0.226 · B 0.112 → 0.112 | 0.219 → 0.225 · B 0.107 → 0.109 | 0.212 → 0.217 · B 0.098 → 0.104 | 0.270 → 0.271 · B 0.125 → 0.127 | 0.555 → 0.556 · B 0.274 → 0.275 | 0.275 → 0.268 · B 0.135 → 0.135 | 0.240 → 0.240 · B 0.110 → 0.112 | 0.254 → 0.214 · B 0.094 → 0.099 | 0.266 → 0.205 · B 0.087 → 0.090 | 0.225 → 0.225 · B 0.109 → 0.109 | 0.219 → 0.225 · B 0.107 → 0.109 | n/a |
| massive-choice-de | 0.372 → 0.331 · B 0.168 → 0.166 | 0.354 → 0.339 · B 0.184 → 0.180 | 0.277 → 0.283 · B 0.159 → 0.157 | 0.351 → 0.349 · B 0.185 → 0.183 | 0.814 → 0.793 · B 0.406 → 0.400 | 0.317 → 0.307 · B 0.162 → 0.161 | 0.371 → 0.358 · B 0.183 → 0.180 | 0.409 → 0.301 · B 0.161 → 0.152 | 0.355 → 0.278 · B 0.145 → 0.137 | 0.339 → 0.339 · B 0.180 → 0.180 | 0.354 → 0.339 · B 0.184 → 0.180 | n/a |
| massive-choice-es | 0.405 → 0.326 · B 0.175 → 0.165 | 0.308 → 0.291 · B 0.158 → 0.154 | 0.296 → 0.281 · B 0.136 → 0.135 | 0.356 → 0.341 · B 0.181 → 0.178 | 0.585 → 0.584 · B 0.288 → 0.288 | 0.349 → 0.329 · B 0.174 → 0.169 | 0.286 → 0.285 · B 0.136 → 0.138 | 0.374 → 0.264 · B 0.149 → 0.136 | 0.306 → 0.226 · B 0.128 → 0.111 | 0.291 → 0.291 · B 0.154 → 0.154 | 0.308 → 0.291 · B 0.158 → 0.154 | n/a |
| massive-choice-fr | 0.271 → 0.246 · B 0.116 → 0.118 | 0.293 → 0.272 · B 0.129 → 0.129 | 0.222 → 0.221 · B 0.123 → 0.121 | 0.241 → 0.242 · B 0.117 → 0.117 | 0.522 → 0.530 · B 0.252 → 0.256 | 0.335 → 0.311 · B 0.161 → 0.156 | 0.334 → 0.317 · B 0.170 → 0.166 | 0.448 → 0.301 · B 0.146 → 0.137 | 0.506 → 0.321 · B 0.180 → 0.155 | 0.272 → 0.272 · B 0.129 → 0.129 | 0.293 → 0.272 · B 0.129 → 0.129 | n/a |
| massive-choice-ar | 0.563 → 0.471 · B 0.234 → 0.223 | 0.541 → 0.494 · B 0.261 → 0.249 | 0.522 → 0.506 · B 0.238 → 0.235 | 0.517 → 0.506 · B 0.238 → 0.236 | 0.880 → 0.865 · B 0.390 → 0.391 | 0.469 → 0.457 · B 0.209 → 0.210 | 0.476 → 0.464 · B 0.226 → 0.223 | 0.642 → 0.438 · B 0.251 → 0.221 | 0.538 → 0.418 · B 0.243 → 0.211 | 0.494 → 0.494 · B 0.249 → 0.249 | 0.541 → 0.494 · B 0.261 → 0.249 | n/a |
| massive-choice-hi | 0.477 → 0.403 · B 0.226 → 0.208 | 0.396 → 0.385 · B 0.203 → 0.199 | 0.479 → 0.460 · B 0.204 → 0.203 | 0.386 → 0.385 · B 0.189 → 0.189 | 1.026 → 0.997 · B 0.486 → 0.478 | 0.486 → 0.458 · B 0.224 → 0.221 | 0.429 → 0.415 · B 0.202 → 0.200 | 0.479 → 0.358 · B 0.200 → 0.181 | 0.548 → 0.371 · B 0.190 → 0.175 | 0.385 → 0.385 · B 0.199 → 0.199 | 0.396 → 0.385 · B 0.203 → 0.199 | n/a |
| massive-choice-ru | 0.394 → 0.327 · B 0.171 → 0.158 | 0.380 → 0.346 · B 0.193 → 0.182 | 0.290 → 0.279 · B 0.139 → 0.137 | 0.304 → 0.301 · B 0.143 → 0.145 | 0.699 → 0.692 · B 0.343 → 0.341 | 0.358 → 0.345 · B 0.174 → 0.170 | 0.300 → 0.297 · B 0.151 → 0.149 | 0.346 → 0.261 · B 0.154 → 0.136 | 0.507 → 0.331 · B 0.187 → 0.160 | 0.346 → 0.346 · B 0.182 → 0.182 | 0.380 → 0.346 · B 0.193 → 0.182 | n/a |
| xnli-noul-en | 0.653 → 0.539 · B 0.390 → 0.357 | 0.292 → 0.291 · B 0.155 → 0.162 | 0.328 → 0.321 · B 0.161 → 0.169 | 0.354 → 0.353 · B 0.211 → 0.213 | 0.697 → 0.697 · B 0.504 → 0.503 | 0.335 → 0.331 · B 0.200 → 0.200 | 0.277 → 0.283 · B 0.161 → 0.165 | 0.392 → 0.338 · B 0.220 → 0.206 | 0.352 → 0.302 · B 0.191 → 0.178 | 0.291 → 0.291 · B 0.162 → 0.162 | 0.292 → 0.291 · B 0.155 → 0.162 | n/a |
| xnli-noul-de | 0.600 → 0.534 · B 0.378 → 0.354 | 0.605 → 0.518 · B 0.350 → 0.331 | 0.574 → 0.509 · B 0.330 → 0.315 | 0.497 → 0.465 · B 0.315 → 0.302 | 0.693 → 0.693 · B 0.500 → 0.500 | 0.484 → 0.433 · B 0.302 → 0.280 | 0.560 → 0.497 · B 0.346 → 0.326 | 0.663 → 0.502 · B 0.371 → 0.331 | 0.657 → 0.502 · B 0.351 → 0.322 | 0.518 → 0.518 · B 0.331 → 0.331 | 0.605 → 0.518 · B 0.350 → 0.331 | n/a |
| xnli-noul-es | 0.630 → 0.543 · B 0.398 → 0.364 | 0.543 → 0.474 · B 0.322 → 0.304 | 0.467 → 0.429 · B 0.279 → 0.271 | 0.518 → 0.473 · B 0.332 → 0.312 | 0.685 → 0.686 · B 0.492 → 0.492 | 0.595 → 0.490 · B 0.344 → 0.316 | 0.497 → 0.446 · B 0.304 → 0.287 | 0.638 → 0.481 · B 0.374 → 0.321 | 0.602 → 0.453 · B 0.313 → 0.285 | 0.474 → 0.474 · B 0.304 → 0.304 | 0.543 → 0.474 · B 0.322 → 0.304 | n/a |
| xnli-noul-fr | 0.562 → 0.522 · B 0.362 → 0.347 | 0.448 → 0.412 · B 0.284 → 0.269 | 0.427 → 0.405 · B 0.277 → 0.267 | 0.426 → 0.410 · B 0.278 → 0.266 | 0.686 → 0.687 · B 0.493 → 0.494 | 0.468 → 0.421 · B 0.302 → 0.277 | 0.397 → 0.383 · B 0.251 → 0.245 | 0.486 → 0.403 · B 0.290 → 0.259 | 0.496 → 0.418 · B 0.290 → 0.269 | 0.412 → 0.412 · B 0.269 → 0.269 | 0.448 → 0.412 · B 0.284 → 0.269 | n/a |
| xnli-noul-ar | 0.633 → 0.580 · B 0.413 → 0.393 | 0.589 → 0.527 · B 0.362 → 0.345 | 0.477 → 0.464 · B 0.284 → 0.286 | 0.526 → 0.495 · B 0.342 → 0.327 | 0.693 → 0.693 · B 0.500 → 0.499 | 0.605 → 0.515 · B 0.372 → 0.341 | 0.485 → 0.467 · B 0.305 → 0.299 | 0.642 → 0.502 · B 0.381 → 0.334 | 0.723 → 0.540 · B 0.404 → 0.353 | 0.527 → 0.527 · B 0.345 → 0.345 | 0.589 → 0.527 · B 0.362 → 0.345 | n/a |
| xnli-noul-hi | 0.644 → 0.588 · B 0.446 → 0.408 | 0.575 → 0.533 · B 0.379 → 0.359 | 0.581 → 0.550 · B 0.378 → 0.368 | 0.644 → 0.582 · B 0.429 → 0.400 | 0.692 → 0.692 · B 0.498 → 0.498 | 0.606 → 0.542 · B 0.386 → 0.363 | 0.555 → 0.527 · B 0.370 → 0.355 | 0.663 → 0.533 · B 0.406 → 0.359 | 0.654 → 0.529 · B 0.400 → 0.355 | 0.533 → 0.533 · B 0.359 → 0.359 | 0.575 → 0.533 · B 0.379 → 0.359 | n/a |
| xnli-noul-ru | 0.764 → 0.639 · B 0.491 → 0.442 | 0.476 → 0.446 · B 0.299 → 0.290 | 0.463 → 0.436 · B 0.294 → 0.284 | 0.507 → 0.476 · B 0.330 → 0.315 | 0.689 → 0.690 · B 0.496 → 0.496 | 0.561 → 0.483 · B 0.326 → 0.310 | 0.497 → 0.467 · B 0.311 → 0.304 | 0.636 → 0.500 · B 0.388 → 0.338 | 0.634 → 0.484 · B 0.348 → 0.316 | 0.446 → 0.446 · B 0.290 → 0.290 | 0.476 → 0.446 · B 0.299 → 0.290 | n/a |
| xnli-noul-zh | 0.714 → 0.598 · B 0.432 → 0.403 | 0.459 → 0.421 · B 0.278 → 0.267 | 0.508 → 0.465 · B 0.269 → 0.271 | 0.486 → 0.451 · B 0.304 → 0.291 | 0.688 → 0.688 · B 0.494 → 0.495 | 0.524 → 0.457 · B 0.311 → 0.295 | 0.542 → 0.486 · B 0.320 → 0.307 | 0.596 → 0.461 · B 0.343 → 0.302 | 0.591 → 0.459 · B 0.331 → 0.297 | 0.421 → 0.421 · B 0.267 → 0.267 | 0.459 → 0.421 · B 0.278 → 0.267 | n/a |
| amazon-score-en | 1.542 → 1.570 · B 0.767 → 0.783 | 1.595 → 1.566 · B 0.791 → 0.780 | 1.495 → 1.513 · B 0.749 → 0.757 | 1.539 → 1.544 · B 0.771 → 0.774 | 1.834 → 1.609 · B 0.921 → 0.800 | 1.604 → 1.557 · B 0.771 → 0.773 | 1.507 → 1.536 · B 0.751 → 0.767 | 1.597 → 1.571 · B 0.795 → 0.784 | 1.613 → 1.590 · B 0.796 → 0.791 | 1.566 → 1.566 · B 0.780 → 0.780 | 1.595 → 1.566 · B 0.791 → 0.780 | n/a |
| amazon-score-de | 1.578 → 1.587 · B 0.786 → 0.791 | 1.660 → 1.610 · B 0.827 → 0.801 | 1.594 → 1.592 · B 0.794 → 0.793 | 1.596 → 1.588 · B 0.799 → 0.793 | 1.906 → 1.610 · B 0.965 → 0.800 | 1.727 → 1.625 · B 0.843 → 0.806 | 1.574 → 1.582 · B 0.786 → 0.789 | 1.712 → 1.611 · B 0.846 → 0.801 | 1.701 → 1.621 · B 0.833 → 0.805 | 1.610 → 1.610 · B 0.801 → 0.801 | 1.660 → 1.610 · B 0.827 → 0.801 | n/a |
| amazon-score-es | 1.581 → 1.588 · B 0.788 → 0.791 | 1.627 → 1.596 · B 0.817 → 0.797 | 1.598 → 1.595 · B 0.796 → 0.795 | 1.614 → 1.602 · B 0.804 → 0.799 | 1.880 → 1.608 · B 0.940 → 0.800 | 1.773 → 1.647 · B 0.847 → 0.813 | 1.577 → 1.585 · B 0.786 → 0.790 | 1.719 → 1.610 · B 0.845 → 0.800 | 1.746 → 1.633 · B 0.850 → 0.809 | 1.596 → 1.596 · B 0.797 → 0.797 | 1.627 → 1.596 · B 0.817 → 0.797 | n/a |
| amazon-score-fr | 1.605 → 1.598 · B 0.798 → 0.796 | 1.636 → 1.600 · B 0.818 → 0.797 | 1.607 → 1.601 · B 0.801 → 0.798 | 1.604 → 1.596 · B 0.799 → 0.795 | 1.829 → 1.608 · B 0.921 → 0.799 | 1.716 → 1.625 · B 0.830 → 0.805 | 1.601 → 1.601 · B 0.797 → 0.797 | 1.770 → 1.629 · B 0.863 → 0.808 | 1.709 → 1.621 · B 0.835 → 0.804 | 1.600 → 1.600 · B 0.797 → 0.797 | 1.636 → 1.600 · B 0.818 → 0.797 | n/a |
| amazon-score-ja | 1.641 → 1.614 · B 0.813 → 0.802 | 1.664 → 1.609 · B 0.826 → 0.800 | 1.611 → 1.605 · B 0.802 → 0.799 | 1.620 → 1.606 · B 0.809 → 0.801 | 1.911 → 1.608 · B 0.968 → 0.800 | 1.689 → 1.614 · B 0.815 → 0.800 | 1.588 → 1.591 · B 0.791 → 0.793 | 1.837 → 1.642 · B 0.886 → 0.813 | 1.700 → 1.617 · B 0.832 → 0.803 | 1.609 → 1.609 · B 0.800 → 0.800 | 1.664 → 1.609 · B 0.826 → 0.800 | 1.609 → 1.606 · B 0.800 → 0.799 |
| amazon-score-zh | 1.633 → 1.613 · B 0.809 → 0.802 | 1.664 → 1.614 · B 0.825 → 0.802 | 1.581 → 1.584 · B 0.788 → 0.789 | 1.582 → 1.582 · B 0.791 → 0.789 | 2.008 → 1.613 · B 1.012 → 0.801 | 1.654 → 1.606 · B 0.805 → 0.797 | 1.590 → 1.594 · B 0.792 → 0.793 | 1.700 → 1.603 · B 0.836 → 0.797 | 1.706 → 1.619 · B 0.835 → 0.804 | 1.614 → 1.614 · B 0.802 → 0.802 | 1.664 → 1.614 · B 0.825 → 0.802 | n/a |

### Option-order invariance (100 cases per suite)

| Engine | Suite | Full-perm flip | Reverse flip | Max prob dev | Errors |
|---|---|---|---|---|---|
| krite | agnews-choice | 0.000 | 0.000 | 0.0000 | 0% |
| krite | banking77-choice | 0.000 | 0.000 | 0.0000 | 0% |
| krite | massive-choice-en | 0.000 | 0.000 | 0.0000 | 0% |
| krite | massive-choice-ko | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-broad | agnews-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-broad | banking77-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-broad | massive-choice-en | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-broad | massive-choice-ko | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-broad-s14 | agnews-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-broad-s14 | banking77-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-broad-s14 | massive-choice-en | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-broad-s14 | massive-choice-ko | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-brier | agnews-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-brier | banking77-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-brier | massive-choice-en | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-brier | massive-choice-ko | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-brier-s14 | agnews-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-brier-s14 | banking77-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-brier-s14 | massive-choice-en | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-brier-s14 | massive-choice-ko | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-ord | agnews-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-ord | banking77-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-ord | massive-choice-en | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-ord | massive-choice-ko | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-ord-s14 | agnews-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-ord-s14 | banking77-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-ord-s14 | massive-choice-en | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-ord-s14 | massive-choice-ko | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-e2 | agnews-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-e2 | banking77-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-e2 | massive-choice-en | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-e2 | massive-choice-ko | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-e2-s14 | agnews-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-e2-s14 | banking77-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-e2-s14 | massive-choice-en | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-e2-s14 | massive-choice-ko | 0.000 | 0.000 | 0.0000 | 0% |
| krite-v1 | agnews-choice | 0.000 | 0.000 | 0.0000 | 0% |
| krite-v1 | banking77-choice | 0.000 | 0.000 | 0.0000 | 0% |
| krite-v1 | massive-choice-en | 0.000 | 0.000 | 0.0000 | 0% |
| krite-v1 | massive-choice-ko | 0.000 | 0.000 | 0.0000 | 0% |
| krite-v1-raw | agnews-choice | 0.000 | 0.000 | 0.0000 | 0% |
| krite-v1-raw | banking77-choice | 0.000 | 0.000 | 0.0000 | 0% |
| krite-v1-raw | massive-choice-en | 0.000 | 0.000 | 0.0000 | 0% |
| krite-v1-raw | massive-choice-ko | 0.000 | 0.000 | 0.0000 | 0% |

### Question interference (agnews-choice, 100 cases; alone vs. with fillers)

| Engine | Fillers | Argmax change | Max prob dev |
|---|---|---|---|
| krite | 3 | 0.000 | 0.0000 |
| krite | 15 | 0.000 | 0.0000 |
| arch-late8-broad | 3 | 0.000 | 0.0000 |
| arch-late8-broad | 15 | 0.000 | 0.0000 |
| arch-late8-broad-s14 | 3 | 0.000 | 0.0000 |
| arch-late8-broad-s14 | 15 | 0.000 | 0.0000 |
| arch-late8-brier | 3 | 0.000 | 0.0000 |
| arch-late8-brier | 15 | 0.000 | 0.0000 |
| arch-late8-brier-s14 | 3 | 0.000 | 0.0000 |
| arch-late8-brier-s14 | 15 | 0.000 | 0.0000 |
| arch-late8-ord | 3 | 0.000 | 0.0000 |
| arch-late8-ord | 15 | 0.000 | 0.0000 |
| arch-late8-ord-s14 | 3 | 0.000 | 0.0000 |
| arch-late8-ord-s14 | 15 | 0.000 | 0.0000 |
| arch-late8-e2 | 3 | 0.000 | 0.0000 |
| arch-late8-e2 | 15 | 0.000 | 0.0000 |
| arch-late8-e2-s14 | 3 | 0.000 | 0.0000 |
| arch-late8-e2-s14 | 15 | 0.000 | 0.0000 |
| krite-v1 | 3 | 0.000 | 0.0000 |
| krite-v1 | 15 | 0.000 | 0.0000 |
| krite-v1-raw | 3 | 0.000 | 0.0000 |
| krite-v1-raw | 15 | 0.000 | 0.0000 |

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
| krite-v1 | burst | cold | 512/1/4 | http | 200 | 90.1 | 90.5 | 90.8 | 11.1 | 0% | AC |  |
| krite-v1 | burst | cold | 512/1/4 | runtime | 200 | 89.9 | 90.2 | 90.5 | 11.1 | 0% | AC |  |
| krite-v1-nocache | burst | cold | 512/1/4 | http | 200 | 98.6 | 99.8 | 100.7 | 10.2 | 0% | AC |  |
| krite-v1-nocache | burst | cold | 512/1/4 | runtime | 200 | 98.4 | 99.5 | 100.1 | 10.2 | 0% | AC |  |
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
| krite-v1 | burst | cold | 512/10/4 | http | 200 | 126.8 | 128.1 | 129.1 | 78.8 | 0% | AC |  |
| krite-v1 | burst | cold | 512/10/4 | runtime | 200 | 126.5 | 127.8 | 128.7 | 78.8 | 0% | AC |  |
| krite-v1-nocache | burst | cold | 512/10/4 | http | 200 | 191.2 | 192.2 | 192.5 | 52.6 | 0% | AC |  |
| krite-v1-nocache | burst | cold | 512/10/4 | runtime | 200 | 190.9 | 191.8 | 192.1 | 52.6 | 0% | AC |  |
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
| krite-v1 | burst | cold | 512/30/4 | http | 200 | 224.9 | 225.3 | 225.5 | 133.4 | 0% | AC |  |
| krite-v1 | burst | cold | 512/30/4 | runtime | 200 | 224.5 | 224.8 | 225.0 | 133.4 | 0% | AC |  |
| krite-v1-nocache | burst | cold | 512/30/4 | http | 200 | 551.3 | 741.2 | 814.9 | 54.4 | 0% | AC |  |
| krite-v1-nocache | burst | cold | 512/30/4 | runtime | 200 | 550.7 | 740.6 | 813.9 | 54.4 | 0% | AC |  |
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
| arch-late8-brier | burst | startup | ping probe | http | 1 | 12625.0 | 12625.0 | 12625.0 | 0.1 | 0% | AC |  |
| arch-late8-brier-s14 | burst | startup | ping probe | http | 1 | 11051.9 | 11051.9 | 11051.9 | 0.1 | 0% | AC |  |
| arch-late8-broad | burst | startup | ping probe | http | 1 | 12193.3 | 12193.3 | 12193.3 | 0.1 | 0% | AC |  |
| arch-late8-broad-s14 | burst | startup | ping probe | http | 1 | 11146.8 | 11146.8 | 11146.8 | 0.1 | 0% | AC |  |
| arch-late8-e2 | burst | startup | ping probe | http | 1 | 16140.5 | 16140.5 | 16140.5 | 0.1 | 0% | AC |  |
| arch-late8-e2-s14 | burst | startup | ping probe | http | 1 | 12614.0 | 12614.0 | 12614.0 | 0.1 | 0% | AC |  |
| arch-late8-nocache | burst | startup | ping probe | http | 1 | 11084.6 | 11084.6 | 11084.6 | 0.1 | 0% | AC |  |
| arch-late8-ord | burst | startup | ping probe | http | 1 | 13618.0 | 13618.0 | 13618.0 | 0.1 | 0% | AC |  |
| arch-late8-ord-s14 | burst | startup | ping probe | http | 1 | 12988.4 | 12988.4 | 12988.4 | 0.1 | 0% | AC |  |
| arch-late8-s14 | burst | startup | ping probe | http | 1 | 11687.5 | 11687.5 | 11687.5 | 0.1 | 0% | AC |  |
| krite | burst | startup | ping probe | http | 1 | 523.6 | 523.6 | 523.6 | 1.9 | 0% | AC |  |
| krite-nocache | burst | startup | ping probe | http | 1 | 1383.5 | 1383.5 | 1383.5 | 0.7 | 0% | AC |  |
| krite-v1 | burst | startup | ping probe | http | 1 | 1616.8 | 1616.8 | 1616.8 | 0.6 | 0% | AC |  |
| krite-v1-nocache | burst | startup | ping probe | http | 1 | 1922.5 | 1922.5 | 1922.5 | 0.5 | 0% | AC |  |
| krite-v1-raw | burst | startup | ping probe | http | 1 | 1201.7 | 1201.7 | 1201.7 | 0.8 | 0% | AC |  |
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
| krite-v1 | burst | warm | 512/1/4 | http | 200 | 7.8 | 7.9 | 8.0 | 128.5 | 0% | AC |  |
| krite-v1 | burst | warm | 512/1/4 | runtime | 200 | 7.6 | 7.7 | 7.7 | 128.5 | 0% | AC |  |
| krite-v1-nocache | burst | warm | 512/1/4 | http | 200 | 147.3 | 166.7 | 191.0 | 7.1 | 0% | AC |  |
| krite-v1-nocache | burst | warm | 512/1/4 | runtime | 200 | 146.8 | 166.1 | 190.4 | 7.1 | 0% | AC |  |
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
| krite-v1 | burst | warm | 512/10/4 | http | 200 | 44.1 | 45.3 | 46.8 | 225.7 | 0% | AC |  |
| krite-v1 | burst | warm | 512/10/4 | runtime | 200 | 43.8 | 45.1 | 46.5 | 225.7 | 0% | AC |  |
| krite-v1-nocache | burst | warm | 512/10/4 | http | 200 | 234.0 | 327.2 | 459.2 | 37.9 | 0% | AC |  |
| krite-v1-nocache | burst | warm | 512/10/4 | runtime | 200 | 233.5 | 326.4 | 458.2 | 37.9 | 0% | AC |  |
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
| krite-v1 | burst | warm | 512/30/4 | http | 200 | 141.7 | 142.0 | 142.4 | 211.6 | 0% | AC |  |
| krite-v1 | burst | warm | 512/30/4 | runtime | 200 | 141.4 | 141.6 | 142.0 | 211.6 | 0% | AC |  |
| krite-v1-nocache | burst | warm | 512/30/4 | http | 200 | 542.8 | 602.8 | 735.5 | 54.5 | 0% | AC |  |
| krite-v1-nocache | burst | warm | 512/30/4 | runtime | 200 | 542.2 | 602.1 | 734.9 | 54.5 | 0% | AC |  |
| arch-b | sustained | warm | 512/1/4 | http | 9302 | 63.7 | 66.1 | 67.8 | 15.7 | 0% | AC | last-minute p50 64.0 |
| arch-d2 | sustained | warm | 512/1/4 | http | 34343 | 17.0 | 19.4 | 20.3 | 57.8 | 0% | AC | last-minute p50 17.6 |
| arch-late4 | sustained | warm | 512/1/4 | http | 79279 | 7.3 | 8.2 | 9.1 | 134.5 | 0% | AC | last-minute p50 7.9 |
| krite | sustained | warm | 512/1/4 | http | 52185 | 8.1 | 18.8 | 20.4 | 88.6 | 0% | AC | last-minute p50 7.9 |
| krite-v1 | sustained | warm | 512/1/4 | http | 72079 | 8.1 | 9.0 | 9.4 | 122.0 | 0% | AC | last-minute p50 8.5 |

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
| arch-late8-broad | arch-late8-broad | 3.45 | 1.32 |
| arch-late8-broad-s14 | arch-late8-broad-s14 | 3.47 | 1.34 |
| arch-late8-brier | arch-late8-brier | 3.49 | 1.41 |
| arch-late8-brier-s14 | arch-late8-brier-s14 | 3.48 | 1.39 |
| arch-late8-ord | arch-late8-ord | 3.81 | 1.09 |
| arch-late8-ord-s14 | arch-late8-ord-s14 | 3.41 | 0.92 |
| arch-late8-e2 | arch-late8-e2 | 3.50 | 1.30 |
| arch-late8-e2-s14 | arch-late8-e2-s14 | 3.60 | 1.31 |
| krite-v1-raw | krite-0.15b-v1 | 3.70 | 1.07 |
| krite-v1 | krite-0.15b-v1 | 4.46 | 1.28 |

<!-- GENERATED:END -->

### Stage A: broad mixture

`study` (`arch-late8`, `arch-late8-s14`) vs `broad` (`arch-late8-broad`, `arch-late8-broad-s14`), one
epoch each, everything else unchanged. Raw output:
[`stage-a.json`](../benchmarks/results/arch/stage-a.json).

| Rule | Value | Pass |
|---|---|---|
| shortfall gain | 1.659 → 0.694 (gain 0.965) | yes |
| seed agreement | seed 13: 1.416 → 0.783; seed 14: 1.902 → 0.606 | yes |
| in-domain drop | −0.006 (in-domain accuracy rose) | yes |
| invariants | flip rate 0; invariance ≤ 1.9e-6; interference ≤ 9.2e-7 | yes |

**Verdict: adopt `broad`.** Shortfall by suite, seed 13 / seed 14 (suites at 0 for all four engines
omitted):

| Suite | Target | `study` | `broad` |
|---|---|---|---|
| agnews-choice | 0.925 | 0.300 / 0.512 | 0.282 / 0.242 |
| boolq-noul | 0.825 | 0.195 / 0.220 | 0.137 / 0.150 |
| sst5-score | 0.728 | 0.264 / 0.289 | 0.219 / 0.186 |
| xnli-noul (8 languages) | 0.740–0.902 | 0.657 / 0.672 | 0.042 / 0.027 |
| amazon-score (6 languages) | 0.017–0.077 | 0 / 0.208 | 0.101 / 0 |

MultiNLI closes xnli almost entirely (xnli-noul-en 0.755 → 0.910 at seed 13). agnews, boolq, and sst5
move by 0.02–0.09 each and remain the shortfall. Amazon QWK falls from `krite-0.15b-v0`'s 0.09–0.39
to −0.05–0.36 and becomes seed-dependent; its targets are low enough that only seed 13's ja, es, and
zh miss them.

### Stage B: loss

**B1: Brier term.** `broad` (`arch-late8-broad`, `-s14`) vs `broad` + Brier (`arch-late8-brier`,
`-s14`). Raw output: [`stage-b1.json`](../benchmarks/results/arch/stage-b1.json).

| Rule | Value | Pass |
|---|---|---|
| shortfall gain | 0.694 → 2.072 (gain −1.378) | no |
| seed agreement | seed 13: 0.783 → 0.662; seed 14: 0.606 → 3.481 | no |
| in-domain drop | 0.073 | no |
| invariants | flip rate 0; invariance ≤ 1.6e-6; interference ≤ 7.3e-7 | yes |

**Verdict: reject; `broad` with cross-entropy stays the winner.** The two seeds disagree. Seed 13
improves slightly (agnews, boolq, sst5, and amazon each move by 0.02–0.07). Seed 14's training loss
(CE + Brier) sits near its starting value (1.6–1.8) for the first 1,500 steps, where seed 13 is at
0.99, and ends at 1.01 against seed 13's 0.65. Its accuracy falls on every suite family, banking77
and xnli included. A term that leaves training this sensitive to the seed is not adopted.

**B2: ordinal term.** `broad` (`arch-late8-broad`, `-s14`) vs `broad` + ordinal (`arch-late8-ord`,
`-s14`); the base is unchanged because B1 was rejected. Raw output:
[`stage-b2.json`](../benchmarks/results/arch/stage-b2.json).

| Rule | Value | Pass |
|---|---|---|
| shortfall gain | 0.694 → 0.586 (gain 0.108) | yes |
| seed agreement | seed 13: 0.783 → 0.557; seed 14: 0.606 → 0.615 | no |
| in-domain drop | 0.007 | yes |
| invariants | flip rate 0; invariance ≤ 1.7e-6; interference ≤ 8.6e-7 | yes |

**Verdict: reject; `broad` with cross-entropy stays the winner.** Seed 14 ends 0.009 above its base,
so the rule fails on seed agreement alone. Shortfall by suite, seed 13 / seed 14 (suites at 0 for all
four engines omitted):

| Suite | `broad` | `broad` + ordinal |
|---|---|---|
| agnews-choice | 0.282 / 0.242 | 0.225 / 0.220 |
| boolq-noul | 0.137 / 0.150 | 0.115 / 0.188 |
| sst5-score | 0.219 / 0.186 | 0.127 / 0.153 |
| xnli-noul (8 languages) | 0.042 / 0.027 | 0.088 / 0.054 |
| amazon-score (6 languages) | 0.101 / 0 | 0 / 0 |

The term does what it targets: sst5 improves at both seeds, and seed 13's amazon misses close. Seed
14's gain there is cancelled by boolq (+0.038) and xnli (+0.027), which the term does not touch
directly.

### Stage C: epochs

`broad`, one epoch (`arch-late8-broad`, `-s14`) vs two epochs (`arch-late8-e2`, `-s14`); the base is
unchanged because B1 and B2 were rejected. Raw output:
[`stage-c.json`](../benchmarks/results/arch/stage-c.json).

| Rule | Value | Pass |
|---|---|---|
| shortfall gain | 0.694 → 0.837 (gain −0.143) | no |
| seed agreement | seed 13: 0.783 → 0.748; seed 14: 0.606 → 0.926 | no |
| in-domain drop | −0.016 (in-domain accuracy rose) | yes |
| invariants | flip rate 0; invariance ≤ 2.4e-6; interference ≤ 1.1e-6 | yes |

**Verdict: reject.** Shortfall by suite, seed 13 / seed 14 (suites at 0 for all four engines
omitted):

| Suite | one epoch | two epochs |
|---|---|---|
| agnews-choice | 0.282 / 0.242 | 0.265 / 0.290 |
| boolq-noul | 0.137 / 0.150 | 0.087 / 0.085 |
| sst5-score | 0.219 / 0.186 | 0.213 / 0.244 |
| xnli-noul (8 languages) | 0.042 / 0.027 | 0.087 / 0.114 |
| amazon-score (6 languages) | 0.101 / 0 | 0.095 / 0.193 |

The second epoch fits the trained tasks better: boolq improves by 0.05–0.07 at both seeds and
in-domain accuracy rises. The held-out tasks (agnews, sst5, xnli, amazon) move the other way at
seed 14, which outweighs it.

**Final recipe:** `broad` mixture, cross-entropy, one epoch. The release checkpoint is its seed-13
run, `late8-broad` (shortfall 0.783).

### Release gate

`late8-broad` exported as `krite-0.15b-v1` (Candle probe tests pass), temperatures fitted on
`krite-v1-raw`, then `krite-v1` and `krite-v1-nocache` measured in the standard order. Raw output:
[`release.json`](../benchmarks/results/arch/release.json).

| Gate | Limit | `krite-0.15b-v0` | `krite-0.15b-v1` | Pass |
|---|---|---|---|---|
| accuracy shortfall | 0 | 1.416 | 0.783 | **no** |
| ECE (mean, scaled) | ≤ 0.071 | 0.058 | 0.057 | yes |
| warm latency p50 | ≤ 10 ms | 7.8 ms | 7.8 ms | yes |
| cold latency p50 | ≤ 210 ms | 91 ms | 90 ms | yes |
| throughput (30 questions) | ≥ 175 decisions/s | 183 | 212 | yes |
| option order | flip 0, ≤ 1e-5 | 0, 0 | 0, 0 | yes |
| isolation | ≤ 1e-5 | 5.3e-7 | 1.7e-7 | yes |
| cache on/off | ≤ 1e-5 | 0 | 0 | yes |

**Verdict: not released.** `krite-0.15b-v1` passes every gate except accuracy. Its shortfall is
agnews-choice 0.283, sst5-score 0.219, boolq-noul 0.137, amazon-score zh/ja/es 0.070/0.016/0.015,
and xnli-noul hi/ar 0.025/0.017. Torch and Candle agree on every suite (the shortfall equals
`arch-late8-broad`'s).

The broad mixture closed most of xnli. Each later change (Brier, ordinal, two epochs) improved seed
13 and lost at seed 14. The remaining gap sits in three held-out tasks: topic (agnews), passage QA
(boolq), and fine-grained sentiment (sst5). As fixed in the rules above, the next step is outside
this recipe: a larger encoder (mmBERT-base, ~0.4B) or a revised accuracy target.

### Feasibility: mmBERT-base

1% pilot of `base8-broad` (740 examples, 43 steps), exported and served as `krite-base-pilot` on
Candle Metal, engine alone on AC power. Raw output:
[`feasibility-base.json`](../benchmarks/results/arch/feasibility-base.json).

| Check | Limit | `krite-0.15b-v1` | mmBERT-base pilot | Pass |
|---|---|---|---|---|
| Candle numerics | weight tests pass | pass | pass (3 of 3) | yes |
| warm latency p50 | ≤ 10 ms | 7.8 ms | 18.7 ms | **no** |
| cold latency p50 | ≤ 210 ms | 90 ms | 219 ms | **no** |
| throughput (30 questions) | ≥ 175 decisions/s | 212 | 77 | **no** |
| training memory (MPS peak) | ≤ 10 GiB | 6.1 GiB | 11.7 GiB | no (accumulation not tried) |

**Verdict: infeasible; the encoder stage stops before training.** Every latency cell is about 2.4×
small's (warm 512/10/4: 44 → 130 ms; cold 512/30/4: 225 → 662 ms), close to the 2.6× ratio of
non-embedding compute, so the fp32 Candle path is compute-bound at this size rather than
overhead-bound. The pilot has 307.5M parameters (196.6M embedding, 110.9M other) and trained at
3.2 examples/s. Because the latency rows already fail, the memory fallback was not run.

### Stage E: encoder

Not run: feasibility failed.

### Release gate: mmBERT-base

Not run: feasibility failed. `krite-0.15b-v1` stays the best model.

### Release gate: revised target

`krite-0.15b-v1` on the revised target, from the same measurements as "Release gate" above. Raw output:
[`release.json`](../benchmarks/results/arch/release.json).

| Gate | Limit | `krite-0.15b-v1` | Pass |
|---|---|---|---|
| accuracy (macro, choice and noul) | ≥ 0.795 | 0.783 | **no** |
| QWK (macro, score) | ≥ 0.357 | 0.273 | **no** |
| every other row | as above | unchanged | yes |

**Verdict: not released; shipped as a pre-release with the gap stated.** On the revised target,
accuracy is 1.1 points short and QWK 8.4 points short. Against cbjev, Krite trades accuracy for speed:
on this machine, warm latency is 7.8 ms vs 387 ms and throughput is 212 vs 15.5 decisions/s (30 questions).

Known limitations of `krite-0.15b-v1`. These are the suites furthest below the best baseline:

| Suite | `krite-0.15b-v1` | Best ≤ 0.45B baseline |
|---|---|---|
| agnews-choice (topic, English) | 0.642 | 0.945 (Laya) |
| sst5-score (5-level sentiment, QWK) | 0.509 | 0.748 (cbjev) |
| boolq-noul (passage yes/no) | 0.688 | 0.845 (Laya) |
| amazon-score (6 languages, QWK) | −0.05 to 0.14 | 0.04 to 0.10 (Laya) |
