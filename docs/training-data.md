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
| accuracy | shortfall 0: every suite at or above its target |
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

| Suite | Metric | krite | arch-late8 | arch-late8-s14 | arch-late8-broad | arch-late8-broad-s14 |
|---|---|---|---|---|---|---|
| agnews-choice | accuracy / macro_f1 | 0.625 / 0.621 | 0.625 / 0.621 | 0.412 / 0.379 | 0.642 / 0.597 | 0.682 / 0.625 |
| banking77-choice | accuracy / macro_f1 | 0.902 / 0.900 | 0.902 / 0.900 | 0.910 / 0.907 | 0.880 / 0.880 | 0.912 / 0.909 |
| boolq-noul | accuracy / auroc | 0.630 / 0.678 | 0.630 / 0.678 | 0.605 / 0.637 | 0.688 / 0.787 | 0.675 / 0.739 |
| sst5-score | mae / qwk | 0.881 / 0.464 | 0.881 / 0.464 | 0.919 / 0.439 | 0.875 / 0.509 | 0.833 / 0.543 |
| massive-choice-en | accuracy / macro_f1 | 0.907 / 0.880 | 0.907 / 0.880 | 0.917 / 0.883 | 0.920 / 0.886 | 0.927 / 0.894 |
| massive-choice-ko | accuracy / macro_f1 | 0.887 / 0.847 | 0.887 / 0.847 | 0.882 / 0.850 | 0.890 / 0.884 | 0.880 / 0.836 |
| massive-choice-ja | accuracy / macro_f1 | 0.907 / 0.907 | 0.907 / 0.907 | 0.892 / 0.877 | 0.920 / 0.914 | 0.943 / 0.910 |
| massive-choice-zh-CN | accuracy / macro_f1 | 0.920 / 0.909 | 0.920 / 0.909 | 0.932 / 0.932 | 0.915 / 0.887 | 0.935 / 0.933 |
| massive-choice-de | accuracy / macro_f1 | 0.902 / 0.872 | 0.902 / 0.872 | 0.895 / 0.844 | 0.892 / 0.863 | 0.892 / 0.849 |
| massive-choice-es | accuracy / macro_f1 | 0.912 / 0.882 | 0.912 / 0.882 | 0.907 / 0.897 | 0.902 / 0.886 | 0.895 / 0.885 |
| massive-choice-fr | accuracy / macro_f1 | 0.935 / 0.921 | 0.935 / 0.921 | 0.912 / 0.897 | 0.927 / 0.901 | 0.930 / 0.919 |
| massive-choice-ar | accuracy / macro_f1 | 0.855 / 0.797 | 0.855 / 0.797 | 0.875 / 0.798 | 0.840 / 0.749 | 0.855 / 0.765 |
| massive-choice-hi | accuracy / macro_f1 | 0.860 / 0.731 | 0.860 / 0.731 | 0.865 / 0.759 | 0.863 / 0.741 | 0.848 / 0.750 |
| massive-choice-ru | accuracy / macro_f1 | 0.902 / 0.852 | 0.902 / 0.852 | 0.887 / 0.811 | 0.887 / 0.849 | 0.920 / 0.886 |
| xnli-noul-en | accuracy / auroc | 0.755 / 0.831 | 0.755 / 0.831 | 0.728 / 0.823 | 0.910 / 0.954 | 0.897 / 0.954 |
| xnli-noul-de | accuracy / auroc | 0.730 / 0.826 | 0.730 / 0.826 | 0.723 / 0.806 | 0.812 / 0.896 | 0.818 / 0.897 |
| xnli-noul-es | accuracy / auroc | 0.738 / 0.823 | 0.738 / 0.823 | 0.743 / 0.814 | 0.845 / 0.917 | 0.840 / 0.923 |
| xnli-noul-fr | accuracy / auroc | 0.765 / 0.827 | 0.765 / 0.827 | 0.757 / 0.822 | 0.823 / 0.925 | 0.823 / 0.924 |
| xnli-noul-ar | accuracy / auroc | 0.728 / 0.799 | 0.728 / 0.799 | 0.713 / 0.793 | 0.755 / 0.856 | 0.782 / 0.869 |
| xnli-noul-hi | accuracy / auroc | 0.642 / 0.739 | 0.642 / 0.739 | 0.690 / 0.770 | 0.715 / 0.848 | 0.718 / 0.843 |
| xnli-noul-ru | accuracy / auroc | 0.690 / 0.771 | 0.690 / 0.771 | 0.688 / 0.763 | 0.805 / 0.905 | 0.810 / 0.905 |
| xnli-noul-zh | accuracy / auroc | 0.703 / 0.780 | 0.703 / 0.780 | 0.695 / 0.774 | 0.828 / 0.910 | 0.835 / 0.895 |
| amazon-score-en | mae / qwk | 1.112 / 0.390 | 1.112 / 0.390 | 1.167 / 0.150 | 1.139 / 0.142 | 1.095 / 0.357 |
| amazon-score-de | mae / qwk | 1.130 / 0.308 | 1.130 / 0.308 | 1.201 / 0.001 | 1.182 / 0.039 | 1.136 / 0.106 |
| amazon-score-es | mae / qwk | 1.195 / 0.248 | 1.195 / 0.248 | 1.266 / 0.047 | 1.223 / 0.022 | 1.188 / 0.164 |
| amazon-score-fr | mae / qwk | 1.177 / 0.202 | 1.177 / 0.202 | 1.230 / -0.008 | 1.216 / 0.051 | 1.162 / 0.160 |
| amazon-score-ja | mae / qwk | 1.191 / 0.090 | 1.191 / 0.090 | 1.197 / -0.048 | 1.240 / 0.015 | 1.170 / 0.035 |
| amazon-score-zh | mae / qwk | 1.150 / 0.142 | 1.150 / 0.142 | 1.195 / -0.059 | 1.225 / -0.053 | 1.140 / 0.052 |

### Calibration: ECE raw → temperature-scaled

Evaluation half of each suite. One temperature per engine model and calibration bucket, fit on the pooled other halves of every suite in that bucket (benchmark-spec §9); the classifier has one model, and so one calibrator, per dataset.

| Suite | krite | arch-late8 | arch-late8-s14 | arch-late8-broad | arch-late8-broad-s14 |
|---|---|---|---|---|---|
| agnews-choice | 0.152 → 0.171 | 0.152 → 0.171 | 0.256 → 0.105 | 0.207 → 0.085 | 0.183 → 0.085 |
| banking77-choice | 0.046 → 0.048 | 0.046 → 0.048 | 0.052 → 0.047 | 0.073 → 0.048 | 0.045 → 0.037 |
| boolq-noul | 0.070 → 0.054 | 0.070 → 0.054 | 0.104 → 0.093 | 0.075 → 0.107 | 0.063 → 0.039 |
| sst5-score | 0.176 → 0.044 | 0.176 → 0.044 | 0.109 → 0.056 | 0.176 → 0.056 | 0.140 → 0.075 |
| massive-choice-en | 0.056 → 0.031 | 0.056 → 0.031 | 0.063 → 0.048 | 0.061 → 0.035 | 0.052 → 0.048 |
| massive-choice-ko | 0.071 → 0.052 | 0.071 → 0.052 | 0.056 → 0.046 | 0.067 → 0.045 | 0.041 → 0.055 |
| massive-choice-ja | 0.043 → 0.048 | 0.043 → 0.048 | 0.058 → 0.038 | 0.032 → 0.042 | 0.035 → 0.059 |
| massive-choice-zh-CN | 0.042 → 0.040 | 0.042 → 0.040 | 0.032 → 0.037 | 0.041 → 0.053 | 0.052 → 0.053 |
| massive-choice-de | 0.049 → 0.045 | 0.049 → 0.045 | 0.051 → 0.042 | 0.075 → 0.070 | 0.056 → 0.055 |
| massive-choice-es | 0.064 → 0.039 | 0.064 → 0.039 | 0.053 → 0.034 | 0.049 → 0.032 | 0.063 → 0.035 |
| massive-choice-fr | 0.039 → 0.037 | 0.039 → 0.037 | 0.044 → 0.040 | 0.039 → 0.039 | 0.040 → 0.034 |
| massive-choice-ar | 0.078 → 0.024 | 0.078 → 0.024 | 0.046 → 0.037 | 0.097 → 0.082 | 0.063 → 0.054 |
| massive-choice-hi | 0.080 → 0.054 | 0.080 → 0.054 | 0.041 → 0.065 | 0.046 → 0.043 | 0.047 → 0.031 |
| massive-choice-ru | 0.069 → 0.036 | 0.069 → 0.036 | 0.064 → 0.070 | 0.083 → 0.060 | 0.052 → 0.052 |
| xnli-noul-en | 0.166 → 0.090 | 0.166 → 0.090 | 0.144 → 0.070 | 0.058 → 0.075 | 0.052 → 0.062 |
| xnli-noul-de | 0.119 → 0.054 | 0.119 → 0.054 | 0.120 → 0.058 | 0.137 → 0.104 | 0.106 → 0.073 |
| xnli-noul-es | 0.146 → 0.045 | 0.146 → 0.045 | 0.125 → 0.084 | 0.103 → 0.066 | 0.069 → 0.051 |
| xnli-noul-fr | 0.108 → 0.056 | 0.108 → 0.056 | 0.115 → 0.066 | 0.080 → 0.046 | 0.076 → 0.049 |
| xnli-noul-ar | 0.108 → 0.070 | 0.108 → 0.070 | 0.142 → 0.122 | 0.105 → 0.052 | 0.061 → 0.064 |
| xnli-noul-hi | 0.166 → 0.098 | 0.166 → 0.098 | 0.084 → 0.068 | 0.099 → 0.062 | 0.098 → 0.052 |
| xnli-noul-ru | 0.179 → 0.087 | 0.179 → 0.087 | 0.176 → 0.108 | 0.083 → 0.038 | 0.069 → 0.051 |
| xnli-noul-zh | 0.147 → 0.087 | 0.147 → 0.087 | 0.105 → 0.057 | 0.096 → 0.076 | 0.103 → 0.091 |
| amazon-score-en | 0.103 → 0.145 | 0.103 → 0.145 | 0.034 → 0.055 | 0.083 → 0.056 | 0.068 → 0.089 |
| amazon-score-de | 0.046 → 0.072 | 0.046 → 0.072 | 0.057 → 0.013 | 0.116 → 0.047 | 0.046 → 0.031 |
| amazon-score-es | 0.026 → 0.039 | 0.026 → 0.039 | 0.056 → 0.013 | 0.096 → 0.021 | 0.035 → 0.018 |
| amazon-score-fr | 0.033 → 0.021 | 0.033 → 0.021 | 0.050 → 0.013 | 0.087 → 0.074 | 0.041 → 0.037 |
| amazon-score-ja | 0.029 → 0.023 | 0.029 → 0.023 | 0.059 → 0.017 | 0.104 → 0.025 | 0.032 → 0.032 |
| amazon-score-zh | 0.034 → 0.010 | 0.034 → 0.010 | 0.062 → 0.021 | 0.113 → 0.059 | 0.045 → 0.036 |
| **mean over suites** | 0.087 → 0.058 | 0.087 → 0.058 | 0.084 → 0.054 | 0.089 → 0.057 | 0.066 → 0.052 |

### Calibration: NLL and Brier, raw → scaled

| Suite | krite | arch-late8 | arch-late8-s14 | arch-late8-broad | arch-late8-broad-s14 |
|---|---|---|---|---|---|
| agnews-choice | 0.988 → 0.979 · B 0.548 → 0.552 | 0.988 → 0.979 · B 0.548 → 0.552 | 1.285 → 1.168 · B 0.740 → 0.664 | 1.317 → 0.923 · B 0.558 → 0.494 | 1.081 → 0.873 · B 0.515 → 0.475 |
| banking77-choice | 0.365 → 0.324 · B 0.153 → 0.154 | 0.365 → 0.324 · B 0.153 → 0.154 | 0.291 → 0.283 · B 0.125 → 0.127 | 0.404 → 0.379 · B 0.196 → 0.190 | 0.277 → 0.272 · B 0.126 → 0.128 |
| boolq-noul | 0.621 → 0.631 · B 0.435 → 0.441 | 0.621 → 0.631 · B 0.435 → 0.441 | 0.653 → 0.660 · B 0.462 → 0.467 | 0.549 → 0.566 · B 0.372 → 0.382 | 0.593 → 0.592 · B 0.407 → 0.407 |
| sst5-score | 1.580 → 1.507 · B 0.796 → 0.759 | 1.580 → 1.507 · B 0.796 → 0.759 | 1.480 → 1.500 · B 0.762 → 0.757 | 1.553 → 1.443 · B 0.775 → 0.735 | 1.469 → 1.428 · B 0.769 → 0.743 |
| massive-choice-en | 0.320 → 0.269 · B 0.150 → 0.142 | 0.320 → 0.269 · B 0.150 → 0.142 | 0.294 → 0.273 · B 0.154 → 0.150 | 0.288 → 0.261 · B 0.134 → 0.130 | 0.220 → 0.212 · B 0.117 → 0.116 |
| massive-choice-ko | 0.405 → 0.329 · B 0.177 → 0.162 | 0.405 → 0.329 · B 0.177 → 0.162 | 0.335 → 0.317 · B 0.150 → 0.148 | 0.380 → 0.352 · B 0.169 → 0.167 | 0.393 → 0.377 · B 0.163 → 0.163 |
| massive-choice-ja | 0.228 → 0.215 · B 0.105 → 0.111 | 0.228 → 0.215 · B 0.105 → 0.111 | 0.267 → 0.254 · B 0.140 → 0.136 | 0.235 → 0.224 · B 0.109 → 0.111 | 0.173 → 0.182 · B 0.081 → 0.086 |
| massive-choice-zh-CN | 0.236 → 0.226 · B 0.112 → 0.112 | 0.236 → 0.226 · B 0.112 → 0.112 | 0.191 → 0.199 · B 0.094 → 0.097 | 0.219 → 0.225 · B 0.107 → 0.109 | 0.212 → 0.217 · B 0.098 → 0.104 |
| massive-choice-de | 0.372 → 0.331 · B 0.168 → 0.166 | 0.372 → 0.331 · B 0.168 → 0.166 | 0.353 → 0.343 · B 0.178 → 0.177 | 0.354 → 0.339 · B 0.184 → 0.180 | 0.277 → 0.283 · B 0.159 → 0.157 |
| massive-choice-es | 0.405 → 0.326 · B 0.175 → 0.165 | 0.405 → 0.326 · B 0.175 → 0.165 | 0.352 → 0.326 · B 0.163 → 0.160 | 0.308 → 0.291 · B 0.158 → 0.154 | 0.296 → 0.281 · B 0.136 → 0.135 |
| massive-choice-fr | 0.271 → 0.246 · B 0.116 → 0.118 | 0.271 → 0.246 · B 0.116 → 0.118 | 0.265 → 0.254 · B 0.140 → 0.137 | 0.293 → 0.272 · B 0.129 → 0.129 | 0.222 → 0.221 · B 0.123 → 0.121 |
| massive-choice-ar | 0.563 → 0.471 · B 0.234 → 0.223 | 0.563 → 0.471 · B 0.234 → 0.223 | 0.392 → 0.377 · B 0.188 → 0.183 | 0.541 → 0.494 · B 0.261 → 0.249 | 0.522 → 0.506 · B 0.238 → 0.235 |
| massive-choice-hi | 0.477 → 0.403 · B 0.226 → 0.208 | 0.477 → 0.403 · B 0.226 → 0.208 | 0.361 → 0.351 · B 0.184 → 0.179 | 0.396 → 0.385 · B 0.203 → 0.199 | 0.479 → 0.460 · B 0.204 → 0.203 |
| massive-choice-ru | 0.394 → 0.327 · B 0.171 → 0.158 | 0.394 → 0.327 · B 0.171 → 0.158 | 0.366 → 0.336 · B 0.178 → 0.168 | 0.380 → 0.346 · B 0.193 → 0.182 | 0.290 → 0.279 · B 0.139 → 0.137 |
| xnli-noul-en | 0.653 → 0.539 · B 0.390 → 0.357 | 0.653 → 0.539 · B 0.390 → 0.357 | 0.609 → 0.544 · B 0.399 → 0.367 | 0.292 → 0.291 · B 0.155 → 0.162 | 0.328 → 0.321 · B 0.161 → 0.169 |
| xnli-noul-de | 0.600 → 0.534 · B 0.378 → 0.354 | 0.600 → 0.534 · B 0.378 → 0.354 | 0.559 → 0.526 · B 0.369 → 0.351 | 0.605 → 0.518 · B 0.350 → 0.331 | 0.574 → 0.509 · B 0.330 → 0.315 |
| xnli-noul-es | 0.630 → 0.543 · B 0.398 → 0.364 | 0.630 → 0.543 · B 0.398 → 0.364 | 0.600 → 0.549 · B 0.396 → 0.370 | 0.543 → 0.474 · B 0.322 → 0.304 | 0.467 → 0.429 · B 0.279 → 0.271 |
| xnli-noul-fr | 0.562 → 0.522 · B 0.362 → 0.347 | 0.562 → 0.522 · B 0.362 → 0.347 | 0.623 → 0.568 · B 0.408 → 0.385 | 0.448 → 0.412 · B 0.284 → 0.269 | 0.427 → 0.405 · B 0.277 → 0.267 |
| xnli-noul-ar | 0.633 → 0.580 · B 0.413 → 0.393 | 0.633 → 0.580 · B 0.413 → 0.393 | 0.670 → 0.611 · B 0.454 → 0.424 | 0.589 → 0.527 · B 0.362 → 0.345 | 0.477 → 0.464 · B 0.284 → 0.286 |
| xnli-noul-hi | 0.644 → 0.588 · B 0.446 → 0.408 | 0.644 → 0.588 · B 0.446 → 0.408 | 0.625 → 0.606 · B 0.432 → 0.419 | 0.575 → 0.533 · B 0.379 → 0.359 | 0.581 → 0.550 · B 0.378 → 0.368 |
| xnli-noul-ru | 0.764 → 0.639 · B 0.491 → 0.442 | 0.764 → 0.639 · B 0.491 → 0.442 | 0.753 → 0.659 · B 0.513 → 0.464 | 0.476 → 0.446 · B 0.299 → 0.290 | 0.463 → 0.436 · B 0.294 → 0.284 |
| xnli-noul-zh | 0.714 → 0.598 · B 0.432 → 0.403 | 0.714 → 0.598 · B 0.432 → 0.403 | 0.628 → 0.595 · B 0.424 → 0.407 | 0.459 → 0.421 · B 0.278 → 0.267 | 0.508 → 0.465 · B 0.269 → 0.271 |
| amazon-score-en | 1.542 → 1.570 · B 0.767 → 0.783 | 1.542 → 1.570 · B 0.767 → 0.783 | 1.586 → 1.578 · B 0.786 → 0.788 | 1.595 → 1.566 · B 0.791 → 0.780 | 1.495 → 1.513 · B 0.749 → 0.757 |
| amazon-score-de | 1.578 → 1.587 · B 0.786 → 0.791 | 1.578 → 1.587 · B 0.786 → 0.791 | 1.701 → 1.618 · B 0.819 → 0.802 | 1.660 → 1.610 · B 0.827 → 0.801 | 1.594 → 1.592 · B 0.794 → 0.793 |
| amazon-score-es | 1.581 → 1.588 · B 0.788 → 0.791 | 1.581 → 1.588 · B 0.788 → 0.791 | 1.730 → 1.630 · B 0.826 → 0.806 | 1.627 → 1.596 · B 0.817 → 0.797 | 1.598 → 1.595 · B 0.796 → 0.795 |
| amazon-score-fr | 1.605 → 1.598 · B 0.798 → 0.796 | 1.605 → 1.598 · B 0.798 → 0.796 | 1.729 → 1.630 · B 0.827 → 0.807 | 1.636 → 1.600 · B 0.818 → 0.797 | 1.607 → 1.601 · B 0.801 → 0.798 |
| amazon-score-ja | 1.641 → 1.614 · B 0.813 → 0.802 | 1.641 → 1.614 · B 0.813 → 0.802 | 1.717 → 1.624 · B 0.825 → 0.805 | 1.664 → 1.609 · B 0.826 → 0.800 | 1.611 → 1.605 · B 0.802 → 0.799 |
| amazon-score-zh | 1.633 → 1.613 · B 0.809 → 0.802 | 1.633 → 1.613 · B 0.809 → 0.802 | 1.678 → 1.611 · B 0.816 → 0.801 | 1.664 → 1.614 · B 0.825 → 0.802 | 1.581 → 1.584 · B 0.788 → 0.789 |

### Option-order invariance (100 cases per suite)

| Engine | Suite | Full-perm flip | Reverse flip | Max prob dev | Errors |
|---|---|---|---|---|---|
| krite | agnews-choice | 0.000 | 0.000 | 0.0000 | 0% |
| krite | banking77-choice | 0.000 | 0.000 | 0.0000 | 0% |
| krite | massive-choice-en | 0.000 | 0.000 | 0.0000 | 0% |
| krite | massive-choice-ko | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8 | agnews-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8 | banking77-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8 | massive-choice-en | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8 | massive-choice-ko | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-s14 | agnews-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-s14 | banking77-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-s14 | massive-choice-en | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-s14 | massive-choice-ko | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-broad | agnews-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-broad | banking77-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-broad | massive-choice-en | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-broad | massive-choice-ko | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-broad-s14 | agnews-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-broad-s14 | banking77-choice | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-broad-s14 | massive-choice-en | 0.000 | 0.000 | 0.0000 | 0% |
| arch-late8-broad-s14 | massive-choice-ko | 0.000 | 0.000 | 0.0000 | 0% |

### Question interference (agnews-choice, 100 cases; alone vs. with fillers)

| Engine | Fillers | Argmax change | Max prob dev |
|---|---|---|---|
| krite | 3 | 0.000 | 0.0000 |
| krite | 15 | 0.000 | 0.0000 |
| arch-late8 | 3 | 0.000 | 0.0000 |
| arch-late8 | 15 | 0.000 | 0.0000 |
| arch-late8-s14 | 3 | 0.000 | 0.0000 |
| arch-late8-s14 | 15 | 0.000 | 0.0000 |
| arch-late8-broad | 3 | 0.000 | 0.0000 |
| arch-late8-broad | 15 | 0.000 | 0.0000 |
| arch-late8-broad-s14 | 3 | 0.000 | 0.0000 |
| arch-late8-broad-s14 | 15 | 0.000 | 0.0000 |

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
| arch-late8-broad | burst | startup | ping probe | http | 1 | 12193.3 | 12193.3 | 12193.3 | 0.1 | 0% | AC |  |
| arch-late8-broad-s14 | burst | startup | ping probe | http | 1 | 11146.8 | 11146.8 | 11146.8 | 0.1 | 0% | AC |  |
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
| arch-late8-broad | arch-late8-broad | 3.45 | 1.32 |
| arch-late8-broad-s14 | arch-late8-broad-s14 | 3.47 | 1.34 |

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

Not run yet.

### Stage C: epochs

Not run yet.

### Release gate

Not run yet.
