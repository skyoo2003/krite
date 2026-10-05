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
| sib200 | `Davlan/sib200` @ `38977a66` (10 suite languages, 7 topics) | choice (noul) | TBD | TBD | CC-BY-SA-4.0 | broad |
| mnli | `nyu-mll/multi_nli` @ `da70db2a` (neutral dropped for noul) | noul (3-way choice) | TBD | TBD | OANC, CC-BY-3.0, CC-BY-SA-3.0 (per genre) | broad |
| pawsx | `google-research-datasets/paws-x` @ `4cd8187c` (de en es fr ja ko zh) | noul | TBD | TBD | PAWS-X terms (free use for any purpose) | broad |
| goemotions | `google-research-datasets/go_emotions` @ `add49224` (`simplified`) | score, K = 3 | TBD | TBD | Apache-2.0 | broad |

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
<!-- GENERATED:END -->

### Stage A: broad mixture

Not run yet.

### Stage B: loss

Not run yet.

### Stage C: epochs

Not run yet.

### Release gate

Not run yet.
