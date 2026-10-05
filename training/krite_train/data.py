"""Training examples from permissively licensed train splits only (docs/training-data.md).

Two mixtures: `study`, the architecture study's frozen mixture, and `broad`, which adds sources,
question-type views, and augmentations for the release recipe. Every source is pinned to a Hugging
Face revision. Dataset text stays in the HF cache and in memory; only counts and hashes are written
out (train_meta.json). The held-out evaluation datasets (agnews, xnli, sst5, amazon) are never read
here, and every row whose state appears in a built suite is dropped (leakage guard), so the in-domain
suites (banking77, massive, boolq) stay unseen too.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from krite_bench import data as kb

MAX_K = 12  # most candidates a training choice question gets
SIB_LANGS = {
    "en": "eng_Latn",
    "ko": "kor_Hang",
    "ja": "jpn_Jpan",
    "zh-CN": "zho_Hans",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "ru": "rus_Cyrl",
}
PAWSX_LANGS = ["en", "de", "es", "fr", "ja", "ko", "zh"]


@dataclass(frozen=True)
class Source:
    repo: str
    revision: str
    files: tuple[str, ...]
    license: str
    type: str
    cap: int


SOURCES = {
    "banking77": Source(
        kb.DATASETS["banking77"].repo,
        kb.DATASETS["banking77"].revision,
        ("data/train-00000-of-00001.parquet",),
        "CC-BY-4.0",
        "choice",
        4000,
    ),
    "clinc": Source(
        "clinc/clinc_oos",
        "155b9c710419136e17307b80d0a13e68cd46b4ec",
        ("plus/train-00000-of-00001.parquet",),
        "CC-BY-3.0",
        "choice",
        4000,
    ),
    "massive": Source(
        kb.DATASETS["massive"].repo,
        kb.DATASETS["massive"].revision,
        tuple(f"train/{lg}.json.gz" for lg in kb.MASSIVE_LANGS),
        "CC-BY-4.0",
        "choice",
        8000,  # split evenly across languages
    ),
    "dbpedia": Source(
        "fancyzhx/dbpedia_14",
        "9abd46cf7fc8b4c64290f26993c540b92aa145ac",
        ("dbpedia_14/train-00000-of-00001.parquet",),
        "CC-BY-SA-3.0",
        "choice",
        4000,
    ),
    "boolq": Source(
        kb.DATASETS["boolq"].repo,
        kb.DATASETS["boolq"].revision,
        ("data/train-00000-of-00001.parquet",),
        "CC-BY-SA-3.0",
        "noul",
        4000,
    ),
    "snli": Source(
        "stanfordnlp/snli",
        "cdb5c3d5eed6ead6e5a341c8e56e669bb666725b",
        ("plain_text/train-00000-of-00001.parquet",),
        "CC-BY-SA-4.0",
        "noul",
        4000,
    ),
    "civil": Source(
        "google/civil_comments",
        "f2970eb3a55777454c94069077cc8d9b5866312d",
        ("data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"),
        "CC0-1.0",
        "score",
        4000,  # split evenly across levels
    ),
    # Broad mixture only (docs/training-data.md). Appended so the study sources keep their rng index.
    "sib200": Source(
        "Davlan/sib200",
        "38977a667f6fc264d5c26ec57a01e16db040b358",
        tuple(f"data/{c}/train.tsv" for c in SIB_LANGS.values()),
        "CC-BY-SA-4.0",
        "choice",
        7000,  # split evenly across languages
    ),
    "mnli": Source(
        "nyu-mll/multi_nli",
        "da70db2af9d09693783c3320c4249840212ee221",
        ("data/train-00000-of-00001.parquet",),
        "OANC, CC-BY-3.0, CC-BY-SA-3.0 (per genre)",
        "noul",
        6000,
    ),
    "pawsx": Source(
        "google-research-datasets/paws-x",
        "4cd8187c404bda33cb1f62b49b001115862acf37",
        tuple(f"{lg}/train-00000-of-00001.parquet" for lg in PAWSX_LANGS),
        "PAWS-X terms (free use for any purpose)",
        "noul",
        7000,  # split evenly across languages
    ),
    "goemotions": Source(
        "google-research-datasets/go_emotions",
        "add492243ff905527e67aeb8b80c082af02207c3",
        ("simplified/train-00000-of-00001.parquet",),
        "Apache-2.0",
        "score",
        6000,  # split evenly across levels
    ),
}
MIXTURES = {
    "study": ("banking77", "clinc", "massive", "dbpedia", "boolq", "snli", "civil"),
    "broad": tuple(SOURCES),
}
CAPS = {"broad": {"boolq": 8000}}  # per-mixture cap overrides
LANG_GROUPS = {"massive": kb.MASSIVE_LANGS, "sib200": list(SIB_LANGS), "pawsx": PAWSX_LANGS}

CIVIL_LEVELS = ["not toxic", "slightly toxic", "moderately toxic", "very toxic", "extremely toxic"]
CIVIL_EDGES = [0.2, 0.4, 0.6, 0.8]
SENTIMENT_LEVELS = [("negative", "neutral", "positive"), ("unhappy", "neutral", "happy")]
LEVEL_GROUPS = {"civil": len(CIVIL_LEVELS), "goemotions": 3}
# GoEmotions authors' sentiment grouping; ambiguous emotions (confusion, curiosity, realization, surprise) are absent.
POLARITY = {
    **dict.fromkeys(
        "anger annoyance disappointment disapproval disgust embarrassment "
        "fear grief nervousness remorse sadness".split(),
        0,
    ),
    "neutral": 1,
    **dict.fromkeys(
        "admiration amusement approval caring desire excitement gratitude joy love optimism pride relief".split(), 2
    ),
}

# Never equal to an evaluation suite's instructions (tested), so every suite stays zero-shot in wording.
TEMPLATES = {
    "banking77": [
        "What does this bank customer want to do?",
        "Which banking request is this message about?",
        "Classify the customer's banking intent.",
    ],
    "clinc": [
        "What is the user asking the assistant to do?",
        "Which intent does this request belong to?",
        "Pick the intent of this user utterance.",
    ],
    "massive": [
        "What does the user want the voice assistant to do?",
        "Which command intent fits this utterance?",
        "Classify the intent of this spoken request.",
    ],
    "dbpedia": [
        "What kind of entity does this text describe?",
        "Which category fits this encyclopedia entry?",
        "Classify the subject of this article.",
    ],
    "boolq": [
        "Does the passage answer the question with yes?",
        "According to the passage, is the answer yes?",
        "Is the answer to the question true, given the passage?",
    ],
    "snli": [
        "Is the hypothesis true given the premise?",
        "Does the hypothesis follow from the premise?",
        "Given the premise, must the hypothesis hold?",
    ],
    "civil": [
        "How toxic is this comment?",
        "Rate the toxicity of this text.",
        "How offensive or hostile is this comment?",
    ],
    "sib200": [
        "What is the topic of this sentence?",
        "Which subject area does this text belong to?",
        "Classify the subject of this sentence.",
    ],
    "mnli": [
        "Is the hypothesis true if the premise is?",
        "Does the premise support the hypothesis?",
        "Given the premise, is the hypothesis correct?",
    ],
    "pawsx": [
        "Do the two sentences mean the same thing?",
        "Is the second sentence a paraphrase of the first?",
        "Do both sentences say the same?",
    ],
    "goemotions": [
        "How positive or negative is this comment?",
        "Rate the sentiment of this comment.",
        "What is the emotional tone of this text?",
    ],
}

# Broad mixture: extra question types per source, (type, count).
VIEWS = {
    "banking77": [("noul", 1000)],
    "clinc": [("noul", 1000)],
    "massive": [("noul", 2000)],
    "dbpedia": [("noul", 1000)],
    "sib200": [("noul", 1000)],
    "civil": [("noul", 1000)],
    "snli": [("choice", 2000)],
    "mnli": [("choice", 3000)],
}
NLI_CRITERIA = [
    ("entailment", "the hypothesis follows"),
    ("neutral", "neither follows nor contradicts"),
    ("contradiction", "the hypothesis is false"),
]
VIEW_TEMPLATES = {  # "{label}" is the asked criterion
    "intent": ["Is the user's intent {label}?", "Does this request ask for {label}?", "Is this message about {label}?"],
    "topic": [
        "Is this text about {label}?",
        "Does the topic {label} fit this text?",
        "Is {label} the subject of this text?",
    ],
    "toxic": [
        "Is this comment toxic?",
        "Does this text contain toxic language?",
        "Is this comment hostile or offensive?",
    ],
    "nli": [
        "How does the hypothesis relate to the premise?",
        "What is the relation between the premise and the hypothesis?",
        "Classify the premise-hypothesis pair.",
    ],
}
DOMAINS = {"banking77": "intent", "clinc": "intent", "massive": "intent", "dbpedia": "topic", "sib200": "topic"}
OTHER_DOMAIN = {"intent": "topic", "topic": "intent"}

# Broad mixture augmentations.
STATE_FORMAT_P, FOREIGN_P = 0.3, 0.2
DICT_STATE_SOURCES = {"boolq", "snli", "mnli", "pawsx"}
STATE_KEYS = ["text", "message", "input", "content", "observation", "utterance"]
KEY_SYNONYMS = {
    "passage": ["context", "document"],
    "question": ["query", "user_question"],
    "premise": ["statement", "context"],
    "hypothesis": ["claim"],
    "sentence1": ["text_a", "first"],
    "sentence2": ["text_b", "second"],
}
CHANNELS = ["chat", "email", "voice", "api"]
EXTRA_FIELDS = {
    "turn": lambda rng: int(rng.integers(1, 21)),
    "channel": lambda rng: CHANNELS[int(rng.integers(len(CHANNELS)))],
    "step": lambda rng: int(rng.integers(1, 51)),
}


def canonical_state(state: object) -> str:
    """Must match benchmarks/baselines/shims/common.py canonical_state."""
    if isinstance(state, str):
        return state
    return json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def label_names(path: str, column: str) -> list[str]:
    """ClassLabel names stored by `datasets` in the parquet schema metadata."""
    meta = json.loads(pq.read_schema(path).metadata[b"huggingface"])
    return meta["info"]["features"][column]["names"]


def _paths(src: Source) -> list[str]:
    return [kb.hf_hub_download(src.repo, f, repo_type="dataset", revision=src.revision) for f in src.files]


def read_tsv(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))


def polarity(names: list[str]) -> int | None:
    """GoEmotions labels → 0 negative / 1 neutral / 2 positive; None when ambiguous or mixed."""
    levels = {POLARITY.get(n) for n in names}
    return levels.pop() if len(levels) == 1 and None not in levels else None


def nli_records(name: str) -> list[tuple[str, dict, int]]:
    """[(lang, {premise, hypothesis}, label)] with label 0 entailment, 1 neutral, 2 contradiction."""
    path = _paths(SOURCES[name])[0]
    if name == "snli":
        rows = kb._read_rows(path)
    else:
        rows = pq.read_table(path, columns=["premise", "hypothesis", "label"]).to_pylist()
    return [
        ("en", {"premise": r["premise"], "hypothesis": r["hypothesis"]}, r["label"])
        for r in rows
        if r["label"] in (0, 1, 2)
    ]


def records(name: str) -> list[tuple[str, object, object]]:
    """[(lang, state, gold)] for one source; gold is a label name, a bool (noul), or a level index (score)."""
    src = SOURCES[name]
    paths = _paths(src)
    if name == "banking77":
        return [("en", r["text"], kb._label_name(r["label_text"])) for r in kb._read_rows(paths[0])]
    if name == "clinc":
        names = label_names(paths[0], "intent")
        rows = kb._read_rows(paths[0])
        return [("en", r["text"], kb._label_name(names[r["intent"]])) for r in rows if names[r["intent"]] != "oos"]
    if name == "massive":
        return [(r["lang"], r["text"], kb._label_name(r["label"])) for p in paths for r in kb._read_rows(p)]
    if name == "dbpedia":
        names = label_names(paths[0], "label")
        return [("en", r["content"].strip(), names[r["label"]]) for r in kb._read_rows(paths[0])]
    if name == "boolq":
        rows = kb._read_rows(paths[0])
        return [("en", {"passage": r["passage"], "question": r["question"]}, bool(r["answer"])) for r in rows]
    if name in ("snli", "mnli"):  # neutral dropped: noul asks entailment vs. contradiction
        return [(lg, state, label == 0) for lg, state, label in nli_records(name) if label in (0, 2)]
    if name == "civil":
        out = []
        for p in paths:
            t = pq.read_table(p, columns=["text", "toxicity"])
            levels = np.digitize(t.column("toxicity").to_numpy(), CIVIL_EDGES)
            out += [("en", text, int(lv)) for text, lv in zip(t.column("text").to_pylist(), levels, strict=True)]
        return out
    if name == "sib200":
        return [(lg, r["text"], r["category"]) for lg, p in zip(SIB_LANGS, paths, strict=True) for r in read_tsv(p)]
    if name == "pawsx":
        out = []
        for lg, p in zip(PAWSX_LANGS, paths, strict=True):
            for r in pq.read_table(p, columns=["sentence1", "sentence2", "label"]).to_pylist():
                if r["sentence1"] and r["sentence2"]:
                    out.append((lg, {"sentence1": r["sentence1"], "sentence2": r["sentence2"]}, r["label"] == 1))
        return out
    if name == "goemotions":
        meta = json.loads(pq.read_schema(paths[0]).metadata[b"huggingface"])
        names = meta["info"]["features"]["labels"]["feature"]["names"]
        rows = pq.read_table(paths[0], columns=["text", "labels"]).to_pylist()
        levels = [(r["text"], polarity([names[i] for i in r["labels"]])) for r in rows]
        return [("en", text, lv) for text, lv in levels if lv is not None]
    raise ValueError(f"unknown source {name}")


def sample_choice(gold: str, labels: list[str], rng: np.random.Generator) -> tuple[list[str], int]:
    """Gold plus seeded distractors, shuffled; K uniform in 2..min(MAX_K, len(labels))."""
    k = int(rng.integers(2, min(MAX_K, len(labels)) + 1))
    others = [x for x in labels if x != gold]
    names = [gold, *rng.choice(others, size=k - 1, replace=False).tolist()]
    names = [names[int(j)] for j in rng.permutation(k)]
    return names, names.index(gold)


def make_example(name: str, lang: str, state: object, gold, labels: list[str], rng: np.random.Generator) -> dict:
    qtype = SOURCES[name].type
    if qtype == "choice":
        cands, g = sample_choice(gold, labels, rng)
    elif qtype == "noul":
        cands, g = ["true", "false"], 0 if gold else 1
    elif name == "civil":
        cands, g = list(CIVIL_LEVELS), int(gold)
    else:
        cands, g = list(SENTIMENT_LEVELS[int(rng.integers(len(SENTIMENT_LEVELS)))]), int(gold)
    return _example(name, lang, qtype, state, TEMPLATES[name][int(rng.integers(len(TEMPLATES[name])))], cands, g)


def _example(name: str, lang: str, qtype: str, state: object, instructions: str, cands: list, gold: int) -> dict:
    return {
        "source": name,
        "lang": lang,
        "type": qtype,
        "state": canonical_state(state),
        "instructions": instructions,
        "candidates": [c if isinstance(c, tuple) else (c, None) for c in cands],
        "gold": gold,
    }


def _pick(recs: list, n: int, rng: np.random.Generator) -> list:
    return [recs[int(i)] for i in rng.permutation(len(recs))[:n]]


def suite_states(data_dir: Path | None = None) -> set[str]:
    """Canonical states of every built evaluation case (leakage guard)."""
    states = set()
    for s in kb.suites():
        if ((data_dir or kb.DATA_DIR) / f"{s.id}.jsonl").exists():
            states |= {canonical_state(c["request"]["state"]) for c in kb.load_suite(s.id, data_dir)}
    if not states:
        raise SystemExit("build the suites first (`krite-bench data`): the leakage guard needs them")
    return states


def drop_leaked(recs: list, held: set[str]) -> list:
    return [r for r in recs if canonical_state(r[1]) not in held]


def source_examples(
    name: str, recs: list, scale: float, rng: np.random.Generator, cap: int | None = None
) -> list[dict]:
    src = SOURCES[name]
    cap = int((cap or src.cap) * scale)
    if name in LANG_GROUPS:  # even split per language
        groups = {lg: [r for r in recs if r[0] == lg] for lg in LANG_GROUPS[name]}
    elif name in LEVEL_GROUPS:  # stratified per level
        groups = {lv: [r for r in recs if r[2] == lv] for lv in range(LEVEL_GROUPS[name])}
    else:
        groups = {"": recs}
    per = cap // len(groups)
    short = {g: len(v) for g, v in groups.items() if len(v) < per}
    if short:
        raise SystemExit(f"{name}: groups below {per} rows: {short}")
    picked = [r for g in groups.values() for r in _pick(g, per, rng)]
    labels = sorted({r[2] for r in recs}) if src.type == "choice" else []
    return [make_example(name, lang, state, gold, labels, rng) for lang, state, gold in picked]


def view_examples(name: str, recs: list, scale: float, held: set[str], rng: np.random.Generator) -> list[dict]:
    """Broad mixture: the source's rows re-asked as another question type (VIEWS)."""
    out = []
    for qtype, count in VIEWS.get(name, []):
        n = int(count * scale)
        if name in DOMAINS:  # noul: is the gold label, or a random other label, the right one?
            labels = sorted({r[2] for r in recs})
            tmpl = VIEW_TEMPLATES[DOMAINS[name]]
            for lang, state, gold in _pick(recs, n, rng):
                others = [x for x in labels if x != gold]
                ask = gold if rng.random() < 0.5 else others[int(rng.integers(len(others)))]
                ins = tmpl[int(rng.integers(len(tmpl)))].format(label=ask)
                out.append(_example(name, lang, qtype, state, ins, ["true", "false"], 0 if ask == gold else 1))
        elif name == "civil":  # noul: toxic means level >= 2 (toxicity >= 0.4)
            toxic = _pick([r for r in recs if r[2] >= 2], n // 2, rng)
            clean = _pick([r for r in recs if r[2] < 2], n - n // 2, rng)
            tmpl = VIEW_TEMPLATES["toxic"]
            for (lang, state, _), yes in [(r, True) for r in toxic] + [(r, False) for r in clean]:
                ins = tmpl[int(rng.integers(len(tmpl)))]
                out.append(_example(name, lang, qtype, state, ins, ["true", "false"], 0 if yes else 1))
        else:  # snli, mnli: 3-way choice with described criteria
            tmpl = VIEW_TEMPLATES["nli"]
            for lang, state, label in _pick(drop_leaked(nli_records(name), held), n, rng):
                ins = tmpl[int(rng.integers(len(tmpl)))]
                out.append(_example(name, lang, qtype, state, ins, NLI_CRITERIA, label))
    return [{**e, "view": True} for e in out]


def format_state(state: str, is_json: bool, rng: np.random.Generator) -> str:
    """A text state wrapped in a JSON object, or a JSON state with synonym keys; maybe unrelated fields added."""
    if is_json:
        obj = {}
        src = json.loads(state)
        for k, v in src.items():
            syn = KEY_SYNONYMS.get(k)
            new = syn[int(rng.integers(len(syn)))] if syn and rng.random() < 0.5 else k
            obj[k if new in src or new in obj else new] = v
    else:
        obj = {STATE_KEYS[int(rng.integers(len(STATE_KEYS)))]: state}
    if rng.random() < 0.5:
        for k in rng.choice(sorted(EXTRA_FIELDS), size=int(rng.integers(1, 3)), replace=False).tolist():
            obj.setdefault(k, EXTRA_FIELDS[k](rng))
    return canonical_state(obj)


def foreign(cands: list, gold: int, pool: list[str], rng: np.random.Generator) -> list:
    """Replace min(2, K - 2) non-gold candidates with labels from `pool`; K and gold index unchanged."""
    names = [c for c, _ in cands]
    pool = [x for x in pool if x not in names]
    m = min(2, len(names) - 2)
    slots = rng.choice([i for i in range(len(names)) if i != gold], size=m, replace=False).tolist()
    for i, x in zip(slots, rng.choice(pool, size=m, replace=False).tolist(), strict=True):
        names[i] = x
    return [(n, None) for n in names]


def augment(ex: dict, labels_by_domain: dict[str, list[str]], rng: np.random.Generator) -> dict:
    """Broad mixture: state formatting and irrelevant candidates (docs/training-data.md); returns a new dict."""
    ex = dict(ex)
    if rng.random() < STATE_FORMAT_P:
        ex["state"] = format_state(ex["state"], ex["source"] in DICT_STATE_SOURCES, rng)
    k = len(ex["candidates"])
    if ex["type"] == "choice" and ex["source"] in DOMAINS and k >= 3 and rng.random() < FOREIGN_P:
        pool = labels_by_domain[OTHER_DOMAIN[DOMAINS[ex["source"]]]]
        ex["candidates"] = foreign(ex["candidates"], ex["gold"], pool, rng)
    return ex


def build(
    seed: int = kb.SEED, scale: float = 1.0, data_dir: Path | None = None, mixture: str = "study"
) -> tuple[list[dict], dict]:
    """One mixture's sources, leakage-filtered and shuffled. Returns (examples, meta with mixture and train sha256)."""
    held = suite_states(data_dir)
    broad = mixture == "broad"
    cached: dict[str, list] = {}  # DOMAINS records, read once for the label pools and once for sampling
    labels_by_domain: dict[str, list[str]] = {}
    if broad:
        pools: dict[str, set[str]] = {}
        for name, dom in DOMAINS.items():
            cached[name] = records(name)
            pools.setdefault(dom, set()).update(r[2] for r in cached[name])
        labels_by_domain = {d: sorted(v) for d, v in pools.items()}
    out, meta = [], {"mixture": mixture}
    for i, name in enumerate(SOURCES):
        if name not in MIXTURES[mixture]:
            continue
        rng = np.random.default_rng([seed, i])
        recs = cached.pop(name) if name in cached else records(name)
        kept = drop_leaked(recs, held)
        exs = source_examples(name, kept, scale, rng, CAPS.get(mixture, {}).get(name))
        src = SOURCES[name]
        meta[name] = {
            "repo": src.repo,
            "revision": src.revision,
            "license": src.license,
            "n": len(exs),
            "dropped_leaked": len(recs) - len(kept),
        }
        if broad:
            views = view_examples(name, kept, scale, held, np.random.default_rng([seed, i, 1]))
            meta[name]["views"] = dict(Counter(e["type"] for e in views))
            exs = [
                augment(ex, labels_by_domain, np.random.default_rng([seed, i, 2, j]))
                for j, ex in enumerate(exs + views)
            ]
        out += exs
        extra = f", {meta[name]['views']} views" if broad else ""
        print(f"{name}: {meta[name]['n']} examples ({len(recs) - len(kept)} leaked rows dropped{extra})", flush=True)
    rng = np.random.default_rng(seed)
    out = [out[int(i)] for i in rng.permutation(len(out))]
    meta["train_sha256"] = hashlib.sha256(json.dumps(out, ensure_ascii=False).encode()).hexdigest()
    return out, meta


def batches(examples: list[dict], size: int, seed: int = kb.SEED) -> list[list[dict]]:
    """Batches of one candidate count each (fixed tensor shapes on MPS); ragged tails dropped; seeded order."""
    by_k: dict[int, list[dict]] = {}
    for ex in examples:
        by_k.setdefault(len(ex["candidates"]), []).append(ex)
    out = [g[i : i + size] for g in by_k.values() for i in range(0, len(g) - size + 1, size)]
    rng = np.random.default_rng(seed)
    return [out[int(i)] for i in rng.permutation(len(out))]
