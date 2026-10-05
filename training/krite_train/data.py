"""Training examples for architecture studies, from permissively licensed train splits only.

Every source is pinned to a Hugging Face revision. Dataset text stays in the HF cache and in memory;
only counts and hashes are written out (train_meta.json). The held-out evaluation datasets (agnews,
xnli, sst5, amazon) are never read here, and every row whose state appears in a built suite is
dropped (leakage guard), so the in-domain suites (banking77, massive, boolq) stay unseen too.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from krite_bench import data as kb

MAX_K = 12  # most candidates a training choice question gets


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
}

CIVIL_LEVELS = ["not toxic", "slightly toxic", "moderately toxic", "very toxic", "extremely toxic"]
CIVIL_EDGES = [0.2, 0.4, 0.6, 0.8]

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
    if name == "snli":
        rows = kb._read_rows(paths[0])
        return [
            ("en", {"premise": r["premise"], "hypothesis": r["hypothesis"]}, r["label"] == 0)
            for r in rows
            if r["label"] in (0, 2)
        ]
    if name == "civil":
        out = []
        for p in paths:
            t = pq.read_table(p, columns=["text", "toxicity"])
            levels = np.digitize(t.column("toxicity").to_numpy(), CIVIL_EDGES)
            out += [("en", text, int(lv)) for text, lv in zip(t.column("text").to_pylist(), levels, strict=True)]
        return out
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
    else:
        cands, g = list(CIVIL_LEVELS), int(gold)
    return {
        "source": name,
        "lang": lang,
        "type": qtype,
        "state": canonical_state(state),
        "instructions": TEMPLATES[name][int(rng.integers(len(TEMPLATES[name])))],
        "candidates": [(c, None) for c in cands],
        "gold": g,
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


def source_examples(name: str, recs: list, scale: float, rng: np.random.Generator) -> list[dict]:
    src = SOURCES[name]
    cap = int(src.cap * scale)
    if name == "massive":  # even split per language
        groups = {lg: [r for r in recs if r[0] == lg] for lg in kb.MASSIVE_LANGS}
    elif name == "civil":  # stratified per level
        groups = {lv: [r for r in recs if r[2] == lv] for lv in range(len(CIVIL_LEVELS))}
    else:
        groups = {"": recs}
    per = cap // len(groups)
    short = {g: len(v) for g, v in groups.items() if len(v) < per}
    if short:
        raise SystemExit(f"{name}: groups below {per} rows: {short}")
    picked = [r for g in groups.values() for r in _pick(g, per, rng)]
    labels = sorted({r[2] for r in recs}) if src.type == "choice" else []
    return [make_example(name, lang, state, gold, labels, rng) for lang, state, gold in picked]


def build(seed: int = kb.SEED, scale: float = 1.0, data_dir: Path | None = None) -> tuple[list[dict], dict]:
    """All sources, leakage-filtered and shuffled. Returns (examples, per-source meta with the train sha256)."""
    held = suite_states(data_dir)
    out, meta = [], {}
    for i, name in enumerate(SOURCES):
        rng = np.random.default_rng([seed, i])
        recs = records(name)
        kept = drop_leaked(recs, held)
        exs = source_examples(name, kept, scale, rng)
        src = SOURCES[name]
        meta[name] = {
            "repo": src.repo,
            "revision": src.revision,
            "license": src.license,
            "n": len(exs),
            "dropped_leaked": len(recs) - len(kept),
        }
        out += exs
        print(f"{name}: {len(exs)} examples ({len(recs) - len(kept)} leaked rows dropped)", flush=True)
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
