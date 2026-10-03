"""Deterministic benchmark suites (benchmark-spec §11) and synthetic latency/interference fixtures.

Cases are Protocol v1 requests plus gold labels. Dataset text is written only to the gitignored
benchmarks/data/ directory; only ids and hashes leave it (manifest.json).
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "benchmarks" / "data"
SCHEMA_DIR = ROOT / "docs" / "protocol"
SEED = 13
N_CASES = 400

TOKENIZER_REPO = "jhu-clsp/mmBERT-small"
TOKENIZER_REVISION = "abc32620dd4f6ab06f5fbe905dc25f310618e09f"


@dataclass(frozen=True)
class Dataset:
    repo: str
    revision: str
    license: str
    instructions: str


DATASETS = {
    "agnews": Dataset(
        "fancyzhx/ag_news",
        "eb185aade064a813bc0b7f42de02595523103ca4",
        "unspecified; eval-only",
        "Which topic is this news article about?",
    ),
    "banking77": Dataset(
        "mteb/banking77",
        "18072d2685ea682290f7b8924d94c62acc19c0b2",
        "CC-BY-4.0 (upstream PolyAI/banking77)",
        "Which customer-support intent does this banking message express?",
    ),
    "massive": Dataset(
        "mteb/amazon_massive_intent",
        "940fd47a81eaa7f2cc7b129674d945d618ac38c2",
        "CC-BY-4.0 (upstream AmazonScience/massive)",
        "Which intent does this voice-assistant command express?",
    ),
    "boolq": Dataset(
        "google/boolq",
        "35b264d03638db9f4ce671b711558bf7ff0f80d5",
        "CC-BY-SA-3.0",
        "Based on the passage, is the answer to the question yes?",
    ),
    "xnli": Dataset(
        "facebook/xnli",
        "b8dd5d7af51114dbda02c0e3f6133f332186418e",
        "CC-BY-NC-4.0; eval-only",
        "Does the premise entail the hypothesis?",
    ),
    "sst5": Dataset(
        "SetFit/sst5",
        "e51bdcd8cd3a30da231967c1a249ba59361279a3",
        "unspecified; eval-only",
        "How positive is the sentiment of this movie review?",
    ),
    "amazon": Dataset(
        "mteb/amazon_reviews_multi",
        "c379a6705fec24a2493fa68e011692605f44e119",
        "Amazon Multilingual Reviews Corpus terms (research); eval-only",
        "How many stars did the reviewer give the product?",
    ),
}

AGNEWS_LABELS = ["World", "Sports", "Business", "Sci/Tech"]
SST5_LEVELS = ["very negative", "negative", "neutral", "positive", "very positive"]
STAR_LEVELS = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
MASSIVE_LANGS = ["en", "ko", "ja", "zh-CN", "de", "es", "fr", "ar", "hi", "ru"]
XNLI_LANGS = ["en", "de", "es", "fr", "ar", "hi", "ru", "zh"]
AMAZON_LANGS = ["en", "de", "es", "fr", "ja", "zh"]


@dataclass(frozen=True)
class Suite:
    id: str
    dataset: str
    type: str
    lang: str
    file: str
    k: int | None = None  # choice only: gold + (k - 1) seeded distractors; None = all labels


def suites() -> list[Suite]:
    out = [
        Suite("agnews-choice", "agnews", "choice", "en", "data/test-00000-of-00001.parquet"),
        Suite("banking77-choice", "banking77", "choice", "en", "data/test-00000-of-00001.parquet", 12),
        Suite("boolq-noul", "boolq", "noul", "en", "data/validation-00000-of-00001.parquet"),
        Suite("sst5-score", "sst5", "score", "en", "test.jsonl"),
    ]
    out += [Suite(f"massive-choice-{lg}", "massive", "choice", lg, f"test/{lg}.json.gz", 8) for lg in MASSIVE_LANGS]
    out += [Suite(f"xnli-noul-{lg}", "xnli", "noul", lg, f"{lg}/test-00000-of-00001.parquet") for lg in XNLI_LANGS]
    out += [Suite(f"amazon-score-{lg}", "amazon", "score", lg, f"{lg}/test.jsonl") for lg in AMAZON_LANGS]
    return out


def bucket(qtype: str, k: int) -> str:
    """Calibration bucket from the ARCHITECTURE.md Glossary table."""
    if qtype == "noul":
        return "noul"
    if qtype == "choice":
        return f"choice/{k}" if k <= 4 else ("choice/5-8" if k <= 8 else "choice/9+")
    return f"score/{k}" if k <= 5 else "score/6+"


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _download(suite: Suite) -> str:
    ds = DATASETS[suite.dataset]
    return hf_hub_download(ds.repo, suite.file, repo_type="dataset", revision=ds.revision)


def _read_rows(path: str) -> list[dict]:
    if path.endswith(".parquet"):
        return pq.read_table(path).to_pylist()
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _label_name(s: str) -> str:
    return s.replace("_", " ")


def _records(suite: Suite, rows: list[dict]) -> tuple[list[tuple[str, object, object]], list]:
    """Return ([(source_id, state, gold)], candidate names) for a suite. gold: name | level | bool."""
    d = suite.dataset
    if d == "agnews":
        recs = [(str(i), r["text"], AGNEWS_LABELS[int(r["label"])]) for i, r in enumerate(rows)]
        return recs, AGNEWS_LABELS
    if d in ("banking77", "massive"):
        key = "label_text" if d == "banking77" else "label"
        recs = [(str(r.get("id", i)), r["text"], _label_name(r[key])) for i, r in enumerate(rows)]
        return recs, sorted({g for _, _, g in recs})
    if d == "boolq":
        recs = [
            (str(i), {"passage": r["passage"], "question": r["question"]}, bool(r["answer"]))
            for i, r in enumerate(rows)
        ]
        return recs, [True, False]
    if d == "xnli":  # 0 entailment, 1 neutral (dropped), 2 contradiction
        recs = [
            (str(i), {"premise": r["premise"], "hypothesis": r["hypothesis"]}, int(r["label"]) == 0)
            for i, r in enumerate(rows)
            if int(r["label"]) != 1
        ]
        return recs, [True, False]
    if d == "sst5":
        return [(str(i), r["text"], int(r["label"])) for i, r in enumerate(rows)], SST5_LEVELS
    if d == "amazon":
        return [(str(r["id"]), r["text"], int(r["label"])) for r in rows], STAR_LEVELS
    raise ValueError(f"unknown dataset {d}")


def build_suite(suite: Suite, rows: list[dict], n: int = N_CASES, seed: int = SEED) -> list[dict]:
    rng = np.random.default_rng([seed, int(hashlib.sha256(suite.id.encode()).hexdigest()[:8], 16)])
    recs, labels = _records(suite, rows)
    picks = rng.choice(len(recs), size=min(n, len(recs)), replace=False)
    instructions = DATASETS[suite.dataset].instructions
    cases = []
    for idx in picks:
        source_id, state, gold = recs[int(idx)]
        question: dict = {"type": suite.type, "instructions": instructions}
        if suite.type == "choice":
            others = [x for x in labels if x != gold]
            k = suite.k or len(labels)
            names = [gold, *rng.choice(others, size=k - 1, replace=False).tolist()]
            names = [names[int(j)] for j in rng.permutation(len(names))]  # gold position is random
            question["criteria"] = {name: None for name in names}
            k_eff = len(names)
        elif suite.type == "score":
            question["criteria"] = list(labels)
            k_eff = len(labels)
        else:
            k_eff = 2
        cases.append(
            {
                "case_id": f"{suite.id}/{source_id}",
                "suite": suite.id,
                "dataset": suite.dataset,
                "lang": suite.lang,
                "type": suite.type,
                "bucket": bucket(suite.type, k_eff),
                "request": {"state": state, "questions": {"q": question}},
                "gold": {"q": gold},
            }
        )
    return cases


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
    return sha256_file(path)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_all(out_dir: Path = DATA_DIR, only: list[str] | None = None) -> dict:
    """Build suites into out_dir and return the manifest (also written to manifest.json).

    With `only`, entries for the other suites already in out_dir/manifest.json are kept.
    """
    unknown = set(only or ()) - {s.id for s in suites()}
    if unknown:
        raise ValueError(f"unknown suite ids: {sorted(unknown)}")
    manifest = {"seed": SEED, "instructions_lang": "en", "suites": {}}
    if only and (out_dir / "manifest.json").exists():
        manifest["suites"] = json.loads((out_dir / "manifest.json").read_text())["suites"]
    for suite in suites():
        if only and suite.id not in only:
            continue
        path = _download(suite)
        cases = build_suite(suite, _read_rows(path))
        ds = DATASETS[suite.dataset]
        manifest["suites"][suite.id] = {
            "dataset": ds.repo,
            "revision": ds.revision,
            "file": suite.file,
            "file_sha256": sha256_file(path),
            "license": ds.license,
            "lang": suite.lang,
            "type": suite.type,
            "n": len(cases),
            "cases_sha256": write_jsonl(out_dir / f"{suite.id}.jsonl", cases),
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_suite(suite_id: str, data_dir: Path | None = None) -> list[dict]:
    path = (data_dir or DATA_DIR) / f"{suite_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `krite-bench data` first")
    return read_jsonl(path)


# --- synthetic fixtures -------------------------------------------------------------------------

_WORDS = (
    "the of and to in is was for on that with as by at from this be are have it an not but or "
    "which one all were had their there been has more when will would who out up so can time if "
    "about into than them only other new some could these two may first then do any like my now "
    "over such our man me even most made after also did many before must through back years where "
    "much your way well down should because each just those people how too little state good very "
    "make world still own see men work long get here between both life being under never day same "
    "another know while last might us great old year off come since against go came right used take "
    "three himself few house use during without again place around however home small found"
).split()


@cache
def tokenizer():
    from tokenizers import Tokenizer

    path = hf_hub_download(TOKENIZER_REPO, "tokenizer.json", revision=TOKENIZER_REVISION)
    return Tokenizer.from_file(path)


def count_tokens(text: str) -> int:
    """mmBERT-small tokens of the text content, excluding special tokens."""
    return len(tokenizer().encode(text, add_special_tokens=False).ids)


@cache
def _single_token_words() -> tuple[str, ...]:
    tok = tokenizer()
    words = tuple(w for w in _WORDS if len(tok.encode(" " + w, add_special_tokens=False).ids) == 1)
    if len(words) < 50:
        raise RuntimeError(f"only {len(words)} single-token words; word list unsuitable")
    return words


def synthetic_state(n_tokens: int, seed: int | list[int]) -> str:
    """Seeded word salad of exactly n_tokens mmBERT-small tokens."""
    words = _single_token_words()
    rng = np.random.default_rng(seed)
    text = " ".join(words[int(i)] for i in rng.integers(0, len(words), size=n_tokens))
    if count_tokens(text) != n_tokens:
        raise RuntimeError(f"synthetic state has {count_tokens(text)} tokens, wanted {n_tokens}")
    return text


def latency_request(state_tokens: int, questions: int, k: int, seed: int | list[int]) -> dict:
    qs = {
        f"q{i}": {
            "type": "choice",
            "instructions": f"Question {i}: which option fits the text best?",
            "criteria": {f"option {j}": None for j in range(k)},
        }
        for i in range(questions)
    }
    return {"state": synthetic_state(state_tokens, seed), "questions": qs}


FILLER_QUESTIONS = [
    {"type": "noul", "instructions": "Is the text written in a formal register?"},
    {"type": "score", "instructions": "How positive is the tone?", "criteria": SST5_LEVELS},
    {
        "type": "choice",
        "instructions": "Who is the most likely audience?",
        "criteria": {"general public": None, "specialists": None, "children": None},
    },
    {"type": "noul", "instructions": "Does the text mention a specific person?"},
    {"type": "noul", "instructions": "Does the text mention a number?"},
    {
        "type": "score",
        "instructions": "How urgent does the text sound?",
        "criteria": ["not urgent", "somewhat urgent", "urgent", "very urgent"],
    },
    {
        "type": "choice",
        "instructions": "What is the text's main purpose?",
        "criteria": {"inform": None, "persuade": None, "entertain": None, "request": None},
    },
    {"type": "noul", "instructions": "Is the text longer than two sentences?"},
    {"type": "noul", "instructions": "Does the text contain a question?"},
    {
        "type": "score",
        "instructions": "How technical is the vocabulary?",
        "criteria": ["plain", "some jargon", "heavy jargon"],
    },
    {
        "type": "choice",
        "instructions": "Which region is most relevant?",
        "criteria": {"Americas": None, "Europe": None, "Asia": None, "Africa": None, "none": None},
    },
    {"type": "noul", "instructions": "Does the text express an opinion?"},
    {"type": "noul", "instructions": "Is the text about a past event?"},
    {
        "type": "score",
        "instructions": "How controversial is the topic?",
        "criteria": ["not at all", "slightly", "moderately", "highly"],
    },
    {
        "type": "choice",
        "instructions": "Which tense dominates the text?",
        "criteria": {"past": None, "present": None, "future": None},
    },
]


def interference_request(case: dict, n_fillers: int) -> dict:
    """The case's target question `q` plus the first n_fillers filler questions, same state."""
    qs = {"q": case["request"]["questions"]["q"]}
    qs.update({f"f{i}": FILLER_QUESTIONS[i] for i in range(n_fillers)})
    return {"state": case["request"]["state"], "questions": qs}
