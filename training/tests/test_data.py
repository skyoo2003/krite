import json

import numpy as np
import pytest
from krite_bench import data as kb

from krite_train import data as d


def test_choice_sampling_is_seeded_and_contains_gold():
    labels = [f"label {i}" for i in range(20)]
    for seed in range(50):
        a = d.sample_choice("label 3", labels, np.random.default_rng(seed))
        b = d.sample_choice("label 3", labels, np.random.default_rng(seed))
        names, gold = a
        assert a == b
        assert names[gold] == "label 3"
        assert 2 <= len(names) <= d.MAX_K and len(set(names)) == len(names)


def test_noul_and_score_examples_keep_fixed_candidates():
    rng = np.random.default_rng(0)
    noul = d.make_example("snli", "en", {"premise": "p", "hypothesis": "h"}, False, [], rng)
    assert [c for c, _ in noul["candidates"]] == ["true", "false"] and noul["gold"] == 1
    assert noul["state"] == '{"hypothesis":"h","premise":"p"}'
    score = d.make_example("civil", "en", "text", 3, [], rng)
    assert [c for c, _ in score["candidates"]] == d.CIVIL_LEVELS and score["gold"] == 3


def test_batches_have_uniform_k():
    rng = np.random.default_rng(1)
    labels = [f"l{i}" for i in range(12)]
    exs = [d.make_example("clinc", "en", f"s{i}", "l0", labels, rng) for i in range(400)]
    bs = d.batches(exs, 8)
    assert bs and all(len(b) == 8 and len({len(e["candidates"]) for e in b}) == 1 for b in bs)
    assert bs == d.batches(exs, 8)


def test_templates_never_equal_eval_instructions():
    used = {t for ts in d.TEMPLATES.values() for t in ts}
    used |= {t.format(label="x") for ts in d.VIEW_TEMPLATES.values() for t in ts}
    assert used.isdisjoint({ds.instructions for ds in kb.DATASETS.values()})
    assert set(d.TEMPLATES) == set(d.SOURCES)


def test_leakage_guard_drops_suite_states(tmp_path):
    case = {"request": {"state": {"question": "q", "passage": "p"}, "questions": {}}}
    (tmp_path / "boolq-noul.jsonl").write_text(json.dumps(case) + "\n")
    held = d.suite_states(tmp_path)
    recs = [("en", {"passage": "p", "question": "q"}, True), ("en", {"passage": "x", "question": "q"}, True)]
    assert d.drop_leaked(recs, held) == recs[1:]


def test_leakage_guard_needs_built_suites(tmp_path):
    with pytest.raises(SystemExit):
        d.suite_states(tmp_path)


def test_read_tsv_keeps_quotes(tmp_path):
    p = tmp_path / "t.tsv"
    p.write_text("index_id\tcategory\ttext\n1\tsports\tHe said \"go\" and 'left'\n")
    assert d.read_tsv(str(p)) == [{"index_id": "1", "category": "sports", "text": "He said \"go\" and 'left'"}]


def test_goemotions_polarity():
    assert d.polarity(["joy"]) == 2 and d.polarity(["joy", "love"]) == 2
    assert d.polarity(["neutral"]) == 1 and d.polarity(["anger", "sadness"]) == 0
    assert d.polarity(["joy", "anger"]) is None
    assert d.polarity(["surprise"]) is None and d.polarity(["joy", "curiosity"]) is None
    assert d.polarity([]) is None


def test_sentiment_score_example_uses_three_levels():
    seen = set()
    for seed in range(20):
        ex = d.make_example("goemotions", "en", "text", 2, [], np.random.default_rng(seed))
        names = tuple(c for c, _ in ex["candidates"])
        assert ex["type"] == "score" and names in d.SENTIMENT_LEVELS and ex["gold"] == 2
        seen.add(names)
    assert seen == set(d.SENTIMENT_LEVELS)


def synthetic(n: int, labels: int, lang: str = "en") -> list:
    return [(lang, f"state {i}", f"l{i % labels}") for i in range(n)]


def test_noul_view_from_choice_asks_one_label():
    exs = d.view_examples("banking77", synthetic(3000, 5), 1.0, set(), np.random.default_rng(0))
    assert len(exs) == 1000 and all(e["view"] and e["type"] == "noul" for e in exs)
    states = {f"state {i}": f"l{i % 5}" for i in range(3000)}
    for e in exs:
        assert [c for c, _ in e["candidates"]] == ["true", "false"]
        asked = next(lb for lb in (f"l{i}" for i in range(5)) if lb in e["instructions"])
        assert e["gold"] == (0 if asked == states[e["state"]] else 1)
    assert 0.4 <= sum(e["gold"] == 0 for e in exs) / len(exs) <= 0.6


def test_civil_noul_view_is_balanced():
    recs = [("en", f"c{i}", i % 5) for i in range(5000)]
    exs = d.view_examples("civil", recs, 1.0, set(), np.random.default_rng(0))
    assert len(exs) == 1000 and sum(e["gold"] == 0 for e in exs) == 500
    level = {f"c{i}": i % 5 for i in range(5000)}
    assert all((e["gold"] == 0) == (level[e["state"]] >= 2) for e in exs)


def test_nli_choice_view_uses_descriptions(monkeypatch):
    rows = [("en", {"premise": f"p{i}", "hypothesis": "h"}, i % 3) for i in range(9000)]
    monkeypatch.setattr(d, "nli_records", lambda name: rows)
    exs = d.view_examples("snli", [], 1.0, set(), np.random.default_rng(0))
    assert len(exs) == 2000 and all(e["type"] == "choice" and e["candidates"] == d.NLI_CRITERIA for e in exs)
    gold = {d.canonical_state(s): lab for _, s, lab in rows}
    assert all(e["gold"] == gold[e["state"]] for e in exs)


def test_state_formatting_preserves_content():
    for seed in range(50):
        out = d.format_state("hello world", False, np.random.default_rng(seed))
        obj = json.loads(out)
        assert "hello world" in obj.values() and out == d.canonical_state(obj)
        assert out == d.format_state("hello world", False, np.random.default_rng(seed))
        state = d.canonical_state({"passage": "p", "question": "q"})
        obj = json.loads(d.format_state(state, True, np.random.default_rng(seed)))
        assert {"p", "q"} <= set(obj.values())
        keys = set(obj) - {"turn", "channel", "step"}
        assert len(keys) == 2 and keys <= {
            "passage",
            "question",
            *d.KEY_SYNONYMS["passage"],
            *d.KEY_SYNONYMS["question"],
        }
    # A synonym that collides with another key is skipped.
    state = d.canonical_state({"premise": "a", "context": "b"})
    for seed in range(50):
        obj = json.loads(d.format_state(state, True, np.random.default_rng(seed)))
        assert {"a", "b"} <= set(obj.values())


def test_foreign_distractors_keep_gold_and_k():
    labels = {"intent": [f"i{j}" for j in range(30)], "topic": [f"t{j}" for j in range(20)]}
    rng = np.random.default_rng(0)
    for seed in range(200):
        r = np.random.default_rng(seed)
        names, gold = d.sample_choice("i0", labels["intent"], r)
        ex = {
            "source": "banking77",
            "type": "choice",
            "state": "s",
            "candidates": [(n, None) for n in names],
            "gold": gold,
        }
        out = d.foreign(ex["candidates"], gold, labels["topic"], rng)
        new = [c for c, _ in out]
        assert len(new) == len(names) and new[gold] == "i0" and len(set(new)) == len(new)
        replaced = [n for n in new if n not in names]
        assert all(n.startswith("t") for n in replaced) and len(replaced) == min(2, len(names) - 2)
    ex = {"source": "banking77", "type": "choice", "state": "s", "candidates": [("i0", None), ("i1", None)], "gold": 0}
    for seed in range(50):
        assert d.augment(ex, labels, np.random.default_rng(seed))["candidates"] == ex["candidates"]


@pytest.mark.network
def test_build_small_scale():
    exs, meta = d.build(scale=0.01)
    assert {e["source"] for e in exs} == set(d.MIXTURES["study"])
    assert len(meta["train_sha256"]) == 64 and meta["mixture"] == "study"


STUDY_SHA_1PCT = "6e3fa02ae2f5cb4a7e31e3e59ee95eb549b5363299b8f6b44c31e433b064a0d6"


@pytest.mark.network
def test_study_mixture_is_frozen():
    assert d.build(scale=0.01, mixture="study")[1]["train_sha256"] == STUDY_SHA_1PCT


@pytest.mark.network
def test_build_broad_small_scale():
    exs, meta = d.build(scale=0.01, mixture="broad")
    assert {e["source"] for e in exs} == set(d.SOURCES) and meta["mixture"] == "broad"
    views = [e for e in exs if e.get("view")]
    assert {e["type"] for e in views} == {"noul", "choice"}
    assert meta["boolq"]["n"] == 80 and meta["mnli"]["views"] == {"choice": 30}


@pytest.mark.network
def test_nli_option_fits():
    from krite_train import model as m

    tok = m.tokenizer()
    for name, desc in d.NLI_CRITERIA:
        assert len(m._ids(tok, "\n" + m.option_text(name, desc))) <= m.MAX_OPTION
