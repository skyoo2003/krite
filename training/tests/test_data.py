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


@pytest.mark.network
def test_build_small_scale():
    exs, meta = d.build(scale=0.01)
    assert {e["source"] for e in exs} == set(d.SOURCES)
    assert len(meta["train_sha256"]) == 64
