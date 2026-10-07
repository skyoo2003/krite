import json

import jsonschema
import pytest

from krite_bench import data as d

SCHEMA = json.loads((d.SCHEMA_DIR / "request.schema.json").read_text())


def _rows(n=50):
    labels = [f"label_{i}" for i in range(20)]
    return [{"id": str(i), "text": f"text {i}", "label": labels[i % 20]} for i in range(n)]


def test_build_suite_deterministic_and_valid():
    suite = d.Suite("massive-choice-xx", "massive", "choice", "xx", "unused", 8)
    a = d.build_suite(suite, _rows(), n=30)
    b = d.build_suite(suite, _rows(), n=30)
    assert a == b
    for case in a:
        jsonschema.validate(case["request"], SCHEMA)
        names = list(case["request"]["questions"]["q"]["criteria"])
        assert len(names) == 8 == len(set(names))
        assert case["gold"]["q"] in names
        assert case["bucket"] == "choice/5-8"


def test_gold_position_varies():
    suite = d.Suite("massive-choice-xx", "massive", "choice", "xx", "unused", 8)
    cases = d.build_suite(suite, _rows(), n=40)
    positions = {list(c["request"]["questions"]["q"]["criteria"]).index(c["gold"]["q"]) for c in cases}
    assert len(positions) > 3


def test_buckets():
    assert d.bucket("choice", 4) == "choice/4"
    assert d.bucket("choice", 12) == "choice/9+"
    assert d.bucket("score", 5) == "score/5"
    assert d.bucket("score", 7) == "score/6+"
    assert d.bucket("noul", 2) == "noul"


def test_interference_request_keeps_target():
    case = {"request": {"state": "s", "questions": {"q": {"type": "noul", "instructions": "i"}}}}
    req = d.interference_request(case, 15)
    jsonschema.validate(req, SCHEMA)
    assert len(req["questions"]) == 16 and req["questions"]["q"] == case["request"]["questions"]["q"]


@pytest.mark.network
def test_synthetic_state_exact_tokens_and_distinct():
    states = {d.synthetic_state(512, s) for s in range(3)}
    assert len(states) == 3
    assert all(d.count_tokens(s) == 512 for s in states)
    jsonschema.validate(d.latency_request(64, 4, 8, 0), SCHEMA)


def test_root_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("KRITE_ROOT", str(tmp_path))
    assert d.find_root() == tmp_path.resolve()


def test_root_defaults_to_checkout(monkeypatch):
    monkeypatch.delenv("KRITE_ROOT", raising=False)
    assert (d.find_root() / "benchmarks" / "krite_bench").is_dir()


def test_bundled_schemas_match_spec(monkeypatch):
    monkeypatch.delenv("KRITE_ROOT", raising=False)
    spec = d.find_root() / "docs" / "protocol"
    names = sorted(p.name for p in spec.glob("*.schema.json"))
    assert names == sorted(p.name for p in d.SCHEMA_DIR.glob("*.schema.json"))
    for n in names:
        assert (d.SCHEMA_DIR / n).read_bytes() == (spec / n).read_bytes(), n
