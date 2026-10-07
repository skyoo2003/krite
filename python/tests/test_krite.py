"""The binding against a tiny random model on the CPU: shapes, sums, errors, and threads, never answer quality."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

import krite

STATE = "The customer was charged twice for March."
REQUEST = {
    "state": STATE,
    "questions": {
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": None, "technical": None, "other": None},
        },
        "urgency": {"type": "score", "instructions": "How urgent is it?", "criteria": ["low", "medium", "high"]},
        "refund": {"type": "noul", "instructions": "Did they ask for a refund?"},
    },
}


@pytest.fixture(scope="module")
def k(tiny_model):
    return krite.Krite(tiny_model, device="cpu")


def test_answers_every_question_type(k):
    r = k.decide(REQUEST)
    assert (r["model"], k.model_id) == ("tiny", "tiny")
    a = r["answers"]
    assert a["route"]["choice"] in {"billing", "technical", "other"}
    assert a["urgency"]["legend"] == {"0": "low", "1": "medium", "2": "high"}
    for q in ("route", "urgency"):
        assert abs(sum(a[q]["probabilities"].values()) - 1) < 1e-6
    assert 0 <= a["refund"]["noul"] <= 1
    assert r["usage"]["input_tokens"] > 0
    assert r["latency_ms"] >= 0


def test_jev_alias_and_unknown_model(k):
    assert k.decide({**REQUEST, "model": "jev-latest"})["model"] == "tiny"
    with pytest.raises(krite.KriteError) as e:
        k.decide({**REQUEST, "model": "nope"})
    assert (e.value.status, e.value.type, e.value.param) == (422, "unknown_model", "model")


def test_invalid_requests(k):
    with pytest.raises(krite.KriteError) as e:
        k.decide("{")
    assert (e.value.status, e.value.type) == (422, "invalid_request")
    many = {f"q{i}": {"type": "noul"} for i in range(65)}
    with pytest.raises(krite.KriteError) as e:
        k.decide({"state": STATE, "questions": many})
    assert e.value.type == "too_many_questions"
    with pytest.raises(krite.KriteError) as e:
        k.decide(b" " * (4 * 1024 * 1024 + 1))
    assert e.value.type == "invalid_request"
    assert "max is" in e.value.message


def test_dict_str_bytes_agree(k):
    text = json.dumps(REQUEST)
    rs = [k.decide(REQUEST), k.decide(text), k.decide(text.encode())]
    assert all((r["answers"], r["usage"]) == (rs[0]["answers"], rs[0]["usage"]) for r in rs)


def test_threads(k):
    reqs = [REQUEST, {**REQUEST, "state": "Did they ask which team should handle a refund?"}] * 4
    want = [k.decide(r)["answers"] for r in reqs[:2]]
    with ThreadPoolExecutor(4) as ex:
        got = list(ex.map(lambda r: k.decide(r)["answers"], reqs))
    assert got == want * 4


def test_load_errors(tmp_path, tiny_model):
    with pytest.raises(RuntimeError, match="krite.json missing"):
        krite.Krite(tmp_path / "none", device="cpu")
    with pytest.raises(ValueError, match="device"):
        krite.Krite(tiny_model, device="gpu")
