"""End-to-end harness tests against the deterministic fake engine (no ML)."""

import json
import subprocess
import sys
import time

import pytest

from krite_bench import data, env, runners
from krite_bench.client import Engine, engine_config

SHIM = data.ROOT / "benchmarks" / "baselines" / "shims" / "fake_shim.py"
EXAMPLES = data.ROOT / "docs" / "protocol" / "examples"
SPEC_FIELDS = (
    "engine model backend precision layer cache_state mode state_tokens questions options n "
    "p50_ms p95_ms p99_ms decisions_per_sec env_ref commit timestamp"
).split()


def _start(name: str, *args: str):
    port = engine_config(name)["port"]
    proc = subprocess.Popen([sys.executable, str(SHIM), "--port", str(port), *args])
    eng = Engine.from_config(name)
    for _ in range(100):
        if eng.call({"state": "x", "questions": {"q": {"type": "noul", "instructions": "i"}}}).ok:
            return proc, eng
        time.sleep(0.05)
    proc.kill()
    pytest.fail(f"{name} did not start on port {port}")


@pytest.fixture(scope="module")
def fake():
    proc, eng = _start("fake")
    yield eng
    eng.close()
    proc.kill()


@pytest.fixture(scope="module")
def biased():
    proc, eng = _start("fake-biased", "--biased")
    yield eng
    eng.close()
    proc.kill()


@pytest.fixture
def tiny_suite(tmp_path, monkeypatch):
    rows = [{"id": str(i), "text": f"text {i}", "label": f"label_{i % 10}"} for i in range(40)]
    cases = data.build_suite(data.Suite("massive-choice-xx", "massive", "choice", "xx", "unused", 4), rows, n=12)
    data.write_jsonl(tmp_path / "massive-choice-xx.jsonl", cases)
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    return tmp_path


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("example", ["choice.request.json", "multi.request.json"])
def test_protocol_examples_round_trip(fake, example):
    r = fake.call(json.loads((EXAMPLES / example).read_text()))
    assert r.ok, r.error
    assert r.runtime_ms and r.runtime_ms > 0


def test_unreachable_engine_is_error_not_exception():
    r = Engine("nobody", 1).call({"state": "x", "questions": {"q": {"type": "noul", "instructions": "i"}}})
    assert not r.ok and r.error


def test_quality_writes_summary(fake, tiny_suite, tmp_path):
    runners.run_quality("fake", ["massive-choice-xx"], tmp_path)
    row = _rows(tmp_path / "quality.jsonl")[0]
    assert row["n"] == 12 and row["error_rate"] == 0 and 0 <= row["accuracy"] <= 1
    assert len(_rows(tmp_path / "pred-fake-massive-choice-xx.jsonl")) == 12


def test_invariance_fake_is_invariant_and_biased_is_not(fake, biased, tiny_suite, tmp_path):
    runners.run_invariance("fake", ["massive-choice-xx"], 5, tmp_path)
    runners.run_invariance("fake-biased", ["massive-choice-xx"], 5, tmp_path)
    clean, bad = _rows(tmp_path / "invariance.jsonl")
    assert clean["full_permutation_flip_rate"] == 0 and clean["max_prob_dev"] == 0
    assert bad["full_permutation_flip_rate"] > 0


def test_interference_fake_is_zero(fake, tmp_path, monkeypatch):
    rows = [{"text": f"news {i}", "label": i % 4} for i in range(20)]
    data.write_jsonl(
        tmp_path / "agnews-choice.jsonl",
        data.build_suite(data.Suite("agnews-choice", "agnews", "choice", "en", "unused"), rows, n=6),
    )
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    runners.run_interference("fake", n=6, out=tmp_path)
    for row in _rows(tmp_path / "interference.jsonl"):
        assert row["argmax_change_rate"] == 0 and row["max_prob_dev"] == 0


@pytest.mark.network
def test_burst_primary_cell_rows(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(runners, "MEASURED", 30)
    env.record(tmp_path)
    runners.run_burst(["fake"], "primary", "cold", tmp_path)
    rows = _rows(tmp_path / "latency.jsonl")
    assert {r["layer"] for r in rows} == {"http", "runtime"}
    for r in rows:
        assert all(f in r for f in SPEC_FIELDS)
        assert r["n"] == 30 and r["state_tokens"] == 512 and r["options"] == 4


@pytest.mark.network
def test_cold_requests_have_unique_states():
    states = {runners._request_for((64, 1, 2), "cold", i)["state"] for i in range(10)}
    warm = {runners._request_for((64, 1, 2), "warm", i)["state"] for i in range(10)}
    assert len(states) == 10 and len(warm) == 1


def test_from_jev_maps_dialect_onto_protocol_v1():
    from krite_bench.client import _validator, from_jev

    jev = {
        "model": "m",
        "answers": {
            "s": {
                "type": "score",
                "score": 0.5,
                "legend": {"0": "lo", "1": "hi"},
                "probabilities": {"0": 0.5, "1": 0.5},
                "confidence": 0.0,
            },
            "n": {"type": "noul", "noul": 0.7, "confidence": 0.7},
        },
        "usage": {"input_tokens": 3, "output_tokens": 9},
        "routing": {"model": "english"},
    }
    out = from_jev(jev)
    assert _validator().is_valid(out)
    assert out["answers"]["s"]["probabilities"] == {"lo": 0.5, "hi": 0.5} and out["latency_ms"] == 0


def test_invariance_skips_identity_and_non_choice_suites(fake, tiny_suite, tmp_path):
    data.write_jsonl(tiny_suite / "boolq-noul.jsonl", [{"type": "noul"}])
    runners.run_invariance("fake", ["boolq-noul", "massive-choice-xx"], 5, tmp_path)
    (row,) = _rows(tmp_path / "invariance.jsonl")
    # K=4: one base call plus the 23 non-identity orders (the reverse is one of them) per case.
    assert row["suite"] == "massive-choice-xx" and row["cases"] == 5 and row["calls"] == 5 * 24


@pytest.mark.network
def test_cold_states_differ_across_cells():
    a = runners._request_for((64, 1, 2), "cold", 0)["state"]
    b = runners._request_for((64, 4, 2), "cold", 0)["state"]
    assert a != b


def test_latency_row_with_no_successful_calls():
    row = runners.latency_row("fake", "http", "cold", "burst", (64, 1, 2), [], 0, 120_000.0, None)
    assert row["p50_ms"] is None and row["decisions_per_sec"] is None and row["n"] == 0


def test_throughput_counts_failed_call_time():
    # One 1 ms success plus one 120 s timeout is ~0.0083 decisions/sec, not 1000.
    row = runners.latency_row("fake", "http", "cold", "burst", (64, 1, 2), [1.0], 1, 120_001.0, None)
    assert abs(row["decisions_per_sec"] - 1 / 120.001) < 1e-9


def test_answer_mismatch():
    from krite_bench.client import answer_mismatch

    q = {"q": {"type": "choice", "instructions": "i", "criteria": {"a": None, "b": None}}}
    assert answer_mismatch(q, {"q": {"type": "noul", "noul": 0.5}})
    assert answer_mismatch(q, {"q": {"type": "choice", "probabilities": {"a": 1.0}}})
    assert answer_mismatch(q, {"q": {"type": "choice", "choice": "b", "probabilities": {"a": 0.4, "b": 0.6}}}) is None
    # The returned choice must be an offered, top-probability candidate.
    assert answer_mismatch(q, {"q": {"type": "choice", "choice": "zzz", "probabilities": {"a": 0.99, "b": 0.01}}})
    assert answer_mismatch(q, {"q": {"type": "choice", "choice": "b", "probabilities": {"a": 0.99, "b": 0.01}}})
    assert answer_mismatch(q, {"q": {"type": "choice", "choice": "b", "probabilities": {"a": 0.5, "b": 0.5}}}) is None
