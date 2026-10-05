import json

from krite_bench.data import suites

from krite_train import study


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def results(tmp_path, d_acc: float, d_warm: float) -> None:
    quality, latency = [], []
    for engine, acc, warm in (("arch-b", 0.80, 200.0), ("arch-d2", d_acc, d_warm), ("arch-d1", d_acc - 0.1, 1.0)):
        for s in suites():
            metric = {"qwk": 0.5} if s.type == "score" else {"accuracy": acc}
            quality.append({"engine": engine, "suite": s.id, "error_rate": 0.0, **metric})
        latency.append(
            {
                "engine": engine,
                "layer": "runtime",
                "cache_state": "warm",
                "mode": "burst",
                "state_tokens": 512,
                "questions": 1,
                "options": 4,
                "p50_ms": warm,
            }
        )
    write(tmp_path / "quality.jsonl", quality)
    write(tmp_path / "latency.jsonl", latency)
    write(tmp_path / "invariance.jsonl", [{"engine": "arch-d2", "full_permutation_flip_rate": 0.0, "max_prob_dev": 0}])
    write(tmp_path / "interference.jsonl", [{"engine": "arch-d2", "max_prob_dev": 1e-8}])
    for e in ("arch-d2", "arch-d2-nocache"):
        for sid in study.CACHE_SUITES:
            write(tmp_path / f"pred-{e}-{sid}.jsonl", [{"case_id": "c1", "error": None, "probs": [0.25, 0.75]}])


def test_verdict_accepts_close_fast_d(tmp_path):
    results(tmp_path, d_acc=0.79, d_warm=10.0)
    v = study.verdict(tmp_path)
    assert v["d_best"] == "arch-d2"
    assert v["accept_d"] and not v["rerun_seed_14"]


def test_verdict_rejects_slow_or_worse_d(tmp_path):
    results(tmp_path, d_acc=0.79, d_warm=50.0)
    assert not study.verdict(tmp_path)["accept_d"]
    results(tmp_path, d_acc=0.775, d_warm=10.0)
    v = study.verdict(tmp_path)
    assert not v["accept_d"] and v["rerun_seed_14"]
    results(tmp_path, d_acc=0.70, d_warm=10.0)
    assert not study.verdict(tmp_path)["rerun_seed_14"]
