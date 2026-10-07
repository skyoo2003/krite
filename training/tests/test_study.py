import json

import pytest
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


def baseline(tmp_path) -> dict[str, float]:
    """Laya acc 0.80 / qwk 0.60, cbjev acc 0.70 / qwk 0.70 on every suite → targets 0.78 / 0.68."""
    base = tmp_path / "baselines"
    base.mkdir()
    rows = []
    for engine, acc, qwk in (("laya", 0.80, 0.60), ("cbjev", 0.70, 0.70)):
        for s in suites():
            metric = {"qwk": qwk} if s.type == "score" else {"accuracy": acc}
            rows.append({"engine": engine, "suite": s.id, "error_rate": 0.0, **metric})
    write(base / "quality.jsonl", rows)
    return study.targets(base)


def engine_rows(engine: str, acc: float, qwk: float, in_domain: float | None = None) -> list[dict]:
    rows = []
    for s in suites():
        a = in_domain if in_domain is not None and s.dataset in study.IN_DOMAIN else acc
        metric = {"qwk": qwk} if s.type == "score" else {"accuracy": a}
        rows.append({"engine": engine, "suite": s.id, "error_rate": 0.0, **metric})
    return rows


def invariance_rows(engines: list[str], flip: float = 0.0) -> tuple[list[dict], list[dict]]:
    return (
        [{"engine": e, "full_permutation_flip_rate": flip, "max_prob_dev": 0.0} for e in engines],
        [{"engine": e, "max_prob_dev": 1e-8} for e in engines],
    )


def test_targets_take_best_baseline_minus_margin(tmp_path):
    tg = baseline(tmp_path)
    for s in suites():
        assert abs(tg[s.id] - (0.68 if s.type == "score" else 0.78)) < 1e-12


def test_targets_need_baselines(tmp_path):
    (tmp_path / "quality.jsonl").write_text("")
    with pytest.raises(SystemExit):
        study.targets(tmp_path)


def test_shortfall_sums_only_misses(tmp_path):
    tg = baseline(tmp_path)
    write(tmp_path / "quality.jsonl", engine_rows("x", 0.79, 0.60))
    total, per = study.shortfall(tmp_path, "x", tg)
    n_score = sum(s.type == "score" for s in suites())
    assert abs(total - 0.08 * n_score) < 1e-9
    assert all(v == 0.0 for s, v in per.items() if not s.startswith(("sst5", "amazon")))
    write(tmp_path / "quality.jsonl", engine_rows("x", 0.79, 0.60)[1:])
    with pytest.raises(RuntimeError):
        study.shortfall(tmp_path, "x", tg)


def stage_results(tmp_path, new: tuple[float, float], in_domain: float | None = None, flip: float = 0.0) -> None:
    """Base pair at acc 0.70 / 0.68; new pair at the given accuracies (score suites at target)."""
    rows = engine_rows("b13", 0.70, 0.68) + engine_rows("b14", 0.68, 0.68)
    rows += engine_rows("n13", new[0], 0.68, in_domain) + engine_rows("n14", new[1], 0.68, in_domain)
    write(tmp_path / "quality.jsonl", rows)
    inv, itf = invariance_rows(["n13", "n14"], flip)
    write(tmp_path / "invariance.jsonl", inv)
    write(tmp_path / "interference.jsonl", itf)


def test_stage_rule(tmp_path):
    tg = baseline(tmp_path)
    run = lambda: study.stage(tmp_path, ["b13", "b14"], ["n13", "n14"], tg)  # noqa: E731
    stage_results(tmp_path, (0.72, 0.70))
    v = run()
    assert v["adopt"], v["rules"]
    stage_results(tmp_path, (0.7025, 0.6825))  # mean gain 0.0025 x 21 suites = 0.0525
    assert not run()["adopt"]
    stage_results(tmp_path, (0.76, 0.675))  # big mean gain, seed 14 worse
    v = run()
    assert not v["adopt"] and v["rules"]["seed agreement"]["pass"] is False
    stage_results(tmp_path, (0.75, 0.73), in_domain=0.67)  # in-domain drop 0.02
    assert not run()["adopt"]
    stage_results(tmp_path, (0.72, 0.70), flip=0.01)
    assert not run()["adopt"]


def release_results(tmp_path, warm: float = 7.8, acc: float = 0.79, calibration: bool = True) -> None:
    write(tmp_path / "quality.jsonl", engine_rows("k", acc, 0.70))
    inv, itf = invariance_rows(["k"])
    write(tmp_path / "invariance.jsonl", inv)
    write(tmp_path / "interference.jsonl", itf)
    cells = [("warm", 1, warm), ("cold", 1, 91.0), ("warm", 30, 160.0)]
    write(
        tmp_path / "latency.jsonl",
        [
            {
                "engine": "k",
                "layer": "http",
                "cache_state": c,
                "mode": "burst",
                "state_tokens": 512,
                "questions": q,
                "options": 4,
                "p50_ms": p,
            }
            for c, q, p in cells
        ],
    )
    cal = [{"engine": "k-raw", "suite": s.id, "scaled_eval_half": {"ece": 0.05}} for s in suites()]
    write(tmp_path / "calibration.jsonl", cal if calibration else [])
    for e in ("k", "k-nocache"):
        for sid in study.CACHE_SUITES:
            write(tmp_path / f"pred-{e}-{sid}.jsonl", [{"case_id": "c1", "error": None, "probs": [0.25, 0.75]}])


def test_release_gate(tmp_path):
    tg = baseline(tmp_path)
    run = lambda: study.release(tmp_path, "k", "k-raw", "k-nocache", tg)  # noqa: E731
    release_results(tmp_path)
    v = run()
    assert v["release"], v["gates"]
    release_results(tmp_path, warm=10.5)
    v = run()
    assert not v["release"] and not v["gates"]["warm latency (ms)"]["pass"]
    release_results(tmp_path, acc=0.77)
    assert not run()["release"]
    release_results(tmp_path, calibration=False)
    v = run()
    assert not v["release"] and v["gates"]["ECE"]["value"] is None


def test_feasibility_reads_only_latency(tmp_path):
    release_results(tmp_path)
    v = study.feasibility(tmp_path, "k")
    assert v["feasible"] and set(v["gates"]) == {
        "warm latency (ms)",
        "cold latency (ms)",
        "decisions/sec (30 questions)",
    }
    release_results(tmp_path, warm=10.5)
    v = study.feasibility(tmp_path, "k")
    assert not v["feasible"] and not v["gates"]["warm latency (ms)"]["pass"]
    v = study.feasibility(tmp_path, "untimed")
    assert not v["feasible"] and v["gates"]["cold latency (ms)"]["value"] is None
