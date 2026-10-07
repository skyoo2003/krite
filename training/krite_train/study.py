"""Pre-registered rules over krite-bench results: the architecture verdict (docs/architecture-study.md),
and the release recipe's stage rule and release gate (docs/training-data.md).

uv run python -m krite_train.study compare --a arch-d2 --b arch-d2-nocache
uv run python -m krite_train.study verdict
uv run python -m krite_train.study stage --base arch-late8,arch-late8-s14 --new arch-late8-broad,arch-late8-broad-s14
uv run python -m krite_train.study release --engine krite-v1 --raw krite-v1-raw --nocache krite-v1-nocache
uv run python -m krite_train.study feasibility --engine krite-base-pilot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from krite_bench.data import ROOT, suites

OUT = ROOT / "benchmarks" / "results" / "arch"
B = "arch-b"
D_ARMS = (
    "arch-d1",
    "arch-d2",
    "arch-d4",
    "arch-d2-emb",
    "arch-d1-pool",
    "arch-d1-lr",
    "arch-d1-set",
    "arch-late4",
    "arch-late8",
    "arch-late6",
)
SEED_14 = {
    "arch-b": "arch-b-s14",
    "arch-late4": "arch-late4-s14",
    "arch-late8": "arch-late8-s14",
}  # robustness check, not part of the rule
IN_DOMAIN = {"banking77", "massive", "boolq"}  # datasets whose train splits are in the mixture
TOL = 1e-5
CACHE_SUITES = ("agnews-choice", "boolq-noul", "massive-choice-ko", "amazon-score-ja")
PRIMARY = (512, 1, 4)
# Rule name prefix -> largest passing value.
LIMITS = {"Q1": 0.02, "Q2": 0.05, "L1": 1.0, "I1 flip": 0.0, "I": TOL}


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def _latest(rows: list[dict], *keys: str) -> dict[tuple, dict]:
    return {tuple(r.get(k) for k in keys): r for r in rows}


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def compare(out: Path, a: str, b: str, suite_ids=CACHE_SUITES) -> float:
    """Max |Δ probability| between two engines' predictions on the same cases."""
    worst = 0.0
    for sid in suite_ids:
        ra, rb = (_rows(out / f"pred-{e}-{sid}.jsonl") for e in (a, b))
        if not ra or [x["case_id"] for x in ra] != [x["case_id"] for x in rb]:
            raise RuntimeError(f"{sid}: pred files of {a} and {b} missing or not aligned")
        for x, y in zip(ra, rb, strict=True):
            if x["error"] or y["error"]:
                raise RuntimeError(f"{sid}/{x['case_id']}: error in a compared call")
            worst = max(worst, max(abs(p - q) for p, q in zip(x["probs"], y["probs"], strict=True)))
    return worst


def group_means(out: Path, engine: str) -> dict[str, float | None]:
    """Mean accuracy (choice + noul) per in-domain / held-out group, and mean QWK over score suites."""
    q = _latest(_rows(out / "quality.jsonl"), "engine", "suite")
    acc: dict[str, list[float]] = {"in_domain": [], "held_out": []}
    qwk: list[float] = []
    for s in suites():
        r = q.get((engine, s.id))
        if r is None:
            continue
        if r["error_rate"]:
            raise RuntimeError(f"{engine} {s.id}: error_rate {r['error_rate']}")
        if s.type == "score":
            qwk.append(r["qwk"])
        else:
            acc["in_domain" if s.dataset in IN_DOMAIN else "held_out"].append(r["accuracy"])
    return {"acc_in_domain": _mean(acc["in_domain"]), "acc_held_out": _mean(acc["held_out"]), "qwk": _mean(qwk)}


def warm_p50(out: Path, engine: str) -> float | None:
    keys = ("engine", "layer", "cache_state", "mode", "state_tokens", "questions", "options")
    r = _latest(_rows(out / "latency.jsonl"), *keys).get((engine, "runtime", "warm", "burst", *PRIMARY))
    return r["p50_ms"] if r else None


def invariants(out: Path, engine: str) -> dict[str, float]:
    inv = [r for r in _rows(out / "invariance.jsonl") if r["engine"] == engine]
    itf = [r for r in _rows(out / "interference.jsonl") if r["engine"] == engine]
    if not inv or not itf:
        raise RuntimeError(f"{engine}: invariance or interference results missing")
    return {
        "flip_rate": max(r["full_permutation_flip_rate"] for r in inv),
        "invariance_max_dev": max(r["max_prob_dev"] for r in inv),
        "interference_max_dev": max(r["max_prob_dev"] for r in itf),
    }


def passes(name: str, value: float | None) -> bool:
    cap = next(v for k, v in LIMITS.items() if name.startswith(k))
    return value is not None and value <= cap


def verdict(out: Path) -> dict:
    arms = {e: {**group_means(out, e), "warm_p50_ms": warm_p50(out, e)} for e in (B, *D_ARMS)}
    ds = [e for e in D_ARMS if arms[e]["acc_held_out"] is not None and arms[e]["warm_p50_ms"] is not None]
    if not ds or None in arms[B].values():
        raise SystemExit(f"incomplete results in {out}: need {B} and at least one tower arm (quality + latency)")
    top = max(arms[e]["acc_held_out"] for e in ds)
    best = min((e for e in ds if top - arms[e]["acc_held_out"] <= 0.005), key=lambda e: arms[e]["warm_p50_ms"])
    b, d = arms[B], arms[best]
    inv = invariants(out, best)
    # I3 on D-best's own cache when its no-cache twin was measured, else on the d2 pair.
    cached = best if (out / f"pred-{best}-nocache-{CACHE_SUITES[0]}.jsonl").exists() else "arch-d2"
    values = {
        "Q1 in-domain accuracy gap (B - D)": b["acc_in_domain"] - d["acc_in_domain"],
        "Q1 held-out accuracy gap (B - D)": b["acc_held_out"] - d["acc_held_out"],
        "Q2 score QWK gap (B - D)": b["qwk"] - d["qwk"],
        "L1 D warm p50 x 5 / B warm p50": d["warm_p50_ms"] * 5 / b["warm_p50_ms"],
        "I1 flip rate": inv["flip_rate"],
        "I1 invariance max dev": inv["invariance_max_dev"],
        "I2 interference max dev": inv["interference_max_dev"],
        f"I3 cache on/off max dev ({cached})": compare(out, cached, f"{cached}-nocache"),
    }
    rules = {k: {"value": v, "pass": passes(k, v)} for k, v in values.items()}
    q1 = [k for k in values if k.startswith("Q1")]
    q1_miss = max(values[k] for k in q1) - LIMITS["Q1"]
    return {
        "arms": arms,
        "d_best": best,
        "rules": rules,
        "accept_d": all(r["pass"] for r in rules.values()),
        "rerun_seed_14": 0 < q1_miss <= 0.01,
        "seed_14": {e: group_means(out, e) for e in SEED_14.values()},
    }


# --- release recipe (docs/training-data.md): rules fixed before training ---------------------------

BASELINES = ROOT / "benchmarks" / "results" / "baselines"
# Best zero-shot engines <= 0.45B (docs/baselines.md); Laya ran non-English suites on its multilingual weights.
TARGET_ENGINES = ("laya", "laya-ml", "cbjev")
MARGIN = 0.02
STAGE_GAIN, IN_DOMAIN_DROP = 0.10, 0.01
# Release gate: PRD Key Hypothesis, docs/baselines.md "Derived Krite targets".
RELEASE = {"ece": 0.071, "warm_ms": 10.0, "cold_ms": 210.0, "dec_per_s": 175.0}


def _value(r: dict, qtype: str) -> float:
    return r["qwk"] if qtype == "score" else r["accuracy"]


def targets(baselines: Path = BASELINES) -> dict[str, float]:
    """Per-suite target: best of TARGET_ENGINES minus MARGIN (latest quality row per engine and suite)."""
    q = _latest(_rows(baselines / "quality.jsonl"), "engine", "suite")
    out = {}
    for s in suites():
        vals = [_value(q[(e, s.id)], s.type) for e in TARGET_ENGINES if (e, s.id) in q]
        if not vals:
            raise SystemExit(f"{s.id}: no baseline quality rows for {TARGET_ENGINES} in {baselines}")
        out[s.id] = max(vals) - MARGIN
    return out


def values(out: Path, engine: str) -> dict[str, float]:
    """Accuracy (choice, noul) or QWK (score) per suite; an unmeasured suite raises, it is not a pass."""
    q = _latest(_rows(out / "quality.jsonl"), "engine", "suite")
    vals = {}
    for s in suites():
        r = q.get((engine, s.id))
        if r is None:
            raise RuntimeError(f"{engine} {s.id}: no quality row")
        if r["error_rate"]:
            raise RuntimeError(f"{engine} {s.id}: error_rate {r['error_rate']}")
        vals[s.id] = _value(r, s.type)
    return vals


def shortfall(out: Path, engine: str, tg: dict[str, float]) -> tuple[float, dict[str, float]]:
    """(sum, per-suite) of max(0, target - value)."""
    per = {sid: max(0.0, tg[sid] - v) for sid, v in values(out, engine).items()}
    return sum(per.values()), per


# Release accuracy (revised after the recipe search, docs/training-data.md "Targets"): whole engines,
# not a per-suite best that no engine reaches. Laya's non-English rows come from its multilingual weights.
REFERENCE_ENGINES = {"laya": ("laya", "laya-ml"), "cbjev": ("cbjev",)}


def macro(vals: dict[str, float]) -> dict[str, float]:
    """Mean over datasets of the per-dataset mean: accuracy over choice and noul, QWK over score."""
    by: dict[tuple[bool, str], list[float]] = {}
    for s in suites():
        by.setdefault((s.type == "score", s.dataset), []).append(vals[s.id])
    means = {k: sum(v) / len(v) for k, v in by.items()}
    return {
        "accuracy": _mean([m for (score, _), m in means.items() if not score]),
        "qwk": _mean([m for (score, _), m in means.items() if score]),
    }


def reference(baselines: Path = BASELINES) -> dict:
    """Per macro metric, the best whole reference engine minus MARGIN."""
    q = _latest(_rows(baselines / "quality.jsonl"), "engine", "suite")
    engines = {}
    for name, rows in REFERENCE_ENGINES.items():
        vals = {}
        for s in suites():
            r = next((q[(e, s.id)] for e in rows if (e, s.id) in q), None)
            if r is None:
                raise SystemExit(f"{name} {s.id}: no baseline quality row in {baselines}")
            vals[s.id] = _value(r, s.type)
        engines[name] = macro(vals)
    return {
        "engines": engines,
        "targets": {m: max(e[m] for e in engines.values()) - MARGIN for m in ("accuracy", "qwk")},
    }


def _invariant_rows(out: Path, engines: list[str]) -> dict[str, dict]:
    rows = {}
    for e in engines:
        inv = invariants(out, e)
        rows[f"I1 flip rate ({e})"] = {"value": inv["flip_rate"], "pass": inv["flip_rate"] == 0}
        for k in ("invariance_max_dev", "interference_max_dev"):
            rows[f"{k} ({e})"] = {"value": inv[k], "pass": inv[k] <= TOL}
    return rows


def stage(out: Path, base: list[str], new: list[str], tg: dict[str, float]) -> dict:
    """Stage rule on paired seeds (base[i] and new[i] share a seed); invariants on every new engine."""
    if len(base) != len(new):
        raise SystemExit("--base and --new need one engine per seed, in the same seed order")
    sb = {e: shortfall(out, e, tg)[0] for e in base}
    sn = {e: shortfall(out, e, tg)[0] for e in new}
    gain = _mean(list(sb.values())) - _mean(list(sn.values()))
    worse = [n for b, n in zip(base, new, strict=True) if sn[n] >= sb[b]]
    drop = _mean([group_means(out, e)["acc_in_domain"] for e in base]) - _mean(
        [group_means(out, e)["acc_in_domain"] for e in new]
    )
    rules = {
        "shortfall gain": {"value": gain, "pass": gain >= STAGE_GAIN},
        "seed agreement": {"value": worse, "pass": not worse},
        "in-domain drop": {"value": drop, "pass": drop <= IN_DOMAIN_DROP},
        **_invariant_rows(out, new),
    }
    return {"base": sb, "new": sn, "rules": rules, "adopt": all(r["pass"] for r in rules.values())}


def latency_p50(out: Path, engine: str, cache_state: str, questions: int) -> float | None:
    """http burst p50 at 512 tokens, 4 options."""
    keys = ("engine", "layer", "cache_state", "mode", "state_tokens", "questions", "options")
    r = _latest(_rows(out / "latency.jsonl"), *keys).get((engine, "http", cache_state, "burst", 512, questions, 4))
    return r["p50_ms"] if r else None


def mean_scaled_ece(out: Path, raw: str) -> float | None:
    """Mean over suites of the temperature-scaled ECE on the half not used for the fit."""
    rows = _latest(_rows(out / "calibration.jsonl"), "engine", "suite")
    return _mean([r["scaled_eval_half"]["ece"] for (e, _), r in rows.items() if e == raw])


def _at_most(v, cap):
    return {"value": v, "limit": cap, "pass": v is not None and v <= cap}


def _at_least(v, floor):
    return {"value": v, "limit": floor, "pass": v is not None and v >= floor}


def latency_gates(out: Path, engine: str) -> dict:
    """The release gate's latency rows; they depend on shapes, not trained weights."""
    warm30 = latency_p50(out, engine, "warm", 30)
    return {
        "warm latency (ms)": _at_most(latency_p50(out, engine, "warm", 1), RELEASE["warm_ms"]),
        "cold latency (ms)": _at_most(latency_p50(out, engine, "cold", 1), RELEASE["cold_ms"]),
        "decisions/sec (30 questions)": _at_least(30_000 / warm30 if warm30 else None, RELEASE["dec_per_s"]),
    }


def feasibility(out: Path, engine: str) -> dict:
    """Latency rows of the release gate on an untrained export, before any full training run."""
    gates = latency_gates(out, engine)
    return {"engine": engine, "gates": gates, "feasible": all(g["pass"] for g in gates.values())}


def release(out: Path, engine: str, raw: str, nocache: str, tg: dict[str, float], ref: dict) -> dict:
    """Release gate. Missing data fails a gate, never passes it. The per-suite shortfall against the
    best baseline per suite is reported, not gated."""
    _, per = shortfall(out, engine, tg)
    m = macro(values(out, engine))
    inv = invariants(out, engine)
    gates = {
        "accuracy (macro, choice and noul)": _at_least(m["accuracy"], ref["targets"]["accuracy"]),
        "QWK (macro, score)": _at_least(m["qwk"], ref["targets"]["qwk"]),
        "ECE": _at_most(mean_scaled_ece(out, raw), RELEASE["ece"]),
        **latency_gates(out, engine),
        "I1 flip rate": _at_most(inv["flip_rate"], 0.0),
        "I1 invariance max dev": _at_most(inv["invariance_max_dev"], TOL),
        "I2 interference max dev": _at_most(inv["interference_max_dev"], TOL),
        "I3 cache on/off max dev": _at_most(compare(out, engine, nocache), TOL),
    }
    return {
        "engine": engine,
        "gates": gates,
        "shortfall": {s: v for s, v in per.items() if v > 0},
        "release": all(g["pass"] for g in gates.values()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["compare", "verdict", "stage", "release", "feasibility"])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--a", default="arch-d2")
    ap.add_argument("--b", default="arch-d2-nocache")
    ap.add_argument("--tol", type=float, default=TOL, help="compare: max |Δp| allowed")
    ap.add_argument("--baselines", type=Path, default=BASELINES, help="stage/release: baseline results directory")
    ap.add_argument("--base", help="stage: comma-separated engines, one per seed")
    ap.add_argument("--new", help="stage: comma-separated engines, same seed order as --base")
    ap.add_argument("--engine", default="krite-v1", help="release: calibrated engine; feasibility: engine to time")
    ap.add_argument("--raw", default="krite-v1-raw", help="release: engine the temperatures were fitted on")
    ap.add_argument("--nocache", default="krite-v1-nocache", help="release: the engine with both caches off")
    a = ap.parse_args()
    if a.cmd == "compare":
        dev = compare(a.out, a.a, a.b)
        print(f"max |Δp| {a.a} vs {a.b}: {dev:.3g}")
        if dev > a.tol:
            raise SystemExit(f"FAIL: above {a.tol}")
    elif a.cmd == "verdict":
        print(json.dumps(verdict(a.out), indent=2))
    elif a.cmd == "stage":
        if not a.base or not a.new:
            raise SystemExit("stage needs --base and --new")
        print(json.dumps(stage(a.out, a.base.split(","), a.new.split(","), targets(a.baselines)), indent=2))
    elif a.cmd == "feasibility":
        print(json.dumps(feasibility(a.out, a.engine), indent=2))
    else:
        ref = reference(a.baselines)
        v = release(a.out, a.engine, a.raw, a.nocache, targets(a.baselines), ref)
        print(json.dumps({**v, "reference": ref["engines"]}, indent=2))


if __name__ == "__main__":
    main()
