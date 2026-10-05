"""Architecture-study verdict from krite-bench results (pre-registered rule; docs/architecture-study.md).

uv run python -m krite_train.study compare --a arch-d2 --b arch-d2-nocache
uv run python -m krite_train.study verdict
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["compare", "verdict"])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--a", default="arch-d2")
    ap.add_argument("--b", default="arch-d2-nocache")
    a = ap.parse_args()
    if a.cmd == "compare":
        dev = compare(a.out, a.a, a.b)
        print(f"max |Δp| {a.a} vs {a.b}: {dev:.3g}")
        if dev > TOL:
            raise SystemExit(f"FAIL: above {TOL}")
    else:
        print(json.dumps(verdict(a.out), indent=2))


if __name__ == "__main__":
    main()
