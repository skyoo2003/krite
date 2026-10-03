"""Quality, invariance, interference, latency, and memory runners.

Every runner talks to engines only through client.Engine (POST /v1/systemone). Failed calls
become error rows that stay in the denominator (`error_rate`); limit rejections are counted
separately as `not_supported_rate`.
"""

from __future__ import annotations

import itertools
import json
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from . import metrics
from .client import Engine, engine_config
from .data import SEED, interference_request, latency_request, load_suite, suites
from .env import commit, latest, now, power_source, therm

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "baselines"
WARMUP, MEASURED = 20, 200
BUDGET_S = 15 * 60
CELLS = {
    "primary": [(512, 1, 4)],
    "aux": [(512, 1, 4), (512, 10, 4), (512, 30, 4)],
    "matrix": [(s, q, k) for s in (64, 512, 2048) for q in (1, 4, 16, 64) for k in (2, 4, 8)],
}


def append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def base_row(engine: str) -> dict:
    cfg = engine_config(engine)
    return {
        "engine": engine,
        "model": cfg.get("model", engine),
        "backend": cfg.get("backend", "unknown"),
        "precision": cfg.get("precision", "unknown"),
    }


def names_of(question: dict) -> list[str]:
    if question["type"] == "noul":
        return ["true", "false"]
    return list(question["criteria"])


def distribution(question: dict, answer: dict) -> list[float]:
    """Answer probabilities in request candidate order (noul: [P(true), P(false)])."""
    if question["type"] == "noul":
        return [answer["noul"], 1.0 - answer["noul"]]
    return [answer["probabilities"][n] for n in names_of(question)]


def gold_index(question: dict, gold) -> int:
    if question["type"] == "noul":
        return 0 if gold else 1
    if question["type"] == "score":
        return int(gold)
    return names_of(question).index(gold)


# --- quality -------------------------------------------------------------------------------------


def run_quality(engine: str, suite_ids: list[str], out: Path = RESULTS_DIR, sample_memory: bool = False) -> None:
    eng = Engine.from_config(engine)
    sampler = MemorySampler(engine) if sample_memory else None
    if sampler:
        sampler.start()
    try:
        for sid in suite_ids:
            cases = load_suite(sid)
            preds, errors, unsupported = [], 0, 0
            pred_path = out / f"pred-{engine}-{sid}.jsonl"
            pred_path.unlink(missing_ok=True)
            for i, case in enumerate(cases):
                q = case["request"]["questions"]["q"]
                r = eng.call(case["request"], case["dataset"])
                row = {"case_id": case["case_id"], "http_ms": r.http_ms, "error": r.error}
                if r.ok:
                    p = distribution(q, r.response["answers"]["q"])
                    g = gold_index(q, case["gold"]["q"])
                    preds.append({"probs": p, "gold": g, "names": names_of(q), "bucket": case["bucket"], "index": i})
                    row.update(probs=p, gold=g)
                else:
                    unsupported += r.not_supported
                    errors += not r.not_supported
                append(pred_path, row)
            summary = metrics.summarize(cases[0]["type"], preds) if preds else {}
            append(
                out / "quality.jsonl",
                {
                    **base_row(engine),
                    "suite": sid,
                    "lang": cases[0]["lang"],
                    "type": cases[0]["type"],
                    "n": len(cases),
                    "error_rate": errors / len(cases),
                    "not_supported_rate": unsupported / len(cases),
                    **summary,
                    "commit": commit(),
                    "timestamp": now(),
                },
            )
            print(f"{engine} {sid}: acc={summary.get('accuracy', float('nan')):.3f} errors={errors}", flush=True)
    finally:
        eng.close()
        if sampler:
            sampler.stop(out)
    run_calibration(engine, out)


def run_calibration(engine: str, out: Path = RESULTS_DIR) -> None:
    """Refit the engine's calibrator from every pred file it has and append one row per suite.

    One temperature per calibration bucket per model: pooled over all suites of the engine, except for
    engines that load a model per dataset (`request_model`, the classifier), which get one per dataset.
    """
    per_dataset = "request_model" in engine_config(engine)
    groups: dict[str, dict[str, tuple[list[dict], int]]] = {}
    for sid in [s.id for s in suites()]:
        pred_path = out / f"pred-{engine}-{sid}.jsonl"
        if not pred_path.exists():
            continue
        cases = load_suite(sid)
        rows = [json.loads(line) for line in pred_path.read_text().splitlines() if line.strip()]
        if [r["case_id"] for r in rows] != [c["case_id"] for c in cases]:
            raise RuntimeError(f"{pred_path} does not match suite {sid}; rerun quality")
        preds = [
            {
                "probs": r["probs"],
                "gold": r["gold"],
                "names": names_of(c["request"]["questions"]["q"]),
                "bucket": c["bucket"],
                "index": i,
            }
            for i, (r, c) in enumerate(zip(rows, cases, strict=True))
            if r.get("error") is None
        ]
        if preds:
            groups.setdefault(cases[0]["dataset"] if per_dataset else "", {})[sid] = (preds, len(cases))
    for group in groups.values():
        for sid, block in metrics.pooled_calibration(group).items():
            append(
                out / "calibration.jsonl",
                {**base_row(engine), "suite": sid, **block, "commit": commit(), "timestamp": now()},
            )


# --- invariance and interference ---------------------------------------------------------------


def subset(cases: list[dict], n: int, seed: int = SEED) -> list[dict]:
    idx = np.random.default_rng(seed).choice(len(cases), size=min(n, len(cases)), replace=False)
    return [cases[int(i)] for i in sorted(idx)]


def permutations(k: int, n_random: int = 24, seed: int = SEED) -> list[tuple[int, ...]]:
    """All permutations when k <= 5, else n_random seeded ones (benchmark-spec §7)."""
    if k <= 5:
        return list(itertools.permutations(range(k)))
    rng = np.random.default_rng(seed)
    return [tuple(int(i) for i in rng.permutation(k)) for _ in range(n_random)]


def with_order(case: dict, order: tuple[int, ...]) -> dict:
    q = case["request"]["questions"]["q"]
    names = list(q["criteria"])
    criteria = {names[i]: q["criteria"][names[i]] for i in order}  # new dict: insertion order is the order
    return {"state": case["request"]["state"], "questions": {"q": {**q, "criteria": criteria}}}


def _probs_by_name(r) -> dict[str, float]:
    return r.response["answers"]["q"]["probabilities"]


def run_invariance(engine: str, suite_ids: list[str], n: int = 100, out: Path = RESULTS_DIR) -> None:
    """Option-order invariance on choice suites (ARCHITECTURE I1); other suite types are skipped.

    The full-permutation flip rate covers every non-identity order (or the seeded sample when K > 5);
    the reverse flip rate reuses the reversed order from that set, or calls it once more when sampled out.
    """
    eng = Engine.from_config(engine)
    try:
        for sid in suite_ids:
            cases = subset(load_suite(sid), n)
            if cases[0]["type"] != "choice":
                print(f"{engine} {sid}: skipped ({cases[0]['type']} suite; invariance covers choice only)", flush=True)
                continue
            flips, rev_flips, devs, errors, calls = [], [], [], 0, 0
            for case in cases:
                k = len(case["request"]["questions"]["q"]["criteria"])
                identity, reverse = tuple(range(k)), tuple(reversed(range(k)))
                orders = [o for o in permutations(k) if o != identity]
                reverse_extra = reverse not in orders
                base = eng.call(with_order(case, identity), case["dataset"])
                calls += 1
                if not base.ok:
                    errors += 1
                    continue
                base_choice, base_p = base.response["answers"]["q"]["choice"], _probs_by_name(base)
                for order in orders + [reverse] * reverse_extra:
                    r = eng.call(with_order(case, order), case["dataset"])
                    calls += 1
                    if not r.ok:
                        errors += 1
                        continue
                    flipped = r.response["answers"]["q"]["choice"] != base_choice
                    if order == reverse:
                        rev_flips.append(flipped)
                    if order in orders:
                        flips.append(flipped)
                    devs.append(metrics.max_prob_dev(base_p, _probs_by_name(r)))
            append(
                out / "invariance.jsonl",
                {
                    **base_row(engine),
                    "suite": sid,
                    "cases": len(cases),
                    "calls": calls,
                    "error_rate": errors / max(calls, 1),
                    "full_permutation_flip_rate": float(np.mean(flips)) if flips else None,
                    "reverse_flip_rate": float(np.mean(rev_flips)) if rev_flips else None,
                    "max_prob_dev": max(devs) if devs else None,
                    "mean_prob_dev": float(np.mean(devs)) if devs else None,
                    "commit": commit(),
                    "timestamp": now(),
                },
            )
            print(f"{engine} {sid}: flip={np.mean(flips) if flips else 'n/a'} errors={errors}", flush=True)
    finally:
        eng.close()


def run_interference(engine: str, suite_id: str = "agnews-choice", n: int = 100, out: Path = RESULTS_DIR) -> None:
    eng = Engine.from_config(engine)
    fillers_set = (3, 15)
    try:
        cases = subset(load_suite(suite_id), n)
        changes = {f: [] for f in fillers_set}
        devs = {f: [] for f in fillers_set}
        errors = dict.fromkeys(fillers_set, 0)
        for case in cases:
            alone = eng.call(interference_request(case, 0), case["dataset"])  # one alone call per case
            for f in fillers_set:
                mixed = eng.call(interference_request(case, f), case["dataset"])
                if not (alone.ok and mixed.ok):
                    errors[f] += 1
                    continue
                a, m = alone.response["answers"]["q"], mixed.response["answers"]["q"]
                changes[f].append(a["choice"] != m["choice"])
                devs[f].append(metrics.max_prob_dev(_probs_by_name(alone), _probs_by_name(mixed)))
        for fillers in fillers_set:
            append(
                out / "interference.jsonl",
                {
                    **base_row(engine),
                    "suite": suite_id,
                    "fillers": fillers,
                    "cases": len(cases),
                    "error_rate": errors[fillers] / len(cases),
                    "argmax_change_rate": float(np.mean(changes[fillers])) if changes[fillers] else None,
                    "max_prob_dev": max(devs[fillers]) if devs[fillers] else None,
                    "commit": commit(),
                    "timestamp": now(),
                },
            )
    finally:
        eng.close()


# --- latency ------------------------------------------------------------------------------------


def percentiles(ms: list[float]) -> dict:
    """None when no call succeeded (the row's error_rate is then 1.0)."""
    return {f"p{q}_ms": float(np.percentile(ms, q)) if ms else None for q in (50, 95, 99)}


def latency_row(engine, layer, cache_state, mode, cell, times_ms, ok_calls, total_ms, env_ref, extra=None) -> dict:
    """`decisions_per_sec` = answered questions / time of every measured call, failed calls included."""
    s, q, k = cell
    return {
        **base_row(engine),
        "layer": layer,
        "cache_state": cache_state,
        "mode": mode,
        "state_tokens": s,
        "questions": q,
        "options": k,
        "n": len(times_ms),
        **percentiles(times_ms),
        "decisions_per_sec": q * ok_calls / (total_ms / 1000) if ok_calls else None,
        "env_ref": env_ref,
        "commit": commit(),
        "timestamp": now(),
        **(extra or {}),
    }


def _request_for(cell, cache_state: str, i: int) -> dict:
    s, q, k = cell
    # cold: a state unique per (cell, iteration), so no earlier cell of the same run can warm it.
    # Across runs the states repeat: restart engines before a cold run.
    seed = [s, q, k, 10_000 + i] if cache_state == "cold" else 10_000
    return latency_request(s, q, k, seed)


def run_burst(engines: list[str], cells: str, cache_state: str, out: Path = RESULTS_DIR) -> None:
    """Burst mode: WARMUP + MEASURED calls per cell, interleaved per call across engines."""
    env_ref = latest(out)
    clients = {e: Engine.from_config(e) for e in engines}
    try:
        for cell in CELLS[cells]:
            therm_before = therm()
            live = dict(clients)
            warm_ms: dict[str, list[float]] = {e: [] for e in engines}
            for i in range(WARMUP):
                req = _request_for(cell, cache_state, i)
                for e, c in list(live.items()):
                    r = c.call(req)
                    if r.not_supported or not r.ok:
                        _status_row(out, e, cell, cache_state, r)
                        del live[e]
                    else:
                        warm_ms[e].append(r.http_ms)
            n = MEASURED
            if any(statistics.median(warm_ms[e]) * (WARMUP + MEASURED) / 1000 > BUDGET_S for e in live):
                n = 50
            http: dict[str, list[float]] = {e: [] for e in live}
            runtime: dict[str, list[float]] = {e: [] for e in live}
            errors = dict.fromkeys(live, 0)
            total_ms = dict.fromkeys(live, 0.0)
            for i in range(WARMUP, WARMUP + n):
                req = _request_for(cell, cache_state, i)
                for e, c in live.items():
                    r = c.call(req)
                    total_ms[e] += r.http_ms
                    if not r.ok:
                        errors[e] += 1
                        continue
                    http[e].append(r.http_ms)
                    if r.runtime_ms:
                        runtime[e].append(r.runtime_ms)
            extra_common = {
                "power_source": power_source(),
                "truncated": n < MEASURED,
                "interleaved_with": sorted(live),
                "therm_before": therm_before,
                "therm_after": therm(),
            }
            for e in live:
                extra = {**extra_common, "error_rate": errors[e] / n}
                common = (len(http[e]), total_ms[e], env_ref, extra)
                append(out / "latency.jsonl", latency_row(e, "http", cache_state, "burst", cell, http[e], *common))
                if runtime[e]:
                    row = latency_row(e, "runtime", cache_state, "burst", cell, runtime[e], *common)
                    append(out / "latency.jsonl", row)
                p50 = percentiles(http[e])["p50_ms"]
                print(f"{e} {cell} {cache_state}: http p50={p50} ms errors={errors[e]}", flush=True)
    finally:
        for c in clients.values():
            c.close()


def _status_row(out: Path, engine: str, cell, cache_state: str, r) -> None:
    status = "not_supported" if r.not_supported else "error"
    append(
        out / "status.jsonl",
        {
            "engine": engine,
            "status": status,
            "cell": list(cell),
            "cache_state": cache_state,
            "reason": r.error,
            "timestamp": now(),
        },
    )


def run_sustained(engine: str, seconds: int = 600, out: Path = RESULTS_DIR) -> None:
    """Sustained mode: primary cell, warm, for `seconds`; p50 per 60 s window."""
    env_ref = latest(out)
    cell = CELLS["primary"][0]
    eng = Engine.from_config(engine)
    req = _request_for(cell, "warm", 0)
    try:
        eng.call(req)  # prime the state cache
        therm_before = therm()
        windows: list[list[float]] = []
        all_http, errors, total_ms = [], 0, 0.0
        start = time.monotonic()
        while (elapsed := time.monotonic() - start) < seconds:
            w = int(elapsed // 60)
            while len(windows) <= w:
                windows.append([])
            r = eng.call(req)
            total_ms += r.http_ms
            if r.ok:
                windows[w].append(r.http_ms)
                all_http.append(r.http_ms)
            else:
                errors += 1
        window_p50 = [statistics.median(x) if x else None for x in windows]
        extra = {
            "power_source": power_source(),
            "window_p50_ms": window_p50,
            "last_minute_p50_ms": window_p50[-1],
            "error_rate": errors / max(len(all_http) + errors, 1),
            "therm_before": therm_before,
            "therm_after": therm(),
        }
        append(
            out / "latency.jsonl",
            latency_row(engine, "http", "warm", "sustained", cell, all_http, len(all_http), total_ms, env_ref, extra),
        )
    finally:
        eng.close()


def run_startup(engine: str, out: Path = RESULTS_DIR) -> None:
    """Startup: wall time of serve.sh, from process launch until its readiness probe returns 200.

    The probe is a 1-question noul "ping" polled every 0.2 s, not the primary cell, so this is
    process-launch-to-ready time (benchmark-spec §3 startup is approximated, see `probe`).
    Engine must not be running.
    """
    serve = Path(__file__).resolve().parents[1] / "baselines" / "serve.sh"
    t0 = time.perf_counter()
    subprocess.run([str(serve), engine], check=True, capture_output=True, text=True)
    ms = (time.perf_counter() - t0) * 1000
    cell = CELLS["primary"][0]
    extra = {"power_source": power_source(), "probe": "serve.sh noul ping, 0.2 s poll"}
    row = latency_row(engine, "http", "startup", "burst", cell, [ms], 1, ms, latest(out), extra)
    append(out / "latency.jsonl", row)


# --- memory -------------------------------------------------------------------------------------


def listener_pid(port: int) -> int:
    out = subprocess.run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"], capture_output=True, text=True).stdout
    if not out.strip():
        raise RuntimeError(f"nothing listening on port {port}")
    return int(out.split()[0])


def footprint_bytes(pid: int) -> int | None:
    """phys_footprint from macOS `footprint` (includes GPU-backed unified memory)."""
    out = subprocess.run(["footprint", "-p", str(pid)], capture_output=True, text=True).stdout
    m = re.search(r"phys_footprint:\s*([\d.]+)\s*([KMG]?B)", out)
    if not m:
        return None
    scale = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}[m.group(2)]
    return int(float(m.group(1)) * scale)


def rss_bytes(pid: int) -> int | None:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(out) * 1024 if out else None


class MemorySampler:
    """Samples phys_footprint and RSS of the engine's listening process every second."""

    def __init__(self, engine: str):
        self.engine = engine
        self.pid = listener_pid(engine_config(engine)["port"])
        self.peak_footprint = self.peak_rss = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak_footprint = max(self.peak_footprint, footprint_bytes(self.pid) or 0)
            self.peak_rss = max(self.peak_rss, rss_bytes(self.pid) or 0)
            self._stop.wait(1.0)

    def start(self) -> None:
        self._thread.start()

    def stop(self, out: Path) -> None:
        self._stop.set()
        self._thread.join()
        append(
            out / "memory.jsonl",
            {
                **base_row(self.engine),
                "pid": self.pid,
                "peak_phys_footprint_bytes": self.peak_footprint,
                "peak_rss_bytes": self.peak_rss,
                "commit": commit(),
                "timestamp": now(),
            },
        )


def suite_ids(langs: str = "all") -> list[str]:
    """Suites an engine runs: `en` (English weights), `non-en` (multilingual weights), or `all`."""
    if langs == "en":
        return [s.id for s in suites() if s.lang == "en"]
    if langs == "non-en":
        return [s.id for s in suites() if s.lang != "en"]
    return [s.id for s in suites()]
