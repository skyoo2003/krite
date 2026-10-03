"""Renders measured results into docs/baselines.md.

Only the block between the GENERATED markers is rewritten; hand-written sections (findings,
quoted README numbers, derived targets) around it are preserved. The latest row per key wins.
"""

from __future__ import annotations

import json
from pathlib import Path

from .data import suites

START, END = "<!-- GENERATED:START -->", "<!-- GENERATED:END -->"
ENGINES = ["laya", "kev", "semif", "cbjev", "classifier"]
TITLES = {
    "laya": "Laya",
    "kev": "Kev-0.8B",
    "semif": "SemIf/Qwen3-0.6B",
    "cbjev": "cbjev",
    "classifier": "Classifier†",
    "laya-ml": "Laya (multilingual)",
}
POWER = {"AC Power": "AC", "Battery Power": "battery", "mixed (Battery Power, then AC Power)": "battery→AC"}
ALIASES = {"laya-ml": "laya"}  # non-English suites run on Laya's multilingual weights


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _latest(rows: list[dict], *keys: str, alias: bool = True) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    for r in rows:
        if alias:
            r = {**r, "engine": ALIASES.get(r["engine"], r["engine"])}
        out[tuple(r.get(k) for k in keys)] = r
    return out


def _known(rows: dict[tuple, dict]) -> list[tuple[tuple, dict]]:
    """Rows of the reported engines only (fake engines and new engines stay out of the tables)."""
    return [(k, r) for k, r in rows.items() if k[0] in ENGINES]


def _f(x, digits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _status(out: Path) -> dict[str, str]:
    return {r["engine"]: r["reason"] for r in _rows(out / "status.jsonl") if r.get("status") == "not_runnable"}


def _missing(engine: str, status: dict[str, str]) -> str:
    return f"n/a ({status[engine]})" if engine in status else "n/a (not measured)"


def quality_section(out: Path, status: dict[str, str]) -> str:
    q = _latest(_rows(out / "quality.jsonl"), "engine", "suite")
    main = {"choice": ("accuracy", "macro_f1"), "noul": ("accuracy", "auroc"), "score": ("mae", "qwk")}
    body = []
    for s in suites():
        a, b = main[s.type]
        cells = []
        for e in ENGINES:
            r = q.get((e, s.id))
            if not r:
                cells.append(_missing(e, status))
            elif "accuracy" not in r:
                cells.append(f"n/a (error rate {r['error_rate']:.0%})")
            else:
                err = f" ⚠ err {r['error_rate']:.0%}" if r["error_rate"] else ""
                cells.append(f"{_f(r[a])} / {_f(r[b])}{err}")
        body.append([s.id, f"{a} / {b}", *cells])
    return "### Quality\n\n" + _table(["Suite", "Metric", *(TITLES[e] for e in ENGINES)], body)


def calibration_section(out: Path, status: dict[str, str]) -> str:
    q = _latest(_rows(out / "calibration.jsonl"), "engine", "suite")
    body, overall = [], {e: ([], []) for e in ENGINES}
    for s in suites():
        cells = []
        for e in ENGINES:
            r = q.get((e, s.id))
            if not r or "raw_eval_half" not in r:
                cells.append(_missing(e, status))
                continue
            raw, scaled = r["raw_eval_half"]["ece"], r["scaled_eval_half"]["ece"]
            overall[e][0].append(raw)
            overall[e][1].append(scaled)
            cells.append(f"{_f(raw)} → {_f(scaled)}")
        body.append([s.id, *cells])
    body.append(
        ["**mean over suites**"]
        + [f"{_f(sum(r) / len(r))} → {_f(sum(c) / len(c))}" if r else "n/a" for r, c in overall.values()]
    )
    nll_rows = []
    for s in suites():
        cells = []
        for e in ENGINES:
            r = q.get((e, s.id))
            if not r or "raw_eval_half" not in r:
                cells.append("n/a")
                continue
            cells.append(
                f"{_f(r['raw_eval_half']['nll'])} → {_f(r['scaled_eval_half']['nll'])} · "
                f"B {_f(r['raw_eval_half']['brier'])} → {_f(r['scaled_eval_half']['brier'])}"
            )
        nll_rows.append([s.id, *cells])
    return (
        "### Calibration: ECE raw → temperature-scaled\n\n"
        "Evaluation half of each suite. One temperature per engine model and calibration bucket, fit on the "
        "pooled other halves of every suite in that bucket (benchmark-spec §9); the classifier has one model, "
        "and so one calibrator, per dataset.\n\n"
        + _table(["Suite", *(TITLES[e] for e in ENGINES)], body)
        + "\n\n### Calibration: NLL and Brier, raw → scaled\n\n"
        + _table(["Suite", *(TITLES[e] for e in ENGINES)], nll_rows)
    )


def invariance_section(out: Path) -> str:
    inv = _latest(_rows(out / "invariance.jsonl"), "engine", "suite")
    body = [
        [
            TITLES.get(e, e),
            s,
            _f(r["full_permutation_flip_rate"]),
            _f(r["reverse_flip_rate"]),
            _f(r["max_prob_dev"], 4),
            f"{r['error_rate']:.0%}",
        ]
        for (e, s), r in sorted(_known(inv), key=lambda kv: (ENGINES.index(kv[0][0]), kv[0][1]))
    ]
    itf = _latest(_rows(out / "interference.jsonl"), "engine", "fillers")
    body2 = [
        [TITLES.get(e, e), str(f), _f(r["argmax_change_rate"]), _f(r["max_prob_dev"], 4)]
        for (e, f), r in sorted(_known(itf), key=lambda kv: (ENGINES.index(kv[0][0]), kv[0][1]))
    ]
    return (
        "### Option-order invariance (100 cases per suite)\n\n"
        + _table(["Engine", "Suite", "Full-perm flip", "Reverse flip", "Max prob dev", "Errors"], body)
        + "\n\n### Question interference (agnews-choice, 100 cases; alone vs. with fillers)\n\n"
        + _table(["Engine", "Fillers", "Argmax change", "Max prob dev"], body2)
    )


def latency_section(out: Path) -> str:
    rows = _latest(
        _rows(out / "latency.jsonl"),
        "engine",
        "layer",
        "cache_state",
        "mode",
        "state_tokens",
        "questions",
        "options",
    )
    body = []
    for key, r in sorted(rows.items(), key=lambda kv: (kv[0][3], kv[0][2], kv[0][4], kv[0][5], kv[0][6], kv[0][0])):
        e, layer, cache_state, mode, s, q, k = key
        extra = r.get("last_minute_p50_ms")
        note = "truncated n" if r.get("truncated") else ""
        if extra is not None:
            note = f"last-minute p50 {_f(extra, 1)}"
        body.append(
            [
                TITLES.get(e, e),
                mode,
                cache_state,
                "ping probe" if cache_state == "startup" else f"{s}/{q}/{k}",
                layer,
                str(r["n"]),
                _f(r["p50_ms"], 1),
                _f(r["p95_ms"], 1),
                _f(r["p99_ms"], 1),
                _f(r["decisions_per_sec"], 1),
                f"{r.get('error_rate', 0):.0%}",
                POWER.get(r.get("power_source"), r.get("power_source") or "n/a"),
                note,
            ]
        )
    header = [
        "Engine",
        "Mode",
        "Cache",
        "State/Q/K",
        "Layer",
        "n",
        "p50 ms",
        "p95 ms",
        "p99 ms",
        "dec/s",
        "Errors",
        "Power",
        "Note",
    ]
    return "### Latency (MacBook Air M4, concurrency 1)\n\n" + _table(header, body)


def memory_section(out: Path) -> str:
    mem = _latest(_rows(out / "memory.jsonl"), "engine", alias=False)
    body = [
        [
            TITLES.get(e, e),
            r["model"],
            f"{r['peak_phys_footprint_bytes'] / 2**30:.2f}",
            f"{r['peak_rss_bytes'] / 2**30:.2f}",
        ]
        for (e,), r in mem.items()
    ]
    return "### Peak memory during the quality run\n\n" + _table(
        ["Engine", "Model", "phys_footprint GiB", "RSS GiB"], body
    )


def render(out: Path) -> str:
    status = _status(out)
    parts = [
        quality_section(out, status),
        calibration_section(out, status),
        invariance_section(out),
        latency_section(out),
        memory_section(out),
    ]
    return "\n\n".join(parts)


def write(out: Path, dest: Path) -> None:
    block = f"{START}\n\n{render(out)}\n\n{END}"
    text = dest.read_text() if dest.exists() else f"# Baselines\n\n{START}\n{END}\n"
    if START not in text:
        raise SystemExit(f"{dest} lacks the {START} marker")
    head, rest = text.split(START, 1)
    tail = rest.split(END, 1)[1]
    dest.write_text(head + block + tail)
