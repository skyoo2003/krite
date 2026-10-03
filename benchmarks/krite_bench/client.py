"""HTTP client for any /v1/systemone engine, with timing and response-schema validation."""

from __future__ import annotations

import json
import time
import tomllib
from dataclasses import dataclass, field
from functools import cache

import httpx
import jsonschema

from .data import ROOT, SCHEMA_DIR

ENGINES_TOML = ROOT / "benchmarks" / "baselines" / "engines.toml"
LIMIT_ERRORS = {"state_too_long", "too_many_questions", "too_many_options"}
CHOICE_TIE_TOL = 1e-6  # probabilities this close to the top count as tied, so any tie-break rule passes


@cache
def _validator() -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(json.loads((SCHEMA_DIR / "response.schema.json").read_text()))


def engine_config(name: str) -> dict:
    engines = tomllib.loads(ENGINES_TOML.read_text())
    if name not in engines:
        raise KeyError(f"engine {name!r} not in {ENGINES_TOML}")
    return engines[name]


@dataclass
class Result:
    status: int
    http_ms: float
    response: dict | None = None
    runtime_ms: float | None = None
    headers: dict = field(default_factory=dict)
    error: str | None = None  # set on any failure
    not_supported: bool = False  # engine rejected the request with a limit error

    @property
    def ok(self) -> bool:
        return self.error is None


def from_jev(payload: dict) -> dict:
    """Map the Jev-style response dialect (seen in Kev and cbjev) onto Krite Protocol v1 before validation.

    Observed differences, all recorded as protocol compatibility findings in docs/baselines.md:
    score `probabilities` keyed by level index instead of level name; nonzero `usage.output_tokens`;
    missing `latency_ms`; extra top-level fields (e.g. `routing`) and noul `confidence`.
    The returned `latency_ms` is 0 when the engine does not report one (no runtime layer).
    """
    if not isinstance(payload, dict):
        return payload
    out = {k: payload[k] for k in ("model", "answers", "usage") if k in payload}
    out["latency_ms"] = payload.get("latency_ms", 0)
    answers = {}
    for qid, answer in (payload.get("answers") or {}).items():
        a = dict(answer)
        probs, legend = a.get("probabilities"), a.get("legend")
        if a.get("type") == "score" and probs and legend and set(probs) <= set(legend):
            a["probabilities"] = {legend[k]: v for k, v in probs.items()}
        if a.get("type") == "noul":
            a = {"type": "noul", "noul": a.get("noul")}
        answers[qid] = a
    out["answers"] = answers
    if isinstance(out.get("usage"), dict):
        out["usage"] = {"input_tokens": out["usage"].get("input_tokens", 0), "output_tokens": 0}
    return out


def answer_mismatch(questions: dict, answers: dict) -> str | None:
    """The schema allows any answer type and probability keys; the runners need them to match the request."""
    for qid, q in questions.items():
        a = answers[qid]
        if a.get("type") != q["type"]:
            return f"answer {qid}: type {a.get('type')!r}, question type {q['type']!r}"
        probs = a.get("probabilities") or {}
        if q["type"] != "noul" and not set(q["criteria"]) <= set(probs):
            return f"answer {qid}: probabilities do not cover every candidate name"
        if q["type"] == "choice":
            # The choice must be an offered candidate with the top probability; the runners read `choice`.
            top = max(probs[n] for n in q["criteria"])
            if a.get("choice") not in q["criteria"] or probs[a["choice"]] < top - CHOICE_TIE_TOL:
                return f"answer {qid}: choice {a.get('choice')!r} is not a top-probability candidate"
    return None


class Engine:
    def __init__(
        self, name: str, port: int, request_model: str | None = None, dialect: str = "krite", timeout: float = 120.0
    ):
        self.name, self.port, self.request_model, self.dialect = name, port, request_model, dialect
        self.url = f"http://127.0.0.1:{port}/v1/systemone"
        self.http = httpx.Client(timeout=timeout)  # keep-alive by default

    @classmethod
    def from_config(cls, name: str) -> Engine:
        cfg = engine_config(name)
        return cls(name, cfg["port"], cfg.get("request_model"), cfg.get("dialect", "krite"))

    def call(self, request: dict, dataset: str | None = None) -> Result:
        if self.request_model:
            request = {**request, "model": self.request_model.format(dataset=dataset or "")}
        body = json.dumps(request, ensure_ascii=False).encode()
        t0 = time.perf_counter_ns()
        try:
            r = self.http.post(self.url, content=body, headers={"content-type": "application/json"})
        except httpx.HTTPError as e:
            return Result(0, (time.perf_counter_ns() - t0) / 1e6, error=f"{type(e).__name__}: {e}")
        http_ms = (time.perf_counter_ns() - t0) / 1e6
        headers = dict(r.headers)
        try:
            payload = r.json()
        except ValueError:
            return Result(r.status_code, http_ms, headers=headers, error=f"non-JSON body (HTTP {r.status_code})")
        if r.status_code != 200:
            err = payload.get("error") if isinstance(payload, dict) else None
            etype = err.get("type") if isinstance(err, dict) else None
            msg = f"HTTP {r.status_code}: {json.dumps(payload)[:300]}"
            return Result(
                r.status_code, http_ms, payload, headers=headers, error=msg, not_supported=etype in LIMIT_ERRORS
            )
        if self.dialect == "jev":
            payload = from_jev(payload)
        problems = sorted(_validator().iter_errors(payload), key=str)
        if problems:
            return Result(200, http_ms, payload, headers=headers, error=f"schema: {problems[0].message[:300]}")
        if set(payload["answers"]) != set(request["questions"]):
            return Result(200, http_ms, payload, headers=headers, error="answers do not cover every question id")
        mismatch = answer_mismatch(request["questions"], payload["answers"])
        if mismatch:
            return Result(200, http_ms, payload, headers=headers, error=mismatch)
        runtime = payload.get("latency_ms") or None
        return Result(200, http_ms, payload, runtime_ms=runtime, headers=headers)

    def close(self) -> None:
        self.http.close()
