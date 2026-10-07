"""Protocol v1 server for engines without a /v1/systemone endpoint.

Stdlib only: this file runs inside each engine's own venv. An engine shim supplies
`decide(state, questions) -> {qid: {candidate_name: probability}}` (noul candidates are
"true"/"false"); this module builds Protocol v1 answers, errors, and timing.
"""

from __future__ import annotations

import json
import math
import time
import traceback
import uuid
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

Decide = Callable[[object, dict], dict[str, dict[str, float]]]
MAX_BODY = 4 * 1024 * 1024


class RequestError(Exception):
    def __init__(self, message: str, param: str | None = None, type_: str = "invalid_request"):
        super().__init__(message)
        self.param, self.type = param, type_


def canonical_state(state: object) -> str:
    """String states pass through; object states use sorted-key compact JSON (JCS for float-free objects)."""
    if isinstance(state, str):
        return state
    return json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def candidates(q: dict) -> list[str]:
    if q["type"] == "choice":
        return list(q["criteria"])
    if q["type"] == "score":
        return list(q["criteria"])
    return ["true", "false"]


def confidence(p: list[float]) -> float:
    if len(p) == 1:
        return 1.0
    h = -sum(x * math.log(x) for x in p if x > 0)
    return min(1.0, max(0.0, 1.0 - h / math.log(len(p))))


def build_answer(q: dict, probs: dict[str, float]) -> dict:
    names = candidates(q)
    total = sum(probs[n] for n in names)
    p = [min(1.0, max(0.0, probs[n] / total)) for n in names]
    if q["type"] == "noul":
        return {"type": "noul", "noul": p[0]}
    if q["type"] == "choice":
        top = max(p)
        best = min((i for i in range(len(names)) if p[i] == top), key=lambda i: names[i])
        return {
            "type": "choice",
            "choice": names[best],
            "probabilities": dict(zip(names, p, strict=True)),
            "confidence": confidence(p),
        }
    return {
        "type": "score",
        "score": sum(i * x for i, x in enumerate(p)),
        "legend": {str(i): n for i, n in enumerate(names)},
        "probabilities": {str(i): x for i, x in enumerate(p)},
        "confidence": confidence(p),
    }


def validate(body: dict) -> None:
    """Minimal structural checks; full validation is the JSON Schema's job on the harness side."""
    if not isinstance(body, dict) or "state" not in body or not isinstance(body.get("questions"), dict):
        raise RequestError("request needs `state` and `questions`")
    if not body["questions"]:
        raise RequestError("`questions` is empty", "questions")
    for qid, q in body["questions"].items():
        t = q.get("type")
        if t not in ("choice", "score", "noul") or not isinstance(q.get("instructions"), str):
            raise RequestError("question needs a valid `type` and `instructions`", f"questions.{qid}")
        if t == "choice" and not (isinstance(q.get("criteria"), dict) and q["criteria"]):
            raise RequestError("choice criteria must be a non-empty object", f"questions.{qid}.criteria")
        if t == "score" and not (isinstance(q.get("criteria"), list) and q["criteria"]):
            raise RequestError("score criteria must be a non-empty array", f"questions.{qid}.criteria")


def serve(
    decide: Decide,
    port: int,
    model_id: str,
    device: str = "cpu",
    count_tokens: Callable[[str], int] | None = None,
) -> None:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # keep stdout quiet during latency runs
            pass

        def _send(self, status: int, payload: dict, extra: dict[str, str]) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            for k, v in extra.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            rid = uuid.uuid4().hex
            headers = {"x-krite-request-id": rid, "x-typesafe-request-id": rid, "x-bench-device": device}
            if self.path != "/v1/systemone":
                self._send(404, _error("invalid_request", f"unknown path {self.path}", None), headers)
                return
            try:
                length = int(self.headers.get("content-length", "0"))
                if length > MAX_BODY:
                    raise RequestError("request body exceeds 4 MiB")
                body = json.loads(self.rfile.read(length))
                t0 = time.perf_counter_ns()
                validate(body)
                state = canonical_state(body["state"])
                raw = decide(state, body["questions"])
                answers = {qid: build_answer(q, raw[qid]) for qid, q in body["questions"].items()}
                latency_ms = (time.perf_counter_ns() - t0) / 1e6
                tokens = count_tokens(state) if count_tokens else 0
                resp = {
                    "model": model_id,
                    "answers": answers,
                    "usage": {"input_tokens": tokens, "output_tokens": 0},
                    "latency_ms": latency_ms,
                }
                self._send(200, resp, headers)
            except RequestError as e:  # engines raise RequestError for limits (type state_too_long, ...)
                self._send(422, _error(e.type, str(e) or "invalid", e.param), headers)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                self._send(422, _error("invalid_request", f"body is not JSON: {e}", None), headers)
            except Exception as e:  # noqa: BLE001 - engine failures become 500 bodies, logged to stderr
                traceback.print_exc()
                self._send(500, _error("internal_error", f"{type(e).__name__}: {e}", None), headers)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"serving {model_id} on 127.0.0.1:{port} ({device})", flush=True)
    server.serve_forever()


def _error(type_: str, message: str, param: str | None) -> dict:
    return {"error": {"type": type_, "message": message, "param": param}}


def pick_device() -> str:
    try:
        import torch

        return "mps" if torch.backends.mps.is_available() else "cpu"
    except ImportError:
        return "cpu"
