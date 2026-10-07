"""Krite in process: the same runtime as `krite serve`, without the server.

    import krite

    k = krite.Krite("krite-0.15b-v1")  # a model directory, e.g. from `hf download skyoo2003/krite-0.15b-v1`
    r = k.decide({
        "state": "Hi, I was charged twice for my March subscription.",
        "questions": {"refund_requested": {"type": "noul"}},
    })
    r["answers"]["refund_requested"]["noul"]

`decide` takes a Protocol v1 request and returns the response that `POST /v1/systemone` would, as a dict;
an error response raises `KriteError` with the same status and body. Calls run one at a time (one runtime
lock) and release the GIL, so threads may share one `Krite`.
"""

import json
from collections.abc import Mapping
from os import PathLike
from typing import Any

from ._krite import Engine

__all__ = ["Krite", "KriteError"]


class KriteError(Exception):
    """A Protocol v1 error response: `status` 422 (or 500) and the body's `type`, `message`, `param`."""

    def __init__(self, status: int, type: str, message: str, param: str | None = None):
        super().__init__(f"{type}: {message}")
        self.status, self.type, self.message, self.param = status, type, message, param


class Krite:
    """A model directory loaded on Metal (`device="auto"`, when available) or the CPU (`device="cpu"`).

    Cache budgets are MiB; 0 turns a cache off. `raw=True` serves uncalibrated probabilities.
    """

    def __init__(
        self,
        model: str | PathLike[str],
        *,
        device: str = "auto",
        state_cache_mb: int = 1024,
        candidate_cache_mb: int = 64,
        raw: bool = False,
    ):
        self._engine = Engine(model, device, state_cache_mb, candidate_cache_mb, raw)

    @property
    def model_id(self) -> str:
        return self._engine.model_id

    def decide(self, request: Mapping[str, Any] | str | bytes) -> dict[str, Any]:
        if isinstance(request, Mapping):
            request = json.dumps(request)
        body = request.encode() if isinstance(request, str) else request
        status, text = self._engine.decide(body)
        out = json.loads(text)
        if status != 200:
            e = out["error"]
            raise KriteError(status, e["type"], e["message"], e.get("param"))
        return out
