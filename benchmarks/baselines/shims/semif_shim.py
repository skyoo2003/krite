"""SemIf (MIT) with Qwen3-0.6B behind Protocol v1, on torch MPS.

SemIf scores one question per row. Each Protocol v1 question becomes one row, and every request
(one question or many) goes through SemIf's shared-state-prefix scorer, so one-question and
multi-question answers come from the same scorer. States whose tokens merge with the prompt suffix
have no shared prefix; for those, every question uses SemIf's direct scorer.
SemIf's MLX backend accepts only Qwen3.5 and its torch loader requires CUDA, so this shim loads
the pinned checkpoint on MPS itself (same settings: bf16, no remote code) and calls SemIf's
device-agnostic scorers unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import torch  # noqa: E402
import transformers  # noqa: E402
from common import (  # noqa: E402
    RequestError,
    candidates,
    pick_device,  # noqa: E402
    serve,
)
from semif_phase1 import direct, shared  # noqa: E402

MAX_OPTIONS = 16  # SemIf answer slots are letters A-P
MAX_TOKENS = 4096  # SemIf's default prompt limit; it never truncates


def row_for(qid: str, state: str, q: dict) -> dict:
    names = candidates(q)
    if not 2 <= len(names) <= MAX_OPTIONS:
        raise RequestError(f"SemIf supports 2-{MAX_OPTIONS} options", f"questions.{qid}.criteria", "too_many_options")
    if q["type"] == "noul":
        descriptions = {"true": "yes", "false": "no"}
        for key in ("true", "false"):
            if isinstance(q.get("criteria"), dict) and q["criteria"].get(key):
                descriptions[key] = f"{descriptions[key]}: {q['criteria'][key]}"
    elif q["type"] == "choice":
        descriptions = {n: n if not q["criteria"][n] else f"{n}: {q['criteria'][n]}" for n in names}
    else:
        descriptions = {n: n for n in names}
    return {
        "id": qid,
        "state": state,
        "question": q["instructions"],
        "options": [{"id": n, "description": descriptions[n]} for n in names],
    }


def make_decide(model, tokenizer, metadata):
    def decide(state: str, questions: dict) -> dict:
        rows = [row_for(qid, state, q) for qid, q in questions.items()]
        try:
            try:
                results, _ = shared.score_shared(model, tokenizer, rows, metadata, MAX_TOKENS)
            except ValueError as e:
                if "state prefix does not match" not in str(e):
                    raise
                # The state's last tokens merge with the prompt suffix, so no shared prefix exists for this
                # state; every question of the request then uses the direct scorer.
                results = [direct.score(model, tokenizer, row, metadata, MAX_TOKENS) for row in rows]
        except ValueError as e:
            if "exceed limit" in str(e):  # SemIf refuses to truncate prompts over MAX_TOKENS
                raise RequestError(str(e), "state", "state_too_long") from e
            raise
        return {r["id"]: dict(zip(r["option_ids"], r["probabilities"], strict=True)) for r in results}

    return decide


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True, help="40-character commit sha")
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    device = pick_device()
    kw = {"revision": a.revision, "trust_remote_code": False}
    tokenizer = transformers.AutoTokenizer.from_pretrained(a.model, **kw)
    model = transformers.AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, **kw).to(device).eval()
    metadata = {"source": a.model, "revision": a.revision, "dtype": "bfloat16", "device": device}
    serve(make_decide(model, tokenizer, metadata), a.port, f"semif-{a.model}", device)
