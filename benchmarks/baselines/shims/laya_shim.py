"""Laya (Apache-2.0) behind Protocol v1. Uses Laya's own probabilities; never recomputes them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import laya  # noqa: E402
from common import candidates, pick_device, serve  # noqa: E402


def make_decide(agent):
    def decide(state: str, questions: dict) -> dict:
        # common.canonical_state serialized object states; Laya accepts dicts, so pass objects through.
        try:
            parsed = json.loads(state)
            state_in = parsed if isinstance(parsed, dict) else state
        except ValueError:
            state_in = state
        answers = agent.predict(state_in, questions)["answers"]
        out = {}
        for qid, q in questions.items():
            a = answers[qid]
            if q["type"] == "noul":
                out[qid] = {"true": a["noul"], "false": 1.0 - a["noul"]}
            elif q["type"] == "score":  # Laya keys score probabilities by level index
                out[qid] = {name: a["probabilities"][str(i)] for i, name in enumerate(candidates(q))}
            else:
                out[qid] = {name: a["probabilities"][name] for name in candidates(q)}
        return out

    return decide


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    device = pick_device()
    revision = laya.PINNED_REVISIONS.get(a.model, a.revision)
    agent = laya.load(a.model, device=device, revision=revision)
    serve(make_decide(agent), a.port, a.model, device)  # usage.input_tokens = 0: not exposed per state
