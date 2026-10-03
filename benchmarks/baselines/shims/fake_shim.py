"""Deterministic fake engine for harness tests. No ML.

Each candidate's logit is a hash of (state, instructions, candidate name), so the fake is
permutation-invariant and question-isolated by construction. --biased adds a bonus to the
first-listed candidate as a negative control for the invariance runner.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import candidates, serve  # noqa: E402


def logit(*parts: str) -> float:
    h = hashlib.sha256("\x00".join(parts).encode()).digest()
    return int.from_bytes(h[:4], "big") / 2**32 * 4.0


def make_decide(biased: bool):
    def decide(state: str, questions: dict) -> dict:
        out = {}
        for qid, q in questions.items():
            names = candidates(q)
            z = [logit(state, q["instructions"], n) + (3.0 if biased and i == 0 else 0.0) for i, n in enumerate(names)]
            e = [math.exp(v - max(z)) for v in z]
            out[qid] = {n: v / sum(e) for n, v in zip(names, e, strict=True)}
        return out

    return decide


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--biased", action="store_true")
    a = ap.parse_args()
    serve(make_decide(a.biased), a.port, "fake-biased" if a.biased else "fake", count_tokens=lambda s: len(s.split()))
