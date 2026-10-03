"""Model-layer reference: mmBERT-small encoder forward on torch, same token ids as `krite bench-encoder`.

Runs in the `classifier` venv (torch + transformers). Prints one JSON object on stdout in the same
shape as `krite bench-encoder`: {"backend", "precision", "results": {tokens: {times_ms, truncated, probe}}}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch
from transformers import AutoModel

REPO, REVISION = "jhu-clsp/mmBERT-small", "abc32620dd4f6ab06f5fbe905dc25f310618e09f"
BUDGET_S = 15 * 60  # above this, a cell drops to 50 measured runs (benchmark-spec §12 `truncated`)


def bench_ids(n: int) -> list[int]:
    """Keep in sync with `bench_ids` in crates/krite-candle/src/lib.rs."""
    return [2] + [(1000 + 7919 * i) % 256000 for i in range(n - 2)] + [1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="64,512,2048")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    model = AutoModel.from_pretrained(REPO, revision=REVISION, dtype=torch.float32).to(a.device).eval()
    sync = torch.mps.synchronize if a.device == "mps" else (lambda: None)

    @torch.no_grad()
    def forward(ids: torch.Tensor) -> tuple[float, torch.Tensor]:
        t0 = time.perf_counter()
        h = model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state
        sync()  # MPS is asynchronous; stop the clock only when the device has finished
        return (time.perf_counter() - t0) * 1000, h

    results = {}
    for s in (int(x) for x in a.tokens.split(",")):
        ids = torch.tensor([bench_ids(s)], device=a.device)
        warm = sorted(forward(ids)[0] for _ in range(a.warmup))
        median = warm[len(warm) // 2] if warm else 0.0
        truncated = median * (a.warmup + a.n) / 1000 > BUDGET_S
        times = [forward(ids)[0] for _ in range(min(a.n, 50) if truncated else a.n)]
        h = forward(ids)[1][0]
        probe = [h[p, :8].float().cpu().tolist() for p in (0, s // 2, s - 1)]
        print(f"{s} tokens: p50 {sorted(times)[len(times) // 2]:.1f} ms over {len(times)} runs", file=sys.stderr)
        results[str(s)] = {"times_ms": times, "truncated": truncated, "probe": probe}
    attn = model.config._attn_implementation
    print(json.dumps({"backend": f"torch-{a.device}", "precision": "fp32", "attn": attn, "results": results}))


if __name__ == "__main__":
    main()
