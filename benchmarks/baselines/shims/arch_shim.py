"""Architecture-study arms (training/krite_train) behind Protocol v1.

The joint arm (B) encodes one sequence per question. Tower arms (D) encode the state once and keep the
state memory in an LRU cache keyed by the canonical state; late-interaction arms also cache each
candidate's lower-layer states by its token ids. `--no-state-cache` turns both caches off so a hit and a
miss can be compared (ARCHITECTURE.md I3).
"""

from __future__ import annotations

import argparse
import sys
import threading
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import torch  # noqa: E402
from common import RequestError, pick_device, serve  # noqa: E402
from krite_train import model as m  # noqa: E402

MAX_STATE = 8190  # content tokens, as in krite serve
MAX_SEQ = 8192  # encoder positions
CACHE_STATES = 256
CACHE_CANDIDATES = 4096
CHUNK = 64  # candidate rows per tower call (whole questions); bounds the per-candidate state copies


def criteria(q: dict) -> list[tuple[str, str | None]]:
    if q["type"] == "choice":
        return list(q["criteria"].items())
    if q["type"] == "score":
        return [(name, None) for name in q["criteria"]]
    given = q.get("criteria") or {}
    return [("true", given.get("true")), ("false", given.get("false"))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--no-state-cache", action="store_true")
    a = ap.parse_args()
    device = pick_device()
    arm, net, tok = m.load(a.ckpt, device)
    sp = m.specials(tok)
    cache: OrderedDict[str, torch.Tensor] = OrderedDict()
    cand_cache: OrderedDict[tuple[int, ...], torch.Tensor] = OrderedDict()
    lock = threading.Lock()  # ponytail: one lock; the harness runs concurrency 1

    def content(state: str) -> list[int]:
        ids = tok(state, add_special_tokens=False)["input_ids"]
        if len(ids) > MAX_STATE:
            raise RequestError(f"state has {len(ids)} tokens; max is {MAX_STATE}", "state", "state_too_long")
        return ids

    def joint(state: str, questions: dict) -> dict[str, torch.Tensor]:
        sids = content(state)
        rows, marks, qtypes = [], [], []
        for qid, q in questions.items():
            options = [m.option_text(n, d) for n, d in criteria(q)]
            ids, mk = m.joint_ids(tok, q["type"], q["instructions"], options, sids)
            if len(ids) > MAX_SEQ:
                raise RequestError(f"state and question {qid!r} exceed {MAX_SEQ} tokens", "state", "state_too_long")
            rows.append(ids)
            marks.append(mk)
            qtypes.append(m.QTYPES[q["type"]])
        ids, mask = m.pad(rows)
        kmax = max(map(len, marks))
        pos = torch.tensor([mk + [-1] * (kmax - len(mk)) for mk in marks], device=device)
        qt = torch.tensor(qtypes, device=device)
        e = net(ids.to(device), mask.to(device), pos, pos >= 0, qt)
        return {qid: e[i, : len(marks[i])] for i, qid in enumerate(questions)}

    def memory(state: str) -> torch.Tensor:
        if not a.no_state_cache and state in cache:
            cache.move_to_end(state)
            return cache[state]
        ids, mask = m.pad([[sp["bos"], *content(state), sp["eos"]]])
        h = net.encode_state(ids.to(device), mask.to(device))
        if not a.no_state_cache:
            cache[state] = h
            if len(cache) > CACHE_STATES:
                cache.popitem(last=False)
        return h

    def lower(rows: list[list[int]]) -> torch.Tensor:
        """Lower-layer candidate states, padded like `m.pad(rows)`; misses run as one batch."""
        keys = [tuple(r) for r in rows]
        miss = list(dict.fromkeys(k for k in keys if a.no_state_cache or k not in cand_cache))
        fresh = {}
        if miss:
            ids, mask = m.pad([list(k) for k in miss])
            x = net.lower(ids.to(device), mask.to(device))
            fresh = {k: x[i, : len(k)] for i, k in enumerate(miss)}
            if not a.no_state_cache:
                cand_cache.update(fresh)
                while len(cand_cache) > CACHE_CANDIDATES:
                    cand_cache.popitem(last=False)
        out = torch.zeros((len(rows), max(map(len, rows)), net.encoder.config.hidden_size), device=device)
        for i, k in enumerate(keys):
            if k in fresh:
                h = fresh[k]
            else:
                cand_cache.move_to_end(k)
                h = cand_cache[k]
            out[i, : len(k)] = h
        return out

    def tower(state: str, questions: dict) -> dict[str, torch.Tensor]:
        h = memory(state)
        h_mask = torch.ones((1, h.shape[-2]), dtype=torch.long, device=device)
        # Chunks hold whole questions (set attention spans one question), about CHUNK rows each.
        chunks, rows, spans = [[]], 0, {}
        for qid, q in questions.items():
            if chunks[-1] and rows + len(criteria(q)) > CHUNK:
                chunks.append([])
                rows = 0
            chunks[-1].append(qid)
            rows += len(criteria(q))
        for chunk in chunks:
            ids_rows, qtypes, group, starts = [], [], [], {}
            for g, qid in enumerate(chunk):
                q = questions[qid]
                cands = criteria(q)
                starts[qid] = (len(ids_rows), len(ids_rows) + len(cands))
                ids_rows += [m.candidate_ids(tok, q["instructions"], n, d) for n, d in cands]
                qtypes += [m.QTYPES[q["type"]]] * len(cands)
                group += [g] * len(cands)
            ids, mask = m.pad(ids_rows)
            owner = torch.zeros(len(ids_rows), dtype=torch.long, device=device)
            qt = torch.tensor(qtypes, device=device)
            extra = {"lower": lower(ids_rows)} if hasattr(net, "lower") else {}
            e = net.energies(
                h, h_mask, owner, ids.to(device), mask.to(device), qt, torch.tensor(group, device=device), **extra
            )
            spans.update({qid: e[s:t] for qid, (s, t) in starts.items()})
        return spans

    @torch.no_grad()
    def decide(state: str, questions: dict) -> dict:
        with lock:
            energies = joint(state, questions) if net.kind == "joint" else tower(state, questions)
            out = {}
            for qid, q in questions.items():
                p = torch.softmax(energies[qid].cpu().double(), -1).tolist()  # MPS has no float64
                out[qid] = dict(zip([n for n, _ in criteria(q)], p, strict=True))
            return out

    model_id = f"arch-{arm}" + ("-nocache" if a.no_state_cache else "")
    serve(decide, a.port, model_id, device, count_tokens=lambda s: len(tok(s, add_special_tokens=False)["input_ids"]))


if __name__ == "__main__":
    main()
