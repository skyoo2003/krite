"""Trains one arm from mmBERT-small (krite_train.model.ARMS) on a training mixture (krite_train.data).

    uv run python -m krite_train.train --arm d2            # full run → training/ckpt/d2/
    uv run python -m krite_train.train --arm b --scale 0.02 # pilot

Arms use the study mixture and cross-entropy unless their spec says otherwise (`mixture`, `brier`,
`ordinal`, `epochs`; docs/training-data.md). Arms with the same data spec and seed see the same
examples in the same order.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from krite_bench.data import SEED, TOKENIZER_REPO, TOKENIZER_REVISION
from transformers import get_linear_schedule_with_warmup

from . import data
from . import model as m

CKPT_DIR = Path(__file__).resolve().parents[1] / "ckpt"
JOINT_LEN = 384  # B rows: JOINT_HEAD + MAX_STATE_TRAIN + <eos>, rounded up
STATE_LEN = m.MAX_STATE_TRAIN + 2
LR_ENCODER, LR_NEW = 5e-5, 3e-4
BRIER_W, ORD_W = 1.0, 1.0  # fixed before the loss runs (docs/training-data.md)


def losses(e: torch.Tensor, gold: torch.Tensor, qtype: torch.Tensor, spec: dict) -> torch.Tensor:
    """CE, plus Brier (spec "brier") and, on score rows only, squared CDF distance (spec "ordinal")."""
    loss = F.cross_entropy(e, gold)
    if not (spec.get("brier") or spec.get("ordinal")):
        return loss
    p, y = e.softmax(-1), F.one_hot(gold, e.size(-1)).to(e.dtype)
    if spec.get("brier"):
        loss = loss + BRIER_W * (p - y).pow(2).sum(-1).mean()
    score = qtype == m.QTYPES["score"]
    # Batches share K, not type: the ordinal term must skip choice and noul rows.
    if spec.get("ordinal") and score.any():
        d = (p[score].cumsum(-1) - y[score].cumsum(-1)).pow(2).sum(-1) / (e.size(-1) - 1)
        loss = loss + ORD_W * d.mean()
    return loss


def tokenize(tok, ex: dict) -> dict:
    """Token ids for both layouts, computed once per example."""
    state = tok(ex["state"], add_special_tokens=False)["input_ids"][: m.MAX_STATE_TRAIN]
    sp = m.specials(tok)
    joint, markers = m.joint_ids(
        tok, ex["type"], ex["instructions"], [m.option_text(n, d) for n, d in ex["candidates"]], state
    )
    if len(joint) > JOINT_LEN:
        raise RuntimeError(f"joint row has {len(joint)} tokens; JOINT_LEN is {JOINT_LEN}")
    return {
        "state": [sp["bos"], *state, sp["eos"]],
        "cands": [m.candidate_ids(tok, ex["instructions"], n, d) for n, d in ex["candidates"]],
        "joint": joint,
        "markers": markers,
        "qtype": m.QTYPES[ex["type"]],
        "gold": ex["gold"],
    }


def energies(net, batch: list[dict], device: str) -> torch.Tensor:
    """(B, K) energies for one batch of equal-K tokenized examples."""
    qtype = torch.tensor([t["qtype"] for t in batch], device=device)
    if net.kind == "joint":
        ids, mask = m.pad([t["joint"] for t in batch], JOINT_LEN)
        pos = torch.tensor([t["markers"] for t in batch], device=device)
        return net(ids.to(device), mask.to(device), pos, torch.ones_like(pos, dtype=torch.bool), qtype)
    k = len(batch[0]["cands"])
    s_ids, s_mask = m.pad([t["state"] for t in batch], STATE_LEN)
    c_ids, c_mask = m.pad([c for t in batch for c in t["cands"]], m.MAX_CANDIDATE)
    s_mask = s_mask.to(device)
    mem = net.encode_state(s_ids.to(device), s_mask)
    owner = torch.arange(len(batch), device=device).repeat_interleave(k)
    e = net.energies(mem, s_mask, owner, c_ids.to(device), c_mask.to(device), qtype.repeat_interleave(k))
    return e.view(len(batch), k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(m.ARMS))
    ap.add_argument("--scale", type=float, default=1.0, help="fraction of every source cap (pilot: 0.02)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=CKPT_DIR)
    a = ap.parse_args()

    spec = m.ARMS[a.arm]
    seed = spec.get("seed", a.seed)
    torch.manual_seed(seed)
    examples, sources = data.build(seed, a.scale, mixture=spec.get("mixture", "study"))
    tok = m.tokenizer()
    t0 = time.time()
    toks = [tokenize(tok, ex) for ex in examples]
    index = {id(ex): i for i, ex in enumerate(examples)}
    epochs, lr_new = spec.get("epochs", 1), spec.get("lr_new", LR_NEW)
    batches = [
        [toks[index[id(ex)]] for ex in b]
        for e in range(epochs)
        for b in data.batches(examples, a.batch, seed + e)  # epoch 0 is the order every 1-epoch arm uses
    ]
    print(f"{len(examples)} examples, {len(batches)} batches, tokenized in {time.time() - t0:.0f}s", flush=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = m.build(a.arm).to(device).train()
    # Without checkpointing the MPS allocator peaked at 11.5 GiB (d2, K=12) and a 16 GB machine swapped.
    net.encoder.gradient_checkpointing_enable()
    enc = {id(p) for p in net.encoder.parameters()}
    opt = torch.optim.AdamW(
        [
            {"params": [p for p in net.parameters() if id(p) in enc], "lr": LR_ENCODER},
            {"params": [p for p in net.parameters() if id(p) not in enc], "lr": lr_new},
        ],
        weight_decay=0.01,
    )
    steps = len(batches)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    peak, t0, window = 0, time.time(), []
    for step, batch in enumerate(batches):
        gold = torch.tensor([t["gold"] for t in batch], device=device)
        qtype = torch.tensor([t["qtype"] for t in batch], device=device)
        loss = losses(energies(net, batch, device), gold, qtype, spec)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        window.append(loss.item())
        if device == "mps":  # every step: cached buffers for each K shape otherwise pile up
            peak = max(peak, torch.mps.driver_allocated_memory())
            torch.mps.empty_cache()
        if step % 100 == 0 or step == steps - 1:
            rate = (step + 1) * a.batch / (time.time() - t0)
            mean = sum(window) / len(window)
            print(f"step {step}/{steps} loss {mean:.4f} ({time.time() - t0:.0f}s, {rate:.1f} ex/s)", flush=True)
            window = []

    seconds = round(time.time() - t0)
    meta = {
        "arm": a.arm,
        **m.ARMS[a.arm],
        "base": TOKENIZER_REPO,
        "revision": TOKENIZER_REVISION,
        "examples": len(examples),
        "trained_examples": steps * a.batch,
        "train_sha256": sources.pop("train_sha256"),
        "mixture": sources.pop("mixture"),
        "sources": sources,
        "loss": {"brier": BRIER_W if spec.get("brier") else 0, "ordinal": ORD_W if spec.get("ordinal") else 0},
        "seed": seed,
        "scale": a.scale,
        "steps": steps,
        "batch": a.batch,
        "lr_encoder": LR_ENCODER,
        "lr_new": lr_new,
        "epochs": epochs,
        "grad_checkpointing": True,
        "device": device,
        "seconds": seconds,
        "peak_mps_bytes": peak,
    }
    out = a.out / a.arm
    m.save(net.cpu(), a.arm, tok, out, meta)
    print(f"saved {out} ({seconds}s, peak {peak / 2**30:.1f} GiB)", flush=True)


if __name__ == "__main__":
    main()
