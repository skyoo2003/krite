"""Fine-tunes the fixed-label classifier baseline (mmBERT-small, same init as Krite), one per dataset.

Runs in the `classifier` venv. This baseline sees each dataset's train split, which Krite and the
other baselines do not, so it is reported as a fixed-label upper reference, not as a peer.

    python classifier_train.py --dataset agnews
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from krite_bench import data as d  # noqa: E402

BASE = "jhu-clsp/mmBERT-small"
CKPT_DIR = Path(__file__).parent / "classifier_ckpt"

# dataset -> (train files, label names). Label names match the option names used in the suites.
TRAIN = {
    "agnews": (["data/train-00000-of-00001.parquet"], d.AGNEWS_LABELS),
    "banking77": (["data/train-00000-of-00001.parquet"], None),
    "massive": (["train/en.json.gz"], None),  # English only: non-English suites are zero-shot
    "boolq": (["data/train-00000-of-00001.parquet"], ["true", "false"]),
    "xnli": (["en/train-00000-of-00001.parquet"], ["true", "false"]),  # English only
    "sst5": (["train.jsonl"], d.SST5_LEVELS),
    "amazon": ([f"{lg}/train.jsonl" for lg in d.AMAZON_LANGS], d.STAR_LEVELS),
}


def examples(dataset: str) -> tuple[list[tuple[str, str | None, str]], list[str]]:
    """Return ([(text, text_pair, label_name)], label names)."""
    files, labels = TRAIN[dataset]
    ds = d.DATASETS[dataset]
    rows = []
    for f in files:
        path = d.hf_hub_download(ds.repo, f, repo_type="dataset", revision=ds.revision)
        rows += d._read_rows(path)
    out = []
    for r in rows:
        if dataset == "agnews":
            out.append((r["text"], None, d.AGNEWS_LABELS[int(r["label"])]))
        elif dataset in ("banking77", "massive"):
            out.append((r["text"], None, d._label_name(r["label_text"] if dataset == "banking77" else r["label"])))
        elif dataset == "boolq":
            out.append((r["question"], r["passage"], "true" if r["answer"] else "false"))
        elif dataset == "xnli" and int(r["label"]) != 1:
            out.append((r["premise"], r["hypothesis"], "true" if int(r["label"]) == 0 else "false"))
        elif dataset == "sst5":
            out.append((r["text"], None, d.SST5_LEVELS[int(r["label"])]))
        elif dataset == "amazon":
            out.append((r["text"], None, d.STAR_LEVELS[int(r["label"])]))
    return out, labels or sorted({lab for _, _, lab in out})


def test_texts(dataset: str) -> set[str]:
    """State texts of every built test case for this dataset (leakage guard)."""
    texts = set()
    for s in d.suites():
        if s.dataset == dataset and (d.DATA_DIR / f"{s.id}.jsonl").exists():
            for c in d.load_suite(s.id):
                st = c["request"]["state"]
                texts.add(json.dumps(st, sort_keys=True) if isinstance(st, dict) else st)
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(TRAIN))
    ap.add_argument("--max-examples", type=int, default=20_000)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    a = ap.parse_args()

    rows, labels = examples(a.dataset)
    held_out = test_texts(a.dataset)
    if not held_out:
        raise SystemExit("build the suites first (`krite-bench data`): the leakage guard needs them")

    def state_key(t: str, p: str | None) -> str:
        if a.dataset == "boolq":
            return json.dumps({"passage": p, "question": t}, sort_keys=True)
        if a.dataset == "xnli":
            return json.dumps({"hypothesis": p, "premise": t}, sort_keys=True)
        return t

    before = len(rows)
    rows = [r for r in rows if state_key(r[0], r[1]) not in held_out]
    rng = np.random.default_rng(d.SEED)
    rows = [rows[int(i)] for i in rng.permutation(len(rows))[: a.max_examples]]
    train_hash = hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()
    print(f"{a.dataset}: {len(rows)} train examples ({before - len(rows)} dropped), {len(labels)} labels")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # Dynamic padding made the MPS allocator cache buffers for every new batch shape; on a 16 GB
    # machine the process grew past 15 GB and thrashed swap. Fixed-length padding keeps one shape.
    tok = AutoTokenizer.from_pretrained(BASE, revision=d.TOKENIZER_REVISION)
    label2id = {lab: i for i, lab in enumerate(labels)}
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE,
        revision=d.TOKENIZER_REVISION,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id=label2id,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = (len(rows) + a.batch - 1) // a.batch
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    model.train()
    t0 = time.time()
    for step in range(steps):
        batch = rows[step * a.batch : (step + 1) * a.batch]
        texts, pairs = [b[0] for b in batch], [b[1] for b in batch]
        enc = tok(
            texts,
            pairs if pairs[0] is not None else None,
            truncation=True,
            max_length=a.max_len,
            padding="max_length",
            return_tensors="pt",
        ).to(device)
        y = torch.tensor([label2id[b[2]] for b in batch], device=device)
        loss = model(**enc, labels=y).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        if device == "mps" and step % 50 == 0:
            torch.mps.empty_cache()
        if step % 100 == 0:
            print(f"step {step}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)

    out = CKPT_DIR / a.dataset
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "train_meta.json").write_text(
        json.dumps(
            {
                "base": BASE,
                "revision": d.TOKENIZER_REVISION,
                "dataset": a.dataset,
                "examples": len(rows),
                "train_sha256": train_hash,
                "epochs": 1,
                "lr": a.lr,
                "batch": a.batch,
                "max_len": a.max_len,
                "device": device,
                "seconds": round(time.time() - t0),
            },
            indent=2,
        )
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
