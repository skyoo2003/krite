"""Export a late-interaction checkpoint as a model directory for `krite serve --model <dir>`.

The directory holds `model.safetensors` (the torch state dict, same key names), the encoder's
`config.json` and `tokenizer.json` (the checkpoint re-saves them in a format the Rust side does not
read; the token ids are identical), `krite.json` (model id, tower shape, calibration temperatures), and
`probe.json`: torch CPU energies on synthetic text that the Rust weight tests compare against.

    uv run python -m krite_train.export --ckpt ckpt/late8
    uv run python -m krite_train.export --ckpt ckpt/late8 \\
        --calibration ../benchmarks/results/arch/calibration.jsonl --engine krite-raw

Temperatures are fitted on the probabilities an engine served, so they are absolute only when that
engine served raw probabilities (`krite serve --raw`); fitting on a calibrated engine would yield a
correction near 1.0 and silently drop the calibration. `require_raw` enforces this from engines.toml.
The directory is written to a staging directory and swapped in only when every file is written.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from krite_bench.data import TOKENIZER_REPO, TOKENIZER_REVISION
from safetensors.torch import save_file

from krite_train import model as m

ENGINES = Path(__file__).resolve().parents[2] / "benchmarks/baselines/engines.toml"

PROBE_CANDIDATES = [
    ("choice", "Which topic fits the text?", "sports", None),
    ("choice", "Which topic fits the text?", "world", "news about countries"),
    ("noul", "Is it positive?", "true", None),
    ("noul", "Is it positive?", "false", None),
    ("score", "How urgent is it?", "low", None),
    ("score", "How urgent is it?", "high", None),
    ("choice", " ".join(f"instruction{i}" for i in range(60)), "long", "x " * 40),
]


def require_raw(engines: Path, engine: str) -> None:
    """Exit unless `engine` serves uncalibrated probabilities (its start command passes `--raw`)."""
    start = tomllib.loads(engines.read_text()).get(engine, {}).get("start", "")
    if "--raw" not in start.split():
        raise SystemExit(
            f"engine {engine!r} does not serve raw probabilities (`--raw`); fit temperatures on one that does"
        )


def base_files(d: Path, meta: dict) -> None:
    """Copy `config.json` and `tokenizer.json` of the encoder the checkpoint was trained from."""
    repo, rev = meta.get("base", TOKENIZER_REPO), meta.get("revision", TOKENIZER_REVISION)
    for f in ("config.json", "tokenizer.json"):
        shutil.copyfile(hf_hub_download(repo, f, revision=rev), d / f)


def publish(out: Path, write: Callable[[Path], None]) -> None:
    """Run `write` into a staging directory, then swap it in for `out`; a failure leaves `out` untouched."""
    staging, old = out.with_name(out.name + ".staging"), out.with_name(out.name + ".old")
    for d in (staging, old):
        shutil.rmtree(d, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        write(staging)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    # ponytail: two renames leave `out` missing for an instant; a versioned symlink if a server reloads live
    if out.exists():
        out.rename(old)
    staging.rename(out)
    shutil.rmtree(old, ignore_errors=True)


def temperatures(path: Path | None, engine: str) -> dict[str, float]:
    """Per-bucket temperatures of `engine` from calibration.jsonl rows; the latest row wins."""
    if path is None:
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = sorted((r for r in rows if r["engine"] == engine), key=lambda r: r["timestamp"])
    out: dict[str, float] = {}
    for r in rows:
        out.update(r["temperature"])
    if not out:
        print(f"no calibration rows for {engine!r} in {path}; temperatures stay 1.0", file=sys.stderr)
    return out


@torch.no_grad()
def probe(net: m.LateModel, tok) -> dict:
    state_text = " ".join(f"word{i}" for i in range(150))
    sids = m.state_ids(tok, state_text)
    ids, mask = m.pad([sids])
    mem = net.encode_state(ids, mask)
    cands = [
        {"kind": k, "instructions": ins, "name": n, "desc": d, "ids": m.candidate_ids(tok, ins, n, d)}
        for k, ins, n, d in PROBE_CANDIDATES
    ]
    cid, cm = m.pad([c["ids"] for c in cands])
    owner = torch.zeros(len(cands), dtype=torch.long)
    qt = torch.tensor([m.QTYPES[c["kind"]] for c in cands])
    e = net.energies(mem, mask, owner, cid, cm, qt)
    return {"state_text": state_text, "state_ids": sids, "candidates": cands, "energies": e.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, help="default: <ckpt>/candle")
    ap.add_argument("--model-id", default="krite-0.15b-v0")
    ap.add_argument("--calibration", type=Path, help="calibration.jsonl from krite-bench quality")
    ap.add_argument("--engine", default="krite-raw", help="engine whose temperatures to ship; must serve --raw")
    a = ap.parse_args()
    out = a.out or a.ckpt / "candle"
    if a.calibration:
        require_raw(ENGINES, a.engine)
    arm, net, tok = m.load(a.ckpt, "cpu")
    if not isinstance(net, m.LateModel):
        raise SystemExit("export supports late-interaction arms only")
    temps = temperatures(a.calibration, a.engine)
    meta = json.loads((a.ckpt / "train_meta.json").read_text())

    def write(d: Path) -> None:
        save_file({k: v.contiguous() for k, v in net.state_dict().items()}, d / "model.safetensors")
        base_files(d, meta)
        manifest = {
            "model_id": a.model_id,
            "arm": arm,
            "late_layers": m.ARMS[arm]["late_layers"],
            "max_candidate_tokens": m.MAX_CANDIDATE,
            "max_option_tokens": m.MAX_OPTION,
            "temperatures": temps,
            "source": {"ckpt": str(a.ckpt), "train_sha256": meta["train_sha256"], "seed": meta["seed"]},
        }
        (d / "krite.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (d / "probe.json").write_text(json.dumps(probe(net, tok)) + "\n")

    publish(out, write)
    for f in sorted(out.iterdir()):
        print(f"{f}  {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
