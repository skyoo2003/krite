"""Fixed-label classifier baseline behind Protocol v1.

The request `model` field (`classifier-<dataset>`) selects the checkpoint. The softmax runs over the
checkpoint's full label set and is then restricted to the offered options and renormalized.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402
import torch  # noqa: E402
from common import RequestError, candidates, pick_device  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

CKPT_DIR = Path(__file__).resolve().parents[1] / "classifier_ckpt"
MAX_LEN = 256


class Classifiers:
    def __init__(self, device: str):
        self.device, self.loaded = device, {}

    def get(self, dataset: str):
        if dataset not in self.loaded:
            path = CKPT_DIR / dataset
            if not path.exists():
                raise RequestError(f"no checkpoint for {dataset!r}", "model", "unknown_model")
            tok = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()
            self.loaded[dataset] = (tok, model)
        return self.loaded[dataset]


def text_inputs(state: str) -> tuple[str, str | None]:
    """Mirror classifier_train.py: pair datasets are (question, passage) / (premise, hypothesis)."""
    try:
        obj = json.loads(state)
    except ValueError:
        return state, None
    if isinstance(obj, dict) and "passage" in obj:
        return obj["question"], obj["passage"]
    if isinstance(obj, dict) and "premise" in obj:
        return obj["premise"], obj["hypothesis"]
    return state, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    device = pick_device()
    models = Classifiers(device)
    current = threading.local()  # ThreadingHTTPServer: validate() and decide() share the request thread

    @torch.no_grad()
    def decide(state: str, questions: dict) -> dict:
        tok, model = models.get(current.dataset)
        text, pair = text_inputs(state)
        enc = tok(text, pair, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
        probs = torch.softmax(model(**enc).logits[0].float(), -1).tolist()
        by_label = {model.config.id2label[i]: p for i, p in enumerate(probs)}
        out = {}
        for qid, q in questions.items():
            names = candidates(q)
            missing = [n for n in names if n not in by_label]
            if missing:
                raise RequestError(f"options not in label set: {missing[:3]}", f"questions.{qid}.criteria")
            total = sum(by_label[n] for n in names)
            out[qid] = {n: by_label[n] / total for n in names}
        return out

    # The request `model` picks the checkpoint; wrap validate() to capture it before decide() runs.
    original_validate = common.validate

    def validate(body: dict) -> None:
        original_validate(body)
        model_id = body.get("model") or ""
        dataset = model_id.removeprefix("classifier-")
        if not model_id.startswith("classifier-") or not dataset or "/" in dataset or dataset.startswith("."):
            raise RequestError("model must be classifier-<dataset>", "model", "unknown_model")
        current.dataset = dataset

    common.validate = validate
    common.serve(decide, a.port, "classifier-mmbert-small", device)


if __name__ == "__main__":
    main()
