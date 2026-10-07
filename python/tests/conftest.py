"""A tiny random model directory, written with the standard library: the tests need no weights or network."""

import json
import math
import random
import struct

import pytest

D, INTER, LAYERS = 32, 48, 3
SPECIAL = [("[UNK]", 0), ("<eos>", 1), ("<bos>", 2)]
WORDS = (
    "the customer was charged twice for march which team should handle this billing technical other "
    "how urgent is it low medium high did they ask a refund"
).split()
VOCAB = {**dict(SPECIAL), **{w: i + len(SPECIAL) for i, w in enumerate(WORDS)}}
V = 32
assert len(VOCAB) <= V

CONFIG = {
    "vocab_size": V,
    "hidden_size": D,
    "num_hidden_layers": LAYERS,
    "num_attention_heads": 2,
    "intermediate_size": INTER,
    "max_position_embeddings": 128,
    "layer_norm_eps": 1e-5,
    "global_attn_every_n_layers": 3,
    "global_rope_theta": 160000.0,
    "local_attention": 64,
    "local_rope_theta": 10000.0,
    "bos_token_id": 2,
    "eos_token_id": 1,
}
MANIFEST = {
    "model_id": "tiny",
    "late_layers": 1,
    "max_candidate_tokens": 32,
    "max_option_tokens": 14,
    "temperatures": {"noul": 1.5},
    "release_date": "2026-01-01",
}


def tokenizer() -> dict:
    """WordLevel on lowercased whitespace words; `<bos> … <eos>` around single sequences, like the real one."""
    special = {t: {"id": t, "ids": [i], "tokens": [t]} for t, i in SPECIAL if t != "[UNK]"}
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": i, "content": t, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False}
            | {"special": True}
            for t, i in SPECIAL
        ],
        "normalizer": {"type": "Lowercase"},
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "<bos>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "<eos>", "type_id": 0}},
            ],
            "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
            "special_tokens": special,
        },
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": VOCAB, "unk_token": "[UNK]"},
    }


def weights() -> dict[str, tuple[tuple[int, ...], list[float]]]:
    """The tensors `LateModel::load` reads: norms 1, everything else Gaussian (σ 0.5)."""
    rng = random.Random(0)

    def rand(*shape: int):
        return shape, [rng.gauss(0.0, 0.5) for _ in range(math.prod(shape))]

    def ones(n: int):
        return (n,), [1.0] * n

    t = {"encoder.embeddings.tok_embeddings.weight": rand(V, D), "encoder.embeddings.norm.weight": ones(D)}
    for i in range(LAYERS):
        p = f"encoder.layers.{i}."
        t[p + "attn.Wqkv.weight"] = rand(3 * D, D)
        t[p + "attn.Wo.weight"] = rand(D, D)
        if i:  # layer 0 has no attention norm
            t[p + "attn_norm.weight"] = ones(D)
        t[p + "mlp.Wi.weight"] = rand(2 * INTER, D)
        t[p + "mlp.Wo.weight"] = rand(D, INTER)
        t[p + "mlp_norm.weight"] = ones(D)
    t["encoder.final_norm.weight"] = ones(D)
    t["type_emb.weight"] = rand(3, D)
    t["scorer.0.weight"], t["scorer.0.bias"] = ones(D), rand(D)
    t["scorer.1.weight"], t["scorer.1.bias"] = rand(D, D), rand(D)
    t["scorer.3.weight"], t["scorer.3.bias"] = rand(1, D), rand(1)
    return t


def safetensors(tensors: dict[str, tuple[tuple[int, ...], list[float]]]) -> bytes:
    """8-byte little-endian header length, the JSON header (padded to 8 bytes), then little-endian f32 data."""
    header, blobs, offset = {}, [], 0
    for name, (shape, values) in tensors.items():
        blob = struct.pack(f"<{len(values)}f", *values)
        header[name] = {"dtype": "F32", "shape": list(shape), "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    return struct.pack("<Q", len(h)) + h + b"".join(blobs)


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny")
    (d / "config.json").write_text(json.dumps(CONFIG))
    (d / "krite.json").write_text(json.dumps(MANIFEST))
    (d / "tokenizer.json").write_text(json.dumps(tokenizer()))
    (d / "model.safetensors").write_bytes(safetensors(weights()))
    return d
