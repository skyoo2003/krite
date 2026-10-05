"""Architecture-study models: the joint encoder (B) and the state-memory decision tower (D).

Both start from the same mmBERT-small encoder. B reads question, options, and state in one sequence
(Laya-style), so it re-encodes the state per question and depends on option order. D encodes the
state alone (cacheable) and scores every candidate independently against it, so permuting candidates
permutes energies and questions never see each other. Token layouts live here so training and the
serving shim build byte-identical inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from krite_bench.data import TOKENIZER_REPO, TOKENIZER_REVISION
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

QTYPES = {"choice": 0, "score": 1, "noul": 2}
NEG = -1e4  # energy of padded candidates
MAX_STATE_TRAIN = 254  # content tokens per state during training
MAX_CANDIDATE = 32  # tokens per D candidate, <bos>/<eos> included (also the tower's position table)
MAX_OPTION = 14  # content tokens kept from "\n<name>[: <desc>]" before the instructions get the rest
JOINT_HEAD = 128  # tokens before the state in a B sequence

ARMS = {
    "b": {"kind": "joint", "head_layers": 2},
    "d1": {"kind": "tower", "layers": 1, "option_encoder": "shared"},
    "d2": {"kind": "tower", "layers": 2, "option_encoder": "shared"},
    "d4": {"kind": "tower", "layers": 4, "option_encoder": "shared"},
    "d2-emb": {"kind": "tower", "layers": 2, "option_encoder": "emb"},
    # Follow-up arms on d1 (docs/architecture-study.md, "Follow-up"): one change each.
    "d1-pool": {"kind": "tower", "layers": 1, "option_encoder": "shared", "state_pool": True},
    "d1-lr": {"kind": "tower", "layers": 1, "option_encoder": "shared", "lr_new": 1e-3, "epochs": 2},
    "d1-set": {"kind": "tower", "layers": 1, "option_encoder": "shared", "set_attention": True},
    "late4": {"kind": "late", "late_layers": 4},
    # Second round on late4: deeper interaction, and a second seed (data sample + init) for late4 and b.
    "late8": {"kind": "late", "late_layers": 8},
    "late4-s14": {"kind": "late", "late_layers": 4, "seed": 14},
    "b-s14": {"kind": "joint", "head_layers": 2, "seed": 14},
    # Third round: between late4 and late8, and a second seed for late8.
    "late6": {"kind": "late", "late_layers": 6},
    "late8-s14": {"kind": "late", "late_layers": 8, "seed": 14},
    # Release recipe (docs/training-data.md), stage A: the broad mixture.
    "late8-broad": {"kind": "late", "late_layers": 8, "mixture": "broad"},
    "late8-broad-s14": {"kind": "late", "late_layers": 8, "mixture": "broad", "seed": 14},
}
MODEL_KEYS = ("layers", "option_encoder", "state_pool", "set_attention", "small_init")


def scorer(d: int) -> nn.Module:
    return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))


# Adapted from Laya (Apache-2.0), laya/common.py DecisionModel; act head and temperature removed.
class JointModel(nn.Module):
    kind = "joint"

    def __init__(self, encoder: nn.Module, head_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        layer = nn.TransformerEncoderLayer(d, max(1, d // 64), 4 * d, dropout, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
        self.type_emb = nn.Embedding(len(QTYPES), d)
        self.scorer = scorer(d)

    def forward(self, ids, mask, marker_pos, marker_mask, qtype) -> torch.Tensor:
        """Energies (rows, K_max); padded markers get NEG."""
        h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        h = self.head(h + self.type_emb(qtype)[:, None], src_key_padding_mask=~mask.bool())
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        e = self.scorer(torch.gather(h, 1, idx)).squeeze(-1)
        return e.masked_fill(~marker_mask, NEG)


class TowerLayer(nn.Module):
    """Pre-norm: self-attention within one candidate, cross-attention to its state memory, FFN."""

    def __init__(self, d: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.cross = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, x_pad, mem, mem_pad):
        h = self.n1(x)
        x = x + self.drop(self.self_attn(h, h, h, key_padding_mask=x_pad, need_weights=False)[0])
        h = self.n2(x)
        x = x + self.drop(self.cross(h, mem, mem, key_padding_mask=mem_pad, need_weights=False)[0])
        return x + self.drop(self.ff(self.n3(x)))


class TowerModel(nn.Module):
    """State memory + per-candidate tower. Optional: a pooled-state term in the scorer (`state_pool`)
    and one set-attention layer between the pooled candidates of a question (`set_attention`); both
    keep candidates permutation-equivariant and questions isolated."""

    kind = "tower"

    def __init__(
        self,
        encoder: nn.Module,
        layers: int = 2,
        option_encoder: str = "shared",
        dropout: float = 0.1,
        state_pool: bool = False,
        set_attention: bool = False,
        small_init: bool = False,
    ):
        super().__init__()
        if option_encoder not in ("shared", "emb"):
            raise ValueError(f"option_encoder must be shared or emb, not {option_encoder!r}")
        self.encoder, self.option_encoder = encoder, option_encoder
        d, heads = encoder.config.hidden_size, max(1, encoder.config.hidden_size // 64)
        self.layers = nn.ModuleList(TowerLayer(d, heads, dropout) for _ in range(layers))
        self.type_emb = nn.Embedding(len(QTYPES), d)
        self.pos_emb = nn.Embedding(MAX_CANDIDATE, d)
        if small_init:  # nn.Embedding defaults to N(0, 1): as large as the normalized encoder output
            nn.init.normal_(self.type_emb.weight, std=0.02)
            nn.init.normal_(self.pos_emb.weight, std=0.02)
        self.norm = nn.LayerNorm(d)
        self.state_proj = nn.Linear(d, d) if state_pool else None
        self.set_norm = nn.LayerNorm(d) if set_attention else None
        self.set_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True) if set_attention else None
        self.scorer = scorer(3 * d if state_pool else d)

    def encode_state(self, ids, mask) -> torch.Tensor:
        """State memory H_state (rows, S, d). Never sees questions."""
        return self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state

    def energies(self, mem, mem_mask, owner, cand_ids, cand_mask, qtype, group=None) -> torch.Tensor:
        """One energy per candidate row. `owner[i]` is the memory row candidate i attends to; `group[i]` its
        question (set attention only; defaults to `owner`, one question per memory row)."""
        if self.option_encoder == "shared":
            x = self.encoder(input_ids=cand_ids, attention_mask=cand_mask).last_hidden_state
        else:
            x = self.encoder.embeddings(input_ids=cand_ids)
        x = x + self.type_emb(qtype)[:, None] + self.pos_emb.weight[: x.size(1)]
        pad = ~cand_mask.bool()
        m, m_pad = mem[owner], ~mem_mask.bool()[owner]
        for layer in self.layers:
            x = layer(x, pad, m, m_pad)
        w = cand_mask[..., None].to(x.dtype)
        c = (self.norm(x) * w).sum(1) / w.sum(1)
        if self.set_attn is not None:
            g = owner if group is None else group
            h = self.set_norm(c)[None]
            c = c + self.set_attn(h, h, h, attn_mask=g[:, None] != g[None, :], need_weights=False)[0][0]
        if self.state_proj is not None:
            mw = mem_mask[..., None].to(mem.dtype)
            s = self.state_proj(((mem * mw).sum(1) / mw.sum(1))[owner])
            c = torch.cat([c, s, c * s], -1)
        return self.scorer(c).squeeze(-1)


class LateModel(nn.Module):
    """Late interaction inside the encoder: candidates run the lower layers alone, then in the top
    `late_layers` layers their tokens also attend to the state's hidden states at that layer (pretrained
    attention weights). The state side never sees candidates, so its per-layer states are cacheable."""

    kind = "tower"

    def __init__(self, encoder: nn.Module, late_layers: int = 4):
        super().__init__()
        self.encoder, self.split = encoder, len(encoder.layers) - late_layers
        d = encoder.config.hidden_size
        self.type_emb = nn.Embedding(len(QTYPES), d)
        nn.init.normal_(self.type_emb.weight, std=0.02)
        self.scorer = scorer(d)

    def _masks(self, h, mask):
        from transformers.masking_utils import create_bidirectional_mask, create_bidirectional_sliding_window_mask

        kw = {"config": self.encoder.config, "inputs_embeds": h, "attention_mask": mask}
        return {
            "full_attention": create_bidirectional_mask(**kw),
            "sliding_attention": create_bidirectional_sliding_window_mask(**kw),
        }

    def _rope(self, h, positions):
        return {t: self.encoder.rotary_emb(h, positions, t) for t in set(self.encoder.config.layer_types)}

    def _run(self, h, mask, layers):
        masks, rope = self._masks(h, mask), self._rope(h, torch.arange(h.size(1), device=h.device)[None])
        for layer in layers:
            h = layer(h, attention_mask=masks[layer.attention_type], position_embeddings=rope[layer.attention_type])
        return h

    def encode_state(self, ids, mask) -> torch.Tensor:
        """Rotated keys and values of the state in each top layer: (late_layers, 2, rows, heads, S, head_dim)."""
        from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb

        enc = self.encoder
        h = self._run(enc.embeddings(input_ids=ids), mask, enc.layers[: self.split])
        heads = enc.config.num_attention_heads
        pos = torch.arange(h.size(1), device=h.device)[None]
        out = []
        for i, layer in enumerate(enc.layers[self.split :]):
            k, v = layer.attn.Wqkv(layer.attn_norm(h)).view(*h.shape[:2], 3, heads, -1)[:, :, 1:].unbind(2)
            cos, sin = enc.rotary_emb(h, pos, layer.attention_type)
            k, _ = apply_rotary_pos_emb(k.transpose(1, 2), k.transpose(1, 2), cos, sin)
            out.append(torch.stack([k, v.transpose(1, 2)]))
            if i < len(enc.layers) - self.split - 1:
                h = self._run(h, mask, [layer])
        return torch.stack(out)

    def lower(self, cand_ids, cand_mask) -> torch.Tensor:
        """Candidate hidden states after the lower layers; depends on the candidate text only."""
        return self._run(self.encoder.embeddings(input_ids=cand_ids), cand_mask, self.encoder.layers[: self.split])

    def energies(self, mem, mem_mask, owner, cand_ids, cand_mask, qtype, group=None, lower=None) -> torch.Tensor:
        """`lower` (rows, T, d), when given, replaces running the lower layers on `cand_ids`.

        With one state (serving), all candidates go in one packed row with a block-diagonal mask, so the
        state's keys and values are not copied per candidate; the result is the same as one row each.
        """
        from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb

        enc = self.encoder
        x = self.lower(cand_ids, cand_mask) if lower is None else lower
        n, t, d = x.shape
        heads = enc.config.num_attention_heads
        c_pos = mem_mask.sum(1)[owner][:, None] + torch.arange(t, device=x.device)[None]  # after their state
        packed = mem.size(2) == 1
        if packed:
            cand = torch.arange(n, device=x.device).repeat_interleave(t)
            own = (cand[:, None] == cand[None, :]) & cand_mask.bool().reshape(1, n * t)
            keep = torch.cat([mem_mask.bool().expand(n * t, -1), own], 1)[None, None]
            x, c_pos, rows, length = x.reshape(1, n * t, d), c_pos.reshape(1, n * t), 1, n * t
        else:
            keep = torch.cat([mem_mask.bool()[owner], cand_mask.bool()], 1)[:, None, None, :]
            rows, length = n, t
        bias = torch.zeros(keep.shape, dtype=x.dtype, device=x.device).masked_fill(~keep, float("-inf"))
        late = enc.layers[self.split :]
        rope = {a: enc.rotary_emb(x, c_pos, a) for a in {layer.attention_type for layer in late}}
        for i, layer in enumerate(late):
            attn = layer.attn
            qkv = attn.Wqkv(layer.attn_norm(x)).view(rows, length, 3, heads, -1).unbind(2)
            q, k, v = (z.transpose(1, 2) for z in qkv)
            q, k = apply_rotary_pos_emb(q, k, *rope[layer.attention_type])
            sk, sv = (mem[i, 0], mem[i, 1]) if packed else (mem[i, 0][owner], mem[i, 1][owner])
            o = nn.functional.scaled_dot_product_attention(
                q,
                torch.cat([sk, k], 2),
                torch.cat([sv, v], 2),
                attn_mask=bias,
                dropout_p=attn.attention_dropout if self.training else 0.0,
            )
            x = x + attn.out_drop(attn.Wo(o.transpose(1, 2).reshape(rows, length, d)))
            x = x + layer.mlp(layer.mlp_norm(x))
        x = x.reshape(n, t, d)
        x = enc.final_norm(x)
        w = cand_mask[..., None].to(x.dtype)
        c = (x * w).sum(1) / w.sum(1) + self.type_emb(qtype)
        return self.scorer(c).squeeze(-1)


# --- token layouts ---------------------------------------------------------------------------------


def pad(rows: list[list[int]], width: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad id rows with 0 (<pad>); returns (ids, attention mask)."""
    width = width or max(map(len, rows))
    ids = torch.zeros((len(rows), width), dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.tensor(r)
        mask[i, : len(r)] = 1
    return ids, mask


def specials(tok) -> dict[str, int]:
    return {"bos": tok.bos_token_id, "eos": tok.eos_token_id, "sep": tok.sep_token_id, "mask": tok.mask_token_id}


def _ids(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False)["input_ids"]


def option_text(name: str, desc: str | None) -> str:
    return f"{name}: {desc}" if desc else name


def state_ids(tok, state: str, limit: int | None = None) -> list[int]:
    sp, ids = specials(tok), _ids(tok, state)
    return [sp["bos"], *(ids[:limit] if limit else ids), sp["eos"]]


def candidate_ids(tok, instructions: str, name: str, desc: str | None) -> list[int]:
    """`<bos> instructions \\n name[: desc] <eos>`, at most MAX_CANDIDATE tokens; the option keeps priority."""
    sp = specials(tok)
    opt = _ids(tok, "\n" + option_text(name, desc))[:MAX_OPTION]
    ins = _ids(tok, instructions)[: MAX_CANDIDATE - 2 - len(opt)]
    return [sp["bos"], *ins, *opt, sp["eos"]]


def joint_ids(tok, qtype: str, instructions: str, options: list[str], state: list[int]) -> tuple[list[int], list[int]]:
    """`<bos> "<type> question: <instructions>" <sep> (<mask> option)×K <sep> state <eos>`; returns (ids, markers).

    `state` is content ids (no specials). Options are cut evenly when the head would exceed JOINT_HEAD.
    """
    sp = specials(tok)
    opts = [[sp["mask"], *_ids(tok, " " + o)[:30]] for o in options]
    room = JOINT_HEAD - 3 - sum(map(len, opts))
    if room < 8:
        per = max(2, (JOINT_HEAD - 11) // len(opts))
        opts = [o[:per] for o in opts]
        room = JOINT_HEAD - 3 - sum(map(len, opts))
    ids = [sp["bos"], *_ids(tok, f"{qtype} question: {instructions}")[: max(1, room)], sp["sep"]]
    markers = []
    for o in opts:
        markers.append(len(ids))
        ids += o
    return [*ids, sp["sep"], *state, sp["eos"]], markers


# --- build, save, load -----------------------------------------------------------------------------


def build(arm: str, encoder: nn.Module | None = None) -> nn.Module:
    spec = ARMS[arm]
    if encoder is None:
        encoder = AutoModel.from_pretrained(TOKENIZER_REPO, revision=TOKENIZER_REVISION, dtype=torch.float32)
    if spec["kind"] == "joint":
        return JointModel(encoder, spec["head_layers"])
    if spec["kind"] == "late":
        return LateModel(encoder, spec["late_layers"])
    return TowerModel(encoder, **{k: v for k, v in spec.items() if k in MODEL_KEYS})


def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_REPO, revision=TOKENIZER_REVISION)


def save(model: nn.Module, arm: str, tok, out: Path, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    model.encoder.config.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "arm.json").write_text(json.dumps({"arm": arm, **ARMS[arm]}, indent=2) + "\n")
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def load(path: Path, device: str) -> tuple[str, nn.Module, object]:
    arm = json.loads((path / "arm.json").read_text())["arm"]
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(path))
    model = build(arm, encoder)
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True))
    return arm, model.to(device).eval(), AutoTokenizer.from_pretrained(path)
