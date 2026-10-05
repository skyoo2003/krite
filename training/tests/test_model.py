"""Architectural invariants on a tiny random encoder (CPU, eval mode, no downloads)."""

import torch
from transformers import ModernBertConfig, ModernBertModel

from krite_train.model import NEG, JointModel, LateModel, TowerModel

TOL = 1e-5


def encoder() -> ModernBertModel:
    torch.manual_seed(0)
    cfg = ModernBertConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        global_attn_every_n_layers=1,
        local_attention=8,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
        cls_token_id=1,
        sep_token_id=1,
    )
    return ModernBertModel(cfg)


def padded(rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, rows))
    ids = torch.tensor([r + [0] * (width - len(r)) for r in rows])
    return ids, (ids != 0).long()


VARIANTS = {
    "shared": {"option_encoder": "shared"},
    "emb": {"option_encoder": "emb"},
    "pool": {"state_pool": True},
    "set": {"set_attention": True},
    "late": None,
}


def tower(variant: str):
    torch.manual_seed(1)
    if VARIANTS[variant] is None:
        return LateModel(encoder(), late_layers=1).eval()
    return TowerModel(encoder(), layers=2, **VARIANTS[variant]).eval()


STATE = [2, 10, 11, 12, 13, 14, 1]
CANDS = [[2, 20, 21, 1], [2, 22, 1], [2, 23, 24, 25, 26, 1], [2, 27, 28, 1], [2, 29, 30, 31, 1]]


@torch.no_grad()
def score(model, cands: list[list[int]], qtypes: list[int], group: list[int] | None = None) -> torch.Tensor:
    mem_ids, mem_mask = padded([STATE])
    mem = model.encode_state(mem_ids, mem_mask)
    ids, mask = padded(cands)
    owner = torch.zeros(len(cands), dtype=torch.long)
    g = torch.tensor(group) if group else owner
    return model.energies(mem, mem_mask, owner, ids, mask, torch.tensor(qtypes), g)


def test_tower_is_permutation_equivariant():
    for mode in VARIANTS:
        m = tower(mode)
        base = score(m, CANDS, [0] * 5)
        perm = [3, 0, 4, 1, 2]
        permuted = score(m, [CANDS[i] for i in perm], [0] * 5)
        assert (permuted - base[perm]).abs().max() <= TOL, mode


def test_tower_question_isolation():
    for mode in VARIANTS:
        m = tower(mode)
        alone = score(m, CANDS[:2], [0, 0])
        other = [[2, 40, 41, 42, 43, 44, 45, 46, 1], [2, 47, 1]]  # longer rows change the padding
        together = score(m, CANDS[:2] + other, [0, 0, 2, 2], group=[0, 0, 1, 1])
        assert (together[:2] - alone).abs().max() <= TOL, mode


@torch.no_grad()
def test_late_energies_depend_on_the_state():
    m = tower("late")
    ids, mask = padded(CANDS[:2])
    owner = torch.zeros(2, dtype=torch.long)
    out = []
    for state in (STATE, [2, 50, 51, 52, 1]):
        s_ids, s_mask = padded([state])
        out.append(
            m.energies(m.encode_state(s_ids, s_mask), s_mask, owner, ids, mask, torch.zeros(2, dtype=torch.long))
        )
    assert (out[0] - out[1]).abs().max() > 1e-4


@torch.no_grad()
def test_tower_state_memory_ignores_batch_neighbours():
    m = tower("shared")
    ids, mask = padded([STATE])
    alone = m.encode_state(ids, mask)
    ids2, mask2 = padded([STATE, [2, 50, 51, 52, 53, 54, 55, 56, 57, 1]])
    assert (m.encode_state(ids2, mask2)[0, : len(STATE)] - alone[0]).abs().max() <= TOL


@torch.no_grad()
def test_joint_gathers_one_energy_per_marker():
    torch.manual_seed(2)
    m = JointModel(encoder(), head_layers=2).eval()
    ids, mask = padded([[2, 5, 6, 1, 4, 20, 4, 21, 4, 22, 1, 10, 11, 1], [2, 5, 1, 4, 23, 1, 12, 1]])
    marker_pos = torch.tensor([[4, 6, 8], [3, -1, -1]])
    marker_mask = marker_pos >= 0
    e = m(ids, mask, marker_pos, marker_mask, torch.tensor([0, 0]))
    assert e.shape == (2, 3)
    assert (e[1, 1:] == NEG).all() and (e[0] > NEG).all()


@torch.no_grad()
def test_late_cached_lower_layers_match():
    """The shim caches each candidate's lower-layer states alone and re-pads them."""
    m = tower("late")
    s_ids, s_mask = padded([STATE])
    mem = m.encode_state(s_ids, s_mask)
    ids, mask = padded(CANDS)
    owner, qt = torch.zeros(5, dtype=torch.long), torch.zeros(5, dtype=torch.long)
    lower = torch.zeros(5, ids.size(1), 32)
    for i, c in enumerate(CANDS):
        lower[i, : len(c)] = m.lower(*padded([c]))[0]
    full = m.energies(mem, s_mask, owner, ids, mask, qt)
    assert (m.energies(mem, s_mask, owner, ids, mask, qt, lower=lower) - full).abs().max() <= TOL


@torch.no_grad()
def test_late_packed_row_matches_one_row_per_candidate():
    m = tower("late")
    s_ids, s_mask = padded([STATE])
    ids, mask = padded(CANDS)
    owner, qt = torch.zeros(5, dtype=torch.long), torch.zeros(5, dtype=torch.long)
    packed = m.energies(m.encode_state(s_ids, s_mask), s_mask, owner, ids, mask, qt)
    two_ids, two_mask = padded([STATE, STATE])  # two state rows take the per-candidate path
    rows = m.energies(m.encode_state(two_ids, two_mask), two_mask, owner, ids, mask, qt)
    assert (packed - rows).abs().max() <= TOL
