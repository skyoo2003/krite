"""Training losses on tiny tensors (no encoder, no downloads)."""

import torch
import torch.nn.functional as F

from krite_train import model as m
from krite_train import train

SCORE, CHOICE = m.QTYPES["score"], m.QTYPES["choice"]


def test_default_spec_is_plain_cross_entropy():
    e = torch.randn(4, 3)
    gold = torch.tensor([0, 2, 1, 0])
    qtype = torch.tensor([CHOICE, SCORE, CHOICE, SCORE])
    assert torch.equal(train.losses(e, gold, qtype, {}), F.cross_entropy(e, gold))


def test_brier_adds_squared_error():
    e = torch.log(torch.tensor([[0.75, 0.25]]))
    gold = torch.tensor([0])
    got = train.losses(e, gold, torch.tensor([CHOICE]), {"brier": True})
    want = -torch.log(torch.tensor(0.75)) + train.BRIER_W * (0.25**2 + 0.25**2)
    assert torch.allclose(got, want)


def test_ordinal_only_on_score_rows_and_grows_with_distance():
    gold = torch.tensor([0])
    near = torch.log(torch.tensor([[0.5, 0.5, 1e-9, 1e-9, 1e-9]]))
    far = torch.log(torch.tensor([[0.5, 1e-9, 1e-9, 1e-9, 0.5]]))

    def extra(e, qtype):
        q = torch.tensor([qtype])
        return train.losses(e, gold, q, {"ordinal": True}) - train.losses(e, gold, q, {})

    assert extra(far, CHOICE).abs() < 1e-6
    assert extra(near, SCORE) > 0 and extra(far, SCORE) > extra(near, SCORE)


def test_losses_have_gradients():
    e = torch.randn(3, 5, requires_grad=True)
    loss = train.losses(
        e, torch.tensor([0, 4, 2]), torch.tensor([SCORE, CHOICE, SCORE]), {"brier": True, "ordinal": True}
    )
    loss.backward()
    assert e.grad is not None and torch.isfinite(e.grad).all()
