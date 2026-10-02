"""Contrastive learning tests (InfoNCE + projection)."""

from __future__ import annotations

import pytest
import torch

from app.models.contrastive import ContrastiveProjection, InfoNCELoss


def test_infonce_prefers_close_positive():
    loss_function = InfoNCELoss(temperature=0.07)
    anchor = torch.tensor([[1.0, 0.0]])
    positive = torch.tensor([[1.0, 0.0]])

    good_negative = torch.tensor([[[-1.0, 0.0]]])  # opposite direction
    loss_good = loss_function(anchor, positive, good_negative)

    bad_negative = torch.tensor([[[1.0, 0.0]]])  # same direction (bad contrast)
    loss_bad = loss_function(anchor, positive, bad_negative)

    assert loss_good.item() < loss_bad.item()
    assert loss_good.item() < 1e-6  # nearly perfectly separable


def test_infonce_multi_negative_monotonic():
    loss_function = InfoNCELoss(temperature=0.2)
    anchor = torch.tensor([[1.0, 0.0]])
    positive = torch.tensor([[0.9, 0.1]])
    negatives_one = torch.tensor([[[-1.0, 0.0]]])
    negatives_two = torch.tensor([[[-1.0, 0.0], [0.5, 0.5]]])
    loss_one = loss_function(anchor, positive, negatives_one)
    loss_two = loss_function(anchor, positive, negatives_two)
    assert loss_two.item() > loss_one.item()


def test_infonce_gradients_flow():
    anchor = torch.randn(2, 8, requires_grad=True)
    positive = torch.randn(2, 8, requires_grad=True)
    negatives = torch.randn(2, 1, 8, requires_grad=True)
    loss = InfoNCELoss(temperature=0.1)(anchor, positive, negatives)
    loss.backward()
    assert anchor.grad is not None and torch.isfinite(anchor.grad).all()
    assert positive.grad is not None and torch.isfinite(positive.grad).all()
    assert negatives.grad is not None and torch.isfinite(negatives.grad).all()


def test_temperature_must_be_positive():
    with pytest.raises(ValueError):
        InfoNCELoss(temperature=0.0)


def test_projection_is_l2_normalised():
    projection = ContrastiveProjection(input_dim=16, projection_dim=8)
    z = projection(torch.randn(4, 16))
    assert z.shape == (4, 8)
    norms = z.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)
