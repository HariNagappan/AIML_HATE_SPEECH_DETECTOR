"""Context interaction tests — exact D / M / concatenation dimensions."""

from __future__ import annotations

import torch

from app.models.context_interaction import (
    ContextInteraction,
    compute_interaction_components,
)


def test_interaction_components_768_dims():
    e_c = torch.randn(2, 768)
    e_p = torch.randn(2, 768)
    components = compute_interaction_components(e_c, e_p)

    # Project spec §27: D [batch,768], M [batch,768], E_int [batch,3072].
    assert components["D"].shape == (2, 768)
    assert components["M"].shape == (2, 768)
    assert components["concat"].shape == (2, 3072)


def test_interaction_values_are_correct():
    e_c = torch.tensor([[1.0, -2.0]])
    e_p = torch.tensor([[0.5, -3.0]])
    components = compute_interaction_components(e_c, e_p)
    assert torch.allclose(components["D"], torch.tensor([[0.5, 1.0]]))
    assert torch.allclose(components["M"], torch.tensor([[0.5, 6.0]]))


def test_interaction_layer_output_dim():
    layer = ContextInteraction(hidden_size=768, interaction_dim=1024)
    out = layer(torch.randn(4, 768), torch.randn(4, 768))
    assert out.shape == (4, 1024)


def test_no_context_fallback_shape():
    layer = ContextInteraction(hidden_size=768, interaction_dim=1024)
    out = layer(torch.randn(3, 768), None)
    assert out.shape == (3, 1024)


def test_simple_mode_dims():
    layer = ContextInteraction(hidden_size=64, interaction_dim=32, mode="simple")
    assert layer.concat_size == 128
    out = layer(torch.randn(2, 64), torch.randn(2, 64))
    assert out.shape == (2, 32)


def test_unknown_mode_raises():
    import pytest

    with pytest.raises(ValueError):
        ContextInteraction(hidden_size=64, mode="bogus")
