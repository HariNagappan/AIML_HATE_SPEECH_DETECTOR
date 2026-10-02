"""Context–comment interaction.

Implements exactly the operations of the project architecture::

    D = |E_c − E_p|
    M = E_c ⊙ E_p
    E_int = [E_c ; E_p ; D ; M]          # 4 × hidden_size = 3072 for BERT-base

followed by a configurable fully connected interaction layer::

    4·hidden → interaction_dim → GELU/ReLU → Dropout → E_int_processed

Two modes support the ablations ("BERT + context" vs "BERT + context
interaction"):

* ``mlp``    – the full ``[E_c; E_p; D; M]`` vector → MLP (proposed model)
* ``simple`` – ``[E_c; E_p]`` only, without the engineered D/M features
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


def compute_interaction_components(
    e_c: torch.Tensor, e_p: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Return ``D``, ``M`` and the raw concatenation ``[E_c; E_p; D; M]``.

    Shapes: ``D``/``M`` are ``[batch, hidden]``, ``concat`` is
    ``[batch, 4 * hidden]`` (3072 for BERT-base with hidden size 768).
    """
    d = torch.abs(e_c - e_p)
    m = e_c * e_p
    return {"D": d, "M": m, "concat": torch.cat([e_c, e_p, d, m], dim=-1)}


class ContextInteraction(nn.Module):
    """Configurable interaction layer (see module docstring)."""

    def __init__(
        self,
        hidden_size: int = 768,
        interaction_dim: int = 1024,
        dropout: float = 0.2,
        activation: str = "gelu",
        mode: str = "mlp",
    ):
        super().__init__()
        if mode not in {"mlp", "simple"}:
            raise ValueError(f"Unknown interaction mode: {mode!r}")
        self.hidden_size = int(hidden_size)
        self.interaction_dim = int(interaction_dim)
        self.mode = mode
        self.concat_size = self.hidden_size * (4 if mode == "mlp" else 2)
        self.projection = nn.Linear(self.concat_size, self.interaction_dim)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    @property
    def output_dim(self) -> int:
        return self.interaction_dim

    def forward(
        self, e_c: torch.Tensor, e_p: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return the processed context-aware representation ``[B, interaction_dim]``.

        ``E_p = None`` (no context available) uses the documented fallback:
        a zero context embedding (then ``D = |E_c|``, ``M = 0``). This is a
        deliberate, simple choice — the design keeps a future N-context
        extension open.
        """
        if e_p is None:
            e_p = torch.zeros_like(e_c)
        if self.mode == "mlp":
            raw = compute_interaction_components(e_c, e_p)["concat"]
        else:
            raw = torch.cat([e_c, e_p], dim=-1)
        return self.dropout(self.activation(self.projection(raw)))
