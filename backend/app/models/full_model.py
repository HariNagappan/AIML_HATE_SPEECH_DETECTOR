"""Full model: shared BERT → context interaction → multi-task heads.

Architecture (see README)::

    current comment ─┐
                     ├─ shared BERT ─→ E_c, E_p ─→ interaction ─→ final representation
    previous context ┘       │
                             ▼
         ┌───────────────┬───────────────┬──────────────────┐
         ▼               ▼               ▼                  ▼
     hate head      target head     reason head     evidence head (token-level)
                                    + contrastive projection (training only)

Every component can be switched off via constructor flags — the ablation
experiments rely on this (see ``evaluation/ablation.py``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from app.models.bert_encoder import BertEncoder
from app.models.context_interaction import ContextInteraction
from app.models.contrastive import ContrastiveProjection
from app.models.evidence_head import EvidenceHead
from app.models.hate_head import HateHead
from app.models.reason_head import ReasonHead
from app.models.target_head import TargetHead


class FullModel(nn.Module):
    architecture = "full"

    def __init__(
        self,
        encoder: BertEncoder,
        hate_labels,
        target_labels=None,
        reason_labels=None,
        *,
        interaction_dim: int = 1024,
        interaction_dropout: float = 0.2,
        interaction_mode: str = "mlp",
        use_context: bool = True,
        use_contrastive: bool = True,
        use_target: bool = True,
        use_reason: bool = False,
        use_evidence: bool = True,
        contrastive_dim: int = 256,
        hate_mode: str = "multiclass",
        hate_threshold: float = 0.5,
    ):
        super().__init__()
        self.encoder = encoder
        self.use_context = bool(use_context)

        hidden_size = encoder.hidden_size
        self.interaction: Optional[ContextInteraction] = None
        if self.use_context:
            self.interaction = ContextInteraction(
                hidden_size=hidden_size,
                interaction_dim=interaction_dim,
                dropout=interaction_dropout,
                mode=interaction_mode,
            )
            self.final_dim = int(interaction_dim)
        else:
            self.final_dim = int(hidden_size)

        self.hate_head = HateHead(
            self.final_dim, hate_labels, mode=hate_mode, threshold=hate_threshold
        )
        self.target_head = (
            TargetHead(self.final_dim, target_labels)
            if (use_target and target_labels)
            else None
        )
        self.reason_head = (
            ReasonHead(self.final_dim, reason_labels)
            if (use_reason and reason_labels)
            else None
        )
        self.evidence_head = EvidenceHead(hidden_size) if use_evidence else None
        self.contrastive_projection = (
            ContrastiveProjection(self.final_dim, contrastive_dim)
            if (use_contrastive and self.use_context)
            else None
        )

    @property
    def supports_context(self) -> bool:
        return self.use_context

    # --- encoding helpers --------------------------------------------------------
    def _encode_current(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        current_inputs_embeds: Optional[torch.Tensor] = None,
    ):
        encoded = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=current_inputs_embeds,
        )
        return encoded["cls"], encoded["last_hidden_state"]

    def _encode_context(
        self,
        context_input_ids: torch.Tensor,
        context_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Encode the context with the SAME (shared) encoder weights."""
        return self.encoder(
            input_ids=context_input_ids, attention_mask=context_attention_mask
        )["cls"]

    def encode_pair(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        context_input_ids: Optional[torch.Tensor] = None,
        context_attention_mask: Optional[torch.Tensor] = None,
        context_present: Optional[torch.Tensor] = None,
        current_inputs_embeds: Optional[torch.Tensor] = None,
    ):
        """Return ``(E_c, E_p, token_states)``; ``E_p`` is ``None`` without context."""
        e_c, token_states = self._encode_current(
            input_ids=input_ids,
            attention_mask=attention_mask,
            current_inputs_embeds=current_inputs_embeds,
        )
        e_p = None
        if self.use_context and context_input_ids is not None:
            e_p = self._encode_context(context_input_ids, context_attention_mask)
            if context_present is not None:
                # Rows without a real context fall back to a zero vector.
                e_p = e_p * context_present.to(e_p.dtype).unsqueeze(-1)
        return e_c, e_p, token_states

    # --- forward ---------------------------------------------------------------------
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        context_input_ids: Optional[torch.Tensor] = None,
        context_attention_mask: Optional[torch.Tensor] = None,
        context_present: Optional[torch.Tensor] = None,
        current_inputs_embeds: Optional[torch.Tensor] = None,
        return_representations: bool = False,
    ) -> Dict[str, torch.Tensor]:
        e_c, e_p, token_states = self.encode_pair(
            input_ids=input_ids,
            attention_mask=attention_mask,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            context_present=context_present,
            current_inputs_embeds=current_inputs_embeds,
        )

        if self.interaction is not None:
            final_representation = self.interaction(e_c, e_p)
        else:
            final_representation = e_c

        outputs: Dict[str, torch.Tensor] = {
            "hate_logits": self.hate_head(final_representation)
        }
        if self.target_head is not None:
            outputs["target_logits"] = self.target_head(final_representation)
        if self.reason_head is not None:
            outputs["reason_logits"] = self.reason_head(final_representation)
        if self.evidence_head is not None:
            outputs["rationale_logits"] = self.evidence_head(token_states)
        if self.contrastive_projection is not None:
            outputs["projection"] = self.contrastive_projection(final_representation)

        if return_representations:
            outputs.update(
                {
                    "e_c": e_c,
                    "e_p": e_p,
                    "final_representation": final_representation,
                    "token_states": token_states,
                }
            )
        return outputs

    # --- contrastive training helper ----------------------------------------------
    def contrastive_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        positive_context_input_ids: torch.Tensor,
        positive_context_attention_mask: torch.Tensor,
        negative_context_input_ids: torch.Tensor,
        negative_context_attention_mask: torch.Tensor,
        negatives_per_anchor: int = 1,
        context_present: Optional[torch.Tensor] = None,
    ):
        """Return ``(z_anchor, z_positive, z_negative)`` for the InfoNCE loss.

        * ``z_anchor``   – comment-only representation (no context)
        * ``z_positive`` – representation with the correct preceding context
        * ``z_negative`` – representations with contrasting contexts, ``[B, N, D]``
        """
        if self.contrastive_projection is None or self.interaction is None:
            raise RuntimeError(
                "contrastive_embeddings requires use_contrastive and use_context"
            )
        e_c, _ = self._encode_current(input_ids=input_ids, attention_mask=attention_mask)
        z_anchor = self.contrastive_projection(self.interaction(e_c))

        e_p_positive = self._encode_context(
            positive_context_input_ids, positive_context_attention_mask
        )
        if context_present is not None:
            e_p_positive = e_p_positive * context_present.to(e_p_positive.dtype).unsqueeze(-1)
        z_positive = self.contrastive_projection(self.interaction(e_c, e_p_positive))

        e_p_negative = self._encode_context(
            negative_context_input_ids, negative_context_attention_mask
        )
        batch_size = e_c.size(0)
        n = max(1, int(negatives_per_anchor))
        e_c_repeated = e_c.repeat_interleave(n, dim=0) if n > 1 else e_c
        z_negative_flat = self.contrastive_projection(
            self.interaction(e_c_repeated, e_p_negative)
        )
        z_negative = z_negative_flat.view(batch_size, n, -1)
        return z_anchor, z_positive, z_negative
