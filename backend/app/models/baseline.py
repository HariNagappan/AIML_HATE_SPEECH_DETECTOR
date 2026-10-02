"""BERT baseline (Experiment 1): ``[CLS]`` → hate classifier. No context.

Purpose: compare "BERT only" against the proposed context-aware model
(README "Baseline vs proposed system").
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from app.models.bert_encoder import BertEncoder
from app.models.hate_head import HateHead


class BaselineModel(nn.Module):
    """Current comment → shared BERT → [CLS] → hate head."""

    architecture = "baseline"
    supports_context = False

    def __init__(
        self,
        encoder: BertEncoder,
        hate_labels,
        mode: str = "multiclass",
        threshold: float = 0.5,
    ):
        super().__init__()
        self.encoder = encoder
        self.hate_head = HateHead(
            encoder.hidden_size, hate_labels, mode=mode, threshold=threshold
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        current_inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # ``current_inputs_embeds`` is the unified parameter name used by the
        # attribution code for both model architectures.
        embeddings = current_inputs_embeds if current_inputs_embeds is not None else inputs_embeds
        encoded = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=embeddings
        )
        cls = encoded["cls"]
        return {"hate_logits": self.hate_head(cls), "cls": cls}

    @staticmethod
    def build(
        encoder: BertEncoder,
        hate_labels,
        mode: str = "multiclass",
        threshold: float = 0.5,
    ) -> "BaselineModel":
        return BaselineModel(encoder, hate_labels, mode=mode, threshold=threshold)
