"""Shared BERT encoder.

One set of BERT weights encodes **both** the current comment and the previous
context (weight sharing — see README, "Why is the encoder shared?").
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from app.datasets.tokenizer import TokenizerWrapper


class BertEncoder(nn.Module):
    """Wrapper around a Hugging Face BERT model.

    ``freeze_mode``:

    * ``full``    – all BERT weights fine-tuned
    * ``frozen``  – BERT weights frozen (only the behavioural heads train)
    * ``partial`` – embeddings + the bottom ``frozen_layers`` layers frozen
    """

    def __init__(
        self,
        bert: nn.Module,
        hidden_size: Optional[int] = None,
        freeze_mode: str = "full",
        frozen_layers: int = 0,
    ):
        super().__init__()
        if freeze_mode not in {"full", "frozen", "partial"}:
            raise ValueError(f"Unknown freeze_mode: {freeze_mode!r}")
        self.bert = bert
        self.hidden_size = int(hidden_size or bert.config.hidden_size)  # type: ignore[attr-defined]
        self.freeze_mode = freeze_mode
        self.frozen_layers = int(frozen_layers)
        self._apply_freeze()

    # --- constructors -----------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls, model_name: str, freeze_mode: str = "full", frozen_layers: int = 0
    ) -> "BertEncoder":
        from transformers import AutoModel

        bert = AutoModel.from_pretrained(model_name)
        return cls(bert, freeze_mode=freeze_mode, frozen_layers=frozen_layers)

    @classmethod
    def random_init(
        cls,
        hidden_size: int = 768,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 12,
        vocab_size: int = 30522,
        intermediate_size: Optional[int] = None,
        max_position_embeddings: int = 512,
        freeze_mode: str = "full",
        frozen_layers: int = 0,
    ) -> "BertEncoder":
        """Small randomly initialised BERT — for tests/dev without downloads."""
        from transformers import BertConfig, BertModel

        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size or hidden_size * 4,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=2,
            pad_token_id=0,
        )
        return cls(
            BertModel(config), freeze_mode=freeze_mode, frozen_layers=frozen_layers
        )

    def model_name(self) -> str:
        return str(getattr(self.bert.config, "_name_or_path", "") or "unknown")

    # --- freezing ------------------------------------------------------------------
    def _apply_freeze(self) -> None:
        if self.freeze_mode == "frozen":
            for parameter in self.bert.parameters():
                parameter.requires_grad = False
        elif self.freeze_mode == "partial":
            for parameter in self.bert.embeddings.parameters():
                parameter.requires_grad = False
            layers = list(self.bert.encoder.layer)  # type: ignore[attr-defined]
            for layer in layers[: max(0, self.frozen_layers)]:
                for parameter in layer.parameters():
                    parameter.requires_grad = False

    # --- forward ---------------------------------------------------------------------
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run BERT; returns contextual token states and the [CLS] embedding."""
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Provide either input_ids or inputs_embeds")
        if inputs_embeds is not None:
            # transformers requires exactly one of the two; embeddings take
            # precedence when both are supplied (attribution code path).
            input_ids = None
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
        )
        last_hidden_state = outputs.last_hidden_state
        return {
            "last_hidden_state": last_hidden_state,
            "cls": last_hidden_state[:, 0],
            "token_embeddings": last_hidden_state,
        }

    def get_input_embeddings(self) -> nn.Module:
        return self.bert.get_input_embeddings()  # type: ignore[attr-defined]

    # --- debugging helper ---------------------------------------------------------------
    @torch.no_grad()
    def encode_text(
        self,
        text: str,
        tokenizer: TokenizerWrapper,
        max_length: int = 128,
        device: Optional[torch.device] = None,
        keep_token_embeddings: bool = True,
    ) -> Dict[str, Any]:
        """Full pipeline for one raw text (debugging / inspection).

        Returns token ids, attention mask, token strings, contextual token
        representations (optional, not retained by default users of this
        helper for inference) and the [CLS] embedding — i.e. the actual
        intermediate representations of the encoder.
        """
        encoded = tokenizer.encode(
            text, max_length=max_length, padding=False, truncation=True, return_tensors="pt"
        )
        if device is not None:
            encoded = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
        was_training = self.training
        self.eval()
        outputs = self.forward(
            input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]
        )
        if was_training:
            self.train()

        result: Dict[str, Any] = {
            "input_ids": encoded["input_ids"][0].detach().cpu(),
            "attention_mask": encoded["attention_mask"][0].detach().cpu(),
            "tokens": tokenizer.token_strings(encoded["input_ids"][0].tolist()),
            "cls": outputs["cls"][0].detach().cpu(),
        }
        if keep_token_embeddings:
            result["token_embeddings"] = outputs["token_embeddings"][0].detach().cpu()
        return result
