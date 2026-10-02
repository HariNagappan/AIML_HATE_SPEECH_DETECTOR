"""Attribution-based explanation: Integrated Gradients and gradient × input.

We explain a chosen prediction (hate / target / reason) by attributing it to
the **input tokens of the current comment**, computed through the model's
input embeddings:

* **Integrated Gradients** (Sundararajan et al., 2017) — average gradients
  along the straight path between a baseline embedding (PAD) and the actual
  embedding, multiplied by ``(embedding − baseline)``.
* **gradient × input** — single-point approximation (cheaper, noisier).

This is a *post-hoc attribution* method. It is NOT attention weights, and it
is NOT the supervised rationale head (see README "Evidence extraction").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from app.datasets.tokenizer import TokenizerWrapper, is_special_token

SUPPORTED_METHODS = ("integrated_gradients", "gradient_x_input")


@dataclass
class AttributionResult:
    """Per-token attributions for one specific prediction."""

    method: str
    target: str                      # "hate" | "target" | "reason"
    target_index: int                # class index that was explained
    target_label: Optional[str]      # readable label of the explained class
    predicted_label: Optional[str]
    predicted_confidence: Optional[float]
    tokens: List[str]                # tokenizer tokens (incl. specials)
    token_ids: List[int]
    scores: List[float]              # signed attribution per token (specials zeroed)
    steps: int
    context_used: bool


# --- small helpers --------------------------------------------------------------


def _head_logits(outputs: Dict[str, torch.Tensor], head: str) -> torch.Tensor:
    key = {"hate": "hate_logits", "target": "target_logits", "reason": "reason_logits"}.get(head)
    if key is None:
        raise ValueError(f"Unknown attribution target: {head!r}")
    if key not in outputs:
        raise ValueError(f"The model has no {head} head (cannot compute attribution).")
    return outputs[key]


def _head_module(model, head: str):
    return getattr(model, f"{head}_head", None)


def _predicted_index(model, outputs: Dict[str, torch.Tensor], head: str) -> int:
    logits = _head_logits(outputs, head)
    if logits.shape[-1] == 1:  # binary hate head
        threshold = float(getattr(_head_module(model, head), "threshold", 0.5))
        probability = float(torch.sigmoid(logits[0, 0]))
        return 1 if probability >= threshold else 0
    return int(torch.argmax(logits[0]).item())


def _decode_prediction(model, outputs: Dict[str, torch.Tensor], head: str):
    """Return ``(label, confidence)`` for the predicted class of a head."""
    module = _head_module(model, head)
    labels = getattr(module, "labels", None)
    if labels is None:
        return None, None
    logits = _head_logits(outputs, head)
    if logits.shape[-1] == 1:  # binary
        threshold = float(getattr(module, "threshold", 0.5))
        probability = float(torch.sigmoid(logits[0, 0]))
        positive = probability >= threshold
        return (
            labels[1] if positive else labels[0],
            probability if positive else 1.0 - probability,
        )
    probabilities = torch.softmax(logits[0], dim=-1)
    best = int(torch.argmax(probabilities).item())
    return labels[best], float(probabilities[best].item())


def _forward_with_current_embeds(
    model,
    embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    context_kwargs: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """Forward pass where the current comment uses the given embeddings.

    ``input_ids`` is intentionally not forwarded: transformers (v5+)
    requires exactly one of ``input_ids`` / ``inputs_embeds``.
    """
    if getattr(model, "architecture", "") == "baseline":
        return model(attention_mask=attention_mask, current_inputs_embeds=embeds)
    return model(
        attention_mask=attention_mask,
        current_inputs_embeds=embeds,
        **context_kwargs,
    )


# --- main entry point --------------------------------------------------------------


def compute_attributions(
    model,
    tokenizer: TokenizerWrapper,
    current_text: str,
    context_text: Optional[str] = None,
    *,
    target: str = "hate",
    target_index: Optional[int] = None,
    method: str = "integrated_gradients",
    steps: int = 32,
    max_length: int = 128,
    device: Optional[torch.device] = None,
) -> AttributionResult:
    """Compute per-token attributions of ``current_text`` for one head.

    ``target_index=None`` explains the currently predicted class; pass an
    index to explain a specific class instead (e.g. the ``hate`` class even
    when the model predicted ``normal``).
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown attribution method: {method!r}")
    model.eval()
    device = device or next(model.parameters()).device

    current_encoded = tokenizer.encode(
        current_text, max_length=max_length, padding=False, truncation=True, return_tensors="pt"
    )
    input_ids = current_encoded["input_ids"].to(device)
    attention_mask = current_encoded["attention_mask"].to(device)
    context_used = bool(context_text and context_text.strip())

    context_kwargs: Dict[str, Any] = {}
    if getattr(model, "supports_context", False) and context_used:
        context_encoded = tokenizer.encode(
            context_text, max_length=max_length, padding=False, truncation=True, return_tensors="pt"
        )
        context_kwargs = {
            "context_input_ids": context_encoded["input_ids"].to(device),
            "context_attention_mask": context_encoded["attention_mask"].to(device),
            "context_present": torch.ones(1, device=device),
        }

    # One no-grad pass to decode the current prediction and pick a class.
    with torch.no_grad():
        base_outputs = model(input_ids=input_ids, attention_mask=attention_mask, **context_kwargs)
    if target_index is None:
        target_index = _predicted_index(model, base_outputs, target)
    target_index = int(target_index)
    predicted_label, predicted_confidence = _decode_prediction(model, base_outputs, target)
    labels = getattr(_head_module(model, target), "labels", None)
    target_label = labels[target_index] if labels and target_index < len(labels) else None

    def select_scalar(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Scalar (summed over batch) logit of the explained class."""
        logits = _head_logits(outputs, target)
        if logits.shape[-1] == 1:  # binary head: explain positive or negative side
            value = logits[:, 0]
            return value.sum() if target_index == 1 else -value.sum()
        return logits[:, target_index].sum()

    embedding_layer = model.encoder.get_input_embeddings()
    actual_embeds = embedding_layer(input_ids).detach()
    pad_id = tokenizer.pad_token_id
    baseline_embeds = embedding_layer(torch.full_like(input_ids, pad_id)).detach()

    if method == "integrated_gradients":
        steps = max(1, int(steps))
        average_grads = torch.zeros_like(actual_embeds)
        alphas = torch.linspace(0.0, 1.0, steps + 1, device=device)[1:]
        for alpha in alphas:
            interpolated = (
                baseline_embeds + float(alpha) * (actual_embeds - baseline_embeds)
            ).detach().clone().requires_grad_(True)
            outputs = _forward_with_current_embeds(
                model, interpolated, attention_mask, context_kwargs
            )
            grads = torch.autograd.grad(select_scalar(outputs), interpolated)[0]
            average_grads += grads
        average_grads /= steps
        attributions = ((actual_embeds - baseline_embeds) * average_grads).sum(-1).squeeze(0)
        used_steps = steps
    else:  # gradient_x_input
        embeds = actual_embeds.detach().clone().requires_grad_(True)
        outputs = _forward_with_current_embeds(
            model, embeds, attention_mask, context_kwargs
        )
        grads = torch.autograd.grad(select_scalar(outputs), embeds)[0]
        attributions = (embeds.detach() * grads).sum(-1).squeeze(0)
        used_steps = 1

    token_ids = [int(i) for i in input_ids[0].tolist()]
    tokens = tokenizer.token_strings(token_ids)
    mask = attention_mask[0].tolist()
    scores: List[float] = []
    for index, token in enumerate(tokens):
        if mask[index] == 0 or is_special_token(token):
            scores.append(0.0)
        else:
            scores.append(float(attributions[index].item()))

    return AttributionResult(
        method=method,
        target=target,
        target_index=target_index,
        target_label=target_label,
        predicted_label=predicted_label,
        predicted_confidence=predicted_confidence,
        tokens=tokens,
        token_ids=token_ids,
        scores=scores,
        steps=used_steps,
        context_used=context_used,
    )
