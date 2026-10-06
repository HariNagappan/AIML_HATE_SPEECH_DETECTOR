"""Model lifecycle + inference service.

The FastAPI app uses one shared :class:`InferenceService` instance (loaded at
startup): the tokenizer and model are loaded **once** and reused for every
request — BERT is never loaded per request.

Also the home of checkpoint loading: :func:`build_model_from_checkpoint`
rebuilds the exact architecture from the metadata stored in a checkpoint, so a
trained model can be served without retraining.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from app import __version__
from app.core.config import Settings, get_settings, resolve_device
from app.core.logging import get_logger
from app.datasets.tokenizer import TokenizerWrapper, token_character_offsets
from app.reasoning.attribution import compute_attributions
from app.reasoning.evidence_extractor import evidence_to_dict, extract_evidence
from app.reasoning.context_reasoner import build_context_reasoning, fallback_reasoning
from app.reasoning.reason_explainer import explain_reason
from app.reasoning.structured_explanation import build_explanation

logger = get_logger("services.inference")


@dataclass
class ModelBundle:
    """Everything needed to serve a model: model, tokenizer, device, metadata."""

    model: Any
    tokenizer: TokenizerWrapper
    device: torch.device
    model_kwargs: Dict[str, Any]
    trained_heads: List[str]
    checkpoint_path: Optional[str] = None
    version: str = __version__


# --- model builders ------------------------------------------------------------------


def _build_from_kwargs(
    model_kwargs: Dict[str, Any], settings: Settings
) -> Tuple[Any, TokenizerWrapper]:
    """Rebuild the model + tokenizer given (checkpoint) metadata."""
    from app.models.baseline import BaselineModel
    from app.models.bert_encoder import BertEncoder
    from app.models.full_model import FullModel

    encoder_name = str(model_kwargs.get("encoder_name", settings.encoder_name))
    freeze_mode = str(model_kwargs.get("freeze_mode", "full"))
    frozen_layers = int(model_kwargs.get("frozen_layers", 0))

    if bool(model_kwargs.get("random_init")):
        encoder = BertEncoder.random_init(
            hidden_size=int(model_kwargs.get("hidden_size", 768)),
            num_hidden_layers=int(model_kwargs.get("random_num_layers", 2)),
            num_attention_heads=int(model_kwargs.get("random_num_heads", 12)),
            freeze_mode=freeze_mode,
            frozen_layers=frozen_layers,
        )
    else:
        encoder = BertEncoder.from_pretrained(
            encoder_name, freeze_mode=freeze_mode, frozen_layers=frozen_layers
        )

    hate_mode = str(model_kwargs.get("hate_mode", "multiclass"))
    hate_threshold = float(model_kwargs.get("hate_threshold", settings.hate_threshold))
    architecture = str(model_kwargs.get("architecture", "full"))

    if architecture == "baseline":
        model = BaselineModel(
            encoder,
            model_kwargs["hate_labels"],
            mode=hate_mode,
            threshold=hate_threshold,
        )
    else:
        model = FullModel(
            encoder,
            model_kwargs["hate_labels"],
            model_kwargs.get("target_labels") or [],
            model_kwargs.get("reason_labels") or [],
            interaction_dim=int(
                model_kwargs.get("interaction_dim", settings.interaction_hidden_dim)
            ),
            interaction_dropout=float(
                model_kwargs.get("interaction_dropout", settings.interaction_dropout)
            ),
            interaction_mode=str(model_kwargs.get("interaction_mode", "mlp")),
            use_context=bool(model_kwargs.get("use_context", True)),
            use_contrastive=bool(model_kwargs.get("use_contrastive", True)),
            use_target=bool(model_kwargs.get("use_target", True)),
            use_reason=bool(model_kwargs.get("use_reason", False)),
            use_evidence=bool(model_kwargs.get("use_evidence", True)),
            contrastive_dim=int(model_kwargs.get("contrastive_dim", 256)),
            hate_mode=hate_mode,
            hate_threshold=hate_threshold,
        )

    tokenizer = TokenizerWrapper.from_pretrained(encoder_name)
    return model, tokenizer


def build_model_from_checkpoint(
    checkpoint_path: str, settings: Optional[Settings] = None
) -> ModelBundle:
    """Load a trained checkpoint into a ready-to-serve :class:`ModelBundle`."""
    settings = settings or get_settings()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_kwargs = dict(payload.get("model_kwargs") or {})
    model, tokenizer = _build_from_kwargs(model_kwargs, settings)
    state = payload.get("model_state")
    if state is None:
        raise ValueError(f"Checkpoint {checkpoint_path!r} does not contain model_state")
    model.load_state_dict(state)
    device = resolve_device(settings.device)
    model.to(device)
    model.eval()
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_kwargs=model_kwargs,
        trained_heads=list(payload.get("trained_heads") or []),
        checkpoint_path=str(checkpoint_path),
    )


def build_untrained_bundle(settings: Optional[Settings] = None) -> ModelBundle:
    """Build a model WITHOUT trained weights (dev/tests).

    The API reports ``trained=false`` and per-head availability so untrained
    heads never fabricate predictions.
    """
    settings = settings or get_settings()
    random_init = str(settings.init_backbone).lower() == "random"
    model_kwargs: Dict[str, Any] = {
        "architecture": "full",
        "encoder_name": settings.encoder_name,
        "random_init": random_init,
        "hidden_size": 768,
        "random_num_layers": 2,
        "random_num_heads": 12,
        "hate_labels": settings.hate_label_list,
        "target_labels": settings.target_label_list,
        "reason_labels": settings.reason_label_list,
        "freeze_mode": "full",
        "frozen_layers": 0,
        "interaction_dim": settings.interaction_hidden_dim,
        "interaction_dropout": settings.interaction_dropout,
        "interaction_mode": "mlp",
        "use_context": True,
        "use_contrastive": True,
        "use_target": True,
        "use_reason": True,
        "use_evidence": True,
        "contrastive_dim": settings.contrastive_projection_dim,
        "hate_mode": "multiclass",
        "hate_threshold": settings.hate_threshold,
    }
    model, tokenizer = _build_from_kwargs(model_kwargs, settings)
    device = resolve_device(settings.device)
    model.to(device)
    model.eval()
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_kwargs=model_kwargs,
        trained_heads=[],
        checkpoint_path=None,
    )


def _resolve_default_checkpoint(settings: Settings) -> Optional[str]:
    """Find a previously trained checkpoint when MODEL_CHECKPOINT is unset.

    Priority:
    1. ``checkpoints/default.json`` pointer — ``{"checkpoint": "<path>"}``
    2. conventional paths: checkpoints/cc_context/best.pt,
       checkpoints/hx_full/best.pt, checkpoints/full/best.pt
    """
    pointer = settings.checkpoints_dir / "default.json"
    if pointer.exists():
        try:
            data = json.loads(pointer.read_text(encoding="utf-8"))
            raw = str(data.get("checkpoint") or "").strip()
            if raw:
                candidate = Path(raw)
                if not candidate.is_absolute():
                    candidate = settings.backend_dir / candidate
                if candidate.exists():
                    return str(candidate)
                logger.warning(
                    "default.json points to a missing checkpoint: %s", candidate
                )
        except Exception as exc:  # noqa: BLE001 - fall back to candidates below
            logger.warning("Could not read %s: %s", pointer, exc)

    for relative in (
        "checkpoints/cc_context/best.pt",
        "checkpoints/hx_full/best.pt",
        "checkpoints/full/best.pt",
    ):
        candidate = settings.backend_dir / relative
        if candidate.exists():
            return str(candidate)
    return None


def load_checkpoint_metrics(checkpoint_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load the evaluation metrics stored next to a checkpoint.

    ``scripts/evaluate.py`` writes ``<checkpoint-stem>.metrics.json`` beside
    the evaluated checkpoint; the API exposes its hate-head summary as the
    ``metrics`` field of ``/api/v1/model/info``. Returns ``None`` when the file
    is missing (or unreadable) so unavailability is reported explicitly
    instead of being guessed.
    """
    if not checkpoint_path:
        return None
    path = Path(checkpoint_path).with_name(
        Path(checkpoint_path).stem + ".metrics.json"
    )
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - a bad file must not break model info
        logger.warning("Could not read metrics file %s: %s", path, exc)
        return None

    hate = payload.get("hate") or {}

    def number(key: str) -> Optional[float]:
        value = hate.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    return {
        "split": payload.get("split"),
        "dataset": payload.get("dataset"),
        "num_examples": payload.get("num_examples"),
        "evaluated_at": payload.get("evaluated_at"),
        "accuracy": number("accuracy"),
        "macro_precision": number("macro_precision"),
        "macro_recall": number("macro_recall"),
        "macro_f1": number("macro_f1"),
        "weighted_precision": number("weighted_precision"),
        "weighted_recall": number("weighted_recall"),
        "weighted_f1": number("weighted_f1"),
    }


# --- prediction decoding ----------------------------------------------------------------


def decode_head_prediction(
    model, outputs: Dict[str, torch.Tensor], head: str
) -> Optional[Dict[str, Any]]:
    """Decode one head's logits into ``label`` / ``confidence`` / ``probabilities``."""
    module = getattr(model, f"{head}_head", None)
    key = f"{head}_logits"
    if module is None or key not in outputs:
        return None
    labels = [str(label) for label in getattr(module, "labels", [])]
    logits = outputs[key][0]

    if logits.shape[-1] == 1:  # explicit binary mode
        threshold = float(getattr(module, "threshold", 0.5))
        probability = float(torch.sigmoid(logits[0]).item())
        positive = probability >= threshold
        return {
            "label": labels[1] if positive else labels[0],
            "confidence": probability if positive else 1.0 - probability,
            "probabilities": {labels[0]: 1.0 - probability, labels[1]: probability},
        }

    probabilities = torch.softmax(logits, dim=-1)
    best = int(torch.argmax(probabilities).item())
    return {
        "label": labels[best],
        "confidence": float(probabilities[best].item()),
        "probabilities": {
            label: float(probability)
            for label, probability in zip(labels, probabilities.tolist())
        },
    }


# --- the service -------------------------------------------------------------------------


class InferenceService:
    """Loads the model once and answers prediction / explanation requests."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._bundle: Optional[ModelBundle] = None
        self.load_error: Optional[str] = None

    # --- lifecycle -------------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._bundle is not None

    def load(self) -> ModelBundle:
        """Load the configured checkpoint once (or an untrained model)."""
        if self._bundle is not None:
            return self._bundle
        try:
            checkpoint = self.settings.checkpoint_path
            if not checkpoint and self.settings.auto_load_checkpoint:
                checkpoint = _resolve_default_checkpoint(self.settings)
                if checkpoint:
                    logger.info(
                        "No MODEL_CHECKPOINT set — auto-detected trained checkpoint: %s",
                        checkpoint,
                    )
            if checkpoint:
                logger.info("Loading trained checkpoint: %s", checkpoint)
                self._bundle = build_model_from_checkpoint(checkpoint, self.settings)
                logger.info(
                    "Model loaded. Trained heads: %s", self._bundle.trained_heads
                )
            else:
                logger.info(
                    "No trained checkpoint found — building an untrained model "
                    "(heads will report available=false until trained)."
                )
                self._bundle = build_untrained_bundle(self.settings)
            self.load_error = None
        except Exception as exc:  # noqa: BLE001 - re-raised, message preserved
            self.load_error = str(exc)
            raise
        return self._bundle

    def _require_bundle(self) -> ModelBundle:
        if self._bundle is None:
            try:
                self.load()
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Model not available: {self.load_error or exc}"
                ) from exc
        assert self._bundle is not None
        return self._bundle

    # --- inference helpers --------------------------------------------------------
    def _forward_single(
        self, bundle: ModelBundle, text: str, context_text: Optional[str]
    ) -> Dict[str, torch.Tensor]:
        model = bundle.model
        encoded = bundle.tokenizer.encode(
            text,
            max_length=self.settings.max_seq_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        kwargs: Dict[str, Any] = {
            "input_ids": encoded["input_ids"].to(bundle.device),
            "attention_mask": encoded["attention_mask"].to(bundle.device),
        }
        if getattr(model, "supports_context", False) and context_text:
            context_encoded = bundle.tokenizer.encode(
                context_text,
                max_length=self.settings.max_seq_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
            )
            kwargs.update(
                {
                    "context_input_ids": context_encoded["input_ids"].to(bundle.device),
                    "context_attention_mask": context_encoded["attention_mask"].to(
                        bundle.device
                    ),
                    "context_present": torch.ones(1, device=bundle.device),
                }
            )
        return model(**kwargs)

    # --- public API ------------------------------------------------------------------
    def predict(self, text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Run the full pipeline; returns the structured explanation object."""
        bundle = self._require_bundle()
        model = bundle.model
        trained = set(bundle.trained_heads)
        context_text = context.strip() if context and context.strip() else None
        context_used = context_text is not None

        hate_prediction: Optional[Dict[str, Any]] = None
        target_prediction: Optional[Dict[str, Any]] = None
        reason_prediction: Optional[Dict[str, Any]] = None
        evidence_spans: List[Any] = []

        if trained & {"hate", "target", "reason"}:
            model.eval()
            with torch.no_grad():
                outputs = self._forward_single(bundle, text, context_text)
            if "hate" in trained:
                hate_prediction = decode_head_prediction(model, outputs, "hate")
            if "target" in trained:
                target_prediction = decode_head_prediction(model, outputs, "target")
            if "reason" in trained:
                reason_prediction = decode_head_prediction(model, outputs, "reason")

            if "hate" in trained:
                attribution = compute_attributions(
                    model,
                    bundle.tokenizer,
                    text,
                    context_text,
                    target="hate",
                    method=self.settings.evidence_method,
                    steps=self.settings.evidence_ig_steps,
                    max_length=self.settings.max_seq_length,
                    device=bundle.device,
                )
                offsets = token_character_offsets(
                    bundle.tokenizer, text, max_length=self.settings.max_seq_length
                )
                evidence_spans = extract_evidence(
                    attribution,
                    bundle.tokenizer,
                    top_k=self.settings.evidence_top_k,
                    min_score_frac=self.settings.evidence_min_score,
                    offsets=offsets or None,
                )

        reason_explanation = None
        if reason_prediction is not None:
            reason_explanation = explain_reason(
                reason=reason_prediction.get("label"),
                target=target_prediction.get("label") if target_prediction else None,
                evidence=[span.text for span in evidence_spans],
                context_used=context_used,
            )

        # Context-relationship reasoning: built deterministically from the
        # prediction, the attribution spans and a transparent reference
        # analysis of the two comments. A reasoning failure must never fail
        # the classification request (spec §16) - degrade explicitly instead.
        try:
            reasoning = build_context_reasoning(
                label=hate_prediction.get("label") if hate_prediction else None,
                evidence_spans=evidence_spans,
                previous_comment=context,
                current_comment=text,
                context_used=context_used,
                target_label=(
                    target_prediction.get("label") if target_prediction else None
                ),
            )
        except Exception as exc:  # noqa: BLE001 - never break classification
            logger.warning("Context reasoning failed: %s", exc)
            reasoning = fallback_reasoning(
                context_used=context_used,
                context_available=bool(context and context.strip()),
            )

        return build_explanation(
            hate=hate_prediction,
            target=target_prediction,
            reason=reason_prediction,
            evidence=evidence_spans,
            context_used=context_used,
            reason_explanation=reason_explanation,
            evidence_available="hate" in trained,
            reasoning=reasoning,
        )

    def explain(
        self,
        text: str,
        context: Optional[str] = None,
        target: str = "hate",
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Detailed attribution output for a chosen head (hate/target/reason)."""
        bundle = self._require_bundle()
        trained = set(bundle.trained_heads)
        context_used = bool(context and context.strip())

        if target not in trained:
            return {
                "available": False,
                "message": (
                    f"The '{target}' head is not trained/available — "
                    "attribution requires a trained head."
                ),
                "target": target,
                "method": self.settings.evidence_method,
                "steps": 0,
                "predicted_label": None,
                "predicted_confidence": None,
                "explained_label": None,
                "context_used": context_used,
                "tokens": [],
                "spans": [],
                "reason_explanation": None,
            }

        attribution = compute_attributions(
            bundle.model,
            bundle.tokenizer,
            text,
            context,
            target=target,
            method=self.settings.evidence_method,
            steps=self.settings.evidence_ig_steps,
            max_length=self.settings.max_seq_length,
            device=bundle.device,
        )
        offsets = token_character_offsets(
            bundle.tokenizer, text, max_length=self.settings.max_seq_length
        )
        spans = extract_evidence(
            attribution,
            bundle.tokenizer,
            top_k=top_k or self.settings.evidence_top_k,
            min_score_frac=self.settings.evidence_min_score,
            offsets=offsets or None,
        )

        reason_explanation = None
        if target == "hate" and "reason" in trained:
            with torch.no_grad():
                outputs = self._forward_single(bundle, text, context)
            target_prediction = (
                decode_head_prediction(bundle.model, outputs, "target")
                if "target" in trained
                else None
            )
            reason_prediction = decode_head_prediction(bundle.model, outputs, "reason")
            if reason_prediction is not None:
                reason_explanation = explain_reason(
                    reason=reason_prediction.get("label"),
                    target=(
                        target_prediction.get("label") if target_prediction else None
                    ),
                    evidence=[span.text for span in spans],
                    context_used=attribution.context_used,
                )

        return {
            "available": True,
            "message": None,
            "target": target,
            "method": attribution.method,
            "steps": attribution.steps,
            "predicted_label": attribution.predicted_label,
            "predicted_confidence": attribution.predicted_confidence,
            "explained_label": attribution.target_label,
            "context_used": attribution.context_used,
            "tokens": [
                {"index": index, "token": token, "score": float(score)}
                for index, (token, score) in enumerate(
                    zip(attribution.tokens, attribution.scores)
                )
            ],
            "spans": [evidence_to_dict(span) for span in spans],
            "reason_explanation": reason_explanation,
        }

    def model_info(self) -> Dict[str, Any]:
        """Status information for ``GET /api/v1/model/info``."""
        settings = self.settings
        bundle = self._bundle
        if bundle is None:
            return {
                "model_name": settings.encoder_name,
                "device": str(resolve_device(settings.device)),
                "hidden_size": None,
                "num_labels": {},
                "trained": False,
                "trained_heads": [],
                "version": __version__,
                "architecture": None,
                "checkpoint": settings.checkpoint_path,
                "loaded": False,
                "metrics": None,
                "error": self.load_error,
            }
        model = bundle.model
        num_labels: Dict[str, int] = {
            "hate": len(getattr(model.hate_head, "labels", []))
        }
        if getattr(model, "target_head", None) is not None:
            num_labels["target"] = len(model.target_head.labels)
        if getattr(model, "reason_head", None) is not None:
            num_labels["reason"] = len(model.reason_head.labels)
        return {
            "model_name": bundle.model_kwargs.get("encoder_name", settings.encoder_name),
            "device": str(bundle.device),
            "hidden_size": getattr(model.encoder, "hidden_size", None),
            "num_labels": num_labels,
            "trained": bool(bundle.trained_heads),
            "trained_heads": list(bundle.trained_heads),
            "version": bundle.version,
            "architecture": bundle.model_kwargs.get("architecture", "full"),
            "checkpoint": bundle.checkpoint_path,
            "loaded": True,
            "metrics": load_checkpoint_metrics(bundle.checkpoint_path),
            "error": self.load_error,
        }


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    """Process-wide singleton used by the API routes."""
    return InferenceService()


def reset_inference_service() -> None:
    """Testing helper: forget the cached service (e.g. after env changes)."""
    get_inference_service.cache_clear()


def warmup_inference_service() -> Optional[str]:
    """Best-effort model load at app startup; returns an error string on failure."""
    service = get_inference_service()
    try:
        service.load()
        return None
    except Exception as exc:  # noqa: BLE001 - reported, not fatal
        logger.exception("Model warmup failed: %s", exc)
        return str(exc)
