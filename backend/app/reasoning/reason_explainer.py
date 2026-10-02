"""Deterministic reason explanation layer.

After the model predicts a reason category and the attribution stage extracts
evidence spans, this module connects the two with fixed, auditable templates:

    reason + target + evidence  →  {summary, details, grounded_in}

Design rules (see README "Reason explanation"):

* **Deterministic** — no generative model, no free-form text generation.
* **Grounded** — an explanation only uses the predicted reason, the predicted
  target, and the evidence spans that were actually extracted. It never adds
  information that is not present in those fields.
* **Honest** — it describes the classification and the attribution result; it
  does NOT claim to know the author's intent and does NOT claim that the
  context *caused* the prediction.

Templates are configurable: edit :data:`REASON_TEMPLATES` below, or drop a
``configs/reason_templates.yaml`` file into the backend with the same keys
(its entries are merged over the defaults).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from app.core.config import BACKEND_DIR

# --- default templates (deterministic, per reason category) -------------------

REASON_TEMPLATES: Dict[str, Dict[str, str]] = {
    "insult": {
        "summary": (
            "The highlighted language uses derogatory or abusive wording "
            "toward the targeted group."
        ),
        "details": (
            "The phrase {evidence} contains degrading or abusive wording directed "
            "at the targeted group{target_clause}, which corresponds to the "
            "predicted insult category."
        ),
    },
    "dehumanization": {
        "summary": (
            "The highlighted language dehumanizes the targeted group by portraying "
            "its members in a degrading non-human way."
        ),
        "details": (
            "The phrase {evidence} portrays members of the targeted group"
            "{target_clause} as less than human or as something non-human in a "
            "degrading way, which corresponds to the predicted dehumanization "
            "category."
        ),
    },
    "negative_stereotyping": {
        "summary": (
            "The highlighted language assigns a negative generalized "
            "characteristic to the targeted group."
        ),
        "details": (
            "The phrase {evidence} attributes a negative generalized characteristic "
            "to the targeted group{target_clause}, which corresponds to the "
            "predicted negative stereotyping category."
        ),
    },
    "threat": {
        "summary": (
            "The highlighted language communicates a threat of harm toward the "
            "targeted group."
        ),
        "details": (
            "The phrase {evidence} expresses or communicates a threat of harm "
            "toward the targeted group{target_clause}, which corresponds to the "
            "predicted threat category."
        ),
    },
    "exclusion": {
        "summary": (
            "The highlighted language expresses exclusion by advocating that "
            "members of the targeted group be removed or kept out."
        ),
        "details": (
            "The phrase {evidence} advocates removing or keeping out members of "
            "the targeted group{target_clause}, which corresponds to the predicted "
            "exclusion category."
        ),
    },
    "discrimination": {
        "summary": (
            "The highlighted language expresses discriminatory treatment toward "
            "the targeted group."
        ),
        "details": (
            "The phrase {evidence} advocates or expresses unequal treatment of the "
            "targeted group{target_clause}, which corresponds to the predicted "
            "discrimination category."
        ),
    },
    "incitement_to_violence": {
        "summary": (
            "The highlighted language encourages or promotes violence against the "
            "targeted group."
        ),
        "details": (
            "The phrase {evidence} encourages, calls for, or promotes violent "
            "action against the targeted group{target_clause}, which corresponds to "
            "the predicted incitement-to-violence category."
        ),
    },
    "other": {
        "summary": (
            "The highlighted evidence supports the model's classification as "
            "harmful content, but it does not fit the predefined reason categories "
            "more specifically."
        ),
        "details": (
            "The phrase {evidence} contributed to the model's classification"
            "{target_clause}; the predicted reason category is 'other'."
        ),
    },
}

_NO_EVIDENCE_DETAILS = (
    "No specific supporting phrases were extracted for this analysis, so the "
    "explanation reflects only the predicted reason category."
)
_CONTEXT_NOTE = "The analysis included the provided conversational context."


def _normalize_reason(reason: str) -> str:
    return str(reason).strip().lower().replace(" ", "_").replace("-", "_")


def _load_template_overrides() -> Dict[str, Dict[str, str]]:
    """Merge ``configs/reason_templates.yaml`` (if present) over the defaults."""
    path = BACKEND_DIR / "configs" / "reason_templates.yaml"
    if not path.exists():
        return {}
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(data, dict):
            return {
                str(key): {str(k): str(v) for k, v in value.items()}
                for key, value in data.items()
                if isinstance(value, dict)
            }
    except Exception:  # noqa: BLE001 - overrides are best-effort
        return {}
    return {}


def _active_templates(
    templates: Optional[Mapping[str, Mapping[str, str]]] = None
) -> Dict[str, Dict[str, str]]:
    merged: Dict[str, Dict[str, str]] = {
        key: dict(value) for key, value in REASON_TEMPLATES.items()
    }
    for source in (_load_template_overrides(), templates or {}):
        for key, value in source.items():
            merged.setdefault(str(key), {})
            merged[str(key)].update({str(k): str(v) for k, v in value.items()})
    return merged


def format_evidence_quotes(evidence: List[str], max_quotes: int = 2) -> str:
    """Format evidence texts as readable quotes: ``'a' and 'b'``.

    Returns an empty string when there is nothing to quote.
    """
    quotes = [f"'{text}'" for text in evidence[:max_quotes]]
    if not quotes:
        return ""
    if len(quotes) == 1:
        return quotes[0]
    return " and ".join(quotes)


def explain_reason(
    reason: Optional[str],
    target: Optional[str] = None,
    evidence: Optional[List[str]] = None,
    *,
    context_used: bool = False,
    templates: Optional[Mapping[str, Mapping[str, str]]] = None,
) -> Optional[Dict[str, Any]]:
    """Build a deterministic, evidence-grounded reason explanation.

    Returns ``None`` when no reason is available — nothing is fabricated
    (see README "Reason explanation" and the availability rules).
    """
    if reason is None or not str(reason).strip():
        return None

    reason_label = str(reason).strip()
    evidence_texts = [
        str(item).strip() for item in (evidence or []) if str(item).strip()
    ]
    active = _active_templates(templates)
    template = active.get(_normalize_reason(reason_label)) or active.get("other") or {}

    clean_target = str(target).strip() if target is not None and str(target).strip() else None
    target_clause = f" (predicted target: {clean_target})" if clean_target else ""

    if evidence_texts:
        quotes = format_evidence_quotes(evidence_texts)
        summary = template.get("summary") or (
            f"The highlighted evidence supports the predicted reason category "
            f"({reason_label})."
        )
        details = template.get("details") or (
            "The phrase {evidence} is associated with the model's predicted "
            "{reason} category."
        )
        summary = summary.format(evidence=quotes, target_clause=target_clause)
        details = details.format(
            evidence=quotes, target_clause=target_clause, reason=reason_label
        )
    else:
        summary = template.get("summary_no_evidence") or (
            f"The model's predicted reason category is {reason_label}."
        )
        details = template.get("details_no_evidence") or _NO_EVIDENCE_DETAILS
        if clean_target:
            details = f"{details.rstrip('.')}{target_clause}."

    if context_used:
        details = f"{details.rstrip()} {_CONTEXT_NOTE}"

    return {
        "summary": summary,
        "details": details,
        "grounded_in": {
            "reason": reason_label,
            "target": clean_target,
            "evidence": evidence_texts,
        },
    }
