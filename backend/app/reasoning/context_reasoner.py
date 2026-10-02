"""Context-relationship reasoning layer (transparent, deterministic).

Answers *"why was this classified the way it was?"* by analysing the
relationship between the **previous comment** and the **current comment** —
without claiming that the classifier itself produced human-readable
explanations.

Scope and honesty rules
-----------------------
* **Not model attention.** No attention weights are read, fabricated, or
  presented as reasoning.
* **Not a second model.** The classification comes from the trained heads;
  this layer only *analyses* the input that produced it.
* **Not an LLM.** Every sentence is composed from fixed templates, grounded
  in three things: the predicted label, the extracted evidence spans
  (Integrated Gradients), and a small reference analysis of the two comments.

What it does
------------
1. **Reference analysis** — an auditable pronoun -> antecedent heuristic that
   links e.g. ``"They"`` (current comment) to ``"group of immigrants"``
   (previous comment), with exact character offsets on both sides. It is
   deliberately simple (no parsing model is used) and its output is presented
   *as* a heuristic link (``method`` field marks the analysis as
   deterministic).
2. **Evidence assembly** — current-comment spans come from the attribution
   pipeline; previous-comment spans are the link targets. Every item carries
   ``source`` (``current_comment`` / ``previous_comment``), offsets, and a
   one-line grounded rationale.
3. **Summary composition** — fixed templates keyed by the prediction's tone
   (hate-like / offensive / counter / neutral), presence of context links,
   and abstention rules: when the evidence cannot establish a relationship,
   the summary says so instead of inventing certainty.

The API response keeps classification and reasoning strictly separate; this
module only builds the ``reasoning`` block.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from app.reasoning.reason_explainer import format_evidence_quotes

#: Marker stored in the reasoning block so clients can see how it was produced.
METHOD = "deterministic evidence + context-reference analysis"

_TOKEN_RE = re.compile(r"\S+")
_TRIM_CHARS = string.punctuation + "“”‘’…'\""

# --- vocabulary used by the reference heuristic (all deliberately small) ----

LINKABLE_PRONOUNS = {
    "they",
    "them",
    "their",
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
}

DETERMINERS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "some",
    "any",
    "all",
    "both",
    "each",
    "every",
    "no",
}

PREPOSITIONS = {
    "of",
    "for",
    "from",
    "in",
    "on",
    "with",
    "to",
    "at",
    "by",
    "about",
    "against",
}

QUANTITY_NOUNS = {
    "group",
    "groups",
    "crowd",
    "crowds",
    "bunch",
    "lot",
    "number",
    "kind",
    "type",
    "class",
    "set",
    "mass",
    "wave",
    "stream",
    "gang",
    "band",
    "pack",
    "community",
    "communities",
}

_COMMON_VERBS = {
    "saw", "see", "seen", "said", "say", "says", "went", "go", "goes", "get",
    "got", "make", "made", "take", "took", "give", "gave", "think", "thought",
    "know", "knew", "want", "wanted", "like", "liked", "love", "loved",
    "tell", "told", "ask", "asked", "move", "moved", "come", "came", "look",
    "looked", "seem", "seemed", "keep", "kept", "let", "put", "run", "ran",
    "read", "find", "found", "call", "called", "show", "showed", "hear",
    "heard", "play", "played", "miss", "missed", "work", "worked", "live",
    "lived", "believe", "believed", "hold", "held", "bring", "brought",
    "happen", "happened", "write", "wrote", "provide", "sit", "sat", "stand",
    "stood", "lose", "lost", "pay", "paid", "meet", "met", "include",
    "included", "continue", "continued", "learn", "learned", "change",
    "changed", "lead", "led", "understand", "understood", "watch", "watched",
    "follow", "followed", "stop", "stopped", "create", "created", "speak",
    "spoke", "allow", "allowed", "add", "added", "spend", "spent", "grow",
    "grew", "open", "opened", "walk", "walked", "offer", "offered",
    "remember", "consider", "appear", "buy", "bought", "wait", "waited",
    "serve", "served", "send", "sent", "expect", "expected", "build", "built",
    "stay", "stayed", "fall", "fell", "cut", "reach", "reached", "kill",
    "killed", "remain", "remained", "were", "was", "are", "is", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "can", "could", "should", "may", "might", "must", "protesting", "protest",
    "protested",
}

_STOPWORDS = (
    DETERMINERS
    | PREPOSITIONS
    | LINKABLE_PRONOUNS
    | {
        "i", "you", "we", "us", "our", "ours", "it", "its", "me", "my",
        "mine", "your", "yours", "and", "or", "but", "not", "no", "nor",
        "very", "too", "so", "just", "only", "also", "there", "here", "when",
        "where", "why", "how", "what", "which", "who", "whom", "whose",
        "than", "then", "now", "up", "down", "off", "over", "under", "again",
        "more", "most", "less", "least", "such", "own", "same", "other",
        "another", "because", "while", "during", "after", "before", "between",
        "into", "onto", "out", "as", "if", "that", "this", "these", "those",
        "an", "the", "a", "at", "by", "for", "from", "in", "of", "on", "to",
        "with", "is", "are",
    }
)

#: Words that end in "s" but are not plural nouns (guards the heuristic).
_S_FALSE_POSITIVES = {
    "says", "goes", "does", "was", "has", "is", "yes", "news", "always",
    "perhaps", "sometimes", "less", "unless", "hers", "theirs", "his",
    "those", "these", "this", "bus", "status", "virus", "versus",
}

#: Adverbs / location words excluded from the antecedent fallback (too vague).
_ADVERBS = {
    "outside",
    "inside",
    "downtown",
    "upstairs",
    "downstairs",
    "tonight",
    "yesterday",
    "today",
    "tomorrow",
    "everywhere",
    "somewhere",
    "anywhere",
    "nowhere",
    "away",
    "ahead",
    "abroad",
    "overseas",
    "online",
    "home",
}

#: Small, documented marker set for welcoming/supportive language (§12).
_POSITIVE_MARKERS = {
    "welcome", "welcomes", "welcoming", "hope", "hopefully", "enjoy",
    "enjoys", "congratulations", "congrats", "thanks", "thank", "glad",
    "happy", "cheers", "celebrate", "supportive", "wonderful", "lovely",
    "appreciate", "proud", "kindness", "kind",
}


# --- tokenisation -----------------------------------------------------------


@dataclass
class _Token:
    raw: str
    clean: str
    start: int
    end: int


def _tokens(text: str) -> List[_Token]:
    tokens: List[_Token] = []
    for match in _TOKEN_RE.finditer(text or ""):
        raw = match.group(0)
        tokens.append(
            _Token(
                raw=raw,
                clean=raw.strip(_TRIM_CHARS).lower(),
                start=match.start(),
                end=match.end(),
            )
        )
    return tokens


def _is_plural_nounish(token: _Token) -> bool:
    clean = token.clean
    return (
        len(clean) >= 4
        and clean not in _STOPWORDS
        and clean not in _COMMON_VERBS
        and clean not in _S_FALSE_POSITIVES
        and clean.endswith("s")
        and not clean.endswith("ss")
    )


def _is_content_word(token: _Token) -> bool:
    clean = token.clean
    return (
        len(clean) >= 4
        and clean not in _STOPWORDS
        and clean not in _COMMON_VERBS
        and clean not in _ADVERBS
        and not clean.endswith(("ly", "ing", "ed"))
    )


def _is_expander(token: _Token) -> bool:
    clean = token.clean
    return clean in DETERMINERS or clean in PREPOSITIONS or clean in QUANTITY_NOUNS


# --- reference analysis -----------------------------------------------------


@dataclass
class ReferenceLink:
    """One heuristic pronoun -> antecedent link across the two comments."""

    from_text: str  # the pronoun, as written in the current comment
    from_start: int
    from_end: int
    to_text: str  # the antecedent phrase, as written in the previous comment
    to_start: int
    to_end: int
    pronoun: str
    relation: str = "refers_to"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_text": self.from_text,
            "from_start": self.from_start,
            "from_end": self.from_end,
            "to_text": self.to_text,
            "to_start": self.to_start,
            "to_end": self.to_end,
            "pronoun": self.pronoun,
            "relation": self.relation,
        }


def _find_antecedent(previous_tokens: Sequence[_Token]) -> Optional[Tuple[int, int]]:
    """Best-effort antecedent: right-most plural noun (else content word).

    The span is widened leftwards over determiners / prepositions / quantity
    nouns (e.g. ``"a group of immigrants"``), then stripped of leading
    determiners (``"group of immigrants"``).
    """
    if not previous_tokens:
        return None

    anchor: Optional[int] = None
    for index in range(len(previous_tokens) - 1, -1, -1):
        if _is_plural_nounish(previous_tokens[index]):
            anchor = index
            break
    if anchor is None:
        for index in range(len(previous_tokens) - 1, -1, -1):
            if _is_content_word(previous_tokens[index]):
                anchor = index
                break
    if anchor is None:
        return None

    start = anchor
    cursor = anchor - 1
    while cursor >= 0 and _is_expander(previous_tokens[cursor]):
        start = cursor
        cursor -= 1
    while start < anchor and previous_tokens[start].clean in DETERMINERS:
        start += 1
    return (start, anchor)


def find_reference_links(
    previous_comment: Optional[str],
    current_comment: str,
    max_links: int = 3,
) -> List[ReferenceLink]:
    """Find pronoun -> antecedent links from the current comment into the previous one.

    Deterministic and auditable: the antecedent is the right-most plural noun
    phrase (or, failing that, the right-most content word) of the previous
    comment; each linkable pronoun in the current comment is linked to it at
    most once.
    """
    previous = (previous_comment or "").strip()
    if not previous or not current_comment:
        return []

    previous_tokens = _tokens(previous)
    current_tokens = _tokens(current_comment)
    antecedent = _find_antecedent(previous_tokens)
    if antecedent is None:
        return []
    target_start_tok, target_end_tok = antecedent
    target_text = previous[
        previous_tokens[target_start_tok].start : previous_tokens[target_end_tok].end
    ]

    links: List[ReferenceLink] = []
    seen: set = set()
    for token in current_tokens:
        if token.clean in LINKABLE_PRONOUNS and token.clean not in seen:
            seen.add(token.clean)
            links.append(
                ReferenceLink(
                    from_text=current_comment[token.start : token.end],
                    from_start=token.start,
                    from_end=token.end,
                    to_text=target_text,
                    to_start=previous_tokens[target_start_tok].start,
                    to_end=previous_tokens[target_end_tok].end,
                    pronoun=token.clean,
                )
            )
            if len(links) >= max_links:
                break
    return links


# --- summary composition ----------------------------------------------------


def _pretty_label(label: str) -> str:
    return str(label).replace("_", " ").replace("-", " ").strip().capitalize()


def _label_tone(label: Optional[str]) -> str:
    key = str(label or "").strip().lower().replace("-", " ").replace("_", " ")
    compact = key.replace(" ", "")
    if compact in {"hate", "hateful", "hatespeech"}:
        return "hate"
    if key in {"offensive", "offence"}:
        return "offensive"
    if compact in {"counter", "counterspeech"}:
        return "counter"
    return "neutral"


def _has_positive_language(current_comment: str) -> bool:
    return any(token.clean in _POSITIVE_MARKERS for token in _tokens(current_comment))


def _compose_summary(
    *,
    label: Optional[str],
    spans: Sequence[Any],
    links: Sequence[ReferenceLink],
    context_available: bool,
    target_label: Optional[str],
    current_comment: str,
) -> str:
    if not label:
        return (
            "No classification was produced (model unavailable), so no reasoning "
            "could be derived. "
            + (
                "A previous comment was provided."
                if context_available
                else "No previous comment was provided."
            )
        )

    pretty = _pretty_label(label)
    tone = _label_tone(label)
    quotes = format_evidence_quotes([str(span.text) for span in spans], max_quotes=2)
    parts: List[str] = []

    if tone == "hate":
        if links:
            link = links[0]
            parts.append(
                f'The current comment uses "{link.from_text}" to refer to '
                f'"{link.to_text}" from the previous comment and expresses hostile '
                "or exclusionary language toward that target."
            )
            parts.append(
                f'The contextual reference matters: on its own, "{link.from_text}" '
                "does not identify who or what is being referred to."
            )
        elif context_available:
            parts.append(
                f"The model classified the comment as {pretty}. The previous comment "
                "was included in the analysis, but no explicit reference relationship "
                "between the two comments was identified."
            )
        else:
            parts.append(
                f"The model classified the comment as {pretty}, based on the current "
                "comment alone - no previous comment was provided, so no "
                "context relationship could be analysed."
            )
        if quotes:
            parts.append(f"Strongest evidence: the phrase {quotes}.")
        if not links:
            if target_label and str(target_label).strip().lower() not in {"none", ""}:
                parts.append(
                    f"The model's target head points to '{_pretty_label(str(target_label))}'."
                )
            elif context_available:
                parts.append(
                    "The available context does not provide an explicit reference to "
                    "which group (if any) the statement targets."
                )

    elif tone == "offensive":
        parts.append(
            f"The model classified the comment as {pretty} - abusive or insulting "
            "language."
        )
        if links:
            link = links[0]
            parts.append(
                f'It refers to "{link.to_text}" from the previous comment, but the '
                "context analysis does not indicate that the insult targets a "
                "protected group."
            )
        elif context_available:
            parts.append(
                "The available context does not indicate that the insult targets a "
                "protected group."
            )
        if quotes:
            parts.append(f"Strongest evidence: the phrase {quotes}.")

    elif tone == "counter":
        parts.append(
            f"The model classified the comment as {pretty} - language that opposes "
            "or rebuts hateful content."
        )
        if context_available:
            parts.append(
                "The previous comment was included in the analysis; no exclusionary "
                "relationship toward it was identified."
            )
        if quotes:
            parts.append(f"Strongest evidence: the phrase {quotes}.")

    else:  # neutral / non-hateful
        parts.append(f"The model classified the comment as {pretty}.")
        positive = _has_positive_language(current_comment) or any(
            _has_positive_language(str(getattr(span, "text", ""))) for span in spans
        )
        if positive:
            parts.append(
                "The comment contains welcoming or supportive language"
                + (f' toward the subject of "{links[0].to_text}"' if links else "")
                + "; no hateful or exclusionary language toward a protected group "
                "was identified."
            )
        else:
            parts.append(
                "No group-directed hostility was identified in the extracted evidence."
            )
        if context_available:
            parts.append("The previous comment was included in the analysis.")
        if quotes:
            parts.append(f"Strongest evidence: the phrase {quotes}.")

    return " ".join(parts)


# --- public builder ---------------------------------------------------------


def build_context_reasoning(
    *,
    label: Optional[str],
    evidence_spans: Iterable[Any] = (),
    previous_comment: Optional[str] = None,
    current_comment: str = "",
    context_used: bool = False,
    target_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the structured ``reasoning`` block for one prediction.

    ``evidence_spans`` are :class:`~app.reasoning.evidence_extractor.EvidenceSpan`
    objects (attribution output for the current comment). ``label`` is the
    predicted hate-style label (or ``None`` when the model is unavailable).
    """
    previous = (previous_comment or "").strip()
    context_available = bool(previous)

    spans = [span for span in evidence_spans if str(getattr(span, "text", "")).strip()][:3]
    links: List[ReferenceLink] = []
    if context_available and current_comment:
        try:
            links = find_reference_links(previous, current_comment)
        except Exception:  # noqa: BLE001 - the heuristic must never break inference
            links = []

    evidence_items: List[Dict[str, Any]] = []
    for span in spans:
        item: Dict[str, Any] = {
            "text": str(span.text),
            "source": "current_comment",
            "type": "current_span",
            "reason": (
                "Strongest attribution signal "
                f"(score {float(getattr(span, 'score', 0.0)):.2f}) for the predicted "
                "classification."
            ),
            "score": float(getattr(span, "score", 0.0)),
        }
        start = getattr(span, "start", None)
        end = getattr(span, "end", None)
        if start is not None and end is not None:
            item["start"] = int(start)
            item["end"] = int(end)
        evidence_items.append(item)

    for link in links:
        evidence_items.append(
            {
                "text": link.to_text,
                "source": "previous_comment",
                "type": "context_target",
                "reason": (
                    f'Identified by the reference analysis as the phrase '
                    f'"{link.from_text}" refers to.'
                ),
                "start": link.to_start,
                "end": link.to_end,
            }
        )

    summary = _compose_summary(
        label=label,
        spans=spans,
        links=links,
        context_available=context_available,
        target_label=target_label,
        current_comment=current_comment,
    )

    return {
        "summary": summary,
        "method": METHOD,
        "context_used": bool(context_used),
        "context_available": context_available,
        "links": [link.to_dict() for link in links],
        "evidence": evidence_items,
    }


def fallback_reasoning(
    *, context_used: bool = False, context_available: bool = False
) -> Dict[str, Any]:
    """Spec-compliant degradation when reasoning generation fails (§16)."""
    return {
        "summary": "Reasoning unavailable.",
        "method": METHOD,
        "context_used": bool(context_used),
        "context_available": bool(context_available),
        "links": [],
        "evidence": [],
    }
