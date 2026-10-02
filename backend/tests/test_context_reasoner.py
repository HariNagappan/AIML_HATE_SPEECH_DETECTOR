"""Tests for the context-relationship reasoning layer (app/reasoning/context_reasoner)."""

from __future__ import annotations

from app.reasoning.context_reasoner import (
    METHOD,
    build_context_reasoning,
    fallback_reasoning,
    find_reference_links,
)
from app.reasoning.evidence_extractor import EvidenceSpan


def _span(text: str, score: float = 0.8, start: int | None = None, end: int | None = None) -> EvidenceSpan:
    return EvidenceSpan(
        text=text,
        score=score,
        raw_score=score,
        token_indices=[],
        start=start,
        end=end,
    )


# --- reference analysis -----------------------------------------------------


class TestReferenceLinks:
    def test_spec_pronoun_example(self):
        previous = "Those immigrants were protesting."
        current = "They should leave."
        links = find_reference_links(previous, current)
        assert len(links) == 1
        link = links[0]
        assert link.from_text == "They"
        assert link.pronoun == "they"
        assert link.to_text == "immigrants"
        assert previous[link.to_start : link.to_end] == "immigrants"
        assert current[link.from_start : link.from_end] == "They"
        assert link.relation == "refers_to"

    def test_group_phrase_expansion(self):
        previous = "I saw a group of immigrants protesting downtown."
        links = find_reference_links(previous, "They should all be kicked out.")
        assert links
        assert links[0].to_text == "group of immigrants"
        assert previous[links[0].to_start : links[0].to_end] == "group of immigrants"

    def test_no_pronoun_means_no_links(self):
        links = find_reference_links(
            "The weather was nice today.", "I love this weather."
        )
        assert links == []

    def test_missing_previous_comment(self):
        assert find_reference_links("", "They should leave.") == []
        assert find_reference_links(None, "They should leave.") == []

    def test_multiple_pronouns_linked_once_each(self):
        links = find_reference_links(
            "Those immigrants were protesting.",
            "They should leave. Their presence is unwanted.",
        )
        assert {link.from_text for link in links} == {"They", "Their"}
        assert all(link.to_text == "immigrants" for link in links)

    def test_ambiguous_context_yields_no_link(self):
        # §14 example: "They were talking outside." has no clear antecedent noun
        links = find_reference_links("They were talking outside.", "They are disgusting.")
        assert links == []

    def test_singular_pronoun_produces_link(self):
        links = find_reference_links(
            "The mayor visited the school.", "He promised new funding."
        )
        assert len(links) == 1
        assert links[0].pronoun == "he"


# --- reasoning builder ------------------------------------------------------


class TestBuildContextReasoning:
    def test_context_hate_with_link(self):
        reasoning = build_context_reasoning(
            label="hate_speech",
            evidence_spans=[
                _span("should all be kicked out", 0.9, start=9, end=31)
            ],
            previous_comment="I saw a group of immigrants protesting downtown.",
            current_comment="They should all be kicked out.",
            context_used=True,
        )

        assert reasoning["method"] == METHOD
        assert reasoning["context_used"] is True
        assert reasoning["context_available"] is True

        summary = reasoning["summary"]
        assert '"They"' in summary
        assert 'group of immigrants' in summary

        assert len(reasoning["links"]) == 1
        link = reasoning["links"][0]
        assert link["from_text"] == "They"
        assert link["to_text"] == "group of immigrants"
        assert link["relation"] == "refers_to"

        sources = {item["source"] for item in reasoning["evidence"]}
        assert sources == {"current_comment", "previous_comment"}
        current_item = next(
            item for item in reasoning["evidence"] if item["source"] == "current_comment"
        )
        assert current_item["type"] == "current_span"
        assert current_item["reason"]
        assert current_item["start"] == 9
        context_item = next(
            item for item in reasoning["evidence"] if item["source"] == "previous_comment"
        )
        assert context_item["type"] == "context_target"
        assert context_item["start"] == link["to_start"]

    def test_no_context_states_it_explicitly(self):
        reasoning = build_context_reasoning(
            label="hate_speech",
            previous_comment="",
            current_comment="You are disgusting.",
            context_used=False,
        )
        assert reasoning["context_used"] is False
        assert reasoning["context_available"] is False
        assert "no previous comment was provided" in reasoning["summary"].lower()
        assert reasoning["links"] == []
        assert reasoning["evidence"] == []

    def test_non_hateful_does_not_invent_hateful_evidence(self):
        reasoning = build_context_reasoning(
            label="normal",
            previous_comment="He moved here from another country.",
            current_comment="Welcome! Hope you enjoy your new home.",
            context_used=True,
        )
        summary = reasoning["summary"].lower()
        assert "welcoming" in summary
        assert "no hateful" in summary
        # no previous-comment "evidence" was invented for a neutral prediction
        assert all(
            item["source"] != "previous_comment" for item in reasoning["evidence"]
        )
        assert reasoning["context_used"] is True

    def test_offensive_distinguishes_insult_from_group_hate(self):
        reasoning = build_context_reasoning(
            label="offensive",
            previous_comment="The player missed the shot.",
            current_comment="What an idiot.",
            context_used=True,
        )
        summary = reasoning["summary"].lower()
        assert "offensive" in summary or "insult" in summary
        assert "protected group" in summary

    def test_ambiguous_negative_without_target_reference(self):
        reasoning = build_context_reasoning(
            label="hate_speech",
            previous_comment="They were talking outside.",
            current_comment="They are disgusting.",
            context_used=True,
        )
        summary = reasoning["summary"].lower()
        assert reasoning["links"] == []
        assert "does not provide an explicit reference" in summary

    def test_counter_speech_tone(self):
        reasoning = build_context_reasoning(
            label="counter_speech",
            previous_comment="They should be kicked out.",
            current_comment="That is a horrible thing to say.",
            context_used=True,
        )
        assert "counter" in reasoning["summary"].lower()

    def test_unavailable_prediction(self):
        reasoning = build_context_reasoning(
            label=None,
            previous_comment="Hello there.",
            current_comment="Hi!",
            context_used=True,
        )
        assert "no reasoning could be derived" in reasoning["summary"].lower()
        assert reasoning["evidence"] == []
        assert reasoning["context_available"] is True

    def test_fallback_shape(self):
        reasoning = fallback_reasoning(context_used=True, context_available=True)
        assert reasoning["summary"] == "Reasoning unavailable."
        assert reasoning["evidence"] == []
        assert reasoning["links"] == []
        assert reasoning["context_used"] is True
        assert reasoning["context_available"] is True
