"""Reason explanation layer tests.

The explanation must be deterministic, grounded only in the given fields,
and must not invent unsupported claims.
"""

from __future__ import annotations

from app.reasoning.reason_explainer import REASON_TEMPLATES, explain_reason

EXPECTED_CATEGORIES = [
    "insult",
    "dehumanization",
    "negative_stereotyping",
    "threat",
    "exclusion",
    "discrimination",
    "incitement_to_violence",
    "other",
]


def test_all_reason_categories_have_templates():
    for category in EXPECTED_CATEGORIES:
        assert category in REASON_TEMPLATES
        template = REASON_TEMPLATES[category]
        assert template.get("summary")
        assert template.get("details")


def test_every_reason_category_produces_an_explanation():
    for category in EXPECTED_CATEGORIES:
        result = explain_reason(
            reason=category,
            target="nationality",
            evidence=["should be kicked out"],
        )
        assert result is not None
        assert result["summary"]
        assert result["details"]
        assert result["grounded_in"]["reason"] == category
        assert result["grounded_in"]["target"] == "nationality"
        assert result["grounded_in"]["evidence"] == ["should be kicked out"]
        assert "should be kicked out" in result["details"]


def test_missing_reason_returns_none():
    assert explain_reason(reason=None, target=None, evidence=[]) is None
    assert explain_reason(reason="", target="gender", evidence=["x"]) is None
    assert explain_reason(reason="   ", target=None, evidence=[]) is None


def test_missing_target_omits_target_clause():
    result = explain_reason(reason="threat", target=None, evidence=["we will find you"])
    assert result is not None
    assert result["grounded_in"]["target"] is None
    assert "predicted target" not in result["details"]
    assert "we will find you" in result["details"]


def test_empty_evidence_produces_honest_fallback():
    result = explain_reason(reason="insult", target="gender", evidence=[])
    assert result is not None
    assert result["grounded_in"]["evidence"] == []
    assert "No specific supporting phrases" in result["details"]
    assert "gender" in result["details"]  # target clause still reported


def test_multiple_evidence_spans():
    result = explain_reason(
        reason="exclusion",
        target="nationality",
        evidence=["get out", "go home", "leave"],
    )
    assert result is not None
    assert result["grounded_in"]["evidence"] == ["get out", "go home", "leave"]
    # the two strongest quotes appear in the details
    assert "get out" in result["details"]
    assert "go home" in result["details"]


def test_unknown_reason_falls_back_to_other_template():
    result = explain_reason(reason="some_new_category", target=None, evidence=["x"])
    assert result is not None
    assert result["grounded_in"]["reason"] == "some_new_category"
    assert result["summary"]


def test_context_note_is_neutral_and_configurable():
    result = explain_reason(
        reason="exclusion",
        target="nationality",
        evidence=["get out"],
        context_used=True,
    )
    assert result is not None
    assert "context" in result["details"].lower()
    assert "caused" not in result["details"].lower()

    without = explain_reason(
        reason="exclusion", target="nationality", evidence=["get out"]
    )
    assert without is not None
    assert "context" not in without["details"].lower()


def test_no_unsupported_claims():
    result = explain_reason(
        reason="exclusion", target="nationality", evidence=["get out"]
    )
    assert result is not None
    combined = f"{result['summary']} {result['details']}".lower()
    for forbidden in ("intent", "meant to", "definitely", "caused", "proves"):
        assert forbidden not in combined


def test_templates_are_overridable():
    custom = {"insult": {"summary": "Custom summary.", "details": "Custom {evidence}."}}
    result = explain_reason(
        reason="insult", target=None, evidence=["dummy"], templates=custom
    )
    assert result is not None
    assert result["summary"] == "Custom summary."
    assert result["details"] == "Custom 'dummy'."


def test_evidence_quotes_formatting_via_details():
    single = explain_reason(reason="threat", target=None, evidence=["watch out"])
    assert single is not None
    assert "'watch out'" in single["details"]
