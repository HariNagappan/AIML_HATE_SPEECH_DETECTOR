import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import ReasoningCard from "./ReasoningCard";
import type { Reasoning } from "../../types/analysis";

const REASONING: Reasoning = {
  summary:
    'The current comment uses "They" to refer to "group of immigrants" from the previous comment and expresses hostile or exclusionary language toward that target.',
  method: "deterministic evidence + context-reference analysis",
  context_used: true,
  context_available: true,
  links: [
    {
      from_text: "They",
      from_start: 0,
      from_end: 4,
      to_text: "group of immigrants",
      to_start: 8,
      to_end: 27,
      pronoun: "they",
      relation: "refers_to",
    },
  ],
  evidence: [
    {
      text: "should all be kicked out",
      source: "current_comment",
      type: "current_span",
      reason: "Strongest attribution signal (score 0.90) for the predicted classification.",
      start: 9,
      end: 31,
      score: 0.9,
    },
    {
      text: "group of immigrants",
      source: "previous_comment",
      type: "context_target",
      reason: 'Identified by the reference analysis as the phrase "They" refers to.',
      start: 8,
      end: 27,
      score: null,
    },
  ],
};

describe("ReasoningCard", () => {
  it("renders the summary, connection chip and per-source evidence", () => {
    render(<ReasoningCard reasoning={REASONING} />);
    expect(
      screen.getByText("Why this was classified this way"),
    ).toBeInTheDocument();
    expect(screen.getByText(REASONING.summary)).toBeInTheDocument();
    expect(screen.getByText("context used")).toBeInTheDocument();
    expect(
      screen.getByRole("button", {
        name: /show connection: they refers to group of immigrants/i,
      }),
    ).toBeInTheDocument();
    expect(screen.getByText("previous comment")).toBeInTheDocument();
    expect(screen.getByText("current comment")).toBeInTheDocument();
    expect(screen.getByText(/not by a language model/i)).toBeInTheDocument();
  });

  it("fires onSelectLink with the link and its index", () => {
    const handler = vi.fn();
    render(<ReasoningCard reasoning={REASONING} onSelectLink={handler} />);
    fireEvent.click(screen.getByRole("button", { name: /show connection/i }));
    expect(handler).toHaveBeenCalledTimes(1);
    expect(handler.mock.calls[0][0].from_text).toBe("They");
    expect(handler.mock.calls[0][1]).toBe(0);
  });

  it("shows the comment-only state when no context was available", () => {
    render(
      <ReasoningCard
        reasoning={{
          ...REASONING,
          context_used: false,
          context_available: false,
          links: [],
          evidence: [],
        }}
      />,
    );
    expect(screen.getByText("comment only")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /show connection/i }),
    ).toBeNull();
  });
});
