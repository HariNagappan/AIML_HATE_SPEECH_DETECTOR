import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import ReasonExplanationCard from "./ReasonExplanationCard";
import type { ReasonExplanation } from "../../types/analysis";

const explanation: ReasonExplanation = {
  summary:
    "The highlighted language expresses exclusion of the targeted group.",
  details:
    "The phrase 'should be kicked out' advocates removing members of the targeted group, which corresponds to the predicted exclusion category.",
  grounded_in: {
    reason: "exclusion",
    target: "nationality",
    evidence: ["should be kicked out"],
  },
};

describe("ReasonExplanationCard", () => {
  it("renders reason, summary, details and the honesty note", () => {
    render(
      <ReasonExplanationCard
        explanation={explanation}
        rankByText={{ "should be kicked out": 2 }}
        onSelectEvidence={() => {}}
      />,
    );
    expect(screen.getByText("Why this classification?")).toBeInTheDocument();
    expect(screen.getByText("Exclusion")).toBeInTheDocument();
    expect(screen.getByText(explanation.summary)).toBeInTheDocument();
    expect(screen.getByText(explanation.details)).toBeInTheDocument();
    expect(screen.getByText(/not by a language model/i)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /highlight evidence span/i }),
    ).toBeInTheDocument();
  });

  it("fires onSelectEvidence when a grounded evidence chip is clicked", () => {
    const handler = vi.fn();
    render(
      <ReasonExplanationCard
        explanation={explanation}
        rankByText={{ "should be kicked out": 2 }}
        onSelectEvidence={handler}
      />,
    );
    fireEvent.click(
      screen.getByRole("button", { name: /highlight evidence span/i }),
    );
    expect(handler).toHaveBeenCalledWith("should be kicked out");
  });

  it("renders without grounded evidence chips when none are listed", () => {
    const without: ReasonExplanation = {
      ...explanation,
      grounded_in: { reason: "other", target: null, evidence: [] },
    };
    render(
      <ReasonExplanationCard
        explanation={without}
        rankByText={{}}
        onSelectEvidence={() => {}}
      />,
    );
    expect(screen.queryByText("Based on")).not.toBeInTheDocument();
  });
});
