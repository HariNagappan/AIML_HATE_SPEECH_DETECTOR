import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import ReasonCard from "./ReasonCard";
import TargetCard from "./TargetCard";
import type { AnalysisResult } from "../../types/analysis";

function buildResult(overrides: Partial<AnalysisResult>): AnalysisResult {
  return {
    prediction: { label: "hate", confidence: 0.9 },
    prediction_available: true,
    target: null,
    target_available: false,
    reason: null,
    reason_available: false,
    evidence: [],
    context_used: false,
    ...overrides,
  };
}

describe("TargetCard", () => {
  it("renders the predicted target group when available", () => {
    render(
      <TargetCard
        result={buildResult({
          target: { label: "nationality", confidence: 0.82 },
          target_available: true,
        })}
      />,
    );
    expect(screen.getByText("Nationality")).toBeInTheDocument();
    expect(screen.getByText(/82% confidence/i)).toBeInTheDocument();
  });

  it("explains unavailability instead of fabricating a target", () => {
    render(<TargetCard result={buildResult({})} />);
    expect(
      screen.getByText(/not available for this model/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/target-annotated training data/i),
    ).toBeInTheDocument();
  });
});

describe("ReasonCard", () => {
  it("renders the predicted reason category when available", () => {
    render(
      <ReasonCard
        result={buildResult({
          reason: { label: "exclusion", confidence: 0.7 },
          reason_available: true,
        })}
      />,
    );
    expect(screen.getByText("Exclusion")).toBeInTheDocument();
    expect(screen.getByText(/70% confidence/i)).toBeInTheDocument();
  });

  it("explains unavailability instead of fabricating a reason", () => {
    render(<ReasonCard result={buildResult({})} />);
    expect(
      screen.getByText(/not available for this model/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/reason-annotated model/i),
    ).toBeInTheDocument();
  });
});
