import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import PredictionCard from "./PredictionCard";
import type { AnalysisResult } from "../../types/analysis";

function buildResult(overrides: Partial<AnalysisResult>): AnalysisResult {
  return {
    prediction: null,
    prediction_available: false,
    target: null,
    target_available: false,
    reason: null,
    reason_available: false,
    evidence: [],
    context_used: false,
    ...overrides,
  };
}

describe("PredictionCard", () => {
  it("shows the predicted label and the full class breakdown", () => {
    render(
      <PredictionCard
        result={buildResult({
          prediction: {
            label: "hate_speech",
            confidence: 0.866,
            probabilities: {
              hate_speech: 0.866,
              counter_speech: 0.101,
              neither: 0.033,
            },
          },
          prediction_available: true,
        })}
      />,
    );

    // top label appears both as the headline and as a class row
    expect(screen.getAllByText("Hate Speech").length).toBeGreaterThanOrEqual(1);
    // the other two classes are visible even though they did not win
    expect(screen.getByText("Counter Speech")).toBeInTheDocument();
    expect(screen.getByText("Neither")).toBeInTheDocument();
    expect(screen.getByText("10%")).toBeInTheDocument();
    expect(screen.getByText("3%")).toBeInTheDocument();
  });

  it("explains unavailability when no trained head is loaded", () => {
    render(<PredictionCard result={buildResult({})} />);
    expect(
      screen.getByText(/not available for this model/i),
    ).toBeInTheDocument();
  });
});
