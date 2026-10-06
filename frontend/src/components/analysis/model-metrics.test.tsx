import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import ModelMetrics from "./ModelMetrics";
import type { ModelInfo } from "../../types/analysis";

const { modelInfoMock } = vi.hoisted(() => ({ modelInfoMock: vi.fn() }));

vi.mock("../../api/analysis", () => ({
  analysisApi: { modelInfo: modelInfoMock },
}));

function buildModelInfo(metrics: ModelInfo["metrics"]): ModelInfo {
  return {
    model_name: "bert-base-uncased",
    device: "cpu",
    hidden_size: 768,
    num_labels: { hate: 3 },
    trained: true,
    trained_heads: ["hate"],
    version: "0.1.0",
    architecture: "context",
    checkpoint: "checkpoints/cc_context/best.pt",
    loaded: true,
    error: null,
    metrics,
  };
}

const METRICS: NonNullable<ModelInfo["metrics"]> = {
  split: "test",
  dataset: "counter_context",
  num_examples: 713,
  evaluated_at: "2026-10-06T12:00:00+00:00",
  accuracy: 0.5947,
  macro_precision: 0.5512,
  macro_recall: 0.5784,
  macro_f1: 0.5501,
  weighted_precision: 0.5901,
  weighted_recall: 0.5947,
  weighted_f1: 0.5923,
};

beforeEach(() => {
  modelInfoMock.mockReset();
});

describe("ModelMetrics", () => {
  it("renders accuracy, precision, recall and F1 from the API", async () => {
    modelInfoMock.mockResolvedValue(buildModelInfo(METRICS));
    render(<ModelMetrics />);

    expect(await screen.findByText("59.5%")).toBeInTheDocument();
    expect(screen.getByText("55.1%")).toBeInTheDocument();
    expect(screen.getByText("57.8%")).toBeInTheDocument();
    expect(screen.getByText("55.0%")).toBeInTheDocument();
    expect(screen.getByText(/713 examples/)).toBeInTheDocument();
    expect(screen.getByText(/counter context test split/i)).toBeInTheDocument();
  });

  it("shows an explicit unavailable message when no metrics exist", async () => {
    modelInfoMock.mockResolvedValue(buildModelInfo(null));
    render(<ModelMetrics />);

    expect(
      await screen.findByText(/no evaluation metrics are available/i),
    ).toBeInTheDocument();
  });

  it("shows the offline state when the backend is unreachable", async () => {
    modelInfoMock.mockRejectedValue(new Error("Network Error"));
    render(<ModelMetrics />);

    expect(await screen.findByText(/backend not reachable/i)).toBeInTheDocument();
  });
});
