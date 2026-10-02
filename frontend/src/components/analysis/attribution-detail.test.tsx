import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import AttributionDetail from "./AttributionDetail";

const { explainMock } = vi.hoisted(() => ({ explainMock: vi.fn() }));

vi.mock("../../api/analysis", () => ({
  analysisApi: { explain: explainMock },
}));

const REQUEST = { text: "They should be kicked out.", context: "About immigrants" };

const OK_RESPONSE = {
  available: true,
  message: null,
  target: "hate",
  method: "integrated_gradients",
  steps: 16,
  predicted_label: "hate_speech",
  predicted_confidence: 0.86,
  explained_label: "hate_speech",
  context_used: true,
  tokens: [
    { index: 0, token: "[CLS]", score: 0.01 },
    { index: 1, token: "kicked", score: 0.8 },
    { index: 2, token: "out", score: 0.4 },
  ],
  spans: [],
};

beforeEach(() => {
  explainMock.mockReset();
});

describe("AttributionDetail", () => {
  it("renders nothing when disabled or without a request", () => {
    const { container } = render(
      <AttributionDetail request={null} enabled={true} />,
    );
    expect(container).toBeEmptyDOMElement();
    const second = render(<AttributionDetail request={REQUEST} enabled={false} />);
    expect(second.container).toBeEmptyDOMElement();
  });

  it("loads attributions and renders token chips, filtering special tokens", async () => {
    explainMock.mockResolvedValue(OK_RESPONSE);
    render(<AttributionDetail request={REQUEST} enabled={true} />);

    fireEvent.click(
      screen.getByRole("button", { name: /load token-level attribution/i }),
    );

    expect(await screen.findByText("kicked")).toBeInTheDocument();
    expect(screen.getByText("out")).toBeInTheDocument();
    expect(screen.queryByText("[CLS]")).toBeNull();
    expect(screen.getByText("integrated_gradients")).toBeInTheDocument();
    expect(explainMock).toHaveBeenCalledWith({
      text: REQUEST.text,
      context: REQUEST.context,
      target: "hate",
    });
  });

  it("shows the backend message when attribution is unavailable", async () => {
    explainMock.mockResolvedValue({
      ...OK_RESPONSE,
      available: false,
      message: "Model not loaded.",
      tokens: [],
    });
    render(<AttributionDetail request={REQUEST} enabled={true} />);
    fireEvent.click(
      screen.getByRole("button", { name: /load token-level attribution/i }),
    );
    expect(await screen.findByText("Model not loaded.")).toBeInTheDocument();
  });

  it("offers a retry when the request fails", async () => {
    explainMock.mockRejectedValue(new Error("boom"));
    render(<AttributionDetail request={REQUEST} enabled={true} />);
    fireEvent.click(
      screen.getByRole("button", { name: /load token-level attribution/i }),
    );
    expect(
      await screen.findByRole("button", { name: /retry/i }),
    ).toBeInTheDocument();
  });
});
