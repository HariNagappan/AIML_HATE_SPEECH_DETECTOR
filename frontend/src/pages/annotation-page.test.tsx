import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import AnnotationPage from "./AnnotationPage";
import { useAnnotationStore } from "../store/annotationStore";

const SAMPLES = [
  { id: "s1", text: "first comment" },
  { id: "s2", text: "second comment" },
];

beforeEach(() => {
  useAnnotationStore.setState({
    samples: [],
    labels: {},
    index: 0,
    source: null,
  });
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => ({
      ok: true,
      json: async () => SAMPLES,
    })) as unknown as typeof fetch,
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
});

function renderPage() {
  return render(
    <MemoryRouter>
      <AnnotationPage />
    </MemoryRouter>,
  );
}

describe("AnnotationPage", () => {
  it("loads samples and advances when a category button is clicked", async () => {
    renderPage();
    expect(await screen.findByText("first comment")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /exclusion/i }));

    expect(await screen.findByText("second comment")).toBeInTheDocument();
    expect(screen.getByText(/1 \/ 2 labeled/)).toBeInTheDocument();
    expect(useAnnotationStore.getState().labels["s1"]).toBe("exclusion");
  });

  it("supports keyboard labeling (digit keys)", async () => {
    renderPage();
    expect(await screen.findByText("first comment")).toBeInTheDocument();

    fireEvent.keyDown(window, { key: "3" }); // negative_stereotyping

    expect(await screen.findByText("second comment")).toBeInTheDocument();
    expect(useAnnotationStore.getState().labels["s1"]).toBe(
      "negative_stereotyping",
    );
  });
});
