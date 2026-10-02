/**
 * Interaction tests for the evidence components:
 * hover attribution readout, clicking list items, unavailable/empty states.
 */
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import EvidenceList from "./EvidenceList";
import EvidenceViewer from "./EvidenceViewer";
import type { EvidenceItem } from "../../types/analysis";

const KICKED_OUT: EvidenceItem = { text: "kicked out", score: 0.84 };

/** EvidenceViewer is controlled — wrap it so hover can update activeIndex. */
function InteractiveViewer({
  text,
  evidence,
  rankByIndex,
}: {
  text: string;
  evidence: EvidenceItem[];
  rankByIndex: Map<number, number>;
}) {
  const [active, setActive] = useState<number | null>(null);
  return (
    <EvidenceViewer
      text={text}
      evidence={evidence}
      headAvailable={true}
      activeIndex={active}
      onActivate={setActive}
      rankByIndex={rankByIndex}
    />
  );
}

describe("EvidenceViewer states", () => {
  const emptyProps = {
    text: "They are ruining everything.",
    evidence: [] as EvidenceItem[],
    activeIndex: null,
    onActivate: () => {},
    rankByIndex: new Map<number, number>(),
  };

  it("shows the unavailable message when the head is missing", () => {
    render(<EvidenceViewer {...emptyProps} headAvailable={false} />);
    expect(
      screen.getByText(/evidence extraction unavailable/i),
    ).toBeInTheDocument();
  });

  it("shows the empty message when no spans were returned", () => {
    render(<EvidenceViewer {...emptyProps} headAvailable={true} />);
    expect(
      screen.getByText(/no evidence spans were returned/i),
    ).toBeInTheDocument();
  });

  it("shows the attribution readout when a highlighted span is hovered", async () => {
    const text = "They are ruining everything and should be kicked out.";
    render(
      <InteractiveViewer
        text={text}
        evidence={[KICKED_OUT]}
        rankByIndex={new Map([[0, 1]])}
      />,
    );
    // initial hint
    expect(
      screen.getByText(/hover, focus or tap a highlighted span/i),
    ).toBeInTheDocument();
    // hover the highlighted span in the original comment
    fireEvent.mouseEnter(
      screen.getByRole("button", { name: /kicked out/i }),
    );
    expect(await screen.findByText(/attribution 0\.84/)).toBeInTheDocument();
  });
});

describe("EvidenceList", () => {
  const evidence: EvidenceItem[] = [
    { text: "low span", score: 0.2 },
    { text: "high span", score: 0.9 },
  ];

  it("reports the original index of the clicked row via onSelect", () => {
    const onSelect = vi.fn();
    render(
      <EvidenceList
        evidence={evidence}
        activeIndex={null}
        onActivate={() => {}}
        onSelect={onSelect}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /high span/i }));
    expect(onSelect).toHaveBeenCalledWith(1);
  });

  it("falls back to onActivate when onSelect is not provided", () => {
    const onActivate = vi.fn();
    render(
      <EvidenceList
        evidence={evidence}
        activeIndex={null}
        onActivate={onActivate}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /low span/i }));
    expect(onActivate).toHaveBeenCalledWith(0);
  });

  it("renders nothing when the evidence list is empty", () => {
    const { container } = render(
      <EvidenceList evidence={[]} activeIndex={null} onActivate={() => {}} />,
    );
    expect(container).toBeEmptyDOMElement();
  });
});
