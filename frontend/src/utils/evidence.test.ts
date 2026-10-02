import { describe, expect, it } from "vitest";
import { buildHighlight, rankEvidence } from "./evidence";
import type { EvidenceItem } from "../types/analysis";

describe("buildHighlight", () => {
  it("uses backend character offsets when provided", () => {
    const text = "They are ruining everything and should be kicked out.";
    const start = text.indexOf("kicked out");
    const evidence: EvidenceItem[] = [
      {
        text: "kicked out",
        score: 0.84,
        start,
        end: start + "kicked out".length,
      },
    ];
    const result = buildHighlight(text, evidence);
    expect(result.allMatched).toBe(true);
    const highlighted = result.segments.filter((s) => s.evidenceIndex === 0);
    expect(highlighted).toHaveLength(1);
    expect(highlighted[0].text).toBe("kicked out");
  });

  it("consumes repeated phrases left-to-right", () => {
    const text = "bad bad worse bad";
    const evidence: EvidenceItem[] = [
      { text: "bad", score: 0.9 },
      { text: "bad", score: 0.5 },
    ];
    const result = buildHighlight(text, evidence);
    const starts = result.matches.map((m) => m.start).sort((a, b) => a - b);
    expect(starts).toEqual([0, 4]);
    // the third occurrence is not re-highlighted
    expect(result.matches.some((m) => m.start === 12)).toBe(false);
  });

  it("falls back to plain text when a span cannot be mapped", () => {
    const text = "hello world";
    const evidence: EvidenceItem[] = [{ text: "not present", score: 0.9 }];
    const result = buildHighlight(text, evidence);
    expect(result.allMatched).toBe(false);
    expect(result.matches).toHaveLength(0);
    expect(result.segments.every((s) => s.evidenceIndex === null)).toBe(true);
  });
});

describe("rankEvidence", () => {
  it("ranks by score descending and preserves original indices", () => {
    const evidence: EvidenceItem[] = [
      { text: "a", score: 0.2 },
      { text: "b", score: 0.9 },
      { text: "c", score: 0.5 },
    ];
    const ranked = rankEvidence(evidence);
    expect(ranked.map((entry) => entry.index)).toEqual([1, 2, 0]);
    expect(ranked.map((entry) => entry.rank)).toEqual([1, 2, 3]);
  });
});
