import { beforeEach, describe, expect, it } from "vitest";
import {
  buildAnnotationJsonl,
  labeledCount,
  useAnnotationStore,
} from "./annotationStore";
import type { AnnotationSample } from "../types/annotation";

const SAMPLES: AnnotationSample[] = [
  { id: "a", text: "one", context: "previous message" },
  { id: "b", text: "two" },
  { id: "c", text: "three" },
];

beforeEach(() => {
  useAnnotationStore.setState({
    samples: [],
    labels: {},
    index: 0,
    source: null,
  });
});

describe("buildAnnotationJsonl", () => {
  it("exports only labeled samples, in sample order, with optional context", () => {
    const jsonl = buildAnnotationJsonl(SAMPLES, { b: "threat", a: "exclusion" });
    const lines = jsonl.split("\n");
    expect(lines).toHaveLength(2);
    expect(JSON.parse(lines[0])).toEqual({
      id: "a",
      text: "one",
      context: "previous message",
      reason: "exclusion",
    });
    expect(JSON.parse(lines[1])).toEqual({
      id: "b",
      text: "two",
      reason: "threat",
    });
  });

  it("omits context when absent and returns empty string with no labels", () => {
    const jsonl = buildAnnotationJsonl([SAMPLES[1]], { b: "other" });
    expect(jsonl).not.toContain("context");
    expect(buildAnnotationJsonl(SAMPLES, {})).toBe("");
  });
});

describe("annotation store flow", () => {
  it("labels the current sample and advances to the next unlabeled", () => {
    useAnnotationStore.getState().loadSamples(SAMPLES, "test");
    useAnnotationStore.getState().setLabel("insult");
    const state = useAnnotationStore.getState();
    expect(state.labels["a"]).toBe("insult");
    expect(state.index).toBe(1);
  });

  it("skip advances past labeled entries and wraps around", () => {
    useAnnotationStore.getState().loadSamples(SAMPLES, "test");
    useAnnotationStore.getState().setLabel("other"); // labels a, index 1
    useAnnotationStore.getState().skip(); // index 2
    useAnnotationStore.getState().skip(); // wraps back to 1 (a is labeled)
    expect(useAnnotationStore.getState().index).toBe(1);
  });

  it("moves to the completion index once everything is labeled", () => {
    useAnnotationStore.getState().loadSamples(SAMPLES, "test");
    for (let i = 0; i < SAMPLES.length; i += 1) {
      useAnnotationStore.getState().setLabel("other");
    }
    const state = useAnnotationStore.getState();
    expect(labeledCount(state.samples, state.labels)).toBe(3);
    expect(state.index).toBe(SAMPLES.length);
  });

  it("back never goes below zero and reset clears labels", () => {
    useAnnotationStore.getState().loadSamples(SAMPLES, "test");
    useAnnotationStore.getState().back();
    expect(useAnnotationStore.getState().index).toBe(0);
    useAnnotationStore.getState().setLabel("threat");
    useAnnotationStore.getState().reset();
    const state = useAnnotationStore.getState();
    expect(state.labels).toEqual({});
    expect(state.index).toBe(0);
  });
});
