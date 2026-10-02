/**
 * Annotation workspace store.
 *
 * Labels are kept in the browser (localStorage, like the analysis history) —
 * nothing leaves the machine. `buildAnnotationJsonl` produces exactly the
 * format `backend/scripts/import_reason_annotations.py` consumes, so the
 * exported file can be merged into the processed datasets and used to train
 * the reason head.
 */
import { create } from "zustand";
import { persist } from "zustand/middleware";
import type {
  AnnotationSample,
  ExportedAnnotation,
  ReasonCategory,
} from "../types/annotation";

export const ANNOTATION_STORAGE_KEY =
  "context-aware-hate-speech.annotation-workspace.v1";

interface AnnotationState {
  samples: AnnotationSample[];
  /** sample id → chosen reason category */
  labels: Record<string, ReasonCategory>;
  /** current sample index; `samples.length` means "nothing left unlabeled" */
  index: number;
  /** where the current sample set came from (display only) */
  source: string | null;

  loadSamples: (samples: AnnotationSample[], source: string) => void;
  setLabel: (reason: ReasonCategory) => void;
  skip: () => void;
  back: () => void;
  gotoFirstUnlabeled: () => void;
  reset: () => void;
}

/** First unlabeled index at/after `from`, wrapping around; length if none. */
export function nextUnlabeledIndex(
  samples: AnnotationSample[],
  labels: Record<string, ReasonCategory>,
  from: number,
): number {
  const total = samples.length;
  for (let step = 0; step < total; step += 1) {
    const i = (from + step) % total;
    if (!(samples[i].id in labels)) {
      return i;
    }
  }
  return total;
}

export function labeledCount(
  samples: AnnotationSample[],
  labels: Record<string, ReasonCategory>,
): number {
  return samples.reduce((acc, s) => (s.id in labels ? acc + 1 : acc), 0);
}

export function labelDistribution(
  samples: AnnotationSample[],
  labels: Record<string, ReasonCategory>,
): Record<string, number> {
  const dist: Record<string, number> = {};
  for (const sample of samples) {
    const label = labels[sample.id];
    if (label) {
      dist[label] = (dist[label] ?? 0) + 1;
    }
  }
  return dist;
}

/** JSONL export — only labeled samples, in sample order. */
export function buildAnnotationJsonl(
  samples: AnnotationSample[],
  labels: Record<string, ReasonCategory>,
): string {
  const lines: string[] = [];
  for (const sample of samples) {
    const reason = labels[sample.id];
    if (!reason) {
      continue;
    }
    const payload: ExportedAnnotation = {
      id: sample.id,
      text: sample.text,
      ...(sample.context ? { context: sample.context } : {}),
      reason,
    };
    lines.push(JSON.stringify(payload));
  }
  return lines.join("\n");
}

export const useAnnotationStore = create<AnnotationState>()(
  persist(
    (set, get) => ({
      samples: [],
      labels: {},
      index: 0,
      source: null,

      loadSamples: (samples, source) =>
        set({ samples, labels: {}, index: 0, source }),

      setLabel: (reason) => {
        const { samples, labels, index } = get();
        const sample = samples[index];
        if (!sample) {
          return;
        }
        const nextLabels = { ...labels, [sample.id]: reason };
        set({
          labels: nextLabels,
          index: nextUnlabeledIndex(samples, nextLabels, index + 1),
        });
      },

      skip: () => {
        const { samples, labels, index } = get();
        set({ index: nextUnlabeledIndex(samples, labels, index + 1) });
      },

      back: () => {
        const { index } = get();
        set({ index: Math.max(0, index - 1) });
      },

      gotoFirstUnlabeled: () => {
        const { samples, labels } = get();
        set({ index: nextUnlabeledIndex(samples, labels, 0) });
      },

      reset: () => set({ labels: {}, index: 0 }),
    }),
    {
      name: ANNOTATION_STORAGE_KEY,
      partialize: (state) => ({
        samples: state.samples,
        labels: state.labels,
        index: state.index,
        source: state.source,
      }),
    },
  ),
);
