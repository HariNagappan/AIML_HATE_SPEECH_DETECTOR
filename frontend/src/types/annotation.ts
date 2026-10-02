/**
 * Reason-annotation types for the annotation workspace.
 *
 * These mirror the canonical reason categories the backend trains on
 * (`app/datasets/label_mapping.py` → CANONICAL_REASON) and the JSONL format
 * `scripts/import_reason_annotations.py` consumes.
 */

export const REASON_CATEGORIES = [
  "insult",
  "dehumanization",
  "negative_stereotyping",
  "threat",
  "exclusion",
  "discrimination",
  "incitement_to_violence",
  "other",
] as const;

export type ReasonCategory = (typeof REASON_CATEGORIES)[number];

/** One comment offered for reason annotation. */
export interface AnnotationSample {
  id: string;
  text: string;
  context?: string | null;
}

/** One exported annotation line (JSONL). */
export interface ExportedAnnotation {
  id: string;
  text: string;
  context?: string;
  reason: ReasonCategory;
}
