const LOWERCASE_JOINERS = new Set([
  "or",
  "and",
  "of",
  "the",
  "to",
  "in",
  "for",
  "with",
]);

/**
 * `"negative_stereotyping"` → `"Negative stereotyping"`,
 * `"NATIONALITY"` → `"Nationality"`, `"hate"` → `"Hate"`.
 */
export function formatLabel(label: string): string {
  const cleaned = String(label ?? "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (!cleaned) {
    return "—";
  }
  return cleaned
    .toLowerCase()
    .split(" ")
    .map((word, index) => {
      if (index > 0 && LOWERCASE_JOINERS.has(word)) {
        return word;
      }
      return word.charAt(0).toUpperCase() + word.slice(1);
    })
    .join(" ");
}

/** `0.9412` → `"94%"`. */
export function formatConfidence(value: number): string {
  if (!Number.isFinite(value)) {
    return "—";
  }
  return `${Math.round(value * 100)}%`;
}

/** `0.8412` → `"0.84"` (attribution scores). */
export function formatScore(value: number): string {
  if (!Number.isFinite(value)) {
    return "—";
  }
  return value.toFixed(2);
}

/** Relative time, e.g. `"2 minutes ago"`. */
export function timeAgo(timestamp: number): string {
  const seconds = Math.max(0, Math.floor((Date.now() - timestamp) / 1000));
  if (seconds < 30) {
    return "just now";
  }
  if (seconds < 60) {
    return `${seconds} seconds ago`;
  }
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    return minutes === 1 ? "1 minute ago" : `${minutes} minutes ago`;
  }
  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    return hours === 1 ? "1 hour ago" : `${hours} hours ago`;
  }
  const days = Math.floor(hours / 24);
  return days === 1 ? "yesterday" : `${days} days ago`;
}

/** Absolute local timestamp for titles/tooltips. */
export function formatDateTime(timestamp: number): string {
  return new Date(timestamp).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

export type SemanticTone = "normal" | "offensive" | "hate" | "neutral";

/**
 * Display-only heuristic mapping a hate-style label to a semantic tone for
 * the known backend label sets. Colour is never the only signal — every tone
 * is accompanied by labels and icons. Unknown labels render neutral.
 */
export function labelTone(label: string): SemanticTone {
  const key = String(label ?? "")
    .trim()
    .toLowerCase()
    .replace(/[_-]+/g, " ");
  if (key === "hate" || key === "hatespeech" || key === "hate speech") {
    return "hate";
  }
  if (key === "offensive") {
    return "offensive";
  }
  if (key === "normal" || key === "neither") {
    return "normal";
  }
  return "neutral";
}
