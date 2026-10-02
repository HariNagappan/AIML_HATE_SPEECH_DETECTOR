import type { EvidenceItem } from "../types/analysis";

/**
 * Utilities that map evidence spans onto the original comment for inline
 * highlighting.
 *
 * Rules (see project spec §30):
 * - Preserve the exact original text (segments are slices of the original).
 * - Prefer backend-provided character offsets when they line up with the text.
 * - Otherwise use a robust matcher: exact → normalised (case/whitespace
 *   tolerant, with an index map back to the original text).
 * - Handle repeated phrases by consuming occurrences left-to-right.
 * - Never use naive `String.replace()`.
 * - If some spans cannot be mapped reliably, the caller should fall back to
 *   the evidence list (partial highlighting is still shown).
 */

export interface EvidenceMatch {
  /** Index into the original evidence array. */
  evidenceIndex: number;
  start: number;
  end: number;
}

export interface HighlightSegment {
  text: string;
  /** Index into the original evidence array, or null for plain text. */
  evidenceIndex: number | null;
}

export interface HighlightResult {
  segments: HighlightSegment[];
  matches: EvidenceMatch[];
  /** True when every non-empty evidence item was located in the text. */
  allMatched: boolean;
  /** Number of non-empty evidence items considered. */
  considered: number;
}

interface Range {
  start: number;
  end: number;
}

interface NormalizedText {
  normalized: string;
  /** map[i] = index in the original string of normalized character i. */
  map: number[];
}

/** Lowercase + collapse whitespace runs to single spaces, keeping an index map. */
function normalizeWithMap(source: string): NormalizedText {
  let normalized = "";
  const map: number[] = [];
  let pendingSpaceIndex = -1;

  for (let i = 0; i < source.length; i += 1) {
    const ch = source[i];
    if (/\s/.test(ch)) {
      if (normalized.length > 0 && pendingSpaceIndex === -1) {
        pendingSpaceIndex = i;
      }
      continue;
    }
    if (pendingSpaceIndex !== -1) {
      normalized += " ";
      map.push(pendingSpaceIndex);
      pendingSpaceIndex = -1;
    }
    normalized += ch.toLowerCase();
    map.push(i);
  }
  return { normalized, map };
}

function locate(text: string, needle: string, from: number): Range | null {
  const searchFrom = Math.max(0, from);
  const exact = text.indexOf(needle, searchFrom);
  if (exact !== -1) {
    return { start: exact, end: exact + needle.length };
  }

  // Normalised fallback (case / whitespace differences).
  const haystack = normalizeWithMap(text);
  const target = normalizeWithMap(needle);
  if (!target.normalized) {
    return null;
  }

  let cursor = 0;
  while (cursor <= haystack.normalized.length - target.normalized.length) {
    const found = haystack.normalized.indexOf(target.normalized, cursor);
    if (found === -1) {
      return null;
    }
    const lastIndex = found + target.normalized.length - 1;
    const start = haystack.map[found];
    const end = haystack.map[lastIndex] + 1;
    if (start >= searchFrom) {
      return { start, end };
    }
    cursor = found + 1;
  }
  return null;
}

function overlaps(a: Range, b: Range): boolean {
  return a.start < b.end && b.start < a.end;
}

function overlapsAny(candidate: Range, taken: Range[]): boolean {
  return taken.some((range) => overlaps(candidate, range));
}

function spansAgree(textSlice: string, evidenceText: string): boolean {
  return normalizeWithMap(textSlice).normalized === normalizeWithMap(evidenceText).normalized;
}

interface PendingItem {
  index: number;
  needle: string;
  score: number;
}

export function buildHighlight(
  text: string,
  evidence: EvidenceItem[],
): HighlightResult {
  const taken: Range[] = [];
  const matches: EvidenceMatch[] = [];

  const nonEmpty: number[] = [];
  const pending: PendingItem[] = [];

  evidence.forEach((item, index) => {
    const needle = typeof item.text === "string" ? item.text.trim() : "";
    if (!needle) {
      return;
    }
    nonEmpty.push(index);

    // 1) Prefer backend offsets when they line up with the text.
    if (
      typeof item.start === "number" &&
      typeof item.end === "number" &&
      item.start >= 0 &&
      item.end <= text.length &&
      item.end > item.start
    ) {
      const range: Range = { start: item.start, end: item.end };
      if (
        spansAgree(text.slice(range.start, range.end), needle) &&
        !overlapsAny(range, taken)
      ) {
        taken.push(range);
        matches.push({ evidenceIndex: index, start: range.start, end: range.end });
        return;
      }
    }

    pending.push({ index, needle, score: Number.isFinite(item.score) ? item.score : 0 });
  });

  // 2) Text matching for everything else. Higher scores claim spans first so
  //    conflicts resolve in favour of the strongest evidence.
  pending.sort((a, b) => b.score - a.score || a.index - b.index);

  const searchFromByNeedle = new Map<string, number>();

  for (const item of pending) {
    const from = searchFromByNeedle.get(item.needle) ?? 0;
    let candidate = locate(text, item.needle, from);

    if (candidate && overlapsAny(candidate, taken)) {
      // Try the next occurrence (repeated phrases are consumed left-to-right).
      candidate = locate(text, item.needle, candidate.end);
    }
    if (!candidate || overlapsAny(candidate, taken)) {
      continue; // cannot map reliably — the list view still shows this item
    }

    taken.push(candidate);
    matches.push({ evidenceIndex: item.index, start: candidate.start, end: candidate.end });
    searchFromByNeedle.set(item.needle, candidate.end);
  }

  // 3) Sweep in text order (drop any residual overlaps defensively).
  matches.sort((a, b) => a.start - b.start || a.end - b.end);
  const finalMatches: EvidenceMatch[] = [];
  let cursor = 0;
  for (const match of matches) {
    if (match.start < cursor) {
      continue;
    }
    finalMatches.push(match);
    cursor = match.end;
  }

  const segments: HighlightSegment[] = [];
  let position = 0;
  for (const match of finalMatches) {
    if (match.start > position) {
      segments.push({ text: text.slice(position, match.start), evidenceIndex: null });
    }
    segments.push({
      text: text.slice(match.start, match.end),
      evidenceIndex: match.evidenceIndex,
    });
    position = match.end;
  }
  if (position < text.length) {
    segments.push({ text: text.slice(position), evidenceIndex: null });
  }

  return {
    segments,
    matches: finalMatches,
    allMatched: finalMatches.length === nonEmpty.length,
    considered: nonEmpty.length,
  };
}

export interface RankedEvidence {
  item: EvidenceItem;
  /** Index into the original evidence array. */
  index: number;
  /** 1-based rank (strongest first). */
  rank: number;
}

/** Evidence ranked by attribution score, descending (stable for ties). */
export function rankEvidence(evidence: EvidenceItem[]): RankedEvidence[] {
  return evidence
    .map((item, index) => ({ item, index }))
    .sort(
      (a, b) =>
        (Number.isFinite(b.item.score) ? b.item.score : 0) -
          (Number.isFinite(a.item.score) ? a.item.score : 0) || a.index - b.index,
    )
    .map((entry, position) => ({ ...entry, rank: position + 1 }));
}
