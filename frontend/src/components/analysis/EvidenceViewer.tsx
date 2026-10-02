import { useMemo } from "react";
import { Highlighter, Info } from "lucide-react";
import Card from "../common/Card";
import Badge from "../common/Badge";
import { buildHighlight } from "../../utils/evidence";
import { formatScore } from "../../utils/formatting";
import type { EvidenceItem } from "../../types/analysis";

interface EvidenceViewerProps {
  /** The analyzed comment (original text, unmodified). */
  text: string;
  evidence: EvidenceItem[];
  /** Whether the hate head was available (a prediction was produced). */
  headAvailable: boolean;
  activeIndex: number | null;
  onActivate: (index: number | null) => void;
  /** 1-based display rank per evidence index (matches the evidence list). */
  rankByIndex: Map<number, number>;
  /** Previous comment text, rendered as a second ("context") block. */
  contextText?: string | null;
  /** Evidence spans located in the previous comment (offsets into contextText). */
  contextEvidence?: EvidenceItem[];
  /**
   * Display-only extra spans injected into the current-comment highlight
   * (e.g. the referring pronoun found by the context analysis).
   */
  currentExtras?: EvidenceItem[];
  /** Active connection line, e.g. `“They” → “group of immigrants”`. */
  connection?: string | null;
}

export default function EvidenceViewer({
  text,
  evidence,
  headAvailable,
  activeIndex,
  onActivate,
  rankByIndex,
  contextText = null,
  contextEvidence = [],
  currentExtras = [],
  connection = null,
}: EvidenceViewerProps) {
  const displayEvidence = useMemo(
    () => [...evidence, ...currentExtras],
    [evidence, currentExtras],
  );
  const highlight = useMemo(
    () => buildHighlight(text, displayEvidence),
    [text, displayEvidence],
  );
  const hasContext = Boolean(contextText && contextText.trim());
  const contextHighlight = useMemo(
    () => buildHighlight(contextText ?? "", contextEvidence),
    [contextText, contextEvidence],
  );

  if (!headAvailable) {
    return (
      <Card title="Evidence" muted>
        <div className="flex items-start gap-3">
          <Info className="mt-0.5 h-4 w-4 shrink-0 text-ink-mute" aria-hidden="true" />
          <p className="text-sm text-ink-soft">
            Evidence extraction unavailable for this analysis.
          </p>
        </div>
      </Card>
    );
  }

  if (evidence.length === 0 && currentExtras.length === 0 && contextEvidence.length === 0) {
    return (
      <Card title="Evidence">
        <p className="text-sm text-ink-soft">
          No evidence spans were returned for this analysis.
        </p>
      </Card>
    );
  }

  const activeItem =
    activeIndex !== null && activeIndex < displayEvidence.length
      ? displayEvidence[activeIndex]
      : null;
  const activeIsOriginal = activeIndex !== null && activeIndex < evidence.length;

  return (
    <Card
      title="Evidence"
      titleIcon={<Highlighter className="h-3.5 w-3.5" aria-hidden="true" />}
      action={<Badge tone="info">attribution-based</Badge>}
    >
      <p className="max-w-3xl text-xs leading-relaxed text-ink-mute">
        Evidence highlights are attribution-based signals from the model. They
        indicate which text spans contributed strongly to the prediction; they
        are not generated natural-language explanations.
      </p>

      {hasContext ? <p className="kicker mt-4">Current comment</p> : null}
      <div
        className={[
          "rounded-lg border border-edge bg-raised/50 p-4 sm:p-5",
          hasContext ? "mt-2" : "mt-4",
        ].join(" ")}
      >
        <p className="max-w-[75ch] break-words text-[15px] leading-8 text-ink">
          {highlight.segments.map((segment, segmentIndex) =>
            segment.evidenceIndex === null ? (
              <span key={segmentIndex}>{segment.text}</span>
            ) : (
              <button
                key={segmentIndex}
                type="button"
                className={[
                  "inline cursor-pointer rounded-[4px] border-b-2 border-accent/60 bg-accent/15 px-0.5 text-ink",
                  "transition-colors hover:bg-accent/25 focus-visible:bg-accent/25",
                  activeIndex === segment.evidenceIndex
                    ? "bg-accent/30 ring-1 ring-accent/50"
                    : "",
                ].join(" ")}
                aria-label={`Evidence span "${segment.text}", attribution ${formatScore(
                  displayEvidence[segment.evidenceIndex]?.score ?? 0,
                )}`}
                onClick={() =>
                  onActivate(
                    activeIndex === segment.evidenceIndex
                      ? null
                      : segment.evidenceIndex,
                  )
                }
                onMouseEnter={() => onActivate(segment.evidenceIndex)}
                onMouseLeave={() => onActivate(null)}
                onFocus={() => onActivate(segment.evidenceIndex)}
                onBlur={() => onActivate(null)}
              >
                {rankByIndex.get(segment.evidenceIndex) !== undefined ? (
                  <span
                    className="mr-1 align-super font-mono text-[10px] text-accent"
                    aria-hidden="true"
                  >
                    {rankByIndex.get(segment.evidenceIndex)}
                  </span>
                ) : null}
                {segment.text}
              </button>
            ),
          )}
        </p>
      </div>

      {hasContext ? (
        <div className="mt-3 rounded-lg border border-dashed border-edge-strong bg-raised/30 p-4 sm:p-5">
          <p className="kicker mb-2">Previous comment (context)</p>
          <p className="max-w-[75ch] break-words text-[15px] leading-8 text-ink-soft">
            {contextHighlight.segments.map((segment, segmentIndex) =>
              segment.evidenceIndex === null ? (
                <span key={segmentIndex}>{segment.text}</span>
              ) : (
                <mark
                  key={segmentIndex}
                  className="rounded-[4px] border-b-2 border-accent/60 bg-accent/10 px-0.5 text-ink"
                >
                  {segment.text}
                </mark>
              ),
            )}
          </p>
        </div>
      ) : null}

      {connection ? (
        <p className="mt-3 font-mono text-xs text-accent" role="status">
          Connection: {connection}
        </p>
      ) : null}

      <div className="mt-3 min-h-[1.5rem]" aria-live="polite">
        {activeItem ? (
          <p className="font-mono text-xs text-ink-soft">
            {activeIsOriginal
              ? `Evidence ${rankByIndex.get(activeIndex as number) ?? "-"} · `
              : "Referring span · "}
            “{activeItem.text}”
            {activeIsOriginal
              ? ` · attribution ${formatScore(activeItem.score)}`
              : ""}
          </p>
        ) : (
          <p className="text-xs text-ink-mute">
            Hover, focus or tap a highlighted span to see its attribution score.
          </p>
        )}
      </div>

      {!highlight.allMatched ? (
        <p className="mt-1 text-[11px] text-ink-mute/80">
          Some spans could not be mapped onto the exact text — see the evidence
          list below for all returned spans.
        </p>
      ) : null}
    </Card>
  );
}
