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
}

export default function EvidenceViewer({
  text,
  evidence,
  headAvailable,
  activeIndex,
  onActivate,
  rankByIndex,
}: EvidenceViewerProps) {
  const highlight = useMemo(() => buildHighlight(text, evidence), [text, evidence]);

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

  if (evidence.length === 0) {
    return (
      <Card title="Evidence">
        <p className="text-sm text-ink-soft">
          No evidence spans were returned for this analysis.
        </p>
      </Card>
    );
  }

  const activeItem =
    activeIndex !== null && activeIndex >= 0 && activeIndex < evidence.length
      ? evidence[activeIndex]
      : null;

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

      <div className="mt-4 rounded-lg border border-edge bg-raised/50 p-4 sm:p-5">
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
                  evidence[segment.evidenceIndex]?.score ?? 0,
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

      <div className="mt-3 min-h-[1.5rem]" aria-live="polite">
        {activeItem ? (
          <p className="font-mono text-xs text-ink-soft">
            Evidence {activeIndex !== null ? rankByIndex.get(activeIndex) ?? "-" : "-"}{' '}
            · “{activeItem.text}” · attribution {formatScore(activeItem.score)}
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
