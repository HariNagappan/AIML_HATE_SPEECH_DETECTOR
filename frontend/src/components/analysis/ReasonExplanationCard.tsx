import { Lightbulb } from "lucide-react";
import Card from "../common/Card";
import Badge from "../common/Badge";
import { formatLabel } from "../../utils/formatting";
import type { ReasonExplanation } from "../../types/analysis";

interface ReasonExplanationCardProps {
  explanation: ReasonExplanation;
  /** 1-based display rank per evidence text (first occurrence). */
  rankByText: Record<string, number>;
  /** Activate the matching evidence span in the evidence section. */
  onSelectEvidence: (text: string) => void;
}

/**
 * Deterministic "Why this classification?" section — connects the predicted
 * reason to the extracted evidence. Text comes from the backend explanation
 * layer; this component only renders it.
 */
export default function ReasonExplanationCard({
  explanation,
  rankByText,
  onSelectEvidence,
}: ReasonExplanationCardProps) {
  const { grounded_in: grounding } = explanation;

  return (
    <Card
      title="Why this classification?"
      titleIcon={<Lightbulb className="h-3.5 w-3.5" aria-hidden="true" />}
      action={<Badge tone="info">grounded</Badge>}
      className="!border-accent/30"
    >
      <p className="kicker">Reason</p>
      <p className="mt-1 text-xl font-semibold text-ink">
        {formatLabel(grounding.reason)}
      </p>

      <p className="mt-3 max-w-[75ch] text-sm leading-relaxed text-ink">
        {explanation.summary}
      </p>

      <div className="mt-4 rounded-lg border border-edge bg-raised/40 p-4">
        <p className="kicker">Why?</p>
        <p className="mt-1.5 max-w-[75ch] text-sm leading-relaxed text-ink-soft">
          {explanation.details}
        </p>
      </div>

      {grounding.evidence.length > 0 ? (
        <div className="mt-4">
          <p className="kicker">Based on</p>
          <ul className="mt-2 flex flex-wrap gap-2">
            {grounding.evidence.map((text) => {
              const rank = rankByText[text];
              return (
                <li key={text}>
                  <button
                    type="button"
                    onClick={() => onSelectEvidence(text)}
                    className="inline-flex max-w-full items-center gap-2 rounded-md border border-edge bg-panel px-2.5 py-1 text-left text-xs text-ink-soft transition-colors hover:border-edge-strong hover:text-ink"
                    aria-label={`Highlight evidence span: ${text}`}
                  >
                    {rank ? (
                      <span
                        className="font-mono text-[10px] text-accent"
                        aria-hidden="true"
                      >
                        {rank}
                      </span>
                    ) : null}
                    <span className="min-w-0 break-words">“{text}”</span>
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      ) : null}

      <p className="mt-4 text-[11px] leading-relaxed text-ink-mute/80">
        Generated from the predicted reason and the extracted evidence — not by
        a language model. This explanation describes the model's structured
        output; it is not a claim about the author's intent.
      </p>
    </Card>
  );
}
