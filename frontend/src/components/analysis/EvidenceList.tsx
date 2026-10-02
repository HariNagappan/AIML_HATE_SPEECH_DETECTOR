import Card from "../common/Card";
import { ListOrdered } from "lucide-react";
import { rankEvidence } from "../../utils/evidence";
import { formatScore } from "../../utils/formatting";
import type { EvidenceItem } from "../../types/analysis";

interface EvidenceListProps {
  evidence: EvidenceItem[];
  activeIndex: number | null;
  onActivate: (index: number | null) => void;
  /** Called on click (e.g. to scroll to the reason explanation). */
  onSelect?: (index: number | null) => void;
}

/** Structured, score-ranked list of the evidence spans (strongest first). */
export default function EvidenceList({
  evidence,
  activeIndex,
  onActivate,
  onSelect,
}: EvidenceListProps) {
  if (evidence.length === 0) {
    return null;
  }

  const ranked = rankEvidence(evidence);

  return (
    <Card
      title="Evidence list"
      titleIcon={<ListOrdered className="h-3.5 w-3.5" aria-hidden="true" />}
    >
      <ul className="divide-y divide-edge/70">
        {ranked.map(({ item, index, rank }) => (
          <li key={index}>
            <button
              type="button"
              className={[
                "flex w-full items-start justify-between gap-3 rounded-md px-2 py-2.5 text-left transition-colors",
                "hover:bg-raised/60 focus-visible:bg-raised/60",
                activeIndex === index ? "bg-raised/70" : "",
              ].join(" ")}
              onClick={() =>
                (onSelect ?? onActivate)(activeIndex === index ? null : index)
              }
              onMouseEnter={() => onActivate(index)}
              onMouseLeave={() => onActivate(null)}
            >
              <span className="flex min-w-0 items-start gap-3">
                <span
                  className="mt-0.5 shrink-0 font-mono text-[11px] tabular-nums text-ink-mute"
                  aria-hidden="true"
                >
                  {rank}
                </span>
                <span className="min-w-0 break-words text-sm text-ink">
                  “{item.text}”
                </span>
              </span>
              <span className="shrink-0 font-mono text-xs tabular-nums text-ink-soft">
                {formatScore(item.score)}
              </span>
            </button>
          </li>
        ))}
      </ul>
      <p className="mt-3 text-[11px] text-ink-mute/80">
        Higher attribution scores indicate a stronger association with the
        model's prediction — an attribution signal, not a human-verified causal
        explanation.
      </p>
    </Card>
  );
}
