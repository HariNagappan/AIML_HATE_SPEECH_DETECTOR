import { ArrowUpRight } from "lucide-react";
import Badge from "../common/Badge";
import {
  formatConfidence,
  formatLabel,
  labelTone,
  timeAgo,
} from "../../utils/formatting";
import type { HistoryEntry } from "../../types/analysis";

interface HistoryItemProps {
  entry: HistoryEntry;
  onOpen: (entry: HistoryEntry) => void;
}

export default function HistoryItem({ entry, onOpen }: HistoryItemProps) {
  const { result } = entry;
  const prediction = result.prediction;
  const tone = prediction ? labelTone(prediction.label) : "neutral";

  return (
    <button
      type="button"
      onClick={() => onOpen(entry)}
      className="group w-full rounded-xl border border-edge bg-panel p-4 text-left transition-colors hover:border-edge-strong hover:bg-raised/40 focus-visible:border-edge-strong sm:p-5"
      aria-label={`Open stored analysis from ${timeAgo(entry.timestamp)}`}
    >
      <div className="flex flex-wrap items-center gap-2">
        {prediction ? (
          <Badge tone={tone === "neutral" ? "neutral" : tone}>
            {formatLabel(prediction.label)}
          </Badge>
        ) : (
          <Badge tone="neutral">unavailable</Badge>
        )}
        {prediction ? (
          <span className="font-mono text-xs text-ink-mute">
            {formatConfidence(prediction.confidence)}
          </span>
        ) : null}
        {result.target ? (
          <Badge tone="neutral">{formatLabel(result.target.label)}</Badge>
        ) : null}
        {result.reason ? (
          <Badge tone="neutral">{formatLabel(result.reason.label)}</Badge>
        ) : null}
      </div>

      <p className="mt-3 line-clamp-2 break-words text-sm text-ink-soft">
        “{entry.request.text}”
      </p>

      <div className="mt-2.5 flex items-center justify-between gap-3">
        <span className="text-[11px] text-ink-mute">
          {timeAgo(entry.timestamp)}
        </span>
        <span className="flex items-center gap-1 text-[11px] text-ink-mute opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100">
          Open
          <ArrowUpRight className="h-3 w-3" aria-hidden="true" />
        </span>
      </div>
    </button>
  );
}
