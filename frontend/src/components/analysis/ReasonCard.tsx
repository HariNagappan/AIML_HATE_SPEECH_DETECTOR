import { Info, Lightbulb } from "lucide-react";
import Card from "../common/Card";
import ConfidenceBar from "../common/ConfidenceBar";
import { formatConfidence, formatLabel } from "../../utils/formatting";
import type { AnalysisResult } from "../../types/analysis";

interface ReasonCardProps {
  result: AnalysisResult;
}

export default function ReasonCard({ result }: ReasonCardProps) {
  const { reason, reason_available } = result;

  if (!reason_available || !reason) {
    // Untrained / unavailable reason head — never fabricate a reason.
    return (
      <Card title="Reason" muted>
        <div className="flex items-start gap-3">
          <Info className="mt-0.5 h-4 w-4 shrink-0 text-ink-mute" aria-hidden="true" />
          <div>
            <p className="text-sm font-medium text-ink-soft">
              Not available for this model
            </p>
            <p className="mt-0.5 text-xs text-ink-mute">
              Reason classification needs a reason-annotated model — label data
              in the Annotate tab to enable it.
            </p>
          </div>
        </div>
      </Card>
    );
  }

  return (
    <Card
      title="Reason"
      titleIcon={<Lightbulb className="h-3.5 w-3.5" aria-hidden="true" />}
    >
      <p className="text-xl font-semibold text-ink">{formatLabel(reason.label)}</p>
      <p className="mt-1 font-mono text-xs text-ink-mute">
        {formatConfidence(reason.confidence)} confidence
      </p>
      <div className="mt-4">
        <ConfidenceBar
          value={reason.confidence}
          tone="accent"
          showValue={false}
          size="sm"
        />
      </div>
    </Card>
  );
}
