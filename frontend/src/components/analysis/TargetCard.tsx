import { Crosshair, Info } from "lucide-react";
import Card from "../common/Card";
import ConfidenceBar from "../common/ConfidenceBar";
import { formatConfidence, formatLabel } from "../../utils/formatting";
import type { AnalysisResult } from "../../types/analysis";

interface TargetCardProps {
  result: AnalysisResult;
}

export default function TargetCard({ result }: TargetCardProps) {
  const { target, target_available } = result;

  if (!target_available || !target) {
    return (
      <Card title="Target" muted>
        <div className="flex items-start gap-3">
          <Info className="mt-0.5 h-4 w-4 shrink-0 text-ink-mute" aria-hidden="true" />
          <div>
            <p className="text-sm font-medium text-ink-soft">
              Not available for this model
            </p>
            <p className="mt-0.5 text-xs text-ink-mute">
              Target classification requires target-annotated training data.
            </p>
          </div>
        </div>
      </Card>
    );
  }

  return (
    <Card
      title="Target"
      titleIcon={<Crosshair className="h-3.5 w-3.5" aria-hidden="true" />}
    >
      <p className="text-xl font-semibold text-ink">{formatLabel(target.label)}</p>
      <p className="mt-1 font-mono text-xs text-ink-mute">
        {formatConfidence(target.confidence)} confidence
      </p>
      <div className="mt-4">
        <ConfidenceBar
          value={target.confidence}
          tone="accent"
          showValue={false}
          size="sm"
        />
      </div>
    </Card>
  );
}
