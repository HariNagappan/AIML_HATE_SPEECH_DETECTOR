import { AlertTriangle, CheckCircle2, Gauge, Info, ShieldAlert } from "lucide-react";
import Card from "../common/Card";
import ConfidenceBar from "../common/ConfidenceBar";
import {
  formatConfidence,
  formatLabel,
  labelTone,
  type SemanticTone,
} from "../../utils/formatting";
import type { AnalysisResult } from "../../types/analysis";

const TONE_TEXT: Record<SemanticTone, string> = {
  hate: "text-tone-hate",
  offensive: "text-tone-offensive",
  normal: "text-tone-normal",
  neutral: "text-ink-soft",
};

const TONE_ICONS = {
  hate: ShieldAlert,
  offensive: AlertTriangle,
  normal: CheckCircle2,
  neutral: Info,
} as const;

interface PredictionCardProps {
  result: AnalysisResult;
}

export default function PredictionCard({ result }: PredictionCardProps) {
  const { prediction, prediction_available } = result;

  if (!prediction_available || !prediction) {
    return (
      <Card title="Prediction" muted>
        <div className="flex items-start gap-3">
          <Info className="mt-0.5 h-4 w-4 shrink-0 text-ink-mute" aria-hidden="true" />
          <div>
            <p className="text-sm font-medium text-ink-soft">
              Not available for this model
            </p>
            <p className="mt-0.5 text-xs text-ink-mute">
              The hate-classification head has no trained checkpoint loaded.
            </p>
          </div>
        </div>
      </Card>
    );
  }

  const tone = labelTone(prediction.label);
  const Icon = TONE_ICONS[tone];
  const probabilities = prediction.probabilities
    ? Object.entries(prediction.probabilities).sort(([, a], [, b]) => b - a)
    : [];

  return (
    <Card
      title="Prediction"
      titleIcon={<Gauge className="h-3.5 w-3.5" aria-hidden="true" />}
    >
      <p
        className={[
          "flex items-center gap-3 text-3xl font-semibold tracking-tight sm:text-4xl",
          TONE_TEXT[tone],
        ].join(" ")}
      >
        <Icon className="h-7 w-7 shrink-0" aria-hidden="true" />
        {formatLabel(prediction.label)}
      </p>
      <p className="mt-2 font-mono text-xs text-ink-mute">
        {formatConfidence(prediction.confidence)} confidence
      </p>
      <div className="mt-5 max-w-md">
        <ConfidenceBar
          value={prediction.confidence}
          tone={tone === "neutral" ? "neutral" : tone}
          label="Confidence"
        />
      </div>

      {probabilities.length > 0 ? (
        <div className="mt-5 max-w-md border-t border-edge pt-4">
          <p className="kicker">All classes</p>
          <div className="mt-3 space-y-3">
            {probabilities.map(([label, value]) => (
              <ConfidenceBar
                key={label}
                value={value}
                tone={labelTone(label)}
                size="sm"
                label={formatLabel(label)}
              />
            ))}
          </div>
        </div>
      ) : null}
    </Card>
  );
}
