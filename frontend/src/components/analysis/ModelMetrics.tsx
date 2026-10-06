import { useEffect, useState } from "react";
import { BarChart3, Server } from "lucide-react";
import Card from "../common/Card";
import Badge from "../common/Badge";
import { analysisApi } from "../../api/analysis";
import { toApiError } from "../../api/client";
import { formatLabel, formatPercent } from "../../utils/formatting";
import type { ApiErrorInfo, ModelInfo } from "../../types/analysis";

interface MetricRowProps {
  label: string;
  value: number | null | undefined;
}

function MetricRow({ label, value }: MetricRowProps) {
  return (
    <div className="flex items-center justify-between gap-x-4 border-b border-edge/60 py-2 last:border-b-0">
      <span className="text-xs text-ink-mute">{label}</span>
      <span className="font-mono text-sm font-medium text-ink">
        {value == null ? "—" : formatPercent(value)}
      </span>
    </div>
  );
}

/**
 * Evaluation scores of the served checkpoint (accuracy, macro
 * precision/recall/F1), sourced from /api/v1/model/info. The values come from
 * an offline test-split evaluation (scripts/evaluate.py) — never hardcoded
 * and never derived from live predictions. When no evaluation exists the
 * card says so explicitly instead of inventing numbers.
 */
export default function ModelMetrics() {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [error, setError] = useState<ApiErrorInfo | null>(null);

  useEffect(() => {
    const controller = new AbortController();

    (async () => {
      try {
        const info = await analysisApi.modelInfo(controller.signal);
        setModelInfo(info);
        setError(null);
      } catch (caught) {
        if (!controller.signal.aborted) {
          setError(toApiError(caught));
        }
      }
    })();

    return () => controller.abort();
  }, []);

  const metrics = modelInfo?.metrics ?? null;
  const datasetLabel = metrics?.dataset
    ? `${formatLabel(metrics.dataset)} test split`
    : "held-out test split";

  return (
    <Card
      title="Evaluation metrics"
      titleIcon={<BarChart3 className="h-3.5 w-3.5" aria-hidden="true" />}
      action={
        error ? (
          <Badge tone="danger">offline</Badge>
        ) : metrics ? (
          <Badge tone="info">test split</Badge>
        ) : modelInfo ? (
          <Badge tone="neutral">not evaluated</Badge>
        ) : (
          <Badge tone="neutral">checking…</Badge>
        )
      }
    >
      {error ? (
        <div className="flex items-start gap-2.5">
          <Server className="mt-0.5 h-4 w-4 shrink-0 text-ink-mute" aria-hidden="true" />
          <p className="text-sm text-ink-soft">
            Backend not reachable — start it with{" "}
            <code className="rounded border border-edge bg-raised px-1.5 py-0.5 font-mono text-[11px]">
              uvicorn app.main:app --reload
            </code>
          </p>
        </div>
      ) : metrics ? (
        <div>
          <MetricRow label="Accuracy" value={metrics.accuracy} />
          <MetricRow label="Precision (macro)" value={metrics.macro_precision} />
          <MetricRow label="Recall (macro)" value={metrics.macro_recall} />
          <MetricRow label="F1 (macro)" value={metrics.macro_f1} />
          <p className="mt-3 text-[11px] leading-relaxed text-ink-mute/80">
            Macro-averaged scores from an offline evaluation of the served
            checkpoint on the {datasetLabel}
            {typeof metrics.num_examples === "number"
              ? ` (${metrics.num_examples} examples)`
              : ""}
            . These numbers come from evaluation runs — never from live
            predictions.
          </p>
        </div>
      ) : modelInfo ? (
        <p className="text-sm text-ink-soft">
          No evaluation metrics are available for the served checkpoint yet.
          Run{" "}
          <code className="rounded border border-edge bg-raised px-1.5 py-0.5 font-mono text-[11px]">
            python scripts/evaluate.py
          </code>{" "}
          to generate them.
        </p>
      ) : (
        <p className="text-sm text-ink-mute">Checking evaluation metrics.</p>
      )}
    </Card>
  );
}
