import { useEffect, useState } from "react";
import { Activity, Server } from "lucide-react";
import Card from "../common/Card";
import Badge from "../common/Badge";
import { analysisApi } from "../../api/analysis";
import { toApiError } from "../../api/client";
import type { ApiErrorInfo, ModelInfo } from "../../types/analysis";

interface StatusRowProps {
  label: string;
  children: React.ReactNode;
}

function StatusRow({ label, children }: StatusRowProps) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 border-b border-edge/60 py-2 last:border-b-0">
      <span className="text-xs text-ink-mute">{label}</span>
      <span className="text-xs text-ink-soft">{children}</span>
    </div>
  );
}

function AvailabilityBadge({ available }: { available: boolean }) {
  return (
    <Badge tone={available ? "success" : "neutral"}>
      {available ? "available" : "unavailable"}
    </Badge>
  );
}

/**
 * Small technical status area for the currently served model
 * (sourced from /health and /api/v1/model/info — never hardcoded).
 */
export default function ModelStatus() {
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

  const trainedHeads = new Set(modelInfo?.trained_heads ?? []);
  const reasonAvailable = trainedHeads.has("reason");
  const evidenceAvailable = trainedHeads.has("hate"); // attribution runs on the hate head
  const contextEnabled = modelInfo?.architecture !== "baseline";

  return (
    <Card
      title="Model status"
      titleIcon={<Activity className="h-3.5 w-3.5" aria-hidden="true" />}
      action={
        error ? (
          <Badge tone="danger">offline</Badge>
        ) : modelInfo ? (
          <Badge tone="info">connected</Badge>
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
      ) : modelInfo ? (
        <div>
          <StatusRow label="Model">
            {modelInfo.model_name ?? "—"}
          </StatusRow>
          <StatusRow label="Device">
            <span className="font-mono">{modelInfo.device ?? "—"}</span>
          </StatusRow>
          <StatusRow label="Trained">
            <Badge tone={modelInfo.trained ? "success" : "neutral"}>
              {modelInfo.trained ? "trained checkpoint" : "untrained"}
            </Badge>
          </StatusRow>
          <StatusRow label="Context interaction">
            <Badge tone={contextEnabled ? "info" : "neutral"}>
              {contextEnabled ? "enabled" : "disabled"}
            </Badge>
          </StatusRow>
          <StatusRow label="Reason classification">
            <AvailabilityBadge available={reasonAvailable} />
          </StatusRow>
          <StatusRow label="Evidence extraction">
            <AvailabilityBadge available={evidenceAvailable} />
          </StatusRow>
        </div>
      ) : (
        <p className="text-sm text-ink-mute">Checking backend status…</p>
      )}
    </Card>
  );
}
