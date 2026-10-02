import { AlertTriangle, RefreshCw } from "lucide-react";
import type { ApiErrorInfo } from "../../types/analysis";

interface ErrorStateProps {
  error: ApiErrorInfo;
  onRetry?: () => void;
}

const KIND_TITLES: Record<ApiErrorInfo["kind"], string> = {
  network: "Backend unavailable",
  timeout: "Request timed out",
  validation: "Invalid input",
  model_unavailable: "Model unavailable",
  server: "Server error",
  unknown: "Something went wrong",
};

export default function ErrorState({ error, onRetry }: ErrorStateProps) {
  return (
    <div
      role="alert"
      className="rounded-xl border border-tone-hate/40 bg-tone-hate/5 p-5"
    >
      <div className="flex items-start gap-3">
        <AlertTriangle
          className="mt-0.5 h-5 w-5 shrink-0 text-tone-hate"
          aria-hidden="true"
        />
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold text-ink">{KIND_TITLES[error.kind]}</p>
          <p className="mt-1 text-sm text-ink-soft">{error.message}</p>
          {typeof error.status === "number" ? (
            <p className="mt-1 font-mono text-[11px] text-ink-mute">
              HTTP {error.status}
            </p>
          ) : null}
        </div>
        {onRetry ? (
          <button type="button" onClick={onRetry} className="btn-secondary shrink-0">
            <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />
            Retry
          </button>
        ) : null}
      </div>
    </div>
  );
}
