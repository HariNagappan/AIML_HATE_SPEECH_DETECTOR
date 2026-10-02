import { useState } from "react";
import { Loader2, RefreshCw, Search } from "lucide-react";
import Card from "../common/Card";
import Badge from "../common/Badge";
import ErrorState from "../common/ErrorState";
import { analysisApi } from "../../api/analysis";
import { toApiError } from "../../api/client";
import { formatLabel, formatScore } from "../../utils/formatting";
import type {
  ApiErrorInfo,
  ExplainResponse,
  PredictRequest,
} from "../../types/analysis";

const SPECIAL_TOKENS = new Set(["[CLS]", "[SEP]", "[PAD]", "[UNK]", "<s>", "</s>", "<pad>"]);

interface AttributionDetailProps {
  request: PredictRequest | null;
  enabled: boolean;
}

/**
 * Token-level attribution view (`POST /api/v1/explain`).
 *
 * Shows the same Integrated-Gradients scores that produced the evidence
 * spans — for every token — so the full model attribution is visible, not
 * just the merged spans. Loaded on demand (attribution costs a forward pass
 * per model side).
 */
export default function AttributionDetail({
  request,
  enabled,
}: AttributionDetailProps) {
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">(
    "idle",
  );
  const [data, setData] = useState<ExplainResponse | null>(null);
  const [error, setError] = useState<ApiErrorInfo | null>(null);

  if (!enabled || !request) {
    return null;
  }

  const load = async () => {
    setStatus("loading");
    setError(null);
    try {
      const response = await analysisApi.explain({
        text: request.text,
        context: request.context ?? null,
        target: "hate",
      });
      setData(response);
      setStatus("ready");
    } catch (caught) {
      setError(toApiError(caught));
      setStatus("error");
    }
  };

  const tokens = (data?.tokens ?? []).filter(
    (token) => !SPECIAL_TOKENS.has(token.token),
  );
  const maxAbs =
    tokens.reduce((acc, token) => Math.max(acc, Math.abs(token.score)), 0) || 1;

  return (
    <Card
      title="Attribution detail"
      titleIcon={<Search className="h-3.5 w-3.5" aria-hidden="true" />}
      action={<Badge tone="info">token-level</Badge>}
    >
      <p className="max-w-3xl text-xs leading-relaxed text-ink-mute">
        Per-token attribution from the explanation endpoint — the same
        Integrated-Gradients scores that produce the evidence spans, shown for
        every token of the comment.
      </p>

      {status === "idle" ? (
        <button type="button" className="btn-secondary mt-3" onClick={() => void load()}>
          <Search className="h-3.5 w-3.5" aria-hidden="true" />
          Load token-level attribution
        </button>
      ) : null}

      {status === "loading" ? (
        <p className="mt-3 flex items-center gap-2 text-sm text-ink-soft">
          <Loader2 className="h-4 w-4 animate-spin-slow text-accent" aria-hidden="true" />
          Computing attributions…
        </p>
      ) : null}

      {status === "error" && error ? (
        <div className="mt-3">
          <ErrorState error={error} onRetry={() => void load()} />
        </div>
      ) : null}

      {status === "ready" && data ? (
        data.available ? (
          <>
            <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-[11px] text-ink-mute">
              <span>
                method <span className="font-mono">{data.method || "—"}</span>
              </span>
              <span>
                steps <span className="font-mono">{data.steps || "—"}</span>
              </span>
              <span>
                explained label{" "}
                <span className="font-mono">
                  {data.explained_label ? formatLabel(data.explained_label) : "—"}
                </span>
              </span>
            </div>

            <div className="mt-4 flex flex-wrap gap-1.5 rounded-lg border border-edge bg-raised/50 p-4">
              {tokens.length > 0 ? (
                tokens.map((token) => {
                  const alpha = 0.08 + 0.55 * (Math.abs(token.score) / maxAbs);
                  return (
                    <span
                      key={token.index}
                      title={`${token.token} · attribution ${formatScore(token.score)}`}
                      style={{ backgroundColor: `rgba(94, 166, 248, ${alpha.toFixed(3)})` }}
                      className="rounded px-1.5 py-0.5 font-mono text-[13px] text-ink"
                    >
                      {token.token}
                    </span>
                  );
                })
              ) : (
                <p className="text-sm text-ink-soft">No tokens returned.</p>
              )}
            </div>

            <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
              <p className="text-[11px] text-ink-mute/80">
                Darker chips = stronger influence on the predicted label
                (attribution magnitude). Hover a token for its exact score.
              </p>
              <button
                type="button"
                className="btn-secondary !px-3 !py-1.5 text-xs"
                onClick={() => void load()}
              >
                <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />
                Refresh
              </button>
            </div>
          </>
        ) : (
          <p className="mt-3 text-sm text-ink-soft">
            {data.message ?? "Token attribution is not available for this model."}
          </p>
        )
      ) : null}
    </Card>
  );
}
