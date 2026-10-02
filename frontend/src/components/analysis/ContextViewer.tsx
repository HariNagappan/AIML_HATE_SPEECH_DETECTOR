import { Check, MessagesSquare, Minus } from "lucide-react";
import Card from "../common/Card";
import Badge from "../common/Badge";
import type { PredictRequest } from "../../types/analysis";

interface ContextViewerProps {
  contextUsed: boolean;
  request: PredictRequest | null;
}

export default function ContextViewer({ contextUsed, request }: ContextViewerProps) {
  return (
    <Card
      title="Context"
      titleIcon={<MessagesSquare className="h-3.5 w-3.5" aria-hidden="true" />}
      action={
        contextUsed ? (
          <Badge tone="info" icon={<Check className="h-3 w-3" aria-hidden="true" />}>
            Context used
          </Badge>
        ) : (
          <Badge
            tone="neutral"
            icon={<Minus className="h-3 w-3" aria-hidden="true" />}
          >
            Not provided
          </Badge>
        )
      }
    >
      <p className="text-sm text-ink-soft">
        {contextUsed
          ? "Context was included in this analysis."
          : "No conversational context was provided for this analysis."}
      </p>

      {request ? (
        <div className="mt-4 space-y-3">
          {request.context ? (
            <div className="rounded-lg border border-edge bg-raised/40 p-3.5">
              <p className="kicker">Previous context</p>
              <p className="mt-1.5 break-words text-sm text-ink-soft">
                “{request.context}”
              </p>
            </div>
          ) : null}
          <div className="rounded-lg border border-edge bg-raised/40 p-3.5">
            <p className="kicker">Current comment</p>
            <p className="mt-1.5 break-words text-sm text-ink-soft">“{request.text}”</p>
          </div>
        </div>
      ) : null}
    </Card>
  );
}
