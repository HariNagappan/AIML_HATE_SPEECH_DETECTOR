import { Loader2 } from "lucide-react";

interface LoadingStateProps {
  contextUsed: boolean;
}

/** Illustrative pipeline stages (the backend does not expose step progress). */
const STAGES = [
  "Encoding the comment",
  "Running context-aware interaction",
  "Generating predictions",
  "Extracting evidence spans",
];

export default function LoadingState({ contextUsed }: LoadingStateProps) {
  return (
    <div
      role="status"
      aria-live="polite"
      className="rounded-xl border border-edge bg-panel p-6"
    >
      <div className="flex items-center gap-3">
        <Loader2 className="h-5 w-5 animate-spin-slow text-accent" aria-hidden="true" />
        <p className="text-sm font-medium text-ink">Analyzing comment…</p>
      </div>
      <p className="mt-1 pl-8 text-xs text-ink-mute">
        {contextUsed
          ? "The model is analyzing the comment together with the provided context."
          : "No context provided — the model is analyzing the comment alone."}
      </p>

      <ul className="mt-5 space-y-2 pl-1" aria-hidden="true">
        {STAGES.map((stage, index) => (
          <li
            key={stage}
            className="flex items-center gap-2.5 text-xs text-ink-mute"
          >
            <span
              className="animate-pulse-dot inline-block h-1.5 w-1.5 rounded-full bg-accent"
              style={{ animationDelay: `${index * 0.35}s` }}
            />
            {stage}
          </li>
        ))}
      </ul>

      <p className="mt-4 text-[11px] text-ink-mute/80">
        This indicator is indeterminate — the stages complete together on the
        server, and no progress percentage is shown.
      </p>
    </div>
  );
}
