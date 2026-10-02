import { MAX_CONTEXT_LENGTH, useAnalysisStore } from "../../store/analysisStore";

export default function ContextInput() {
  const context = useAnalysisStore((state) => state.context);
  const setContext = useAnalysisStore((state) => state.setContext);
  const isAnalyzing = useAnalysisStore((state) => state.isAnalyzing);

  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <label htmlFor="context-input" className="text-sm font-medium text-ink">
          Conversation Context{" "}
          <span className="ml-1 font-normal text-ink-mute">(optional)</span>
        </label>
        <span
          className="font-mono text-[11px] tabular-nums text-ink-mute"
          aria-hidden="true"
        >
          {context.length}/{MAX_CONTEXT_LENGTH}
        </span>
      </div>
      <p id="context-help" className="mt-1 text-xs text-ink-mute">
        Optional previous message or conversational context.
      </p>
      <textarea
        id="context-input"
        className="input-base mt-2"
        rows={3}
        maxLength={MAX_CONTEXT_LENGTH}
        placeholder="Enter the previous message or context..."
        value={context}
        onChange={(event) => setContext(event.target.value)}
        disabled={isAnalyzing}
        aria-describedby="context-help"
      />
    </div>
  );
}
