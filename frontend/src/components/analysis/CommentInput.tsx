import { MAX_COMMENT_LENGTH, useAnalysisStore } from "../../store/analysisStore";

interface CommentInputProps {
  /** Called when Ctrl/⌘ + Enter is pressed inside the textarea. */
  onRequestSubmit: () => void;
}

export default function CommentInput({ onRequestSubmit }: CommentInputProps) {
  const currentComment = useAnalysisStore((state) => state.currentComment);
  const setCurrentComment = useAnalysisStore((state) => state.setCurrentComment);
  const isAnalyzing = useAnalysisStore((state) => state.isAnalyzing);
  const showValidation = useAnalysisStore((state) => state.showValidation);

  const isEmpty = currentComment.trim().length === 0;
  const invalid = showValidation && isEmpty;

  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <label htmlFor="comment-input" className="text-sm font-medium text-ink">
          Current Comment{" "}
          <span className="ml-1 font-normal text-tone-hate" aria-hidden="true">
            *
          </span>
        </label>
        <span
          className="font-mono text-[11px] tabular-nums text-ink-mute"
          aria-hidden="true"
        >
          {currentComment.length}/{MAX_COMMENT_LENGTH}
        </span>
      </div>
      <p id="comment-help" className="mt-1 text-xs text-ink-mute">
        The message you want the model to analyze.
      </p>
      <textarea
        id="comment-input"
        className="input-base mt-2"
        rows={5}
        maxLength={MAX_COMMENT_LENGTH}
        placeholder="Enter the comment you want to analyze..."
        value={currentComment}
        onChange={(event) => setCurrentComment(event.target.value)}
        onKeyDown={(event) => {
          if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
            event.preventDefault();
            onRequestSubmit();
          }
        }}
        disabled={isAnalyzing}
        required
        aria-required="true"
        aria-invalid={invalid ? true : undefined}
        aria-describedby={invalid ? "comment-help comment-error" : "comment-help"}
      />
      {invalid ? (
        <p id="comment-error" role="alert" className="mt-1.5 text-xs text-tone-hate">
          Please enter a comment to analyze.
        </p>
      ) : null}
      <p className="mt-1.5 text-[11px] text-ink-mute/80">
        Tip: press Ctrl/⌘ + Enter to analyze.
      </p>
    </div>
  );
}
