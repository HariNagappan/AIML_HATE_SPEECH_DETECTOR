const FILL_CLASSES = {
  neutral: "bg-ink-mute",
  normal: "bg-tone-normal",
  offensive: "bg-tone-offensive",
  hate: "bg-tone-hate",
  accent: "bg-accent",
} as const;

export type ConfidenceTone = keyof typeof FILL_CLASSES;

interface ConfidenceBarProps {
  /** Value in 0..1 (clamped defensively). */
  value: number;
  tone?: ConfidenceTone;
  label?: string;
  showValue?: boolean;
  size?: "sm" | "md";
}

export default function ConfidenceBar({
  value,
  tone = "accent",
  label,
  showValue = true,
  size = "md",
}: ConfidenceBarProps) {
  const clamped = Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0));
  const percent = Math.round(clamped * 100);

  return (
    <div>
      {label || showValue ? (
        <div className="mb-1.5 flex items-baseline justify-between gap-3">
          {label ? <span className="text-xs text-ink-mute">{label}</span> : <span />}
          {showValue ? (
            <span className="font-mono text-sm tabular-nums text-ink-soft">
              {percent}%
            </span>
          ) : null}
        </div>
      ) : null}
      <div
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent}
        aria-label={label ?? "Confidence"}
        className={[
          "w-full overflow-hidden rounded-full bg-raised",
          size === "sm" ? "h-1.5" : "h-2",
        ].join(" ")}
      >
        <div
          className={[
            "h-full rounded-full transition-[width] duration-500 ease-out",
            FILL_CLASSES[tone],
          ].join(" ")}
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}
