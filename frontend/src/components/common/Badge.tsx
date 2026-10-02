import type { ReactNode } from "react";

export type BadgeTone =
  | "neutral"
  | "info"
  | "normal"
  | "offensive"
  | "hate"
  | "success"
  | "warning"
  | "danger";

const TONE_CLASSES: Record<BadgeTone, string> = {
  neutral: "border-edge text-ink-soft bg-raised/60",
  info: "border-accent/40 text-accent bg-accent/10",
  normal: "border-tone-normal/40 text-tone-normal bg-tone-normal/10",
  success: "border-tone-normal/40 text-tone-normal bg-tone-normal/10",
  offensive: "border-tone-offensive/40 text-tone-offensive bg-tone-offensive/10",
  warning: "border-tone-offensive/40 text-tone-offensive bg-tone-offensive/10",
  hate: "border-tone-hate/45 text-tone-hate bg-tone-hate/10",
  danger: "border-tone-hate/45 text-tone-hate bg-tone-hate/10",
};

interface BadgeProps {
  tone?: BadgeTone;
  icon?: ReactNode;
  children: ReactNode;
  className?: string;
}

/** Small labeled chip. Colour is always paired with text (never colour alone). */
export default function Badge({
  tone = "neutral",
  icon,
  children,
  className = "",
}: BadgeProps) {
  return (
    <span
      className={[
        "inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 font-mono text-[11px] uppercase tracking-[0.08em]",
        TONE_CLASSES[tone],
        className,
      ].join(" ")}
    >
      {icon}
      {children}
    </span>
  );
}
