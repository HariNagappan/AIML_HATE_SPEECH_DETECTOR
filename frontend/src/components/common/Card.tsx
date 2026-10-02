import type { ReactNode } from "react";

interface CardProps {
  /** Small uppercase kicker shown at the top of the card. */
  title?: string;
  titleIcon?: ReactNode;
  /** Right-aligned content in the header row (e.g. a badge). */
  action?: ReactNode;
  children: ReactNode;
  className?: string;
  /** Slightly muted styling for unavailable / secondary cards. */
  muted?: boolean;
}

export default function Card({
  title,
  titleIcon,
  action,
  children,
  className = "",
  muted = false,
}: CardProps) {
  return (
    <section
      className={[
        "rounded-xl border bg-panel p-5 sm:p-6",
        muted ? "border-edge/60" : "border-edge",
        className,
      ].join(" ")}
    >
      {title || action ? (
        <header className="mb-4 flex items-center justify-between gap-3">
          {title ? (
            <h2 className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.14em] text-ink-mute">
              {titleIcon}
              {title}
            </h2>
          ) : (
            <span />
          )}
          {action}
        </header>
      ) : null}
      {children}
    </section>
  );
}
