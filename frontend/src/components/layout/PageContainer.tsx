import type { ReactNode } from "react";

interface PageContainerProps {
  children: ReactNode;
  className?: string;
}

export default function PageContainer({
  children,
  className = "",
}: PageContainerProps) {
  return (
    <div
      className={[
        "mx-auto w-full max-w-6xl px-4 py-8 sm:px-6 sm:py-10",
        className,
      ].join(" ")}
    >
      {children}
    </div>
  );
}
