import { ShieldCheck } from "lucide-react";
import { NavLink } from "react-router-dom";

const NAV_ITEMS = [
  { to: "/", label: "Analyze", end: true },
  { to: "/history", label: "History", end: false },
  { to: "/about", label: "About", end: false },
];

export default function Header() {
  return (
    <header className="sticky top-0 z-40 border-b border-edge bg-base/90 backdrop-blur-sm">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-x-4 gap-y-2 px-4 py-3 sm:px-6">
        <NavLink
          to="/"
          className="flex items-center gap-2.5"
          aria-label="Context-Aware Hate Speech Analyzer — home"
        >
          <span className="flex h-8 w-8 items-center justify-center rounded-lg border border-edge bg-raised">
            <ShieldCheck className="h-4 w-4 text-accent" aria-hidden="true" />
          </span>
          <span className="flex flex-col leading-tight">
            <span className="font-mono text-[13px] font-semibold tracking-tight text-ink">
              Context-Aware Analyzer
            </span>
            <span className="text-[10.5px] text-ink-mute">
              structured, evidence-based hate speech analysis
            </span>
          </span>
        </NavLink>

        <nav aria-label="Primary" className="flex items-center gap-1">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                [
                  "rounded-md px-3 py-1.5 text-sm transition-colors",
                  isActive
                    ? "bg-raised text-ink"
                    : "text-ink-soft hover:bg-raised/60 hover:text-ink",
                ].join(" ")
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  );
}
