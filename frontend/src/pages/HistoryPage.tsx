import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Clock, Trash2 } from "lucide-react";
import PageContainer from "../components/layout/PageContainer";
import HistoryItem from "../components/history/HistoryItem";
import { useAnalysisStore } from "../store/analysisStore";
import type { HistoryEntry } from "../types/analysis";

export default function HistoryPage() {
  const history = useAnalysisStore((state) => state.history);
  const clearHistory = useAnalysisStore((state) => state.clearHistory);
  const loadFromHistory = useAnalysisStore((state) => state.loadFromHistory);
  const navigate = useNavigate();
  const [confirmingClear, setConfirmingClear] = useState(false);

  const openEntry = (entry: HistoryEntry) => {
    loadFromHistory(entry);
    navigate("/");
  };

  return (
    <PageContainer>
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="kicker">local history</p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight text-ink">
            Recent Analyses
          </h1>
          <p className="mt-1 text-sm text-ink-mute">
            Stored in your browser (localStorage) — {history.length}{" "}
            {history.length === 1 ? "item" : "items"}.
          </p>
        </div>

        {history.length > 0 ? (
          confirmingClear ? (
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs text-ink-mute">
                Delete all {history.length} items?
              </span>
              <button
                type="button"
                className="btn-secondary !border-tone-hate/50 !text-tone-hate"
                onClick={() => {
                  clearHistory();
                  setConfirmingClear(false);
                }}
              >
                <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
                Yes, clear
              </button>
              <button
                type="button"
                className="btn-secondary"
                onClick={() => setConfirmingClear(false)}
              >
                Cancel
              </button>
            </div>
          ) : (
            <button
              type="button"
              className="btn-secondary"
              onClick={() => setConfirmingClear(true)}
            >
              <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
              Clear history
            </button>
          )
        ) : null}
      </div>

      {history.length === 0 ? (
        <div className="mt-10 rounded-xl border border-dashed border-edge bg-panel/50 p-8 text-center">
          <Clock className="mx-auto h-5 w-5 text-ink-mute" aria-hidden="true" />
          <p className="mt-3 text-sm text-ink-soft">No analyses yet.</p>
          <p className="mt-1 text-xs text-ink-mute">
            Run one from the{" "}
            <Link to="/" className="text-accent underline-offset-2 hover:underline">
              Analyze page
            </Link>{" "}
            and it will appear here.
          </p>
        </div>
      ) : (
        <ul className="mt-6 space-y-3">
          {history.map((entry) => (
            <li key={entry.id}>
              <HistoryItem entry={entry} onOpen={openEntry} />
            </li>
          ))}
        </ul>
      )}
    </PageContainer>
  );
}
