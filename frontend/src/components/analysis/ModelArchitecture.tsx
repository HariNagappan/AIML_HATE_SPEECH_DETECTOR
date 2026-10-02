import { ChevronDown, Workflow } from "lucide-react";

const ARCHITECTURE_DIAGRAM = `Current Comment ──┐
                  ├──► Shared BERT encoder
Context ──────────┘
                       │
                       ▼
              Context interaction   D = |E_c − E_p|,  M = E_c ⊙ E_p
                       │
                       ▼
             Contrastive representation   (training only)
                       │
                       ▼
              Final representation
          ┌────────────┼────────────┐
          ▼            ▼            ▼
        Hate         Target       Reason
          │            │            │
          └────────────┼────────────┘
                       ▼
              Evidence extraction   (attribution: Integrated Gradients)`;

const DESCRIPTIONS = [
  {
    term: "BERT",
    detail: "Provides contextual language representations.",
  },
  {
    term: "Context Interaction",
    detail: "Combines the current comment with conversational context.",
  },
  {
    term: "Contrastive Learning",
    detail:
      "Encourages the model to distinguish relevant context from contrasting context.",
  },
  {
    term: "Classification Heads",
    detail: "Predict hate label, target, and reason.",
  },
  {
    term: "Evidence Extraction",
    detail: "Identifies text spans associated with the model prediction.",
  },
];

/** Collapsible architecture explainer (native <details> for accessibility). */
export default function ModelArchitecture() {
  return (
    <details className="group rounded-xl border border-edge bg-panel">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 p-5 [&::-webkit-details-marker]:hidden sm:p-6">
        <span className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.14em] text-ink-mute">
          <Workflow className="h-3.5 w-3.5" aria-hidden="true" />
          How the model works
        </span>
        <ChevronDown
          className="h-4 w-4 shrink-0 text-ink-mute transition-transform group-open:rotate-180"
          aria-hidden="true"
        />
      </summary>

      <div className="border-t border-edge px-5 pb-6 pt-5 sm:px-6">
        <pre className="overflow-x-auto rounded-lg border border-edge bg-raised/40 p-4 font-mono text-[11.5px] leading-6 text-ink-soft">
          {ARCHITECTURE_DIAGRAM}
        </pre>

        <dl className="mt-5 grid gap-x-8 gap-y-4 sm:grid-cols-2">
          {DESCRIPTIONS.map((entry) => (
            <div key={entry.term}>
              <dt className="font-mono text-xs font-semibold uppercase tracking-[0.08em] text-ink">
                {entry.term}
              </dt>
              <dd className="mt-1 text-sm text-ink-soft">{entry.detail}</dd>
            </div>
          ))}
        </dl>

        <p className="mt-5 text-[11px] text-ink-mute/80">
          The encoder is a pretrained BERT model that is fine-tuned for this
          task — it is not trained from scratch. Evidence is attribution-based
          and does not involve generated text.
        </p>
      </div>
    </details>
  );
}
