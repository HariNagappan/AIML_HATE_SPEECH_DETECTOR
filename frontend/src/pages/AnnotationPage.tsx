import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, Check, Download, Eraser, Upload } from "lucide-react";
import PageContainer from "../components/layout/PageContainer";
import Card from "../components/common/Card";
import Badge from "../components/common/Badge";
import ErrorState from "../components/common/ErrorState";
import {
  buildAnnotationJsonl,
  labelDistribution,
  labeledCount,
  useAnnotationStore,
} from "../store/annotationStore";
import {
  REASON_CATEGORIES,
  type AnnotationSample,
} from "../types/annotation";
import { formatLabel } from "../utils/formatting";
import type { ApiErrorInfo } from "../types/analysis";

/**
 * Annotation workspace — label comments with reason categories.
 *
 * The exported JSONL (`reason_annotations.jsonl`) is exactly what
 * `backend/scripts/import_reason_annotations.py` consumes; after merging and
 * a training run the reason head exists and the analysis UI displays the
 * Reason card + "Why this classification?" section automatically.
 */
export default function AnnotationPage() {
  const samples = useAnnotationStore((state) => state.samples);
  const labels = useAnnotationStore((state) => state.labels);
  const index = useAnnotationStore((state) => state.index);
  const source = useAnnotationStore((state) => state.source);
  const loadSamples = useAnnotationStore((state) => state.loadSamples);
  const setLabel = useAnnotationStore((state) => state.setLabel);
  const skip = useAnnotationStore((state) => state.skip);
  const back = useAnnotationStore((state) => state.back);
  const gotoFirstUnlabeled = useAnnotationStore((state) => state.gotoFirstUnlabeled);
  const reset = useAnnotationStore((state) => state.reset);

  const [loadError, setLoadError] = useState<ApiErrorInfo | null>(null);
  const [exported, setExported] = useState(false);

  // Load the built-in sample set on first visit (persisted state wins).
  useEffect(() => {
    if (samples.length > 0) {
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const response = await fetch("/annotation-samples.json");
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const data = (await response.json()) as AnnotationSample[];
        const valid =
          Array.isArray(data) &&
          data.length > 0 &&
          data.every(
            (item) =>
              typeof item?.id === "string" && typeof item?.text === "string",
          );
        if (!valid) {
          throw new Error("unexpected sample format");
        }
        if (!cancelled) {
          loadSamples(data, "built-in sample (Counter Context train split)");
        }
      } catch (caught) {
        if (!cancelled) {
          setLoadError({
            kind: "network",
            message: `Could not load annotation samples (${
              caught instanceof Error ? caught.message : "unknown error"
            }). Run the app with "npm run dev" or "npm run preview" so the sample file is served.`,
          });
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [samples.length, loadSamples]);

  // Keyboard: 1-8 label & advance, S skip, ArrowLeft back.
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) {
        return;
      }
      const digit = Number.parseInt(event.key, 10);
      if (!Number.isNaN(digit) && digit >= 1 && digit <= REASON_CATEGORIES.length) {
        event.preventDefault();
        setLabel(REASON_CATEGORIES[digit - 1]);
        return;
      }
      if (event.key === "s" || event.key === " ") {
        event.preventDefault();
        skip();
      }
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        back();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [setLabel, skip, back]);

  const total = samples.length;
  const done = labeledCount(samples, labels);
  const distribution = useMemo(
    () => labelDistribution(samples, labels),
    [samples, labels],
  );
  const current = index < total ? samples[index] : null;
  const currentLabel = current ? labels[current.id] : undefined;
  const percent = total > 0 ? Math.round((done / total) * 100) : 0;

  const handleExport = () => {
    const jsonl = buildAnnotationJsonl(samples, labels);
    if (!jsonl) {
      return;
    }
    const blob = new Blob([jsonl + "\n"], { type: "application/x-ndjson" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "reason_annotations.jsonl";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
    setExported(true);
  };

  const handleImport = async (file: File) => {
    try {
      const text = await file.text();
      const items: AnnotationSample[] = [];
      text.split(/\r?\n/).forEach((line, lineIndex) => {
        const trimmed = line.trim();
        if (!trimmed) {
          return;
        }
        const parsed = JSON.parse(trimmed) as Record<string, unknown>;
        const itemText =
          typeof parsed.text === "string" ? parsed.text.trim() : "";
        if (!itemText) {
          return;
        }
        items.push({
          id:
            typeof parsed.id === "string" && parsed.id
              ? parsed.id
              : `import-${lineIndex + 1}`,
          text: itemText,
          context:
            typeof parsed.context === "string" ? parsed.context : undefined,
        });
      });
      if (items.length === 0) {
        throw new Error("no usable lines (need JSONL with a 'text' field)");
      }
      const replace = window.confirm(
        `Replace the current set with ${items.length} imported comments? Labels will be reset.`,
      );
      if (replace) {
        loadSamples(items, file.name);
        setExported(false);
      }
    } catch (caught) {
      setLoadError({
        kind: "validation",
        message: `Could not import file: ${
          caught instanceof Error ? caught.message : "unknown error"
        }`,
      });
    }
  };

  const handleReset = () => {
    if (window.confirm("Clear all labels for this sample set?")) {
      reset();
      setExported(false);
    }
  };

  return (
    <PageContainer>
      {/* intro */}
      <div className="max-w-3xl">
        <p className="kicker">data curation · reason labels</p>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight text-ink sm:text-3xl">
          Annotation Workspace
        </h1>
        <p className="mt-2 text-sm leading-relaxed text-ink-soft">
          Label comments with the reason category that applies. The reason head
          of the model is trained only on real annotations — this workspace
          creates them; the export feeds directly into the training pipeline.
        </p>
      </div>

      {loadError ? (
        <div className="mt-6">
          <ErrorState error={loadError} />
        </div>
      ) : null}

      {/* progress + actions */}
      <div className="mt-6 rounded-xl border border-edge bg-panel p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-sm font-medium text-ink">
              {done} / {total} labeled{" "}
              <span className="text-ink-mute">({percent}%)</span>
            </p>
            {source ? (
              <p className="mt-0.5 text-[11px] text-ink-mute">Source: {source}</p>
            ) : null}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              className="btn-primary !px-3.5 !py-2 text-xs"
              onClick={handleExport}
              disabled={done === 0}
            >
              <Download className="h-3.5 w-3.5" aria-hidden="true" />
              Export JSONL
            </button>
            <label className="btn-secondary cursor-pointer !px-3.5 !py-2 text-xs">
              <Upload className="h-3.5 w-3.5" aria-hidden="true" />
              Import file
              <input
                type="file"
                accept=".jsonl,.json,.txt"
                className="hidden"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) {
                    void handleImport(file);
                  }
                  event.target.value = "";
                }}
              />
            </label>
            <button
              type="button"
              className="btn-secondary !px-3.5 !py-2 text-xs"
              onClick={handleReset}
              disabled={done === 0}
            >
              <Eraser className="h-3.5 w-3.5" aria-hidden="true" />
              Reset labels
            </button>
          </div>
        </div>

        <div className="mt-3 h-1.5 overflow-hidden rounded bg-raised">
          <div
            className="h-full rounded bg-accent transition-[width]"
            style={{ width: `${percent}%` }}
          />
        </div>

        {done > 0 ? (
          <div className="mt-3 flex flex-wrap gap-1.5">
            {REASON_CATEGORIES.filter((category) => distribution[category]).map(
              (category) => (
                <Badge key={category} tone="neutral">
                  {formatLabel(category)} · {distribution[category]}
                </Badge>
              ),
            )}
          </div>
        ) : null}

        {exported ? (
          <p className="mt-3 rounded-md border border-edge bg-raised/60 px-3 py-2 text-xs text-ink-soft">
            Downloaded <span className="font-mono">reason_annotations.jsonl</span> —
            move it to <span className="font-mono">backend/data/</span> and run the
            merge + training commands shown below.
          </p>
        ) : null}
      </div>

      {/* current sample */}
      {current ? (
        <div className="mt-6 rounded-xl border border-edge bg-panel p-5">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <p className="kicker">
              sample {index + 1} of {total}
            </p>
            <p className="font-mono text-[11px] text-ink-mute">{current.id}</p>
          </div>
          <p className="mt-3 break-words text-lg leading-relaxed text-ink">
            {current.text}
          </p>
          {current.context ? (
            <div className="mt-3 rounded-md border-l-2 border-edge-strong bg-raised/40 px-3 py-2">
              <p className="text-[11px] uppercase tracking-wide text-ink-mute">
                context
              </p>
              <p className="mt-0.5 break-words text-sm text-ink-soft">
                {current.context}
              </p>
            </div>
          ) : null}

          <div className="mt-5 grid grid-cols-2 gap-2 sm:grid-cols-4">
            {REASON_CATEGORIES.map((category, categoryIndex) => {
              const selected = currentLabel === category;
              return (
                <button
                  key={category}
                  type="button"
                  onClick={() => setLabel(category)}
                  className={[
                    "flex items-center justify-between gap-2 rounded-lg border px-3 py-2.5 text-left text-[13px] transition-colors",
                    selected
                      ? "border-accent bg-accent/15 text-ink"
                      : "border-edge bg-panel text-ink-soft hover:bg-raised hover:text-ink",
                  ].join(" ")}
                >
                  <span>{formatLabel(category)}</span>
                  <kbd className="font-mono text-[10px] text-ink-mute">
                    {categoryIndex + 1}
                  </kbd>
                </button>
              );
            })}
          </div>

          <div className="mt-4 flex flex-wrap items-center justify-between gap-2">
            <p className="text-[11px] text-ink-mute">
              Keys 1–8 label and advance · S skip · ← back
            </p>
            <div className="flex gap-2">
              <button
                type="button"
                className="btn-secondary !px-3 !py-1.5 text-xs"
                onClick={back}
                disabled={index === 0}
              >
                <ArrowLeft className="h-3.5 w-3.5" aria-hidden="true" />
                Back
              </button>
              {currentLabel ? (
                <button
                  type="button"
                  className="btn-secondary !px-3 !py-1.5 text-xs"
                  onClick={skip}
                >
                  <Check className="h-3.5 w-3.5" aria-hidden="true" />
                  Next
                </button>
              ) : (
                <button
                  type="button"
                  className="btn-secondary !px-3 !py-1.5 text-xs"
                  onClick={skip}
                >
                  Skip
                </button>
              )}
            </div>
          </div>
        </div>
      ) : total > 0 ? (
        <div className="mt-6 rounded-xl border border-edge bg-panel p-6 text-center">
          <p className="text-sm font-medium text-ink">
            {done === total
              ? `All ${total} samples labeled.`
              : `Reached the end — ${total - done} unlabeled remain.`}
          </p>
          <p className="mt-1 text-xs text-ink-mute">
            Export when ready, or continue with the remaining items.
          </p>
          <div className="mt-4 flex justify-center gap-2">
            {done < total ? (
              <button type="button" className="btn-secondary text-xs" onClick={gotoFirstUnlabeled}>
                Continue labeling
              </button>
            ) : null}
            <button
              type="button"
              className="btn-primary !px-3.5 !py-2 text-xs"
              onClick={handleExport}
              disabled={done === 0}
            >
              <Download className="h-3.5 w-3.5" aria-hidden="true" />
              Export JSONL
            </button>
          </div>
        </div>
      ) : null}

      {/* how to use */}
      <div className="mt-6">
        <Card title="How to use these labels">
          <ol className="max-w-3xl list-decimal space-y-1.5 pl-5 text-sm text-ink-soft">
            <li>
              Export the JSONL above and move it to{" "}
              <span className="font-mono text-ink">backend/data/reason_annotations.jsonl</span>.
            </li>
            <li>
              Merge it into the processed splits (matches by{" "}
              <span className="font-mono">id</span>):
            </li>
          </ol>
          <pre className="mt-3">{`python scripts/import_reason_annotations.py \\
    --annotations data/reason_annotations.jsonl \\
    --merge-into data/processed/counter_context_train.jsonl \\
    --output data/processed/counter_context_reason_train.jsonl
python scripts/train.py --config cc_reason
# then serve: point MODEL_CHECKPOINT (or checkpoints/default.json)
# at checkpoints/cc_reason/best.pt`}</pre>
          <p className="mt-2 text-xs text-ink-mute">
            Labels are stored in this browser only (localStorage) until you
            export. Around 100+ labeled comments make a workable demo; more is
            better. Reason head output appears in the Analyze page once a
            reason-trained checkpoint is served.
          </p>
        </Card>
      </div>
    </PageContainer>
  );
}
