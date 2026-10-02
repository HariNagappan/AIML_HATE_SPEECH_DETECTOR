import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { Eraser, FlaskConical } from "lucide-react";
import PageContainer from "../components/layout/PageContainer";
import Badge from "../components/common/Badge";
import AnalyzeButton from "../components/analysis/AnalyzeButton";
import AttributionDetail from "../components/analysis/AttributionDetail";
import CommentInput from "../components/analysis/CommentInput";
import ContextInput from "../components/analysis/ContextInput";
import ContextViewer from "../components/analysis/ContextViewer";
import EvidenceList from "../components/analysis/EvidenceList";
import EvidenceViewer from "../components/analysis/EvidenceViewer";
import ModelArchitecture from "../components/analysis/ModelArchitecture";
import ModelStatus from "../components/analysis/ModelStatus";
import PredictionCard from "../components/analysis/PredictionCard";
import ReasonCard from "../components/analysis/ReasonCard";
import ReasonExplanationCard from "../components/analysis/ReasonExplanationCard";
import ReasoningCard from "../components/analysis/ReasoningCard";
import TargetCard from "../components/analysis/TargetCard";
import ErrorState from "../components/common/ErrorState";
import LoadingState from "../components/common/LoadingState";
import { useAnalysisStore } from "../store/analysisStore";
import { demoExamples } from "../utils/demoExamples";
import { rankEvidence } from "../utils/evidence";
import type { ReasoningLink } from "../types/analysis";

export default function AnalyzePage() {
  const context = useAnalysisStore((state) => state.context);
  const isAnalyzing = useAnalysisStore((state) => state.isAnalyzing);
  const result = useAnalysisStore((state) => state.result);
  const lastRequest = useAnalysisStore((state) => state.lastRequest);
  const error = useAnalysisStore((state) => state.error);
  const analyze = useAnalysisStore((state) => state.analyze);
  const clearForm = useAnalysisStore((state) => state.clearForm);
  const setContext = useAnalysisStore((state) => state.setContext);
  const setCurrentComment = useAnalysisStore((state) => state.setCurrentComment);

  const [activeEvidence, setActiveEvidence] = useState<number | null>(null);
  const [activeLink, setActiveLink] = useState<{
    link: ReasoningLink;
    index: number;
  } | null>(null);
  const explanationRef = useRef<HTMLDivElement | null>(null);
  const evidenceRef = useRef<HTMLDivElement | null>(null);

  // New result ⇒ reset the hovered/selected evidence span and active link.
  useEffect(() => {
    setActiveEvidence(null);
    setActiveLink(null);
  }, [result]);

  // Activating a context link scrolls the evidence section into view.
  useEffect(() => {
    if (activeLink) {
      evidenceRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [activeLink]);

  const { rankByIndex, rankByText } = useMemo(() => {
    const indexMap = new Map<number, number>();
    const textMap: Record<string, number> = {};
    if (result) {
      for (const entry of rankEvidence(result.evidence)) {
        indexMap.set(entry.index, entry.rank);
        if (!(entry.item.text in textMap)) {
          textMap[entry.item.text] = entry.rank;
        }
      }
    }
    return { rankByIndex: indexMap, rankByText: textMap };
  }, [result]);

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    void analyze();
  };

  const handleEvidenceSelect = (index: number | null) => {
    setActiveEvidence(index);
    if (index !== null && result?.reason_explanation) {
      explanationRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "nearest",
      });
    }
  };

  const selectEvidenceByText = (text: string) => {
    if (!result) {
      return;
    }
    const index = result.evidence.findIndex((item) => item.text === text);
    setActiveEvidence(index >= 0 ? index : null);
  };

  const reasoning = result?.reasoning ?? null;
  const contextText = lastRequest?.context?.trim() ? lastRequest.context : null;

  const contextEvidence = useMemo(() => {
    if (!reasoning) {
      return [];
    }
    return reasoning.evidence
      .filter((item) => item.source === "previous_comment")
      .map((item) => ({
        text: item.text,
        score: item.score ?? 0,
        start: item.start ?? undefined,
        end: item.end ?? undefined,
      }));
  }, [reasoning]);

  const currentExtras = useMemo(() => {
    if (!reasoning) {
      return [];
    }
    return reasoning.links.map((link) => ({
      text: link.from_text,
      score: 0,
      start: link.from_start,
      end: link.from_end,
    }));
  }, [reasoning]);

  const handleSelectLink = (link: ReasoningLink, index: number) => {
    setActiveLink((previous) =>
      previous?.index === index ? null : { link, index },
    );
  };

  return (
    <PageContainer>
      {/* ---------- intro ---------- */}
      <div className="max-w-3xl">
        <p className="kicker">context-aware · evidence-based</p>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight text-ink sm:text-3xl">
          Hate Speech Analysis
        </h1>
        <p className="mt-2 text-sm leading-relaxed text-ink-soft">
          Analyze a comment with optional conversational context. The model
          returns a structured prediction and the text spans that contributed
          to it.
        </p>
      </div>

      {/* ---------- input form ---------- */}
      <form
        onSubmit={handleSubmit}
        noValidate
        className="mt-8 grid gap-5 rounded-xl border border-edge bg-panel p-5 sm:p-6"
      >
        <ContextInput />
        <CommentInput onRequestSubmit={() => void analyze()} />

        <div className="flex flex-wrap items-center gap-3">
          <AnalyzeButton />
          <button
            type="button"
            className="btn-secondary"
            onClick={clearForm}
            disabled={isAnalyzing}
          >
            <Eraser className="h-3.5 w-3.5" aria-hidden="true" />
            Clear
          </button>
          <span className="text-[11px] text-ink-mute">
            Clear resets the form and results — history is kept.
          </span>
        </div>

        <div className="border-t border-edge pt-4">
          <p className="kicker flex items-center gap-2">
            <FlaskConical className="h-3.5 w-3.5" aria-hidden="true" />
            Demo inputs
          </p>
          <p className="mt-1 text-[11px] text-ink-mute">
            Fill the form with an example comment. Demo inputs only — results
            always come from the backend model, never from the examples.
          </p>
          <div className="mt-2 flex flex-wrap gap-2">
            {demoExamples.map((example) => (
              <button
                key={example.id}
                type="button"
                className="btn-secondary !px-3 !py-1.5 text-xs"
                onClick={() => {
                  setContext(example.context);
                  setCurrentComment(example.comment);
                }}
                disabled={isAnalyzing}
                aria-label={`Fill the form with demo example: ${example.label}`}
              >
                {example.label}
              </button>
            ))}
          </div>
        </div>
      </form>

      {/* ---------- results ---------- */}
      <div className="mt-8 space-y-4">
        {error && !isAnalyzing ? (
          <ErrorState error={error} onRetry={() => void analyze()} />
        ) : null}

        {isAnalyzing ? (
          <LoadingState contextUsed={context.trim().length > 0} />
        ) : result ? (
          <>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h2 className="text-lg font-semibold tracking-tight text-ink">
                Analysis Result
              </h2>
              {result.context_used ? (
                <Badge tone="info">Context-aware analysis</Badge>
              ) : (
                <Badge tone="neutral">Comment only</Badge>
              )}
            </div>

            <div className="grid gap-4 lg:grid-cols-2">
              <div className="lg:col-span-2">
                <PredictionCard result={result} />
              </div>
              {/* <TargetCard result={result} />
              <ReasonCard result={result} /> */}
              {reasoning ? (
                <div className="lg:col-span-2">
                  <ReasoningCard
                    reasoning={reasoning}
                    onSelectLink={handleSelectLink}
                    activeLinkIndex={activeLink?.index ?? null}
                  />
                </div>
              ) : null}
              <div className="lg:col-span-2" ref={evidenceRef}>
                <EvidenceViewer
                  text={lastRequest?.text ?? ""}
                  evidence={result.evidence}
                  headAvailable={result.prediction_available}
                  activeIndex={activeEvidence}
                  onActivate={setActiveEvidence}
                  rankByIndex={rankByIndex}
                  contextText={contextText}
                  contextEvidence={contextEvidence}
                  currentExtras={currentExtras}
                  connection={
                    activeLink
                      ? `“${activeLink.link.from_text}” → “${activeLink.link.to_text}”`
                      : null
                  }
                />
              </div>
              {result.evidence.length > 0 ? (
                <div className="lg:col-span-2">
                  <EvidenceList
                    evidence={result.evidence}
                    activeIndex={activeEvidence}
                    onActivate={setActiveEvidence}
                    onSelect={handleEvidenceSelect}
                  />
                </div>
              ) : null}
              <div className="lg:col-span-2">
                <AttributionDetail
                  request={lastRequest}
                  enabled={Boolean(result.prediction_available)}
                />
              </div>
              {result.reason_explanation ? (
                <div className="lg:col-span-2" ref={explanationRef}>
                  <ReasonExplanationCard
                    explanation={result.reason_explanation}
                    rankByText={rankByText}
                    onSelectEvidence={selectEvidenceByText}
                  />
                </div>
              ) : null}
              <ContextViewer contextUsed={result.context_used} request={lastRequest} />
              <ModelStatus />
            </div>

            <ModelArchitecture />
          </>
        ) : (
          <div className="rounded-xl border border-dashed border-edge bg-panel/50 p-6 text-center">
            <p className="text-sm text-ink-mute">
              Results will appear here after you analyze a comment.
            </p>
          </div>
        )}
      </div>
    </PageContainer>
  );
}
