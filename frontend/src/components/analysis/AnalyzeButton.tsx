import { Loader2, Search } from "lucide-react";
import { useAnalysisStore } from "../../store/analysisStore";

export default function AnalyzeButton() {
  const isAnalyzing = useAnalysisStore((state) => state.isAnalyzing);

  return (
    <button
      type="submit"
      className="btn-primary min-w-[190px]"
      disabled={isAnalyzing}
      aria-busy={isAnalyzing}
    >
      {isAnalyzing ? (
        <>
          <Loader2 className="h-4 w-4 animate-spin-slow" aria-hidden="true" />
          Analyzing…
        </>
      ) : (
        <>
          <Search className="h-4 w-4" aria-hidden="true" />
          Analyze Comment
        </>
      )}
    </button>
  );
}
