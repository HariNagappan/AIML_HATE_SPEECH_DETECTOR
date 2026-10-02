import { create } from "zustand";
import { persist } from "zustand/middleware";
import { analysisApi } from "../api/analysis";
import { toApiError } from "../api/client";
import type {
  AnalysisResult,
  ApiErrorInfo,
  HistoryEntry,
  PredictRequest,
} from "../types/analysis";

export const MAX_COMMENT_LENGTH = 5000;
export const MAX_CONTEXT_LENGTH = 5000;
export const MAX_HISTORY_ENTRIES = 50;

interface AnalysisState {
  // form
  context: string;
  currentComment: string;
  showValidation: boolean;

  // request lifecycle
  isAnalyzing: boolean;
  result: AnalysisResult | null;
  lastRequest: PredictRequest | null;
  error: ApiErrorInfo | null;

  // local history (persisted to localStorage)
  history: HistoryEntry[];

  // actions
  setContext: (value: string) => void;
  setCurrentComment: (value: string) => void;
  analyze: () => Promise<void>;
  clearForm: () => void;
  loadFromHistory: (entry: HistoryEntry) => void;
  clearHistory: () => void;
}

function createHistoryId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `h-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export const useAnalysisStore = create<AnalysisState>()(
  persist(
    (set, get) => ({
      context: "",
      currentComment: "",
      showValidation: false,
      isAnalyzing: false,
      result: null,
      lastRequest: null,
      error: null,
      history: [],

      setContext: (value) => set({ context: value }),
      setCurrentComment: (value) => set({ currentComment: value }),

      analyze: async () => {
        const { context, currentComment, isAnalyzing } = get();
        if (isAnalyzing) {
          return; // guard against duplicate submissions
        }

        const trimmedComment = currentComment.trim();
        if (!trimmedComment) {
          set({
            showValidation: true,
            error: {
              kind: "validation",
              message: "Please enter a comment to analyze.",
            },
          });
          return;
        }

        const trimmedContext = context.trim();
        const request: PredictRequest = {
          text: trimmedComment,
          context: trimmedContext ? trimmedContext : null,
        };

        set({ isAnalyzing: true, error: null, showValidation: false });

        try {
          const result = await analysisApi.analyzeComment(request);
          const entry: HistoryEntry = {
            id: createHistoryId(),
            timestamp: Date.now(),
            request,
            result,
          };
          set((state) => ({
            isAnalyzing: false,
            result,
            lastRequest: request,
            error: null,
            history: [entry, ...state.history].slice(0, MAX_HISTORY_ENTRIES),
          }));
        } catch (caught) {
          set({
            isAnalyzing: false,
            result: null,
            lastRequest: null,
            error: toApiError(caught),
          });
        }
      },

      clearForm: () =>
        set({
          context: "",
          currentComment: "",
          showValidation: false,
          isAnalyzing: false,
          result: null,
          lastRequest: null,
          error: null,
        }),

      loadFromHistory: (entry) =>
        set({
          context: entry.request.context ?? "",
          currentComment: entry.request.text,
          showValidation: false,
          result: entry.result,
          lastRequest: entry.request,
          error: null,
        }),

      clearHistory: () => set({ history: [] }),
    }),
    {
      name: "context-aware-hate-speech.history",
      // Only the history is persisted — form/result state stays in memory.
      partialize: (state) => ({ history: state.history }),
    },
  ),
);
