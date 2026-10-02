import { apiClient } from "./client";
import type {
  AnalysisResult,
  ExplainRequest,
  ExplainResponse,
  HealthStatus,
  ModelInfo,
  PredictRequest,
} from "../types/analysis";

/**
 * Analysis API — UI components use this layer; they never call axios directly.
 */
export const analysisApi = {
  /** `POST /api/v1/predict` — classify a comment (with optional context). */
  async analyzeComment(
    request: PredictRequest,
    signal?: AbortSignal,
  ): Promise<AnalysisResult> {
    const { data } = await apiClient.post<AnalysisResult>(
      "/api/v1/predict",
      request,
      { signal },
    );
    return data;
  },

  /** `GET /health` — liveness + model-loaded flag. */
  async health(signal?: AbortSignal): Promise<HealthStatus> {
    const { data } = await apiClient.get<HealthStatus>("/health", { signal });
    return data;
  },

  /** `GET /api/v1/model/info` — model metadata for the status panel. */
  async modelInfo(signal?: AbortSignal): Promise<ModelInfo> {
    const { data } = await apiClient.get<ModelInfo>("/api/v1/model/info", {
      signal,
    });
    return data;
  },

  /** `POST /api/v1/explain` — per-token attribution details (optional). */
  async explain(
    request: ExplainRequest,
    signal?: AbortSignal,
  ): Promise<ExplainResponse> {
    const { data } = await apiClient.post<ExplainResponse>(
      "/api/v1/explain",
      request,
      { signal },
    );
    return data;
  },
};
