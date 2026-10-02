import axios, { AxiosError } from "axios";
import type { ApiErrorInfo } from "../types/analysis";

/** Backend base URL — configured via VITE_API_BASE_URL (see .env.example). */
export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL?.trim() || "http://localhost:8000";

/** Shared axios instance — the only place the frontend talks HTTP. */
export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60_000,
  headers: { "Content-Type": "application/json" },
});

/** Extract a human-readable message from a FastAPI error body. */
function detailMessage(data: unknown): string | null {
  if (typeof data === "string" && data.trim()) {
    return data;
  }
  if (data && typeof data === "object" && "detail" in data) {
    const detail = (data as { detail?: unknown }).detail;
    if (typeof detail === "string" && detail.trim()) {
      return detail;
    }
  }
  return null;
}

/** Map any thrown value to a friendly, typed error for the UI. */
export function toApiError(error: unknown): ApiErrorInfo {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError;
    if (axiosError.code === "ECONNABORTED" || axiosError.code === "ETIMEDOUT") {
      return {
        kind: "timeout",
        message: "The analysis took too long. Please try again.",
      };
    }

    const response = axiosError.response;
    if (!response) {
      return {
        kind: "network",
        message:
          "Unable to connect to the analysis server. Check that the backend is running.",
      };
    }

    const status = response.status;
    const detail = detailMessage(response.data);

    if (status === 400 || status === 422) {
      return {
        kind: "validation",
        message: detail ?? "Please check your input and try again.",
        status,
      };
    }
    if (status === 503) {
      return {
        kind: "model_unavailable",
        message: detail ?? "The analysis model is currently unavailable.",
        status,
      };
    }
    if (status >= 500) {
      return {
        kind: "server",
        message: detail ?? "The server encountered an error. Please try again.",
        status,
      };
    }
    return {
      kind: "unknown",
      message: detail ?? `Request failed with status ${status}.`,
      status,
    };
  }

  return {
    kind: "unknown",
    message: error instanceof Error ? error.message : "Something went wrong.",
  };
}
