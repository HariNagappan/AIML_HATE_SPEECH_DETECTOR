/**
 * State-component tests: loading indicator copy and backend error rendering
 * (part of the spec's frontend test matrix).
 */
import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import ErrorState from "./ErrorState";
import LoadingState from "./LoadingState";
import type { ApiErrorInfo } from "../../types/analysis";

describe("LoadingState", () => {
  it("renders the indeterminate pipeline with context-aware copy", () => {
    render(<LoadingState contextUsed={true} />);
    expect(screen.getByText(/analyzing comment…/i)).toBeInTheDocument();
    expect(
      screen.getByText(/together with the provided context/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/no progress percentage/i)).toBeInTheDocument();
  });

  it("switches the copy when no context was provided", () => {
    render(<LoadingState contextUsed={false} />);
    expect(screen.getByText(/no context provided/i)).toBeInTheDocument();
  });
});

describe("ErrorState", () => {
  const error: ApiErrorInfo = {
    kind: "network",
    message: "Backend not reachable.",
  };

  it("renders a kind-specific title and the error message", () => {
    render(<ErrorState error={error} />);
    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByText("Backend unavailable")).toBeInTheDocument();
    expect(screen.getByText("Backend not reachable.")).toBeInTheDocument();
  });

  it("shows the HTTP status when present", () => {
    render(<ErrorState error={{ kind: "server", message: "boom", status: 500 }} />);
    expect(screen.getByText("HTTP 500")).toBeInTheDocument();
  });

  it("invokes onRetry, and hides the Retry button when no handler is given", () => {
    const onRetry = vi.fn();
    const { unmount } = render(<ErrorState error={error} onRetry={onRetry} />);
    fireEvent.click(screen.getByRole("button", { name: /retry/i }));
    expect(onRetry).toHaveBeenCalledTimes(1);

    unmount();
    render(<ErrorState error={error} />);
    expect(screen.queryByRole("button", { name: /retry/i })).toBeNull();
  });
});
