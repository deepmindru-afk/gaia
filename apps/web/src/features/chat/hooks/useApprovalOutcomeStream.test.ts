// @vitest-environment jsdom

import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useApprovalOutcomeStream } from "@/features/chat/hooks/useApprovalOutcomeStream";

const listeners = vi.hoisted(() => ({
  impl: null as null | ((url: string, opts: Record<string, unknown>) => void),
}));

vi.mock("@microsoft/fetch-event-source", () => ({
  fetchEventSource: (url: string, opts: Record<string, unknown>) =>
    listeners.impl?.(url, opts),
}));

describe("useApprovalOutcomeStream", () => {
  beforeEach(() => {
    listeners.impl = null;
  });

  it("stays running until a terminal outcome arrives, then settles", async () => {
    let onmessage: ((event: { data: string }) => void) | null = null;
    listeners.impl = (_url, opts) => {
      onmessage = opts.onmessage as (event: { data: string }) => void;
    };
    const { result } = renderHook(() => useApprovalOutcomeStream());
    act(() => {
      result.current.start("ap_1");
    });
    expect(result.current.phase).toBe("running");
    act(() => {
      onmessage?.({
        data: JSON.stringify({ status: "running", approval_id: "ap_1" }),
      });
    });
    expect(result.current.phase).toBe("running");
    act(() => {
      onmessage?.({
        data: JSON.stringify({ status: "executed", approval_id: "ap_1" }),
      });
    });
    await waitFor(() => expect(result.current.phase).toBe("done"));
    expect(result.current.outcome).toBe("executed");
  });

  it("ends the stream on socket error without an outcome", async () => {
    let onError: (() => void) | null = null;
    listeners.impl = (_url, opts) => {
      onError = opts.onerror as () => void;
    };
    const { result } = renderHook(() => useApprovalOutcomeStream());
    act(() => {
      result.current.start("ap_1");
    });
    act(() => {
      onError?.();
    });
    await waitFor(() => expect(result.current.phase).toBe("done"));
    expect(result.current.outcome).toBeNull();
  });

  it("ignores malformed frames", async () => {
    let onmessage: ((event: { data: string }) => void) | null = null;
    listeners.impl = (_url, opts) => {
      onmessage = opts.onmessage as (event: { data: string }) => void;
    };
    const { result } = renderHook(() => useApprovalOutcomeStream());
    act(() => {
      result.current.start("ap_1");
    });
    act(() => {
      onmessage?.({ data: "not-json{{{ " });
    });
    expect(result.current.phase).toBe("running");
    expect(result.current.outcome).toBeNull();
  });
});
