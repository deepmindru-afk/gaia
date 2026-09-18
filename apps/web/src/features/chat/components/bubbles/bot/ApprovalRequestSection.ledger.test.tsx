// @vitest-environment jsdom
import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { ApprovalRequestData } from "@shared/chat";
import ApprovalRequestSection from "@/features/chat/components/bubbles/bot/ApprovalRequestSection";
import { chatApi } from "@/features/chat/api/chatApi";

vi.mock("@/features/chat/api/chatApi", () => ({
  chatApi: { postApprovalDecision: vi.fn() },
}));

vi.mock("@/features/chat/hooks/useMarkApprovalDecided", () => ({
  useMarkApprovalDecided: () => vi.fn(),
}));

vi.mock("@/lib/toast", () => ({
  toast: { error: vi.fn() },
}));

const card = (overrides: Partial<ApprovalRequestData> = {}): ApprovalRequestData => ({
  approval_id: "ap_1",
  tool_call_id: "call-1",
  gated_tool_name: "GMAIL_SEND_EMAIL",
  integration_name: "gmail",
  summary: "Send email",
  args_preview: { to: "b@x" },
  status: "pending",
  feedback: null,
  timeout_seconds: 300,
  ...overrides,
});

describe("ApprovalRequestSection ledger UX", () => {
  beforeEach(() => {
    vi.useRealTimers();
    vi.mocked(chatApi.postApprovalDecision).mockReset().mockResolvedValue(undefined);
  });

  it("shows the card age when the ledger provides it", () => {
    render(
      <ApprovalRequestSection data={card({ age_seconds: 172800 })} onDecided={() => {}} />,
    );
    expect(screen.getByText("asked 2d ago")).toBeDefined();
  });

  it("asks in place before submitting a day-old approval", async () => {
    vi.useFakeTimers();
    render(
      <ApprovalRequestSection
        data={card({ age_seconds: 259200 })}
        onDecided={() => {}}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /^approve$/i }));
    // Re-confirm appears; nothing submitted yet.
    expect(screen.getByText(/still want this/i)).toBeDefined();
    expect(chatApi.postApprovalDecision).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /yes, still/i }));
    await vi.advanceTimersByTimeAsync(5000);
    expect(chatApi.postApprovalDecision).toHaveBeenCalledTimes(1);
    vi.useRealTimers();
  });

  it("holds a fresh approve behind commit grace and cancels cleanly", async () => {
    vi.useFakeTimers();
    render(
      <ApprovalRequestSection data={card({ age_seconds: 5 })} onDecided={() => {}} />,
    );
    fireEvent.click(screen.getByRole("button", { name: /^approve$/i }));
    expect(chatApi.postApprovalDecision).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: /cancel/i })).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: /cancel/i }));
    await vi.advanceTimersByTimeAsync(5000);
    expect(chatApi.postApprovalDecision).not.toHaveBeenCalled();
    vi.useRealTimers();
  });

  it("submits the rendered row version with the decision", async () => {
    vi.useFakeTimers();
    const onDecided = vi.fn();
    render(
      <ApprovalRequestSection
        data={card({ age_seconds: 5, ledger_version: 4 })}
        onDecided={onDecided}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /^approve$/i }));
    await vi.advanceTimersByTimeAsync(5000);
    expect(chatApi.postApprovalDecision).toHaveBeenCalledWith("ap_1", {
      decision: "approve",
      feedback: undefined,
      scope: "once",
      v: 4,
    });
    vi.useRealTimers();
  });
});
