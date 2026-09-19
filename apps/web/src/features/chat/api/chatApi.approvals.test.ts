// @vitest-environment jsdom

import { beforeEach, describe, expect, it, vi } from "vitest";
import { chatApi } from "@/features/chat/api/chatApi";

vi.mock("@/lib/api/service", () => ({
  apiService: { post: vi.fn() },
}));

import { apiService } from "@/lib/api/service";

const post = vi.mocked(apiService.post);

describe("chatApi approval decisions", () => {
  beforeEach(() => {
    post.mockReset();
  });

  it("maps 410 to not_found instead of painting the tapped verdict", async () => {
    post.mockRejectedValue({ response: { status: 410 } });
    const outcome = await chatApi.postApprovalDecision("ap_1", {
      decision: "approve",
    });
    expect(outcome).toEqual({ success: false, reason: "not_found" });
  });

  it("chunks batch decisions to the server 25-item cap", async () => {
    post.mockImplementation(async (_url: string, payload: unknown) => ({
      outcomes: (
        payload as { decisions: { approval_id: string }[] }
      ).decisions.map((d) => ({
        approval_id: d.approval_id,
        resolved: true,
        reason: null,
      })),
    }));
    const decisions = Array.from({ length: 30 }, (_, i) => ({
      approval_id: `ap_${i}`,
      decision: "approve" as const,
    }));
    const response = await chatApi.postApprovalBatchDecision({ decisions });
    expect(post).toHaveBeenCalledTimes(2);
    expect(
      (post.mock.calls[0][1] as { decisions: unknown[] }).decisions,
    ).toHaveLength(25);
    expect(
      (post.mock.calls[1][1] as { decisions: unknown[] }).decisions,
    ).toHaveLength(5);
    expect(response.outcomes).toHaveLength(30);
  });
});
