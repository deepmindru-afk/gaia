import { describe, expect, it } from "vitest";

import { apiRowHasLiveApproval } from "./approvalFlag";

describe("apiRowHasLiveApproval", () => {
  it("lights only on an explicit true", () => {
    expect(apiRowHasLiveApproval({ has_live_approval: true })).toBe(true);
  });

  it("stays dark on false, missing, null, and garbage", () => {
    expect(apiRowHasLiveApproval({ has_live_approval: false })).toBe(false);
    expect(apiRowHasLiveApproval({})).toBe(false);
    expect(apiRowHasLiveApproval({ has_live_approval: null })).toBe(false);
    expect(apiRowHasLiveApproval({ has_live_approval: 1 })).toBe(false);
    expect(apiRowHasLiveApproval({ has_live_approval: "yes" })).toBe(false);
  });

  it("stays dark on non-objects", () => {
    expect(apiRowHasLiveApproval(null)).toBe(false);
    expect(apiRowHasLiveApproval(undefined)).toBe(false);
    expect(apiRowHasLiveApproval("true")).toBe(false);
  });
});
