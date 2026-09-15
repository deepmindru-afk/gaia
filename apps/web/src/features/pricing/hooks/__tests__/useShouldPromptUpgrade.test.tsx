// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";

import type { UserSubscriptionStatus } from "../../api/pricingApi";
import { useShouldPromptUpgrade } from "../usePricing";

const { mockGetSubscriptionStatus } = vi.hoisted(() => ({
  mockGetSubscriptionStatus: vi.fn(),
}));

vi.mock("@/features/auth/hooks/useUser", () => ({
  useUser: () => ({ id: "user-1" }),
}));

vi.mock("../../api/pricingApi", async (importOriginal) => {
  const original =
    await importOriginal<typeof import("../../api/pricingApi")>();
  return {
    ...original,
    pricingApi: {
      ...original.pricingApi,
      getSubscriptionStatus: mockGetSubscriptionStatus,
    },
  };
});

const FREE_STATUS: UserSubscriptionStatus = {
  user_id: "user-1",
  is_subscribed: false,
  can_upgrade: true,
  can_downgrade: false,
};

function renderUpgradeHook() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
  return renderHook(() => useShouldPromptUpgrade(), { wrapper });
}

describe("useShouldPromptUpgrade", () => {
  it("prompts a confirmed free user", async () => {
    mockGetSubscriptionStatus.mockResolvedValueOnce(FREE_STATUS);
    const { result } = renderUpgradeHook();
    await waitFor(() => expect(result.current).toBe(true));
  });

  it("stays silent for a subscribed user", async () => {
    mockGetSubscriptionStatus.mockResolvedValueOnce({
      ...FREE_STATUS,
      is_subscribed: true,
    });
    const { result } = renderUpgradeHook();
    await waitFor(() => expect(mockGetSubscriptionStatus).toHaveBeenCalled());
    expect(result.current).toBe(false);
  });

  it("fails open when the status request errors, so upgrade paths stay visible", async () => {
    mockGetSubscriptionStatus.mockRejectedValueOnce(new Error("network down"));
    const { result } = renderUpgradeHook();
    // The bug: an error resolved exactly like loading/subscribed (false),
    // hiding every upgrade entry point for the session.
    await waitFor(() => expect(result.current).toBe(true));
  });
});
