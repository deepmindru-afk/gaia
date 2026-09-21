// @vitest-environment jsdom

/**
 * Regression: the clear keyed off the route param, which is stale for a
 * replaceState'd new chat, so deciding a gate never cleared the live session and
 * the approval flag stayed stuck. It must key off `activeConversationId`.
 */
import { renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { useMarkApprovalDecided } from "@/features/chat/hooks/useMarkApprovalDecided";
import { useChatStore } from "@/stores/chatStore";
import { useStreamStore } from "@/stores/streamStore";

const CONVO = "conv-live";

const resetStores = (): void => {
  for (const key of Object.keys(useStreamStore.getState().sessions)) {
    useStreamStore.getState().endSession(key);
  }
  useChatStore.setState({ activeConversationId: null });
};

describe("useMarkApprovalDecided", () => {
  afterEach(resetStores);

  it("clears the active conversation's gate without a matching route param", () => {
    resetStores();
    const store = useStreamStore.getState();
    store.startSession(CONVO);
    store.updateSession(CONVO, { awaitingApproval: true });
    // The route param is absent (a fresh chat rewrote the URL); the live session
    // is only reachable via activeConversationId.
    useChatStore.setState({ activeConversationId: CONVO });

    const { result } = renderHook(() => useMarkApprovalDecided());
    result.current();

    expect(useStreamStore.getState().sessions[CONVO]?.awaitingApproval).toBe(
      false,
    );
  });
});
