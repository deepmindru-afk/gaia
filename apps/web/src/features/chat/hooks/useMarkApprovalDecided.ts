import { useCallback } from "react";
import { useChatStore } from "@/stores/chatStore";
import { useStreamStore } from "@/stores/streamStore";

/**
 * Swap "waiting on you" for a resuming state the moment the user decides, since
 * the resolved frame can take seconds to arrive on the stream.
 *
 * Takes the owning conversation when the caller knows it (the card's message
 * context — a sheet or a background conversation's card can outlive the active
 * one), falling back to the live `activeConversationId` and then the pending
 * new-chat key, the same resolution the stream store uses — the route param is
 * stale for a replaceState'd new chat, so keying off it left the flag stuck.
 */
export function useMarkApprovalDecided(): (conversationId?: string) => void {
  const clearAwaitingApproval = useStreamStore(
    (state) => state.clearAwaitingApproval,
  );
  return useCallback(
    (conversationId?: string) => {
      const key =
        conversationId ??
        useChatStore.getState().activeConversationId ??
        useStreamStore.getState().pendingNewConversationKey;
      if (key) clearAwaitingApproval(key);
    },
    [clearAwaitingApproval],
  );
}
