import { useCallback } from "react";
import { useChatStore } from "@/stores/chatStore";
import { useStreamStore } from "@/stores/streamStore";

/**
 * Swap "waiting on you" for a resuming state the moment the user decides, since
 * the resolved frame can take seconds to arrive on the stream.
 *
 * Keys off the live `activeConversationId` (falling back to the pending
 * new-chat key), the same resolution the stream store uses — the route param is
 * stale for a replaceState'd new chat, so keying off it left the flag stuck.
 */
export function useMarkApprovalDecided(): () => void {
  const clearAwaitingApproval = useStreamStore(
    (state) => state.clearAwaitingApproval,
  );
  return useCallback(() => {
    const key =
      useChatStore.getState().activeConversationId ??
      useStreamStore.getState().pendingNewConversationKey;
    if (key) clearAwaitingApproval(key);
  }, [clearAwaitingApproval]);
}
