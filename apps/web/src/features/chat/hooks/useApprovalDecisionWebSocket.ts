import type { ApprovalRequestData, ApprovalStatus } from "@shared/chat";
import { useCallback, useEffect } from "react";
import type { TypedToolDataEntry } from "@/config/registries/toolRegistry";
import { db } from "@/lib/db/chatDb";
import { wsManager } from "@/lib/websocket/WebSocketManager";
import { useChatStore } from "@/stores/chatStore";

/**
 * WebSocket payload for a settled ledger approval.
 *
 * Emitted by the backend on every decide and revoke (see
 * `publish_ledger_decision` / `publish_ledger_revocation`). Without this, a
 * card decided anywhere but the open tab — another device, the review sheet
 * on a different message, an agent-side revoke — stays actionable forever.
 */
interface ApprovalDecidedEvent {
  type: "hil_approval_decided";
  conversation_id: string;
  approval_id: string;
  status: string;
  feedback?: string | null;
  version?: number | null;
}

const TERMINAL_STATUSES: ApprovalStatus[] = [
  "approved",
  "denied",
  "revoked",
  "executed",
  "failed",
  "unknown",
];

/**
 * Subscribe to `hil_approval_decided` and settle the matching open card.
 *
 * Finds the stored message whose `tool_data` carries the approval id, flips
 * the entry to its terminal status, and writes it back through IndexedDB +
 * the store so the card settles (or the tombstone appears) with no reload.
 * Unknown statuses and unknown ids are ignored — the ledger row stays truth.
 */
export function useApprovalDecisionWebSocket() {
  const handleDecided = useCallback(async (raw: unknown) => {
    const event = raw as ApprovalDecidedEvent;
    const { conversation_id, approval_id, status } = event;
    if (!conversation_id || !approval_id) return;
    if (!(TERMINAL_STATUSES as string[]).includes(status)) return;

    const state = useChatStore.getState();
    const messages = state.messagesByConversation[conversation_id] ?? [];
    const target = messages.find((message) =>
      (message.tool_data ?? []).some(
        (entry) =>
          entry.tool_name === "approval_request" &&
          (entry.data as { approval_id?: string } | undefined)?.approval_id ===
            approval_id,
      ),
    );
    if (!target) return;

    const tool_data: TypedToolDataEntry[] = (target.tool_data ?? []).map(
      (entry) => {
        const data = entry.data as Partial<ApprovalRequestData> | undefined;
        if (
          entry.tool_name !== "approval_request" ||
          data?.approval_id !== approval_id
        ) {
          return entry;
        }
        const next: ApprovalRequestData = {
          ...(data as ApprovalRequestData),
          status: status as ApprovalRequestData["status"],
          feedback: event.feedback ?? data.feedback ?? null,
        };
        return { ...entry, data: next } as TypedToolDataEntry;
      },
    );
    const updated = { ...target, tool_data };
    try {
      await db.putMessage(updated);
    } catch (err) {
      console.error(
        "[useApprovalDecisionWebSocket] Failed to persist card:",
        err,
      );
    }
    useChatStore.getState().addOrUpdateMessage(updated);
  }, []);

  useEffect(() => {
    wsManager.on("hil_approval_decided", handleDecided);
    return () => {
      wsManager.off("hil_approval_decided", handleDecided);
    };
  }, [handleDecided]);
}
