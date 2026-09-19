import type {
  ApprovalRequestData,
  ApprovalStatus,
  ToolDataEntry,
} from "@gaia/shared/chat";
import {
  APPROVAL_REQUEST_TOOL_NAME,
  approvalOutcomeLabel,
} from "@gaia/shared/chat";
import type { Message } from "@/features/chat/api/chat-api";

export interface ApprovalMeta {
  label: string;
  color: string;
}

/**
 * Resolved-card meta for every non-pending status. The card derefs this map
 * for any `data.status !== "pending"`, so all nine ledger states must be
 * present — a missing key crashes the card on ledger-terminal statuses
 * (revoked/executed/failed/unknown).
 */
export const APPROVAL_RESOLVED_META: Record<
  Exclude<ApprovalStatus, "pending">,
  ApprovalMeta
> = {
  auto_approved: { label: "Ran automatically", color: "#34d399" },
  approved: { label: "Approved", color: "#34d399" },
  denied: { label: "Declined", color: "#f87171" },
  timeout: { label: "Timed out", color: "#fbbf24" },
  abandoned: { label: "Dropped", color: "#a1a1aa" },
  executed: { label: "Executed", color: "#34d399" },
  failed: { label: "Failed", color: "#f87171" },
  unknown: { label: "Unknown", color: "#fbbf24" },
  revoked: { label: "Withdrawn", color: "#a1a1aa" },
};

/**
 * Activity-row chips for every settled state. Labels match web's
 * SubagentRow APPROVAL_CHIP (Executed/success, Failed/danger,
 * Unknown/warning); revoked renders as a tombstone-style chip since the
 * mobile timeline has no separate tombstone surface.
 */
export const APPROVAL_CHIP_META: Record<
  Exclude<ApprovalStatus, "pending">,
  ApprovalMeta
> = {
  approved: { label: "Approved", color: "#22c55e" },
  auto_approved: { label: "Auto-approved", color: "#22c55e" },
  denied: { label: "Denied", color: "#ef4444" },
  timeout: { label: "Expired", color: "#71717a" },
  abandoned: { label: "Expired", color: "#71717a" },
  executed: { label: "Executed", color: "#22c55e" },
  failed: { label: "Failed", color: "#ef4444" },
  unknown: { label: "Unknown", color: "#eab308" },
  revoked: { label: "Withdrawn", color: "#71717a" },
};

/** Statuses the backend publishes over `hil_approval_decided`. */
export const TERMINAL_APPROVAL_STATUSES: ApprovalStatus[] = [
  "approved",
  "denied",
  "revoked",
  "executed",
  "failed",
  "unknown",
];

const KNOWN_STATUSES: ReadonlySet<string> = new Set<string>([
  "pending",
  "approved",
  "denied",
  "timeout",
  "abandoned",
  "auto_approved",
  "revoked",
  "executed",
  "failed",
  "unknown",
]);

export function isKnownApprovalStatus(
  status: string,
): status is ApprovalStatus {
  return KNOWN_STATUSES.has(status);
}

/**
 * One-line outcome for a settled card. Shared `approvalOutcomeLabel` returns
 * "" for executed/failed/unknown (libs is read-only from mobile), so fall
 * back to per-status copy here — preferring explicit feedback when present.
 */
export function approvalOutcomeText(data: ApprovalRequestData): string {
  const base = approvalOutcomeLabel(data);
  if (base.trim() !== "") return base;
  const feedback = data.feedback?.trim();
  if (feedback) return feedback;
  switch (data.status) {
    case "executed":
      return "The approved action ran";
    case "failed":
      return "The approved action failed";
    case "unknown":
      return "Outcome unknown";
    default:
      return "";
  }
}

export interface ApprovalDecidedEvent {
  conversation_id: string;
  approval_id: string;
  status: ApprovalStatus;
  feedback: string | null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

/**
 * Parse a raw `hil_approval_decided` frame. Accepts the top-level web shape
 * (`{conversation_id, approval_id, status, feedback}`) and a nested
 * `{data: {...}}` variant. Returns null for unknown statuses/ids — the
 * ledger row stays truth.
 */
export function parseApprovalDecidedEvent(
  raw: unknown,
): ApprovalDecidedEvent | null {
  if (!isRecord(raw)) return null;
  const payload = isRecord(raw.data) ? raw.data : raw;
  const conversation_id = payload.conversation_id;
  const approval_id = payload.approval_id;
  const status = payload.status;
  if (typeof conversation_id !== "string" || conversation_id === "") {
    return null;
  }
  if (typeof approval_id !== "string" || approval_id === "") return null;
  if (typeof status !== "string") return null;
  if (!(TERMINAL_APPROVAL_STATUSES as string[]).includes(status)) return null;
  const feedback = payload.feedback;
  return {
    conversation_id,
    approval_id,
    status: status as ApprovalStatus,
    feedback: typeof feedback === "string" && feedback !== "" ? feedback : null,
  };
}

function applyToToolData(
  toolData: ToolDataEntry[] | undefined,
  approval_id: string,
  status: ApprovalStatus,
  feedback: string | null,
): { toolData: ToolDataEntry[] | undefined; changed: boolean } {
  if (!toolData) return { toolData, changed: false };
  let changed = false;
  const next = toolData.map((entry) => {
    if (entry.tool_name !== APPROVAL_REQUEST_TOOL_NAME) return entry;
    const data = entry.data as Partial<ApprovalRequestData> | null;
    if (!data || data.approval_id !== approval_id) return entry;
    changed = true;
    return {
      ...entry,
      data: {
        ...(data as ApprovalRequestData),
        status,
        feedback: feedback ?? data.feedback ?? null,
      },
    };
  });
  return { toolData: changed ? next : toolData, changed };
}

/**
 * Flip the stored card matching `approval_id` to its terminal status.
 * Pure — operates on a Message snapshot, never mutates the input. Returns
 * the original array by identity when nothing matched so callers can skip
 * store writes and persistence.
 */
export function applyApprovalDecisionToMessages(
  messages: Message[],
  event: ApprovalDecidedEvent,
): { messages: Message[]; changed: boolean } {
  let changed = false;
  const next = messages.map((message) => {
    const { toolData, changed: entryChanged } = applyToToolData(
      message.toolData,
      event.approval_id,
      event.status,
      event.feedback,
    );
    if (!entryChanged) return message;
    changed = true;
    return { ...message, toolData };
  });
  return { messages: changed ? next : messages, changed };
}
