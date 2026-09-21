/**
 * Reads the backend's `has_live_approval` off an API conversation row.
 *
 * Typed loosely on purpose: the generated API types don't carry the field
 * yet (openapi regen is blocked on the pre-existing ToolInfo collision),
 * while the runtime JSON already does. Strict `=== true` so no truthy
 * garbage ever lights the approval dot. Delete this bridge when the
 * generated `ConversationSummary` includes the field.
 */
export function apiRowHasLiveApproval(row: unknown): boolean {
  if (typeof row !== "object" || row === null) {
    return false;
  }
  return (row as { has_live_approval?: unknown }).has_live_approval === true;
}
