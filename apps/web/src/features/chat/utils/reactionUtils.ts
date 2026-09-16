import type { MessageReaction } from "@/config/registries/baseMessageRegistry";
import type { IMessage } from "@/lib/db/chatDb";

/**
 * Fold comms REACT acks onto their target messages for render.
 *
 * A background emoji-ack arrives (and persists) as its own message with
 * `kind: "emoji_ack"` + `reacts_to_message_id`. Rendering it as a bubble
 * would show a stray giant glyph; instead the ack's emoji attaches to the
 * target message's `reactions` and the ack itself is dropped from the render
 * list. The stored records are untouched — folding is pure and re-runnable,
 * so reload, sync, and live push all converge on the same view.
 *
 * An ack whose target is absent (not loaded, another conversation, or no
 * target at all) stays a normal message — the pre-reaction bubble behavior —
 * so the acknowledgment is never lost. Folding is idempotent: re-running over
 * already-folded output changes nothing.
 */
export function foldReactionAcks(messages: IMessage[]): IMessage[] {
  const acks = messages.filter(isReactionAck);
  if (acks.length === 0) return messages;

  const byId = new Map<string, IMessage>();
  for (const message of messages) {
    byId.set(message.id, message);
    if (message.messageId && message.messageId !== message.id) {
      byId.set(message.messageId, message);
    }
  }

  const foldedAckIds = new Set<string>();
  const reactionsByTarget = new Map<string, MessageReaction[]>();
  for (const ack of acks) {
    const targetId = ack.reacts_to_message_id;
    const target = (targetId && byId.get(targetId)) || undefined;
    if (!target || target.id === ack.id) continue;
    foldedAckIds.add(ack.id);
    const list = reactionsByTarget.get(target.id) ?? [];
    if (!list.some((reaction) => reaction.ackId === ack.id)) {
      list.push({ emoji: ack.content, ackId: ack.id });
    }
    reactionsByTarget.set(target.id, list);
  }
  if (foldedAckIds.size === 0) return messages;

  return messages.flatMap((message) => {
    if (foldedAckIds.has(message.id)) return [];
    const extra = reactionsByTarget.get(message.id);
    if (!extra || extra.length === 0) return [message];
    const existing = message.reactions ?? [];
    const merged = [
      ...existing,
      ...extra.filter(
        (reaction) =>
          !existing.some((current) => current.ackId === reaction.ackId),
      ),
    ];
    if (merged.length === existing.length) return [message];
    return [{ ...message, reactions: merged }];
  });
}

function isReactionAck(message: IMessage): boolean {
  return message.kind === "emoji_ack" && message.content.trim().length > 0;
}
