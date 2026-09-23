/**
 * The comms `REACT: <emoji>` control line, as the bot streamer must see it.
 *
 * The rule is the backend's `interpret_comms_output` (apps/api/app/agents/core/
 * comms_directive.py): the whole trimmed turn is one line starting with `REACT:`
 * (any case), and its payload is non-empty once message-break tokens are removed.
 * Only a turn matching it gets an `emoji_ack`; anything else is sent as a reply.
 */
import { NEW_MESSAGE_BREAK_TOKEN } from "../../utils/messageBreakUtils";

const REACT_KEYWORD = "REACT:";

/** `[^\n]` rather than `.`: Python's `.` excludes only `\n`, JS's also `\r` and U+2028/9. */
const REACT_DIRECTIVE_PATTERN = /^REACT:([^\n]*)$/i;

/** The emoji a whole turn reacts with, or null when the backend treats it as a reply. */
export function reactDirectiveEmoji(turnText: string): string | null {
  const match = REACT_DIRECTIVE_PATTERN.exec(turnText.trim());
  if (!match) return null;
  const emoji = match[1].replaceAll(NEW_MESSAGE_BREAK_TOKEN, "").trim();
  return emoji || null;
}

/**
 * Whether a turn streamed so far could still end as a REACT directive — the
 * text the streamer must hold back until the `emoji_ack` or the end of the turn.
 */
export function couldBecomeReactDirective(turnText: string): boolean {
  const candidate = turnText.trimStart();
  const head = candidate.slice(0, REACT_KEYWORD.length).toUpperCase();
  if (!REACT_KEYWORD.startsWith(head)) return false;
  // Past a newline only trailing whitespace may follow, so the turn is final.
  if (!candidate.includes("\n")) return true;
  return reactDirectiveEmoji(candidate) !== null;
}
