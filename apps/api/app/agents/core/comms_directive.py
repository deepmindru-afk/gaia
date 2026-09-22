"""Parse comms' narration output into a turn outcome: reply, silence, or reaction.

When comms narrates a background executor update that is not worth a full message,
its output is a single control line instead of prose:

    ``SILENCE: <reason>``  → deliver nothing (the reason is logged, not shown)
    ``REACT: <emoji>``     → deliver a lightweight emoji acknowledgment

Anything else is an ordinary reply. Parsing is strict — the whole trimmed message
must be one control line — so prose that merely mentions the word never triggers
it, and the safe failure is "treat as a normal reply" (a stray directive shows as
text) rather than silently dropping a real message.
"""

import re

from app.constants.comms import REACT_KEYWORD, SILENCE_KEYWORD, CommsDirectiveKind
from app.constants.general import NEW_MESSAGE_BREAKER
from app.models.agent_models import CommsDirective

# Single logical line only (no DOTALL/MULTILINE): a real directive is one line,
# so any multi-line reply falls through to REPLY and can never be mis-silenced.
_DIRECTIVE_RE = re.compile(rf"^({SILENCE_KEYWORD}|{REACT_KEYWORD}):[ \t]*(.*)$", re.IGNORECASE)


def interpret_comms_output(text: str) -> CommsDirective:
    """Classify comms' final narration text as a reply, a silence, or a reaction."""
    match = _DIRECTIVE_RE.match(text.strip())
    if match:
        keyword, payload = match.group(1).upper(), match.group(2).strip()
        if keyword == SILENCE_KEYWORD:
            return CommsDirective(CommsDirectiveKind.SILENCE, payload)
        # A REACT with no emoji is meaningless — fall back to a normal reply so the
        # user still gets something rather than an empty reaction. The model has
        # been observed trailing the directive with the bubble-separator token
        # ("REACT: 😎<NEW_MESSAGE_BREAK>"); strip it so the badge renders the clean
        # emoji, and a break-only payload still falls through to REPLY.
        payload = payload.replace(NEW_MESSAGE_BREAKER, "").strip()
        if payload:
            return CommsDirective(CommsDirectiveKind.REACT, payload)
    return CommsDirective(CommsDirectiveKind.REPLY, text)
