"""The jev-ultrafast loop: Jev replaces the agent loop, not the chat model.

A port of browser-use/jev-ultrafast (MIT). A step is one snapshot, one Jev
decision request carrying every speculative target head, and a small text call
only when the chosen operation is TYPE_TEXT — no screenshots, no per-step chat
completion. ``run_jev_ultrafast`` is the whole surface: give it a session's CDP
websocket and a goal, and it runs to a terminal status.

This lives beside the ``JevChatModel`` path (``jev/chat_model.py``) rather than
replacing it, so the two can be benchmarked against each other; nothing here is
wired into the runner or the browser tool yet.
"""

from app.services.browser.jev.ultrafast.agent import (
    JevRunResult,
    JevRunStopped,
    JevUltrafastAgent,
)
from app.services.browser.jev.ultrafast.browser import (
    JevExecutionError,
    StalePage,
    UltrafastBrowser,
    fingerprint,
)
from app.services.browser.jev.ultrafast.model import (
    JevTextHelper,
    JevTextHelperError,
    JevUltrafastDecision,
    JevUltrafastDecisionError,
    action_space,
    build_request,
    choose,
    validate_choice,
)
from app.services.browser.jev.ultrafast.run import run_jev_ultrafast

__all__ = [
    "JevExecutionError",
    "JevRunResult",
    "JevRunStopped",
    "JevTextHelper",
    "JevTextHelperError",
    "JevUltrafastAgent",
    "JevUltrafastDecision",
    "JevUltrafastDecisionError",
    "StalePage",
    "UltrafastBrowser",
    "action_space",
    "build_request",
    "choose",
    "fingerprint",
    "run_jev_ultrafast",
    "validate_choice",
]
