"""Turn a Browser-Use action into a human-readable caption.

Used as the fallback caption when the model's own next_goal is absent —
flash mode strips it, or a step produced no goal/thinking text at all — so
the SSE step card (runner.py) and the bot's photo caption
(bot_delivery.py) describe the same step the same way.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict

from app.schemas.browser import BrowserAction

# Actions whose whole meaning is the element they hit — a bare verb reads as
# noise ("Clicking"), the element's text reads as intent ("Clicking Add to cart").
_TARGETED_ACTIONS = {"click", "select_dropdown", "upload_file"}

_TARGET_MAX_CHARS = 40


def _shorten(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= _TARGET_MAX_CHARS:
        return collapsed
    return collapsed[: _TARGET_MAX_CHARS - 1].rstrip() + "…"


class _ActionParams(BaseModel):
    """The argument fields a caption can name, whatever else the action carried."""

    model_config = ConfigDict(extra="ignore")

    url: str | None = None
    query: str | None = None
    text: str | None = None
    coordinate_x: int | None = None
    coordinate_y: int | None = None


def _navigate_caption(params: _ActionParams, _target: str | None) -> str:
    host = urlparse(params.url).hostname if params.url else None
    return f"Opening {host.removeprefix('www.')}" if host else "Opening the page"


def _search_caption(params: _ActionParams, _target: str | None) -> str:
    q = (params.query or params.text or "").strip()
    return f'Searching "{q}"' if q else "Searching"


def _typing_caption(params: _ActionParams, target: str | None) -> str:
    text = (params.text or "").strip()
    if text and target:
        return f'Typing "{_shorten(text)}" into "{_shorten(target)}"'
    if text:
        return f'Typing "{_shorten(text)}"'
    return f'Typing into "{_shorten(target)}"' if target else "Typing"


def _select_dropdown_caption(params: _ActionParams, target: str | None) -> str:
    text = (params.text or "").strip()
    if text:
        return f'Choosing "{text}"'
    return f'Choosing in "{_shorten(target)}"' if target else "Choosing an option"


def _click_caption(params: _ActionParams, target: str | None) -> str:
    if target:
        return f'Clicking "{_shorten(target)}"'
    # A coordinate click resolves no element, so name the point it hit
    # rather than leaving a bare verb with no object at all.
    x, y = params.coordinate_x, params.coordinate_y
    if x is not None and y is not None:
        return f"Clicking at {x}, {y}"
    return "Clicking"


# Actions whose caption depends on the step's params/target.
_DYNAMIC_CAPTIONS: dict[str, Callable[[_ActionParams, str | None], str]] = {
    "navigate": _navigate_caption,
    "search": _search_caption,
    "search_page": _search_caption,
    "input": _typing_caption,
    "send_keys": _typing_caption,
    "select_dropdown": _select_dropdown_caption,
    "click": _click_caption,
}

# Actions whose caption is the same verb every time, regardless of params.
_STATIC_CAPTIONS: dict[str, str] = {
    "scroll": "Scrolling",
    "scroll_to_text": "Scrolling",
    "extract": "Reading the page",
    "read_file": "Reading the page",
    "read_long_content": "Reading the page",
    "find_text": "Reading the page",
    "find_elements": "Reading the page",
    "upload_file": "Uploading a file",
    "go_back": "Going back",
    "wait": "Waiting for the page",
    "request_human_takeover": "Handing this step to you",
    "solve_captcha_with_help": "Handing this step to you",
    "done": "Wrapping up",
}


def describe_action(name: str, params: Mapping[str, object], target: str | None = None) -> str:
    """Return a plain-language phrase for one action, naming its real target (the URL it opens, the text it types) so a caption reads like intent, not "Clicking" five times."""
    dynamic = _DYNAMIC_CAPTIONS.get(name)
    if dynamic is not None:
        return dynamic(_ActionParams.model_validate(params), target)
    return _STATIC_CAPTIONS.get(name) or name.replace("_", " ")


def caption_from_action_list(actions: list[BrowserAction]) -> str:
    """Return the same captions from a step snapshot's structured actions, whose params are real, so a caption can name what was opened or typed."""
    return _dedupe_join([describe_action(a.name, a.inputs, a.target) for a in actions])


def _dedupe_join(parts: list[str]) -> str:
    # de-dupe consecutive repeats ("Clicking; Clicking" → "Clicking")
    return ", ".join(dict.fromkeys(p for p in parts if p))
