"""Static agent prompt templates, per-channel.

These strings are passed verbatim to LangChain as the system prompt. They are
free of user-specific interpolation so they are byte-identical across every
user on a given channel and flag variant. That lets the LLM provider's
implicit prompt cache match across users — the first request of the day on
web warms the cache and every subsequent web user on the same variant hits
it on turn 1.

Each channel's static prompt is the base comms prompt plus the output-format
block that applies to that channel (OpenUI component library on renderable
channels; text-only restrictions everywhere else). Dynamic per-user/per-turn
context (user name, timezone, preferences, memories, current time) is carried
in a separate dynamic-context system message placed AFTER this one.
"""

from typing import Final

from app.agents.prompts.comms_prompts import COMMS_AGENT_PROMPT
from app.agents.prompts.executor_activation_prompt import build_activation_executor_prompt
from app.agents.prompts.openui_prompts import MARKDOWN_ONLY_ADDENDUM, OPENUI_ADDENDUM
from app.agents.workspace.operational_docs import GAIA_CORE
from app.constants.general import NEW_MESSAGE_BREAKER

# Output-format addenda for renderable channels (web, mobile, desktop). Both
# variants are precomputed once at import: the per-user OpenUI decision only
# picks between these two byte-stable strings, so the provider's prompt cache
# sees two buckets per channel instead of one string per user.
_RENDERABLE_ADDENDUM_OPENUI_ON: Final[str] = "\n\n" + OPENUI_ADDENDUM
_RENDERABLE_ADDENDUM_OPENUI_OFF: Final[str] = "\n\n" + MARKDOWN_ONLY_ADDENDUM


# Desktop-app capability addendum. The comms agent has no desktop tools
# itself — it delegates to the executor, which discovers them via retrieval.
_DESKTOP_ADDENDUM: Final[str] = """

## Desktop Context
The user is chatting from the GAIA desktop app on their computer. Through the
executor, GAIA can act on that computer: look at the screen (take_screenshot),
read or write the clipboard, open applications, open URLs in the browser, and
list open windows. When the user references "this", "my screen", or something
they are currently looking at, delegate to the executor and have it take a
screenshot for visual context."""


# Output-format addendum for each text-only channel. These strings are the
# platform-specific formatting rules the LLM must stick to. We inline them
# here so the addendum is byte-identical across every WhatsApp user, etc.
#
# This is the LAST thing the model reads, roughly 27k characters after the Chat
# Bubbles section, so the bubble rule is restated here rather than relied on at
# that distance. Belt and braces: the bot layer also splits paragraphs
# deterministically, so a model that ignores the sentinel still gets split.
def _text_only_addendum(platform_name: str, formatting: str) -> str:
    return f"""

## Platform Context (IMPORTANT)
The user is messaging from **{platform_name}**. This is a text-based messaging platform, so your reply arrives as one or more real chat messages and nothing else.

BUBBLES ON THIS PLATFORM (restated here because it matters most here):
- Every conversational beat is its own bubble, separated by {NEW_MESSAGE_BREAKER}. So is every paragraph of prose: on a messaging app a three-paragraph block is three messages, the way a person actually texts.
- Structured content stays WHOLE in one bubble: a list, a set of bullets, a table written out as a list, a code block, a set of numbered steps. Splitting one of those breaks it into fragments that no longer read as one thing.
- Aim for about 600 characters per bubble. When a bubble runs long, SPLIT IT, and never solve it by cutting content: the data goes out in full across more bubbles.
- So a three-paragraph answer leaves you as: first paragraph, {NEW_MESSAGE_BREAKER}, second paragraph, {NEW_MESSAGE_BREAKER}, third paragraph. Sending a multi-paragraph answer as one message is the single most common way this goes wrong.

OUTPUT RESTRICTIONS for this platform:
- NO HTML, interactive UI components, artifacts, or rich cards, since the user cannot see them
- NO markdown links [text](url). Paste the bare URL on its own line and let the platform link it. This overrides the clickable-markdown rule in Delivering Results, which applies only to the app.
- NO tables. Write the same rows as a flat list.
- NO nested lists. One level of bullets only; a sub-point becomes its own line or its own bubble.
- NO bold and NO italics, at all, in a chat reply. Not for a key number, not for a heading, not to make a point land. Emphasis here is word choice and rhythm; bolded fragments scattered through a chat message are a formatting tic that reads as generated text. The only formatting syntax you may use here is for code: {formatting}
- NO images or embedded media in your response
- The user CANNOT see tool_data UI, MCP apps, or any frontend components
- When showing structured data (search results, calendar events, emails, etc.), format as clean text lists
- Artifacts and HTML content blocks are invisible to the user, so describe results in plain text instead
- When an integration needs to be connected: paste the connect URL directly in your reply. There is no connect button on this platform, so the URL is the only way for the user to connect

WHAT TO DO INSTEAD:
- Present all information as plain text lines and, where there are genuinely separate items, one flat level of bullets
- For data that would normally show as a card/component, write it out as a clear text summary
- For content that would be an artifact, include it directly in your message as text
- Concise here means cutting filler, never cutting data a result carried. Trim your own wrapper words, then split what is left across bubbles. NON-NEGOTIABLE 3 still outranks brevity."""


_WHATSAPP_ADDENDUM: Final[str] = _text_only_addendum(
    "WhatsApp",
    "WhatsApp code formatting: ```code```",
)
_TELEGRAM_ADDENDUM: Final[str] = _text_only_addendum(
    "Telegram",
    "Telegram code formatting: `code`, ```code blocks```",
)
_DISCORD_ADDENDUM: Final[str] = _text_only_addendum(
    "Discord",
    "Discord code formatting: `code`, ```code blocks```, > quotes",
)
_SLACK_ADDENDUM: Final[str] = _text_only_addendum(
    "Slack",
    "Slack code formatting: `code`, ```code blocks```, > quotes",
)


# Pre-assembled static comms prompts per channel. Each is a single Python
# string literal that lives for the process lifetime, so the bytes sent to
# the LLM are identical for every user on that channel *and experiment
# variant*. The ON variants below are the defaults; OFF variants are built by
# ``get_comms_static_prompt(..., openui_enabled=False)``.
COMMS_PROMPT_BY_SOURCE: Final[dict[str, str]] = {
    "web": COMMS_AGENT_PROMPT + _RENDERABLE_ADDENDUM_OPENUI_ON,
    "mobile": COMMS_AGENT_PROMPT + _RENDERABLE_ADDENDUM_OPENUI_ON,
    "desktop": COMMS_AGENT_PROMPT
    + _RENDERABLE_ADDENDUM_OPENUI_ON
    + _DESKTOP_ADDENDUM,
    "whatsapp": COMMS_AGENT_PROMPT + _WHATSAPP_ADDENDUM,
    "telegram": COMMS_AGENT_PROMPT + _TELEGRAM_ADDENDUM,
    "discord": COMMS_AGENT_PROMPT + _DISCORD_ADDENDUM,
    "slack": COMMS_AGENT_PROMPT + _SLACK_ADDENDUM,
}

# Renderable channels whose output format is experiment-gated. Text-only
# channels ignore the OpenUI flag entirely.
_RENDERABLE_SOURCES: Final[frozenset[str]] = frozenset({"web", "mobile", "desktop"})

# Default (web-style) static prompt used when ``source`` is unknown/None.
COMMS_PROMPT_DEFAULT: Final[str] = COMMS_PROMPT_BY_SOURCE["web"]


def get_comms_static_prompt(source: str | None, openui_enabled: bool = True) -> str:
    """Return the per-channel static comms prompt.

    The choice of channel-specific static prompt means the provider's
    implicit prompt cache can match byte-for-byte across all users on the
    same channel. ``openui_enabled`` (resolved per user by
    ``app.services.feature_flags.is_comms_openui_enabled``) swaps the
    output-format block on renderable channels only, so the cache sees at
    most two variants per channel. Unknown sources fall back to web.
    """
    normalized = source.strip().lower() if source else "web"
    if normalized not in COMMS_PROMPT_BY_SOURCE:
        normalized = "web"
    if openui_enabled or normalized not in _RENDERABLE_SOURCES:
        return COMMS_PROMPT_BY_SOURCE[normalized]
    base = COMMS_AGENT_PROMPT + _RENDERABLE_ADDENDUM_OPENUI_OFF
    if normalized == "desktop":
        return base + _DESKTOP_ADDENDUM
    return base


# Legacy name still imported by a few call sites. Kept as an alias for the
# default web-style prompt; all per-channel users should go through
# ``get_comms_static_prompt``.
COMMS_PROMPT_TEMPLATE: Final[str] = COMMS_PROMPT_DEFAULT

# The executor's static prefix carries the always-on operating core (GAIA_CORE):
# user-independent self-knowledge + the self-management capability menu + the
# read_manual topic routing. It is appended here (not interpolated per user) so
# the prompt stays byte-identical across users and rides the provider's
# prompt cache. The base is the activation rewrite of EXECUTOR_AGENT_PROMPT,
# precomputed once (building it anchors against the base text and raises on
# drift, so a stale rewrite can never ship silently).
_EXECUTOR_BASE: Final[str] = build_activation_executor_prompt()


def get_executor_prompt() -> str:
    """Return the static executor prompt (activation doctrine, always)."""
    return _EXECUTOR_BASE + "\n\n" + GAIA_CORE


EXECUTOR_PROMPT_TEMPLATE: Final[str] = get_executor_prompt()
