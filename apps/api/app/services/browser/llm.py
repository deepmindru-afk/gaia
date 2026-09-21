"""LLM factory for the Browser-Use agent.

Jev makes every step decision and a small OpenRouter chat model writes typed
values for it; both ride OPENROUTER_API_KEY. The browser_use import is local
since the package is heavy and only a real browser task needs it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.config.settings import settings
from app.services.browser.exceptions import BrowserUnavailableError
from app.services.browser.jev import build_jev_chat_model

if TYPE_CHECKING:
    from browser_use.llm.base import BaseChatModel


# OpenRouter is OpenAI-wire-compatible; Browser-Use talks to it via ChatOpenAI.
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def build_browser_llm(user_id: str | None = None) -> BaseChatModel:
    """Build the model that drives the Browser-Use agent: Jev, over its text helper.

    user_id is whose spend the writer's calls are metered to.
    """
    if not settings.OPENROUTER_API_KEY:
        raise BrowserUnavailableError(
            "OPENROUTER_API_KEY is not set; Jev decisions and the text model both need it."
        )
    return build_jev_chat_model(text_model=_build_text_model(), user_id=user_id)


def _build_text_model() -> BaseChatModel:
    from browser_use import ChatOpenAI  # noqa: PLC0415 -- heavy optional dep

    # The completion budget covers the model's reasoning tokens too: at 1024 a
    # plan answer was cut off mid-string after the thinking spent the budget, so
    # the cap stays generous and the reasoning effort stays minimal, which
    # answered a URL prompt in 1.3-2.1s with every reply valid (measured 2026-09-22).
    return ChatOpenAI(
        model=settings.BROWSER_USE_JEV_TEXT_MODEL,
        api_key=settings.OPENROUTER_API_KEY,
        base_url=_OPENROUTER_BASE_URL,
        max_completion_tokens=4096,
        reasoning_models=[settings.BROWSER_USE_JEV_TEXT_MODEL],
        reasoning_effort="minimal",
    )
