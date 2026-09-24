"""LLM factory for the Browser-Use agent.

Jev makes every step decision and a small chat model writes typed values for
it. The browser_use import is local since the package is heavy and only a real
browser task needs it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.agents.llm.dev_lane import custom_endpoint, custom_lane_forced
from app.config.settings import settings
from app.constants.llm import (
    DEV_LLM_BROWSER_HEADERS,
    OPENAI_REASONING_EFFORT,
    OPENROUTER_REASONING_EFFORT,
    ReasoningLevel,
)
from app.services.browser.exceptions import BrowserUnavailableError
from app.services.browser.jev import build_jev_chat_model

if TYPE_CHECKING:
    from browser_use.llm.base import BaseChatModel


# OpenRouter is OpenAI-wire-compatible; Browser-Use talks to it via ChatOpenAI.
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# The cap covers reasoning tokens too: at 1024 a plan answer was cut mid-string.
_TEXT_MAX_COMPLETION_TOKENS = 4096
# Minimal effort answered a URL prompt in 1.3-2.1s, every reply valid (measured 2026-09-22).
_TEXT_REASONING = ReasoningLevel.LIGHT


def build_browser_llm(user_id: str | None = None) -> BaseChatModel:
    """Build the model that drives the Browser-Use agent: Jev, over its text helper.

    user_id is whose spend the writer's calls are metered to.
    """
    return build_jev_chat_model(text_model=_build_text_model(), user_id=user_id)


def _build_text_model() -> BaseChatModel:
    """Return the text helper on the forced dev lane's endpoint, else BROWSER_USE_JEV_TEXT_MODEL over OpenRouter.

    Browser-Use's client speaks chat completions only, whatever DEV_LLM_API says;
    it asks for JSON-schema output rather than tools, which OpenAI's reasoning
    models accept there.
    """
    from browser_use import ChatOpenAI  # noqa: PLC0415 -- heavy optional dep

    if custom_lane_forced():
        endpoint = custom_endpoint()
        return ChatOpenAI(
            model=endpoint.model,
            api_key=endpoint.api_key,
            base_url=endpoint.base_url,
            default_headers=DEV_LLM_BROWSER_HEADERS,
            max_completion_tokens=_TEXT_MAX_COMPLETION_TOKENS,
            reasoning_models=[endpoint.model],
            reasoning_effort=OPENAI_REASONING_EFFORT[_TEXT_REASONING],
        )
    if not settings.OPENROUTER_API_KEY:
        raise BrowserUnavailableError(
            "OPENROUTER_API_KEY is not set; Jev decisions and the text model both need it."
        )
    return ChatOpenAI(
        model=settings.BROWSER_USE_JEV_TEXT_MODEL,
        api_key=settings.OPENROUTER_API_KEY,
        base_url=_OPENROUTER_BASE_URL,
        max_completion_tokens=_TEXT_MAX_COMPLETION_TOKENS,
        reasoning_models=[settings.BROWSER_USE_JEV_TEXT_MODEL],
        reasoning_effort=OPENROUTER_REASONING_EFFORT[_TEXT_REASONING],
    )
