"""LLM factory for the Browser-Use agent.

Deliberately decoupled from the chat harness model: browser automation always
gets a strong, vision-capable model chosen by BROWSER_USE_LLM_* settings,
regardless of which model the surrounding conversation runs on. Browser-Use
ships its own chat wrappers (it is not a LangChain model), so this builds one of
those. Imports are local so the heavy browser_use package loads only when a
browser task actually runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.agents.llm.model_catalog import get_openrouter_catalog
from app.config.settings import settings
from app.constants.llm import DEFAULT_MODEL_NAME
from app.services.browser.exceptions import BrowserUnavailableError

if TYPE_CHECKING:
    from browser_use.llm.base import BaseChatModel


# OpenRouter is OpenAI-wire-compatible; Browser-Use talks to it via ChatOpenAI.
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Providers whose chat model cannot take image input; forces use_vision off.
# Browser-Use's DeepSeek wrapper (api.deepseek.com) is text-only; an
# OpenRouter-routed DeepSeek model is judged by the live catalog instead.
_TEXT_ONLY_PROVIDERS = frozenset({"deepseek"})


def _custom_lane_configured() -> bool:
    """Whether the shared custom LLM endpoint (DEV_LLM_*) comms uses is set.

    Mirrors the same three-setting check the comms default LLM makes
    (agents/llm/client.py); the settings are the single source of truth.
    """
    return bool(settings.DEV_LLM_BASE_URL and settings.DEV_LLM_API_KEY and settings.DEV_LLM_MODEL)


def _resolve_browser_lane() -> tuple[str, str, str | None, str | None]:
    """Return (provider, model, api_key, base_url) for the browser LLM.

    Without a browser-specific key this inherits the comms lane (DEV_LLM_*);
    an explicit BROWSER_USE_LLM_API_KEY selects a separate lane.
    """
    # An explicit browser-specific key opts into a deliberately separate lane.
    if settings.BROWSER_USE_LLM_API_KEY:
        provider = settings.BROWSER_USE_LLM_PROVIDER.lower()
        return (
            provider,
            settings.BROWSER_USE_LLM_MODEL,
            settings.BROWSER_USE_LLM_API_KEY,
            settings.BROWSER_USE_LLM_BASE_URL,
        )
    # Otherwise inherit exactly what comms runs on, so one key powers both:
    #   dev  → the custom endpoint (DEV_LLM_*)
    #   prod → OpenRouter + the default chat model (get_default_llm's fallback)
    if _custom_lane_configured():
        return "openai", settings.DEV_LLM_MODEL, settings.DEV_LLM_API_KEY, settings.DEV_LLM_BASE_URL
    return "openrouter", DEFAULT_MODEL_NAME, settings.OPENROUTER_API_KEY, None


def build_browser_llm() -> BaseChatModel:
    """Build the Browser-Use chat model for the configured provider.

    Raises :class:BrowserUnavailableError when the provider is unknown or its
    API key is missing, so the tool reports a clean reason instead of the agent
    failing deep inside a run.
    """
    provider, model, api_key, override_base_url = _resolve_browser_lane()
    if not api_key:
        raise BrowserUnavailableError(
            f"Browser agent LLM not configured: set an API key for provider '{provider}'."
        )

    if provider == "anthropic":
        from browser_use import ChatAnthropic  # noqa: PLC0415 -- heavy optional dep

        return ChatAnthropic(model=model, api_key=api_key)

    if provider == "google":
        from browser_use import ChatGoogle  # noqa: PLC0415 -- heavy optional dep

        return ChatGoogle(model=model, api_key=api_key)

    if provider == "deepseek":
        # Browser-Use's DeepSeek wrapper is NOT re-exported from the top-level
        # `browser_use` package (absent from its `_LAZY_IMPORTS`/`__all__`, unlike
        # ChatOpenAI/ChatGoogle/ChatAnthropic) — import it from its real module.
        from browser_use.llm.deepseek.chat import ChatDeepSeek  # noqa: PLC0415 -- heavy dep

        # Only override base_url when set — ChatDeepSeek's own default already
        # points at DeepSeek's official endpoint.
        if override_base_url:
            return ChatDeepSeek(model=model, api_key=api_key, base_url=override_base_url)
        return ChatDeepSeek(model=model, api_key=api_key)

    if provider in ("openai", "openrouter"):
        from browser_use import ChatOpenAI  # noqa: PLC0415 -- heavy optional dep

        base_url = override_base_url or (_OPENROUTER_BASE_URL if provider == "openrouter" else None)
        # When the endpoint cannot serve `json_schema`, hand Browser-Use its own
        # fallback: schema in the system prompt, plain-text JSON parsed back.
        # Vision is unaffected — image input alone is fine on those lanes.
        schema_in_prompt = settings.BROWSER_USE_LLM_SCHEMA_IN_PROMPT
        kwargs: dict[str, Any] = {
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
            "add_schema_to_system_prompt": schema_in_prompt,
            "dont_force_structured_output": schema_in_prompt,
        }
        effort = settings.BROWSER_USE_LLM_REASONING_EFFORT
        if effort:
            # Naming this model in `reasoning_models` is what makes Browser-Use
            # forward `reasoning_effort` at all — its check is a substring match
            # against the model name, not a capability lookup.
            kwargs["reasoning_effort"] = effort
            kwargs["reasoning_models"] = [model]
        return ChatOpenAI(**kwargs)

    raise BrowserUnavailableError(f"Unknown browser LLM provider: '{provider}'.")


async def resolve_use_vision() -> bool:
    """Whether the browser agent should be sent step screenshots.

    BROWSER_USE_VISION is the operator switch; it is additionally forced off
    when the configured provider/model cannot take images. OpenRouter models
    are checked against the live model catalog rather than curated here.
    """
    if not settings.BROWSER_USE_VISION:
        return False

    # Same lane the agent will actually run on (inherits the custom endpoint when
    # no browser-specific key is set), so the vision check judges the real model.
    provider, model, _api_key, _base_url = _resolve_browser_lane()
    if provider in _TEXT_ONLY_PROVIDERS:
        return False

    # An openrouter-style id ("vendor/model") is judged by the live catalog, so a
    # text-only model inherited from comms turns vision off by itself. A bare id
    # (an OpenAI model like gpt-4o) is assumed vision-capable.
    if provider == "openrouter" or "/" in model:
        catalog = await get_openrouter_catalog()
        return await catalog.accepts_images(model)

    return True
