"""Shared types for the LLM client layer."""

from collections.abc import Callable

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableSerializable
from typing_extensions import TypedDict

from app.constants.llm import LLMProviderName

# configurable_fields() returns a RunnableConfigurableFields, not a BaseChatModel,
# and only RunnableSerializable declares configurable_alternatives().
ProviderLLM = RunnableSerializable[LanguageModelInput, AIMessage]


class LLMProvider(TypedDict):
    name: LLMProviderName
    instance: ProviderLLM


# A fallback may be passed as a ready runnable or as a zero-arg factory, so
# expensive preparation (e.g. re-binding the full tool list) only happens in
# the rare case the primary actually fails. Parametrized to match what callers
# actually build: bind_tools() returns Runnable[LanguageModelInput, AIMessage].
LLMFallback = (
    Runnable[LanguageModelInput, AIMessage]
    | Callable[[], Runnable[LanguageModelInput, AIMessage] | None]
    | None
)
