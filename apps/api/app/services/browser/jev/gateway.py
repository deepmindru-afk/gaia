"""Vercel AI Gateway transport for Jev — the AI SDK's evaluation-model route.

The gateway serves ``POST {base}/evaluation-model`` for ``@ai-sdk/gateway`` only;
there is no OpenAI-compatible surface for evaluation models. This speaks that
route directly: the SDK's protocol headers, ``{state, questions}`` in, and
``{answers, usage}`` back. jev-ultrafast posts the same body straight to
TypeSafe's ``/v1/systemone``; only the envelope differs.
"""

from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from app.constants.browser import (
    JEV_GATEWAY_EVALUATION_SPEC_VERSION,
    JEV_GATEWAY_MAX_ATTEMPTS,
    JEV_GATEWAY_PROTOCOL_VERSION,
    JEV_GATEWAY_TIMEOUT_SECONDS,
)
from app.services.browser.exceptions import BrowserAutomationError

# A criterion or instruction: TypeSafe accepts a string or structured JSON.
JsonInput = str | dict[str, object] | list[object]

_RETRY_STATUSES = frozenset({429, 503, 529})


class JevGatewayError(BrowserAutomationError):
    """The gateway refused or failed the evaluation; no action was executed."""


class JevChoiceQuestion(BaseModel):
    """One ``choice`` question: pick an option name from ``criteria``."""

    type: Literal["choice"] = "choice"
    instructions: JsonInput
    criteria: dict[str, JsonInput]


class JevEvaluationRequest(BaseModel):
    state: dict[str, object]
    questions: dict[str, JevChoiceQuestion]


class JevChoiceAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float] = Field(default_factory=dict)


class JevUsage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    input_tokens: int = Field(default=0, alias="inputTokens")
    output_tokens: int = Field(default=0, alias="outputTokens")


class JevEvaluation(BaseModel):
    """The gateway's answer set plus what it cost and how long it took."""

    model_config = ConfigDict(extra="ignore")

    answers: dict[str, JevChoiceAnswer]
    usage: JevUsage | None = None
    latency_ms: int = 0


class JevGatewayClient:
    """Async client for one gateway credential and model; safe to share per run."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self._url = f"{base_url.rstrip('/')}/evaluation-model"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "ai-gateway-protocol-version": JEV_GATEWAY_PROTOCOL_VERSION,
            "ai-gateway-auth-method": "api-key",
            "ai-evaluation-model-specification-version": JEV_GATEWAY_EVALUATION_SPEC_VERSION,
            "ai-model-id": model,
        }
        self._client = client or httpx.AsyncClient(timeout=JEV_GATEWAY_TIMEOUT_SECONDS)

    async def evaluate(self, request: JevEvaluationRequest) -> JevEvaluation:
        """POST the questions; retries transient 429/503/529 with backoff, like jev-ultrafast."""
        body = request.model_dump(mode="json")
        started = perf_counter()
        for attempt in range(JEV_GATEWAY_MAX_ATTEMPTS):
            try:
                response = await self._client.post(self._url, json=body, headers=self._headers)
            except httpx.HTTPError as exc:
                raise JevGatewayError(
                    f"Jev gateway connection failed: {exc}; no action executed."
                ) from exc
            if response.status_code in _RETRY_STATUSES and attempt < JEV_GATEWAY_MAX_ATTEMPTS - 1:
                await asyncio.sleep(0.5 * 2**attempt)
                continue
            if response.is_error:
                raise JevGatewayError(
                    f"Jev gateway returned HTTP {response.status_code}: "
                    f"{_error_message(response)}; no action executed."
                )
            evaluation = JevEvaluation.model_validate(response.json())
            evaluation.latency_ms = round((perf_counter() - started) * 1000)
            return evaluation
        raise JevGatewayError("Jev gateway unavailable; no action executed.")

    async def aclose(self) -> None:
        await self._client.aclose()


def _error_message(response: httpx.Response) -> str:
    try:
        error = response.json().get("error", {})
    except ValueError:
        return response.text[:200]
    return str(error.get("message") or error.get("type") or response.text[:200])
