"""Jev makes the choice; a small chat model writes text only for TYPE_TEXT.

Port of jev-ultrafast's ``model.py``. ``action_space`` turns one snapshot's
action rows into the indexed element table Jev sees, ``choose`` asks for the
operation and every compatible target in one request, and ``validate_choice``
refuses anything that is not a well-formed distribution over exactly the
offered ids — an invalid answer executes nothing.

Transport differs from the reference and only there: decisions go through
``jev/gateway.py`` (OpenRouter's decisions endpoint on ``OPENROUTER_API_KEY``)
because we have no TypeSafe credential, and the text helper posts to
OpenRouter's OpenAI-compatible chat completions with the same key.
"""

from __future__ import annotations

import asyncio
from collections.abc import Collection
from dataclasses import dataclass, field
import json
import math
from time import perf_counter
from typing import Any

import httpx

from app.constants.browser import (
    JEV_PROBABILITY_SUM_TOLERANCE,
    JEV_RECENT_ACTIONS,
    JEV_TEXT_HELPER_RECENT_ACTIONS,
    JEV_TEXT_VALUE_MAX_CHARS,
    JEV_ULTRAFAST_TEXT_MAX_TOKENS,
    JEV_ULTRAFAST_TEXT_TIMEOUT_SECONDS,
    JevOperation,
    SensitiveCategory,
)
from app.services.browser.exceptions import BrowserAutomationError
from app.services.browser.jev.gateway import (
    JevChoiceAnswer,
    JevChoiceQuestion,
    JevEvaluationRequest,
    JevGatewayClient,
    JevUsage,
    JsonInput,
)
from app.services.browser.jev.prompts import (
    HUMAN_RULES,
    NEXT_ACTION,
    REQUEST_HUMAN_CRITERION,
    SOLVE_CAPTCHA_CRITERION,
    TAKEOVER_REASON,
    TARGET,
    TEXT_VALUE,
)

# One row of snapshot.js's action table, or one page state it returns. The shape
# is owned by the browser-side snapshot (see snapshot.js) and is deliberately
# heterogeneous — a row carries `value`/`current_value`/`checked` only when the
# observed control has them — so it stays JSON rather than a Python model.
Action = dict[str, Any]
PageState = dict[str, Any]

# snapshot.js action kind -> the operation Jev is offered for it.
_OPERATIONS = {"click": "CLICK", "fill": "TYPE_TEXT", "select": "SELECT"}

_OPERATION_LABELS = {
    "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
    "TYPE_TEXT": (
        "Enter or replace text in an editable field. "
        "A small LLM will supply the value from the goal."
    ),
    "SELECT": "Select an observed dropdown value.",
}

# The two operations GAIA adds on top of the reference's action space: controls
# with no target head, offered only when the caller supplied a handler for them.
_HANDOFF_LABELS = {
    JevOperation.REQUEST_HUMAN: REQUEST_HUMAN_CRITERION,
    JevOperation.SOLVE_CAPTCHA: SOLVE_CAPTCHA_CRITERION,
}

# Element attributes snapshot.js reports, carried into the table and the criteria.
_STATE_KEYS = ("role", "value", "checked", "selected", "expanded")
_CRITERION_KEYS = ("role", "checked", "selected", "expanded")

_RETRY_STATUSES = frozenset({429, 503, 529})
_TEXT_MAX_ATTEMPTS = 3


class JevUltrafastDecisionError(BrowserAutomationError):
    """Jev's answer did not name an offered operation/target; no action executed."""


class JevTextHelperError(BrowserAutomationError):
    """The text helper produced no usable field value; nothing was typed."""


@dataclass(frozen=True, slots=True)
class JevUltrafastDecision:
    """One decision cycle: the operation, the target it consumed, and what it cost."""

    choice: str
    operation: str
    target: str | None
    confidence: float
    probabilities: dict[str, float]
    operation_probabilities: dict[str, float]
    target_probabilities: dict[str, float] = field(default_factory=dict)
    target_confidence: float | None = None
    model: str = ""
    usage: JevUsage | None = None
    latency_ms: int = 0


@dataclass(frozen=True, slots=True)
class TextHelperCall:
    """What the text helper cost for one generated field value."""

    model: str
    latency_ms: int
    usage: dict[str, Any]


def validate_choice(answer: JevChoiceAnswer | None, ids: Collection[str]) -> JevChoiceAnswer:
    """The choice is offered, the distribution covers exactly the offered ids, sums
    to ~1, and the choice is its argmax. Anything else executes nothing."""
    offered = set(ids)
    probabilities = answer.probabilities if answer is not None else {}
    numbers = [*probabilities.values(), answer.confidence if answer is not None else None]
    valid = (
        answer is not None
        and bool(probabilities)
        and answer.choice in offered
        and set(probabilities) == offered
        and all(n is not None and math.isfinite(n) and 0 <= n <= 1 for n in numbers)
        and abs(sum(probabilities.values()) - 1) < JEV_PROBABILITY_SUM_TOLERANCE
        and probabilities[answer.choice] >= max(probabilities.values()) - 1e-6
    )
    if not valid or answer is None:
        raise JevUltrafastDecisionError("Invalid Jev response; no action executed.")
    return answer


def action_space(
    actions: list[Action],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Action]], dict[str, Action]]:
    """One index per observed element; each operation has its own valid target choices."""
    elements: list[dict[str, Any]] = []
    indices: dict[int, str] = {}
    targets: dict[str, dict[str, Action]] = {}
    controls: dict[str, Action] = {}
    for action in actions:
        kind = action["kind"]
        if kind not in _OPERATIONS:
            controls[action["id"].upper()] = action
            continue
        node = action["node"]
        if node not in indices:
            index = str(len(elements) + 1)
            indices[node] = index
            element = {k: action[k] for k in _STATE_KEYS if k in action}
            element.update(index=index, label=action["label"].split(" → ")[0], operations=[])
            if kind == "select":
                element["value"] = action.get("current_value", "")
                element["options"] = []
            elements.append(element)
        index = indices[node]
        operation = _OPERATIONS[kind]
        group = targets.setdefault(operation, {})
        element = elements[int(index) - 1]
        if operation not in element["operations"]:
            element["operations"].append(operation)
        target = index
        if kind == "select":
            target = f"{index}:{len(element['options']) + 1}"
            element["options"].append(
                {"index": target, "label": action["label"], "value": action["value"]}
            )
        group[target] = action
    return elements, targets, controls


def build_request(
    state: PageState,
    goal: str,
    history: list[dict[str, Any]],
    handoffs: Collection[JevOperation] = (),
) -> tuple[JevEvaluationRequest, dict[str, str], dict[str, dict[str, Action]], dict[str, Action]]:
    """The one request: the operation head plus a speculative head per operation.

    ``handoffs`` are the human-in-the-loop operations the caller can actually
    service; like DONE and BLOCKED they are controls with no target head, and an
    operation with no handler is never offered.
    """
    elements, targets, controls = action_space(state["actions"])
    operations = {key: _OPERATION_LABELS[key] for key in targets}
    operations.update({key: value["label"] for key, value in controls.items()})
    operations.update({op.value: _HANDOFF_LABELS[op] for op in handoffs})
    operations.update(
        DONE="Every requirement is visibly satisfied.",
        BLOCKED="No supported operation can progress.",
    )
    rules: JsonInput = [NEXT_ACTION, HUMAN_RULES] if handoffs else NEXT_ACTION
    questions: dict[str, JevChoiceQuestion] = {
        "operation": JevChoiceQuestion(
            criteria=dict(operations), instructions={"goal": goal, "rules": rules}
        )
    }
    for operation, candidates in targets.items():
        criteria: dict[str, JsonInput] = {
            index: {
                "element": f"[{index}] {a['label']}",
                "current_value": a.get("current_value", a.get("value", "")),
                **{k: a[k] for k in _CRITERION_KEYS if k in a},
            }
            for index, a in candidates.items()
        }
        questions[operation.lower() + "_target"] = JevChoiceQuestion(
            criteria=criteria,
            instructions={"goal": goal, "operation": operation, "rules": [NEXT_ACTION, TARGET]},
        )
    request = JevEvaluationRequest(
        state={
            "page": {k: state[k] for k in ("url", "title", "text")},
            "elements": elements,
            "recent_actions": [
                {k: h.get(k) for k in ("action", "kind", "text", "page_changed")}
                for h in history[-JEV_RECENT_ACTIONS:]
            ],
        },
        questions=questions,
    )
    return request, operations, targets, controls


async def choose(
    client: JevGatewayClient,
    state: PageState,
    goal: str,
    history: list[dict[str, Any]],
    handoffs: Collection[JevOperation] = (),
) -> JevUltrafastDecision:
    """Ask for the operation and every compatible target in one round trip."""
    request, operations, targets, controls = build_request(state, goal, history, handoffs)
    evaluation = await client.evaluate(request)
    operation_answer = validate_choice(evaluation.answers.get("operation"), operations)
    operation = operation_answer.choice
    target: str | None = None
    target_answer: JevChoiceAnswer | None = None
    probabilities: dict[str, float] = {}
    if operation in targets:
        # Unused target heads cannot cause an action. Validate the head the operation selected.
        target_answer = validate_choice(
            evaluation.answers.get(operation.lower() + "_target"), targets[operation]
        )
        target = target_answer.choice
        choice = targets[operation][target]["id"]
        probabilities = {
            a["id"]: target_answer.probabilities[index] for index, a in targets[operation].items()
        }
    else:
        choice = controls[operation]["id"] if operation in controls else operation
        probabilities[choice] = operation_answer.probabilities[operation]
    return JevUltrafastDecision(
        choice=choice,
        operation=operation,
        target=target,
        confidence=operation_answer.confidence or 0.0,
        probabilities=probabilities,
        operation_probabilities=operation_answer.probabilities,
        target_probabilities=target_answer.probabilities if target_answer else {},
        target_confidence=target_answer.confidence if target_answer else None,
        model=client.model,
        usage=evaluation.usage,
        latency_ms=evaluation.latency_ms,
    )


def field_context(
    goal: str, action: Action | None, page: PageState, history: list[dict[str, Any]]
) -> dict[str, Any]:
    """Everything the text helper is allowed to see. Identity of this dict is what
    makes a generated value reusable across a stale-page retry. ``action`` is None
    for the answers that are about the page rather than one field (a handoff reason)."""
    return {
        "goal": goal,
        "field": {k: action.get(k) for k in ("label", "role", "value")} if action else None,
        "page": {"title": page["title"], "text": page["text"][:6000]},
        "recent_actions": [
            {k: h.get(k) for k in ("action", "text")}
            for h in history[-JEV_TEXT_HELPER_RECENT_ACTIONS:]
        ],
    }


class JevTextHelper:
    """OpenAI-compatible chat completions for the one call TYPE_TEXT needs."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        url: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self._url = url
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = client or httpx.AsyncClient(timeout=JEV_ULTRAFAST_TEXT_TIMEOUT_SECONDS)

    async def field_text(
        self, context: dict[str, Any], *, instructions: str = TEXT_VALUE
    ) -> tuple[str, TextHelperCall]:
        """The value to type, or a refusal — the executor never guesses one itself.

        ``instructions`` swaps the prompt for the other single-``text`` answers
        this helper writes (a CAPTCHA challenge); the validation is the same.
        """
        result, call = await self._complete(instructions, context)
        return _parse_text_value(result), call

    async def takeover_reason(
        self, context: dict[str, Any]
    ) -> tuple[str, SensitiveCategory, TextHelperCall]:
        """The directive shown to the user, and why the step needs them."""
        result, call = await self._complete(TAKEOVER_REASON, context)
        text, category = _parse_takeover_reason(result)
        return text, category, call

    async def _complete(
        self, instructions: str, context: dict[str, Any]
    ) -> tuple[dict[str, Any], TextHelperCall]:
        started = perf_counter()
        result = await self._post(
            {
                "model": self.model,
                "max_tokens": JEV_ULTRAFAST_TEXT_MAX_TOKENS,
                "response_format": {"type": "json_object"},
                # Mercury (and the OpenRouter-routed models generally) take the
                # reasoning switch here; the reference's DeepSeek branch is the
                # one base URL we never point at.
                "reasoning": {"enabled": False},
                "messages": [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": json.dumps(context)},
                ],
            }
        )
        return result, TextHelperCall(
            model=self.model,
            latency_ms=round((perf_counter() - started) * 1000),
            usage=result.get("usage", {}),
        )

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(_TEXT_MAX_ATTEMPTS):
            try:
                response = await self._client.post(self._url, json=body, headers=self._headers)
            except httpx.HTTPError as exc:
                raise JevTextHelperError(
                    f"Text helper connection failed: {exc}; nothing typed."
                ) from exc
            if response.status_code in _RETRY_STATUSES and attempt < _TEXT_MAX_ATTEMPTS - 1:
                await asyncio.sleep(0.5 * 2**attempt)
                continue
            if response.is_error:
                raise JevTextHelperError(
                    f"Text helper returned HTTP {response.status_code}; nothing typed."
                )
            parsed: dict[str, Any] = response.json()
            return parsed
        raise JevTextHelperError("Text helper unavailable; nothing typed.")

    async def aclose(self) -> None:
        await self._client.aclose()


def _parse_text_value(result: dict[str, Any]) -> str:
    """Exactly one JSON key, ``text``, holding a non-empty in-budget string."""
    return _answer(result, {"text"})[0]


def _parse_takeover_reason(result: dict[str, Any]) -> tuple[str, SensitiveCategory]:
    """``text`` under the same rules, plus the category saying why a human is needed."""
    value, output = _answer(result, {"text", "category"})
    try:
        category = SensitiveCategory(output["category"])
    except ValueError:
        raise JevTextHelperError(
            "Text helper returned no valid field value; nothing typed."
        ) from None
    if category is SensitiveCategory.NONE:
        raise JevTextHelperError("Text helper returned no valid field value; nothing typed.")
    return value, category


def _answer(result: dict[str, Any], keys: set[str]) -> tuple[str, dict[str, Any]]:
    """The answer carries exactly ``keys`` and a non-empty in-budget ``text``."""
    try:
        output = json.loads(result["choices"][0]["message"]["content"])
        value = output["text"]
    except (ValueError, KeyError, TypeError, IndexError):
        raise JevTextHelperError(
            "Text helper returned no valid field value; nothing typed."
        ) from None
    if (
        set(output) != keys
        or not isinstance(value, str)
        or not value.strip()
        or len(value) > JEV_TEXT_VALUE_MAX_CHARS
    ):
        raise JevTextHelperError("Text helper returned no valid field value; nothing typed.")
    return value, output
