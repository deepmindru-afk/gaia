"""Jev as the Browser-Use agent's decision model.

Browser-Use asks its chat model for an AgentOutput every step; this class
answers by having Jev decide the operation and target from the current page
state instead of returning a completion, and using the small text model only
to write a typed value when the decision needs one. Everything downstream
(execution, step cards, takeover, history, replay) is unchanged; calls that
are not a step decision go straight to the text model.
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import replace
import json
import re
from time import perf_counter
from typing import TYPE_CHECKING, TypedDict, TypeVar, cast, get_args, overload

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import CoreSchema, core_schema

from app.config.settings import settings
from app.constants.browser import (
    BROWSER_GUIDANCE_MAX_ELEMENTS,
    BROWSER_GUIDANCE_PAGE_TEXT_MAX_CHARS,
    BROWSER_GUIDANCE_RECENT_ACTIONS,
    BROWSER_RUN_BLOCKED_SUMMARY,
    JEV_DONE_REASK_BUDGET,
    JEV_MIN_DONE_CONFIDENCE,
    JEV_TEXT_HELPER_RECENT_ACTIONS,
    JEV_TEXT_VALUE_MAX_CHARS,
    BrowserHandoffAction,
    JevNoteSource,
    JevOperation,
    SensitiveCategory,
)
from app.constants.log_tags import LogTag
from app.patches.browser_use_deferred_screenshot_patch import defer_screenshots_for
from app.schemas.browser import AgentGuidanceRequest, GuidanceAction, GuidanceElement
from app.services.browser.exceptions import BrowserUnavailableError
from app.services.browser.jev.gateway import JevGatewayClient, JevGatewayError
from app.services.browser.jev.live_values import read_live_values
from app.services.browser.jev.observation import JevElement, JevObservation, observe
from app.services.browser.jev.policy import (
    JevDecision,
    JevDecisionError,
    JevHistoryEntry,
    choose,
)
from app.services.browser.jev.prompts import (
    CAPTCHA_CHALLENGE,
    DONE_SUMMARY,
    GUIDANCE_REASON,
    TAKEOVER_REASON,
    TEXT_VALUE,
    URL_VALUE,
)
from app.services.browser.jev.seen_text import SeenText
from app.services.browser.jev.viewport import NodeHandles, ViewportBox, read_viewport
from app.services.browser.run_contract import GuidanceGate
from shared.py.wide_events import log

if TYPE_CHECKING:
    from browser_use.browser.session import BrowserSession
    from browser_use.llm.base import BaseChatModel
    from browser_use.llm.messages import BaseMessage
    from browser_use.llm.views import ChatInvokeCompletion

T = TypeVar("T", bound=BaseModel)


class _WaitAction(TypedDict, total=False):
    """One Browser-Use action, read only for whether it fell back to waiting."""

    wait: dict[str, object]


class _ActionField(TypedDict):
    """AgentOutput's fields, read only for the action model they carry."""

    action: FieldInfo


class _RootField(TypedDict, total=False):
    """A RootModel's fields, read only for the union member it wraps."""

    root: FieldInfo


# Browser-Use action name → the Jev operations it makes available. Text entry
# is `input` in Browser-Use 0.11 and `input_text` before it; both are known.
_OPERATIONS_BY_ACTION: dict[str, tuple[JevOperation, ...]] = {
    "click": (JevOperation.CLICK,),
    "input": (JevOperation.TYPE_TEXT,),
    "input_text": (JevOperation.TYPE_TEXT,),
    "select_dropdown": (JevOperation.SELECT,),
    "scroll": (JevOperation.SCROLL_UP, JevOperation.SCROLL_DOWN),
    "wait": (JevOperation.WAIT,),
    "navigate": (JevOperation.NAVIGATE,),
    "go_back": (JevOperation.GO_BACK,),
    BrowserHandoffAction.REQUEST_HUMAN_TAKEOVER: (JevOperation.REQUEST_HUMAN,),
    BrowserHandoffAction.SOLVE_CAPTCHA_WITH_HELP: (JevOperation.SOLVE_CAPTCHA,),
    "done": (JevOperation.DONE, JevOperation.BLOCKED),
}
_DEFAULT_TAKEOVER_REASON = "Complete this step in the live browser"
_DEFAULT_GUIDANCE_REASON = "No operation on this page moves the task forward."
_DEFAULT_CAPTCHA_CHALLENGE = "Solve the CAPTCHA, then continue"
_USER_REQUEST = re.compile(r"<user_request>\s*(.*?)\s*</user_request>", re.DOTALL)
# Neither ends the run on a real next action, so neither counts as an alternative
# to an unconfident DONE.
_TERMINAL_OPERATIONS = frozenset({JevOperation.DONE, JevOperation.BLOCKED})
_HANDOFF_OPERATIONS = frozenset({JevOperation.REQUEST_HUMAN, JevOperation.SOLVE_CAPTCHA})
_JEV_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
# Vercel AI Gateway evaluation endpoint: same {model, state, questions} body
# and {answers, usage} response as the OpenRouter decisions route.
_JEV_VERCEL_EVALUATE_URL = "https://ai-gateway.vercel.sh/v1/evaluate"


class _TextValue(BaseModel):
    text: str | None = None


class _TakeoverReason(BaseModel):
    text: str | None = None
    category: str = SensitiveCategory.IRREVERSIBLE.value


class JevChatModel:
    """Browser-Use BaseChatModel whose step decisions come from Jev."""

    _verified_api_keys = True

    #: The step photo in flight, rendered while the decision is; class-level so a
    #: model built without __init__ (the runner tests do) still answers None.
    _shot: asyncio.Task[str | None] | None = None

    def __init__(
        self, *, client: JevGatewayClient, text_model: BaseChatModel, provider: str = "openrouter"
    ) -> None:
        self.model = client.model
        self.text_model = text_model
        self._provider = provider
        self._client = client
        self._browser: BrowserSession | None = None
        self._task: str | None = None
        self._history: list[JevHistoryEntry] = []
        self._last_fingerprint: str | None = None
        self._steps = 0
        #: Actual gateway-reported spend across this run's decisions (Vercel
        #: reports cost 0 on free allowance); None the moment one decision
        #: arrives without cost metadata, so metering falls back to the table.
        self._gateway_cost_usd: float | None = 0.0
        self._viewport: dict[int, ViewportBox] = {}
        self._done_reasks_left = JEV_DONE_REASK_BUDGET
        self._guidance_allowed: GuidanceGate | None = None
        self._observation: JevObservation | None = None
        self._seen_text = SeenText()
        self._handles = NodeHandles()
        self._shot = None
        #: One-shot: guidance just arrived, so this next step may not give up on it.
        self._blocked_suppressed = False

    def viewport_points(self) -> dict[int, tuple[float, float]]:
        """Return the last observation's on-screen centres by Browser-Use index, for the UI pulse."""
        return {
            index: (round(box.cx, 4), round(box.cy, 4))
            for index, box in self._viewport.items()
            if box.on_screen
        }

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def actual_cost_usd(self) -> float | None:
        """Gateway-reported spend for this run's decisions, or None when any decision lacked cost metadata."""
        return self._gateway_cost_usd

    @property
    def name(self) -> str:
        return self.model

    @property
    def model_name(self) -> str:
        return self.model

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: type, handler: object) -> CoreSchema:
        # Browser-Use stores its model in pydantic settings; the Protocol asks for this.
        return core_schema.any_schema()

    def bind(
        self, browser: BrowserSession, task: str, guidance_allowed: GuidanceGate | None = None
    ) -> None:
        """Give the policy the session whose observations it decides on, the raw goal, and the gate that says whether a blocked step may ask the agent for guidance."""
        self._browser = browser
        self._task = task
        self._guidance_allowed = guidance_allowed
        defer_screenshots_for(browser)

    @overload
    async def ainvoke(
        self, messages: list[BaseMessage], output_format: None = None, **kwargs: object
    ) -> ChatInvokeCompletion[str]: ...

    @overload
    async def ainvoke(
        self, messages: list[BaseMessage], output_format: type[T], **kwargs: object
    ) -> ChatInvokeCompletion[T]: ...

    async def ainvoke(
        self, messages: list[BaseMessage], output_format: type[T] | None = None, **kwargs: object
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        if output_format is None or not _is_agent_output(output_format):
            return await self.text_model.ainvoke(messages, output_format, **kwargs)
        return await self._decide(messages, output_format)

    async def _decide(
        self, messages: list[BaseMessage], output_format: type[T]
    ) -> ChatInvokeCompletion[T]:
        from browser_use.llm.views import (  # noqa: PLC0415 -- heavy optional dep
            ChatInvokeCompletion,
            ChatInvokeUsage,
        )

        if self._browser is None:
            raise BrowserUnavailableError("Jev policy has no browser session bound.")
        t0 = perf_counter()
        state = await self._browser.get_browser_state_summary(cached=True, include_screenshot=False)
        t1 = perf_counter()
        selector_map = getattr(getattr(state, "dom_state", None), "selector_map", None) or {}
        self._handles.on_page(getattr(state, "url", None))
        screen = await read_viewport(self._browser, selector_map, self._handles)
        t2 = perf_counter()
        self._viewport = screen.boxes
        # The engine idles while Jev and the text helper think; render the step's
        # photo then, not inside the state read where it queued ahead of the DOM.
        self._shot = asyncio.create_task(self._capture_screenshot())
        live = await read_live_values(self._browser)
        t3 = perf_counter()
        observation = observe(state, live, screen)
        t4 = perf_counter()
        log.info(
            f"{LogTag.BROWSER} Jev step input built (step={self._steps + 1} "
            f"state={round((t1 - t0) * 1000)}ms viewport={round((t2 - t1) * 1000)}ms "
            f"live={round((t3 - t2) * 1000)}ms observe={round((t4 - t3) * 1000)}ms "
            f"elements={len(observation.elements)})",
            step=self._steps + 1,
            state_ms=round((t1 - t0) * 1000),
            viewport_ms=round((t2 - t1) * 1000),
            live_values_ms=round((t3 - t2) * 1000),
            observe_ms=round((t4 - t3) * 1000),
            elements=len(observation.elements),
        )
        self._observation = observation
        self._seen_text.record(observation.url, observation.text)
        self._settle_previous_step(observation)
        goal = self._effective_goal(messages)
        registered = _registered_actions(output_format)
        offered = _offered_operations(registered)
        if self._latest_note_from(JevNoteSource.USER):
            # The user answered a takeover with an instruction; handing the same step
            # back ignores what they said, then blames them when it times out. A later
            # agent note says how to proceed and never restores what they declined.
            offered -= _HANDOFF_OPERATIONS
        if self._blocked_suppressed:
            # The agent just said how to proceed; giving up on the same step would
            # spend a guidance round and never try what it said.
            offered -= {JevOperation.BLOCKED}
            self._blocked_suppressed = False
        self._steps += 1

        try:
            decision = await self._choose(observation, goal, offered)
        except JevDecisionError as exc:
            # Nothing executes on a malformed answer; a WAIT keeps the loop honest
            # and the failure shows up in Jev's next recent_actions. On the step
            # Browser-Use narrows to `done` only, the honest answer is a failed done.
            log.warning(f"{LogTag.BROWSER} Jev decision rejected", error_type=type(exc).__name__)
            self._remember("WAIT", "error", str(exc))
            fallback = _idle_action(output_format)
            return ChatInvokeCompletion(
                completion=_output(output_format, fallback, next(iter(fallback)).upper(), str(exc)),
                usage=None,
            )

        action, text = await self._action_for(decision, observation, goal, registered)
        wait_action: _WaitAction = cast(_WaitAction, action)
        kind = (
            "error"
            if wait_action.get("wait") and decision.operation is not JevOperation.WAIT
            else (decision.operation.value.lower())
        )
        self._remember(decision.label, kind, text)
        evaluation = decision.evaluation
        if self._gateway_cost_usd is not None:
            cost = evaluation.gateway_cost_usd if evaluation is not None else None
            self._gateway_cost_usd = (self._gateway_cost_usd + cost) if cost is not None else None
        log.info(
            f"{LogTag.BROWSER} Jev step decided (step={self._steps} "
            f"{decision.operation.value} p={decision.confidence:.2f} "
            f"{decision.evaluation.latency_ms if decision.evaluation else None}ms)",
            step=self._steps,
            provider=self._provider,
            operation=decision.operation.value,
            target=decision.target,
            confidence=round(decision.confidence, 3),
            latency_ms=decision.evaluation.latency_ms if decision.evaluation else None,
        )
        usage = decision.evaluation.usage if decision.evaluation else None
        return ChatInvokeCompletion(
            completion=_output(
                output_format,
                action,
                decision.label,
                f"Step {self._steps}: {decision.label} (p={decision.confidence:.2f})",
            ),
            usage=ChatInvokeUsage(
                prompt_tokens=usage.input_tokens,
                prompt_cached_tokens=None,
                prompt_cache_creation_tokens=None,
                prompt_image_tokens=None,
                completion_tokens=usage.output_tokens,
                total_tokens=usage.input_tokens + usage.output_tokens,
            )
            if usage
            else None,
        )

    async def _choose(
        self, observation: JevObservation, goal: str, offered: frozenset[JevOperation]
    ) -> JevDecision:
        """Ask Jev for this step, re-asking without DONE while the run's re-ask budget holds.

        A DONE under the floor ends the run on whatever page is showing, and the
        closing summary then reads like a confident answer to a goal never met.
        """
        decision = await choose(self._client, observation, goal, self._history, offered)
        if (
            decision.operation is not JevOperation.DONE
            or decision.confidence >= JEV_MIN_DONE_CONFIDENCE
            or self._done_reasks_left <= 0
            or not offered - _TERMINAL_OPERATIONS
        ):
            return decision
        log.info(
            f"{LogTag.BROWSER} Jev DONE below the confidence floor; re-asking without it",
            step=self._steps,
            confidence=round(decision.confidence, 3),
        )
        self._done_reasks_left -= 1
        alternative = await choose(
            self._client, observation, goal, self._history, offered - {JevOperation.DONE}
        )
        # A re-ask that surfaces only a less sure WAIT spends a whole step (about
        # 6s on a long page) to change nothing; the unsure DONE was the better read.
        return alternative if alternative.confidence >= decision.confidence else decision

    async def _action_for(
        self, decision: JevDecision, observation: JevObservation, goal: str, registered: set[str]
    ) -> tuple[dict[str, dict[str, object]], str | None]:
        """Return the Browser-Use action a decision executes as, plus any text the helper wrote."""
        action: tuple[dict[str, dict[str, object]], str | None] | None = None
        match decision.operation:
            case JevOperation.DONE | JevOperation.BLOCKED:
                action = await self._terminal_action(decision, observation, goal, registered)
            case JevOperation.REQUEST_HUMAN | JevOperation.SOLVE_CAPTCHA:
                action = await self._handoff_action(decision, observation, goal)
            case JevOperation.CLICK | JevOperation.TYPE_TEXT | JevOperation.SELECT:
                action = await self._element_action(decision, observation, goal, registered)
            case (
                JevOperation.SCROLL_UP
                | JevOperation.SCROLL_DOWN
                | JevOperation.WAIT
                | JevOperation.GO_BACK
                | JevOperation.NAVIGATE
            ):
                action = await self._control_action(decision, observation, goal)
        if action is None:
            raise JevDecisionError(f"Jev chose {decision.operation.value} without a usable target.")
        return action

    async def _terminal_action(
        self, decision: JevDecision, observation: JevObservation, goal: str, registered: set[str]
    ) -> tuple[dict[str, dict[str, object]], str | None]:
        if decision.operation is JevOperation.BLOCKED:
            return await self._blocked_action(goal, observation, registered)
        summary = await self._field_text(
            DONE_SUMMARY, goal, observation, None, seen_text=self._seen_text.text
        )
        return {"done": {"text": summary or "Completed the task.", "success": True}}, summary

    async def _blocked_action(
        self, goal: str, observation: JevObservation, registered: set[str]
    ) -> tuple[dict[str, dict[str, object]], str | None]:
        """Ask the agent that started the run how to proceed, or end the run failed when nobody can answer."""
        action = BrowserHandoffAction.REQUEST_AGENT_GUIDANCE
        if action not in registered or not await self._may_ask_for_guidance():
            return {"done": {"text": BROWSER_RUN_BLOCKED_SUMMARY, "success": False}}, None
        reason = await self._field_text(GUIDANCE_REASON, goal, observation, None)
        text = reason or _DEFAULT_GUIDANCE_REASON
        return {action: {"reason": text}}, text

    async def _may_ask_for_guidance(self) -> bool:
        return self._guidance_allowed is not None and await self._guidance_allowed()

    async def _handoff_action(
        self, decision: JevDecision, observation: JevObservation, goal: str
    ) -> tuple[dict[str, dict[str, object]], str | None]:
        if decision.operation is JevOperation.SOLVE_CAPTCHA:
            challenge = await self._field_text(CAPTCHA_CHALLENGE, goal, observation, None)
            text = challenge or _DEFAULT_CAPTCHA_CHALLENGE
            return {BrowserHandoffAction.SOLVE_CAPTCHA_WITH_HELP: {"challenge": text}}, text
        reason = await self._structured(_TakeoverReason, TAKEOVER_REASON, goal, observation, None)
        text = (reason.text if reason else None) or _DEFAULT_TAKEOVER_REASON
        category = reason.category if reason else SensitiveCategory.IRREVERSIBLE.value
        return _takeover(text, category), text

    async def _element_action(
        self, decision: JevDecision, observation: JevObservation, goal: str, registered: set[str]
    ) -> tuple[dict[str, dict[str, object]], str | None] | None:
        """None when the decision named no usable element, which the caller rejects."""
        element = decision.element
        if element is None:
            return None
        match decision.operation:
            case JevOperation.CLICK:
                return {"click": {"index": element.browser_index}}, None
            case JevOperation.TYPE_TEXT:
                value = await self._field_text(TEXT_VALUE, goal, observation, element)
                if value is None:
                    # A value the goal did not supply is never invented; the human
                    # supplies it instead, per the takeover policy.
                    return _takeover(f"Enter the {element.label}"), None
                input_action = "input" if "input" in registered else "input_text"
                return {
                    input_action: {"index": element.browser_index, "text": value, "clear": True}
                }, value
            case JevOperation.SELECT if decision.option is not None:
                return {
                    "select_dropdown": {
                        "index": element.browser_index,
                        "text": decision.option.label,
                    }
                }, decision.option.label
        return None

    async def _control_action(
        self, decision: JevDecision, observation: JevObservation, goal: str
    ) -> tuple[dict[str, dict[str, object]], str | None] | None:
        """None when the operation is not one of the controls, which the caller rejects."""
        match decision.operation:
            case JevOperation.SCROLL_UP:
                return {"scroll": {"down": False, "pages": 1.0}}, None
            case JevOperation.SCROLL_DOWN:
                return {"scroll": {"down": True, "pages": 1.0}}, None
            case JevOperation.WAIT:
                return {"wait": {"seconds": 1}}, None
            case JevOperation.GO_BACK:
                return {"go_back": {}}, None
            case JevOperation.NAVIGATE:
                url = await self._field_text(URL_VALUE, goal, observation, None)
                if not url or not url.lower().startswith(("http://", "https://")):
                    return {
                        "wait": {"seconds": 1}
                    }, "NAVIGATE needs a URL the goal implies; none found"
                return {"navigate": {"url": url, "new_tab": False}}, url
        return None

    async def _field_text(
        self,
        instructions: str,
        goal: str,
        observation: JevObservation,
        field: JevElement | None,
        seen_text: str | None = None,
    ) -> str | None:
        answer = await self._structured(
            _TextValue, instructions, goal, observation, field, seen_text
        )
        value = answer.text if answer else None
        if not value or not value.strip() or len(value) > JEV_TEXT_VALUE_MAX_CHARS:
            return None
        return value

    async def _structured(
        self,
        output: type[T],
        instructions: str,
        goal: str,
        observation: JevObservation,
        field: JevElement | None,
        seen_text: str | None = None,
    ) -> T | None:
        """Goal, field, page and recent actions in; one small JSON value out."""
        from browser_use.llm.messages import (  # noqa: PLC0415 -- heavy optional dep
            SystemMessage,
            UserMessage,
        )

        context: dict[str, object] = {
            "goal": goal,
            "field": {"label": field.label, "role": field.role, "value": field.value}
            if field
            else None,
            "page": {"title": observation.title, "url": observation.url, "text": observation.text},
            "recent_actions": [
                {"action": h.action, "text": h.text}
                for h in self._history[-JEV_TEXT_HELPER_RECENT_ACTIONS:]
            ],
            "latest_note": self._latest_note(),
        }
        if seen_text:
            context["seen_on_this_page"] = seen_text
        t0 = perf_counter()
        try:
            result = await self.text_model.ainvoke(
                [SystemMessage(content=instructions), UserMessage(content=json.dumps(context))],
                output,
            )
        except Exception as exc:
            log.warning(f"{LogTag.BROWSER} Jev text helper failed", error_type=type(exc).__name__)
            return None
        log.info(
            f"{LogTag.BROWSER} Jev text helper answered (step={self._steps} "
            f"text_ms={round((perf_counter() - t0) * 1000)})",
            step=self._steps,
            text_ms=round((perf_counter() - t0) * 1000),
        )
        return result.completion

    def note_from_user(self, note: str | None) -> None:
        """Attach what the user said when handing the browser back to the step that asked.

        The takeover step is the last history entry: _remember runs inside
        _decide, before Browser-Use executes the action that blocks on the human.
        """
        self._attach_note(note, JevNoteSource.USER)

    def note_from_agent(self, note: str | None) -> None:
        """Attach the instruction the agent that started the run sent back to the blocked step that asked for it."""
        self._attach_note(note, JevNoteSource.AGENT)
        self._blocked_suppressed = bool(note)

    def _attach_note(self, note: str | None, source: JevNoteSource) -> None:
        if not self._history:
            raise RuntimeError("No step to attach a note to")
        self._history[-1] = replace(
            self._history[-1], note=note, note_source=source if note else None
        )

    async def take_step_screenshot(self) -> str | None:
        """Return the photo captured for the step being decided, base64 PNG, or None."""
        if self._shot is None:
            return None
        shot, self._shot = self._shot, None
        return await shot

    async def _capture_screenshot(self) -> str | None:
        if self._browser is None:
            return None
        try:
            return base64.b64encode(await self._browser.take_screenshot()).decode()
        except Exception as exc:  # a missing photo must never cost the step
            log.warning(
                f"{LogTag.BROWSER} Jev step screenshot failed", error_type=type(exc).__name__
            )
            return None

    def guidance_request(self, reason: str) -> AgentGuidanceRequest:
        """Return what the blocked step shows the agent: the page it is on, what it can see, and what it just tried."""
        observation = self._observation
        return AgentGuidanceRequest(
            reason=reason,
            task=self._task or "",
            url=observation.url if observation else "",
            title=observation.title if observation else "",
            page_text=observation.text[:BROWSER_GUIDANCE_PAGE_TEXT_MAX_CHARS]
            if observation
            else "",
            elements=[
                GuidanceElement(index=e.index, label=e.label, role=e.role)
                for e in (observation.elements if observation else ())[
                    :BROWSER_GUIDANCE_MAX_ELEMENTS
                ]
            ],
            recent_actions=[
                GuidanceAction(action=h.action, page_changed=h.page_changed)
                for h in self._history[-BROWSER_GUIDANCE_RECENT_ACTIONS:]
            ],
            user_notes=[
                h.note for h in self._history if h.note and h.note_source is JevNoteSource.USER
            ],
        )

    def _settle_previous_step(self, observation: JevObservation) -> None:
        if self._history and self._last_fingerprint is not None:
            # replace(), not a rebuild: a note attached to this step must survive.
            self._history[-1] = replace(
                self._history[-1],
                page_changed=observation.fingerprint != self._last_fingerprint,
            )
        self._last_fingerprint = observation.fingerprint

    def _effective_goal(self, messages: list[BaseMessage]) -> str:
        """Return the goal to decide and answer against: what the user changed, how to proceed, then the task.

        The user's instruction leads and says it overrides -- appended at the end
        it read as an aside, and the closing answer reported the original task as
        unfinished instead. Agent guidance only says how to proceed, so it never
        displaces what the user asked for.
        """
        goal = self._task or _goal_from_messages(messages)
        user_note = self._latest_note_from(JevNoteSource.USER)
        agent_note = self._latest_note_from(JevNoteSource.AGENT)
        if not user_note and not agent_note:
            return goal
        lines = []
        if user_note:
            lines.append(
                f"Latest instruction from the user, which overrides the task below: {user_note}"
            )
        if agent_note:
            lines.append(
                "Guidance from the assistant that planned this task, on how to proceed: "
                f"{agent_note}"
            )
        lines.append(f"{'Original task' if user_note else 'Task, which still stands'}: {goal}")
        return "\n".join(lines)

    def _latest_note_from(self, source: JevNoteSource) -> str | None:
        return next(
            (h.note for h in reversed(self._history) if h.note and h.note_source is source), None
        )

    def _latest_note(self) -> str | None:
        return next((h.note for h in reversed(self._history) if h.note), None)

    def _remember(self, action: str, kind: str, text: str | None) -> None:
        self._history.append(JevHistoryEntry(action=action, kind=kind, text=text))


def _takeover(
    reason: str, category: str = SensitiveCategory.IRREVERSIBLE.value
) -> dict[str, dict[str, object]]:
    return {BrowserHandoffAction.REQUEST_HUMAN_TAKEOVER: {"reason": reason, "category": category}}


def _idle_action(output_format: type[BaseModel]) -> dict[str, dict[str, object]]:
    """Return an action that changes nothing and is valid for this step's schema."""
    if "wait" in _registered_actions(output_format):
        return {"wait": {"seconds": 1}}
    return {"done": {"text": BROWSER_RUN_BLOCKED_SUMMARY, "success": False}}


def _is_agent_output(output_format: type[BaseModel]) -> bool:
    return "action" in getattr(output_format, "model_fields", {})


def _offered_operations(registered: set[str]) -> frozenset[JevOperation]:
    """Only operations whose Browser-Use action is registered for this run are offered."""
    return frozenset(
        op for name, ops in _OPERATIONS_BY_ACTION.items() if name in registered for op in ops
    )


def _registered_actions(output_format: type[BaseModel]) -> set[str]:
    """Return the action names in AgentOutput.action's element model.

    Browser-Use builds that model two ways: one model with an optional field per
    action, or (0.11+) a RootModel over a union of single-field models.
    """
    action_fields: _ActionField = cast(_ActionField, output_format.model_fields)
    action_model = get_args(action_fields["action"].annotation)[0]
    fields: _RootField = cast(_RootField, getattr(action_model, "model_fields", {}))
    if "root" not in fields:
        return set(fields)
    members = get_args(fields["root"].annotation) or (fields["root"].annotation,)
    return {name for member in members for name in getattr(member, "model_fields", {})}


def _output(
    output_format: type[T], action: dict[str, dict[str, object]], goal: str, memory: str
) -> T:
    return output_format.model_validate({"memory": memory, "next_goal": goal, "action": [action]})


def _goal_from_messages(messages: list[BaseMessage]) -> str:
    """Return the task as Browser-Use's own prompt carries it, for an unbound adapter."""
    for message in reversed(messages):
        text = getattr(message, "text", None)
        if not isinstance(text, str):
            continue
        match = _USER_REQUEST.search(text)
        if match:
            return match.group(1)
    return ""


def build_jev_chat_model(*, text_model: BaseChatModel) -> JevChatModel:
    """Return the Jev policy over the configured gateway, with text_model as its text helper.

    BROWSER_JEV_PROVIDER selects "openrouter" (default) or "vercel" (Vercel AI
    Gateway). Raises BrowserUnavailableError when the selected gateway's key
    is not configured.
    """
    provider = settings.BROWSER_JEV_PROVIDER
    if provider == "vercel":
        api_key = settings.BROWSER_JEV_VERCEL_API_KEY
        if not api_key:
            raise BrowserUnavailableError(
                "Jev provider is vercel but BROWSER_JEV_VERCEL_API_KEY is not set."
            )
        client = JevGatewayClient(
            api_key=api_key,
            model=settings.BROWSER_JEV_VERCEL_MODEL,
            url=_JEV_VERCEL_EVALUATE_URL,
        )
        return JevChatModel(client=client, text_model=text_model, provider="vercel")
    api_key = settings.OPENROUTER_API_KEY
    if not api_key:
        raise BrowserUnavailableError("Jev is enabled but OPENROUTER_API_KEY is not set.")
    client = JevGatewayClient(
        api_key=api_key,
        model=settings.BROWSER_USE_JEV_MODEL,
        url=_JEV_DECISIONS_URL,
    )
    return JevChatModel(client=client, text_model=text_model, provider="openrouter")


__all__ = ["JevChatModel", "JevGatewayError", "build_jev_chat_model"]
