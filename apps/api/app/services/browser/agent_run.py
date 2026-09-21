"""Drive one Browser-Use Agent over the session's CDP endpoint.

Jev is its model; every executed step is reported back to the runner through
RunHooks, and the agent's history is read into a RunOutcome.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from app.constants.browser import (
    BROWSER_TAKEOVER_PREAMBLE,
    BROWSER_VIEWPORT_HEIGHT,
    BROWSER_VIEWPORT_WIDTH,
)
from app.constants.log_tags import LogTag
from app.schemas.browser import BrowserAction, BrowserActionOutput
from app.services.browser.captions import caption_from_action_list
from app.services.browser.exceptions import BrowserUnavailableError
from app.services.browser.jev import JevChatModel
from app.services.browser.run_contract import (
    BrowserRunConfig,
    RunHooks,
    RunOutcome,
    RunUsage,
    StepClock,
    StepFrame,
)
from app.services.browser.session import BrowserHostSession
from app.services.browser.tools import build_browser_tools
from shared.py.wide_events import log

if TYPE_CHECKING:
    from browser_use.agent.views import AgentHistoryList, AgentOutput
    from browser_use.browser.views import BrowserStateSummary
    from browser_use.llm.base import BaseChatModel
    from pydantic import BaseModel


# Attributes worth naming an otherwise-unlabelled control by, in the order a
# person would recognise it. `value` covers <input type="submit" value="Submit">.
_LABEL_ATTRIBUTES = ("aria-label", "value", "title", "placeholder", "alt", "name", "id")

_OUTPUT_MAX_CHARS = 1000

#: Caption for a step that produced no action to describe — one whose actions
#: errored or whose observation stalled. It lives here rather than in
#: captions.py, which only ever names an action the agent actually chose.
STEP_ERROR_CAPTION = "That didn't respond, trying again"


class _ActionInputs(TypedDict, total=False):
    """One Browser-Use action's arguments, read only for the element it targets."""

    index: int


def _element_label(state: BrowserStateSummary, index: object) -> str | None:
    """Return the on-page name of the element an action targets, by its DOM index.

    Prefer the accessibility name (populated even for icon-only buttons), then
    visible text, then labelling attributes, then the tag name, so a control is
    never described as a bare verb when anything identifies it.
    """
    if not isinstance(index, int):
        return None
    selector_map = getattr(getattr(state, "dom_state", None), "selector_map", None) or {}
    node = selector_map.get(index)
    if node is None:
        return None
    try:
        ax_node = getattr(node, "ax_node", None)
        candidates = [getattr(ax_node, "name", None), node.get_meaningful_text_for_llm()]
        attributes = getattr(node, "attributes", None) or {}
        candidates += [attributes.get(attr) for attr in _LABEL_ATTRIBUTES]
        for candidate in candidates:
            text = (candidate or "").strip()
            if text:
                return text
        # Last resort: the tag itself ("Clicking BUTTON" still beats "Clicking").
        tag = (getattr(node, "node_name", "") or "").strip()
        return tag.lower() or None
    except Exception as exc:
        # A DOM node shape we don't recognise must not kill the step — the caption
        # just loses this element's name. Logged so a systematic shape change shows up.
        log.warning(
            f"{LogTag.BROWSER} Could not resolve element label from DOM node",
            error_type=type(exc).__name__,
        )
        return None


def _extract_actions(
    agent_output: AgentOutput,
    state: BrowserStateSummary | None = None,
    points: dict[int, tuple[float, float]] | None = None,
) -> list[BrowserAction]:
    """Return the step's actions as the agent's own tool calls — name, arguments, and the on-page text of whatever each one targets."""
    actions: list[BrowserAction] = []
    for action in getattr(agent_output, "action", None) or []:
        dumped = action.model_dump(exclude_none=True) if hasattr(action, "model_dump") else {}
        for action_name, params in dumped.items():
            raw_inputs = params if isinstance(params, dict) else {}
            typed_inputs: _ActionInputs = raw_inputs
            index = typed_inputs.get("index")
            target = _element_label(state, index) if state is not None else None
            # The centre the page itself reported for this element this step; the
            # snapshot's own boxes are fabricated on some engines (see jev/viewport.py).
            point = (points or {}).get(index) if isinstance(index, int) else None
            actions.append(
                BrowserAction(name=action_name, inputs=raw_inputs, target=target, point=point)
            )
    return actions


def _summarize_action_result(result: object) -> str | None:
    """One action's outcome as short display text, or None when there is nothing worth showing (a click that succeeded silently needs no output row)."""
    error = getattr(result, "error", None)
    if error:
        text = str(error)
    else:
        text = str(
            getattr(result, "extracted_content", None)
            or getattr(result, "long_term_memory", None)
            or ""
        )
    collapsed = " ".join(text.split())
    if not collapsed:
        return None
    return (
        collapsed
        if len(collapsed) <= _OUTPUT_MAX_CHARS
        else collapsed[: _OUTPUT_MAX_CHARS - 1].rstrip() + "…"
    )


def outcome_from_history(history: AgentHistoryList[BaseModel]) -> RunOutcome:
    """Return what the agent's history says the run achieved, and what it cost."""
    # Fallbacks kept in place, not an early return: an early return discards a
    # final_result() read before a later failure. Mutation-exempt because every
    # consumer collapses falsy values to one answer, so no other falsy value differs.
    final = None  # pragma: no mutate
    is_done = False  # pragma: no mutate
    is_successful: bool | None = None  # pragma: no mutate
    try:
        final = history.final_result()
        is_done = history.is_done()
        is_successful = history.is_successful()
    except Exception as exc:
        log.warning(
            f"{LogTag.BROWSER} Could not read browser history result",
            error_type=type(exc).__name__,
        )

    success = bool(is_done and is_successful is not False)
    summary = final or (
        "Completed the browser task." if success else "Could not complete the browser task."
    )
    return RunOutcome(success=success, summary=str(summary), usage=_usage_from_history(history))


def _usage_from_history(history: AgentHistoryList[BaseModel]) -> list[RunUsage]:
    """Return Browser-Use's per-model token totals under the names it billed them.

    Populated only when Agent.run returns normally; timeout, cancellation and
    CDP failure never reach a history.
    """
    usage = history.usage
    if usage is None:
        return []
    return [
        RunUsage(
            model_name=model_name,
            input_tokens=stats.prompt_tokens,
            output_tokens=stats.completion_tokens,
        )
        for model_name, stats in usage.by_model.items()
    ]


class BrowserAgentRun:
    """Run one Browser-Use Agent that decides and executes this task's steps."""

    def __init__(
        self,
        *,
        session: BrowserHostSession,
        llm: BaseChatModel | None,
        config: BrowserRunConfig,
        hooks: RunHooks,
        step_timeout: float,
    ) -> None:
        self._session = session
        self._llm = llm
        self._config = config
        self._hooks = hooks
        self._step_timeout = step_timeout
        self._agent: Any = None
        self._clock = StepClock()
        self._last_step = 0
        self._frames = 0
        self._framed = False

    async def execute(self, task: str) -> RunOutcome:
        from browser_use import Agent, Browser  # noqa: PLC0415 -- heavy optional dep

        if self._llm is None:
            raise BrowserUnavailableError("The Browser-Use agent needs a chat model to drive it.")

        browser = Browser(
            cdp_url=self._session.cdp_url,
            viewport={"width": BROWSER_VIEWPORT_WIDTH, "height": BROWSER_VIEWPORT_HEIGHT},
            # The live-view surface DPR is set host-side (screencast.py); Browser-Use
            # ignores device_scale_factor when connecting over CDP, so leave it at 1 here.
            device_scale_factor=1,
            no_viewport=False,
        )
        # Stealth fingerprinting is injected on every page by browser_use_stealth_patch
        # (app/patches), which hooks Browser-Use's per-target CDP session accessor so
        # new tabs are covered too — no per-run registration needed here.

        agent_kwargs: dict[str, Any] = {
            "task": task + BROWSER_TAKEOVER_PREAMBLE,
            "llm": self._llm,
            "browser": browser,
            # Browser-Use's prompt suggests todo.md for long tasks, but small
            # models write it even for 3-step forms — a measured ~7s and one
            # whole step of pure bookkeeping. Steer it to act directly.
            "extend_system_message": (
                "Do NOT create or update todo.md (or any planning file) unless the task "
                "genuinely needs more than 10 steps. For short tasks, act on the page "
                "directly from the first step."
            ),
            "register_new_step_callback": self._on_step,
            "register_should_stop_callback": self._hooks.should_stop,
            # Jev decides from structured state; no model on this path takes images.
            "use_vision": False,
            # Browser-Use's post-run judge bills a whole extra call and nothing
            # here reads its verdict; it also judged against an instruction the
            # user had already cancelled and logged the run as a failure.
            "use_judge": False,
            "flash_mode": self._config.flash_mode,
            "max_actions_per_step": self._config.max_actions_per_step,
            "step_timeout": self._step_timeout,
            "tools": build_browser_tools(
                solve_captcha=self._config.solve_captcha,
                handle_takeover=self._takeover,
                handle_guidance=self._guidance,
            ),
        }
        if isinstance(self._llm, JevChatModel):
            # Jev reads the structured observation from the session itself, with the
            # raw task (not the takeover preamble) as its goal. Its text helper is the
            # extraction model so Browser-Use meters those tokens under their own name.
            self._llm.bind(browser, task, self._hooks.guidance_allowed)
            agent_kwargs["page_extraction_llm"] = self._llm.text_model
        self._agent = Agent(**agent_kwargs)

        history = await self._agent.run(
            max_steps=self._config.max_steps, on_step_end=self._on_step_end
        )
        return outcome_from_history(history)

    def stop(self) -> None:
        if self._agent is not None:
            self._agent.stop()

    async def _takeover(self, reason: str, category: str) -> str:
        """Hand the browser to the user, then give the note they left to both readers: Jev's own state, and the action result Browser-Use records for this step."""
        note = await self._hooks.takeover(reason, category)
        if isinstance(self._llm, JevChatModel):
            self._llm.note_from_user(note)
        return note or "The user finished that step in the live browser."

    async def _guidance(self, reason: str) -> str:
        """Ask the agent that started this run how to proceed, and give its instruction to both readers.

        Only reached when the gate already said an agent is there to answer; the
        hook raises rather than returning when none arrives, which ends the run.
        """
        if self._hooks.guidance is None or not isinstance(self._llm, JevChatModel):
            raise BrowserUnavailableError("This run has no agent to ask for guidance.")
        instruction = await self._hooks.guidance(self._llm.guidance_request(reason))
        self._llm.note_from_agent(instruction)
        return instruction

    def _emit_frame(
        self,
        *,
        goal: str,
        actions: list[BrowserAction],
        state: BrowserStateSummary | None = None,
        raw_screenshot: str | None = None,
    ) -> None:
        """Emit one frame under the next number the user sees.

        Numbered by frames emitted, never by Browser-Use's step counter: that one
        advances for steps that never reach a frame, so the user saw 2, 3, 4, 7.
        """
        self._frames += 1
        self._last_step = self._frames
        self._hooks.step(
            StepFrame(
                index=self._frames,
                goal=goal,
                actions=actions,
                url=getattr(state, "url", None),
                title=getattr(state, "title", None),
                raw_screenshot=raw_screenshot or getattr(state, "screenshot", None),
                since_prev_ms=self._clock.tick(),
            )
        )

    async def _on_step(
        self, browser_state_summary: BrowserStateSummary, agent_output: AgentOutput, n_steps: int
    ) -> None:
        """Fire after the model picks actions, before they execute."""
        del n_steps  # Browser-Use's counter; the frame's own number is what the user reads
        self._framed = True
        points = self._llm.viewport_points() if isinstance(self._llm, JevChatModel) else {}
        step_actions = _extract_actions(agent_output, browser_state_summary, points)
        raw_screenshot = (
            await self._llm.take_step_screenshot()
            if isinstance(self._llm, JevChatModel)
            else getattr(browser_state_summary, "screenshot", None)
        )
        # Never the model's own next_goal/thinking: Jev fills both with its raw
        # decision label ("CLICK [6] Log In"). The caption describes what the
        # step does, named after the element it resolved.
        self._emit_frame(
            goal=caption_from_action_list(step_actions),
            actions=step_actions,
            state=browser_state_summary,
            raw_screenshot=raw_screenshot,
        )

    async def _on_step_end(self, agent: object) -> None:
        """Mirror each executed action's result into the thread after a step ends.

        register_new_step_callback fires before the actions execute, so only
        on_step_end sees state.last_result (one entry per action, in order).
        Keyed by self._last_step so each output lands on the row _on_step emitted.
        """
        state = getattr(agent, "state", None)
        results = getattr(state, "last_result", None) or []
        framed, self._framed = self._framed, False
        if not framed and any(getattr(result, "error", None) for result in results):
            # The step died before the model picked anything (an action error, a
            # watchdog timeout on the observation), so nothing else will ever
            # speak for it and the user just watches the card sit there.
            self._emit_frame(goal=STEP_ERROR_CAPTION, actions=[])
        if self._hooks.action_results is None:
            return
        outputs = [
            BrowserActionOutput(position=position, output=text)
            for position, result in enumerate(results)
            if (text := _summarize_action_result(result))
        ]
        if outputs:
            await self._hooks.action_results(self._last_step, outputs)
