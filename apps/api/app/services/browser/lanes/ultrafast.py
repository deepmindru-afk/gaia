"""The ultrafast lane: the ported jev-ultrafast loop decides and executes steps.

Jev chooses one operation and one observed target per step and a small text model
writes only the values that get typed — no screenshots to the model, no per-step
chat completion. Measured at 1.65x the speed and 1/23rd the cost of the
Browser-Use lane on the same task.

This module is the adapter, not a second runner: it drives the loop one ``tick``
at a time so the runner can bound each step, stop between steps, and stream the
same :class:`~app.schemas.browser.BrowserStepSnapshot` the other lane emits. The
loop itself learns nothing about cards, bots or live view — it is handed the
runner's takeover callback and hands back history entries.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict

from app.constants.browser import JevOperation, SensitiveCategory
from app.schemas.browser import BrowserAction
from app.services.browser.captions import caption_from_action_list
from app.services.browser.jev.ultrafast import (
    JevRunResult,
    JevRunStopped,
    JevUltrafastAgent,
    build_jev_clients,
)
from app.services.browser.lanes.base import (
    BrowserRunConfig,
    LaneHooks,
    LaneOutcome,
    LaneUsage,
    StepClock,
    StepFrame,
)
from app.services.browser.session import BrowserHostSession

# The loop's only non-terminal status: a tick leaves "ready" when it can go again,
# and "done"/"blocked" when it cannot.
_READY = "ready"
_DONE = "done"

# The loop captures JPEG (jev/ultrafast/browser.py), not Browser-Use's PNG.
_SCREENSHOT_MEDIA_TYPE = "image/jpeg"

# One history entry's kind -> the action name the shared captions know it by, so
# both lanes describe the same step the same way (see services/browser/captions.py).
_ACTION_NAMES = {
    "click": "click",
    "fill": "input",
    "select": "select_dropdown",
    "scroll": "scroll",
    "wait": "wait",
}
_HANDOFF_ACTION_NAMES = {
    JevOperation.REQUEST_HUMAN.value: "request_human_takeover",
    JevOperation.SOLVE_CAPTCHA.value: "solve_captcha_with_help",
}

_SUCCESS_SUMMARY = "Completed the browser task."
_FAILURE_SUMMARY = "Could not complete the browser task."


class UltrafastLane:
    """The jev-ultrafast loop over the session's CDP endpoint, one tick per step."""

    def __init__(
        self,
        *,
        session: BrowserHostSession,
        config: BrowserRunConfig,
        hooks: LaneHooks,
        step_timeout: float,
    ) -> None:
        self._session = session
        self._config = config
        self._hooks = hooks
        self._step_timeout = step_timeout
        self._clock = StepClock()
        self._emitted = 0
        self._stopped = False

    async def execute(self, task: str) -> LaneOutcome:
        client, text_helper = build_jev_clients()
        agent = await JevUltrafastAgent.start(
            self._session.cdp_url,
            task,
            client=client,
            text_helper=text_helper,
            screenshots=self._config.stream_screenshots,
            on_takeover=self._hand_off,
            # No handler, no offer: Jev is never shown an operation the run
            # cannot service (see the loop's `handoffs`).
            on_captcha=self._hand_off if self._config.solve_captcha else None,
        )
        try:
            success, summary = await self._drive(agent)
            result = agent.result()
        finally:
            await agent.close()
            await client.aclose()
            await text_helper.aclose()
        return LaneOutcome(success=success, summary=summary, usage=_usage(result, client.model))

    def stop(self) -> None:
        self._stopped = True

    async def _hand_off(self, reason: str, category: SensitiveCategory) -> str:
        """Jev asked for the human; the runner owns the live view and the wait."""
        return await self._hooks.takeover(reason, category.value)

    async def _drive(self, agent: JevUltrafastAgent) -> tuple[bool, str]:
        """Tick to a terminal status, emitting every executed step and stopping
        between steps when the runner says so."""
        while agent.status == _READY and not self._stopped:
            if await self._hooks.should_stop():
                return False, "Browser task stopped."
            try:
                await asyncio.wait_for(agent.tick(), timeout=self._step_timeout)
            except JevRunStopped as exc:
                # A budget ran out mid-step. Whatever executed before it still
                # happened, so report it before reporting the stop.
                self._emit_steps(agent)
                return False, str(exc)
            self._emit_steps(agent)
        if self._stopped:
            return False, "Browser task stopped."
        success = agent.status == _DONE
        return success, _SUCCESS_SUMMARY if success else _FAILURE_SUMMARY

    def _emit_steps(self, agent: JevUltrafastAgent) -> None:
        """Every history entry the last tick appended, as a step frame each."""
        for entry in agent.history[self._emitted :]:
            action = _action(entry)
            self._hooks.step(
                StepFrame(
                    index=int(entry["step"]),
                    goal=caption_from_action_list([action]),
                    actions=[action],
                    url=entry.get("url"),
                    title=agent.page.get("title"),
                    raw_screenshot=agent.page.get("screenshot"),
                    since_prev_ms=self._clock.tick(),
                    screenshot_media_type=_SCREENSHOT_MEDIA_TYPE,
                )
            )
        self._emitted = len(agent.history)


def _action(entry: dict[str, object]) -> BrowserAction:
    """One history entry as the action that actually executed: the operation Jev
    chose, the value it typed, and the label of the element it landed on."""
    kind = str(entry.get("kind", ""))
    text = entry.get("text")
    label = str(entry.get("action", "")) or None
    if kind not in _ACTION_NAMES:
        # A handoff executes no browser input: its "target" is the person, and
        # the generated reason is what they were asked to do.
        return BrowserAction(
            name=_HANDOFF_ACTION_NAMES.get(str(entry.get("choice")), kind),
            inputs={"reason": text} if text else {},
            target=None,
        )
    return BrowserAction(
        name=_ACTION_NAMES[kind],
        inputs={"text": text} if text else {},
        target=label,
    )


def _usage(result: JevRunResult, jev_model: str) -> list[LaneUsage]:
    """Both models the lane bills: Jev per decision — including the terminal one,
    which executes nothing — and the text helper per generated value."""
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for decision in result.decisions:
        usage = decision.get("usage") or {}
        totals[jev_model][0] += int(usage.get("input_tokens", 0))
        totals[jev_model][1] += int(usage.get("output_tokens", 0))
    for call in result.text_calls:
        usage = call.get("usage") or {}
        model = str(call["model"])
        totals[model][0] += int(usage.get("prompt_tokens", 0))
        totals[model][1] += int(usage.get("completion_tokens", 0))
    return [
        LaneUsage(model_name=model, input_tokens=tokens[0], output_tokens=tokens[1])
        for model, tokens in totals.items()
    ]
