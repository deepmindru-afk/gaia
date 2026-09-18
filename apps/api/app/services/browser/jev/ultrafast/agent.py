"""The complete agent loop. Typed choices, observable state, bounded execution.

Port of jev-ultrafast's ``agent.py``. A step is one snapshot, one Jev request,
and — only when the chosen operation is TYPE_TEXT — one small text call. The
reference's ``command("tick"/"predict"/"act")`` string dispatch is three methods
here; the control flow, the freshness checks and the order of every mutation are
unchanged. Decisions are consumed before anything mutates, so a stale-page retry
can never double-click.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from app.constants.browser import (
    JEV_ULTRAFAST_MAX_STEPS,
    JEV_ULTRAFAST_NO_PROGRESS_STEPS,
    JevOperation,
    SensitiveCategory,
)
from app.services.browser.exceptions import BrowserAutomationError, BrowserHandoffCancelled
from app.services.browser.jev.gateway import JevGatewayClient
from app.services.browser.jev.prompts import CAPTCHA_CHALLENGE
from app.services.browser.jev.ultrafast.browser import StalePage, UltrafastBrowser
from app.services.browser.jev.ultrafast.model import (
    Action,
    JevTextHelper,
    JevUltrafastDecision,
    PageState,
    TextHelperCall,
    action_space,
    choose,
    field_context,
)

_TERMINAL = {"done", "blocked"}
_FINAL_CHOICES = {"DONE", "BLOCKED"}
# A handoff executes no browser input, so it is its own history kind: Jev sees it
# in recent_actions (and so does not ask for the same human twice), the no-progress
# stop still counts it, and the action budget — which bounds mutations — does not.
_HANDOFF_KIND = "handoff"

# ``(reason, category)`` in, the completed handoff out. The handler owns the live
# view, the message to the user, keeping the session alive and the wait; it raises
# :class:`BrowserHandoffCancelled` when the user cancelled or the wait timed out.
JevHandoffHandler = Callable[[str, SensitiveCategory], Awaitable[object]]


class JevRunStopped(BrowserAutomationError):
    """The run hit one of its budgets, or was asked to continue after it ended."""


@dataclass(slots=True)
class JevRunResult:
    """What a finished run produced: every executed step and how it ended."""

    status: str
    goal: str
    url: str
    elapsed_ms: int
    history: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    text_calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.status == "done"


@dataclass(slots=True)
class _PendingText:
    """A generated value survives a stale-page retry only while its entire
    helper input is unchanged, and is discarded after a successful mutation."""

    context: dict[str, Any]
    text: str
    helper: TextHelperCall


class JevUltrafastAgent:
    """One goal, one attached page, one decision per step."""

    def __init__(
        self,
        browser: UltrafastBrowser,
        page: PageState,
        *,
        goal: str,
        client: JevGatewayClient,
        text_helper: JevTextHelper,
        screenshots: bool = False,
        on_takeover: JevHandoffHandler | None = None,
        on_captcha: JevHandoffHandler | None = None,
    ) -> None:
        self.browser = browser
        self.client = client
        self.text_helper = text_helper
        self.screenshots = screenshots
        self.handoffs: dict[JevOperation, JevHandoffHandler] = {
            operation: handler
            for operation, handler in (
                (JevOperation.REQUEST_HUMAN, on_takeover),
                (JevOperation.SOLVE_CAPTCHA, on_captcha),
            )
            if handler is not None
        }
        self.goal = goal
        self.page = page
        self.decision: JevUltrafastDecision | None = None
        self.status = "ready"
        self.history: list[dict[str, Any]] = []
        self.decisions: list[dict[str, Any]] = []
        self.text_calls: list[dict[str, Any]] = []
        self.elapsed_ms = 0
        self.started_at: float | None = None
        self.pending_text: _PendingText | None = None

    @classmethod
    async def start(
        cls,
        cdp_url: str,
        goal: str,
        *,
        client: JevGatewayClient,
        text_helper: JevTextHelper,
        start_url: str | None = None,
        screenshots: bool = False,
        on_takeover: JevHandoffHandler | None = None,
        on_captcha: JevHandoffHandler | None = None,
    ) -> JevUltrafastAgent:
        task = goal.strip()
        if not task:
            raise ValueError("Supply a task")
        browser = await UltrafastBrowser.attach(cdp_url, start_url=start_url)
        try:
            page = await browser.observe(screenshot=screenshots)
        except BaseException:
            await browser.close()
            raise
        return cls(
            browser,
            page,
            goal=task,
            client=client,
            text_helper=text_helper,
            screenshots=screenshots,
            on_takeover=on_takeover,
            on_captcha=on_captcha,
        )

    @property
    def elements(self) -> list[dict[str, Any]]:
        """The indexed element table as Jev sees it for the current page."""
        return action_space(self.page["actions"])[0]

    async def tick(self) -> None:
        """One decision cycle. A page that moved under the decision costs one
        re-observation, never an action."""
        try:
            await self.predict()
            await self.act(self.page["fingerprint"])
        except StalePage:
            self.decision = None
            self.status = "ready"
            self.page = await self.browser.observe(screenshot=self.screenshots)
            self._mark_elapsed()

    async def predict(self) -> None:
        if self.started_at is None:
            self.started_at = perf_counter()
        if not await self.browser.fresh(self.page):
            self.page = await self.browser.observe(screenshot=self.screenshots)
        self.decision = None
        if self.status in _TERMINAL:
            raise JevRunStopped("This run has stopped.")
        if len(self.decisions) >= JEV_ULTRAFAST_MAX_STEPS * 2:
            raise JevRunStopped("Reached the run's model-call budget")
        decision = await choose(
            self.client, self.page, self.goal, self.history, self.handoffs.keys()
        )
        self.decision = decision
        self.decisions.append(
            {
                "choice": decision.choice,
                "operation": decision.operation,
                "target": decision.target,
                "confidence": decision.confidence,
                "latency_ms": decision.latency_ms,
                # Every decision is billed, including the terminal one that
                # executes nothing, so a run cannot be priced from history alone.
                "usage": decision.usage.model_dump() if decision.usage else {},
                "fingerprint": self.page["fingerprint"],
                "elapsed_ms": self._elapsed(),
            }
        )
        self.status = "predicted"

    async def act(self, fingerprint: str) -> None:
        decision, page = self.decision, self.page
        if decision is None or fingerprint != page["fingerprint"]:
            raise JevRunStopped("Observe and choose before acting")
        # Consume once, before any mutation or model call. A retry cannot double-click.
        self.decision = None
        selected = decision.choice
        if selected in _FINAL_CHOICES:
            await self._finish(selected, page)
            return
        if selected in self.handoffs:
            await self._hand_off(JevOperation(selected), decision, page)
            return
        action = next(a for a in page["actions"] if a["id"] == selected)
        if self._executed() >= JEV_ULTRAFAST_MAX_STEPS:
            self.status = "blocked"
            raise JevRunStopped(f"Stopped at the {JEV_ULTRAFAST_MAX_STEPS}-action budget")
        text, helper = await self._text_for(action, page)
        # UltrafastBrowser.act rechecks freshness immediately before input,
        # including after text generation.
        await self.browser.act(action, page, text=text)
        self.pending_text = None
        self._mark_elapsed()
        # Record execution before observing. A stale post-action observation
        # must not erase the action that already happened.
        self.history.append(self._entry(action, decision, text, helper, page))
        await self._settle(page)

    async def _hand_off(
        self, operation: JevOperation, decision: JevUltrafastDecision, page: PageState
    ) -> None:
        """Jev asked for the human. The handler owns the live view, the message to
        the user and the wait; the loop only re-observes what it hands back."""
        if not await self.browser.fresh(page):
            raise StalePage("Page changed before the handoff. Choose again.")
        reason, category, helper = await self._handoff_reason(operation, page)
        action: Action = {
            "id": operation.value,
            "kind": _HANDOFF_KIND,
            "label": f"{operation.value} {reason}",
        }
        self._mark_elapsed()
        self.history.append(self._entry(action, decision, reason, helper, page))
        try:
            await self.handoffs[operation](reason, category)
        except BrowserHandoffCancelled:
            # The user cancelled, or the wait for them ran out. Nothing else can
            # advance the goal without them, so the run ends the way BLOCKED does.
            self.status = "blocked"
            self._mark_elapsed()
            self.history[-1].update(elapsed_ms=self.elapsed_ms)
            return
        await self._settle(page)

    async def _handoff_reason(
        self, operation: JevOperation, page: PageState
    ) -> tuple[str, SensitiveCategory, TextHelperCall]:
        """Jev decides *that* a human is needed; the text helper writes what to tell
        them, under the same validation as a typed value."""
        context = field_context(self.goal, None, page, self.history)
        if operation is JevOperation.SOLVE_CAPTCHA:
            reason, helper = await self.text_helper.field_text(
                context, instructions=CAPTCHA_CHALLENGE
            )
            category = SensitiveCategory.NONE
        else:
            reason, category, helper = await self.text_helper.takeover_reason(context)
        self.text_calls.append(
            {
                "model": helper.model,
                "latency_ms": helper.latency_ms,
                "usage": helper.usage,
                "field": operation.value,
                "value": reason,
            }
        )
        return reason, category, helper

    async def _settle(self, page: PageState) -> None:
        """Observe the page the last history entry landed on, then judge progress."""
        self.page = await self.browser.observe(screenshot=self.screenshots)
        self._mark_elapsed()
        self.history[-1].update(
            page_changed=self.page["fingerprint"] != page["fingerprint"],
            url=self.page["url"],
            elapsed_ms=self.elapsed_ms,
        )
        self.status = "blocked" if self._no_progress() else "ready"

    async def run(self) -> JevRunResult:
        """Tick until the run reaches a terminal status, then report it."""
        while self.status not in _TERMINAL:
            await self.tick()
        return self.result()

    def result(self) -> JevRunResult:
        return JevRunResult(
            status=self.status,
            goal=self.goal,
            url=self.page["url"],
            elapsed_ms=self.elapsed_ms,
            history=self.history,
            decisions=self.decisions,
            text_calls=self.text_calls,
        )

    async def close(self) -> None:
        await self.browser.close()

    async def _finish(self, selected: str, page: PageState) -> None:
        if not await self.browser.fresh(page):
            self.status = "ready"
            raise StalePage("Page changed since the decision. Choose again.")
        self.status = "done" if selected == "DONE" else "blocked"
        self._mark_elapsed()

    async def _text_for(
        self, action: Action, page: PageState
    ) -> tuple[str | None, TextHelperCall | None]:
        """The value to type, generated only for ``fill`` and reused only for an
        identical retry context."""
        if action["kind"] != "fill":
            return None, None
        if not await self.browser.fresh(page):
            raise StalePage("Page changed before text generation. Choose again.")
        context = field_context(self.goal, action, page, self.history)
        pending = self.pending_text
        if pending is not None and pending.context == context:
            return pending.text, pending.helper
        text, helper = await self.text_helper.field_text(context)
        self.pending_text = _PendingText(context, text, helper)
        self.text_calls.append(
            {
                "model": helper.model,
                "latency_ms": helper.latency_ms,
                "usage": helper.usage,
                "field": action["label"],
                "value": text,
            }
        )
        return text, helper

    def _entry(
        self,
        action: Action,
        decision: JevUltrafastDecision,
        text: str | None,
        helper: TextHelperCall | None,
        page: PageState,
    ) -> dict[str, Any]:
        return {
            "step": len(self.history) + 1,
            "action": action["label"],
            "kind": action["kind"],
            "choice": decision.choice,
            "probability": decision.probabilities[decision.choice],
            "confidence": decision.confidence,
            "latency_ms": decision.latency_ms,
            "text": text,
            "text_helper": helper.model if helper else None,
            "text_latency_ms": helper.latency_ms if helper else 0,
            "operation": decision.operation,
            "target": decision.target,
            "page_changed": None,
            "url": page["url"],
            "usage": decision.usage.model_dump() if decision.usage else {},
            "executed_ms": self.elapsed_ms,
            "elapsed_ms": self.elapsed_ms,
        }

    def _executed(self) -> int:
        """Executed browser actions. A handoff mutates nothing, so it costs no budget."""
        return sum(1 for entry in self.history if entry.get("kind") != _HANDOFF_KIND)

    def _no_progress(self) -> bool:
        repeated = self.history[-JEV_ULTRAFAST_NO_PROGRESS_STEPS:]
        return len(repeated) == JEV_ULTRAFAST_NO_PROGRESS_STEPS and all(
            h["page_changed"] is False and h["kind"] != "wait" for h in repeated
        )

    def _elapsed(self) -> int:
        return round((perf_counter() - (self.started_at or perf_counter())) * 1000)

    def _mark_elapsed(self) -> None:
        self.elapsed_ms = self._elapsed()
