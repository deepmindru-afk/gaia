"""The seam between the runner and whichever loop actually drives the browser.

``BrowserTaskRunner`` owns everything about a run that is not the stepping
itself: the progress card, the human handoff, cancellation, the budgets, the
metering, the replay link. A *lane* owns only "decide and execute the steps" —
Browser-Use's agent (``lanes/browser_use.py``) — and reaches back through
:class:`LaneHooks`.

A lane therefore never learns about SSE, Redis, bots or live-view links: it hands
the runner a :class:`StepFrame` per executed step, calls ``takeover`` when it
needs the human, asks ``should_stop`` between steps, and returns what the run
produced as a :class:`LaneOutcome`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Protocol

from app.schemas.browser import BrowserAction, BrowserActionOutput

# Per-action results, keyed to the step whose rows the thread mirror emitted.
ActionResultsFn = Callable[[int, list[BrowserActionOutput]], None]


@dataclass(frozen=True)
class BrowserRunConfig:
    """One browser run's tuning knobs — every field is a ``BROWSER_USE_*`` setting."""

    max_steps: int
    max_actions_per_step: int
    task_timeout_seconds: int
    step_timeout_seconds: int
    handoff_timeout_seconds: int
    stream_screenshots: bool
    solve_captcha: bool
    flash_mode: bool = True


@dataclass(frozen=True)
class StepFrame:
    """One executed step, captured off the lane's loop for a deferred emit."""

    index: int
    goal: str
    actions: list[BrowserAction]
    url: str | None
    title: str | None
    raw_screenshot: str | None
    since_prev_ms: int
    #: What ``raw_screenshot`` actually is. Browser-Use captures PNG; the
    #: ultrafast loop captures JPEG, and a frame stored under the wrong type is
    #: served under the wrong type.
    screenshot_media_type: str = "image/png"


@dataclass(frozen=True)
class LaneUsage:
    """One model's token spend over a run, under the name it is billed as."""

    model_name: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class LaneOutcome:
    """What a lane's loop produced, before the runner judges cancellation."""

    success: bool
    summary: str
    usage: list[LaneUsage] = field(default_factory=list)


@dataclass(frozen=True)
class LaneHooks:
    """The runner's side of the seam, as a lane sees it.

    ``step`` is deliberately synchronous: the runner schedules the emit (the
    screenshot upload is a CDN round-trip) so recording a step never taxes the
    lane's loop.
    """

    step: Callable[[StepFrame], None]
    takeover: Callable[[str, str], Awaitable[str]]
    should_stop: Callable[[], Awaitable[bool]]
    action_results: ActionResultsFn | None = None


class BrowserLane(Protocol):
    """One implementation of "decide and execute this task's steps"."""

    async def execute(self, task: str) -> LaneOutcome:
        """Run the task to a terminal state. Budgets are the runner's; this only
        raises what the runner maps onto a result."""
        ...

    def stop(self) -> None:
        """Tell the loop to stop — the runner calls this when a budget expires,
        so the loop is really told, not merely abandoned."""
        ...


class StepClock:
    """Wall-clock between steps — the lane's think + execute time, per step."""

    def __init__(self) -> None:
        self._last = 0.0

    def tick(self) -> int:
        """Milliseconds since the previous step; 0 for the first one."""
        now = perf_counter()
        elapsed = round((now - self._last) * 1000) if self._last else 0
        self._last = now
        return elapsed
