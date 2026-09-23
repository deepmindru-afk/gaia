"""Watch the primary engine under a run, and cut the run short once the engine stops answering.

Browser-Use learns of a frozen engine only through its own timeouts: a SIGSTOPped
engine cost a 15 s click timeout and two 120 s state reads, ~285 s, before the run
returned and the host was asked. The watchdog asks the host on a fixed beat
instead. The host's answer reflects whether the engine answers CDP at all, never
how long a page takes, so a heavy page on a live engine is left alone; and a run
paused on the user or the agent is never judged, since nothing steps while it waits.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from time import perf_counter

from app.constants.browser import (
    BROWSER_ENGINE_WATCH_CANCEL_GRACE_SECONDS,
    BROWSER_ENGINE_WATCH_INTERVAL_SECONDS,
    BROWSER_ENGINE_WATCH_STRIKES,
    EngineFailure,
)
from app.constants.log_tags import LogTag
from app.services.browser.agent_run import BrowserAgentRun
from app.services.browser.run_contract import RunOutcome
from app.services.browser.session import BrowserHostSession, engine_failure
from shared.py.wide_events import log

#: Whether the run is waiting on someone (a handoff, a guidance ask) and must not be judged.
PausedFn = Callable[[], bool]


async def run_watched(
    agent_run: BrowserAgentRun, task: str, session: BrowserHostSession, *, paused: PausedFn
) -> RunOutcome | EngineFailure:
    """Run the agent beside the watchdog; return its outcome, or how the engine failed when the watchdog cut it short.

    However this ends (an outcome, a raise, the caller cancelled) the watchdog
    stops, and a run still unwinding gets a bounded grace, not the caller's whole wait.
    """
    run = asyncio.create_task(agent_run.execute(task))
    watch = asyncio.create_task(_watch(session, agent_run, run, paused))
    try:
        await asyncio.wait({run, watch}, return_when=asyncio.FIRST_COMPLETED)
        if run.done() and not run.cancelled():
            return run.result()
        return watch.result()
    finally:
        watch.cancel()
        if not run.done():
            # A second cancel would cut Browser-Use's own teardown short, leaving
            # its event loops and watchdog tasks running for good.
            if not run.cancelling():
                run.cancel()
            unwind_started = perf_counter()
            await asyncio.wait({run}, timeout=BROWSER_ENGINE_WATCH_CANCEL_GRACE_SECONDS)
            log.info(
                f"{LogTag.BROWSER} Browser run unwound",
                browser={"session_id": session.session_id, "operation": "engine_watch"},
                duration_ms=round((perf_counter() - unwind_started) * 1000),
            )
            if not run.done():
                log.warning(
                    f"{LogTag.BROWSER} Browser run outlived its unwind grace",
                    browser={"session_id": session.session_id, "operation": "engine_watch"},
                )


async def _watch(
    session: BrowserHostSession,
    agent_run: BrowserAgentRun,
    run: asyncio.Task[RunOutcome],
    paused: PausedFn,
) -> EngineFailure:
    """Read the engine's liveness on a beat; after enough unanswered reads in a row, cut the run and return the failure."""
    strikes = 0
    while True:
        await asyncio.sleep(BROWSER_ENGINE_WATCH_INTERVAL_SECONDS)
        if paused():
            strikes = 0
            continue
        failure = await engine_failure(session)
        # A handoff may have begun while the read was in flight.
        if failure is None or paused():
            strikes = 0
            continue
        strikes += 1
        log.warning(
            f"{LogTag.BROWSER} Browser engine did not answer the watchdog",
            browser={"session_id": session.session_id, "operation": "engine_watch"},
            engine_failure=failure.value,
            strikes=strikes,
        )
        if strikes >= BROWSER_ENGINE_WATCH_STRIKES:
            # In the same tick as the last paused() read, so a handoff cannot
            # start between the judgement and the cut.
            run.cancel()
            await agent_run.abandon()
            return failure
