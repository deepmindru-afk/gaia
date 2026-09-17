"""Give keystrokes human variance without making the agent noticeably slower.

Browser-Use types on two fixed timers (5ms key hold, 1ms inter-key gap; a single
10ms gap in the _type_to_page fallback), so every keystroke is identical to the
microsecond. Behavioural checks read event.timeStamp deltas, and that zero-variance
metronome survives every fingerprint defence because it comes from our own input.

Only the distribution changes: each delay is scaled by a draw averaging ~1.9x,
about 100ms on a 20-character field. The residual tell (still faster than a
human's ~60ms/char) is accepted; the variance is what defeats the check. The RNG
is seeded per user so one person's rhythm stays consistent across tasks.

The shim is armed only while a typing method is on the stack, and it rebinds the
watchdog module's asyncio name to a proxy rather than patching asyncio.sleep.
Pinned to browser-use==0.11.13; the imports fail loudly if the methods move.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import contextvars
import random
from typing import Any, ParamSpec, TypeVar

from browser_use.browser.watchdogs import default_action_watchdog as watchdog_module
from browser_use.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog

from app.services.browser.fingerprint import current_fingerprint_seed

# Only the per-keystroke timers are this short; every other wait in the module
# is 50ms or longer, so this threshold separates rhythm from page timing.
_KEYSTROKE_DELAY_CEILING_SECONDS = 0.010
# Log-normal scale per delay, the shape human inter-key intervals take (clustered, with an
# occasional long pause). mu/sigma give a mean scale of ~1.9x; the cap keeps a tail draw
# from stalling.
_SCALE_MU = 0.49
_SCALE_SIGMA = 0.55
_MAX_SCALE = 6.0

P = ParamSpec("P")
R = TypeVar("R")

_rng: contextvars.ContextVar[random.Random | None] = contextvars.ContextVar(
    "browser_typing_rng", default=None
)


async def _sleep(delay: float, result: object = None) -> object:
    rng = _rng.get()
    if rng is not None and delay <= _KEYSTROKE_DELAY_CEILING_SECONDS:
        delay *= min(rng.lognormvariate(_SCALE_MU, _SCALE_SIGMA), _MAX_SCALE)
    return await asyncio.sleep(delay, result)


class _AsyncioProxy:
    """The stdlib asyncio module with only sleep swapped, for one module's use."""

    sleep = staticmethod(_sleep)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 -- delegates to an arbitrary asyncio module attribute, so Any is the honest return type
        return getattr(asyncio, name)


def _arm_typing_rhythm(
    method: Callable[P, Awaitable[R]],
) -> Callable[P, Awaitable[R]]:
    """Enable the jitter for the duration of one typing action."""

    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        # A private Random, not the global module RNG: seeded per user and scoped
        # to this call, so nothing else in the process has its randomness moved.
        token = _rng.set(random.Random(current_fingerprint_seed()))
        try:
            return await method(*args, **kwargs)
        finally:
            _rng.reset(token)

    return wrapper


watchdog_module.asyncio = _AsyncioProxy()  # type: ignore[assignment, attr-defined]  # inject an asyncio proxy into browser-use's watchdog module so its sleeps carry the typing rhythm
DefaultActionWatchdog._input_text_element_node_impl = _arm_typing_rhythm(  # type: ignore[method-assign, assignment]  # rebind browser-use's watchdog method with the rhythm-armed wrapper
    DefaultActionWatchdog._input_text_element_node_impl
)
DefaultActionWatchdog._type_to_page = _arm_typing_rhythm(  # type: ignore[method-assign, assignment]  # rebind browser-use's watchdog method with the rhythm-armed wrapper
    DefaultActionWatchdog._type_to_page
)
