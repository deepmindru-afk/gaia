"""The hedged writer call: a slow first answer is beaten by a spare, failures fall through."""

import asyncio

import pytest

from app.services.browser.jev.hedge import first_answer

pytestmark = pytest.mark.unit


def _calls(*behaviours):
    """Each call runs the next behaviour: (seconds to wait, value or exception)."""
    started: list[int] = []

    async def call():
        index = len(started)
        started.append(index)
        delay, outcome = behaviours[index]
        await asyncio.sleep(delay)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return call, started


async def test_a_fast_answer_starts_no_spare() -> None:
    call, started = _calls((0.0, "first"))

    assert await first_answer(call, hedge_after=0.05, deadline=1) == "first"
    assert started == [0]


async def test_a_slow_first_call_is_beaten_by_the_spare() -> None:
    call, started = _calls((1.0, "slow"), (0.0, "spare"))

    assert await first_answer(call, hedge_after=0.05, deadline=2) == "spare"
    assert started == [0, 1]


async def test_a_failure_is_raised_not_retried() -> None:
    call, started = _calls((0.0, RuntimeError("boom")), (0.0, "spare"))

    with pytest.raises(RuntimeError, match="boom"):
        await first_answer(call, hedge_after=10, deadline=20)
    assert started == [0]


async def test_the_spare_still_answers_when_the_slow_first_call_fails() -> None:
    call, _ = _calls((0.2, RuntimeError("boom")), (0.3, "spare"))

    assert await first_answer(call, hedge_after=0.05, deadline=2) == "spare"


async def test_no_answer_within_the_deadline_times_out() -> None:
    call, _ = _calls((5.0, "late"), (5.0, "late"))

    with pytest.raises(TimeoutError):
        await first_answer(call, hedge_after=0.05, deadline=0.2)
