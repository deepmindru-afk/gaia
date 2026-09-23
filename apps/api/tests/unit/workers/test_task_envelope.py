"""The ARQ task envelope's deadline: a job cut off at its time limit says so.

ARQ enforces a job timeout by cancelling the task, which a wide-event boundary
records as a clean cancel. The envelope owns each task's deadline so the
cut-off reads as a failure with reason task_timeout, and ARQ's own timeout
sits past it as a backstop only.
"""

import asyncio
from collections.abc import Mapping
from unittest.mock import patch

import pytest

from app.workers.config.worker_settings import ARQ_BACKSTOP_GRACE_SECONDS
from app.workers.task_envelope import arq_function, arq_task
from tests.helpers import WideEventRecorder

pytestmark = pytest.mark.unit


async def _outlives_its_deadline(ctx: Mapping[str, object]) -> str:
    await asyncio.sleep(60)
    return "finished"


async def _raises_its_own_timeout(ctx: Mapping[str, object]) -> str:
    raise TimeoutError("upstream read timed out")


async def test_a_task_cut_off_at_its_deadline_fails_with_reason_task_timeout() -> None:
    recorder = WideEventRecorder()
    wrapped = arq_task(_outlives_its_deadline, timeout_seconds=0.01)
    with (
        patch("shared.py.wide_events._loguru", recorder),
        pytest.raises(TimeoutError),
    ):
        await wrapped({"job_id": "j1", "job_try": 1})

    event = recorder.event("_outlives_its_deadline")
    assert event["outcome"] == "failed"
    assert event["reason"] == "task_timeout"
    assert event["timeout_seconds"] == 0.01


async def test_the_task_body_still_cleans_up_when_its_deadline_cuts_it_off() -> None:
    cleaned_up: list[bool] = []

    async def _holds_a_slot(ctx: Mapping[str, object]) -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cleaned_up.append(True)
            raise

    with (
        patch("shared.py.wide_events._loguru", WideEventRecorder()),
        pytest.raises(TimeoutError),
    ):
        await arq_task(_holds_a_slot, timeout_seconds=0.01)({})

    assert cleaned_up == [True]


async def test_a_timeout_the_task_raised_itself_is_not_the_deadline() -> None:
    recorder = WideEventRecorder()
    with (
        patch("shared.py.wide_events._loguru", recorder),
        pytest.raises(TimeoutError),
    ):
        await arq_task(_raises_its_own_timeout, timeout_seconds=60)({})

    event = recorder.event("_raises_its_own_timeout")
    assert event["outcome"] == "failed"
    assert "reason" not in event


def test_a_task_registered_with_its_own_deadline_gets_arqs_backstop_past_it() -> None:
    registered = arq_function(_outlives_its_deadline, name="slow_job", timeout_seconds=7200)

    assert registered.name == "slow_job"
    assert registered.timeout_s == 7200 + ARQ_BACKSTOP_GRACE_SECONDS
