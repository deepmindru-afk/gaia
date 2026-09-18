"""What the user is told when an executor run dies instead of finishing.

The crash path hands comms the run's terminal text. A raw exception string is
not a story, so comms invents one ("The browser task got cancelled partway
through... Want me to run it again?") — wrong about what happened AND an
invitation to re-run work that may have half-landed.
"""

import asyncio
from unittest.mock import AsyncMock, patch

from langgraph.errors import GraphRecursionError
import pytest

from app.agents.core.background import executor_runner as er
from app.constants.executor import EXECUTOR_STEP_LIMIT_MESSAGE


async def _run_with(side_effect: BaseException) -> er._ExecutorResult:
    """Run _execute_executor with the graph execution raising side_effect."""
    with (
        patch.object(er, "prepare_executor_execution", AsyncMock(return_value=(object(), None))),
        patch.object(er, "make_redis_stream_writer", lambda _sid: None),
        patch.object(er, "execute_subagent_stream", AsyncMock(side_effect=side_effect)),
    ):
        return await er._execute_executor("do the thing", {}, "stream-1")


@pytest.mark.unit
class TestExecutorCrashText:
    async def test_a_crash_never_hands_comms_the_raw_exception(self) -> None:
        result = await _run_with(RuntimeError("Watchdog timed out after 60s"))

        assert result.type == "error"
        assert "Watchdog timed out" not in result.text

    async def test_a_crash_tells_comms_not_to_offer_a_re_run(self) -> None:
        result = await _run_with(RuntimeError("boom"))

        assert result.text == er.EXECUTOR_CRASH_MESSAGE
        lowered = result.text.lower()
        assert "again" not in lowered
        assert "how they" in lowered or "how the user" in lowered

    async def test_an_exception_with_no_message_still_says_what_happened(self) -> None:
        # str(exc) is "" for plenty of real failures, and an empty error block is
        # exactly when comms invented a story and offered to re-run it.
        result = await _run_with(TimeoutError())

        assert result.type == "error"
        assert result.text == er.EXECUTOR_CRASH_MESSAGE

    async def test_a_user_stop_is_left_to_the_cancellation_path(self) -> None:
        # Swallowing CancelledError here would break cooperative cancellation:
        # a Stop is finalized as cancelled, not narrated as a crash.
        with pytest.raises(asyncio.CancelledError):
            await _run_with(asyncio.CancelledError())

    async def test_the_recursion_limit_keeps_its_own_actionable_message(self) -> None:
        result = await _run_with(GraphRecursionError("limit"))

        assert result.text == EXECUTOR_STEP_LIMIT_MESSAGE
