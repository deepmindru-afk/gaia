"""Background results reach the executor inbox — no collect call needed."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.core.subagents.subagent_runner import SubagentOutcome

MOD = "app.agents.core.background.subagent_runner"


def _ctx(**overrides: object) -> SimpleNamespace:
    base: dict[str, object] = {
        "agent_name": "gmail_agent",
        "integration_id": "gmail",
        "config": {"configurable": {"thread_id": "t1", "conversation_id": "c1"}},
        "configurable": {"thread_id": "t1", "conversation_id": "c1"},
        "initial_state": {"intent": "triage"},
        "user_id": "u1",
        "stream_id": "s1",
    }
    base.update(overrides)
    return SimpleNamespace(
        subagent_graph=MagicMock(),
        agent_name=base["agent_name"],
        config=base["config"],
        configurable=base["configurable"],
        integration_id=base["integration_id"],
        initial_state=base["initial_state"],
        user_id=base["user_id"],
        stream_id=base["stream_id"],
    )


def _harness(execute_result: object) -> tuple[MagicMock, MagicMock]:
    """Mocked inbox class + stream driver; returns (inbox_instance, writer)."""
    inbox = MagicMock()
    inbox.append = AsyncMock()
    writer = MagicMock()
    return inbox, writer


@pytest.mark.unit
class TestResultDelivery:
    async def test_completion_lands_in_the_executor_inbox(self) -> None:
        from app.agents.core.background.subagent_runner import run_subagent_background
        from app.constants.agents import AgentTag

        inbox, writer = _harness(None)
        with (
            patch(
                f"{MOD}.execute_subagent_stream",
                new=AsyncMock(return_value=SubagentOutcome(text="archived 3")),
            ),
            patch(f"{MOD}.make_redis_stream_writer", return_value=writer),
            patch(f"{MOD}.ExecutorInbox", return_value=inbox),
            patch(f"{MOD}.RunningSubagents"),
            patch(f"{MOD}._wake_if_executor_rested", new=AsyncMock()),
        ):
            await run_subagent_background(_ctx(), stream_id="s1")

        inbox.append.assert_awaited_once()
        _, text, tag = inbox.append.await_args.args
        assert "gmail_agent" in text and "archived 3" in text
        assert tag is AgentTag.SUBAGENT_RESULT

    async def test_failure_lands_in_the_executor_inbox(self) -> None:
        from app.agents.core.background.subagent_runner import run_subagent_background

        inbox, writer = _harness(None)
        with (
            patch(
                f"{MOD}.execute_subagent_stream",
                new=AsyncMock(side_effect=RuntimeError("boom")),
            ),
            patch(f"{MOD}.make_redis_stream_writer", return_value=writer),
            patch(f"{MOD}.ExecutorInbox", return_value=inbox),
            patch(f"{MOD}.RunningSubagents"),
            patch(f"{MOD}._wake_if_executor_rested", new=AsyncMock()),
        ):
            await run_subagent_background(_ctx(), stream_id="s1")

        inbox.append.assert_awaited_once()
        assert "boom" in inbox.append.await_args.args[1]

    async def test_park_announces_itself_instead_of_going_silent(self) -> None:
        """A parked approval has no review path yet; silence until expiry
        would strand it invisibly. The inbox note keeps a witness."""
        from app.agents.core.background.subagent_runner import run_subagent_background

        inbox, writer = _harness(None)
        with (
            patch(
                f"{MOD}.execute_subagent_stream",
                new=AsyncMock(
                    return_value=SubagentOutcome(text="", interrupt={"approval_id": "ap1"})
                ),
            ),
            patch(f"{MOD}.make_redis_stream_writer", return_value=writer),
            patch(f"{MOD}.ExecutorInbox", return_value=inbox),
            patch(f"{MOD}.RunningSubagents"),
            patch(f"{MOD}.stamp_subagent_resume", new=AsyncMock()),
            patch(f"{MOD}._wake_if_executor_rested", new=AsyncMock()),
        ):
            await run_subagent_background(_ctx(), stream_id="s1")

        inbox.append.assert_awaited_once()
        text = inbox.append.await_args.args[1]
        assert "ap1" in text and "approval" in text
