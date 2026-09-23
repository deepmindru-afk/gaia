"""The running-subagent registry — the executor's addressable handle on live workers.

Proves the executor can enumerate exactly what is running and name one by id, that a
finished subagent leaves the registry (so a stale id can never be steered), and that
one checkpoint thread carries at most one live run.
"""

from unittest.mock import patch

import fakeredis.aioredis
import pytest

from app.agents.core.background import running_registry as registry
from app.agents.core.background.running_registry import RunningSubagents
from app.models.agent_models import RunningSubagent

pytestmark = pytest.mark.unit

CONV = "conv-1"


def _sub(subagent_id: str, integration: str = "gmail", thread: str = "") -> RunningSubagent:
    return RunningSubagent(
        subagent_id=subagent_id,
        subagent_thread_id=thread or f"{integration}_executor_{CONV}",
        integration_id=integration,
        agent_name=f"{integration}_agent",
        task_summary="search mail",
        started_at="2026-09-05T10:00:00Z",
    )


@pytest.fixture
async def redis():
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    with patch.object(registry, "redis_cache") as cache:
        cache.client = client
        yield client
    await client.aclose()


class TestRunningSubagents:
    async def test_register_then_list_and_get_by_id(self, redis) -> None:
        reg = RunningSubagents(CONV)
        sub = _sub("s1")
        assert await reg.claim(sub)
        assert await reg.list() == [sub]
        assert await reg.get("s1") == sub

    async def test_deregister_removes_only_the_named_one(self, redis) -> None:
        reg = RunningSubagents(CONV)
        assert await reg.claim(_sub("s1", "gmail"))
        assert await reg.claim(_sub("s2", "slack"))
        await reg.deregister(_sub("s1", "gmail"))
        remaining = await reg.list()
        assert [s.subagent_id for s in remaining] == ["s2"]
        assert await reg.get("s1") is None

    async def test_a_finished_subagent_cannot_be_addressed(self, redis) -> None:
        reg = RunningSubagents(CONV)
        assert await reg.claim(_sub("s1"))
        await reg.deregister(_sub("s1"))
        assert await reg.list() == []
        assert await reg.get("s1") is None

    async def test_registries_are_isolated_per_conversation(self, redis) -> None:
        assert await RunningSubagents(CONV).claim(_sub("s1"))
        assert await RunningSubagents("other-conv").list() == []


class TestThreadClaim:
    async def test_a_second_run_on_a_held_thread_is_refused(self, redis) -> None:
        reg = RunningSubagents(CONV)
        assert await reg.claim(_sub("s1", thread="notion-mcp_executor_conv-1"))
        refused = _sub("s2", thread="notion-mcp_executor_conv-1")
        assert not await reg.claim(refused)
        assert [s.subagent_id for s in await reg.list()] == ["s1"]
        assert await reg.holds_thread("notion-mcp_executor_conv-1")

    async def test_the_thread_frees_when_its_run_deregisters(self, redis) -> None:
        reg = RunningSubagents(CONV)
        first = _sub("s1", thread="spawn_conv-1_call-1")
        assert await reg.claim(first)
        await reg.deregister(first)
        assert not await reg.holds_thread("spawn_conv-1_call-1")
        assert await reg.claim(_sub("s1", thread="spawn_conv-1_call-1"))

    async def test_distinct_threads_run_side_by_side(self, redis) -> None:
        reg = RunningSubagents(CONV)
        assert await reg.claim(_sub("s1", thread="spawn_conv-1_a"))
        assert await reg.claim(_sub("s2", thread="spawn_conv-1_b"))
        assert {s.subagent_id for s in await reg.list()} == {"s1", "s2"}
