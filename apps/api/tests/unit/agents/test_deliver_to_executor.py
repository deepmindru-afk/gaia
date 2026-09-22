"""Unit tests for deliver_to_executor's handoff to the background executor.

One invariant: work appended while a run is finalizing must still get a run.
The old run's carry reads the inbox exactly once, so an entry that lands after
that read and after the lock release would sit unseen until unrelated future
activity. The append is therefore followed by a recheck that starts a carry
run when the lock is free.
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.agents.core.background import (
    executor_channel as ec,
    executor_queue as eq,
    executor_runner as er,
)
from app.agents.core.background.executor_channel import ExecutorInbox
from app.agents.core.background.executor_queue import build_lock_value, release_lock_if_owned
from app.agents.core.background.session import ExecutorRun, RunKind
from app.constants.agents import AgentTag
from app.constants.cache import EXECUTOR_BUSY_PREFIX
from app.constants.executor import EXECUTOR_CARRY_TASK
from app.models.agent_models import InboxEntry
from app.models.user_models import AuthenticatedUser

pytestmark = pytest.mark.unit

CONVERSATION = "conv-1"
TASK = "summarize the thread"


def _seams(*, busy: list[bool], started: bool):
    """Patch deliver_to_executor's seams: the busy fast-path, run start, inbox."""
    stack = patch.object(er, "is_executor_busy", new_callable=AsyncMock, side_effect=busy)
    start = patch.object(er, "_start_executor_run", new_callable=AsyncMock, return_value=started)
    append = patch.object(er.ExecutorInbox, "append", new_callable=AsyncMock)
    return stack, start, append


class TestMidFinalizeWorkGetsARun:
    async def test_an_append_after_the_old_runs_carry_starts_a_carry_run(self) -> None:
        """Busy at entry (old run finalizing), free after the append (released, carry missed it)."""
        busy, start, append = _seams(busy=[True, False], started=True)
        with busy, start as start_mock, append as append_mock:
            await er.deliver_to_executor(CONVERSATION, AuthenticatedUser(user_id="u1"), TASK)

        append_mock.assert_awaited_once()  # the work is never dropped from the inbox
        start_mock.assert_awaited_once()  # ...nor left without a run to drain it
        assert start_mock.await_args.args[2] == EXECUTOR_CARRY_TASK

    async def test_a_live_run_absorbs_the_append_with_no_extra_start(self) -> None:
        busy, start, append = _seams(busy=[True, True], started=False)
        with busy, start as start_mock, append as append_mock:
            await er.deliver_to_executor(CONVERSATION, AuthenticatedUser(user_id="u1"), TASK)

        append_mock.assert_awaited_once()
        start_mock.assert_not_awaited()

    async def test_an_idle_conversation_starts_a_run_for_the_task_itself(self) -> None:
        busy, start, append = _seams(busy=[False], started=True)
        with busy, start as start_mock, append as append_mock:
            await er.deliver_to_executor(CONVERSATION, AuthenticatedUser(user_id="u1"), TASK)

        start_mock.assert_awaited_once()
        assert start_mock.await_args.args[2] == TASK
        append_mock.assert_not_awaited()


class TestTaggedWorkTravelsThroughTheInbox:
    async def test_an_idle_conversation_gets_the_entry_framed_and_a_carry_run(self) -> None:
        # A subagent result must reach the model framed as a subagent result, so it is
        # never the run's bare task, even when the conversation is idle.
        busy, start, append = _seams(busy=[False], started=True)
        with busy, start as start_mock, append as append_mock:
            await er.deliver_to_executor(
                CONVERSATION, AuthenticatedUser(user_id="u1"), TASK, tag=AgentTag.SUBAGENT_RESULT
            )

        append_mock.assert_awaited_once()
        assert append_mock.await_args.args[1:] == (TASK, AgentTag.SUBAGENT_RESULT)
        start_mock.assert_awaited_once()
        assert start_mock.await_args.args[2] == EXECUTOR_CARRY_TASK


class _FakeRedisClient:
    """Enough of the raw Redis surface for the busy lock and the inbox list.

    set does its NX check and its write with no await between them, the same
    atomicity single-threaded Redis gives the claim.
    """

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.lists: dict[str, list[str]] = {}

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def set(
        self, key: str, value: str, ex: int | None = None, nx: bool = False
    ) -> bool | None:
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    async def delete(self, key: str) -> None:
        self.store.pop(key, None)

    async def rpush(self, key: str, value: str) -> int:
        self.lists.setdefault(key, []).append(value)
        return len(self.lists[key])

    async def expire(self, key: str, ttl: int) -> bool:
        return key in self.lists or key in self.store

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        values = self.lists.get(key, [])
        return values[start:] if end == -1 else values[start : end + 1]

    async def lrem(self, key: str, count: int, value: str) -> int:
        values = self.lists.get(key, [])
        if value not in values:
            return 0
        values.remove(value)
        return 1


class _FakeRedisCache:
    def __init__(self) -> None:
        self.client = _FakeRedisClient()

    async def delete(self, key: str) -> None:
        await self.client.delete(key)


class TestTheRealLockAndInbox:
    """The same invariant, driven through the real busy lock and the real inbox.

    The class above patches the seams that decide, so it pins the branch table
    and nothing else. This one reproduces the interleaving itself: the old run's
    whole finalize tail — its ownership-checked release and its one-time carry
    read — lands between the busy check and the append.
    """

    async def test_an_entry_landing_between_the_release_and_the_carry_gets_a_run(self) -> None:
        cache = _FakeRedisCache()
        cache.client.store[f"{EXECUTOR_BUSY_PREFIX}{CONVERSATION}"] = build_lock_value(
            "stream-a", "task-a"
        )
        old_run = ExecutorRun(
            stream_id="stream-a",
            conversation_id=CONVERSATION,
            user=AuthenticatedUser(user_id="u1"),
            kind=RunKind.QUEUED,
            task_id="task-a",
            user_message_id=None,
        )
        real_append = ExecutorInbox.append

        async def finalize_the_old_run_then_append(
            inbox: ExecutorInbox, entry_id: str, text: str, tag: AgentTag | None = None
        ) -> InboxEntry:
            await release_lock_if_owned(CONVERSATION, "stream-a", "task-a")
            await er._carry_pending_into_new_run(old_run, None)
            return await real_append(inbox, entry_id, text, tag)

        with (
            patch.object(eq, "redis_cache", cache),
            patch.object(ec, "redis_cache", cache),
            patch.object(eq, "StreamManager", AsyncMock()),
            patch.object(eq, "websocket_manager", AsyncMock()),
            patch.object(er, "_spawn_detached_run") as spawn,
            patch.object(ExecutorInbox, "append", finalize_the_old_run_then_append),
        ):
            await er.deliver_to_executor(CONVERSATION, AuthenticatedUser(user_id="u1"), TASK)
            pending = await ExecutorInbox(CONVERSATION).read()

        spawn.assert_called_once()
        assert spawn.call_args.args[0].task == EXECUTOR_CARRY_TASK
        # Left in the inbox on purpose: the new run's drain hook injects it and
        # retires it only once the thread holds it.
        assert [entry.text for entry in pending] == [TASK]
