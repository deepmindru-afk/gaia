"""The delegation runner: one lifecycle for every subagent the executor delegates to.

Redis is real (fakeredis) so the thread claim, the dispatch claim and the inbox are
the real mechanisms; the subagent graph run itself (execute_subagent_stream) and the
client edges (stream start, websocket, message persistence) are the doubled seams.
"""

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis.aioredis
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphInterrupt
import pytest

from app.agents.core.background import executor_queue
from app.agents.core.background.running_registry import RunningSubagents
from app.agents.core.subagents import delegation
from app.agents.core.subagents.delegation import (
    Delegation,
    SubagentDisplay,
    delegate,
    runs_in_background,
)
from app.agents.core.subagents.subagent_runner import SubagentExecutionContext, SubagentOutcome
from app.constants.agents import AgentTag
from app.constants.hil import SUBAGENT_RESUME_CONFIG_KEY
from app.db.redis import redis_cache
from app.models.agent_models import AgentConfigurable, RunningSubagent, SubagentKind
from app.models.hil_models import HILApprovalStatus
from app.utils import background_tasks
from tests.unit.services.hil.conftest import make_record

pytestmark = pytest.mark.unit

MODULE = "app.agents.core.subagents.delegation"
CONVERSATION = "conv-d"
THREAD = f"spawn_{CONVERSATION}_call-1"

LIVE: AgentConfigurable = {
    "user_id": "u1",
    "conversation_id": CONVERSATION,
    "stream_id": "parent-stream",
    "execution_mode": "interactive",
    "bot_message_id": "bot-msg-1",
}


def _delegation(parent: AgentConfigurable | None = None, **ctx_overrides: Any) -> Delegation:
    parent = dict(parent if parent is not None else LIVE)
    configurable = {**parent, "thread_id": THREAD, **ctx_overrides}
    return Delegation(
        ctx=SubagentExecutionContext(
            subagent_graph=MagicMock(),
            agent_name="spawned_subagent",
            config={"configurable": dict(configurable)},
            configurable=configurable,
            integration_id="spawn",
            initial_state={},
            user_id="u1",
            stream_id=parent.get("stream_id"),
        ),
        kind=SubagentKind.SPAWN,
        subagent_id="row-1",
        tool_call_id="call-1",
        task="summarise the report",
        display=SubagentDisplay(name="summarise the report", agent_type="spawned"),
        parent_configurable=parent,
    )


async def _drain() -> None:
    while pending := [
        t
        for t in background_tasks._background_tasks
        if t.get_name() in {delegation.BACKGROUND_SUBAGENT_TASK_NAME, "stream-publish"}
    ]:
        await asyncio.gather(*pending, return_exceptions=True)


@pytest.fixture
async def redis() -> AsyncIterator[Any]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    with patch.object(redis_cache, "redis", client):
        yield client
    await client.aclose()


@pytest.fixture
def client_edges() -> Iterator[SimpleNamespace]:
    """Record the stream, websocket and persistence edges a background run touches."""
    broadcasts: list[dict[str, Any]] = []

    async def _broadcast(_user_id: str, payload: dict[str, Any]) -> None:
        broadcasts.append(payload)

    with (
        patch.object(executor_queue, "StreamManager", AsyncMock()),
        patch.object(executor_queue.websocket_manager, "broadcast_to_user", new=_broadcast),
        patch(f"{MODULE}.conversation_repository") as conversations,
        patch(f"{MODULE}.stream_manager") as streams,
        patch(f"{MODULE}.deliver_to_executor", new=AsyncMock()) as deliver,
    ):
        conversations.append_message_tool_data = AsyncMock(return_value=True)
        streams.is_cancelled = AsyncMock(return_value=False)
        yield SimpleNamespace(broadcasts=broadcasts, conversations=conversations, deliver=deliver)


class TestWhereARunRuns:
    def test_a_live_conversation_runs_in_the_background(self) -> None:
        assert runs_in_background(True, LIVE) is True

    def test_asking_to_wait_waits(self) -> None:
        assert runs_in_background(False, LIVE) is False

    def test_a_headless_run_waits_even_when_asked_for_the_background(self) -> None:
        # A workflow or scheduled todo delivers once: a later landing reaches nobody.
        assert runs_in_background(True, {**LIVE, "execution_mode": "background"}) is False

    def test_a_run_with_no_stream_or_conversation_waits(self) -> None:
        assert runs_in_background(True, {**LIVE, "stream_id": None}) is False
        assert runs_in_background(True, {**LIVE, "conversation_id": ""}) is False


@contextmanager
def _blocking(outcomes: list[Any], recovered: SubagentOutcome | None = None) -> Iterator[Any]:
    writer = MagicMock()
    execute = AsyncMock(side_effect=outcomes)
    with (
        patch(f"{MODULE}.get_stream_writer", return_value=writer),
        patch(f"{MODULE}.execute_subagent_stream", new=execute),
        patch(f"{MODULE}.recover_from_checkpoint", new=AsyncMock(return_value=recovered)),
        patch(f"{MODULE}.resume_for_gate", return_value={"status": "approved"}),
    ):
        yield SimpleNamespace(writer=writer, execute=execute)


def _events(writer: MagicMock) -> list[str]:
    return [next(iter(c.args[0])) for c in writer.call_args_list]


class TestABlockingRun:
    async def test_it_holds_its_thread_for_exactly_its_run(self, redis: Any) -> None:
        held_during: list[bool] = []

        async def _run(**_kwargs: Any) -> SubagentOutcome:
            held_during.append(await RunningSubagents(CONVERSATION).holds_thread(THREAD))
            return SubagentOutcome(text="done")

        with _blocking([]) as h:
            h.execute.side_effect = _run
            result = await delegate(_delegation(), background=False, probe_parked=False)

        assert result == "done"
        assert held_during == [True]
        assert not await RunningSubagents(CONVERSATION).holds_thread(THREAD)
        assert _events(h.writer) == ["subagent_start", "subagent_end"]

    async def test_a_failed_run_frees_its_thread_and_closes_its_row(self, redis: Any) -> None:
        with _blocking([RuntimeError("graph exploded")]) as h, pytest.raises(RuntimeError):
            await delegate(_delegation(), background=False, probe_parked=False)

        assert not await RunningSubagents(CONVERSATION).holds_thread(THREAD)
        assert _events(h.writer) == ["subagent_start", "subagent_end"]

    async def test_a_pause_bubbles_up_with_the_row_left_open(self, redis: Any) -> None:
        paused = SubagentOutcome(text="", interrupt={"approval_id": "a1"})
        with (
            _blocking([paused]) as h,
            patch(f"{MODULE}.resume_for_gate", side_effect=GraphInterrupt()),
            pytest.raises(GraphInterrupt),
        ):
            await delegate(_delegation(), background=False, probe_parked=False)

        assert _events(h.writer) == ["subagent_start"]
        # Released, so the replay that the decision triggers can claim it again.
        assert not await RunningSubagents(CONVERSATION).holds_thread(THREAD)

    async def test_a_held_thread_refuses_in_whole_sentences(self, redis: Any) -> None:
        await RunningSubagents(CONVERSATION).claim(
            RunningSubagent(
                subagent_id="other",
                subagent_thread_id=THREAD,
                integration_id="spawn",
                agent_name="spawned_subagent",
                task_summary="",
                started_at="",
            )
        )
        with _blocking([]) as h:
            result = await delegate(_delegation(), background=False, probe_parked=False)

        h.execute.assert_not_awaited()
        assert result.startswith("summarise the report is already running on this integration")
        assert "message_subagent" in result and "cancel_subagent" in result

    async def test_a_replay_recovers_the_finished_thread_instead_of_rerunning_it(
        self, redis: Any
    ) -> None:
        with _blocking([], recovered=SubagentOutcome(text="recovered")) as h:
            result = await delegate(_delegation(), background=False, probe_parked=True)

        assert result == "recovered"
        h.execute.assert_not_awaited()

    async def test_a_workflow_run_records_every_call_across_a_pause(self, redis: Any) -> None:
        def _outcome(text: str, to: str, *, paused: bool) -> SubagentOutcome:
            return SubagentOutcome(
                text=text,
                interrupt={"approval_id": "a1"} if paused else None,
                run_messages=(
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "GMAIL_SEND", "args": {"to": to}, "id": to}],
                    ),
                    ToolMessage(content="sent", tool_call_id=to),
                ),
            )

        parent = {**LIVE, "workflow_id": "wf-1", "execution_mode": "background"}
        with _blocking(
            [_outcome("", "a@b.c", paused=True), _outcome("both sent", "d@e.f", paused=False)]
        ):
            result = await delegate(_delegation(parent), background=True, probe_parked=False)

        assert result.startswith("both sent")
        assert 'GMAIL_SEND({"to":"a@b.c"})' in result
        assert 'GMAIL_SEND({"to":"d@e.f"})' in result


class TestABackgroundDispatch:
    async def test_it_returns_the_acknowledgement_and_runs_detached(
        self, redis: Any, client_edges: SimpleNamespace
    ) -> None:
        with patch(
            f"{MODULE}.execute_subagent_stream",
            new=AsyncMock(return_value=SubagentOutcome(text="the summary")),
        ):
            ack = await delegate(_delegation(), background=True, probe_parked=False)
            await _drain()

        assert "started in the background as subagent row-1" in ack
        landed = client_edges.deliver.await_args
        assert landed.args[2] == "summarise the report (subagent row-1): the summary"
        assert landed.kwargs["tag"] is AgentTag.SUBAGENT_RESULT
        assert not await RunningSubagents(CONVERSATION).holds_thread(THREAD)

    async def test_a_replay_of_the_dispatching_node_does_not_run_it_twice(
        self, redis: Any, client_edges: SimpleNamespace
    ) -> None:
        execute = AsyncMock(return_value=SubagentOutcome(text="the summary"))
        with patch(f"{MODULE}.execute_subagent_stream", new=execute):
            first = await delegate(_delegation(), background=True, probe_parked=False)
            second = await delegate(_delegation(), background=True, probe_parked=False)
            await _drain()

        assert first == second
        assert execute.await_count == 1

    async def test_a_refused_dispatch_gives_its_claim_back(
        self, redis: Any, client_edges: SimpleNamespace
    ) -> None:
        holder = RunningSubagent(
            subagent_id="other",
            subagent_thread_id=THREAD,
            integration_id="spawn",
            agent_name="spawned_subagent",
            task_summary="",
            started_at="",
        )
        await RunningSubagents(CONVERSATION).claim(holder)
        refused = await delegate(_delegation(), background=True, probe_parked=False)
        await RunningSubagents(CONVERSATION).deregister(holder)

        execute = AsyncMock(return_value=SubagentOutcome(text="ran"))
        with patch(f"{MODULE}.execute_subagent_stream", new=execute):
            retried = await delegate(_delegation(), background=True, probe_parked=False)
            await _drain()

        assert "already running" in refused
        assert "started in the background" in retried
        assert execute.await_count == 1


class TestABackgroundRunOwnsItsStream:
    async def test_it_announces_its_own_stream_folded_into_the_parents_message(
        self, redis: Any, client_edges: SimpleNamespace
    ) -> None:
        seen: dict[str, Any] = {}

        async def _run(**kwargs: Any) -> SubagentOutcome:
            ctx = kwargs["ctx"]
            seen["stream_id"] = ctx.stream_id
            seen["configurable"] = dict(ctx.configurable)
            seen["parent_stream_id"] = ctx.parent_stream_id
            return SubagentOutcome(text="done")

        with patch(f"{MODULE}.execute_subagent_stream", new=_run):
            await delegate(_delegation(), background=True, probe_parked=False)
            await _drain()

        (announced,) = [
            b for b in client_edges.broadcasts if b["type"] == "executor.stream_started"
        ]
        assert announced["stream_id"] == seen["stream_id"] != "parent-stream"
        assert announced["task_id"] == "row-1"
        assert announced["bot_message_id"] == "bot-msg-1"
        # The client upserts only this run's cards: the turn's own run may still be writing.
        assert announced["kind"] == "subagent"
        # The gate reads both off the run's configurable: the card goes on this
        # stream, and the approval record gets the recipe that resumes the run.
        assert seen["configurable"]["stream_id"] == seen["stream_id"]
        assert seen["configurable"][SUBAGENT_RESUME_CONFIG_KEY]["tool_call_id"] == "call-1"
        # A Stop on the dispatching turn still reaches it.
        assert seen["parent_stream_id"] == "parent-stream"

    async def test_its_frames_are_saved_into_the_parents_message(
        self, redis: Any, client_edges: SimpleNamespace
    ) -> None:
        with patch(
            f"{MODULE}.execute_subagent_stream",
            new=AsyncMock(return_value=SubagentOutcome(text="done")),
        ):
            await delegate(_delegation(), background=True, probe_parked=False)
            await _drain()

        saved = client_edges.conversations.append_message_tool_data.await_args
        assert saved.args[0] == CONVERSATION
        assert saved.kwargs["message_id"] == "bot-msg-1"
        assert saved.kwargs["entries"], "the run's subagent row must be saved"


class TestABackgroundRunThatParks:
    @staticmethod
    def _paused() -> SubagentOutcome:
        return SubagentOutcome(
            text="", interrupt={"approval_id": "appr-1", "summary": "Send the report"}
        )

    async def test_it_says_what_it_waits_on_and_lands_no_result(
        self, redis: Any, client_edges: SimpleNamespace
    ) -> None:
        pending = make_record(approval_id="appr-1", summary="Send the report")
        with (
            patch(f"{MODULE}.execute_subagent_stream", new=AsyncMock(return_value=self._paused())),
            patch(f"{MODULE}.get_approval", new=AsyncMock(return_value=pending)),
        ):
            await delegate(_delegation(), background=True, probe_parked=False)
            await _drain()

        (announced,) = [c.args[2] for c in client_edges.deliver.await_args_list]
        assert "waiting for the user's approval: Send the report (approval appr-1)" in announced
        assert "cannot be reviewed" not in announced
        assert not await RunningSubagents(CONVERSATION).holds_thread(THREAD)

    async def test_a_decision_that_beat_the_park_resumes_it_at_once(
        self, redis: Any, client_edges: SimpleNamespace
    ) -> None:
        decided = make_record(approval_id="appr-1", status=HILApprovalStatus.APPROVED)
        execute = AsyncMock(side_effect=[self._paused(), SubagentOutcome(text="sent")])
        with (
            patch(f"{MODULE}.execute_subagent_stream", new=execute),
            patch(f"{MODULE}.get_approval", new=AsyncMock(return_value=decided)),
            patch(f"{MODULE}.mark_resumed", new=AsyncMock()) as resumed,
        ):
            await delegate(_delegation(), background=True, probe_parked=False)
            await _drain()

        assert execute.await_count == 2
        resume = execute.await_args_list[1].kwargs["resume"]
        assert resume.resume == {
            "status": "approved",
            "feedback": None,
            "scope": "once",
            "approval_id": "appr-1",
        }
        resumed.assert_awaited_once_with("appr-1")
        (landed,) = [c.args[2] for c in client_edges.deliver.await_args_list]
        assert landed.endswith(": sent")
