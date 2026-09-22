"""Background subagent coroutine for non-blocking handoff execution.

Spawned by handoff(background=True) via asyncio.create_task(). Runs the
subagent graph, appends the final result to the conversation's durable
results bucket (Redis — it must survive the executor's approval pause), and
decrements the in-process pending counter.

The executor never collects: landed results go straight to its inbox, and a
parked approval stamps its record and announces itself the same way, so the
executor stays free to steer or keep working.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
import time
from uuid import uuid4

from app.agents.core.background.executor_channel import ExecutorInbox
from app.agents.core.background.executor_queue import claim_collection_wake, is_executor_busy
from app.agents.core.background.executor_runner import deliver_to_executor
from app.agents.core.background.redis_writer import make_redis_stream_writer
from app.agents.core.background.running_registry import RunningSubagents
from app.agents.core.background.session import release_bg_integration
from app.agents.core.subagents.call_record import append_call_record
from app.agents.core.subagents.subagent_runner import (
    SubagentExecutionContext,
    SubagentInitialState,
    execute_subagent_stream,
)
from app.constants.agents import AgentTag
from app.constants.executor import EXECUTOR_COLLECTION_TASK
from app.constants.log_tags import LogTag
from app.models.agent_models import AgentConfigurable, RunningSubagent
from app.models.hil_models import HilInterruptPayload
from app.services.hil.approvals_store import stamp_subagent_resume
from app.utils.agent_utils import (
    IntegrationMetadata,
    SubagentStartDetails,
    format_subagent_end_event,
    format_subagent_start_event,
)
from shared.py.wide_events import get_trace_id, log, wide_task

#: Task name for a background handoff run. Tests drain by this name to wait out
#: exactly the subagents a turn dispatched, not every background task in the process.
BACKGROUND_SUBAGENT_TASK_NAME = "background-subagent-run"


@dataclass(frozen=True)
class BackgroundHandoff:
    """How a background subagent run presents itself and what it releases on exit.

    Icon/name metadata for tool events, the subagent and integration ids, the
    display triple for start/end cards, and whether successful calls are
    recorded (workflow runs only).
    """

    integration_metadata: IntegrationMetadata | None = None
    subagent_id: str | None = None
    display_name: str | None = None
    tool_category: str | None = None
    icon_url: str | None = None
    integration_id: str | None = None
    record_calls: bool = False


async def run_subagent_background(
    ctx: SubagentExecutionContext,
    stream_id: str,
    handoff: BackgroundHandoff | None = None,
) -> None:
    """Run a worker subagent in the background and store its result.

    Designed for asyncio.create_task(); never raises — every exception is caught
    and stored as the subagent's result text. A HIL pause is not an error: the
    graph is checkpointed, so this stamps the approval record with the thread id,
    announces the pause to the executor inbox, and exits.
    """
    handoff = handoff or BackgroundHandoff()
    integration_metadata, subagent_id, integration_id = (
        handoff.integration_metadata,
        handoff.subagent_id,
        handoff.integration_id,
    )
    display_name, tool_category, icon_url = (
        handoff.display_name,
        handoff.tool_category,
        handoff.icon_url,
    )
    record_calls = handoff.record_calls
    configurable: AgentConfigurable = ctx.configurable

    conversation_id = str(configurable.get("conversation_id", ""))
    # This task outlives the spawning executor turn, so it needs its own
    # wide-event boundary or every log.set() is silently discarded.
    # get_trace_id() correlates this event with the run that dispatched it.
    async with wide_task(
        "subagent_run",
        trace_id=get_trace_id() or None,
        agent_name=ctx.agent_name,
        conversation_id=conversation_id,
        stream_id=stream_id,
        subagent_id=subagent_id,
        integration_id=integration_id,
    ):
        try:
            writer = make_redis_stream_writer(stream_id)

            if subagent_id:
                writer(
                    {
                        "subagent_start": format_subagent_start_event(
                            subagent_name=display_name or ctx.agent_name,
                            agent_type="handoff",
                            subagent_id=subagent_id,
                            details=SubagentStartDetails(
                                icon_url=icon_url, tool_category=tool_category
                            ),
                        )
                    }
                )

            # Register the running subagent so the executor can steer or cancel
            # THIS worker by id while it runs (message_subagent / cancel_subagent).
            # Deregistered in `finally`, so a finished OR parked subagent leaves.
            if subagent_id:
                initial_state: SubagentInitialState = ctx.initial_state
                await RunningSubagents(conversation_id).register(
                    RunningSubagent(
                        subagent_id=subagent_id,
                        subagent_thread_id=str(configurable.get("thread_id", "")),
                        integration_id=integration_id or "",
                        agent_name=ctx.agent_name,
                        task_summary=str(initial_state.get("intent", ""))[:200],
                        started_at=datetime.now(UTC).isoformat(),
                    )
                )

            start_time = time.monotonic()
            outcome = await execute_subagent_stream(
                ctx=ctx,
                stream_writer=writer,
                integration_metadata=integration_metadata,
                subagent_id=subagent_id,
            )
            if outcome.paused:
                await _park(ctx, outcome.interrupt or {}, stream_id)
                return
            result = (
                append_call_record(outcome.text, outcome.run_messages)
                if record_calls
                else outcome.text
            )
            duration_ms = int((time.monotonic() - start_time) * 1000)

            if subagent_id:
                writer(
                    {
                        "subagent_end": format_subagent_end_event(
                            subagent_id=subagent_id,
                            duration_ms=duration_ms,
                        )
                    }
                )
            log.info(
                f"{LogTag.AGENT} Background subagent completed",
                agent_name=ctx.agent_name,
                stream_id=stream_id,
            )
            await _deliver_result(conversation_id, ctx.agent_name, result)
        except Exception as e:
            log.error(
                f"{LogTag.AGENT} Background subagent failed",
                agent_name=ctx.agent_name,
                stream_id=stream_id,
                error=str(e),
            )
            await _deliver_result(
                conversation_id, ctx.agent_name, f"Error from {ctx.agent_name}: {e!s}"
            )
        finally:
            if subagent_id:
                await RunningSubagents(conversation_id).deregister(subagent_id)
            if integration_id:
                release_bg_integration(stream_id, integration_id)
            await _wake_if_executor_rested(conversation_id, configurable)


async def _wake_if_executor_rested(conversation_id: str, configurable: AgentConfigurable) -> None:
    """Queue a collection turn when this landing has no live executor to collect it.

    Busy executor -> it will collect itself; headless run -> nothing to wake
    (results have no live audience). Best-effort: a wake failure must not
    crash the task — the marker TTL and the next landing retry it.
    """
    if not conversation_id or str(configurable.get("execution_mode") or "") == "background":
        return
    try:
        # Busy executor → it collects on its own; only a rested one needs waking.
        if not await is_executor_busy(conversation_id) and await claim_collection_wake(
            conversation_id
        ):
            await deliver_to_executor(conversation_id, configurable, EXECUTOR_COLLECTION_TASK)
    except Exception as e:  # create_task coroutine must not raise
        log.error(
            f"{LogTag.AGENT} Could not queue collection wake-up",
            conversation_id=conversation_id,
            error=str(e),
        )


async def _deliver_result(conversation_id: str, agent_name: str, result: str) -> None:
    """Hand a finished background result to the executor's inbox.

    The drain hook injects it before the executor's next reasoning step, so no
    collect call is needed whether the executor is mid-turn or rested. Agent-
    prefixed, because an inbox entry carries no attribution of its own.
    """
    await ExecutorInbox(conversation_id).append(
        str(uuid4()), f"{agent_name}: {result}", AgentTag.SUBAGENT_RESULT
    )


async def _park(
    ctx: SubagentExecutionContext, interrupt: HilInterruptPayload, stream_id: str
) -> None:
    """Record a HIL-paused subagent durably and say so out loud.

    No result is appended — there is none yet. But the pause itself is
    announced to the executor inbox: parked work with no witness rots silently
    until expiry, and the HIL rework that will review it is not built yet.
    """
    configurable: AgentConfigurable = ctx.configurable
    approval_id = str(interrupt.get("approval_id", ""))
    thread_id = str(configurable.get("thread_id", ""))
    if not approval_id or not thread_id:
        # Unresumable pause: without the id pair nothing can ever collect this
        # subagent. Surface it as an error result rather than stranding silently.
        raise RuntimeError(
            f"Background subagent {ctx.agent_name} paused on approval but the pause "
            f"is unresumable (approval_id={approval_id!r}, thread_id={thread_id!r})"
        )
    await stamp_subagent_resume(
        approval_id,
        subagent_thread_id=thread_id,
        subagent_agent_name=ctx.agent_name,
    )
    conversation_id = str(configurable.get("conversation_id", ""))
    if conversation_id:
        await ExecutorInbox(conversation_id).append(
            str(uuid4()),
            f"{ctx.agent_name} is paused waiting for the user's approval "
            f"(approval {approval_id}). Background approvals cannot be reviewed "
            "yet; the request stays pending until it times out.",
            AgentTag.SUBAGENT_RESULT,
        )
    log.info(
        f"{LogTag.HIL} Background subagent parked on approval",
        agent_name=ctx.agent_name,
        approval_id=approval_id,
        subagent_thread_id=thread_id,
        stream_id=stream_id,
    )
