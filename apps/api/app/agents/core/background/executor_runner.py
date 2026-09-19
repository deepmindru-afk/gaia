"""Background executor lifecycle: execute → finalize → hand off the queue.

Spawned by the call_executor tool (live runs) or the previous run's finalize
step (queued runs) via asyncio.create_task(). Runs the executor agent graph
with a Redis stream writer for tool events, then finalizes: signals
executor-done so a waiting chat stream can close its SSE; routes the terminal
outcome through exactly one delivery entry point (deliver_result, or
persist_cancelled_run for self-owned tool_data — see result_delivery); for
queued runs tears down the session and closes the stream; then hands the busy
lock to the next queued task, or releases it.

The executor:busy Redis key prevents concurrent executor spawns per
conversation. TTL of 30 minutes is a safety net — released explicitly.
"""

from dataclasses import dataclass, replace
import time
from typing import NamedTuple
from uuid import uuid4

from langgraph.errors import GraphRecursionError
from langgraph.types import Command
from langsmith import traceable
from pydantic import BaseModel, ConfigDict

from app.agents.core.background.comms_narrator import record_executor_cancellation
from app.agents.core.background.executor_capture import (
    build_returned_to_frontend_note,
    drain_executor_tool_data,
    teardown_executor_capture,
)
from app.agents.core.background.executor_channel import ExecutorInbox, decide_drain
from app.agents.core.background.executor_queue import (
    LockClaim,
    PreparedQueuedTask,
    build_run_item,
    extend_lock_if_owned,
    is_executor_busy,
    prepare_run_from_item,
    release_lock_if_owned,
)
from app.agents.core.background.redis_writer import make_redis_stream_writer
from app.agents.core.background.result_delivery import deliver_result, persist_cancelled_run
from app.agents.core.background.session import (
    ExecutorRun,
    RunIdentity,
    executor_abandoned,
    get_session,
    signal_executor_done,
)
from app.agents.core.subagents.subagent_runner import (
    SubagentExecutionContext,
    execute_subagent_stream,
    prepare_executor_execution,
    thread_messages,
)
from app.constants.agents import AgentTag, wrap_agent_payload
from app.constants.executor import (
    EXECUTOR_APPROVAL_LOST_MESSAGE,
    EXECUTOR_PAUSED,
    EXECUTOR_STEP_LIMIT_MESSAGE,
    MESSAGE_ID_KEY,
    VOICE_TTS_KEY,
)
from app.constants.hil import HIL_PAUSED_LOCK_TTL_SECONDS, HIL_RESUME_CONFIG_KEY
from app.constants.log_tags import LogTag
from app.core.stream_manager import StreamManager
from app.models.agent_models import AgentConfigurable, AgentConfigurableView
from app.models.chat_models import ToolDataEntry
from app.services.analytics_service import AnalyticsEvents, capture_event
from app.services.hil.approvals_store import set_resume_item
from app.services.hil.resume_slot import release_resume_dispatch
from app.services.latency_metrics import (
    observe_executor_active,
    observe_executor_e2e,
    observe_executor_queue_wait,
    observe_executor_run_total,
    observe_executor_ttft,
    span,
)
from app.utils.agent_utils import format_sse_data
from app.utils.background_tasks import spawn_background_task
from shared.py.wide_events import WorkflowContext, get_trace_id, log, wide_task

#: Task name for a queued executor run. Tests drain by this name to wait out
#: exactly the runs a turn handed off, not every background task in the process.
DETACHED_EXECUTOR_TASK_NAME = "detached-executor-run"


@traceable(name="executor_background", run_type="chain")
async def run_executor_background(
    run: ExecutorRun,
    task: str,
    configurable: AgentConfigurable,
    resume: Command | None = None,
) -> None:
    """Run (or resume) the executor agent in background and hand its result to delivery.

    Never raises — exceptions route through comms as an <executor_error> message.
    A paused run (resume continuing a HIL approval) keeps the busy lock instead of
    delivering, since the thread has pending work until the approval resolves.
    """
    # This task outlives the spawning request/turn, so it needs its own
    # wide-event boundary or every log.set() (LLM accounting included) is
    # silently discarded. get_trace_id() correlates it back to the dispatcher.
    run_start = time.perf_counter()
    # The busy-lock queue origin, not ``kind``: a HIL resume is RunKind.QUEUED
    # but never waited on the lock, so it must not label as queued.
    queued = run.queued
    async with wide_task(
        "executor_run",
        trace_id=get_trace_id() or None,
        conversation_id=run.conversation_id,
        stream_id=run.stream_id,
        task_id=run.task_id,
        # Surface the turn came from, carried so auxiliary calls inside this run
        # (handed a bare config) can still name it — without it a web turn is
        # metered as "system", undercounting COGS exactly where it matters most.
        conversation_source=AgentConfigurableView.model_validate(configurable).conversation_source,
        # The workflow run this executor is part of; the workflow task stamped it
        # on ITS boundary, this one is fresh. Without it every model call lands in
        # the ledger with no execution, so "what did this run cost" reads only comms.
        **(
            {
                "workflow": WorkflowContext(
                    id=run.workflow_id, execution_id=run.workflow_execution_id
                )
            }
            if run.workflow_id and run.workflow_execution_id
            else {}
        ),
    ):
        result_text = ""
        result_type = "final"
        run_ctx: SubagentExecutionContext | None = None
        queue_wait_ms = _queue_wait_ms(run, run_start, configurable, queued=queued)
        ttft_ms: float | None = None
        active_ms: float | None = None

        # One lifecycle event per run segment; a resumed run re-enters here.
        executor_user_id = run.user.user_id
        run_props = _run_props(run)
        if executor_user_id:
            capture_event(executor_user_id, AnalyticsEvents.AGENT_RUN_STARTED, run_props)

        try:
            with span() as elapsed_active:
                result = await _execute_executor(task, configurable, run.stream_id, resume)
            active_ms = round(elapsed_active() * 1000.0, 2)
            result_text, result_type, run_ctx = result.text, result.type, result.ctx
            ttft_ms = _executor_ttft_ms(run, run_start)
            if ttft_ms is not None:
                observe_executor_ttft(ttft_ms / 1000.0, queued=queued)
            timing_fields = _timing_fields(queue_wait_ms, ttft_ms, active_ms)
            log.set(executor={"queued": queued, **timing_fields})
            if result.paused_on and not await _record_pause(
                run, task, configurable, result.paused_on
            ):
                # Recording failed, so no decision can ever resume this thread; finalizing
                # as paused would hold the busy lock for its full TTL waiting for a resume
                # that can't come. Fail instead: the lock releases and the sweep closes it.
                result_text, result_type = EXECUTOR_APPROVAL_LOST_MESSAGE, "error"
            # Cancellation and the pause-record outcome are only known here, so
            # read both after the pause decision: the span must carry the same
            # status finalize records, not the pre-pause guess.
            run_cancelled = bool(run.stream_id) and await StreamManager.is_cancelled(run.stream_id)
            observe_executor_active(
                active_ms / 1000.0, status=_active_status(result_type, run_cancelled)
            )
            log.info(
                f"{LogTag.AGENT} Background executor finished",
                result_type=result_type,
                task_id=run.task_id,
                stream_id=run.stream_id,
            )
            _capture_executor_terminal(
                run,
                run_props=run_props,
                queued=queued,
                timing_fields=timing_fields,
                result_type=result_type,
            )
        finally:
            await _finalize_executor_run(run, task, result_text, result_type, run_ctx)
            if resume is not None:
                # This run held the conversation's resume slot (claimed at dispatch).
                # Freeing it AFTER finalize means the next decision can dispatch only
                # once this run's pause/completion bookkeeping is fully written.
                await release_resume_dispatch(run.conversation_id)


def _run_props(run: ExecutorRun) -> dict[str, str]:
    """Build the lifecycle props shared by the start and terminal events."""
    props: dict[str, str] = {
        "agent": "executor",
        "mode": "background",
        "conversation_id": run.conversation_id,
    }
    if run.task_id:
        props["task_id"] = run.task_id
    return props


def _timing_fields(
    queue_wait_ms: float | None, ttft_ms: float | None, active_ms: float | None
) -> dict[str, float]:
    """Collect the measured executor timings, omitting spans that never happened."""
    fields: dict[str, float] = {}
    if queue_wait_ms is not None:
        fields["queue_wait_ms"] = queue_wait_ms
    if ttft_ms is not None:
        fields["executor_ttft_ms"] = ttft_ms
    if active_ms is not None:
        fields["executor_active_ms"] = active_ms
    return fields


def _queue_wait_ms(
    run: ExecutorRun, run_start: float, configurable: AgentConfigurable, *, queued: bool
) -> float | None:
    """Observe dispatch-to-start queue wait, or None without a usable stamp.

    A queued run can survive a restart inside its 1h TTL, mixing monotonic
    epochs — a negative delta is garbage, not a measurement.
    """
    if run.t_dispatch_perf is None:
        return None
    queue_wait_s = run_start - run.t_dispatch_perf
    if queue_wait_s < 0.0:
        return None
    observe_executor_queue_wait(
        queue_wait_s,
        source=str(configurable.get("conversation_source") or "unknown"),
        queued=queued,
    )
    return round(queue_wait_s * 1000.0, 2)


def _active_status(result_type: str, cancelled: bool) -> str:
    """Label the active span with the same statuses finalize records."""
    if result_type == "error":
        return "error"
    if result_type == EXECUTOR_PAUSED:
        return "paused"
    return "cancelled" if cancelled else "success"


def _capture_executor_terminal(
    run: ExecutorRun,
    *,
    run_props: dict[str, str],
    queued: bool,
    timing_fields: dict[str, float],
    result_type: str,
) -> None:
    """Emit the run's terminal lifecycle event with its measured timings."""
    user_id = run.user.user_id
    if not user_id or result_type not in ("final", "error"):
        return
    event = (
        AnalyticsEvents.AGENT_RUN_COMPLETED
        if result_type == "final"
        else AnalyticsEvents.AGENT_RUN_FAILED
    )
    capture_event(
        user_id,
        event,
        {**run_props, "queued": queued, **timing_fields},
        dedupe_key=run.task_id or run.stream_id,
    )


async def _record_pause(
    run: ExecutorRun, task: str, configurable: AgentConfigurable, approval_ids: tuple[str, ...]
) -> bool:
    """Attach this run's re-dispatch context to every approval it paused on.

    Each paused approval gets the same resume context so whichever decision lands
    first can re-dispatch the run. Returns whether every write landed — it is the only thing that makes
    the pause resumable, so a failure is not something the run can carry on
    through: the caller fails the run rather than parking it forever.
    """
    try:
        item = build_run_item(
            task=task,
            configurable=configurable,
            # A pause re-dispatch is a new incarnation: drop the original stamp
            # so the resumed run measures no queue wait for user decision time,
            # and clear the queue origin — it did not wait on the busy lock.
            identity=replace(run.identity, t_dispatch_perf=None, queued=False),
            workflow_execution_id=run.workflow_execution_id,
        )
        for approval_id in approval_ids:
            await set_resume_item(approval_id, item)
        return True
    except Exception as e:  # a lost pause must fail the run, not the process
        log.error(
            f"{LogTag.HIL} Could not record resume context; failing the paused run",
            approval_ids=list(approval_ids),
            stream_id=run.stream_id,
            task_id=run.task_id,
            error=str(e),
        )
        return False


def _executor_ttft_ms(run: ExecutorRun, run_start: float) -> float | None:
    """First executor frame minus dispatch (or run start without a stamp).

    None when no frame was written, or the delta is negative (mixed
    monotonic epochs across a restart) — missing beats zero-filled.
    """
    session = get_session(run.stream_id)
    first_frame = session.executor_first_frame_perf if session is not None else None
    if first_frame is None:
        return None
    base = run.t_dispatch_perf if run.t_dispatch_perf is not None else run_start
    ttft_ms = round((first_frame - base) * 1000.0, 2)
    return ttft_ms if ttft_ms >= 0.0 else None


class _ExecutorResult(NamedTuple):
    """One executor run's terminal shape.

    ``paused_on`` holds the approval id(s) when the run stopped on a HIL
    interrupt instead of finishing — one for a gate pause (batch pauses arrive
    with the HIL rework).

    ``ctx`` is the prepared execution context, kept so finalize can read the
    thread this run actually wrote; ``None`` when preparation itself failed, in
    which case no model call happened and nothing was committed."""

    text: str
    type: str
    paused_on: tuple[str, ...] = ()
    ctx: SubagentExecutionContext | None = None


class _PauseInterrupt(BaseModel):
    """The approval ids a subagent's interrupt payload carries.

    Both are ``object``: the payload is the gate's raw interrupt value, and the
    reader below keeps its own guards (a non-list batch is ignored, a single id
    is stringified) rather than letting validation reject a pause outright.
    """

    model_config = ConfigDict(extra="ignore")

    approval_ids: object = None
    approval_id: object = ""


def _paused_approval_ids(interrupt: _PauseInterrupt) -> tuple[str, ...]:
    """Approval ids from an interrupt payload — batch shape first, then single."""
    batch = interrupt.approval_ids
    if isinstance(batch, list):
        ids = tuple(str(a) for a in batch if a)
        if ids:
            return ids
    single = str(interrupt.approval_id)
    return (single,) if single else ()


async def _execute_executor(
    task: str,
    configurable: AgentConfigurable,
    stream_id: str,
    resume: Command | None = None,
) -> _ExecutorResult:
    """Run the executor agent graph once. Never raises on error.

    Errors return as _ExecutorResult(text, "error"). Tool events stream to the
    session's collector via make_redis_stream_writer. The executor inherits comms'
    model/provider/reasoning from configurable (free -> Gemini, paid -> MiniMax M3).
    """
    ctx: SubagentExecutionContext | None = None
    try:
        with span() as elapsed_prep:
            ctx, error = await prepare_executor_execution(
                task=task,
                configurable=configurable,
                stream_id=stream_id,
            )
        log.set(executor={"prep_ms": round(elapsed_prep() * 1000.0, 2)})
        if error or ctx is None:
            log.error(f"{LogTag.AGENT} Executor prep failed", error=error)
            return _ExecutorResult(error or "Executor agent not available", "error")
        if resume is not None:
            # Tells the handoff tool to probe its subagent thread for a parked
            # interrupt — only a resume replay can encounter one, so fresh runs
            # skip that per-handoff checkpoint read.
            ctx.configurable[HIL_RESUME_CONFIG_KEY] = True
            ctx.config.setdefault("configurable", {})[HIL_RESUME_CONFIG_KEY] = True
        writer = make_redis_stream_writer(stream_id)
        outcome = await execute_subagent_stream(ctx=ctx, stream_writer=writer, resume=resume)
        if outcome.paused:
            approval_ids = _paused_approval_ids(
                _PauseInterrupt.model_validate(outcome.interrupt or {})
            )
            if not approval_ids:
                # Unresumable: nothing can ever re-dispatch this thread. Fail the
                # run loudly rather than leave the conversation's lock held.
                log.error(f"{LogTag.HIL} Executor paused with no approval_id", stream_id=stream_id)
                return _ExecutorResult("Approval request was malformed", "error", ctx=ctx)
            return _ExecutorResult("", EXECUTOR_PAUSED, approval_ids, ctx=ctx)
        return _ExecutorResult(outcome.text, "final", ctx=ctx)
    except GraphRecursionError as e:
        # The executor exhausted its recursion budget. Log the real cause loudly,
        # but hand comms a friendly message instead of the raw traceback string so
        # the user sees actionable guidance rather than an internal error.
        log.error(
            f"{LogTag.AGENT} Executor hit recursion limit",
            stream_id=stream_id,
            error=str(e),
        )
        return _ExecutorResult(EXECUTOR_STEP_LIMIT_MESSAGE, "error", ctx=ctx)
    except Exception as e:
        log.error(f"{LogTag.AGENT} Executor run failed", stream_id=stream_id, error=str(e))
        return _ExecutorResult(str(e), "error", ctx=ctx)


async def _finalize_executor_run(
    run: ExecutorRun,
    task: str,
    result_text: str,
    result_type: str,
    ctx: SubagentExecutionContext | None = None,
) -> None:
    """Post-run cleanup, in order: signal done → deliver → free the lock → hand it on."""
    if result_type == EXECUTOR_PAUSED:
        await _finalize_paused_run(run)
        return

    was_cancelled = bool(run.stream_id) and await StreamManager.is_cancelled(run.stream_id)

    # Snapshot returned-cards BEFORE signalling done: live streams tear down the
    # session in parallel once done_event fires, so reading after would race it.
    # Only meaningful where cards render — a bot/workflow delivery has no card to fall back on.
    build_note = not was_cancelled and run.renders_native_cards
    returned_note = build_returned_to_frontend_note(run.stream_id) if build_note else ""

    # Snapshot cards delivery will persist, same reason: every comms consumer
    # tears the session down the moment done_event fires, so a read from inside
    # delivery comes back empty. None means a live run — comms owns those cards.
    tool_data = drain_executor_tool_data(run.stream_id) if run.executor_owns_tool_data else None

    # Held approval cards go live here, not mid-run: the open client never
    # renders them until a full refresh otherwise. Before the done signal so
    # the session is still alive; failure only skips the live push, never
    # finalize — the drain above already persisted the frames.
    try:
        from app.services.hil.bridge import (  # noqa: PLC0415 -- runner is imported too broadly for a top-level hil import
            flush_held_approval_cards,
        )

        await flush_held_approval_cards(run.stream_id)
    except Exception as e:
        log.error(
            f"{LogTag.AGENT} Held card flush failed",
            stream_id=run.stream_id,
            error_type=type(e).__name__,
        )

    # Signal SSE consumer that tool events are done so it can drain the session
    # into the comms ack and publish [DONE]. Comms re-narration runs in parallel.
    signal_executor_done(
        run.stream_id,
        failed=result_type == "error",
        reason=result_text if result_type == "error" else None,
    )

    # The waiter gave up on this executor and closed its run as failed. A result
    # delivered now would answer a turn that is over, and a collection queued
    # now would start work on it; only the lock release below is still owed.
    abandoned = executor_abandoned(run.stream_id)
    if abandoned:
        log.warning(
            f"{LogTag.AGENT} Executor finished after its waiter gave up; result not delivered",
            stream_id=run.stream_id,
            task_id=run.task_id,
            result_type=result_type,
        )

    # Delivery is best-effort: a failure here must NOT skip the lock release and
    # queue handoff below, or queued tasks strand and the busy lock leaks until
    # its TTL. The lock lifecycle is the load-bearing step — always run it.
    try:
        if not abandoned:
            await _deliver_terminal_outcome(
                run,
                task,
                TerminalOutcome(
                    result_text=result_text,
                    result_type=result_type,
                    was_cancelled=was_cancelled,
                    returned_note=returned_note,
                    tool_data=tool_data,
                ),
            )
    except Exception as e:  # never let delivery failure strand the queue
        log.error(
            f"{LogTag.AGENT} Executor finalize delivery failed",
            stream_id=run.stream_id,
            task_id=run.task_id,
            error=str(e),
        )

    # Release the busy lock now, not at end of finalize: held longer, comms'
    # executor_status hook keeps reading "still running," and a mid-finalize
    # exception would strand it for the full 30-min TTL. Ownership-checked.
    try:
        await release_lock_if_owned(run.conversation_id, run.stream_id, run.task_id)
        await _close_queued_stream(run, was_cancelled)
    except Exception as e:
        log.error(
            f"{LogTag.AGENT} Executor finalize lock release / stream close failed",
            stream_id=run.stream_id,
            task_id=run.task_id,
            error=str(e),
        )

    end_status = (
        "error" if result_type == "error" else ("cancelled" if was_cancelled else "success")
    )
    queued = run.queued
    if run.t_dispatch_perf is not None:
        e2e_s = time.perf_counter() - run.t_dispatch_perf
        if e2e_s >= 0.0:  # mixed-epoch guard, same as queue wait above
            observe_executor_e2e(e2e_s, status=end_status, queued=queued)
    observe_executor_run_total(status=end_status, queued=queued)

    # Work handed over mid-run is normally absorbed by the run itself. The one
    # case it cannot be is a hand-off that lands after this run's LAST model
    # call — there is no further reasoning step to read it. Carry that into a
    # fresh run rather than leaving it to sit. Runs on EVERY terminal path,
    # cancelled included: a Stop targets the running task, not work the user
    # added afterwards.
    await _carry_pending_into_new_run(run, ctx)


async def _finalize_paused_run(run: ExecutorRun) -> None:
    """Close out a run parked on a HIL approval without ending its turn.

    Doesn't deliver a result, drain the queue, or release the busy lock — it stays
    held until resolve_approval resumes this thread. Re-arms the lock's TTL to
    cover the approval window, and still signals SSE so the user sees the approval card.
    """
    if not await extend_lock_if_owned(
        run.conversation_id, run.stream_id, run.task_id, HIL_PAUSED_LOCK_TTL_SECONDS
    ):
        # Someone else owns the conversation (or Redis is down), so this pause is
        # already at risk of being trampled. Nothing to do but say so loudly.
        log.warning(
            f"{LogTag.HIL} Could not extend busy lock for paused run; the approval "
            "may be orphaned if the lock lapses",
            task_id=run.task_id,
            conversation_id=run.conversation_id,
            stream_id=run.stream_id,
        )
    signal_executor_done(run.stream_id)
    await _close_queued_stream(run, was_cancelled=False)
    observe_executor_run_total(status="paused", queued=run.queued)
    log.info(
        f"{LogTag.HIL} Executor paused on approval; busy lock retained",
        task_id=run.task_id,
        conversation_id=run.conversation_id,
        stream_id=run.stream_id,
    )


@dataclass(frozen=True)
class TerminalOutcome:
    """The terminal facts of one executor run, as _finalize_run snapshotted them.

    tool_data is None for a live run, whose cards the comms stream owns.
    """

    result_text: str
    result_type: str
    was_cancelled: bool
    returned_note: str
    tool_data: list[ToolDataEntry] | None


async def _deliver_terminal_outcome(
    run: ExecutorRun,
    task: str,
    outcome: TerminalOutcome,
) -> None:
    """Route the run's terminal outcome to exactly one delivery entry point.

    A cancelled run's already-streamed cards must not vanish: self-owning runs
    persist them here, live runs defer to comms' attach step (None means that) —
    persisting here too would duplicate cards. A completed run with text narrates and delivers.
    """
    if outcome.was_cancelled:
        # Regardless of who owns the tool_data, comms' context must record the
        # cancellation — otherwise its last knowledge stays 'Task accepted...
        # I'm on it' and later turns claim the task is still running or done.
        await record_executor_cancellation(run.conversation_id, run.task_id, task)
        if outcome.tool_data is None:
            log.info(
                f"{LogTag.AGENT} Live executor cancelled; comms stream owns tool_data persistence",
                task_id=run.task_id,
                stream_id=run.stream_id,
            )
        else:
            await persist_cancelled_run(run, outcome.tool_data)
    elif outcome.result_text:
        notification_text, message_id = await deliver_result(
            run,
            outcome.result_text,
            outcome.result_type,
            outcome.returned_note,
            tool_data=outcome.tool_data,
        )
        await _publish_voice_tts(run.stream_id, notification_text, message_id)


async def _publish_voice_tts(
    stream_id: str, notification_text: str | None, message_id: str | None
) -> None:
    """Push the narrated answer on a voice-mode stream so the agent speaks AND bubbles it.

    The frame carries the message_id so the voice agent forwards it as a display
    frame, rendering off the data channel immediately; the later WebSocket push
    (same id) then reconciles in place instead of duplicating. Only live streams are ever voice mode.
    """
    if not notification_text:
        return
    session = get_session(stream_id)
    if session is not None and session.voice_mode:
        await StreamManager.publish_chunk(
            stream_id,
            format_sse_data({VOICE_TTS_KEY: notification_text, MESSAGE_ID_KEY: message_id}),
        )


async def _close_queued_stream(run: ExecutorRun, was_cancelled: bool) -> None:
    """Tear down a queued run's session and close the SSE stream it owns.

    Only queued runs own a stream the frontend subscribed to via
    executor.stream_started; live sessions are torn down by the chat path. A
    cancelled queued stream closes silently — the cancel already told the client
    — so no [DONE] / complete_stream.
    """
    if not run.is_queued:
        return
    teardown_executor_capture(run.stream_id)
    if not was_cancelled:
        await StreamManager.publish_chunk(run.stream_id, "data: [DONE]\n\n")
        await StreamManager.complete_stream(run.stream_id)


def _spawn_detached_run(prepared: PreparedQueuedTask, conversation_id: str) -> None:
    """Spawn a run that owns its own stream, as a GC-tracked background task."""
    spawn_background_task(
        run_executor_background(
            run=prepared.run,
            task=prepared.task,
            configurable=prepared.configurable,
        ),
        name=DETACHED_EXECUTOR_TASK_NAME,
    )

    log.info(
        f"{LogTag.AGENT} Detached executor run spawned",
        task_id=prepared.run.task_id,
        conversation_id=conversation_id,
        stream_id=prepared.run.stream_id,
    )


async def deliver_to_executor(
    conversation_id: str,
    configurable: AgentConfigurable,
    task: str,
    *,
    workflow_execution_id: str | None = None,
) -> None:
    """Give the executor work from outside a comms turn — the one way to do it.

    Whether a run already exists is an implementation detail of delivery, not two
    different behaviours: a live run absorbs the task through its inbox, and an
    idle conversation gets a run started to carry it. Either way the task ends up
    as a message in an executor thread, never as a second parallel answer.

    The busy check is a fast path, not the decision: the claim inside
    ``_start_executor_run`` is atomic, so a run that starts between the two lands
    the task in the inbox instead of racing a second run onto the same thread.
    """
    if not await is_executor_busy(conversation_id) and await _start_executor_run(
        conversation_id, configurable, task, workflow_execution_id=workflow_execution_id
    ):
        return
    await ExecutorInbox(conversation_id).append(str(uuid4()), task)
    log.info(
        f"{LogTag.AGENT} Work handed to the live executor run",
        conversation_id=conversation_id,
    )


async def _prepare_executor_run(
    conversation_id: str,
    configurable: AgentConfigurable,
    task: str,
    *,
    workflow_execution_id: str | None = None,
) -> PreparedQueuedTask | None:
    """Claim the conversation and materialize a detached run for ``task``.

    ``None`` means another run holds the conversation (or Redis is down) and
    nothing was started. Separate from spawning so a caller with inbox entries to
    retire can do it once the run exists but before it can read them back.
    """
    return await prepare_run_from_item(
        conversation_id,
        build_run_item(
            task=task,
            configurable={
                **configurable,
                "thread_id": conversation_id,
                "execution_mode": "interactive",
            },
            identity=RunIdentity(
                conversation_id=conversation_id,
                task_id=str(uuid4()),
                user_message_id=None,
            ),
            workflow_execution_id=workflow_execution_id,
        ),
        claim=LockClaim.ACQUIRE,
    )


async def _start_executor_run(
    conversation_id: str,
    configurable: AgentConfigurable,
    task: str,
    *,
    workflow_execution_id: str | None = None,
) -> bool:
    """Materialize and spawn a detached run for ``task``. Returns whether it started."""
    prepared = await _prepare_executor_run(
        conversation_id, configurable, task, workflow_execution_id=workflow_execution_id
    )
    if prepared is None:
        return False
    _spawn_detached_run(prepared, conversation_id)
    return True


async def _carry_pending_into_new_run(
    run: ExecutorRun, ctx: SubagentExecutionContext | None
) -> None:
    """Start a run for work this one finished too early to absorb.

    What counts as absorbed is read off the thread, not off a marker: an entry
    injected on a model call that then FAILED was never committed, and only the
    checkpoint can tell that apart from one the final call did commit.

    Only WORK starts a run. A stop notice is context for whatever the user does
    next — carrying it made a bare Stop spawn a fresh run whose task was
    "the task you were working on was INTERRUPTED". It stays pending instead,
    for the next run's drain hook to read.
    """
    try:
        inbox = ExecutorInbox(run.conversation_id)
        pending = await inbox.read()
        if not pending:
            return
        drain = decide_drain(pending, await thread_messages(ctx) if ctx else [])
        for entry in drain.retire:
            await inbox.retire(entry)
        carry = drain.inject
        if not any(entry.tag is not AgentTag.EXECUTOR_INTERRUPTED for entry in carry):
            return

        # Framed per entry, so a stop notice and the redirect that follows it
        # stay distinguishable to the model instead of merging into one blob.
        prepared = await _prepare_executor_run(
            run.conversation_id,
            {
                "user_id": run.user.user_id,
                "email": run.user.email or "",
                "user_name": run.user.name or "",
                "user_timezone": run.user.timezone,
            },
            "".join(wrap_agent_payload(entry.tag, entry.text) for entry in carry),
            workflow_execution_id=run.workflow_execution_id,
        )
        if prepared is None:
            log.info(
                f"{LogTag.AGENT} Pending work left for the run that holds the conversation",
                conversation_id=run.conversation_id,
                pending=len(carry),
            )
            return
        # Retired only now the run exists, and before it is spawned: a failed
        # claim must leave the work for whoever won, and the new run's drain hook
        # must not find these still pending and inject them a second time.
        for entry in carry:
            await inbox.retire(entry)
        _spawn_detached_run(prepared, run.conversation_id)
    except Exception as e:  # a failed carry must not break finalize
        log.error(
            f"{LogTag.AGENT} Could not carry pending work into a new run",
            conversation_id=run.conversation_id,
            error_type=type(e).__name__,
        )
