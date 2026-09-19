"""The executor-facing browser-automation tools: start a run, collect it, unstick it.

The executor sees a start, a join and an answer, not the internals. The "do you want me to
use a browser?" confirmation is handled by the shared HIL system (``browser_task``
is registered destructive). This file owns only the turn's half of a run: the
settings and URL gates, the identity read off the run's config, the enqueue, and
the relay that replays the run's cards onto this turn. The run itself belongs to
the ARQ worker, through ``app/services/browser/job_runner.py``.
"""

import asyncio
from dataclasses import dataclass
from typing import Annotated
import uuid

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict

from app.config.settings import settings
from app.constants.browser import (
    BROWSER_JOB_JOINER_REFRESH_SECONDS,
    BROWSER_JOB_POLL_INTERVAL_SECONDS,
    BROWSER_JOB_TASK,
    BrowserSessionStatus,
    HandoffDecision,
)
from app.constants.log_tags import LogTag
from app.decorators import with_doc, with_rate_limiting
from app.models.chat_models import ConversationSource
from app.schemas.browser import BrowserResultSnapshot
from app.schemas.browser_job import BrowserJobRequest, BrowserJobState, BrowserJobStatus
from app.services.browser.agent_guidance import (
    clear_guidance_request,
    get_guidance_request,
    guidance_message,
)
from app.services.browser.handoff import resolve_handoff
from app.services.browser.job_relay import relay_job_events
from app.services.browser.job_runner import agent_result_message
from app.services.browser.jobs import (
    claim_conversation_slot,
    drop_joiner_lease,
    get_conversation_slot,
    get_job_state,
    put_job_state,
    refresh_joiner_lease,
    release_conversation_slot,
    take_joiner_lease,
)
from app.templates.docstrings.browser_tool_docs import (
    BROWSER_TASK,
    GUIDE_BROWSER_TASK,
    WAIT_FOR_BROWSER_TASK,
)
from app.utils.redis_utils import RedisPoolManager
from app.utils.url_safety import assert_safe_url_shape
from app.workers.queue import enqueue_worker_job
from shared.py.wide_events import log, spawn_logged_task

#: Handed back when the worker is gone: the run cannot be resumed and must not be
#: silently retried, so the executor is told the same thing a failed run tells it.
_DEAD_WORKER_RESULT = BrowserResultSnapshot(
    status=BrowserSessionStatus.FAILED,
    success=False,
    summary="the browser worker stopped unexpectedly",
)


@dataclass(frozen=True)
class _RunParams:
    """The run's identity and provenance, read once from the tool's config."""

    user_id: str
    conversation_id: str
    stream_id: str | None
    root_request_id: str | None
    source_category: str | None
    conversation_source: ConversationSource | None


class _RunConfigurable(BaseModel):
    """The keys of the run's configurable this tool reads; the rest is the graph's."""

    model_config = ConfigDict(extra="ignore")

    user_id: str | None = None
    conversation_id: str | None = None
    thread_id: str | None = None
    stream_id: str | None = None
    root_request_id: str | None = None
    source_category: str | None = None
    conversation_source: str | None = None


def _run_params(config: RunnableConfig) -> _RunParams:
    configurable = _RunConfigurable.model_validate(config.get("configurable", {}))
    conv_source = ConversationSource.coerce(configurable.conversation_source)
    return _RunParams(
        user_id=configurable.user_id or "",
        # The USER-facing conversation, never the executor's derived thread_id
        # (executor_<conv>): a handoff is resolved by a chat reply arriving on the
        # comms conversation id, so a prefixed key would never match.
        conversation_id=configurable.conversation_id or configurable.thread_id or "",
        stream_id=configurable.stream_id,
        root_request_id=configurable.root_request_id,
        source_category=configurable.source_category,
        conversation_source=conv_source,
    )


def _job_request(
    params: _RunParams, job_id: str, task: str, start_url: str | None
) -> BrowserJobRequest:
    return BrowserJobRequest(
        job_id=job_id,
        user_id=params.user_id,
        conversation_id=params.conversation_id,
        task=task,
        start_url=start_url,
        stream_id=params.stream_id,
        root_request_id=params.root_request_id,
        source_category=params.source_category,
        conversation_source=params.conversation_source,
    )


@tool
@with_rate_limiting("browser_task")
@with_doc(BROWSER_TASK)
async def browser_task(
    config: RunnableConfig,
    task: Annotated[str, "Clear, self-contained description of what to do in the browser."],
    start_url: Annotated[str | None, "Optional URL to open first."] = None,
) -> str:
    """Start a browser run as a background job and return immediately.

    Claims the conversation's one browser slot, hands the run to the worker, and
    starts the relay that puts the run's cards on this turn's stream.
    """
    params = _run_params(config)
    log.set(browser={"operation": "task", "source_category": params.source_category})

    if not settings.BROWSER_USE_ENABLED:
        return "Browser automation is currently disabled."
    if start_url and not settings.BROWSER_HOST_ALLOW_PRIVATE_NETWORK:
        try:
            assert_safe_url_shape(start_url)
        except ValueError as exc:
            return f"I can't open {start_url}: {exc}. Only public http(s) sites are reachable."

    job_id = uuid.uuid4().hex
    holder = await claim_conversation_slot(params.conversation_id, job_id)
    if holder is not None:
        return (
            f"A browser task is already running in this conversation (job {holder}). "
            "Call wait_for_browser_task() to collect it before starting another."
        )

    request = _job_request(params, job_id, task, start_url)
    await put_job_state(BrowserJobState(job_id=job_id, status=BrowserJobStatus.QUEUED, task=task))
    if not await _enqueue(request):
        await release_conversation_slot(params.conversation_id, job_id)
        return "I couldn't start the browser task right now. Try again in a moment."

    if params.stream_id:
        spawn_logged_task("browser_job_relay", relay_job_events(job_id, params.stream_id))
    return (
        f"Browser task started in the background (job {job_id}). Progress and the "
        "live-view link are streaming into this conversation. Call "
        "wait_for_browser_task() when you need the outcome; if you end the turn first, "
        "the result is delivered to the user as a follow-up."
    )


async def _enqueue(request: BrowserJobRequest) -> bool:
    """Put the run on the worker queue; False when it did not get there.

    A browser run is not idempotent, so there is no in-process fallback: a run
    that cannot be queued has not started, and the user is told so.
    """
    try:
        pool = await RedisPoolManager.get_pool()
        job = await enqueue_worker_job(pool, BROWSER_JOB_TASK, request.model_dump(mode="json"))
    except Exception as exc:
        log.error(
            f"{LogTag.BROWSER} Could not enqueue the browser job",
            error_type=type(exc).__name__,
            browser={"job_id": request.job_id},
        )
        return False
    if job is None:
        log.error(
            f"{LogTag.BROWSER} Browser job was not queued",
            browser={"job_id": request.job_id},
        )
        return False
    return True


@tool
@with_doc(WAIT_FOR_BROWSER_TASK)
async def wait_for_browser_task(
    config: RunnableConfig,
    # NOSONAR python:S7483 — `timeout` is part of this tool's LLM-facing input
    # schema (the model chooses how long to wait); it is not an internal call
    # timeout that an asyncio.timeout() context manager could replace.
    timeout: Annotated[  # NOSONAR python:S7483
        int, "Maximum seconds to wait for the browser task. Default 600."
    ] = 600,
) -> str:
    """Wait for this conversation's background browser task and return its outcome."""
    params = _run_params(config)
    job_id = await get_conversation_slot(params.conversation_id)
    if job_id is None:
        return "No browser task is running."

    stream_id = params.stream_id or ""
    await take_joiner_lease(job_id, stream_id)
    keep_lease = False
    try:
        outcome = await _poll_job(job_id, params.conversation_id, stream_id, timeout)
        keep_lease = outcome.keep_lease
        return outcome.message
    finally:
        # The worker stays silent while this lease is held, so a leaked one loses
        # the result entirely -- except on a guidance request, where this same
        # turn is coming straight back and is still the one joined.
        if not keep_lease:
            await drop_joiner_lease(job_id)


@dataclass(frozen=True)
class _JoinOutcome:
    """What a join returns, and whether this turn is still on the hook for the result."""

    message: str
    keep_lease: bool = False


async def _poll_job(
    job_id: str, conversation_id: str, stream_id: str, timeout: int
) -> _JoinOutcome:
    """Wait on the job's durable state, keeping this turn's claim on the result alive."""
    waited = 0.0
    since_refresh = 0.0
    while True:
        state = await get_job_state(job_id)
        if state is not None and state.status is BrowserJobStatus.DONE:
            return _JoinOutcome(state.agent_message or agent_result_message(_DEAD_WORKER_RESULT))
        pending = await get_guidance_request(job_id)
        if pending is not None:
            return _JoinOutcome(guidance_message(pending.request), keep_lease=True)
        if await get_conversation_slot(conversation_id) is None:
            # Only the worker heartbeats the slot, so its absence under a
            # non-terminal state means the run died with the process that held it.
            log.warning(
                f"{LogTag.BROWSER} Browser job lost its worker",
                browser={"job_id": job_id},
            )
            return _JoinOutcome(agent_result_message(_DEAD_WORKER_RESULT))
        if waited >= timeout:
            return _JoinOutcome(
                "The browser task is still running; it will be delivered to the user "
                "when it finishes."
            )
        await asyncio.sleep(BROWSER_JOB_POLL_INTERVAL_SECONDS)
        waited += BROWSER_JOB_POLL_INTERVAL_SECONDS
        since_refresh += BROWSER_JOB_POLL_INTERVAL_SECONDS
        if since_refresh >= BROWSER_JOB_JOINER_REFRESH_SECONDS:
            await refresh_joiner_lease(job_id, stream_id)
            since_refresh = 0.0


@tool
@with_doc(GUIDE_BROWSER_TASK)
async def guide_browser_task(
    config: RunnableConfig,
    instruction: Annotated[
        str, "ONE concrete next step for the browser operator, or empty when giving up."
    ] = "",
    give_up: Annotated[bool, "True when the task cannot honestly be done."] = False,
    reason: Annotated[str, "Why it cannot be done. Only with give_up."] = "",
) -> str:
    """Answer this conversation's stuck browser task with one instruction, or tell it to stop."""
    params = _run_params(config)
    log.set(browser={"operation": "guide"})

    job_id = await get_conversation_slot(params.conversation_id)
    pending = await get_guidance_request(job_id) if job_id else None
    if job_id is None or pending is None:
        return (
            "No browser task is waiting for guidance. Call wait_for_browser_task() to see "
            "where the run actually is."
        )
    # Still this turn's join: the round trip cost a model call, and an expired
    # lease would have the worker narrate the run behind the executor's back.
    await refresh_joiner_lease(job_id, params.stream_id or "")

    text = instruction.strip()
    if give_up:
        return await _resolve_guidance(
            job_id,
            pending.handoff_id,
            params.user_id,
            HandoffDecision.CANCEL,
            reason.strip(),
            "Told the browser to stop. Call wait_for_browser_task() for its final result.",
        )
    if not text:
        return (
            "An instruction is required. Call guide_browser_task with ONE concrete next "
            "step, or guide_browser_task(give_up=True, reason=...)."
        )
    return await _resolve_guidance(
        job_id,
        pending.handoff_id,
        params.user_id,
        HandoffDecision.CONTINUE,
        text,
        "Sent that to the browser. Call wait_for_browser_task() again to collect the outcome.",
    )


async def _resolve_guidance(
    job_id: str,
    handoff_id: str,
    user_id: str,
    decision: HandoffDecision,
    message: str,
    confirmation: str,
) -> str:
    """Answer the request and withdraw it here, not only in the worker.

    The run clears it too, but a join landing in that window would be handed the
    same stuck page again and spend another instruction answering nothing.
    """
    status = await resolve_handoff(handoff_id, decision, user_id, message)
    await clear_guidance_request(job_id)
    if status is None:
        return "The browser task stopped waiting for guidance; call wait_for_browser_task()."
    return confirmation
