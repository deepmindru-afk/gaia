"""The ARQ worker that runs browser jobs, on their own queue, inside the worker process.

ARQ gives every job a worker starts an in-progress key that lives for the longest
function timeout on that worker, and a job whose worker died is not picked up
again until the key expires. A browser job's deadline is hours, so registered on
the main worker it would hold every crashed 30-minute job for hours. This second
arq Worker keeps that TTL to browser jobs, which cannot resume after a crash
anyway: their browser session died with the process.
"""

import asyncio
import signal
import socket
from typing import Any

from arq.worker import Worker

from app.constants.browser import BROWSER_JOB_QUEUE, BROWSER_JOB_TASK
from app.constants.log_tags import LogTag
from app.services.browser.job_lifetime import browser_job_deadline_seconds
from app.utils.background_tasks import spawn_background_task
from app.workers.config.worker_settings import WorkerSettings
from app.workers.task_envelope import arq_function
from app.workers.tasks.browser_tasks import run_browser_job
from shared.py.wide_events import log

#: Where the running browser worker and its task live in the main worker's ctx.
BROWSER_WORKER_CTX_KEY = "browser_worker"


def build_browser_worker() -> Worker:
    """Build the arq Worker for the browser queue; signals stay with the main worker."""
    # One run per conversation is enforced by the browser slot lease, not by ARQ,
    # and a run is not idempotent (it may already have submitted a form): no retries.
    browser_job = arq_function(
        run_browser_job,
        name=BROWSER_JOB_TASK,
        timeout_seconds=browser_job_deadline_seconds(),
        max_tries=1,
        keep_result=0,
    )
    return Worker(
        functions=[browser_job],
        queue_name=BROWSER_JOB_QUEUE,
        redis_settings=WorkerSettings.redis_settings,
        handle_signals=False,
        max_jobs=WorkerSettings.max_jobs,
        keep_result=0,
        health_check_interval=WorkerSettings.health_check_interval,
        health_check_key=f"{BROWSER_JOB_QUEUE}:health:{socket.gethostname()}",
        allow_abort_jobs=True,
    )


def start_browser_worker(ctx: dict[str, Any]) -> None:
    """Start serving the browser queue beside the main worker, once the process is ready."""
    worker = build_browser_worker()
    task = spawn_background_task(
        worker.async_run(), name="browser_worker", on_done=_stop_process_if_it_ended
    )
    ctx[BROWSER_WORKER_CTX_KEY] = (worker, task)


async def stop_browser_worker(ctx: dict[str, Any]) -> None:
    """Cancel the browser worker's jobs (each ends through its cancel path) and close it."""
    running: tuple[Worker, asyncio.Task[None]] | None = ctx.pop(BROWSER_WORKER_CTX_KEY, None)
    if running is None:
        # Startup failed before it started; there is nothing to stop.
        return
    worker, task = running
    await worker.close()
    # close() only cancels a run that had got as far as starting its poll loop.
    task.cancel()
    await asyncio.wait({task})


def _stop_process_if_it_ended(task: asyncio.Task[None]) -> None:
    """Shut the whole worker down when the browser worker stops on its own.

    A process whose browser worker died would keep running every other job while
    browser jobs queue unserved; stopping it lets the orchestrator restart both.
    """
    if task.cancelled():
        return
    log.error(f"{LogTag.WORKER} Browser worker stopped; shutting the worker down")
    signal.raise_signal(signal.SIGTERM)
