"""The observability envelope every ARQ task runs behind.

arq_task is applied once per task in app.worker at registration time, so
a task cannot reach WorkerSettings without it. It owns the two things every
task needs and no task body should have to remember:

* the worker_task wide-event boundary, carrying the trace id propagated by
  app.workers.queue.enqueue_worker_job plus ARQ's job_id / job_try — the
  latter two are what make a retry chain queryable.
* the Prometheus duration/outcome metrics behind the arq-worker dashboard.
* the task's deadline. ARQ enforces a timeout by cancelling the task, which the
  boundary can only record as ``cancelled``; the envelope cuts the task off
  itself so the event reads ``failed`` with reason ``task_timeout``, and ARQ's
  timeout sits ARQ_BACKSTOP_GRACE_SECONDS past it as a backstop.

Task bodies therefore call log.set(...) directly: the boundary is already
open by the time they run.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
import functools
import time
from typing import Any, TypeVar

from arq.worker import Function, func as arq_func

from app.workers.config.worker_settings import (
    ARQ_BACKSTOP_GRACE_SECONDS,
    WORKER_JOB_TIMEOUT_SECONDS,
)
from app.workers.metrics import TASK_DURATION_SECONDS, TASK_TOTAL
from app.workers.queue import TRACE_ID_KWARG
from shared.py.wide_events import log, wide_task

T = TypeVar("T")

#: The worker_task event's reason when the envelope's deadline cut the task off.
REASON_TASK_TIMEOUT = "task_timeout"


def arq_task(
    func: Callable[..., Coroutine[Any, Any, T]],
    *,
    timeout_seconds: float = WORKER_JOB_TIMEOUT_SECONDS,
) -> Callable[..., Coroutine[Any, Any, T]]:
    """Wrap an ARQ task coroutine in the wide-event + metrics envelope, cut off at timeout_seconds.

    A task given its own timeout_seconds must be registered with ARQ through
    arq_function, or ARQ's default timeout would cancel it first.
    """

    task_name = func.__name__

    @functools.wraps(func)
    async def wrapper(ctx: dict[str, Any], *args: Any, **kwargs: Any) -> T:  # noqa: ANN401 -- ARQ's job API is dynamically typed upstream
        # Absent only when a caller (a test) invokes the task with a bare ctx;
        # omitting the keys beats emitting nulls the dashboards would have to skip.
        job_context = {key: ctx[key] for key in ("job_id", "job_try") if key in ctx}
        start = time.perf_counter()
        status = "success"
        try:
            async with wide_task(
                task_name,
                trace_id=kwargs.pop(TRACE_ID_KWARG, None),
                **job_context,
            ):
                deadline = asyncio.timeout(timeout_seconds)
                try:
                    async with deadline:
                        return await func(ctx, *args, **kwargs)
                except TimeoutError:
                    # A TimeoutError the task raised itself is its own failure.
                    if deadline.expired():
                        log.fail(REASON_TASK_TIMEOUT, timeout_seconds=timeout_seconds)
                    raise
        except Exception:
            status = "error"
            raise
        finally:
            elapsed = time.perf_counter() - start
            TASK_DURATION_SECONDS.labels(task_name=task_name, status=status).observe(elapsed)
            TASK_TOTAL.labels(task_name=task_name, status=status).inc()

    return wrapper


def arq_function(
    func: Callable[..., Coroutine[Any, Any, T]],
    *,
    name: str,
    timeout_seconds: int,
    max_tries: int | None = None,
    keep_result: float | None = None,
) -> Function:
    """Register a task that needs a deadline other than WORKER_JOB_TIMEOUT_SECONDS.

    The envelope's deadline and ARQ's backstop are set from the one number here,
    so they cannot drift into ARQ cutting the task off first.
    """
    return arq_func(
        arq_task(func, timeout_seconds=timeout_seconds),
        name=name,
        timeout=timeout_seconds + ARQ_BACKSTOP_GRACE_SECONDS,
        max_tries=max_tries,
        keep_result=keep_result,
    )
