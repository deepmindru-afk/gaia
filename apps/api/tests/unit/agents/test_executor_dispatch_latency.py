"""Unit tests for executor dispatch + run latency spans (Tasks 5-6).

Dispatch stamps ``t_dispatch_perf``; the runner turns it into queue-wait,
TTFT, active and E2E observations. The LLM/graph itself is never driven —
``run_executor_background`` runs with its execute step stubbed and its
delivery/lock boundaries mocked, so the assertions cover the timing +
PostHog wiring only.
"""

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from prometheus_client import REGISTRY

from app.agents.core.background import executor_runner as er, session as sess
from app.agents.core.background.executor_runner import _ExecutorResult, run_executor_background
from app.agents.core.background.session import ExecutorRun, RunKind, get_session, teardown_session
from app.agents.tools import executor_tool as et
from app.constants.executor import EXECUTOR_PAUSED


def _count(name: str, labels: dict[str, str]) -> float:
    return REGISTRY.get_sample_value(f"{name}_count", labels) or 0.0


def _configurable(stream_id: str) -> dict[str, Any]:
    return {
        "stream_id": stream_id,
        "user_message_id": "umsg-1",
        "bot_message_id": "bmsg-1",
        "user_id": "user-1",
        "thread_id": "conv-1",
    }


def _run(stream_id: str, **overrides: Any) -> ExecutorRun:
    kwargs: dict[str, Any] = {
        "stream_id": stream_id,
        "conversation_id": "conv-1",
        "user": {"user_id": "user-1"},
        "kind": RunKind.LIVE,
        "task_id": "task-1",
        "user_message_id": "umsg-1",
        "bot_message_id": None,
    }
    kwargs.update(overrides)
    return ExecutorRun(**kwargs)


def _mock_redis(held_value: str | None) -> MagicMock:
    """Stand-in for the executor_tool redis_cache binding.

    Patches the module attribute (not the ``client`` property, which would
    lazily open a real connection).
    """
    mock_redis = MagicMock()
    mock_redis.client.get = AsyncMock(return_value=held_value)
    return mock_redis


class TestDispatchLatency:
    def setup_method(self) -> None:
        sess._sessions.clear()

    def teardown_method(self) -> None:
        sess._sessions.clear()

    async def test_busy_lock_queues_and_carries_dispatch_stamp(self) -> None:
        """A held lock queues the task, marks the session, and the queued item
        carries the dispatch stamp the runner later measures queue-wait from."""
        stream_id = "dispatch-queued"
        with (
            patch.object(et, "try_acquire_lock", AsyncMock(return_value=False)),
            patch.object(et, "redis_cache", _mock_redis("other-stream:other-task")),
            patch.object(et, "_acquire_lock_through_redirect", AsyncMock(return_value=False)),
            patch.object(et, "enqueue_task", AsyncMock()) as mock_enqueue,
            patch.object(et, "spawn_background_task") as mock_spawn,
        ):
            result = await et._dispatch_executor(
                task="do the thing",
                task_id="task-q",
                configurable=_configurable(stream_id),  # type: ignore[arg-type] -- minimal dispatch bag
                conversation_id="conv-1",
            )

        assert "queued" in result
        mock_spawn.assert_not_called()
        mock_enqueue.assert_called_once()
        session = get_session(stream_id)
        assert session is not None
        assert session.executor_queued_task_id == "task-q"
        item = mock_enqueue.call_args.args[1]
        assert item["task_id"] == "task-q"
        assert isinstance(item["t_dispatch_perf"], float)

    async def test_free_lock_spawns_live_run_with_dispatch_stamp(self) -> None:
        stream_id = "dispatch-live"
        captured: dict[str, Any] = {}
        real_from_config = ExecutorRun.from_configurable.__func__

        def _capture_from_config(
            cls: Any, configurable: Any, *, identity: Any, workflow_execution_id: Any = None
        ) -> ExecutorRun:
            run = real_from_config(
                cls, configurable, identity=identity, workflow_execution_id=workflow_execution_id
            )
            captured["run"] = run
            return run

        with (
            patch.object(et, "try_acquire_lock", AsyncMock(return_value=True)),
            patch.object(
                et.ExecutorRun, "from_configurable", new=classmethod(_capture_from_config)
            ),
            patch.object(et, "spawn_background_task"),
        ):
            result = await et._dispatch_executor(
                task="do the thing",
                task_id="task-live",
                configurable=_configurable(stream_id),  # type: ignore[arg-type] -- minimal dispatch bag
                conversation_id="conv-1",
            )

        assert "Task accepted (task_id: task-live)" in result
        session = get_session(stream_id)
        assert session is not None and session.executor_spawned is True
        assert isinstance(captured["run"].t_dispatch_perf, float)
        teardown_session(stream_id)

    async def test_same_turn_duplicate_dispatch_emits_nothing(self) -> None:
        stream_id = "dispatch-dup"
        with (
            patch.object(et, "try_acquire_lock", AsyncMock(return_value=False)),
            patch.object(et, "redis_cache", _mock_redis(f"{stream_id}:task-first")),
            patch.object(et, "enqueue_task", AsyncMock()) as mock_enqueue,
            patch.object(et, "spawn_background_task") as mock_spawn,
        ):
            result = await et._dispatch_executor(
                task="do it again",
                task_id="task-second",
                configurable=_configurable(stream_id),  # type: ignore[arg-type] -- minimal dispatch bag
                conversation_id="conv-1",
            )

        assert "already running" in result
        mock_enqueue.assert_not_called()
        mock_spawn.assert_not_called()
        assert get_session(stream_id) is None

    async def test_redirect_wait_is_measured_within_budget(self) -> None:
        """A cancel in flight lets the dispatch wait for the lock and run live;
        the wait is stamped on the wide event within the redirect budget."""
        stream_id = "dispatch-redirect"
        attempts = {"n": 0}

        async def _flaky_acquire(lock_key: str, lock_value: str) -> bool:
            attempts["n"] += 1
            return attempts["n"] >= 3

        with (
            patch.object(et, "try_acquire_lock", side_effect=_flaky_acquire),
            patch.object(et, "redis_cache", _mock_redis("dying-stream:old-task")),
            patch.object(et.StreamManager, "is_cancelled", AsyncMock(return_value=True)),
            patch.object(et, "spawn_background_task"),
            patch.object(et, "log") as mock_log,
        ):
            result = await et._dispatch_executor(
                task="redirected work",
                task_id="task-r",
                configurable=_configurable(stream_id),  # type: ignore[arg-type] -- minimal dispatch bag
                conversation_id="conv-1",
            )

        assert "Task accepted (task_id: task-r)" in result
        waits = [
            call.kwargs["tool"]["redirect_wait_ms"]
            for call in mock_log.set.call_args_list
            if "tool" in call.kwargs and "redirect_wait_ms" in call.kwargs["tool"]
        ]
        assert len(waits) == 1
        assert 0.0 <= waits[0] <= et.REDIRECT_CANCEL_WAIT_S * 1000.0
        teardown_session(stream_id)


class TestExecutorRunLatency:
    def setup_method(self) -> None:
        sess._sessions.clear()

    def teardown_method(self) -> None:
        sess._sessions.clear()

    async def _background(
        self,
        run: ExecutorRun,
        *,
        first_frame_at: float | None = None,
        result: _ExecutorResult | None = None,
        record_pause: bool | AsyncMock = True,
    ) -> MagicMock:
        """Drive run_executor_background with execute stubbed and delivery/lock
        boundaries mocked. Returns the PostHog capture mock."""
        if first_frame_at is not None:
            session = sess.create_session(run.stream_id, run.kind)
            session.executor_first_frame_perf = first_frame_at
        pause_recorder = (
            record_pause
            if isinstance(record_pause, AsyncMock)
            else AsyncMock(return_value=record_pause)
        )
        with (
            patch.object(
                er,
                "_execute_executor",
                AsyncMock(return_value=result or _ExecutorResult("done", "final")),
            ),
            patch.object(er, "_record_pause", pause_recorder),
            patch.object(er, "_finalize_paused_run", AsyncMock()),
            patch.object(er, "_deliver_terminal_outcome", AsyncMock()),
            patch.object(er, "release_lock_if_owned", AsyncMock()),
            patch.object(er, "_close_queued_stream", AsyncMock()),
            patch.object(er, "_queue_collection_if_uncollected", AsyncMock()),
            patch.object(er, "reclaim_stranded_task", AsyncMock(return_value=None)),
            patch.object(er, "capture_event") as mock_capture,
        ):
            await run_executor_background(
                run=run, task="do the thing", configurable={"conversation_source": "web"}
            )
        return mock_capture

    async def test_pause_record_failure_labels_the_active_span_error(self) -> None:
        """A pause whose resume context could not be written is failed as an
        error, so its active span must say ``error`` too — not the pre-pause
        ``paused`` — to agree with the E2E/run-total labels finalize emits."""
        run = _run("exec-pause-lost", t_dispatch_perf=time.perf_counter())
        error_before = _count("executor_active_seconds", {"status": "error"})
        paused_before = _count("executor_active_seconds", {"status": "paused"})

        mock_capture = await self._background(
            run,
            result=_ExecutorResult("", EXECUTOR_PAUSED, ("appr-1",)),
            record_pause=False,
        )

        assert _count("executor_active_seconds", {"status": "error"}) == error_before + 1
        assert _count("executor_active_seconds", {"status": "paused"}) == paused_before
        failed = [c for c in mock_capture.call_args_list if c.args[1] == "agent:run_failed"]
        assert len(failed) == 1

    async def test_recorded_pause_labels_the_active_span_paused(self) -> None:
        """A pause that records cleanly stays ``paused`` on the active span."""
        run = _run("exec-pause-ok", t_dispatch_perf=time.perf_counter())
        paused_before = _count("executor_active_seconds", {"status": "paused"})
        error_before = _count("executor_active_seconds", {"status": "error"})

        await self._background(
            run,
            result=_ExecutorResult("", EXECUTOR_PAUSED, ("appr-1",)),
            record_pause=True,
        )

        assert _count("executor_active_seconds", {"status": "paused"}) == paused_before + 1
        assert _count("executor_active_seconds", {"status": "error"}) == error_before

    async def test_active_histogram_agrees_with_the_active_ms_it_reports(self) -> None:
        """The histogram sample and the ``executor_active_ms`` sent to PostHog/Loki
        are the same measurement: recording the pause is bookkeeping I/O after the
        run went idle, and must not stretch one of them but not the other."""
        run = _run("exec-slow-pause-record", t_dispatch_perf=time.perf_counter())
        sum_before = (
            REGISTRY.get_sample_value("executor_active_seconds_sum", {"status": "error"}) or 0.0
        )

        async def _slow_pause_record(*_args: Any, **_kwargs: Any) -> bool:
            await asyncio.sleep(0.3)
            return False

        mock_capture = await self._background(
            run,
            result=_ExecutorResult("", EXECUTOR_PAUSED, ("appr-1",)),
            record_pause=AsyncMock(side_effect=_slow_pause_record),
        )

        failed = [c for c in mock_capture.call_args_list if c.args[1] == "agent:run_failed"]
        active_ms = failed[0].args[2]["executor_active_ms"]
        observed_s = (
            REGISTRY.get_sample_value("executor_active_seconds_sum", {"status": "error"})
            - sum_before
        )
        assert abs(observed_s - active_ms / 1000.0) < 0.05

    async def test_queue_wait_is_labelled_by_whether_the_run_was_queued(self) -> None:
        """A live run's dispatch->start gap is spawn delay, not queue wait; the two
        must be separable or a real backlog hides behind thousands of ~0s samples."""
        live_before = _count("executor_queue_wait_seconds", {"source": "web", "queued": "false"})
        queued_before = _count("executor_queue_wait_seconds", {"source": "web", "queued": "true"})

        await self._background(_run("exec-live-qw", t_dispatch_perf=time.perf_counter()))
        await self._background(
            _run(
                "exec-queued-qw",
                kind=RunKind.QUEUED,
                queued=True,
                t_dispatch_perf=time.perf_counter(),
            )
        )

        assert (
            _count("executor_queue_wait_seconds", {"source": "web", "queued": "false"})
            == live_before + 1
        )
        assert (
            _count("executor_queue_wait_seconds", {"source": "web", "queued": "true"})
            == queued_before + 1
        )

    async def test_hil_resume_is_not_labelled_queued(self) -> None:
        """A HIL resume runs on its own stream (``RunKind.QUEUED``) but never waited
        on the busy lock, so its run must not count as queued work."""

        def _run_total(queued: str) -> float:
            return (
                REGISTRY.get_sample_value(
                    "executor_run_total", {"status": "success", "queued": queued}
                )
                or 0.0
            )

        resume_before = _run_total("false")
        queued_before = _run_total("true")

        await self._background(_run("exec-resume", kind=RunKind.QUEUED, queued=False))

        assert _run_total("false") == resume_before + 1
        assert _run_total("true") == queued_before

    async def test_live_run_reports_queue_wait_ttft_and_e2e(self) -> None:
        stream_id = "exec-live"
        t_dispatch = time.perf_counter()
        run = _run(stream_id, t_dispatch_perf=t_dispatch)
        active_before = _count("executor_active_seconds", {"status": "success"})
        e2e_before = _count("executor_e2e_seconds", {"status": "success", "queued": "false"})

        mock_capture = await self._background(run, first_frame_at=t_dispatch + 0.4)

        assert _count("executor_active_seconds", {"status": "success"}) == active_before + 1
        assert (
            _count("executor_e2e_seconds", {"status": "success", "queued": "false"})
            == e2e_before + 1
        )
        completed = [
            call for call in mock_capture.call_args_list if call.args[1] == "agent:run_completed"
        ]
        assert len(completed) == 1
        props = completed[0].args[2]
        assert props["queued"] is False
        assert props["queue_wait_ms"] >= 0.0
        assert props["executor_ttft_ms"] >= 400.0
        assert props["executor_active_ms"] >= 0.0

    async def test_run_without_dispatch_stamp_omits_queue_wait(self) -> None:
        """Runs that predate the stamp (queued items written before deploy)
        degrade to missing timings, never zero-filled."""
        stream_id = "exec-legacy"
        run = _run(stream_id)
        mock_capture = await self._background(run)
        completed = [
            call for call in mock_capture.call_args_list if call.args[1] == "agent:run_completed"
        ]
        assert len(completed) == 1
        props = completed[0].args[2]
        assert "queue_wait_ms" not in props
        assert "executor_ttft_ms" not in props

    async def test_mixed_epoch_dispatch_stamp_measures_nothing(self) -> None:
        """A queued run that survived a restart carries a stamp from another
        monotonic epoch — its deltas are garbage and must not reach Prometheus
        or PostHog."""
        stream_id = "exec-restarted"
        run = _run(stream_id, t_dispatch_perf=time.perf_counter() + 3600.0)
        e2e_before = _count("executor_e2e_seconds", {"status": "success", "queued": "false"})
        mock_capture = await self._background(run)
        assert (
            _count("executor_e2e_seconds", {"status": "success", "queued": "false"}) == e2e_before
        )
        completed = [
            call for call in mock_capture.call_args_list if call.args[1] == "agent:run_completed"
        ]
        assert len(completed) == 1
        assert "queue_wait_ms" not in completed[0].args[2]
