"""Turn latency histograms — one benchmark surface for the full chat flow.

Prometheus is the benchmark source of truth (p50/p95/p99, alerts); PostHog
carries the same timings as event props for segmentation; wide events carry
per-turn fields for Loki deep-dives. This module owns the Prometheus half.

Label discipline: low-cardinality labels only
(``source, voice_mode, delegated, queued, status, stage, tool_name,
subagent_id, model, lane, agent, op``). Never user/conversation/stream/task
ids on collectors — those go on ``log.set()`` + PostHog props. ``tool_name``
is only safe for built-in tools; MCP-proxied or dynamically registered calls
collapse to ``tool_name="mcp"`` and keep the real name on the wide event.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import contextlib
import time
from typing import Final

from prometheus_client import Counter, Histogram

from app.services.storage.metrics import _register_once
from shared.py.wide_events import log

_TTFT_BUCKETS: Final[tuple[float, ...]] = (0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)

_TURN_BUCKETS: Final[tuple[float, ...]] = (
    0.05,
    0.1,
    0.25,
    0.5,
    1,
    2.5,
    5,
    10,
    30,
    60,
    120,
    300,
    600,
)

_CHAT_TTFT_SECONDS = _register_once(
    "chat_ttft_seconds",
    lambda: Histogram(
        name="chat_ttft_seconds",
        documentation="Time to first comms response text per chat turn in seconds",
        labelnames=("source", "voice_mode", "status"),
        buckets=_TTFT_BUCKETS,
    ),
)

_CHAT_E2E_ACK_SECONDS = _register_once(
    "chat_e2e_ack_seconds",
    lambda: Histogram(
        name="chat_e2e_ack_seconds",
        documentation="Request accepted to comms ack complete per chat turn in seconds",
        labelnames=("source", "voice_mode", "delegated", "status"),
        buckets=_TURN_BUCKETS,
    ),
)

_CHAT_E2E_FULL_SECONDS = _register_once(
    "chat_e2e_full_seconds",
    lambda: Histogram(
        name="chat_e2e_full_seconds",
        documentation="Request accepted to stream DONE per chat turn in seconds",
        labelnames=("source", "voice_mode", "delegated", "status"),
        buckets=_TURN_BUCKETS,
    ),
)

_LLM_TTFT_SECONDS = _register_once(
    "llm_ttft_seconds",
    lambda: Histogram(
        name="llm_ttft_seconds",
        documentation="True provider time to first token per streaming LLM call in seconds",
        labelnames=("model", "lane", "agent"),
        buckets=_TTFT_BUCKETS,
    ),
)

_COMMS_GRAPH_SECONDS = _register_once(
    "comms_graph_seconds",
    lambda: Histogram(
        name="comms_graph_seconds",
        documentation="Comms graph streaming run duration in seconds",
        labelnames=("status",),
        buckets=_TURN_BUCKETS,
    ),
)

_CONTEXT_ASSEMBLE_SECONDS = _register_once(
    "context_assemble_seconds",
    lambda: Histogram(
        name="context_assemble_seconds",
        documentation="Context assembly duration by stage in seconds",
        labelnames=("stage",),
        buckets=_TTFT_BUCKETS,
    ),
)

_EXECUTOR_QUEUE_WAIT_SECONDS = _register_once(
    "executor_queue_wait_seconds",
    lambda: Histogram(
        name="executor_queue_wait_seconds",
        documentation="Executor dispatch to run start wait in seconds",
        labelnames=("source",),
        buckets=_TURN_BUCKETS,
    ),
)

_EXECUTOR_TTFT_SECONDS = _register_once(
    "executor_ttft_seconds",
    lambda: Histogram(
        name="executor_ttft_seconds",
        documentation="Executor dispatch to first tool-data frame in seconds",
        labelnames=("queued",),
        buckets=_TURN_BUCKETS,
    ),
)

_EXECUTOR_ACTIVE_SECONDS = _register_once(
    "executor_active_seconds",
    lambda: Histogram(
        name="executor_active_seconds",
        documentation="Executor active run time excluding HIL pause in seconds",
        labelnames=("status",),
        buckets=_TURN_BUCKETS,
    ),
)

_EXECUTOR_E2E_SECONDS = _register_once(
    "executor_e2e_seconds",
    lambda: Histogram(
        name="executor_e2e_seconds",
        documentation="Executor dispatch to finalize in seconds",
        labelnames=("status", "queued"),
        buckets=_TURN_BUCKETS,
    ),
)

_TOOL_CALL_SECONDS = _register_once(
    "tool_call_seconds",
    lambda: Histogram(
        name="tool_call_seconds",
        documentation="Per tool call duration in seconds",
        labelnames=("tool_name", "status"),
        buckets=_TTFT_BUCKETS,
    ),
)

_TOOL_RETRIEVAL_SECONDS = _register_once(
    "tool_retrieval_seconds",
    lambda: Histogram(
        name="tool_retrieval_seconds",
        documentation="Tool retrieval fan-out duration in seconds",
        labelnames=("status",),
        buckets=_TTFT_BUCKETS,
    ),
)

_SUBAGENT_RUN_SECONDS = _register_once(
    "subagent_run_seconds",
    lambda: Histogram(
        name="subagent_run_seconds",
        documentation="Subagent active run time excluding pause in seconds",
        labelnames=("subagent_id", "status"),
        buckets=_TURN_BUCKETS,
    ),
)

_HIL_USER_WAIT_SECONDS = _register_once(
    "hil_user_wait_seconds",
    lambda: Histogram(
        name="hil_user_wait_seconds",
        documentation="HIL approval decided_at minus created_at in seconds",
        labelnames=(),
        buckets=_TURN_BUCKETS,
    ),
)

_HIL_DISPATCH_LAG_SECONDS = _register_once(
    "hil_dispatch_lag_seconds",
    lambda: Histogram(
        name="hil_dispatch_lag_seconds",
        documentation="HIL approval resumed_at minus decided_at in seconds",
        labelnames=(),
        buckets=_TTFT_BUCKETS,
    ),
)

_DELIVERY_NARRATION_SECONDS = _register_once(
    "delivery_narration_seconds",
    lambda: Histogram(
        name="delivery_narration_seconds",
        documentation="Executor result narration duration in seconds",
        labelnames=("status",),
        buckets=_TURN_BUCKETS,
    ),
)

_DELIVERY_PERSIST_SECONDS = _register_once(
    "delivery_persist_seconds",
    lambda: Histogram(
        name="delivery_persist_seconds",
        documentation="Result persistence write duration in seconds",
        labelnames=("op",),
        buckets=_TTFT_BUCKETS,
    ),
)

_TRANSPORT_REDIS_PUBLISH_SECONDS = _register_once(
    "transport_redis_publish_seconds",
    lambda: Histogram(
        name="transport_redis_publish_seconds",
        documentation="Redis stream publish duration in seconds",
        labelnames=(),
        buckets=_TTFT_BUCKETS,
    ),
)

_SSE_DELIVERY_SECONDS = _register_once(
    "sse_delivery_seconds",
    lambda: Histogram(
        name="sse_delivery_seconds",
        documentation="SSE subscribe to close delivery duration in seconds",
        labelnames=("status",),
        buckets=_TURN_BUCKETS,
    ),
)

_CHAT_TURN_TOTAL = _register_once(
    "chat_turn_total",
    lambda: Counter(
        name="chat_turn_total",
        documentation="Lifetime chat turn observations",
        labelnames=("source", "delegated", "status"),
    ),
)

_EXECUTOR_RUN_TOTAL = _register_once(
    "executor_run_total",
    lambda: Counter(
        name="executor_run_total",
        documentation="Lifetime executor run observations",
        labelnames=("status", "queued"),
    ),
)

_TOOL_CALL_TOTAL = _register_once(
    "tool_call_total",
    lambda: Counter(
        name="tool_call_total",
        documentation="Lifetime tool call observations",
        labelnames=("tool_name", "status"),
    ),
)

_HIL_PAUSE_TOTAL = _register_once(
    "hil_pause_total",
    lambda: Counter(
        name="hil_pause_total",
        documentation="Lifetime HIL pause observations",
    ),
)


def _bool_label(value: bool | str) -> str:
    return value if isinstance(value, str) else ("true" if value else "false")


def _observe(histogram: Histogram, amount: float, **labels: str) -> None:
    try:
        histogram.labels(**labels).observe(amount)
    except Exception as e:
        log.warning(
            "[metrics] latency observe failed",
            error_type=type(e).__name__,
        )


def _inc(counter: Counter, **labels: str) -> None:
    try:
        if labels:
            counter.labels(**labels).inc()
        else:
            counter.inc()
    except Exception as e:
        log.warning(
            "[metrics] latency counter inc failed",
            error_type=type(e).__name__,
        )


@contextlib.contextmanager
def span() -> Iterator[Callable[[], float]]:
    """Time one span, yielding an ``elapsed()`` reader in seconds."""
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


def observe_chat_ttft(seconds: float, *, source: str, voice_mode: bool, status: str) -> None:
    _observe(
        _CHAT_TTFT_SECONDS,
        seconds,
        source=source,
        voice_mode=_bool_label(voice_mode),
        status=status,
    )


def observe_chat_e2e_ack(
    seconds: float, *, source: str, voice_mode: bool, delegated: bool, status: str
) -> None:
    _observe(
        _CHAT_E2E_ACK_SECONDS,
        seconds,
        source=source,
        voice_mode=_bool_label(voice_mode),
        delegated=_bool_label(delegated),
        status=status,
    )


def observe_chat_e2e_full(
    seconds: float, *, source: str, voice_mode: bool, delegated: bool, status: str
) -> None:
    _observe(
        _CHAT_E2E_FULL_SECONDS,
        seconds,
        source=source,
        voice_mode=_bool_label(voice_mode),
        delegated=_bool_label(delegated),
        status=status,
    )


def observe_chat_turn_total(*, source: str, delegated: bool, status: str) -> None:
    _inc(
        _CHAT_TURN_TOTAL,
        source=source,
        delegated=_bool_label(delegated),
        status=status,
    )


def observe_llm_ttft(seconds: float, *, model: str, lane: str, agent: str) -> None:
    _observe(_LLM_TTFT_SECONDS, seconds, model=model, lane=lane, agent=agent)


def observe_comms_graph(seconds: float, *, status: str) -> None:
    _observe(_COMMS_GRAPH_SECONDS, seconds, status=status)


def observe_context_assemble(seconds: float, *, stage: str) -> None:
    _observe(_CONTEXT_ASSEMBLE_SECONDS, seconds, stage=stage)


def observe_executor_queue_wait(seconds: float, *, source: str) -> None:
    _observe(_EXECUTOR_QUEUE_WAIT_SECONDS, seconds, source=source)


def observe_executor_ttft(seconds: float, *, queued: bool) -> None:
    _observe(_EXECUTOR_TTFT_SECONDS, seconds, queued=_bool_label(queued))


def observe_executor_active(seconds: float, *, status: str) -> None:
    _observe(_EXECUTOR_ACTIVE_SECONDS, seconds, status=status)


def observe_executor_e2e(seconds: float, *, status: str, queued: bool) -> None:
    _observe(
        _EXECUTOR_E2E_SECONDS,
        seconds,
        status=status,
        queued=_bool_label(queued),
    )


def observe_executor_run_total(*, status: str, queued: bool) -> None:
    _inc(_EXECUTOR_RUN_TOTAL, status=status, queued=_bool_label(queued))


def observe_tool_call(seconds: float, *, tool_name: str, status: str) -> None:
    _observe(_TOOL_CALL_SECONDS, seconds, tool_name=tool_name, status=status)
    _inc(_TOOL_CALL_TOTAL, tool_name=tool_name, status=status)


def observe_tool_retrieval(seconds: float, *, status: str) -> None:
    _observe(_TOOL_RETRIEVAL_SECONDS, seconds, status=status)


def observe_subagent_run(seconds: float, *, subagent_id: str, status: str) -> None:
    _observe(_SUBAGENT_RUN_SECONDS, seconds, subagent_id=subagent_id, status=status)


def observe_hil_user_wait(seconds: float) -> None:
    _observe(_HIL_USER_WAIT_SECONDS, seconds)
    _inc(_HIL_PAUSE_TOTAL)


def observe_hil_dispatch_lag(seconds: float) -> None:
    _observe(_HIL_DISPATCH_LAG_SECONDS, seconds)


def observe_delivery_narration(seconds: float, *, status: str) -> None:
    _observe(_DELIVERY_NARRATION_SECONDS, seconds, status=status)


def observe_delivery_persist(seconds: float, *, op: str) -> None:
    _observe(_DELIVERY_PERSIST_SECONDS, seconds, op=op)


def observe_transport_redis_publish(seconds: float) -> None:
    _observe(_TRANSPORT_REDIS_PUBLISH_SECONDS, seconds)


def observe_sse_delivery(seconds: float, *, status: str) -> None:
    _observe(_SSE_DELIVERY_SECONDS, seconds, status=status)


__all__ = [
    "observe_chat_e2e_ack",
    "observe_chat_e2e_full",
    "observe_chat_ttft",
    "observe_chat_turn_total",
    "observe_comms_graph",
    "observe_context_assemble",
    "observe_delivery_narration",
    "observe_delivery_persist",
    "observe_executor_active",
    "observe_executor_e2e",
    "observe_executor_queue_wait",
    "observe_executor_run_total",
    "observe_executor_ttft",
    "observe_hil_dispatch_lag",
    "observe_hil_user_wait",
    "observe_llm_ttft",
    "observe_sse_delivery",
    "observe_subagent_run",
    "observe_tool_call",
    "observe_tool_retrieval",
    "observe_transport_redis_publish",
    "span",
]
