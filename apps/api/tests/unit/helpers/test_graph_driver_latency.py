"""Unit tests for graph-driver latency spans (execute_graph_streaming)."""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import patch

from langchain_core.messages import AIMessage, AIMessageChunk
from prometheus_client import REGISTRY

from app.helpers.agent_helpers import execute_graph_streaming

HELPERS = "app.helpers.agent_helpers"


class _ScriptedGraph:
    def __init__(self, events: list[tuple[Any, ...]]) -> None:
        self._events = events

    def astream(self, *_args: Any, **_kwargs: Any) -> AsyncGenerator[tuple[Any, ...], None]:
        events = self._events

        async def stream() -> AsyncGenerator[tuple[Any, ...], None]:
            for event in events:
                yield event

        return stream()


def _graph_count(status: str) -> float:
    return REGISTRY.get_sample_value("comms_graph_seconds_count", {"status": status}) or 0.0


def _config() -> Any:
    return {"agent_name": "comms_agent", "configurable": {"user_id": "u1"}}


async def test_streaming_run_observes_graph_span() -> None:
    graph = _ScriptedGraph(
        [
            ((), "messages", (AIMessageChunk(id="m1", content="hi"), {})),
            ((), "updates", {"agent": {"messages": [AIMessage(id="m1", content="hi")]}}),
        ]
    )
    before = _graph_count("success")
    frames = [frame async for frame in execute_graph_streaming(graph, {}, _config())]
    assert any(f.startswith("data: ") and "hi" in f for f in frames)
    assert _graph_count("success") == before + 1


async def test_first_text_yield_stamps_pipeline_ttft() -> None:
    graph = _ScriptedGraph(
        [
            ((), "messages", (AIMessageChunk(id="m1", content="hi"), {})),
            ((), "updates", {"agent": {"messages": [AIMessage(id="m1", content="hi")]}}),
        ]
    )
    with patch(f"{HELPERS}.log") as mock_log:
        [frame async for frame in execute_graph_streaming(graph, {}, _config())]
    ttft_sets = [
        call.kwargs["comms_pipeline_ttft_ms"]
        for call in mock_log.set.call_args_list
        if "comms_pipeline_ttft_ms" in call.kwargs
    ]
    assert len(ttft_sets) == 1
    assert ttft_sets[0] >= 0.0


async def test_run_without_text_stamps_no_pipeline_ttft() -> None:
    graph = _ScriptedGraph([])
    with patch(f"{HELPERS}.log") as mock_log:
        frames = [frame async for frame in execute_graph_streaming(graph, {}, _config())]
    assert frames[-2].startswith("nostream: ")
    assert not [
        call for call in mock_log.set.call_args_list if "comms_pipeline_ttft_ms" in call.kwargs
    ]
