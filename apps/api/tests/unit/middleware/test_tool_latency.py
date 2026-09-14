"""Unit tests for per-tool-call latency spans (Task 7)."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp
from prometheus_client import REGISTRY
import pytest

from app.agents.middleware.executor import MiddlewareExecutor


def _count(tool_name: str, status: str) -> float:
    return (
        REGISTRY.get_sample_value(
            "tool_call_seconds_count", {"tool_name": tool_name, "status": status}
        )
        or 0.0
    )


def _invoke(
    executor: MiddlewareExecutor,
    tool_call: dict[str, Any],
    tool: Any,
    invoke_fn: Any,
) -> Any:
    state = {"messages": []}
    config: dict[str, Any] = {"configurable": {"user_id": "user-1"}}
    with (
        patch("app.agents.middleware.executor.create_tool_call_request"),
        patch(
            "app.agents.middleware.executor.BigtoolToolRuntime.from_graph_context",
            return_value=MagicMock(),
        ),
        patch("app.agents.middleware.executor.capture_event"),
    ):
        return executor.wrap_tool_invocation(tool_call, tool, state, config, None, invoke_fn)


async def test_successful_call_observes_tool_span() -> None:
    before = _count("lat-tool-ok", "success")
    invoke_fn = AsyncMock(return_value=ToolMessage(content="ok", tool_call_id="c1"))
    result = await _invoke(
        MiddlewareExecutor([]), {"name": "lat-tool-ok", "args": {}, "id": "c1"}, None, invoke_fn
    )
    assert result.content == "ok"
    assert _count("lat-tool-ok", "success") == before + 1


async def test_failing_call_observes_error_span() -> None:
    before = _count("lat-tool-bad", "error")
    invoke_fn = AsyncMock(side_effect=RuntimeError("tool exploded"))
    with pytest.raises(RuntimeError, match="tool exploded"):
        await _invoke(
            MiddlewareExecutor([]),
            {"name": "lat-tool-bad", "args": {}, "id": "c1"},
            None,
            invoke_fn,
        )
    assert _count("lat-tool-bad", "error") == before + 1


async def test_mcp_tool_collapses_to_mcp_label() -> None:
    """User-defined MCP server tool names are unbounded cardinality — the
    collector sees ``mcp`` while the wide event keeps the real name."""
    before = _count("mcp", "success")
    mcp_tool = MagicMock()
    mcp_tool.tool_connector = MagicMock()
    invoke_fn = AsyncMock(return_value=ToolMessage(content="ok", tool_call_id="c1"))
    await _invoke(
        MiddlewareExecutor([]),
        {"name": "lat-user-defined-tool", "args": {}, "id": "c1"},
        mcp_tool,
        invoke_fn,
    )
    assert _count("mcp", "success") == before + 1
    assert _count("lat-user-defined-tool", "success") == 0.0


async def test_hil_pause_is_neither_success_nor_error() -> None:
    """A gate interrupt is control flow, not a result: it propagates and
    leaves nospan — the pause is measured as HIL wait, not tool time."""
    from langchain.agents.middleware import AgentMiddleware
    from langchain.agents.middleware.types import ToolCallRequest

    class _Gate(AgentMiddleware):  # type: ignore[type-arg]
        async def awrap_tool_call(self, request: ToolCallRequest, handler: Any) -> Any:
            raise GraphBubbleUp("paused")

    for status in ("success", "error"):
        before = _count("lat-tool-gated", status)
        with pytest.raises(GraphBubbleUp):
            await _invoke(
                MiddlewareExecutor([_Gate()]),
                {"name": "lat-tool-gated", "args": {}, "id": "c1"},
                None,
                AsyncMock(),
            )
        assert _count("lat-tool-gated", status) == before
