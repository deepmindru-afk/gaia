"""Unit tests for app.utils.stream_publishers.optional_stream_writer."""

from unittest.mock import MagicMock, patch

from langchain_core.runnables.config import var_child_runnable_config

from app.utils.stream_publishers import optional_stream_writer

MODULE = "app.utils.stream_publishers"


class TestOptionalStreamWriter:
    def test_returns_none_with_no_runnable_context(self) -> None:
        assert optional_stream_writer() is None

    def test_returns_none_under_dispatch_like_bare_config(self) -> None:
        """Ticket redeem invokes tools via dispatch with a synthesized config
        that carries no Pregel runtime — this is the shape that crashed."""
        token = var_child_runnable_config.set({"configurable": {"user_id": "u1"}})
        try:
            assert optional_stream_writer() is None
        finally:
            var_child_runnable_config.reset(token)

    def test_returns_writer_inside_a_graph_run(self) -> None:
        sink = MagicMock()
        with patch(f"{MODULE}.get_stream_writer", return_value=sink):
            assert optional_stream_writer() is sink
