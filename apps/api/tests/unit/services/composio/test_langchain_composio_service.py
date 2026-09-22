"""Tests for LangchainProvider (app/services/composio/langchain_composio_service.py)."""

from typing import Any
from unittest.mock import patch

from pydantic import BaseModel

from app.services.composio.langchain_composio_service import LangchainProvider, StructuredTool

MODULE = "app.services.composio.langchain_composio_service"


class TestObservabilityFailureDoesNotBreakTheToolCall:
    """The invocation-observability log call is wrapped in its own try/except so a logging failure can't take down the tool call."""

    def _action_func(self, execute_tool: Any) -> Any:
        return LangchainProvider()._wrap_action(
            tool="GMAIL_SEND_EMAIL",
            description="Send an email.",
            schema_params={},
            execute_tool=execute_tool,
            keywords={},
            toolkit="gmail",
        )

    def test_a_logging_failure_is_swallowed_and_the_tool_result_still_returns(self) -> None:
        tool_result = {"successful": False, "error": "invalid recipient", "data": None}
        action_func = self._action_func(execute_tool=lambda _tool, _kwargs: tool_result)

        with patch(f"{MODULE}.log") as mock_log:
            mock_log.set.side_effect = RuntimeError("log sink unreachable")
            result = action_func(__runnable_config__={"metadata": {"user_id": "user-1"}})

        # The observability failure must not surface as an exception, and must not
        # swap out the real tool result for anything else.
        assert result == tool_result

    def test_the_failure_is_reported_via_log_debug_with_the_original_error(self) -> None:
        action_func = self._action_func(
            execute_tool=lambda _tool, _kwargs: {"successful": True, "data": {"id": "msg-1"}}
        )

        with patch(f"{MODULE}.log") as mock_log:
            mock_log.set.side_effect = RuntimeError("log sink unreachable")
            action_func(__runnable_config__={"metadata": {"user_id": "user-1"}})

        mock_log.debug.assert_called_once()
        _, kwargs = mock_log.debug.call_args
        assert kwargs["tool"] == "GMAIL_SEND_EMAIL"
        assert kwargs["error"] == "log sink unreachable"
        assert kwargs["error_type"] == "RuntimeError"

    def test_a_successful_invocation_never_touches_the_debug_fallback(self) -> None:
        # Positive control: observability succeeding must not also emit the
        # failure-path debug log — only the caught failure does.
        action_func = self._action_func(
            execute_tool=lambda _tool, _kwargs: {"successful": True, "data": {"id": "msg-1"}}
        )

        with patch(f"{MODULE}.log") as mock_log:
            result = action_func(__runnable_config__={"metadata": {"user_id": "user-1"}})

        assert result == {"successful": True, "data": {"id": "msg-1"}}
        mock_log.debug.assert_not_called()


class _SendArgs(BaseModel):
    recipient: str
    count: int


def _send(recipient: str, count: int) -> str:
    return f"sent {count} to {recipient}"


class TestInvalidArgumentsReturnAFailure:
    def _tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=_send, name="send", description="Send.", args_schema=_SendArgs
        )

    def test_invalid_arguments_come_back_as_a_failure_result(self) -> None:
        result = self._tool().run({"recipient": "a@b.c", "count": "not-a-number"})

        assert isinstance(result, dict)
        assert result["successful"] is False
        assert result["data"] is None
        assert "count" in result["error"]

    def test_valid_arguments_still_run_the_tool(self) -> None:
        assert self._tool().run({"recipient": "a@b.c", "count": 2}) == "sent 2 to a@b.c"
