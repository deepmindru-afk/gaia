"""Unit tests for Laminar turn telemetry.

Turn scopes run against the REAL Laminar SDK (fake key, empty instrument
set, own provider — the global tracer is never touched): scope creation,
output/attribute writes, and exit codes below are SDK truth, not mock
theater. Only the SDK's network edge is out of scope (faked where a test
forces a failure).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from opentelemetry.trace import SpanContext, StatusCode
from opentelemetry.trace.span import NonRecordingSpan

import pytest

import app.config.laminar as laminar_config
import app.services.laminar_service as laminar_module
from app.services.laminar_service import TurnScope, begin_turn, end_turn


@pytest.fixture(scope="module", autouse=True)
def _laminar_test_provider():
    """Initialize the real SDK once: manual spans work, nothing global changes.

    Empty instruments (no LangChain auto-patching of the test process) and
    ``set_global_tracer_provider=False`` (verified: the global provider stays
    a bare Proxy). Spans buffer locally; nothing is ever flushed in tests.
    The SDK sets OTEL_* env vars as a side effect — restored afterwards so
    the hermetic-env guard stays green.
    """
    import os

    from lmnr import Laminar

    saved_env = dict(os.environ)
    Laminar.initialize(
        project_api_key="test-only-fake-key",
        instruments=set(),
        set_global_tracer_provider=False,
    )
    yield
    os.environ.clear()
    os.environ.update(saved_env)


def _settings(api_key: str | None) -> SimpleNamespace:
    return SimpleNamespace(LMNR_PROJECT_API_KEY=api_key)


def _entered_scope() -> TurnScope:
    from lmnr import Laminar

    scope = Laminar.start_as_current_span(
        "comms_agent", user_id="u1", session_id="c1", metadata={"source": "web"}
    )
    span = scope.__enter__()
    return TurnScope(scope=scope, span=span)


class TestBeginTurn:
    def test_returns_none_when_api_key_missing(self) -> None:
        with patch("app.services.laminar_service.settings", _settings(None)):
            assert begin_turn(user_id="u1", conversation_id="c1") is None

    def test_returns_none_when_user_id_missing(self) -> None:
        with patch("app.services.laminar_service.settings", _settings("key-1")):
            assert begin_turn(user_id="", conversation_id="c1") is None

    def test_passes_input_ids_and_filtered_metadata(self) -> None:
        with patch("app.services.laminar_service.settings", _settings("key-1")):
            handle = begin_turn(
                user_id="u1",
                conversation_id="c1",
                user_input="hello",
                properties={"source": "web", "voice_mode": None},
            )

            assert handle is not None
            attrs = dict(handle.span.attributes)
            assert attrs["lmnr.association.properties.user_id"] == "u1"
            assert attrs["lmnr.association.properties.session_id"] == "c1"
            assert attrs["lmnr.span.input"] == '"hello"'
            assert "lmnr.association.properties.metadata.voice_mode" not in attrs
            handle.scope.__exit__(None, None, None)

    def test_begin_failure_returns_none_without_raising(self) -> None:
        with (
            patch("app.services.laminar_service.settings", _settings("key-1")),
            patch("app.services.laminar_service.Laminar") as mock_sdk,
            patch("app.services.laminar_service.log") as mock_log,
        ):
            mock_sdk.start_as_current_span.side_effect = RuntimeError("boom")

            assert begin_turn(user_id="u1", conversation_id="c1") is None
            mock_log.warning.assert_called_once_with(
                "laminar_begin_failed",
                error="boom",
                error_type="RuntimeError",
                conversation_id="c1",
            )

    def test_missing_key_logs_once_per_process(self) -> None:
        laminar_module._disabled_logged = False
        try:
            with (
                patch("app.services.laminar_service.settings", _settings("  ")),
                patch("app.services.laminar_service.log") as mock_log,
            ):
                assert begin_turn(user_id="u1", conversation_id="c1") is None
                assert begin_turn(user_id="u1", conversation_id="c1") is None

                mock_log.info.assert_called_once_with(
                    "laminar_disabled",
                    reason="LMNR_PROJECT_API_KEY unset; turns will not report",
                )
        finally:
            laminar_module._disabled_logged = False


class TestEndTurn:
    def test_none_scope_is_noop(self) -> None:
        end_turn(None, output="hi")

    def test_success_sets_output_and_exits_clean(self) -> None:
        handle = _entered_scope()

        end_turn(handle, output="hi")

        assert dict(handle.span.attributes)["lmnr.span.output"] == '"hi"'
        assert handle.span.status.status_code is StatusCode.UNSET
        handle.scope.__exit__(None, None, None)

    def test_error_exits_with_exception_and_real_traceback(self) -> None:
        handle = _entered_scope()
        try:
            raise RuntimeError("provider down")
        except RuntimeError as error:
            assert error.__traceback__ is not None, "mutant guard: traceback must be real"
            end_turn(handle, output="boom", error=error)

            assert handle.span.status.status_code is StatusCode.ERROR
            assert [e.name for e in handle.span.events] == ["exception"]

    def test_cancelled_sets_tag_and_exits_clean(self) -> None:
        handle = _entered_scope()

        end_turn(handle, output="partial", cancelled=True)

        assert dict(handle.span.attributes)["cancelled"] is True
        assert handle.span.status.status_code is StatusCode.UNSET

    def test_span_update_failure_still_exits_without_raising(self) -> None:
        scope = MagicMock()
        span = MagicMock()
        span.set_output.side_effect = RuntimeError("otel down")
        handle = TurnScope(scope=scope, span=span)
        with patch("app.services.laminar_service.log") as mock_log:
            end_turn(handle, output="hi")

            scope.__exit__.assert_called_once_with(None, None, None)
            mock_log.warning.assert_called_once_with(
                "laminar_span_update_failed", error="otel down", error_type="RuntimeError"
            )

    def test_exit_failure_does_not_raise(self) -> None:
        scope = MagicMock()
        span = MagicMock()
        scope.__exit__.side_effect = RuntimeError("export down")
        handle = TurnScope(scope=scope, span=span)
        with patch("app.services.laminar_service.log") as mock_log:
            end_turn(handle, output="hi")

            mock_log.warning.assert_called_once_with(
                "laminar_end_failed", error="export down", error_type="RuntimeError"
            )

    def test_exit_reraise_of_turn_error_stays_quiet(self) -> None:
        scope = MagicMock()
        try:
            raise RuntimeError("turn blew up")
        except RuntimeError as error:
            caught = error
        scope.__exit__.side_effect = caught
        handle = TurnScope(scope=scope, span=MagicMock())
        with patch("app.services.laminar_service.log") as mock_log:
            end_turn(handle, output="boom", error=caught)

            mock_log.warning.assert_not_called()

    def test_uninitialized_sdk_degrades_without_raising(self) -> None:
        """Without init the SDK hands out NonRecordingSpans (no set_output):
        the turn still closes instead of raising."""
        scope = MagicMock()
        span = NonRecordingSpan(SpanContext(0, 0, False))
        handle = TurnScope(scope=scope, span=span)
        with patch("app.services.laminar_service.log") as mock_log:
            end_turn(handle, output="hi")

            scope.__exit__.assert_called_once_with(None, None, None)
            mock_log.warning.assert_called_once()
            assert mock_log.warning.call_args.args[0] == "laminar_span_update_failed"


class TestFlushGate:
    """Shutdown flushes only after a successful init — a failed init must
    not attempt a flush against an uninitialized SDK."""

    async def test_uninitialized_flush_is_noop(self) -> None:
        laminar_config._initialized = False
        try:
            with patch("app.config.laminar.Laminar") as mock_sdk:
                await laminar_config.flush_laminar()

                mock_sdk.flush.assert_not_called()
        finally:
            laminar_config._initialized = False

    async def test_initialized_flush_calls_sdk(self) -> None:
        laminar_config._initialized = True
        try:
            with patch("app.config.laminar.Laminar") as mock_sdk:
                await laminar_config.flush_laminar()

                mock_sdk.flush.assert_called_once_with()
        finally:
            laminar_config._initialized = False
