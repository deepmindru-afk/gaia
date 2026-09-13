"""Unit tests for Latitude turn telemetry."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services.latitude_service import begin_turn, end_turn


def _settings(api_key: str | None) -> SimpleNamespace:
    return SimpleNamespace(LATITUDE_API_KEY=api_key, LATITUDE_PROJECT="gaia")


class TestBeginTurn:
    def test_returns_none_when_api_key_missing(self) -> None:
        with (
            patch("app.services.latitude_service.settings", _settings(None)),
            patch("app.services.latitude_service.capture") as mock_capture,
        ):
            assert begin_turn(user_id="u1", conversation_id="c1") is None
            mock_capture.start.assert_not_called()

    def test_returns_none_when_user_id_missing(self) -> None:
        with (
            patch("app.services.latitude_service.settings", _settings("key-1")),
            patch("app.services.latitude_service.capture") as mock_capture,
        ):
            assert begin_turn(user_id="", conversation_id="c1") is None
            mock_capture.start.assert_not_called()

    def test_passes_real_ids_and_filtered_metadata(self) -> None:
        scope = MagicMock()
        with (
            patch("app.services.latitude_service.settings", _settings("key-1")),
            patch("app.services.latitude_service.capture") as mock_capture,
        ):
            mock_capture.start.return_value = scope

            result = begin_turn(
                user_id="u1",
                conversation_id="c1",
                properties={"source": "web", "voice_mode": None},
            )

            assert result is scope
            name, options = mock_capture.start.call_args.args
            assert name == "comms_agent"
            assert options["user_id"] == "u1"
            assert options["session_id"] == "c1"
            assert options["project"] == "gaia"
            assert options["metadata"] == {"source": "web"}

    def test_sdk_failure_returns_none_without_raising(self) -> None:
        with (
            patch("app.services.latitude_service.settings", _settings("key-1")),
            patch("app.services.latitude_service.capture") as mock_capture,
        ):
            mock_capture.start.side_effect = RuntimeError("boom")

            assert begin_turn(user_id="u1", conversation_id="c1") is None


class TestEndTurn:
    def test_none_scope_is_noop(self) -> None:
        with patch("app.services.latitude_service.capture") as mock_capture:
            end_turn(None)
            mock_capture.end.assert_not_called()

    def test_success_ends_without_error(self) -> None:
        scope = MagicMock()
        with patch("app.services.latitude_service.capture") as mock_capture:
            end_turn(scope)

            mock_capture.end.assert_called_once_with(scope, None)

    def test_error_ends_with_error(self) -> None:
        scope = MagicMock()
        error = RuntimeError("provider down")
        with patch("app.services.latitude_service.capture") as mock_capture:
            end_turn(scope, error=error)

            mock_capture.end.assert_called_once_with(scope, error)

    def test_cancelled_ends_clean_with_attribute(self) -> None:
        scope = MagicMock()
        span = MagicMock()
        span.is_recording.return_value = True
        with (
            patch("app.services.latitude_service.capture") as mock_capture,
            patch("app.services.latitude_service.trace") as mock_trace,
        ):
            mock_trace.get_current_span.return_value = span

            end_turn(scope, cancelled=True)

            span.set_attribute.assert_called_once_with("cancelled", True)
            mock_capture.end.assert_called_once_with(scope, None)

    def test_error_dominates_cancelled(self) -> None:
        scope = MagicMock()
        error = RuntimeError("provider down")
        with (
            patch("app.services.latitude_service.capture") as mock_capture,
            patch("app.services.latitude_service.trace") as mock_trace,
        ):
            end_turn(scope, error=error, cancelled=True)

            mock_trace.get_current_span.assert_not_called()
            mock_capture.end.assert_called_once_with(scope, error)

    def test_sdk_failure_does_not_raise(self) -> None:
        scope = MagicMock()
        with patch("app.services.latitude_service.capture") as mock_capture:
            mock_capture.end.side_effect = RuntimeError("boom")

            end_turn(scope, error=RuntimeError("x"))
