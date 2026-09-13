"""Unit tests for the turn-telemetry fan-out mapping.

These run the real ``begin_turn_all``/``end_turn_all`` with the vendor
services faked one layer down, so the outcome mapping itself executes —
mocking the services here is the seam, not the thing under test.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.services.turn_telemetry import TurnOutcome, begin_turn_all, end_turn_all


@pytest.fixture
def services() -> MagicMock:
    with (
        patch("app.services.turn_telemetry.agnost_service") as mock_agnost,
        patch("app.services.turn_telemetry.latitude_service") as mock_latitude,
        patch("app.services.turn_telemetry.laminar_service") as mock_laminar,
    ):
        yield MagicMock(agnost=mock_agnost, latitude=mock_latitude, laminar=mock_laminar)


@pytest.mark.unit
class TestOutcomeValues:
    def test_three_distinct_outcomes(self) -> None:
        assert {TurnOutcome.SUCCESS, TurnOutcome.CANCELLED, TurnOutcome.FAILED} == {
            TurnOutcome("success"),
            TurnOutcome("cancelled"),
            TurnOutcome("failed"),
        }


@pytest.mark.unit
class TestBeginFanOut:
    def test_carries_real_ids_input_and_properties(self, services: MagicMock) -> None:
        begin_turn_all(
            user_id="u1",
            conversation_id="c1",
            user_input="hello",
            properties={"source": "web"},
        )

        agnost_kwargs = services.agnost.begin_turn.call_args.kwargs
        assert agnost_kwargs["user_id"] == "u1"
        assert agnost_kwargs["conversation_id"] == "c1"
        assert agnost_kwargs["user_input"] == "hello"
        assert agnost_kwargs["properties"] == {"source": "web"}
        assert services.latitude.begin_turn.call_args.kwargs["user_id"] == "u1"
        assert services.laminar.begin_turn.call_args.kwargs["user_input"] == "hello"


@pytest.mark.unit
class TestEndFanOut:
    def test_success_is_clean_everywhere(self, services: MagicMock) -> None:
        handles = {
            "agnost": MagicMock(),
            "latitude": MagicMock(),
            "laminar": MagicMock(),
        }

        end_turn_all(handles, output="hi")  # type: ignore[typeddict-item]

        agnost_kwargs = services.agnost.end_turn.call_args.kwargs
        assert agnost_kwargs["success"] is True
        assert agnost_kwargs["properties"] == {
            "cancelled": False,
            "has_error": False,
            "outcome": "success",
        }
        assert services.latitude.end_turn.call_args.kwargs == {
            "error": None,
            "cancelled": False,
        }
        laminar_kwargs = services.laminar.end_turn.call_args.kwargs
        assert laminar_kwargs["output"] == "hi"
        assert laminar_kwargs["error"] is None
        assert laminar_kwargs["cancelled"] is False

    def test_error_is_failed_with_same_exception(self, services: MagicMock) -> None:
        handles = {
            "agnost": MagicMock(),
            "latitude": MagicMock(),
            "laminar": MagicMock(),
        }
        error = RuntimeError("provider down")

        end_turn_all(handles, output="boom", error=error)

        agnost_kwargs = services.agnost.end_turn.call_args.kwargs
        assert agnost_kwargs["success"] is False
        assert agnost_kwargs["properties"]["outcome"] == "failed"
        assert services.latitude.end_turn.call_args.kwargs["error"] is error
        assert services.laminar.end_turn.call_args.kwargs["error"] is error

    def test_cancelled_is_not_failed(self, services: MagicMock) -> None:
        handles = {
            "agnost": MagicMock(),
            "latitude": MagicMock(),
            "laminar": MagicMock(),
        }

        end_turn_all(handles, output="partial", cancelled=True)

        agnost_kwargs = services.agnost.end_turn.call_args.kwargs
        assert agnost_kwargs["success"] is False
        assert agnost_kwargs["properties"] == {
            "cancelled": True,
            "has_error": False,
            "outcome": "cancelled",
        }
        assert services.latitude.end_turn.call_args.kwargs == {
            "error": None,
            "cancelled": True,
        }
        assert services.laminar.end_turn.call_args.kwargs["cancelled"] is True

    def test_error_dominates_cancelled(self, services: MagicMock) -> None:
        handles = {
            "agnost": MagicMock(),
            "latitude": MagicMock(),
            "laminar": MagicMock(),
        }
        error = RuntimeError("provider down")

        end_turn_all(handles, output="boom", error=error, cancelled=True)

        agnost_kwargs = services.agnost.end_turn.call_args.kwargs
        assert agnost_kwargs["properties"]["outcome"] == "failed"
        assert agnost_kwargs["properties"]["cancelled"] is False
        assert services.latitude.end_turn.call_args.kwargs["cancelled"] is False

    def test_none_handles_is_noop(self, services: MagicMock) -> None:
        end_turn_all(None, output="hi")

        services.agnost.end_turn.assert_not_called()
        services.latitude.end_turn.assert_not_called()
        services.laminar.end_turn.assert_not_called()
