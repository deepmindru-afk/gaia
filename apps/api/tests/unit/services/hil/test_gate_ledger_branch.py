"""Ledger branch of the approval gate (flag on: register, never interrupt)."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.constants.hil import HIL_STATUS_KWARG

from .conftest import CONVERSATION_ID, STREAM_ID, USER_ID, make_request

MODULE = "app.services.hil.gate"


@pytest.fixture(autouse=True)
def _quiet_log():
    with patch(f"{MODULE}.log"):
        yield


def _ledger(
    live: Any = None, denied: Any = None, registered_id: str = "ap_abc1234567"
) -> MagicMock:
    ledger = MagicMock()
    ledger.find_live = AsyncMock(return_value=live)
    ledger.find_latest_denied = AsyncMock(return_value=denied)
    ledger.register = AsyncMock(return_value=registered_id)
    return ledger


def _gated_request(**overrides: Any):
    return make_request(
        name="GMAIL_SEND_EMAIL",
        args={"to": "b@x"},
        configurable={
            "stream_id": STREAM_ID,
            "user_id": USER_ID,
            "conversation_id": CONVERSATION_ID,
            "user_messages": ["send it"],
            **overrides,
        },
    )


@pytest.mark.unit
class TestLedgerBranch:
    async def test_registers_pending_and_never_interrupts(self) -> None:
        from app.services.hil import gate

        ledger = _ledger()
        with (
            patch(f"{MODULE}.is_hil_ledger_enabled", new=AsyncMock(return_value=True)),
            patch(f"{MODULE}.resolve_policy", new=AsyncMock(return_value="ask")),
            patch(f"{MODULE}.approval_ledger_repository", new=ledger),
            patch(f"{MODULE}._integration_name_for", new=AsyncMock(return_value="gmail")),
            patch(f"{MODULE}.publish_ledger_request", new=AsyncMock()),
            patch(f"{MODULE}.interrupt") as intr,
        ):
            result = await gate.decide_tool_call(_gated_request())

        ledger.register.assert_awaited_once()
        intr.assert_not_called()
        assert result is not None
        assert "PENDING ap_abc1234567" in str(result.content)
        assert "DO NOT retry" in str(result.content)
        assert "revoke_tool" in str(result.content)
        assert result.additional_kwargs[HIL_STATUS_KWARG] == "pending"

    async def test_flag_off_takes_the_old_interrupt_path(self) -> None:
        from app.services.hil import gate

        with (
            patch(f"{MODULE}.is_hil_ledger_enabled", new=AsyncMock(return_value=False)),
            patch(f"{MODULE}.resolve_policy", new=AsyncMock(return_value="ask")),
            patch(f"{MODULE}.approval_ledger_repository", new=_ledger()) as ledger,
            patch(f"{MODULE}._integration_name_for", new=AsyncMock(return_value="gmail")),
            patch(f"{MODULE}.get_approval", new=AsyncMock(return_value=None)),
            patch(f"{MODULE}.recall_declined_call", new=AsyncMock(return_value=None)),
            patch(f"{MODULE}.publish_approval_request", new=AsyncMock()),
            patch(f"{MODULE}.interrupt") as intr,
        ):
            await gate.decide_tool_call(_gated_request())

        ledger.find_live.assert_not_awaited()
        intr.assert_called_once()

    async def test_live_duplicate_returns_existing_id_without_register(self) -> None:
        from app.services.hil import gate

        live = MagicMock(approval_id="ap_live", summary="Send it")
        ledger = _ledger(live=live)
        with (
            patch(f"{MODULE}.is_hil_ledger_enabled", new=AsyncMock(return_value=True)),
            patch(f"{MODULE}.resolve_policy", new=AsyncMock(return_value="ask")),
            patch(f"{MODULE}.approval_ledger_repository", new=ledger),
            patch(f"{MODULE}.interrupt") as intr,
        ):
            result = await gate.decide_tool_call(_gated_request())

        ledger.register.assert_not_awaited()
        intr.assert_not_called()
        assert result is not None
        assert "PENDING ap_live" in str(result.content)
        assert "already requested" in str(result.content)

    async def test_same_run_denied_reissue_is_refused(self) -> None:
        from app.services.hil import gate

        denied = MagicMock(proposing_run_id=STREAM_ID, feedback="too broad", decided_at=None)
        ledger = _ledger(denied=denied)
        with (
            patch(f"{MODULE}.is_hil_ledger_enabled", new=AsyncMock(return_value=True)),
            patch(f"{MODULE}.resolve_policy", new=AsyncMock(return_value="ask")),
            patch(f"{MODULE}.approval_ledger_repository", new=ledger),
            patch(f"{MODULE}.interrupt") as intr,
        ):
            result = await gate.decide_tool_call(_gated_request())

        ledger.register.assert_not_awaited()
        intr.assert_not_called()
        assert result is not None
        assert "REFUSED" in str(result.content)
        assert "DO NOT re-request" in str(result.content)
        assert result.additional_kwargs[HIL_STATUS_KWARG] == "denied"

    async def test_older_denied_run_registers_with_the_why_attached(self) -> None:
        from app.services.hil import gate

        denied = MagicMock(proposing_run_id="other-stream", feedback="wrong day", decided_at=None)
        ledger = _ledger(denied=denied)
        with (
            patch(f"{MODULE}.is_hil_ledger_enabled", new=AsyncMock(return_value=True)),
            patch(f"{MODULE}.resolve_policy", new=AsyncMock(return_value="ask")),
            patch(f"{MODULE}.approval_ledger_repository", new=ledger),
            patch(f"{MODULE}._integration_name_for", new=AsyncMock(return_value="gmail")),
            patch(f"{MODULE}.publish_ledger_request", new=AsyncMock()),
            patch(f"{MODULE}.interrupt") as intr,
        ):
            result = await gate.decide_tool_call(_gated_request())

        ledger.register.assert_awaited_once()
        intr.assert_not_called()
        assert result is not None
        assert "PENDING ap_abc1234567" in str(result.content)
        assert "wrong day" in str(result.content)

    async def test_ledger_failure_fails_closed_without_interrupt(self) -> None:
        from app.services.hil import gate

        ledger = _ledger()
        ledger.find_live = AsyncMock(side_effect=ConnectionError("mongo down"))
        with (
            patch(f"{MODULE}.is_hil_ledger_enabled", new=AsyncMock(return_value=True)),
            patch(f"{MODULE}.resolve_policy", new=AsyncMock(return_value="ask")),
            patch(f"{MODULE}.approval_ledger_repository", new=ledger),
            patch(f"{MODULE}.interrupt") as intr,
        ):
            result = await gate.decide_tool_call(_gated_request())

        intr.assert_not_called()
        assert result is not None
        assert result.additional_kwargs[HIL_STATUS_KWARG] == "error"
