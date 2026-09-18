"""Ledger card publish: register emits a PENDING card, dedup emits nothing."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.hil_models import LedgerState

from .conftest import CONVERSATION_ID, STREAM_ID, USER_ID, make_request

MODULE = "app.services.hil.gate"


@pytest.fixture(autouse=True)
def _quiet_log():
    with patch(f"{MODULE}.log"):
        yield


def _request(**overrides: Any):
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


def _ledger(
    live: Any = None, denied: Any = None, registered_id: str = "ap_abc1234567"
) -> MagicMock:
    ledger = MagicMock()
    ledger.find_live = AsyncMock(return_value=live)
    ledger.find_latest_denied = AsyncMock(return_value=denied)
    ledger.register = AsyncMock(return_value=registered_id)
    return ledger


@pytest.mark.unit
class TestLedgerPublish:
    async def test_register_publishes_pending_card_once(self) -> None:
        from app.services.hil import gate

        ledger = _ledger()
        with (
            patch(f"{MODULE}.is_hil_ledger_enabled", new=AsyncMock(return_value=True)),
            patch(f"{MODULE}.resolve_policy", new=AsyncMock(return_value="ask")),
            patch(f"{MODULE}.approval_ledger_repository", new=ledger),
            patch(f"{MODULE}._integration_name_for", new=AsyncMock(return_value="gmail")),
            patch(f"{MODULE}.publish_ledger_request", new=AsyncMock()) as pub,
        ):
            await gate.decide_tool_call(_request())

        ledger.register.assert_awaited_once()
        pub.assert_awaited_once()
        assert pub.await_args.kwargs["approval_id"] == "ap_abc1234567"
        assert pub.await_args.kwargs["conversation_id"] == CONVERSATION_ID

    async def test_live_duplicate_publishes_nothing(self) -> None:
        from app.services.hil import gate

        live = MagicMock(approval_id="ap_live", summary="Send it")
        live.state = LedgerState.PENDING
        with (
            patch(f"{MODULE}.is_hil_ledger_enabled", new=AsyncMock(return_value=True)),
            patch(f"{MODULE}.resolve_policy", new=AsyncMock(return_value="ask")),
            patch(f"{MODULE}.approval_ledger_repository", new=_ledger(live=live)),
            patch(f"{MODULE}.publish_ledger_request", new=AsyncMock()) as pub,
        ):
            result = await gate.decide_tool_call(_request())

        pub.assert_not_awaited()
        assert result is not None
        assert "ap_live" in str(result.content)
