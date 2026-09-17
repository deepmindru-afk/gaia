"""revoke_tool: withdraw your own pending approval, nothing else."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.tools.revoke_tool import revoke_tool
from app.models.hil_models import LedgerState

MODULE = "app.agents.tools.revoke_tool"

CONFIG = {
    "configurable": {
        "thread_id": "gmail_conv-1",
        "conversation_id": "conv-1",
        "user_id": "u1",
    }
}


def _row(**overrides):
    row = MagicMock()
    row.approval_id = "ap_abc"
    row.conversation_id = "conv-1"
    row.owner_agent = "gmail_conv-1"
    row.summary = "Send it"
    row.state = LedgerState.PENDING
    for key, value in overrides.items():
        setattr(row, key, value)
    return row


def _ledger(row=None, transitioned=True):
    ledger = MagicMock()
    ledger.get_by_approval_id = AsyncMock(return_value=row)
    ledger.transition = AsyncMock(return_value=transitioned)
    return ledger


@pytest.mark.unit
class TestRevokeTool:
    async def test_revokes_own_pending(self) -> None:
        with patch(f"{MODULE}.approval_ledger_repository", new=_ledger(_row())) as ledger:
            result = await revoke_tool.ainvoke({"approval_id": "ap_abc"}, CONFIG)

        ledger.transition.assert_awaited_once_with("ap_abc", LedgerState.PENDING, LedgerState.REVOKED)
        assert result == "Revoked 'ap_abc' (Send it). It will never be asked."

    async def test_unknown_id_reads_as_not_found(self) -> None:
        with patch(f"{MODULE}.approval_ledger_repository", new=_ledger(None)):
            result = await revoke_tool.ainvoke({"approval_id": "ap_abc"}, CONFIG)

        assert result == "No pending approval with id 'ap_abc'."

    async def test_foreign_conversation_reads_as_not_found(self) -> None:
        with patch(
            f"{MODULE}.approval_ledger_repository", new=_ledger(_row(conversation_id="conv-9"))
        ):
            result = await revoke_tool.ainvoke({"approval_id": "ap_abc"}, CONFIG)

        assert result == "No pending approval with id 'ap_abc'."

    async def test_decided_row_is_untouchable(self) -> None:
        with patch(
            f"{MODULE}.approval_ledger_repository", new=_ledger(_row(state=LedgerState.APPROVED))
        ) as ledger:
            result = await revoke_tool.ainvoke({"approval_id": "ap_abc"}, CONFIG)

        ledger.transition.assert_not_awaited()
        assert "already approved" in result

    async def test_foreign_worker_is_refused(self) -> None:
        with patch(
            f"{MODULE}.approval_ledger_repository", new=_ledger(_row(owner_agent="slack_conv-1"))
        ) as ledger:
            result = await revoke_tool.ainvoke({"approval_id": "ap_abc"}, CONFIG)

        ledger.transition.assert_not_awaited()
        assert "belongs to another worker" in result

    async def test_executor_may_revoke_any_worker_row(self) -> None:
        config = {"configurable": {**CONFIG["configurable"], "thread_id": "executor_conv-1"}}
        with patch(
            f"{MODULE}.approval_ledger_repository", new=_ledger(_row(owner_agent="slack_conv-1"))
        ) as ledger:
            result = await revoke_tool.ainvoke({"approval_id": "ap_abc"}, config)

        ledger.transition.assert_awaited_once()
        assert result.startswith("Revoked")

    async def test_lost_race_reports_current_state(self) -> None:
        ledger = _ledger(_row(), transitioned=False)
        ledger.get_by_approval_id = AsyncMock(
            side_effect=[
                _row(),
                _row(state=LedgerState.APPROVED),
            ]
        )
        with patch(f"{MODULE}.approval_ledger_repository", new=ledger):
            result = await revoke_tool.ainvoke({"approval_id": "ap_abc"}, CONFIG)

        assert "already approved" in result
