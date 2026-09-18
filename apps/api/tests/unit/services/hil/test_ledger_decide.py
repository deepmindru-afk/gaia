"""Ledger decide: CAS commit, stale-v refresh, queued approvals, no execution yet."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.hil_models import LedgerState

MODULE = "app.services.hil.ledger_decide"


def _row(**overrides: Any) -> MagicMock:
    row = MagicMock()
    row.approval_id = "ap_abc"
    row.conversation_id = "conv-1"
    row.user_id = "u1"
    row.tool_name = "GMAIL_SEND_EMAIL"
    row.args = {"to": "b@x"}
    row.summary = "Send it"
    row.state = LedgerState.PENDING
    row.v = 3
    row.blocked_by = []
    row.decided_at = None
    row.feedback = None
    for key, value in overrides.items():
        setattr(row, key, value)
    return row


def _repo(row: Any = None) -> MagicMock:
    repo = MagicMock()
    repo.get_by_approval_id = AsyncMock(return_value=row)
    repo.transition = AsyncMock(return_value=True)
    return repo


@pytest.mark.unit
class TestDecideLedger:
    async def test_approve_cas_commits_and_claims_executing(self) -> None:
        from app.services.hil.ledger_decide import decide_ledger

        repo = _repo(_row())
        with (
            patch(f"{MODULE}.approval_ledger_repository", new=repo),
            patch(f"{MODULE}.publish_ledger_decision", new=AsyncMock()),
            patch(f"{MODULE}.schedule_ledger_execution") as sched,
        ):
            outcome = await decide_ledger("ap_abc", user_id="u1", kind="approve", v=3)

        assert outcome.committed is True
        assert outcome.prior_state == LedgerState.PENDING
        assert outcome.state == LedgerState.APPROVED
        assert outcome.queued is False
        repo.transition.assert_awaited_once_with(
            "ap_abc", LedgerState.PENDING, LedgerState.APPROVED,
            decided_by="u1", feedback=None,
        )
        sched.assert_called_once()

    async def test_stale_v_returns_current_row_without_writing(self) -> None:
        from app.services.hil.ledger_decide import decide_ledger

        repo = _repo(_row(state=LedgerState.APPROVED, v=4))
        with (
            patch(f"{MODULE}.approval_ledger_repository", new=repo),
            patch(f"{MODULE}.publish_ledger_decision", new=AsyncMock()),
            patch(f"{MODULE}.schedule_ledger_execution") as sched,
        ):
            outcome = await decide_ledger("ap_abc", user_id="u1", kind="approve", v=3)

        assert outcome.committed is False
        assert outcome.state == LedgerState.APPROVED
        repo.transition.assert_not_awaited()
        sched.assert_not_called()

    async def test_deny_commits_without_scheduling(self) -> None:
        from app.services.hil.ledger_decide import decide_ledger

        repo = _repo(_row())
        with (
            patch(f"{MODULE}.approval_ledger_repository", new=repo),
            patch(f"{MODULE}.publish_ledger_decision", new=AsyncMock()),
            patch(f"{MODULE}.schedule_ledger_execution") as sched,
        ):
            outcome = await decide_ledger(
                "ap_abc", user_id="u1", kind="deny", feedback="nope", v=3
            )

        assert outcome.committed is True
        assert outcome.state == LedgerState.DENIED
        repo.transition.assert_awaited_once_with(
            "ap_abc", LedgerState.PENDING, LedgerState.DENIED,
            decided_by="u1", feedback="nope",
        )
        sched.assert_not_called()

    async def test_blocked_approval_stays_queued_without_scheduling(self) -> None:
        from app.services.hil.ledger_decide import decide_ledger

        repo = _repo(_row(blocked_by=["ap_dep"]))
        with (
            patch(f"{MODULE}.approval_ledger_repository", new=repo),
            patch(f"{MODULE}.publish_ledger_decision", new=AsyncMock()),
            patch(f"{MODULE}.schedule_ledger_execution") as sched,
        ):
            outcome = await decide_ledger("ap_abc", user_id="u1", kind="approve", v=3)

        assert outcome.committed is True
        assert outcome.queued is True
        sched.assert_not_called()

    async def test_unknown_id_raises_not_found(self) -> None:
        from app.services.hil.ledger_decide import decide_ledger
        from app.services.hil.resolution import ApprovalRequestNotFoundError

        with (
            patch(f"{MODULE}.approval_ledger_repository", new=_repo(None)),
            pytest.raises(ApprovalRequestNotFoundError),
        ):
            await decide_ledger("ap_nope", user_id="u1", kind="approve", v=None)
