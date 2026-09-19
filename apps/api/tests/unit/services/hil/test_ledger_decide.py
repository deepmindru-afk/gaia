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
    repo.list_stalled_executing = AsyncMock(return_value=[])
    repo.list_approved_unblocked = AsyncMock(return_value=[])
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
            "ap_abc",
            LedgerState.PENDING,
            LedgerState.APPROVED,
            decided_by="u1",
            feedback=None,
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
            outcome = await decide_ledger("ap_abc", user_id="u1", kind="deny", feedback="nope", v=3)

        assert outcome.committed is True
        assert outcome.state == LedgerState.DENIED
        repo.transition.assert_awaited_once_with(
            "ap_abc",
            LedgerState.PENDING,
            LedgerState.DENIED,
            decided_by="u1",
            feedback="nope",
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

    async def test_stale_v_marks_stale_not_gone(self) -> None:
        from app.services.hil.ledger_decide import decide_ledger

        repo = _repo(_row(state=LedgerState.APPROVED, v=4))
        with (
            patch(f"{MODULE}.approval_ledger_repository", new=repo),
            patch(f"{MODULE}.publish_ledger_decision", new=AsyncMock()),
            patch(f"{MODULE}.schedule_ledger_execution"),
        ):
            outcome = await decide_ledger("ap_abc", user_id="u1", kind="approve", v=3)

        assert outcome.committed is False
        assert outcome.stale is True
        assert outcome.state.value == "approved"

    async def test_unknown_id_raises_not_found(self) -> None:
        from app.services.hil.ledger_decide import decide_ledger
        from app.services.hil.resolution import ApprovalRequestNotFoundError

        with (
            patch(f"{MODULE}.approval_ledger_repository", new=_repo(None)),
            pytest.raises(ApprovalRequestNotFoundError),
        ):
            await decide_ledger("ap_nope", user_id="u1", kind="approve", v=None)


@pytest.mark.unit
class TestDecideHardening:
    async def test_empty_owner_never_matches_any_user(self) -> None:
        from app.services.hil.ledger_decide import decide_ledger
        from app.services.hil.resolution import ApprovalRequestForbiddenError

        repo = _repo(_row(user_id=""))
        with (
            patch(f"{MODULE}.approval_ledger_repository", new=repo),
            pytest.raises(ApprovalRequestForbiddenError),
        ):
            await decide_ledger("ap_abc", user_id="u1", kind="approve", v=None)

        repo.transition.assert_not_awaited()

    async def test_deny_wakes_the_agent_with_feedback(self) -> None:
        from app.services.hil.ledger_decide import decide_ledger

        row = _row()
        with (
            patch(f"{MODULE}.approval_ledger_repository", new=_repo(row)),
            patch(f"{MODULE}.publish_ledger_decision", new=AsyncMock()),
            patch(f"{MODULE}.schedule_ledger_execution"),
            patch(f"{MODULE}._wake_agent", new=AsyncMock()) as wake,
        ):
            await decide_ledger("ap_abc", user_id="u1", kind="deny", feedback="nope", v=3)

        wake.assert_awaited_once_with(row, "DENIED", "nope")

    async def test_queued_approve_wakes_with_blockers(self) -> None:
        from app.services.hil.ledger_decide import decide_ledger

        row = _row(blocked_by=["ap_dep"])
        with (
            patch(f"{MODULE}.approval_ledger_repository", new=_repo(row)),
            patch(f"{MODULE}.publish_ledger_decision", new=AsyncMock()),
            patch(f"{MODULE}.schedule_ledger_execution") as sched,
            patch(f"{MODULE}._wake_agent", new=AsyncMock()) as wake,
        ):
            outcome = await decide_ledger("ap_abc", user_id="u1", kind="approve", v=3)

        assert outcome.queued is True
        sched.assert_not_called()
        wake.assert_awaited_once_with(row, "QUEUED", "waiting on ap_dep")

    async def test_reconcile_heals_stalled_and_orphans(self) -> None:
        from app.models.hil_models import LedgerState as LS
        from app.services.hil.ledger_decide import reconcile_conversation_ledger

        stalled = _row(approval_id="ap_old")
        stalled.state = LS.EXECUTING
        orphan = _row(approval_id="ap_orphan")
        orphan.state = LS.APPROVED
        orphan.blocked_by = []
        repo = _repo()
        repo.list_stalled_executing = AsyncMock(return_value=[stalled])
        repo.list_approved_unblocked = AsyncMock(return_value=[orphan])
        repo.transition = AsyncMock(return_value=True)
        with (
            patch(f"{MODULE}.approval_ledger_repository", new=repo),
            patch(f"{MODULE}.schedule_ledger_execution") as sched,
            patch(f"{MODULE}._wake_agent", new=AsyncMock()) as wake,
        ):
            await reconcile_conversation_ledger("conv-1")

        repo.transition.assert_awaited_once_with("ap_old", LS.EXECUTING, LS.UNKNOWN)
        sched.assert_called_once_with("ap_orphan")
        assert wake.await_count == 1

    async def test_reconcile_ignores_other_conversations(self) -> None:
        from app.services.hil.ledger_decide import reconcile_conversation_ledger

        foreign = _row(approval_id="ap_x")
        foreign.conversation_id = "conv-9"
        repo = _repo()
        repo.list_stalled_executing = AsyncMock(return_value=[foreign])
        repo.list_approved_unblocked = AsyncMock(return_value=[])
        repo.transition = AsyncMock(return_value=True)
        with (
            patch(f"{MODULE}.approval_ledger_repository", new=repo),
            patch(f"{MODULE}.schedule_ledger_execution") as sched,
        ):
            await reconcile_conversation_ledger("conv-1")

        repo.transition.assert_not_awaited()
        sched.assert_not_called()


@pytest.mark.unit
class TestEnvelopeExecution:
    def _exec_repo(self, result: Any) -> MagicMock:
        from app.services.hil.ledger_decide import _execute_ledger_envelope  # noqa: F401

        repo = _repo(_row())
        repo.claim_executing = AsyncMock(return_value=True)
        repo.transition = AsyncMock(return_value=True)
        return repo

    async def test_provider_timeout_lands_unknown_never_failed(self) -> None:
        from app.agents.tools.execute.dispatch import DispatchError, DispatchErrorKind
        from app.models.hil_models import LedgerState as LS
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import _execute_ledger_envelope

        timeout_result = MagicMock(
            ok=False,
            output=None,
            error=DispatchError(kind=DispatchErrorKind.TIMEOUT, detail="slow", hint="x"),
        )
        repo = self._exec_repo(timeout_result)
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(
                ledger_decide, "dispatch_tool", new=AsyncMock(return_value=timeout_result)
            ),
            patch.object(ledger_decide, "_wake_agent", new=AsyncMock()) as wake,
        ):
            await _execute_ledger_envelope("ap_abc")

        repo.transition.assert_awaited_once_with("ap_abc", LS.EXECUTING, LS.UNKNOWN)
        wake.assert_awaited_once_with(repo.get_by_approval_id.return_value, "UNKNOWN", ANY_TEXT)

    async def test_lost_receipt_cas_wakes_with_actual_state(self) -> None:
        from app.models.hil_models import LedgerState as LS
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import _execute_ledger_envelope

        ok_result = MagicMock(ok=True, output="sent", error=None)
        repo = self._exec_repo(ok_result)
        repo.claim_executing = AsyncMock(return_value=True)
        repo.transition = AsyncMock(return_value=False)
        reconciled = _row()
        reconciled.state = LS.UNKNOWN
        repo.get_by_approval_id = AsyncMock(side_effect=[_row(), reconciled])
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(ledger_decide, "dispatch_tool", new=AsyncMock(return_value=ok_result)),
            patch.object(ledger_decide, "_wake_agent", new=AsyncMock()) as wake,
        ):
            await _execute_ledger_envelope("ap_abc")

        wake.assert_awaited_once_with(reconciled, "unknown", "sent")


@pytest.mark.unit
class TestEnvelopeIdentity:
    async def test_envelope_dispatches_with_user_identity_in_config(self) -> None:
        """The approved envelope must run AS the row's user.

        Tools are user-agnostic at resolve time; Composio/MCP wrappers resolve
        per-user auth from config at invocation (configurable AND metadata —
        different wrappers read different keys; the sandbox route already sets
        both). Without metadata the invoke runs as Composio's "default" user,
        finds no connected accounts, and lands UNKNOWN. Pinned from the
        2026-09-18 production trace (ap_5ede5be5eb11).
        """
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import _execute_ledger_envelope

        repo = _repo(_row())
        repo.claim_executing = AsyncMock(return_value=True)
        repo.transition = AsyncMock(return_value=True)
        ok_result = MagicMock(ok=True, output="sent", error=None)
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(
                ledger_decide, "dispatch_tool", new=AsyncMock(return_value=ok_result)
            ) as dispatch,
            patch.object(ledger_decide, "_wake_agent", new=AsyncMock()),
            patch.object(ledger_decide, "publish_ledger_receipt", new=AsyncMock()),
        ):
            await _execute_ledger_envelope("ap_abc")

        config = dispatch.await_args.kwargs["config"]
        assert config["configurable"]["user_id"] == "u1"
        assert config["metadata"]["user_id"] == "u1"


@pytest.mark.unit
class TestExecutionReceipt:
    async def test_executed_publishes_receipt_with_output(self) -> None:
        from app.models.hil_models import LedgerState as LS
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import _execute_ledger_envelope

        repo = _repo(_row())
        repo.claim_executing = AsyncMock(return_value=True)
        repo.transition = AsyncMock(return_value=True)
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(
                ledger_decide,
                "dispatch_tool",
                new=AsyncMock(return_value=MagicMock(ok=True, output="deleted xyz", error=None)),
            ),
            patch.object(ledger_decide, "_wake_agent", new=AsyncMock()),
            patch.object(ledger_decide, "publish_ledger_receipt", new=AsyncMock()) as receipt,
        ):
            await _execute_ledger_envelope("ap_abc")

        repo.transition.assert_awaited_once_with("ap_abc", LS.EXECUTING, LS.EXECUTED)
        receipt.assert_awaited_once()
        assert receipt.await_args.args[1] == "EXECUTED"
        assert "deleted xyz" in str(receipt.await_args.args[2])

    async def test_failed_publishes_receipt_with_error_detail(self) -> None:
        from app.agents.tools.execute.dispatch import DispatchError, DispatchErrorKind
        from app.models.hil_models import LedgerState as LS
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import _execute_ledger_envelope

        failed = MagicMock(
            ok=False,
            output=None,
            error=DispatchError(kind=DispatchErrorKind.INVALID_ARGS, detail="bad date", hint="x"),
        )
        repo = _repo(_row())
        repo.claim_executing = AsyncMock(return_value=True)
        repo.transition = AsyncMock(return_value=True)
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(ledger_decide, "dispatch_tool", new=AsyncMock(return_value=failed)),
            patch.object(ledger_decide, "_wake_agent", new=AsyncMock()),
            patch.object(ledger_decide, "publish_ledger_receipt", new=AsyncMock()) as receipt,
        ):
            await _execute_ledger_envelope("ap_abc")

        repo.transition.assert_awaited_once_with("ap_abc", LS.EXECUTING, LS.FAILED)
        receipt.assert_awaited_once()
        assert receipt.await_args.args[1] == "FAILED"
        assert "bad date" in str(receipt.await_args.args[2])

    async def test_raised_execution_publishes_unknown_receipt_with_cause(self) -> None:
        """An infra raise must still tell the user what happened: UNKNOWN with
        the cause, never a silent row flip and an empty wake."""
        from app.models.hil_models import LedgerState as LS
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import _execute_ledger_envelope

        repo = _repo(_row())
        repo.claim_executing = AsyncMock(return_value=True)
        repo.transition = AsyncMock(return_value=True)
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(
                ledger_decide,
                "dispatch_tool",
                new=AsyncMock(side_effect=ValueError("No connected accounts")),
            ),
            patch.object(ledger_decide, "_wake_agent", new=AsyncMock()) as wake,
            patch.object(ledger_decide, "publish_ledger_receipt", new=AsyncMock()) as receipt,
        ):
            await _execute_ledger_envelope("ap_abc")

        repo.transition.assert_awaited_once_with("ap_abc", LS.EXECUTING, LS.UNKNOWN)
        receipt.assert_awaited_once()
        assert receipt.await_args.args[1] == "UNKNOWN"
        assert "No connected accounts" in str(receipt.await_args.args[2])
        wake.assert_awaited_once()
        assert "No connected accounts" in str(wake.await_args.args[2])


@pytest.mark.unit
class TestReceiptDelivery:
    async def test_receipt_settles_stream_persist_and_broadcast(self) -> None:
        """The triple delivery mirrors the decision frame: live stream when it
        is still open, persisted frame for reload, broadcast for listeners."""
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import publish_ledger_receipt

        row = _row()
        row.proposing_run_id = "stream-1"
        with (
            patch.object(ledger_decide, "_publish_entry", new=AsyncMock()) as publish,
            patch.object(ledger_decide, "_persist_decision_status", new=AsyncMock()) as persist,
            patch.object(ledger_decide, "_broadcast_decision", new=AsyncMock()) as broadcast,
        ):
            await publish_ledger_receipt(row, "EXECUTED", "deleted xyz")

        publish.assert_awaited_once()
        persist.assert_awaited_once()
        assert persist.await_args.args[1] == "approved"
        assert "deleted xyz" in str(persist.await_args.kwargs["feedback"])
        broadcast.assert_awaited_once()
        assert "deleted xyz" in str(broadcast.await_args.args[2])


ANY_TEXT = "provider timed out; may or may not have run — never auto-retried"
