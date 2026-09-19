"""Ledger decide: CAS commit, stale-v refresh, queued approvals, no execution yet."""

from datetime import UTC, datetime, timedelta
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
    return repo


@pytest.mark.unit
class TestDecideLedger:
    async def test_approve_cas_commits_and_delivers_ticket(self) -> None:
        from app.services.hil.ledger_decide import decide_ledger

        repo = _repo(_row())
        with (
            patch(f"{MODULE}.approval_ledger_repository", new=repo),
            patch(f"{MODULE}.publish_ledger_decision", new=AsyncMock()),
            patch(f"{MODULE}._deliver_ticket", new=AsyncMock()) as deliver,
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
        deliver.assert_awaited_once()

    async def test_stale_v_returns_current_row_without_writing(self) -> None:
        from app.services.hil.ledger_decide import decide_ledger

        repo = _repo(_row(state=LedgerState.APPROVED, v=4))
        with (
            patch(f"{MODULE}.approval_ledger_repository", new=repo),
            patch(f"{MODULE}.publish_ledger_decision", new=AsyncMock()),
            patch(f"{MODULE}._deliver_ticket", new=AsyncMock()) as deliver,
        ):
            outcome = await decide_ledger("ap_abc", user_id="u1", kind="approve", v=3)

        assert outcome.committed is False
        assert outcome.state == LedgerState.APPROVED
        repo.transition.assert_not_awaited()
        deliver.assert_not_awaited()

    async def test_deny_commits_without_delivery(self) -> None:
        from app.services.hil.ledger_decide import decide_ledger

        repo = _repo(_row())
        with (
            patch(f"{MODULE}.approval_ledger_repository", new=repo),
            patch(f"{MODULE}.publish_ledger_decision", new=AsyncMock()),
            patch(f"{MODULE}._deliver_ticket", new=AsyncMock()) as deliver,
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
        deliver.assert_not_awaited()

    async def test_blocked_approval_stays_queued_without_delivery(self) -> None:
        from app.services.hil.ledger_decide import decide_ledger

        repo = _repo(_row(blocked_by=["ap_dep"]))
        with (
            patch(f"{MODULE}.approval_ledger_repository", new=repo),
            patch(f"{MODULE}.publish_ledger_decision", new=AsyncMock()),
            patch(f"{MODULE}._deliver_ticket", new=AsyncMock()) as deliver,
        ):
            outcome = await decide_ledger("ap_abc", user_id="u1", kind="approve", v=3)

        assert outcome.committed is True
        assert outcome.queued is True
        deliver.assert_not_awaited()

    async def test_stale_v_marks_stale_not_gone(self) -> None:
        from app.services.hil.ledger_decide import decide_ledger

        repo = _repo(_row(state=LedgerState.APPROVED, v=4))
        with (
            patch(f"{MODULE}.approval_ledger_repository", new=repo),
            patch(f"{MODULE}.publish_ledger_decision", new=AsyncMock()),
            patch(f"{MODULE}._deliver_ticket", new=AsyncMock()),
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
            patch(f"{MODULE}._deliver_ticket", new=AsyncMock()),
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
            patch(f"{MODULE}._deliver_ticket", new=AsyncMock()) as deliver,
            patch(f"{MODULE}._wake_agent", new=AsyncMock()) as wake,
        ):
            outcome = await decide_ledger("ap_abc", user_id="u1", kind="approve", v=3)

        assert outcome.queued is True
        deliver.assert_not_awaited()
        wake.assert_awaited_once_with(row, "QUEUED", "waiting on ap_dep")

    async def test_reconcile_heals_stalled_only(self) -> None:
        """Orphaned APPROVED rows are left alone: the ticket lives in model
        context and only a redeem runs the envelope — never re-nudge."""
        from app.models.hil_models import LedgerState as LS
        from app.services.hil.ledger_decide import reconcile_conversation_ledger

        stalled = _row(approval_id="ap_old")
        stalled.state = LS.EXECUTING
        orphan = _row(approval_id="ap_orphan")
        orphan.state = LS.APPROVED
        orphan.blocked_by = []
        repo = _repo()
        repo.list_stalled_executing = AsyncMock(return_value=[stalled])
        repo.transition = AsyncMock(return_value=True)
        with (
            patch(f"{MODULE}.approval_ledger_repository", new=repo),
            patch(f"{MODULE}._deliver_ticket", new=AsyncMock()) as deliver,
            patch(f"{MODULE}._wake_agent", new=AsyncMock()) as wake,
        ):
            await reconcile_conversation_ledger("conv-1")

        repo.transition.assert_awaited_once_with("ap_old", LS.EXECUTING, LS.UNKNOWN)
        deliver.assert_not_awaited()
        assert wake.await_count == 1

    async def test_reconcile_ignores_other_conversations(self) -> None:
        from app.services.hil.ledger_decide import reconcile_conversation_ledger

        foreign = _row(approval_id="ap_x")
        foreign.conversation_id = "conv-9"
        repo = _repo()
        repo.list_stalled_executing = AsyncMock(return_value=[foreign])
        repo.transition = AsyncMock(return_value=True)
        with (
            patch(f"{MODULE}.approval_ledger_repository", new=repo),
            patch(f"{MODULE}._deliver_ticket", new=AsyncMock()) as deliver,
        ):
            await reconcile_conversation_ledger("conv-1")

        repo.transition.assert_not_awaited()
        deliver.assert_not_awaited()


@pytest.mark.unit
class TestRedeemExecution:
    def _redeem_repo(self) -> MagicMock:
        from app.models.hil_models import LedgerState as LS

        repo = _repo(_row(state=LS.APPROVED))
        repo.claim_executing = AsyncMock(return_value=True)
        repo.transition = AsyncMock(return_value=True)
        return repo

    def _redeem_kwargs(self) -> dict[str, Any]:
        return {
            "user_id": "u1",
            "conversation_id": "conv-1",
            "caller": "executor_conv-1",
        }

    async def test_provider_timeout_lands_unknown_never_failed(self) -> None:
        from app.agents.tools.execute.dispatch import DispatchError, DispatchErrorKind
        from app.models.hil_models import LedgerState as LS
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import redeem_approved

        timeout_result = MagicMock(
            ok=False,
            output=None,
            error=DispatchError(kind=DispatchErrorKind.TIMEOUT, detail="slow", hint="x"),
        )
        repo = self._redeem_repo()
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(
                ledger_decide, "dispatch_tool", new=AsyncMock(return_value=timeout_result)
            ),
        ):
            result = await redeem_approved("ap_abc", **self._redeem_kwargs())

        repo.transition.assert_awaited_once_with("ap_abc", LS.EXECUTING, LS.UNKNOWN)
        assert result.ok is False
        assert result.state == LS.UNKNOWN
        assert "never auto-retried" in result.detail

    async def test_redeem_failed_transition_still_returns_state(self) -> None:
        from app.models.hil_models import LedgerState as LS
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import redeem_approved

        ok_result = MagicMock(ok=True, output="sent", error=None)
        repo = self._redeem_repo()
        repo.transition = AsyncMock(return_value=False)
        reconciled = _row()
        reconciled.state = LS.UNKNOWN
        approved = _row()
        approved.state = LS.APPROVED
        repo.get_by_approval_id = AsyncMock(side_effect=[approved, reconciled])
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(ledger_decide, "dispatch_tool", new=AsyncMock(return_value=ok_result)),
        ):
            result = await redeem_approved("ap_abc", **self._redeem_kwargs())

        assert result.ok is False
        assert "Already unknown" in result.detail


@pytest.mark.unit
class TestRedeemIdentity:
    async def test_redeem_dispatches_with_user_identity_in_config(self) -> None:
        """The approved envelope must run AS the row's user.

        Tools are user-agnostic at resolve time; Composio/MCP wrappers resolve
        per-user auth from config at invocation (configurable AND metadata —
        different wrappers read different keys; the sandbox route already sets
        both). Without metadata the invoke runs as Composio's "default" user,
        finds no connected accounts, and lands UNKNOWN. Pinned from the
        2026-09-18 production trace (ap_5ede5be5eb11).
        """
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import redeem_approved

        repo = _repo(_row(state=LedgerState.APPROVED))
        repo.claim_executing = AsyncMock(return_value=True)
        repo.transition = AsyncMock(return_value=True)
        ok_result = MagicMock(ok=True, output="sent", error=None)
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(
                ledger_decide, "dispatch_tool", new=AsyncMock(return_value=ok_result)
            ) as dispatch,
        ):
            await redeem_approved(
                "ap_abc",
                user_id="u1",
                conversation_id="conv-1",
                caller="executor_conv-1",
            )

        config = dispatch.await_args.kwargs["config"]
        assert config["configurable"]["user_id"] == "u1"
        assert config["metadata"]["user_id"] == "u1"


@pytest.mark.unit
class TestRedeemTerminalStates:
    async def test_executed_returns_output(self) -> None:
        from app.models.hil_models import LedgerState as LS
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import redeem_approved

        repo = _repo(_row(state=LedgerState.APPROVED))
        repo.claim_executing = AsyncMock(return_value=True)
        repo.transition = AsyncMock(return_value=True)
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(
                ledger_decide,
                "dispatch_tool",
                new=AsyncMock(return_value=MagicMock(ok=True, output="deleted xyz", error=None)),
            ),
        ):
            result = await redeem_approved(
                "ap_abc",
                user_id="u1",
                conversation_id="conv-1",
                caller="executor_conv-1",
            )

        repo.transition.assert_awaited_once_with("ap_abc", LS.EXECUTING, LS.EXECUTED)
        assert result.ok is True
        assert result.state == LS.EXECUTED
        assert "deleted xyz" in result.detail

    async def test_failed_returns_error_detail(self) -> None:
        from app.agents.tools.execute.dispatch import DispatchError, DispatchErrorKind
        from app.models.hil_models import LedgerState as LS
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import redeem_approved

        failed = MagicMock(
            ok=False,
            output=None,
            error=DispatchError(kind=DispatchErrorKind.INVALID_ARGS, detail="bad date", hint="x"),
        )
        repo = _repo(_row(state=LedgerState.APPROVED))
        repo.claim_executing = AsyncMock(return_value=True)
        repo.transition = AsyncMock(return_value=True)
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(ledger_decide, "dispatch_tool", new=AsyncMock(return_value=failed)),
        ):
            result = await redeem_approved(
                "ap_abc",
                user_id="u1",
                conversation_id="conv-1",
                caller="executor_conv-1",
            )

        repo.transition.assert_awaited_once_with("ap_abc", LS.EXECUTING, LS.FAILED)
        assert result.ok is False
        assert result.state == LS.FAILED
        assert "bad date" in result.detail

    async def test_raised_execution_returns_unknown_with_cause(self) -> None:
        """An infra raise must still tell the caller what happened: UNKNOWN
        with the cause, never a silent row flip."""
        from app.models.hil_models import LedgerState as LS
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import redeem_approved

        repo = _repo(_row(state=LedgerState.APPROVED))
        repo.claim_executing = AsyncMock(return_value=True)
        repo.transition = AsyncMock(return_value=True)
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(
                ledger_decide,
                "dispatch_tool",
                new=AsyncMock(side_effect=ValueError("No connected accounts")),
            ),
        ):
            result = await redeem_approved(
                "ap_abc",
                user_id="u1",
                conversation_id="conv-1",
                caller="executor_conv-1",
            )

        repo.transition.assert_awaited_once_with("ap_abc", LS.EXECUTING, LS.UNKNOWN)
        assert result.ok is False
        assert result.state == LS.UNKNOWN
        assert "No connected accounts" in result.detail


@pytest.mark.unit
class TestRevokeTicket:
    def _revoke_kwargs(self) -> dict[str, Any]:
        return {
            "conversation_id": "conv-1",
            "caller": "executor_conv-1",
        }

    async def test_revoke_tombstones_pending_row(self) -> None:
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import revoke_ticket

        repo = _repo(_row())
        repo.transition = AsyncMock(return_value=True)
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(ledger_decide, "publish_ledger_revocation", new=AsyncMock()) as tombstone,
        ):
            text = await revoke_ticket("ap_abc", **self._revoke_kwargs())

        repo.transition.assert_awaited_once_with("ap_abc", LedgerState.PENDING, LedgerState.REVOKED)
        tombstone.assert_awaited_once()
        assert "Revoked 'ap_abc'" in text

    async def test_revoke_wrong_conversation_looks_absent(self) -> None:
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import revoke_ticket

        repo = _repo(_row())
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(ledger_decide, "dispatch_tool", new=AsyncMock()) as dispatch,
        ):
            text = await revoke_ticket(
                "ap_abc",
                conversation_id="other-conv",
                caller="executor_other-conv",
            )

        assert "No pending approval" in text
        dispatch.assert_not_awaited()

    async def test_revoke_decided_row_refused(self) -> None:
        from app.models.hil_models import LedgerState as LS
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import revoke_ticket

        repo = _repo(_row(state=LS.APPROVED))
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(ledger_decide, "publish_ledger_revocation", new=AsyncMock()) as tombstone,
        ):
            text = await revoke_ticket("ap_abc", **self._revoke_kwargs())

        assert "already approved" in text
        tombstone.assert_not_awaited()

    async def test_revoke_foreign_worker_refused(self) -> None:
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import revoke_ticket

        row = _row()
        row.owner_agent = "worker-other"
        repo = _repo(row)
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(ledger_decide, "publish_ledger_revocation", new=AsyncMock()) as tombstone,
        ):
            text = await revoke_ticket(
                "ap_abc",
                conversation_id="conv-1",
                caller="worker-unrelated",
            )

        assert "another worker" in text
        tombstone.assert_not_awaited()


ANY_TEXT = "provider timed out; may or may not have run — never auto-retried"


@pytest.mark.unit
class TestApproveDeliversToExecutor:
    async def test_approve_wakes_executor_with_redeem_task(self) -> None:
        """Approval is permission, not execution: commit wakes the model with
        the ticket instead of scheduling a backend run."""
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import decide_ledger

        repo = _repo(_row())
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(ledger_decide, "publish_ledger_decision", new=AsyncMock()),
            patch.object(ledger_decide, "_deliver_ticket", new=AsyncMock()) as deliver,
        ):
            outcome = await decide_ledger("ap_abc", user_id="u1", kind="approve", v=3)

        assert outcome.committed is True
        deliver.assert_awaited_once()
        claimed_row = deliver.await_args.args[0]
        assert claimed_row.approval_id == "ap_abc"


@pytest.mark.unit
class TestRedeemApproved:
    async def test_redeem_runs_stored_envelope_and_returns_output(self) -> None:
        """The model supplies no args: the ticket IS the approval_id and the
        envelope runs verbatim."""
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import redeem_approved

        repo = _repo(_row(state=LedgerState.APPROVED))
        repo.claim_executing = AsyncMock(return_value=True)
        repo.transition = AsyncMock(return_value=True)
        ok_result = MagicMock(ok=True, output="deleted xyz", error=None)
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(
                ledger_decide, "dispatch_tool", new=AsyncMock(return_value=ok_result)
            ) as dispatch,
        ):
            result = await redeem_approved(
                "ap_abc",
                user_id="u1",
                conversation_id="conv-1",
                caller="executor_conv-1",
            )

        dispatch.assert_awaited_once()
        assert dispatch.await_args.kwargs["data"] == {"to": "b@x"}
        assert result.ok is True
        assert "deleted xyz" in result.detail

    async def test_double_redeem_runs_envelope_once(self) -> None:
        """The claim is the single-use CAS: loser is refused, never re-run."""
        from app.models.hil_models import LedgerState as LS
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import redeem_approved

        repo = _repo(_row(state=LedgerState.APPROVED))
        repo.claim_executing = AsyncMock(side_effect=[True, False])
        repo.transition = AsyncMock(return_value=True)
        ok_result = MagicMock(ok=True, output="sent", error=None)
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(
                ledger_decide, "dispatch_tool", new=AsyncMock(return_value=ok_result)
            ) as dispatch,
        ):
            first = await redeem_approved(
                "ap_abc",
                user_id="u1",
                conversation_id="conv-1",
                caller="executor_conv-1",
            )
            second = await redeem_approved(
                "ap_abc",
                user_id="u1",
                conversation_id="conv-1",
                caller="executor_conv-1",
            )

        assert first.ok is True
        assert first.state == LS.EXECUTED
        assert second.ok is False
        dispatch.assert_awaited_once()

    async def test_redeem_wrong_conversation_looks_absent(self) -> None:
        """No existence leak across conversations — mirrors revoke_tool."""
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import redeem_approved
        from app.services.hil.resolution import ApprovalRequestNotFoundError

        repo = _repo(_row(state=LedgerState.APPROVED))
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(ledger_decide, "dispatch_tool", new=AsyncMock()) as dispatch,
            pytest.raises(ApprovalRequestNotFoundError),
        ):
            await redeem_approved(
                "ap_abc",
                user_id="u1",
                conversation_id="other-conv",
                caller="executor_other-conv",
            )
        dispatch.assert_not_awaited()

    async def test_redeem_non_approved_row_refused(self) -> None:
        from app.models.hil_models import LedgerState as LS
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import redeem_approved

        repo = _repo(_row(state=LS.EXECUTED))
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(ledger_decide, "dispatch_tool", new=AsyncMock()) as dispatch,
        ):
            result = await redeem_approved(
                "ap_abc",
                user_id="u1",
                conversation_id="conv-1",
                caller="executor_conv-1",
            )

        assert result.ok is False
        dispatch.assert_not_awaited()


@pytest.mark.unit
class TestTicketCarriesAge:
    async def test_old_approve_commits_and_task_names_age(self) -> None:
        """No server refuse: a 3-day-old approve commits like any other, and
        the ticket wake carries the age so the model judges freshness."""
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import _ticket_task, decide_ledger

        old = _row()
        old.created_at = datetime.now(UTC) - timedelta(days=3)
        repo = _repo(old)
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(ledger_decide, "publish_ledger_decision", new=AsyncMock()),
            patch.object(ledger_decide, "_deliver_ticket", new=AsyncMock()) as deliver,
        ):
            outcome = await decide_ledger("ap_abc", user_id="u1", kind="approve", v=3)

        assert outcome.committed is True
        deliver.assert_awaited_once()
        claimed_row = deliver.await_args.args[0]
        assert claimed_row.approval_id == "ap_abc"
        task = _ticket_task(old)
        assert task.startswith("APPROVAL_READY ap_abc")
        assert "72h" in task
        assert 'execute(tool_name="approve"' in task

    async def test_fresh_approve_still_commits(self) -> None:
        """Guard against over-blocking: a fresh card is unaffected."""
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import decide_ledger

        fresh = _row()
        fresh.created_at = datetime.now(UTC)
        repo = _repo(fresh)
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(ledger_decide, "publish_ledger_decision", new=AsyncMock()),
            patch.object(ledger_decide, "_deliver_ticket", new=AsyncMock()),
        ):
            outcome = await decide_ledger("ap_abc", user_id="u1", kind="approve", v=3)

        assert outcome.committed is True

    async def test_stale_deny_still_passes(self) -> None:
        """Deny is a refusal — age never gates it."""
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import decide_ledger

        old = _row()
        old.created_at = datetime.now(UTC) - timedelta(days=3)
        repo = _repo(old)
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(ledger_decide, "publish_ledger_decision", new=AsyncMock()),
            patch.object(ledger_decide, "_deliver_ticket", new=AsyncMock()),
        ):
            outcome = await decide_ledger("ap_abc", user_id="u1", kind="deny", feedback="nope", v=3)

        assert outcome.committed is True
        assert outcome.state == LedgerState.DENIED
