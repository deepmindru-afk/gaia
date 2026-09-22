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
    row.owner_run_type = ""
    row.owner_id = ""
    # Real datetime: the decision path computes card age from it, and a
    # MagicMock would TypeError on the subtraction, hiding real type errors.
    row.created_at = datetime.now(UTC)
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

    async def test_approve_delivery_failure_falls_back_to_wake(self) -> None:
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import decide_ledger

        repo = _repo(_row())
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(ledger_decide, "publish_ledger_decision", new=AsyncMock()),
            patch.object(
                ledger_decide,
                "_deliver_ticket",
                new=AsyncMock(side_effect=RuntimeError("redis down")),
            ),
            patch.object(ledger_decide, "_wake_agent", new=AsyncMock()) as wake,
        ):
            outcome = await decide_ledger("ap_abc", user_id="u1", kind="approve", v=3)

        assert outcome.committed is True
        wake.assert_awaited_once_with(repo.get_by_approval_id.return_value, "APPROVED", None)

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
class TestDecisionSubmittedEvent:
    async def test_approve_emits_event_with_user_id(self) -> None:
        """Same attribution rule as revoke: the decide path resolves its user from the row, so the event must carry that id explicitly."""
        from app.services.analytics_service import AnalyticsEvents
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import decide_ledger

        repo = _repo(_row())
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(ledger_decide, "publish_ledger_decision", new=AsyncMock()),
            patch.object(ledger_decide, "_deliver_ticket", new=AsyncMock()),
            patch.object(ledger_decide, "capture_event") as capture,
        ):
            outcome = await decide_ledger("ap_abc", user_id="u1", kind="approve", v=3)

        assert outcome.committed is True
        capture.assert_called_once()
        user_id, event, props = capture.call_args.args
        assert user_id == "u1"
        assert event == AnalyticsEvents.HIL_DECISION_SUBMITTED
        assert props["approval_id"] == "ap_abc"
        assert props["decision"] == "approved"
        assert props["ledger_version"] == 4
        assert props["card_age_seconds"] is not None
        assert props["card_age_seconds"] < 60

    async def test_deny_emits_deny_decision(self) -> None:
        from app.services.analytics_service import AnalyticsEvents
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import decide_ledger

        repo = _repo(_row())
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(ledger_decide, "publish_ledger_decision", new=AsyncMock()),
            patch.object(ledger_decide, "capture_event") as capture,
        ):
            await decide_ledger("ap_abc", user_id="u1", kind="deny", v=3)

        assert capture.call_args.args[1] == AnalyticsEvents.HIL_DECISION_SUBMITTED
        assert capture.call_args.args[2]["decision"] == "denied"

    async def test_uncommitted_decisions_emit_nothing(self) -> None:
        """Stale-v and lost-CAS returns decided nothing — an event would count attempts as successes."""
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import decide_ledger

        stale_repo = _repo(_row())
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=stale_repo),
            patch.object(ledger_decide, "publish_ledger_decision", new=AsyncMock()),
            patch.object(ledger_decide, "_deliver_ticket", new=AsyncMock()),
            patch.object(ledger_decide, "capture_event") as capture,
        ):
            outcome = await decide_ledger("ap_abc", user_id="u1", kind="approve", v=99)

        assert outcome.committed is False
        capture.assert_not_called()

        lost_repo = _repo(_row())
        lost_repo.transition = AsyncMock(return_value=False)
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=lost_repo),
            patch.object(ledger_decide, "publish_ledger_decision", new=AsyncMock()),
            patch.object(ledger_decide, "_deliver_ticket", new=AsyncMock()),
            patch.object(ledger_decide, "capture_event") as capture,
        ):
            outcome = await decide_ledger("ap_abc", user_id="u1", kind="approve", v=3)

        assert outcome.committed is False
        capture.assert_not_called()


@pytest.mark.unit
class TestConversationFlagSync:
    async def test_decide_refreshes_the_sidebar_flag(self) -> None:
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import decide_ledger

        repo = _repo(_row())
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(ledger_decide, "publish_ledger_decision", new=AsyncMock()),
            patch.object(ledger_decide, "_deliver_ticket", new=AsyncMock()),
            patch.object(ledger_decide, "capture_event"),
            patch.object(ledger_decide, "sync_conversation_approval_flag", new=AsyncMock()) as sync,
        ):
            await decide_ledger("ap_abc", user_id="u1", kind="approve", v=3)

        sync.assert_awaited_once_with("conv-1", "u1")

    async def test_revoke_refreshes_the_sidebar_flag(self) -> None:
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import revoke_ticket

        repo = _repo(_row())
        repo.transition = AsyncMock(return_value=True)
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(ledger_decide, "publish_ledger_revocation", new=AsyncMock()),
            patch.object(ledger_decide, "capture_event"),
            patch.object(ledger_decide, "sync_conversation_approval_flag", new=AsyncMock()) as sync,
        ):
            await revoke_ticket(
                "ap_abc", user_id="u1", conversation_id="conv-1", caller="executor_conv-1"
            )

        sync.assert_awaited_once_with("conv-1", "u1")

    async def test_refused_revoke_syncs_nothing(self) -> None:
        """A revoke that changed nothing must not touch the flag — the row's state (and any flag it implies) is exactly as it was."""
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import revoke_ticket
        from app.services.hil.resolution import ApprovalRequestForbiddenError

        repo = _repo(_row())
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(ledger_decide, "publish_ledger_revocation", new=AsyncMock()),
            patch.object(ledger_decide, "capture_event"),
            patch.object(ledger_decide, "sync_conversation_approval_flag", new=AsyncMock()) as sync,
            pytest.raises(ApprovalRequestForbiddenError),
        ):
            await revoke_ticket(
                "ap_abc",
                user_id="intruder",
                conversation_id="conv-1",
                caller="executor_conv-1",
            )

        sync.assert_not_called()


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

    async def test_deny_delivers_verdict_to_start_idle_run(self) -> None:
        """A deny must reach the agent even when the conversation is idle: the inbox wake alone is only read by a running run."""
        from app.services.hil.ledger_decide import decide_ledger

        row = _row()
        with (
            patch(f"{MODULE}.approval_ledger_repository", new=_repo(row)),
            patch(f"{MODULE}.publish_ledger_decision", new=AsyncMock()),
            patch(f"{MODULE}._deliver_ticket", new=AsyncMock()),
            patch(f"{MODULE}._deliver_verdict", new=AsyncMock()) as deliver,
            patch(f"{MODULE}._wake_agent", new=AsyncMock()) as wake,
        ):
            outcome = await decide_ledger("ap_abc", user_id="u1", kind="deny", feedback="nope", v=3)

        assert outcome.committed is True
        deliver.assert_awaited_once()
        assert deliver.await_args.args[1] == "DENIED"
        assert "nope" in str(deliver.await_args.args[2])
        wake.assert_not_awaited()

    async def test_deny_delivery_failure_falls_back_to_wake(self) -> None:
        from app.services.hil.ledger_decide import _deliver_verdict

        row = _row()
        with (
            patch(
                "app.agents.core.background.executor_runner.deliver_to_executor",
                new=AsyncMock(side_effect=RuntimeError("redis down")),
            ),
            patch(f"{MODULE}._wake_agent", new=AsyncMock()) as wake,
        ):
            await _deliver_verdict(row, "DENIED", "nope")

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
        """Orphaned APPROVED rows are left alone: the ticket lives in model context and only a redeem runs the envelope — never re-nudge."""
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

    async def test_reconcile_scopes_stalled_scan_to_conversation(self) -> None:
        # The stalled scan must not read every conversation's stalls on each
        # tap: scope rides in the query, not a Python-side discard.
        from app.services.hil.ledger_decide import reconcile_conversation_ledger

        repo = _repo()
        repo.list_stalled_executing = AsyncMock(return_value=[])
        repo.transition = AsyncMock(return_value=True)
        with (
            patch(f"{MODULE}.approval_ledger_repository", new=repo),
            patch(f"{MODULE}._deliver_ticket", new=AsyncMock()) as deliver,
        ):
            await reconcile_conversation_ledger("conv-1")

        repo.list_stalled_executing.assert_awaited_once()
        assert repo.list_stalled_executing.await_args.args[1] == "conv-1"
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

    async def test_redeem_terminal_syncs_the_sidebar_flag(self) -> None:
        """A settled ticket must not leave its sidebar row behind."""
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import redeem_approved

        ok_result = MagicMock(ok=True, output="sent", error=None)
        repo = self._redeem_repo()
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(ledger_decide, "dispatch_tool", new=AsyncMock(return_value=ok_result)),
            patch.object(ledger_decide, "sync_conversation_approval_flag", new=AsyncMock()) as sync,
        ):
            result = await redeem_approved("ap_abc", **self._redeem_kwargs())

        assert result.state.value == "executed"
        sync.assert_awaited_once_with("conv-1", "u1")

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
        """The approved envelope must run AS the row's user."""
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
        """An infra raise must still tell the caller what happened: UNKNOWN with the cause, never a silent row flip."""
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
            "user_id": "u1",
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

    async def test_revoke_emits_event_with_user_id(self) -> None:
        """The revoke event must attribute to the row's user — an anonymous capture would strand it outside the user's funnel, silently."""
        from app.services.analytics_service import AnalyticsEvents
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import revoke_ticket

        repo = _repo(_row())
        repo.transition = AsyncMock(return_value=True)
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(ledger_decide, "publish_ledger_revocation", new=AsyncMock()),
            patch.object(ledger_decide, "capture_event") as capture,
        ):
            await revoke_ticket("ap_abc", **self._revoke_kwargs())

        capture.assert_called_once_with(
            "u1",
            AnalyticsEvents.HIL_REVOKED,
            {
                "approval_id": "ap_abc",
                "ledger_version": 4,
                "revoker": "executor_conv-1",
            },
        )

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
                user_id="u1",
                conversation_id="other-conv",
                caller="executor_other-conv",
            )

        assert "No pending approval" in text
        dispatch.assert_not_awaited()

    async def test_revoke_cross_user_forbidden(self) -> None:
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import revoke_ticket
        from app.services.hil.resolution import ApprovalRequestForbiddenError

        repo = _repo(_row())
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(ledger_decide, "publish_ledger_revocation", new=AsyncMock()) as tombstone,
            pytest.raises(ApprovalRequestForbiddenError),
        ):
            await revoke_ticket(
                "ap_abc",
                user_id="attacker",
                conversation_id="conv-1",
                caller="executor_conv-1",
            )
        repo.transition.assert_not_awaited()
        tombstone.assert_not_awaited()

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
                user_id="u1",
                conversation_id="conv-1",
                caller="worker-unrelated",
            )

        assert "another worker" in text
        tombstone.assert_not_awaited()


ANY_TEXT = "provider timed out; may or may not have run — never auto-retried"


@pytest.mark.unit
class TestApproveDeliversToExecutor:
    async def test_approve_wakes_executor_with_redeem_task(self) -> None:
        """Approval is permission, not execution: commit wakes the model with the ticket instead of scheduling a backend run."""
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
class TestApproveResumesBackgroundOwner:
    async def test_approve_with_todo_owner_resumes(self) -> None:
        """A background todo parked on this approval re-enqueues; the tap never waits on it and never fails for it."""
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import decide_ledger

        row = _row(owner_run_type="todo", owner_id="todo-9")
        repo = _repo(row)
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(ledger_decide, "publish_ledger_decision", new=AsyncMock()),
            patch.object(ledger_decide, "_deliver_ticket", new=AsyncMock()),
            patch.object(ledger_decide, "resume_owner_after_approval", new=AsyncMock()) as resume,
        ):
            outcome = await decide_ledger("ap_abc", user_id="u1", kind="approve", v=3)

        assert outcome.committed is True
        resume.assert_awaited_once_with(row)

    async def test_approve_without_owner_dispatches_to_a_no_op(self) -> None:
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import decide_ledger

        repo = _repo(_row())
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(ledger_decide, "publish_ledger_decision", new=AsyncMock()),
            patch.object(ledger_decide, "_deliver_ticket", new=AsyncMock()),
            patch.object(ledger_decide, "resume_owner_after_approval", new=AsyncMock()) as resume,
        ):
            await decide_ledger("ap_abc", user_id="u1", kind="approve", v=3)

        # Dispatch is unconditional; the no-owner row returns before any claim.
        resume.assert_awaited_once()

    async def test_deny_records_todo_skip_and_resumes_nothing(self) -> None:
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import decide_ledger

        repo = _repo(_row(owner_run_type="todo", owner_id="todo-9"))
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(ledger_decide, "publish_ledger_decision", new=AsyncMock()),
            patch.object(ledger_decide, "_deliver_verdict", new=AsyncMock()),
            patch.object(ledger_decide, "resume_owner_after_approval", new=AsyncMock()) as resume,
            patch.object(ledger_decide, "record_owner_deny", new=AsyncMock()) as record,
        ):
            outcome = await decide_ledger("ap_abc", user_id="u1", kind="deny", feedback="nope", v=3)

        assert outcome.committed is True
        assert outcome.state == LedgerState.DENIED
        resume.assert_not_called()
        record.assert_awaited_once()

    async def test_redeem_runs_stored_envelope_and_returns_output(self) -> None:
        """The model supplies no args: the ticket IS the approval_id and the envelope runs verbatim."""
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
class TestConditionalApproveBecomesDeny:
    async def test_approve_with_feedback_records_denied(self) -> None:
        # Permission-scope guard: an envelope carries no conditions, so
        # "approve + note" must never run the unmodified args. It records
        # as denied with the note attached (chat-classifier parity).
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import decide_ledger

        repo = _repo(_row())
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(ledger_decide, "publish_ledger_decision", new=AsyncMock()),
            patch.object(ledger_decide, "_deliver_ticket", new=AsyncMock()) as deliver,
            patch.object(ledger_decide, "_deliver_verdict", new=AsyncMock()) as verdict,
        ):
            outcome = await decide_ledger(
                "ap_abc", user_id="u1", kind="approve", feedback="cc finance", v=3
            )

        assert outcome.committed is True
        assert outcome.state == LedgerState.DENIED
        repo.transition.assert_awaited_once_with(
            "ap_abc",
            LedgerState.PENDING,
            LedgerState.DENIED,
            decided_by="u1",
            feedback="cc finance",
        )
        deliver.assert_not_awaited()
        verdict.assert_awaited_once()

    async def test_plain_approve_unaffected(self) -> None:
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
        assert outcome.state == LedgerState.APPROVED
        deliver.assert_awaited_once()


@pytest.mark.unit
class TestTicketCarriesAge:
    async def test_old_approve_commits_and_task_names_age(self) -> None:
        """No server refuse: a 3-day-old approve commits like any other, and the ticket wake carries the age so the model judges freshness."""
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


@pytest.mark.unit
class TestCancelLedgerApprovals:
    async def test_cancel_withdraws_user_pending_rows_only(self) -> None:
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import cancel_ledger_approvals

        mine = _row(approval_id="ap_mine")
        approved = _row(approval_id="ap_ticket", state=LedgerState.APPROVED)
        foreign = _row(approval_id="ap_theirs", user_id="u2")
        repo = _repo()
        repo.list_open = AsyncMock(return_value=[mine, approved, foreign])
        repo.transition = AsyncMock(return_value=True)
        repo.get_by_approval_id = AsyncMock(return_value=mine)
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(ledger_decide, "publish_ledger_revocation", new=AsyncMock()) as tombstone,
        ):
            cancelled = await cancel_ledger_approvals("conv-1", "u1")

        assert cancelled == ["ap_mine"]
        repo.transition.assert_awaited_once_with(
            "ap_mine", LedgerState.PENDING, LedgerState.REVOKED
        )
        tombstone.assert_awaited_once()

    async def test_cancel_lost_race_skips_quietly(self) -> None:
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import cancel_ledger_approvals

        repo = _repo()
        repo.list_open = AsyncMock(return_value=[_row()])
        repo.transition = AsyncMock(return_value=False)
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(ledger_decide, "publish_ledger_revocation", new=AsyncMock()) as tombstone,
        ):
            cancelled = await cancel_ledger_approvals("conv-1", "u1")

        assert cancelled == []
        tombstone.assert_not_awaited()


@pytest.mark.unit
class TestRevokeSettlesSessionFrame:
    async def test_revoke_flips_pending_frame_in_proposing_session(self) -> None:
        # BUG: revoke settled mongo + broadcast but never touched the proposing
        # session's in-memory frame; the drain later persisted the stale PENDING
        # frame back over it (production ap_d455 stuck pending post-revoke).
        from app.agents.core.background.session import (
            RunKind,
            create_session,
            get_session,
            teardown_session,
        )
        from app.models.hil_models import HILApprovalStatus
        from app.services.hil import ledger_decide
        from app.services.hil.bridge import _approval_entry, _publish_entry
        from app.services.hil.utils import GatedCall

        stream_id = "stream-settle-test"
        create_session(stream_id, RunKind.LIVE)
        try:
            with patch(
                "app.services.hil.bridge.stream_manager.publish_chunk",
                new=AsyncMock(),
            ):
                await _publish_entry(
                    stream_id,
                    _approval_entry(
                        "ap_abc",
                        GatedCall(name="GMAIL_SEND_EMAIL", id="c1", args={"to": "b@x"}),
                        HILApprovalStatus.PENDING,
                        "Send it",
                        None,
                    ),
                )
            row = _row(proposing_run_id=stream_id, state=LedgerState.REVOKED)
            with (
                patch.object(ledger_decide, "_persist_decision_status", new=AsyncMock()),
                patch.object(ledger_decide, "_broadcast_decision", new=AsyncMock()),
            ):
                await ledger_decide.publish_ledger_revocation(row)
            frames = get_session(stream_id).tool_events
            assert len(frames) == 1
            assert frames[0]["tool_data"]["data"]["status"] == "revoked"
        finally:
            teardown_session(stream_id)


@pytest.mark.unit
class TestRedeemSettlesTerminalFrame:
    async def test_executed_settles_card_as_executed(self) -> None:
        # Production ap_38b3/d74bc/1454/a16e: ledger UNKNOWN while cards stayed
        # approved — redeem settled nothing. Terminal states must settle the frame
        # (persist + broadcast + session) so the card collapses to the outcome chip.
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import redeem_approved

        repo = _repo(_row(state=LedgerState.APPROVED))
        repo.claim_executing = AsyncMock(return_value=True)
        repo.transition = AsyncMock(return_value=True)
        ok_result = MagicMock(ok=True, output="sent", error=None)
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(ledger_decide, "dispatch_tool", new=AsyncMock(return_value=ok_result)),
            patch.object(ledger_decide, "_persist_decision_status", new=AsyncMock()) as persist,
            patch.object(ledger_decide, "_broadcast_decision", new=AsyncMock()) as broadcast,
        ):
            result = await redeem_approved(
                "ap_abc",
                user_id="u1",
                conversation_id="conv-1",
                caller="executor_conv-1",
            )

        assert result.ok is True
        assert result.state == LedgerState.EXECUTED
        persist.assert_awaited_once()
        assert persist.await_args.args[1] == "executed"
        broadcast.assert_awaited_once()
        assert broadcast.await_args.args[1] == "executed"

    async def test_failed_settles_card_as_failed(self) -> None:
        from app.agents.tools.execute.dispatch import DispatchError, DispatchErrorKind
        from app.services.hil import ledger_decide
        from app.services.hil.ledger_decide import redeem_approved

        failed = MagicMock(
            ok=False,
            output=None,
            error=DispatchError(kind=DispatchErrorKind.INVALID_ARGS, detail="bad", hint="x"),
        )
        repo = _repo(_row(state=LedgerState.APPROVED))
        repo.claim_executing = AsyncMock(return_value=True)
        repo.transition = AsyncMock(return_value=True)
        with (
            patch.object(ledger_decide, "approval_ledger_repository", new=repo),
            patch.object(ledger_decide, "dispatch_tool", new=AsyncMock(return_value=failed)),
            patch.object(ledger_decide, "_persist_decision_status", new=AsyncMock()) as persist,
            patch.object(ledger_decide, "_broadcast_decision", new=AsyncMock()) as broadcast,
        ):
            result = await redeem_approved(
                "ap_abc",
                user_id="u1",
                conversation_id="conv-1",
                caller="executor_conv-1",
            )

        assert result.ok is False
        persist.assert_awaited_once()
        assert persist.await_args.args[1] == "failed"
        broadcast.assert_awaited_once()
        assert broadcast.await_args.args[1] == "failed"


@pytest.mark.unit
class TestReclaimDeadHolder:
    async def test_free_lock_means_proceed(self) -> None:
        from app.services.hil import ledger_decide

        with patch.object(ledger_decide, "get_lock_holder", new=AsyncMock(return_value=None)):
            assert await ledger_decide._reclaim_dead_holder("conv-1") is True

    async def test_live_session_means_hold(self) -> None:
        from app.services.hil import ledger_decide

        with (
            patch.object(ledger_decide, "get_lock_holder", new=AsyncMock(return_value="s1:t1")),
            patch.object(ledger_decide, "get_session", return_value=MagicMock()),
        ):
            assert await ledger_decide._reclaim_dead_holder("conv-1") is False

    async def test_fresh_lock_means_hold(self) -> None:
        from types import SimpleNamespace

        from app.constants.cache import EXECUTOR_BUSY_TTL
        from app.services.hil import ledger_decide

        client = SimpleNamespace(ttl=AsyncMock(return_value=EXECUTOR_BUSY_TTL - 10))
        with (
            patch.object(ledger_decide, "get_lock_holder", new=AsyncMock(return_value="s1:t1")),
            patch.object(ledger_decide, "get_session", return_value=None),
            patch.object(ledger_decide, "redis_cache", new=SimpleNamespace(client=client)),
            patch.object(
                ledger_decide,
                "list_pending_for_conversation",
                new=AsyncMock(return_value=[]),
            ),
        ):
            assert await ledger_decide._reclaim_dead_holder("conv-1") is False

    async def test_paused_records_mean_hold(self) -> None:
        from types import SimpleNamespace

        from app.services.hil import ledger_decide

        parked = SimpleNamespace(resume_item={"task": "x"}, subagent_thread_id=None)
        client = SimpleNamespace(ttl=AsyncMock(return_value=100))
        with (
            patch.object(ledger_decide, "get_lock_holder", new=AsyncMock(return_value="s1:t1")),
            patch.object(ledger_decide, "get_session", return_value=None),
            patch.object(ledger_decide, "redis_cache", new=SimpleNamespace(client=client)),
            patch.object(
                ledger_decide,
                "list_pending_for_conversation",
                new=AsyncMock(return_value=[parked]),
            ),
            patch.object(ledger_decide, "break_holder_lock", new=AsyncMock()) as breaker,
        ):
            assert await ledger_decide._reclaim_dead_holder("conv-1") is False
        breaker.assert_not_awaited()

    async def test_dead_holder_is_released(self) -> None:
        from types import SimpleNamespace

        from app.services.hil import ledger_decide

        client = SimpleNamespace(ttl=AsyncMock(return_value=100))
        with (
            patch.object(ledger_decide, "get_lock_holder", new=AsyncMock(return_value="s1:t1")),
            patch.object(ledger_decide, "get_session", return_value=None),
            patch.object(ledger_decide, "redis_cache", new=SimpleNamespace(client=client)),
            patch.object(
                ledger_decide,
                "list_pending_for_conversation",
                new=AsyncMock(return_value=[]),
            ),
            patch.object(
                ledger_decide, "break_holder_lock", new=AsyncMock(return_value=True)
            ) as breaker,
        ):
            assert await ledger_decide._reclaim_dead_holder("conv-1") is True
        breaker.assert_awaited_once_with("conv-1", "s1:t1")
