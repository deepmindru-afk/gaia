"""Decide ledger approvals: CAS commit, instant-eligible execution, agent wake.

The single entry point for every ledger decision source — the web/mobile
decide endpoints, and later the bot classifier. Each supplies approve/deny;
everything else is identical.

Guarantees (mirror ``resolution.py`` for the old path):

* **Exactly once.** The ``PENDING -> decided`` transition is a conditional
  Mongo update. A double tap, a racing second device, and a stale client all
  lose the race and resolve nothing — they get the current row back.
* **Stale clients refresh, never overwrite.** Every render carries the row
  version ``v``; a decide call with a mismatched ``v`` returns the current
  row with ``committed=False``.
* **Execution never blocks the tap.** Approval commits in the request;
  the envelope runs in a canonical background task right after, and the
  agent learns from the executor-inbox DECISIONS wake. Crash between commit
  and receipt lands the row in UNKNOWN, reconciled lazily (never blind-retried).
* **No daemon.** Crash recovery (stalled EXECUTING, orphaned APPROVED) runs
  lazily inside the decide path via :func:`reconcile_conversation_ledger` —
  every decide heals its own conversation as a side effect.
"""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import uuid4

from app.agents.core.background.executor_channel import ExecutorInbox
from app.agents.tools.execute.dispatch import (
    DispatchErrorKind,
    dispatch_config_for,
    dispatch_tool,
)
from app.constants.agents import AgentTag
from app.constants.log_tags import LogTag
from app.core.websocket_manager import websocket_manager
from app.db.redis import redis_cache
from app.db.repositories.approval_ledger import approval_ledger_repository
from app.db.repositories.conversations import conversation_repository
from app.models.hil_models import (
    ApprovalLedgerDocument,
    HILApprovalStatus,
    LedgerState,
)
from app.services.hil.bridge import _approval_entry, _publish_entry
from app.services.hil.resolution import (
    ApprovalRequestForbiddenError,
    ApprovalRequestNotFoundError,
)
from app.services.hil.utils import GatedCall
from app.utils.background_tasks import spawn_background_task
from app.utils.general_utils import clip_text
from shared.py.wide_events import log

DecisionKind = Literal["approve", "deny"]

# EXECUTING rows older than this never had a live claimant: the process died
# holding the claim. Reconciled lazily to UNKNOWN on the decide path — there
# is no sweeper by design, so this cutoff is the only crash detector.
STALLED_EXECUTING_MINUTES = 10

# Bounded parallelism for batch submits: a 12-item approve-all must not fan
# out 12 concurrent provider calls.
_EXECUTION_SEMAPHORE = asyncio.Semaphore(4)


@dataclass(frozen=True)
class LedgerDecision:
    """What one decide call did: committed or current-row refresh."""

    committed: bool
    approval_id: str
    prior_state: LedgerState
    state: LedgerState
    # Approved but blocked_by unresolved: sits in APPROVED as the queue, no
    # execution scheduled. Ordering chains (claim rule + dep_unmet reconciler)
    # arrive in Phase 3; this flag is their seam.
    queued: bool = False
    # The call lost on version, not on state: the row may still be PENDING.
    # Callers must report "stale/refresh", never "not found", or clients drop
    # live cards they should keep.
    stale: bool = False


async def decide_ledger(
    approval_id: str,
    *,
    user_id: str,
    kind: DecisionKind,
    feedback: str | None = None,
    v: int | None = None,
) -> LedgerDecision:
    """Commit one decision, schedule execution, wake the agent. Never blocks."""
    row = await approval_ledger_repository.get_by_approval_id(approval_id)
    if row is None:
        raise ApprovalRequestNotFoundError()
    # Strict match, no empty bypass: a row without an owner must never be
    # decidable by whoever asks first. The gate always stamps a real id.
    if row.user_id != user_id:
        raise ApprovalRequestForbiddenError()

    if v is not None and v != row.v:
        return LedgerDecision(
            committed=False,
            approval_id=approval_id,
            prior_state=row.state,
            state=row.state,
            stale=True,
        )

    target = LedgerState.APPROVED if kind == "approve" else LedgerState.DENIED
    transitioned = await approval_ledger_repository.transition(
        approval_id, LedgerState.PENDING, target, decided_by=user_id, feedback=feedback
    )
    if not transitioned:
        current = await approval_ledger_repository.get_by_approval_id(approval_id)
        state = current.state if current is not None else row.state
        return LedgerDecision(
            committed=False,
            approval_id=approval_id,
            prior_state=row.state,
            state=state,
        )

    log.set(hil={"approval_id": approval_id, "decision": kind, "tool": row.tool_name})
    queued = bool(target is LedgerState.APPROVED and row.blocked_by)
    if target is LedgerState.APPROVED and not queued:
        # Schedule BEFORE the best-effort fan-out below: a cancel or shutdown
        # between commit and claim must still leave a live task holding the
        # execution, never a committed row nobody owns.
        schedule_ledger_execution(approval_id)
    await publish_ledger_decision(row, target, feedback=feedback)
    if target is LedgerState.DENIED:
        # Denials schedule nothing, but the agent still needs the verdict: it
        # exited on the PENDING message and list_open will never show this row
        # again. Without the wake "user said no" arrives nowhere.
        await _wake_agent(row, "DENIED", feedback)
    elif queued:
        await _wake_agent(row, "QUEUED", f"waiting on {','.join(row.blocked_by)}")
    await reconcile_conversation_ledger(row.conversation_id)
    return LedgerDecision(
        committed=True,
        approval_id=approval_id,
        prior_state=LedgerState.PENDING,
        state=target,
        queued=queued,
    )


async def reconcile_conversation_ledger(conversation_id: str) -> None:
    """Heal one conversation's ledger as a decide-path side effect.

    No daemon exists by design, so every decide also repairs: stalled
    EXECUTING rows go UNKNOWN (with a wake, so the agent learns), and
    APPROVED-but-unscheduled rows get their execution task back. Every step
    is CAS-guarded, so concurrent deciders racing here converge instead of
    duplicating.
    """
    cutoff = datetime.now(UTC) - timedelta(minutes=STALLED_EXECUTING_MINUTES)
    for stalled in await approval_ledger_repository.list_stalled_executing(cutoff):
        if stalled.conversation_id != conversation_id:
            continue
        if await approval_ledger_repository.transition(
            stalled.approval_id, LedgerState.EXECUTING, LedgerState.UNKNOWN
        ):
            log.error(
                f"{LogTag.HIL} Ledger execution stalled; reconciled as UNKNOWN, never retried",
                approval_id=stalled.approval_id,
            )
            await _wake_agent(stalled, "UNKNOWN", "stalled execution reconciled; never retried")
    for orphan in await approval_ledger_repository.list_approved_unblocked(conversation_id):
        schedule_ledger_execution(orphan.approval_id)


async def publish_ledger_decision(
    row: ApprovalLedgerDocument,
    status: LedgerState,
    *,
    feedback: str | None = None,
) -> None:
    """Settle the card where the user watches + persist the settled frame.

    Best-effort delivery, never fails the decision: the ledger row is the
    truth and every render re-reads it. Publishes the settled frame to the
    proposing stream when it is still live, persists the settled status for
    reload, and broadcasts so listening clients refresh.
    """
    mapped = (
        HILApprovalStatus.APPROVED if status is LedgerState.APPROVED else HILApprovalStatus.DENIED
    )
    entry = _approval_entry(
        row.approval_id,
        GatedCall(name=row.tool_name, id="", args=row.args),
        mapped,
        row.summary,
        None,
        feedback if feedback is not None else row.feedback,
    )
    if row.proposing_run_id:
        try:
            await _publish_entry(row.proposing_run_id, entry)
        except Exception as e:
            log.warning(
                f"{LogTag.HIL} Ledger decision frame missed its stream",
                approval_id=row.approval_id,
                error_type=type(e).__name__,
            )
    await _persist_decision_status(row, mapped.value)
    await _broadcast_decision(row, mapped.value, feedback if feedback is not None else row.feedback)


async def publish_ledger_receipt(row: ApprovalLedgerDocument, outcome: str, result: object) -> None:
    """Show the user what their approval DID — the other half of the decision.

    The decide path settles the card to approved/denied, but the tool runs
    detached afterwards; without this the user watches an approved card do
    nothing forever. Same triple delivery as the decision frame (live stream
    when still open, persisted frame for reload, broadcast for listeners) on
    the existing approval_request wire shape, so no client changes are needed:
    the settled card's feedback carries the receipt and the outcome chip shows
    it. Best-effort like every other delivery here — the row is the truth.
    """
    receipt = _receipt_text(outcome, result)
    entry = _approval_entry(
        row.approval_id,
        GatedCall(name=row.tool_name, id="", args=row.args),
        HILApprovalStatus.APPROVED,
        row.summary,
        None,
        receipt,
    )
    if row.proposing_run_id:
        try:
            await _publish_entry(row.proposing_run_id, entry)
        except Exception as e:
            log.warning(
                f"{LogTag.HIL} Ledger receipt frame missed its stream",
                approval_id=row.approval_id,
                error_type=type(e).__name__,
            )
    await _persist_decision_status(row, HILApprovalStatus.APPROVED.value, feedback=receipt)
    await _broadcast_decision(row, HILApprovalStatus.APPROVED.value, receipt)


def _receipt_text(outcome: str, result: object) -> str:
    """One user-facing line for a terminal execution outcome."""
    preview = "" if result is None else clip_text(str(result), 300)
    if outcome == "EXECUTED":
        return f"Done: {preview}" if preview else "Done."
    if outcome == "FAILED":
        return f"Failed: {preview}" if preview else "Failed."
    return (
        f"Outcome unknown{(': ' + preview) if preview else ''} — never auto-retried; "
        "check whether it happened before asking again."
    )


async def publish_ledger_revocation(row: ApprovalLedgerDocument) -> None:
    """Surface an agent-side revoke: tombstone, not a silent row change.

    Without this, a revoked card stays actionable in every open client
    forever — the tombstone UI would be unreachable. Best-effort like every
    other delivery here: the row is the truth, frames are hints.
    """
    await _persist_decision_status(row, "revoked")
    await _broadcast_decision(row, "revoked", None)


async def _persist_decision_status(
    row: ApprovalLedgerDocument, status: str, *, feedback: str | None = None
) -> None:
    """Settle the persisted card frame so reload renders the terminal state.

    Isolated on purpose (mirrors ``bridge.publish_decision``): a write error
    here must never fail the decision or revocation it reports on.
    """
    try:
        await conversation_repository.set_message_approval_status(
            row.conversation_id,
            user_id=row.user_id,
            approval_id=row.approval_id,
            status=status,
            feedback=feedback,
        )
    except Exception as e:
        log.warning(
            f"{LogTag.HIL} Ledger decision persist missed; live delivery already attempted",
            approval_id=row.approval_id,
            error_type=type(e).__name__,
        )


async def _broadcast_decision(
    row: ApprovalLedgerDocument, status: str, feedback: str | None
) -> None:
    """Tell listening clients a card settled — same shape for decide + revoke."""
    try:
        await websocket_manager.broadcast_to_user(
            user_id=row.user_id or "",
            message={
                "type": "hil_approval_decided",
                "data": {
                    "conversation_id": row.conversation_id,
                    "approval_id": row.approval_id,
                    "status": status,
                    "feedback": feedback,
                    "version": row.v + 1,
                },
            },
        )
    except Exception as e:
        log.warning(
            f"{LogTag.HIL} Ledger decision broadcast missed",
            approval_id=row.approval_id,
            error_type=type(e).__name__,
        )


def schedule_ledger_execution(approval_id: str) -> None:
    """Run an approved envelope post-response on the canonical task runner.

    ``spawn_background_task`` strong-refs the task to completion and carries
    the request's trace boundary — a bare create_task would risk both GC
    mid-flight and detached logs.
    """
    spawn_background_task(
        _execute_ledger_envelope(approval_id),
        name=f"ledger-execute-{approval_id}",
        on_done=lambda task: (
            log.error(
                f"{LogTag.HIL} Ledger execution task ended with an error",
                approval_id=approval_id,
                error=str(task.exception()),
            )
            if not task.cancelled() and task.exception() is not None
            else None
        ),
    )


async def _execute_ledger_envelope(approval_id: str) -> None:
    """Claim EXECUTING exactly once, run the stored envelope, receipt, wake.

    The semaphore comes FIRST so EXECUTING always means a provider call in
    flight — never "queued behind 8 siblings". The provider timeout is the
    one genuinely ambiguous outcome (may or may not have run), so it lands
    UNKNOWN, never FAILED: FAILED invites a re-proposal that could duplicate
    a send that already happened.
    """
    async with _EXECUTION_SEMAPHORE:
        claimed = await approval_ledger_repository.claim_executing(approval_id)
        if not claimed:
            return
        row = await approval_ledger_repository.get_by_approval_id(approval_id)
        if row is None:
            return
        try:
            result = await dispatch_tool(
                user_id=row.user_id or None,
                tool_name=row.tool_name,
                data=dict(row.args),
                # Identity-bearing config, not a bare configurable: the wrappers
                # resolve per-user auth from this (see dispatch_config_for).
                # A bare configurable ran approvals as Composio's "default" user
                # with no connected accounts — every approval landed UNKNOWN.
                config=dispatch_config_for(row.user_id or ""),
            )
        except Exception as e:
            await approval_ledger_repository.transition(
                approval_id, LedgerState.EXECUTING, LedgerState.UNKNOWN
            )
            cause = f"{type(e).__name__}: {e}"
            log.error(
                f"{LogTag.HIL} Ledger execution raised; reconciled as UNKNOWN, never retried",
                approval_id=approval_id,
                error_type=type(e).__name__,
            )
            await publish_ledger_receipt(row, "UNKNOWN", cause)
            await _wake_agent(row, "UNKNOWN", cause)
            return
        if (
            not result.ok
            and result.error is not None
            and result.error.kind == DispatchErrorKind.TIMEOUT
        ):
            state, outcome = LedgerState.UNKNOWN, "UNKNOWN"
            detail: object = "provider timed out; may or may not have run — never auto-retried"
        elif result.ok:
            state, outcome, detail = LedgerState.EXECUTED, "EXECUTED", result.output
        else:
            state, outcome = LedgerState.FAILED, "FAILED"
            detail = result.error.detail if result.error is not None else "unknown error"
        committed = await approval_ledger_repository.transition(
            approval_id, LedgerState.EXECUTING, state
        )
        if not committed:
            # A reconciler moved the row under us (stalled cutoff fired
            # mid-flight): report what the ledger actually says, not what we
            # attempted. Never a silent overwrite of its verdict.
            current = await approval_ledger_repository.get_by_approval_id(approval_id)
            live = current if current is not None else row
            actual = live.state.value
            log.error(
                f"{LogTag.HIL} Ledger receipt lost its CAS; waking with actual state",
                approval_id=approval_id,
                attempted=state.value,
                actual=actual,
            )
            await _wake_agent(live, actual, detail)
            return
        await publish_ledger_receipt(row, outcome, detail)
        await _wake_agent(row, outcome, detail)


async def _wake_agent(row: ApprovalLedgerDocument, outcome: str, result: object) -> None:
    """One DECISIONS line into the executor inbox — the agent's only signal.

    Inbox is transport, ledger is truth: a lost wake re-surfaces via the next
    OPEN PENDINGS injection, never via re-execution. A missing Redis client is
    a loud error, never a silent drop — ``append`` no-ops without one, so this
    checks first instead of reporting a delivery that never happened.
    """
    if redis_cache.client is None:
        log.error(
            f"{LogTag.HIL} Ledger wake dropped: no Redis client; agent will not learn this outcome",
            approval_id=row.approval_id,
            outcome=outcome,
        )
        return
    preview = "" if result is None else str(result)[:500]
    try:
        await ExecutorInbox(row.conversation_id).append(
            str(uuid4()),
            f"DECISIONS: {row.approval_id}={outcome} {row.summary}"
            + (f" :: {preview}" if preview else ""),
            AgentTag.HIL_DECISION,
        )
    except Exception as e:
        log.error(
            f"{LogTag.HIL} Ledger wake failed; outcome lives on the row only",
            approval_id=row.approval_id,
            outcome=outcome,
            error_type=type(e).__name__,
        )
