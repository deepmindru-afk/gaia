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
  the envelope runs in a strong-ref'd in-process task right after, and the
  agent learns from the executor-inbox DECISIONS wake. Crash between commit
  and receipt lands the row in UNKNOWN, reconciled lazily (never blind-retried).
"""

import asyncio
from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from app.agents.core.background.executor_channel import ExecutorInbox
from app.agents.tools.execute.dispatch import dispatch_tool
from app.constants.agents import AgentTag
from app.constants.log_tags import LogTag
from app.core.websocket_manager import websocket_manager
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
from shared.py.wide_events import log

DecisionKind = Literal["approve", "deny"]

# In-flight ledger executions, strongly referenced until done — a bare
# create_task can be GC'd mid-flight, stranding an APPROVED row in EXECUTING.
_execution_tasks: set[asyncio.Task[None]] = set()

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
    if row.user_id and row.user_id != user_id:
        raise ApprovalRequestForbiddenError()

    if v is not None and v != row.v:
        return LedgerDecision(
            committed=False,
            approval_id=approval_id,
            prior_state=row.state,
            state=row.state,
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
    await publish_ledger_decision(row, target, feedback=feedback)

    queued = bool(target is LedgerState.APPROVED and row.blocked_by)
    if target is LedgerState.APPROVED and not queued:
        schedule_ledger_execution(approval_id)
    if queued:
        log.info(
            f"{LogTag.HIL} Ledger approval queued behind dependencies",
            approval_id=approval_id,
            blocked_by=row.blocked_by,
        )
    return LedgerDecision(
        committed=True,
        approval_id=approval_id,
        prior_state=LedgerState.PENDING,
        state=target,
        queued=queued,
    )


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
        HILApprovalStatus.APPROVED
        if status is LedgerState.APPROVED
        else HILApprovalStatus.DENIED
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
    try:
        await conversation_repository.set_message_approval_status(
            row.conversation_id,
            user_id=row.user_id,
            approval_id=row.approval_id,
            status=mapped.value,
        )
    except Exception as e:
        log.warning(
            f"{LogTag.HIL} Ledger decision persist missed; live frame already sent",
            approval_id=row.approval_id,
            error_type=type(e).__name__,
        )
    try:
        await websocket_manager.broadcast_to_user(
            user_id=row.user_id or "",
            message={
                "type": "hil_approval_decided",
                "data": {
                    "conversation_id": row.conversation_id,
                    "approval_id": row.approval_id,
                    "status": mapped.value,
                    "feedback": feedback if feedback is not None else row.feedback,
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
    """Run an approved envelope post-response, strongly referenced to completion."""
    task = asyncio.create_task(_execute_ledger_envelope(approval_id))
    _execution_tasks.add(task)
    task.add_done_callback(_execution_tasks.discard)


async def _execute_ledger_envelope(approval_id: str) -> None:
    """Claim EXECUTING exactly once, run the stored envelope, receipt, wake."""
    claimed = await approval_ledger_repository.transition(
        approval_id, LedgerState.APPROVED, LedgerState.EXECUTING
    )
    if not claimed:
        return
    row = await approval_ledger_repository.get_by_approval_id(approval_id)
    if row is None:
        return
    try:
        async with _EXECUTION_SEMAPHORE:
            result = await dispatch_tool(
                user_id=row.user_id or None,
                tool_name=row.tool_name,
                data=dict(row.args),
                config={"configurable": {"user_id": row.user_id}},
            )
    except Exception as e:
        await approval_ledger_repository.transition(
            approval_id, LedgerState.EXECUTING, LedgerState.UNKNOWN
        )
        log.error(
            f"{LogTag.HIL} Ledger execution raised; reconciled as UNKNOWN, never retried",
            approval_id=approval_id,
            error_type=type(e).__name__,
        )
        await _wake_agent(row, "UNKNOWN", None)
        return
    if result.ok:
        await approval_ledger_repository.transition(
            approval_id, LedgerState.EXECUTING, LedgerState.EXECUTED
        )
        await _wake_agent(row, "EXECUTED", result.output)
    else:
        await approval_ledger_repository.transition(
            approval_id, LedgerState.EXECUTING, LedgerState.FAILED
        )
        detail = result.error.detail if result.error is not None else "unknown error"
        await _wake_agent(row, "FAILED", detail)


async def _wake_agent(
    row: ApprovalLedgerDocument, outcome: str, result: object
) -> None:
    """One DECISIONS line into the executor inbox — the agent's only signal.

    Inbox is transport, ledger is truth: a lost wake re-surfaces via the next
    OPEN PENDINGS injection, never via re-execution.
    """
    preview = "" if result is None else str(result)[:500]
    await ExecutorInbox(row.conversation_id).append(
        str(uuid4()),
        f"DECISIONS: {row.approval_id}={outcome} {row.summary}"
        + (f" :: {preview}" if preview else ""),
        AgentTag.HIL_DECISION,
    )
