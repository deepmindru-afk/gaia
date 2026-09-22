"""Resume driver: wake a background owner parked on a decided approval.

Live runs resume through the executor inbox (ticket redeem). Background runs
have no live run to wake — their work stopped at the PENDING message — so an
approval must RE-ENQUEUE the owning unit of work, and a denial must leave a
visible trace where the owner actually looks:

- Tracked todos have no chat. Continuity is the activity log plus a fresh
  execution: the resume appends the receipt and re-enqueues the todo, whose
  next execution reads the log and continues. A denial appends the skip.
- Workflows have their own chat AND a run queue. The resume re-queues the
  workflow with the receipt in its trigger context, so the run continues with
  the approval in context and the trace visible in the workflow's conversation.

Every resume costs a fresh human tap (one claim per approval), so no count
cap is needed — the human is the loop breaker. Everything here is
best-effort: a resume failure must never fail the tap that approved.
"""

from app.constants.log_tags import LogTag
from app.db.repositories.approval_ledger import approval_ledger_repository
from app.models.hil_models import ApprovalLedgerDocument
from app.services.analytics_service import AnalyticsEvents, capture_event
from shared.py.wide_events import log


async def resume_owner_after_approval(row: ApprovalLedgerDocument) -> None:
    """Re-enqueue the background owner parked on this approved row. Never raises."""
    if not row.owner_run_type or not row.owner_id:
        return
    try:
        if not await approval_ledger_repository.claim_resume(row.approval_id):
            return
    except Exception as e:
        log.warning(
            f"{LogTag.HIL} resume claim failed; skipping resume",
            approval_id=row.approval_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return
    try:
        if row.owner_run_type == "todo":
            await _resume_todo(row)
        elif row.owner_run_type == "workflow":
            await _resume_workflow(row)
        else:
            log.warning(
                f"{LogTag.HIL} unknown resume owner; skipping",
                approval_id=row.approval_id,
                owner_run_type=row.owner_run_type,
            )
    except Exception as e:
        log.warning(
            f"{LogTag.HIL} owner resume failed; the approval itself stands",
            approval_id=row.approval_id,
            owner_run_type=row.owner_run_type,
            owner_id=row.owner_id,
            error=str(e),
            error_type=type(e).__name__,
        )


async def record_owner_deny(row: ApprovalLedgerDocument, feedback: str | None) -> None:
    """Leave a denial trace where a background owner looks. Never raises."""
    if row.owner_run_type != "todo" or not row.owner_id:
        return
    try:
        from app.services.tracked_todo_service import tracked_todo_service

        what = f" — {feedback!r}" if feedback else ""
        await tracked_todo_service.append_activity_entry(
            todo_id=row.owner_id,
            user_id=row.user_id,
            entry=(f"Approval {row.approval_id} denied{what}: skipped {row.summary}."),
        )
    except Exception as e:
        log.warning(
            f"{LogTag.HIL} deny trace failed",
            approval_id=row.approval_id,
            error=str(e),
            error_type=type(e).__name__,
        )


async def _resume_todo(row: ApprovalLedgerDocument) -> None:
    """Resume in the PARKED conversation: the receipt joins the run's own
    thread instead of a fresh session. The lock/defer machinery stays the
    arbiter of concurrency, exactly like a scheduled fire."""
    from app.utils.redis_utils import RedisPoolManager
    from app.workers.queue import enqueue_worker_job

    pool = await RedisPoolManager.get_pool()
    await enqueue_worker_job(
        pool,
        "resume_tracked_todo",
        row.owner_id,
        row.conversation_id,
        row.approval_id,
        row.summary,
    )
    capture_event(
        row.user_id,
        AnalyticsEvents.HIL_RESUMED,
        {"approval_id": row.approval_id, "owner_run_type": "todo"},
    )


async def _resume_workflow(row: ApprovalLedgerDocument) -> None:
    """Re-queue the workflow with the receipt in its trigger context.

    A resumed run re-enters through the normal queue (deterministic job id
    dedups a racing re-fire), carrying the approval receipt plus the prior
    execution's trace summary so the agent continues past the granted step
    instead of redoing it. The workflow's own conversation shows the whole
    arc: park, approval, continuation.
    """
    from app.services.workflow.execution_service import get_last_run_brief
    from app.services.workflow.queue_service import WorkflowQueueService

    brief = ""
    try:
        brief = await get_last_run_brief(row.owner_id, row.user_id)
    except Exception as e:
        log.warning(
            f"{LogTag.HIL} last-run brief unavailable; resuming without it",
            approval_id=row.approval_id,
            error=str(e),
            error_type=type(e).__name__,
        )
    await WorkflowQueueService.queue_workflow_execution(
        row.owner_id,
        row.user_id,
        {
            "resume_from_approval": row.approval_id,
            "approval_summary": row.summary,
            "approval_result": "granted",
            "prior_run_brief": brief,
        },
    )
    capture_event(
        row.user_id,
        AnalyticsEvents.HIL_RESUMED,
        {"approval_id": row.approval_id, "owner_run_type": "workflow"},
    )
