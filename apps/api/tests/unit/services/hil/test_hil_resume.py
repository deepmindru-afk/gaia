"""Resume driver (app/services/hil/resume.py): background owners wake on verdicts.

Live runs resume through the executor inbox; background runs have no live run,
so approvals re-enqueue the owner and denials leave a trace where the owner
looks. Everything is best-effort and claim-guarded — a resume failure must
never fail the tap.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.hil_models import LedgerState

MODULE = "app.services.hil.resume"


def _row(**overrides: Any) -> MagicMock:
    row = MagicMock()
    row.approval_id = "ap_bg1"
    row.conversation_id = "conv-bg"
    row.user_id = "u1"
    row.tool_name = "GMAIL_SEND_EMAIL"
    row.summary = "Send briefing"
    row.state = LedgerState.APPROVED
    row.owner_run_type = ""
    row.owner_id = ""
    for key, value in overrides.items():
        setattr(row, key, value)
    return row


class TestResumeAfterApproval:
    async def test_no_owner_is_a_no_op(self) -> None:
        from app.services.hil.resume import resume_owner_after_approval

        with patch(f"{MODULE}.approval_ledger_repository") as repo:
            await resume_owner_after_approval(_row())
        repo.claim_resume.assert_not_called()

    async def test_lost_claim_means_another_tap_already_resumed(self) -> None:
        from app.services.hil.resume import resume_owner_after_approval

        with (
            patch(f"{MODULE}.approval_ledger_repository") as repo,
            patch(f"{MODULE}._resume_todo", new=AsyncMock()) as resume,
        ):
            repo.claim_resume = AsyncMock(return_value=False)
            await resume_owner_after_approval(
                _row(owner_run_type="todo", owner_id="todo-1")
            )
        resume.assert_not_awaited()

    async def test_todo_resume_logs_and_reenqueues(self) -> None:
        from app.services.hil import resume as resume_module
        from app.services.hil.resume import resume_owner_after_approval

        with (
            patch.object(
                resume_module, "approval_ledger_repository"
            ) as repo,
            patch(f"{MODULE}._resume_todo", new=AsyncMock()) as resume,
        ):
            repo.claim_resume = AsyncMock(return_value=True)
            await resume_owner_after_approval(
                _row(owner_run_type="todo", owner_id="todo-1")
            )
        resume.assert_awaited_once()

    async def test_unknown_owner_type_claims_but_runs_nothing(self) -> None:
        from app.services.hil.resume import resume_owner_after_approval

        with patch(f"{MODULE}.approval_ledger_repository") as repo:
            repo.claim_resume = AsyncMock(return_value=True)
            with patch(f"{MODULE}.log"):
                await resume_owner_after_approval(
                    _row(owner_run_type="cron", owner_id="x")
                )
        repo.claim_resume.assert_awaited_once()

    async def test_resume_failure_never_raises(self) -> None:
        from app.services.hil.resume import resume_owner_after_approval

        with (
            patch(f"{MODULE}.approval_ledger_repository") as repo,
            patch(
                f"{MODULE}._resume_workflow",
                new=AsyncMock(side_effect=RuntimeError("redis down")),
            ),
        ):
            repo.claim_resume = AsyncMock(return_value=True)
            await resume_owner_after_approval(
                _row(owner_run_type="workflow", owner_id="wf-1")
            )


class TestResumeTodo:
    async def test_resume_enqueues_into_the_parked_conversation(self) -> None:
        # Lazy imports inside _resume_todo bind at call time: patch the source
        # modules, never the resume module's namespace.
        from app.services.hil.resume import _resume_todo

        pool = MagicMock()
        enqueued: dict[str, Any] = {}

        async def _fake_enqueue(pool_arg: Any, fn: str, *args: Any, **kwargs: Any) -> None:
            enqueued.update(fn=fn, args=args)

        with (
            patch(
                "app.utils.redis_utils.RedisPoolManager.get_pool",
                new=AsyncMock(return_value=pool),
            ),
            patch(
                "app.workers.queue.enqueue_worker_job",
                new=_fake_enqueue,
            ),
        ):
            await _resume_todo(_row(owner_run_type="todo", owner_id="todo-9"))
        # The parked conversation — never a fresh session. A fresh uuid would
        # orphan the parked run's thread, checkpoint, and partial results.
        assert enqueued["fn"] == "resume_tracked_todo"
        assert enqueued["args"] == ("todo-9", "conv-bg", "ap_bg1", "Send briefing")


class TestResumeWorkflow:
    async def test_requeued_with_receipt_context(self) -> None:
        from app.services.hil.resume import _resume_workflow

        queued: dict[str, Any] = {}

        async def _fake_queue(workflow_id: str, user_id: str, context: Any) -> bool:
            queued.update(workflow_id=workflow_id, user_id=user_id, context=context)
            return True

        with patch(
            "app.services.workflow.queue_service.WorkflowQueueService.queue_workflow_execution",
            new=_fake_queue,
        ):
            await _resume_workflow(_row(owner_run_type="workflow", owner_id="wf-7"))
        assert queued["workflow_id"] == "wf-7"
        assert queued["context"]["resume_from_approval"] == "ap_bg1"
        assert queued["context"]["approval_result"] == "granted"


class TestRecordDeny:
    async def test_todo_deny_leaves_a_skip_trace(self) -> None:
        from app.services.hil.resume import record_owner_deny

        service = MagicMock()
        service.append_activity_entry = AsyncMock(return_value=True)
        with patch(
            "app.services.tracked_todo_service.tracked_todo_service",
            service,
        ):
            await record_owner_deny(
                _row(owner_run_type="todo", owner_id="todo-3"), "too pricey"
            )
        entry = service.append_activity_entry.await_args.kwargs["entry"]
        assert "denied" in entry and "too pricey" in entry

    async def test_workflow_and_live_denies_record_nothing(self) -> None:
        from app.services.hil.resume import record_owner_deny

        service = MagicMock()
        service.append_activity_entry = AsyncMock()
        with patch(
            "app.services.tracked_todo_service.tracked_todo_service",
            service,
        ):
            await record_owner_deny(_row(owner_run_type="workflow", owner_id="wf-1"), None)
            await record_owner_deny(_row(), "nope")
        service.append_activity_entry.assert_not_called()
