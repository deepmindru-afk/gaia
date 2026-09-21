"""Ledger branch of the approval gate (flag on: register, never interrupt)."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.tools import StructuredTool
from pydantic import BaseModel
import pytest

from app.constants.hil import HIL_STATUS_KWARG

from .conftest import CONVERSATION_ID, STREAM_ID, USER_ID, make_request

MODULE = "app.services.hil.gate"


@pytest.fixture(autouse=True)
def _quiet_log():
    with patch(f"{MODULE}.log"):
        yield


def _ledger(
    live: Any = None, denied: Any = None, registered_id: str = "ap_abc1234567"
) -> MagicMock:
    ledger = MagicMock()
    ledger.find_live = AsyncMock(return_value=live)
    ledger.find_latest_denied = AsyncMock(return_value=denied)
    ledger.register = AsyncMock(return_value=registered_id)
    return ledger


def _gated_request(**overrides: Any):
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


@pytest.mark.unit
class TestLedgerBranch:
    async def test_registers_pending_and_never_interrupts(self) -> None:
        from app.services.hil import gate

        ledger = _ledger()
        with (
            patch(f"{MODULE}.is_hil_ledger_enabled", new=AsyncMock(return_value=True)),
            patch(f"{MODULE}.resolve_policy", new=AsyncMock(return_value="ask")),
            patch(f"{MODULE}.approval_ledger_repository", new=ledger),
            patch(f"{MODULE}._integration_name_for", new=AsyncMock(return_value="gmail")),
            patch(f"{MODULE}.publish_ledger_request", new=AsyncMock()),
            patch(f"{MODULE}.interrupt") as intr,
        ):
            result = await gate.decide_tool_call(_gated_request())

        ledger.register.assert_awaited_once()
        intr.assert_not_called()
        assert result is not None
        assert "PENDING ap_abc1234567" in str(result.content)
        assert "needs the user's explicit approval" in str(result.content)
        assert 'execute(tool_name="revoke"' in str(result.content)
        assert "approve ticket" in str(result.content)
        assert result.additional_kwargs[HIL_STATUS_KWARG] == "pending"

    async def test_background_pending_points_at_the_approvals_tab(self) -> None:
        """A background run has no watcher: the card lives in the Approvals
        tab, so the guidance must say so instead of "when this run ends"."""
        from app.services.hil import gate

        ledger = _ledger()
        request = _gated_request(execution_mode="background")
        with (
            patch(f"{MODULE}.is_hil_ledger_enabled", new=AsyncMock(return_value=True)),
            patch(f"{MODULE}.resolve_policy", new=AsyncMock(return_value="ask")),
            patch(f"{MODULE}.approval_ledger_repository", new=ledger),
            patch(f"{MODULE}._integration_name_for", new=AsyncMock(return_value="gmail")),
            patch(f"{MODULE}.publish_ledger_request", new=AsyncMock()),
            patch(f"{MODULE}.interrupt") as intr,
        ):
            result = await gate.decide_tool_call(request)

        intr.assert_not_called()
        assert result is not None
        assert "Approvals tab" in str(result.content)
        assert "when this run ends" not in str(result.content)

    async def test_background_workflow_run_tags_the_ledger_owner(self) -> None:
        """The resume driver needs to know WHAT parked: a background workflow
        run stamps its owner on the row; nothing else does."""
        from app.services.hil import gate

        ledger = _ledger()
        request = _gated_request(execution_mode="background", workflow_id="wf-1")
        with (
            patch(f"{MODULE}.is_hil_ledger_enabled", new=AsyncMock(return_value=True)),
            patch(f"{MODULE}.resolve_policy", new=AsyncMock(return_value="ask")),
            patch(f"{MODULE}.approval_ledger_repository", new=ledger),
            patch(f"{MODULE}._integration_name_for", new=AsyncMock(return_value="gmail")),
            patch(f"{MODULE}.publish_ledger_request", new=AsyncMock()),
            patch(f"{MODULE}.interrupt"),
        ):
            await gate.decide_tool_call(request)

        assert ledger.register.await_args.kwargs["owner_run_type"] == "workflow"
        assert ledger.register.await_args.kwargs["owner_id"] == "wf-1"

    async def test_background_run_threads_the_owner_to_publish(self) -> None:
        """Publish is what raises the sidebar flag — it must receive the same
        owner the row carries, or background cards never surface."""
        from app.services.hil import gate

        ledger = _ledger()
        request = _gated_request(execution_mode="background", active_todo_id="todo-9")
        with (
            patch(f"{MODULE}.is_hil_ledger_enabled", new=AsyncMock(return_value=True)),
            patch(f"{MODULE}.resolve_policy", new=AsyncMock(return_value="ask")),
            patch(f"{MODULE}.approval_ledger_repository", new=ledger),
            patch(f"{MODULE}._integration_name_for", new=AsyncMock(return_value="gmail")),
            patch(f"{MODULE}.publish_ledger_request", new=AsyncMock()) as pub,
            patch(f"{MODULE}.interrupt"),
        ):
            await gate.decide_tool_call(request)

        assert pub.await_args.kwargs["owner_run_type"] == "todo"
        assert pub.await_args.kwargs["owner_id"] == "todo-9"

    async def test_live_run_publishes_with_no_owner(self) -> None:
        """Live runs resume through the inbox — an owner here would wrongly
        surface (and re-enqueue) them."""
        from app.services.hil import gate

        ledger = _ledger()
        with (
            patch(f"{MODULE}.is_hil_ledger_enabled", new=AsyncMock(return_value=True)),
            patch(f"{MODULE}.resolve_policy", new=AsyncMock(return_value="ask")),
            patch(f"{MODULE}.approval_ledger_repository", new=ledger),
            patch(f"{MODULE}._integration_name_for", new=AsyncMock(return_value="gmail")),
            patch(f"{MODULE}.publish_ledger_request", new=AsyncMock()) as pub,
            patch(f"{MODULE}.interrupt"),
        ):
            await gate.decide_tool_call(_gated_request())

        assert pub.await_args.kwargs["owner_run_type"] == ""
        assert pub.await_args.kwargs["owner_id"] == ""

    async def test_background_todo_run_tags_the_ledger_owner(self) -> None:
        from app.services.hil import gate

        ledger = _ledger()
        request = _gated_request(execution_mode="background", active_todo_id="todo-9")
        with (
            patch(f"{MODULE}.is_hil_ledger_enabled", new=AsyncMock(return_value=True)),
            patch(f"{MODULE}.resolve_policy", new=AsyncMock(return_value="ask")),
            patch(f"{MODULE}.approval_ledger_repository", new=ledger),
            patch(f"{MODULE}._integration_name_for", new=AsyncMock(return_value="gmail")),
            patch(f"{MODULE}.publish_ledger_request", new=AsyncMock()),
            patch(f"{MODULE}.interrupt"),
        ):
            await gate.decide_tool_call(request)

        assert ledger.register.await_args.kwargs["owner_run_type"] == "todo"
        assert ledger.register.await_args.kwargs["owner_id"] == "todo-9"

    async def test_live_run_with_ids_tags_no_owner(self) -> None:
        """Owner tagging is resume-scoped: a live run resumes through the
        executor inbox and must never re-enqueue, even carrying the keys."""
        from app.services.hil import gate

        ledger = _ledger()
        request = _gated_request(workflow_id="wf-1", active_todo_id="todo-9")
        with (
            patch(f"{MODULE}.is_hil_ledger_enabled", new=AsyncMock(return_value=True)),
            patch(f"{MODULE}.resolve_policy", new=AsyncMock(return_value="ask")),
            patch(f"{MODULE}.approval_ledger_repository", new=ledger),
            patch(f"{MODULE}._integration_name_for", new=AsyncMock(return_value="gmail")),
            patch(f"{MODULE}.publish_ledger_request", new=AsyncMock()),
            patch(f"{MODULE}.interrupt"),
        ):
            await gate.decide_tool_call(request)

        assert ledger.register.await_args.kwargs["owner_run_type"] == ""
        assert ledger.register.await_args.kwargs["owner_id"] == ""

    async def test_flag_off_takes_the_old_interrupt_path(self) -> None:
        from app.services.hil import gate

        with (
            patch(f"{MODULE}.is_hil_ledger_enabled", new=AsyncMock(return_value=False)),
            patch(f"{MODULE}.resolve_policy", new=AsyncMock(return_value="ask")),
            patch(f"{MODULE}.approval_ledger_repository", new=_ledger()) as ledger,
            patch(f"{MODULE}._integration_name_for", new=AsyncMock(return_value="gmail")),
            patch(f"{MODULE}.get_approval", new=AsyncMock(return_value=None)),
            patch(f"{MODULE}.recall_declined_call", new=AsyncMock(return_value=None)),
            patch(f"{MODULE}.publish_approval_request", new=AsyncMock()),
            patch(f"{MODULE}.interrupt") as intr,
        ):
            await gate.decide_tool_call(_gated_request())

        ledger.find_live.assert_not_awaited()
        intr.assert_called_once()

    async def test_live_duplicate_returns_existing_id_without_register(self) -> None:
        from app.models.hil_models import LedgerState
        from app.services.hil import gate

        live = MagicMock(approval_id="ap_live", summary="Send it")
        live.state = LedgerState.PENDING
        ledger = _ledger(live=live)
        with (
            patch(f"{MODULE}.is_hil_ledger_enabled", new=AsyncMock(return_value=True)),
            patch(f"{MODULE}.resolve_policy", new=AsyncMock(return_value="ask")),
            patch(f"{MODULE}.approval_ledger_repository", new=ledger),
            patch(f"{MODULE}.interrupt") as intr,
        ):
            result = await gate.decide_tool_call(_gated_request())

        ledger.register.assert_not_awaited()
        intr.assert_not_called()
        assert result is not None
        assert "PENDING ap_live" in str(result.content)
        assert "already requested" in str(result.content)

    async def test_live_approved_row_is_not_called_awaiting_decision(self) -> None:
        from app.models.hil_models import LedgerState
        from app.services.hil import gate

        live = MagicMock(approval_id="ap_live", summary="Send it")
        live.state = LedgerState.APPROVED
        ledger = _ledger(live=live)
        with (
            patch(f"{MODULE}.is_hil_ledger_enabled", new=AsyncMock(return_value=True)),
            patch(f"{MODULE}.resolve_policy", new=AsyncMock(return_value="ask")),
            patch(f"{MODULE}.approval_ledger_repository", new=ledger),
            patch(f"{MODULE}.interrupt") as intr,
        ):
            result = await gate.decide_tool_call(_gated_request())

        ledger.register.assert_not_awaited()
        intr.assert_not_called()
        assert result is not None
        assert "APPROVED ap_live" in str(result.content)
        assert "awaiting redeem" in str(result.content)

    async def test_same_run_denied_reissue_is_refused(self) -> None:
        from app.services.hil import gate

        denied = MagicMock(proposing_run_id=STREAM_ID, feedback="too broad", decided_at=None)
        ledger = _ledger(denied=denied)
        with (
            patch(f"{MODULE}.is_hil_ledger_enabled", new=AsyncMock(return_value=True)),
            patch(f"{MODULE}.resolve_policy", new=AsyncMock(return_value="ask")),
            patch(f"{MODULE}.approval_ledger_repository", new=ledger),
            patch(f"{MODULE}.interrupt") as intr,
        ):
            result = await gate.decide_tool_call(_gated_request())

        ledger.register.assert_not_awaited()
        intr.assert_not_called()
        assert result is not None
        assert "REFUSED" in str(result.content)
        assert "DO NOT re-request" in str(result.content)
        assert result.additional_kwargs[HIL_STATUS_KWARG] == "denied"

    async def test_older_denied_run_registers_with_the_why_attached(self) -> None:
        from app.services.hil import gate

        denied = MagicMock(proposing_run_id="other-stream", feedback="wrong day", decided_at=None)
        ledger = _ledger(denied=denied)
        with (
            patch(f"{MODULE}.is_hil_ledger_enabled", new=AsyncMock(return_value=True)),
            patch(f"{MODULE}.resolve_policy", new=AsyncMock(return_value="ask")),
            patch(f"{MODULE}.approval_ledger_repository", new=ledger),
            patch(f"{MODULE}._integration_name_for", new=AsyncMock(return_value="gmail")),
            patch(f"{MODULE}.publish_ledger_request", new=AsyncMock()),
            patch(f"{MODULE}.interrupt") as intr,
        ):
            result = await gate.decide_tool_call(_gated_request())

        ledger.register.assert_awaited_once()
        intr.assert_not_called()
        assert result is not None
        assert "PENDING ap_abc1234567" in str(result.content)
        assert "wrong day" in str(result.content)

    async def test_ledger_failure_fails_closed_without_interrupt(self) -> None:
        from app.services.hil import gate

        ledger = _ledger()
        ledger.find_live = AsyncMock(side_effect=ConnectionError("mongo down"))
        with (
            patch(f"{MODULE}.is_hil_ledger_enabled", new=AsyncMock(return_value=True)),
            patch(f"{MODULE}.resolve_policy", new=AsyncMock(return_value="ask")),
            patch(f"{MODULE}.approval_ledger_repository", new=ledger),
            patch(f"{MODULE}.interrupt") as intr,
        ):
            result = await gate.decide_tool_call(_gated_request())

        intr.assert_not_called()
        assert result is not None
        assert result.additional_kwargs[HIL_STATUS_KWARG] == "error"


@pytest.mark.unit
class TestLedgerAutoParity:
    async def test_auto_aligned_call_runs_without_card_or_row(self) -> None:
        """Auto mode keeps its intent judge on the ledger path: an aligned
        call clears to run with no card and no ledger row, exactly like the
        barrier path. The flag flip must never change what gets asked."""
        from app.services.hil import gate
        from app.services.hil.intent import IntentDecision

        ledger = _ledger()
        with (
            patch(f"{MODULE}.is_hil_ledger_enabled", new=AsyncMock(return_value=True)),
            patch(f"{MODULE}.resolve_policy", new=AsyncMock(return_value="auto")),
            patch(f"{MODULE}.approval_ledger_repository", new=ledger),
            patch(f"{MODULE}._integration_name_for", new=AsyncMock(return_value="gmail")),
            patch(
                f"{MODULE}._judge",
                new=AsyncMock(return_value=IntentDecision(outcome="accept", reason="asked")),
            ),
            patch(f"{MODULE}.publish_ledger_request", new=AsyncMock()) as pub,
            patch(f"{MODULE}.interrupt") as intr,
        ):
            result = await gate.decide_tool_call(_gated_request())

        assert result is None
        ledger.register.assert_not_awaited()
        pub.assert_not_awaited()
        intr.assert_not_called()

    async def test_auto_misaligned_call_still_registers(self) -> None:
        from app.services.hil import gate
        from app.services.hil.intent import IntentDecision

        ledger = _ledger()
        with (
            patch(f"{MODULE}.is_hil_ledger_enabled", new=AsyncMock(return_value=True)),
            patch(f"{MODULE}.resolve_policy", new=AsyncMock(return_value="auto")),
            patch(f"{MODULE}.approval_ledger_repository", new=ledger),
            patch(f"{MODULE}._integration_name_for", new=AsyncMock(return_value="gmail")),
            patch(
                f"{MODULE}._judge",
                new=AsyncMock(return_value=IntentDecision(outcome="ask", reason="unclear")),
            ),
            patch(f"{MODULE}.publish_ledger_request", new=AsyncMock()),
            patch(f"{MODULE}.interrupt") as intr,
        ):
            result = await gate.decide_tool_call(_gated_request())

        ledger.register.assert_awaited_once()
        intr.assert_not_called()
        assert result is not None
        assert "PENDING ap_abc1234567" in str(result.content)
        assert "auto-approve did not cover" in str(result.content)


class _StrictArgs(BaseModel):
    to: str
    subject: str


def _strict_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=lambda: None,
        name="GMAIL_SEND_EMAIL",
        description="Send.",
        args_schema=_StrictArgs,
    )


@pytest.mark.unit
class TestLedgerBranchValidatesArgs:
    async def test_invalid_args_fail_before_any_card_exists(self) -> None:
        """No card for malformed args: the user must never approve a call the
        model will have to retry. Invalid args fail fast with the schema
        error; nothing registers, nothing publishes, nothing interrupts."""
        from app.services.hil import gate

        ledger = _ledger()
        with (
            patch(f"{MODULE}.is_hil_ledger_enabled", new=AsyncMock(return_value=True)),
            patch(f"{MODULE}.resolve_policy", new=AsyncMock(return_value="ask")),
            patch(f"{MODULE}.approval_ledger_repository", new=ledger),
            patch(f"{MODULE}.gated_tool_object", new=AsyncMock(return_value=_strict_tool())),
            patch(f"{MODULE}._integration_name_for", new=AsyncMock(return_value="gmail")),
            patch(f"{MODULE}.publish_ledger_request", new=AsyncMock()) as pub,
            patch(f"{MODULE}.interrupt") as intr,
        ):
            result = await gate.decide_tool_call(_gated_request())

        ledger.register.assert_not_awaited()
        pub.assert_not_awaited()
        intr.assert_not_called()
        assert result is not None
        assert result.additional_kwargs[HIL_STATUS_KWARG] == "error"
        assert "subject" in str(result.content)
        assert "no approval was requested" in str(result.content)

    async def test_unresolvable_tool_skips_validation(self) -> None:
        """Resolution failure must not gate: execution validates
        authoritatively, the gate only pre-filters what it can read."""
        from app.services.hil import gate

        ledger = _ledger()
        with (
            patch(f"{MODULE}.is_hil_ledger_enabled", new=AsyncMock(return_value=True)),
            patch(f"{MODULE}.resolve_policy", new=AsyncMock(return_value="ask")),
            patch(f"{MODULE}.approval_ledger_repository", new=ledger),
            patch(f"{MODULE}.gated_tool_object", new=AsyncMock(return_value=None)),
            patch(f"{MODULE}._integration_name_for", new=AsyncMock(return_value="gmail")),
            patch(f"{MODULE}.publish_ledger_request", new=AsyncMock()) as pub,
            patch(f"{MODULE}.interrupt") as intr,
        ):
            result = await gate.decide_tool_call(_gated_request())

        ledger.register.assert_awaited_once()
        pub.assert_awaited_once()
        intr.assert_not_called()
        assert result is not None
        assert "PENDING ap_abc1234567" in str(result.content)
