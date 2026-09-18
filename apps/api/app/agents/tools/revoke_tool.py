"""Withdraw your own pending approval (executor-free HIL)."""

from typing import Annotated

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.constants.general import EXECUTOR_THREAD_PREFIX
from app.db.repositories.approval_ledger import approval_ledger_repository
from app.models.agent_models import agent_configurable
from app.models.hil_models import LedgerState


@tool
async def revoke_tool(
    config: RunnableConfig,
    approval_id: Annotated[
        str,
        "The pending approval id from the PENDING message (e.g. 'ap_3f9a1c'). "
        "Only approvals still PENDING can be withdrawn.",
    ],
) -> str:
    """Withdraw a pending approval you requested.

    Use it the moment a requested approval is no longer needed (the plan moved
    on, you found another way, the user countermanded it in chat). The row
    tombstones as REVOKED; the user sees "agent withdrew" and is never asked.
    Only the worker that proposed it (same thread) or the executor can revoke;
    decided rows are untouchable, so report those to the user instead.
    """
    configurable = agent_configurable(config)
    caller_thread = str(configurable.get("thread_id") or "")
    caller_conversation = str(configurable.get("conversation_id") or "")

    row = await approval_ledger_repository.get_by_approval_id(approval_id)
    # Unknown id and foreign conversation read identically: existence must not
    # leak across conversations.
    if row is None or row.conversation_id != caller_conversation:
        return f"No pending approval with id '{approval_id}'."

    if row.state is not LedgerState.PENDING:
        return (
            f"Cannot revoke '{approval_id}': already {row.state.value}. "
            "Report that to the user instead of retrying."
        )

    if caller_thread != row.owner_agent and not caller_thread.startswith(
        EXECUTOR_THREAD_PREFIX
    ):
        return (
            f"Cannot revoke '{approval_id}': it belongs to another worker. "
            "Only its proposer or the executor can withdraw it."
        )

    revoked = await approval_ledger_repository.transition(
        approval_id, LedgerState.PENDING, LedgerState.REVOKED
    )
    if not revoked:
        current = await approval_ledger_repository.get_by_approval_id(approval_id)
        state = current.state.value if current else "gone"
        return f"Cannot revoke '{approval_id}': already {state}."
    # Deferred import: ledger_decide reaches the tool registry (via dispatch),
    # which imports this module — top-level would close a cycle.
    from app.services.hil.ledger_decide import publish_ledger_revocation  # noqa: PLC0415

    revoked_row = await approval_ledger_repository.get_by_approval_id(approval_id)
    if revoked_row is not None:
        await publish_ledger_revocation(revoked_row)
    return f"Revoked '{approval_id}' ({row.summary}). It will never be asked."
