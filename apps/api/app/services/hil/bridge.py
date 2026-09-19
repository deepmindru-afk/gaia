"""Surface an approval request to the user's clients, and remember declines.

The gate pauses its run with LangGraph's ``interrupt()``; nothing here waits. This
module only *publishes*: it records the pending approval durably, pushes the
``approval_request`` tool_data card onto the turn's SSE stream, and wakes clients
that aren't watching it. The decision arrives out-of-band and is applied by
``app/services/hil/resolution.py``, which resumes the paused thread.

Frame delivery mirrors ``make_redis_stream_writer``: every frame is both published
to the replayable stream event log (live + reload) AND appended to the stream
session's tool-event collector so the executor drain path persists it. The gate
only fires inside the detached executor/subagent (comms holds no gated tools),
where ``get_stream_writer`` is unavailable — so this dual write, keyed purely by
``stream_id``, is what makes the card work at every nesting depth.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from typing import Any, cast

from app.agents.core.background.session import get_session
from app.constants.cache import HIL_DECLINED_PREFIX
from app.constants.hil import (
    APPROVAL_REQUEST_TOOL_NAME,
    APPROVAL_TOOL_CATEGORY,
    HIL_APPROVAL_TIMEOUT_SECONDS,
    HIL_CLASSIFIER_MAX_ARG_CHARS,
    HIL_CLASSIFIER_MAX_ARGS,
    HIL_CLASSIFIER_MAX_DETAIL_CHARS,
    HIL_DECLINE_MEMORY_TTL_SECONDS,
    HIL_SUMMARY_MAX_ARG_CHARS,
    HIL_SUMMARY_MAX_ARGS,
)
from app.constants.log_tags import LogTag
from app.core.stream_manager import stream_manager
from app.db.redis import redis_cache
from app.db.repositories.conversations import conversation_repository
from app.models.hil_models import DeclinedCallRecord, HILApprovalRecord, HILApprovalStatus
from app.models.stream_events import ApprovalRequestEntry, ApprovalRequestEntryData
from app.services.hil.approvals_store import record_auto_approval, upsert_pending_approval
from app.services.hil.notify import notify_approval_pending
from app.services.hil.utils import GatedCall
from app.utils.general_utils import clip_text
from shared.py.wide_events import log, spawn_logged_task


@dataclass
class ApprovalOutcome:
    """The resolved result of an approval request: status, optional feedback, and scope."""

    status: HILApprovalStatus
    feedback: str | None = None
    scope: str = "once"


async def publish_approval_request(
    *,
    approval_id: str,
    stream_id: str,
    user_id: str,
    conversation_id: str,
    tool_call: GatedCall,
    summary: str,
    integration_name: str | None,
) -> None:
    """Record the pending approval and surface its card — exactly once.

    The gate re-enters this on every resume replay (the node re-runs from the
    top), so the card and the notification are gated on whether the upsert
    actually created the record. A replay is a no-op.
    """
    created = await upsert_pending_approval(
        approval_id=approval_id,
        user_id=user_id,
        conversation_id=conversation_id,
        stream_id=stream_id,
        tool_name=tool_call.name,
        tool_call_id=tool_call.id,
        args=tool_call.args,
        summary=summary,
        integration_name=integration_name,
    )
    if not created:
        return

    log.set(hil={"approval_id": approval_id, "tool": tool_call.name, "stream_id": stream_id})
    await _publish_entry(
        stream_id,
        _approval_entry(
            approval_id, tool_call, HILApprovalStatus.PENDING, summary, integration_name
        ),
    )
    _schedule_pending_notification(user_id, conversation_id, approval_id, summary)


async def publish_ledger_request(
    *,
    approval_id: str,
    stream_id: str,
    user_id: str,
    conversation_id: str,
    tool_call: GatedCall,
    summary: str,
    integration_name: str | None,
    rationale: str | None = None,
    live: bool = True,
) -> None:
    """Surface a ledger PENDING card — exactly once per registration.

    Same ``approval_request`` wire shape as the barrier path (same tool_name,
    so web TOOL_RENDERERS and bot streaming render it with no registry
    changes); the ledger-only fields (rationale, age 0, version 0) ride as
    optional extras old clients ignore. Called only on fresh registration —
    the dedup hit means the card is already up, so the caller skips this.

    On a live run the card is HELD, not streamed: the frame is recorded on
    the session (so the end-of-run drain persists it) but no SSE chunk or
    push goes out until the run ends. A revoke before then removes the
    unshown frame instead of tombstoning it, so the user never sees work
    the model already withdrew. Background runs hold nothing — nobody is
    watching, and the push is the only signal.
    """
    log.set(hil={"approval_id": approval_id, "tool": tool_call.name, "stream_id": stream_id})
    entry = _approval_entry(
        approval_id,
        tool_call,
        HILApprovalStatus.PENDING,
        summary,
        integration_name,
    )
    entry.data.rationale = rationale
    entry.data.age_seconds = 0
    entry.data.ledger_version = 0
    if not live:
        await _publish_entry(stream_id, entry)
        _schedule_pending_notification(user_id, conversation_id, approval_id, summary)
        return
    frame = {"tool_data": entry.model_dump(), "_held_approval": True}
    session = get_session(stream_id)
    if session is not None:
        session.tool_events.append(frame)


async def publish_decision(
    record: HILApprovalRecord, status: HILApprovalStatus, *, stream_id: str, feedback: str | None
) -> None:
    """Settle this approval's card, on the stream the user is watching NOW.

    Never ``record.stream_id``: that is the stream the request was raised on, and a run
    that paused resumes on a fresh one (``prepare_run_from_item``), leaving the original
    closed. The client follows the new stream via ``executor.stream_started``, so a card
    settled on the old one resolves where nobody is looking.
    """
    await _publish_entry(
        stream_id,
        _approval_entry(
            record.approval_id,
            GatedCall(name=record.tool_name, id=record.tool_call_id, args=record.args),
            status,
            record.summary,
            record.integration_name,
            feedback,
        ),
    )
    # Also settle the PERSISTED frame right now. Final delivery reconciles too,
    # but the run may pause again on a later gate first — a revisit in that
    # window would otherwise render a dead pending card for a decided approval.
    # Isolated on purpose: this is a redraw of an already-decided card, and the
    # caller is the gate, which fails CLOSED. Letting a write error escape here
    # would turn a cosmetic failure into a denial of the user's own decision.
    try:
        await conversation_repository.set_message_approval_status(
            record.conversation_id,
            user_id=record.user_id,
            approval_id=record.approval_id,
            status=status.value,
        )
    except Exception as e:
        log.error(
            f"{LogTag.HIL} Could not settle persisted approval frame; delivery will reconcile",
            approval_id=record.approval_id,
            error=str(e),
            error_type=type(e).__name__,
        )


async def publish_auto_approval(
    *,
    approval_id: str,
    stream_id: str,
    user_id: str,
    conversation_id: str,
    tool_call: GatedCall,
    summary: str,
    integration_name: str | None,
    reason: str,
) -> None:
    """Record and surface an action auto mode ran without asking.

    The card is published already settled, so it needs no decision and wakes nobody — it
    is a receipt, not a request. Auto mode should never mean the user cannot see what was
    done in their name.
    """
    await record_auto_approval(
        approval_id=approval_id,
        user_id=user_id,
        conversation_id=conversation_id,
        stream_id=stream_id,
        tool_name=tool_call.name,
        tool_call_id=tool_call.id,
        args=tool_call.args,
        summary=summary,
        integration_name=integration_name,
        reason=reason,
    )
    await _publish_entry(
        stream_id,
        _approval_entry(
            approval_id,
            tool_call,
            HILApprovalStatus.AUTO_APPROVED,
            summary,
            integration_name,
            auto_reason=reason,
        ),
    )


async def remember_declined_call(
    stream_id: str, tool_name: str, args: dict[str, Any], feedback: str | None
) -> None:
    """Record that the user declined this exact call for the rest of the turn."""
    if not redis_cache.redis:
        return
    record: DeclinedCallRecord = {"feedback": feedback}
    await redis_cache.set(
        _declined_key(stream_id, tool_name, args),
        record,
        ttl=HIL_DECLINE_MEMORY_TTL_SECONDS,
    )


async def recall_declined_call(
    stream_id: str, tool_name: str, args: dict[str, Any]
) -> ApprovalOutcome | None:
    """The prior decline for this exact call in this turn, if any — so the gate
    can auto-deny a retry with the user's original feedback and never re-prompt."""
    if not redis_cache.redis:
        return None
    raw = await redis_cache.get(_declined_key(stream_id, tool_name, args))
    if not raw:
        return None
    # Correct by construction: the only writer is ``remember_declined_call`` above.
    record = cast(DeclinedCallRecord, raw)
    return ApprovalOutcome(status=HILApprovalStatus.DENIED, feedback=record.get("feedback"))


def build_summary(tool_name: str, args: dict[str, Any], integration_name: str | None) -> str:
    """Deterministic one-line summary of a gated call (no LLM in the hot path)."""
    label = tool_name.replace("_", " ").strip().capitalize()
    if integration_name:
        label = f"{label} ({integration_name})"
    parts = _summary_arg_parts(args)
    return f"{label} — {', '.join(parts)}" if parts else label


def build_action_detail(summary: str, args: dict[str, Any]) -> str:
    """Richer rendering of a gated call for the conversational classifier.

    The card's one-line ``summary`` (tool + integration identity, truncated args)
    as the label, plus every argument up to a bound with non-scalar values as
    compact JSON — so the classifier sees the full content (recipient, subject,
    body, ...) the summary omits. The total is capped by
    ``HIL_CLASSIFIER_MAX_DETAIL_CHARS``; the per-value clip only stops one
    pathological arg from eating the whole budget. No LLM here."""
    lines = [summary]
    arg_lines = []
    for key, value in list((args or {}).items())[:HIL_CLASSIFIER_MAX_ARGS]:
        rendered = (
            value if isinstance(value, (str, int, float, bool)) else json.dumps(value, default=str)
        )
        arg_lines.append(f"  {key}: {clip_text(str(rendered), HIL_CLASSIFIER_MAX_ARG_CHARS)}")
    if arg_lines:
        lines.append("Arguments:")
        lines.extend(arg_lines)
    return clip_text("\n".join(lines), HIL_CLASSIFIER_MAX_DETAIL_CHARS)


# --- internals -----------------------------------------------------------------


def _schedule_pending_notification(
    user_id: str, conversation_id: str, approval_id: str, summary: str
) -> None:
    """Wake clients not watching the stream. Detached — a notify failure must
    never block the gate."""
    spawn_logged_task(
        "approval_pending_notification",
        notify_approval_pending(user_id, conversation_id, approval_id, summary),
        user={"id": user_id},
        conversation_id=conversation_id,
        approval_id=approval_id,
    )


async def _publish_entry(stream_id: str, entry: ApprovalRequestEntry) -> None:
    """Deliver a frame live (replayable event log) AND record it for persistence.

    The session append mirrors ``make_redis_stream_writer`` so the executor
    drain path persists the card; the SSE publish reaches live/reloaded clients.
    Both carry the same plain-dict frame the rest of the tool_data pipeline
    (``stream_utils``, the bot bridge, the frontend parser) reads.
    """
    frame = {"tool_data": entry.model_dump()}
    await stream_manager.publish_chunk(stream_id, f"data: {json.dumps(frame)}\n\n")
    session = get_session(stream_id)
    if session is not None:
        session.tool_events.append(frame)


def settle_session_approval_frame(
    stream_id: str,
    approval_id: str,
    status: str,
    feedback: str | None = None,
    *,
    drop_if_unpublished: bool = False,
) -> bool:
    """Flip an already-recorded card frame to its terminal status, in place.

    Revoke and late decisions otherwise miss the persisted message (it is
    created at drain time, after the run ends) while the stale PENDING frame
    drains back onto it — resurrecting a settled card. Mutating in place
    means the drain upsert is a no-op safety net, not a resurrection path.

    With ``drop_if_unpublished``, a still-held frame (never streamed live) is
    removed instead of flipped: a revoke before the run ends means the user
    never saw the card, so there is nothing to tombstone. Best-effort like
    every other delivery here: returns whether a frame was settled or
    dropped, never raises.
    """
    try:
        session = get_session(stream_id)
        if session is None:
            return False
        settled = False
        kept: list[dict[str, Any]] = []
        for event in session.tool_events:
            tool_data = event.get("tool_data") if isinstance(event, dict) else None
            if not isinstance(tool_data, dict):
                kept.append(event)
                continue
            if tool_data.get("tool_name") != APPROVAL_REQUEST_TOOL_NAME:
                kept.append(event)
                continue
            data = tool_data.get("data")
            if not isinstance(data, dict) or data.get("approval_id") != approval_id:
                kept.append(event)
                continue
            if drop_if_unpublished and event.get("_held_approval") is True:
                settled = True
                continue
            data["status"] = status
            if feedback is not None:
                data["feedback"] = feedback
            kept.append(event)
            settled = True
        session.tool_events[:] = kept
        return settled
    except Exception:
        return False


def _approval_entry(
    approval_id: str,
    tool_call: GatedCall,
    status: HILApprovalStatus,
    summary: str,
    integration_name: str | None,
    feedback: str | None = None,
    auto_reason: str | None = None,
) -> ApprovalRequestEntry:
    return ApprovalRequestEntry(
        tool_name=APPROVAL_REQUEST_TOOL_NAME,
        tool_category=APPROVAL_TOOL_CATEGORY,
        data=ApprovalRequestEntryData(
            approval_id=approval_id,
            tool_call_id=tool_call.id,
            gated_tool_name=tool_call.name,
            integration_name=integration_name,
            summary=summary,
            args_preview=tool_call.args,
            status=status,
            feedback=feedback,
            auto_reason=auto_reason,
            timeout_seconds=int(HIL_APPROVAL_TIMEOUT_SECONDS),
        ),
        timestamp=datetime.now(UTC).isoformat(),
    )


def _summary_arg_parts(args: dict[str, Any]) -> list[str]:
    """A few short ``key: value`` scalars for the card's one-line summary."""
    parts: list[str] = []
    for key, value in (args or {}).items():
        if len(parts) >= HIL_SUMMARY_MAX_ARGS:
            break
        if isinstance(value, (str, int, float, bool)):
            parts.append(f"{key}: {clip_text(str(value), HIL_SUMMARY_MAX_ARG_CHARS)}")
    return parts


def _declined_key(stream_id: str, tool_name: str, args: dict[str, Any]) -> str:
    return f"{HIL_DECLINED_PREFIX}{stream_id}:{tool_name}:{_args_hash(args)}"


def _args_hash(args: dict[str, Any]) -> str:
    # md5 over the canonical args — a cache key, not a security boundary.
    payload = json.dumps(args or {}, sort_keys=True, default=str)
    return hashlib.md5(payload.encode(), usedforsecurity=False).hexdigest()  # nosec B324
