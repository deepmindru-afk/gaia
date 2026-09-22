"""Auto mode's memory (app/services/hil/intent.py: AutoHistory).

Past decisions bias the judge: a tool the user keeps denying stops
auto-approving until the pattern flips. A history the store cannot produce
reads as "no signal", never as authorization.
"""

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

from app.models.hil_models import ApprovalLedgerDocument, LedgerState
from app.services.hil.intent import (
    AutoContext,
    AutoHistory,
    IntentDecision,
    JudgedCall,
    _Verdict,
    judge_intent,
    summarize_history,
)

MODULE = "app.services.hil.intent"
USER_TURNS = ["draft an email to bob about the deck", "looks good, send it"]


def _row(state: LedgerState, **overrides: Any) -> ApprovalLedgerDocument:
    return ApprovalLedgerDocument(
        **{
            "approval_id": "ap_x",
            "conversation_id": "c",
            "user_id": "u",
            "fingerprint": "f",
            "tool_name": "send_email",
            "args": {},
            "summary": "s",
            "state": state,
            "decided_at": datetime.now(UTC),
            **overrides,
        }
    )


def test_counts_and_last_deny_come_from_the_rows() -> None:
    now = datetime.now(UTC)
    rows = [
        _row(LedgerState.DENIED, feedback="too pricey", decided_at=now),
        _row(LedgerState.EXECUTED, decided_at=now - timedelta(days=1)),
        _row(LedgerState.DENIED, feedback="older", decided_at=now - timedelta(days=2)),
    ]
    h = summarize_history(rows)
    assert (h.approved_recent, h.denied_recent) == (1, 2)
    assert h.last_deny_feedback == "too pricey"


def test_blank_rows_are_no_signal() -> None:
    assert summarize_history([]) == AutoHistory()


async def _judge(history: AutoHistory) -> IntentDecision:
    """Real judge_intent, mocked LLM saying allow with a genuine quote."""
    llm_verdict = _Verdict(
        authorized_scope="email bob the deck",
        authorizing_quote="draft an email to bob about the deck",
        action_effect="sends an email",
        scope_gap="",
        risk_factors=[],
        injected_instructions=False,
        verdict="allow",
        reason="You asked to email Bob the deck.",
    )
    with patch(f"{MODULE}.ainvoke_structured", new=AsyncMock(return_value=llm_verdict)):
        return await judge_intent(
            AutoContext(user_id="u", history=history),
            user_messages=USER_TURNS,
            call=JudgedCall(
                tool_name="send_email",
                description="Send an email.",
                args={"to": "bob@example.com"},
                summary="Send email",
            ),
            prior_calls=[],
        )


async def test_repeated_denies_downgrade_an_allowed_call_to_ask() -> None:
    history = AutoHistory(approved_recent=0, denied_recent=3)
    d = await _judge(history)
    assert d.outcome == "ask"
    assert "denied" in d.reason


async def test_mostly_approved_history_leaves_an_allowed_call_accepted() -> None:
    history = AutoHistory(approved_recent=5, denied_recent=1)
    d = await _judge(history)
    assert d.outcome == "accept"


async def test_no_history_is_no_signal_and_leaves_an_allowed_call_accepted() -> None:
    d = await _judge(AutoHistory())
    assert d.outcome == "accept"


async def test_a_single_deny_blocks_accept_until_an_approval_lands() -> None:
    # Boundary: 0 approved vs 1 denied is a deny pattern, not a blank slate.
    assert (await _judge(AutoHistory(denied_recent=1))).outcome == "ask"
    assert (await _judge(AutoHistory(approved_recent=1, denied_recent=1))).outcome == "ask"
    assert (await _judge(AutoHistory(approved_recent=2, denied_recent=1))).outcome == "accept"
