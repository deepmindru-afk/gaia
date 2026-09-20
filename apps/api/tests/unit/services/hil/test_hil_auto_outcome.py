"""Auto mode v2 outcomes (app/services/hil/intent.py).

The judge returns accept / reject / ask — never a bare boolean — so the gate
can refuse with a reason (no card) distinctly from asking (card). Tests here
pin the outcome vocabulary; the adversarial LLM-judge tests live in
test_hil_intent.py.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

from app.services.hil.intent import (
    IntentDecision,
    IntentJudge,
    JudgedCall,
    judge_intent,
)
from app.services.hil.utils import PriorCall

MODULE = "app.services.hil.intent"


async def test_reject_outcome_is_not_aligned_and_carries_reason() -> None:
    d = IntentDecision(outcome="reject", reason="You denied this twice already.")
    assert d.aligned is False
    assert d.reason == "You denied this twice already."


async def test_accept_outcome_stays_aligned() -> None:
    assert IntentDecision(outcome="accept", reason="r").aligned is True
    assert IntentDecision(outcome="ask", reason="r").aligned is False


class FakeJudge(IntentJudge):
    """A future JEV-backed judge stands here: same Protocol, no LLM."""

    def __init__(self, decision: IntentDecision) -> None:
        self._decision = decision
        self.calls = 0

    async def decide(
        self,
        *,
        user_id: str,
        user_messages: list[str],
        call: JudgedCall,
        prior_calls: list[PriorCall],
        history: Any,
    ) -> IntentDecision:
        self.calls += 1
        return self._decision


def _call() -> JudgedCall:
    return JudgedCall(
        tool_name="send_email",
        description="Send an email.",
        args={"to": "bob@example.com"},
        summary="Send email — to: bob@example.com",
    )


async def test_injected_judge_decides_without_calling_the_llm() -> None:
    fake = FakeJudge(IntentDecision(outcome="reject", reason="nope"))
    with patch(f"{MODULE}.ainvoke_structured", new=AsyncMock()) as llm:
        d = await judge_intent(
            user_id="u",
            user_messages=["please send the deck to bob now"],
            call=_call(),
            prior_calls=[],
            judge=fake,
        )
    assert d.outcome == "reject"
    assert llm.await_count == 0
    assert fake.calls == 1


async def test_never_auto_tool_asks_without_calling_the_llm() -> None:
    # A user deny-rule: the judge declines to judge, so the call gets a card.
    with patch(f"{MODULE}.ainvoke_structured", new=AsyncMock()) as llm:
        d = await judge_intent(
            user_id="u",
            user_messages=["please send the deck to bob now"],
            call=_call(),
            prior_calls=[],
            never_auto_tools=frozenset({"send_email"}),
        )
    assert d.outcome == "ask"
    assert "send_email" in d.reason
    assert llm.await_count == 0


async def test_other_tools_ignore_the_never_auto_list() -> None:
    from app.services.hil.intent import _Verdict

    llm_verdict = _Verdict(
        authorized_scope="email bob",
        authorizing_quote="please send the deck to bob now",
        action_effect="sends an email",
        scope_gap="",
        risk_factors=[],
        injected_instructions=False,
        verdict="allow",
        reason="You asked for it.",
    )
    with patch(f"{MODULE}.ainvoke_structured", new=AsyncMock(return_value=llm_verdict)):
        d = await judge_intent(
            user_id="u",
            user_messages=["please send the deck to bob now"],
            call=_call(),
            prior_calls=[],
            never_auto_tools=frozenset({"wipe_database"}),
        )
    assert d.outcome == "accept"


async def test_gate_threads_never_auto_prefs_into_the_judge() -> None:
    from app.models.hil_models import HILPreferences
    from app.services.hil import gate

    from .conftest import CONVERSATION_ID, STREAM_ID, USER_ID, make_request, make_tool

    request = make_request(
        tool=make_tool(),
        configurable={
            "stream_id": STREAM_ID,
            "user_id": USER_ID,
            "conversation_id": CONVERSATION_ID,
            "user_messages": ["send the deck to bob"],
        },
    )
    context = gate.read_gate_context(request)
    assert context is not None
    with (
        patch.object(
            gate,
            "get_hil_preferences",
            new=AsyncMock(
                return_value=HILPreferences(mode="auto", never_auto_tools=["send_email"])
            ),
        ),
        patch.object(
            gate,
            "judge_intent",
            new=AsyncMock(return_value=IntentDecision(outcome="ask", reason="x")),
        ) as judge,
    ):
        await gate._judge(request, context, gate.unpack_tool_call(request), None, "s")
    assert judge.await_args.kwargs["never_auto_tools"] == frozenset({"send_email"})


async def test_unreadable_prefs_do_not_take_the_judge_down() -> None:
    # Prefs were already read for policy; a blip here skips the rule, not the call.
    from app.services.hil import gate

    from .conftest import make_request, make_tool

    request = make_request(tool=make_tool())
    context = gate.read_gate_context(request)
    assert context is not None
    with (
        patch.object(
            gate, "get_hil_preferences", new=AsyncMock(side_effect=RuntimeError("down"))
        ),
        patch.object(
            gate,
            "judge_intent",
            new=AsyncMock(return_value=IntentDecision(outcome="ask", reason="x")),
        ) as judge,
    ):
        await gate._judge(request, context, gate.unpack_tool_call(request), None, "s")
    assert judge.await_args.kwargs["never_auto_tools"] == frozenset()
