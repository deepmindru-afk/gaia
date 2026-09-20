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
