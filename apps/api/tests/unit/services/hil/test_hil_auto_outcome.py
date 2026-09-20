"""Auto mode v2 outcomes (app/services/hil/intent.py).

The judge returns accept / reject / ask — never a bare boolean — so the gate
can refuse with a reason (no card) distinctly from asking (card). Tests here
pin the outcome vocabulary; the adversarial LLM-judge tests live in
test_hil_intent.py.
"""

from app.services.hil.intent import IntentDecision


async def test_reject_outcome_is_not_aligned_and_carries_reason() -> None:
    d = IntentDecision(outcome="reject", reason="You denied this twice already.")
    assert d.aligned is False
    assert d.reason == "You denied this twice already."


async def test_accept_outcome_stays_aligned() -> None:
    assert IntentDecision(outcome="accept", reason="r").aligned is True
    assert IntentDecision(outcome="ask", reason="r").aligned is False
