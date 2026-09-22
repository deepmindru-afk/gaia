"""JEV choice judge for auto mode (app/services/hil/intent.py's other half).

One Decisions API call returns authorized / forbidden / unclear with a
calibrated confidence; code maps it to accept / reject / ask and applies the
checks a classifier must never grade itself on (target grounding, history
bias). Transport failure degrades to the injected LLM fallback — the gate's
behavior on a JEV outage is today's behavior, not an open gate.

Question text lives in prompts.py (JEV_QUESTION); thresholds in
constants/hil.py. Both are tuned through the hil-judge calibration suite,
never by hand here.
"""

from collections.abc import Mapping
from datetime import UTC, datetime
import re

import httpx

from app.config.settings import settings
from app.constants.hil import (
    HIL_JEV_ACCEPT_LINE,
    HIL_JEV_MODEL_NAME,
    HIL_JEV_REJECT_FLOOR,
    HIL_JEV_TIMEOUT_SECONDS,
    HIL_JEV_URL,
)
from app.constants.log_tags import LogTag
from app.services.hil.intent import (
    AutoHistory,
    AutoOutcome,
    IntentDecision,
    IntentJudge,
    JudgedCall,
    _history_blocks,
    _LLMIntentJudge,
    history_line,
    ungrounded_targets,
)
from app.services.hil.prompts import (
    JEV_FORBID_QUESTION,
    JEV_QUESTION,
    JEV_QUESTIONS_VERSION,
)
from app.services.hil.utils import PriorCall
from shared.py.wide_events import log


def map_jev_choice(
    choice: str, confidence: float, *, accept_line: float, reject_floor: float
) -> AutoOutcome:
    """JEV verdict to gate outcome. Pure — the offline sweep reuses this."""
    if choice == "authorized" and confidence >= accept_line:
        return "accept"
    if choice == "forbidden" and confidence >= reject_floor:
        return "reject"
    return "ask"


#: Earlier-turn language that trips the forbid double-check (accept path only).
#: Crisp prohibitions, not moods: "don't send anything yet" is a temporary
#: boundary (asks on its own merits), while these claim the act itself is off
#: limits. Bare "no" is deliberately absent — "no, send it" lifts, not forbids.
_FORBID_PATTERNS = (
    r"\bnever\b",
    r"\bdon'?t\b",
    r"\bdo not\b",
    r"\bdoesn'?t\b",
    r"\bcan'?t\b",
    r"\bcannot\b",
    r"\bwon'?t\b",
    r"\bcancel that\b",
    r"\bforbid",
    r"\bprohibit",
    r"\bkeep\b.{0,24}\bprivate\b",
    r"\bkeep (everything|all)\b",
    r"\bhold (everything|all)\b",
    r"\bdon'?t touch\b",
)

#: Later-turn language that lifts a forbid. Searched only AFTER the forbid turn.
_LIFT_PATTERNS = (
    r"\bactually\b",
    r"\bgo ahead\b",
    r"\bnever mind\b",
    r"\bnevermind\b",
    r"\byes do it\b",
    r"\bdo it now\b",
    r"\boverride\b",
    r"\bignore (that|my last|the last)\b",
    r"\bchange of plans\b",
    r"\bon second thought\b",
)


def forbid_tripwire(user_messages: list[str]) -> bool:
    """Whether an earlier turn forbids with no later lift — check needed.

    A tripwire, not a verdict: it only ever downgrades an accept into a
    focused forbid-check (or an ask), never into a run. Matching is
    deterministic keyword/regex on the user's own words; the JUDGMENT of
    whether the forbid covers this call belongs to the focused JEV question.
    """
    for index, turn in enumerate(user_messages):
        lowered = turn.lower()
        if any(re.search(pattern, lowered) for pattern in _FORBID_PATTERNS):
            later = " ".join(user_messages[index + 1 :]).lower()
            if not any(re.search(pattern, later) for pattern in _LIFT_PATTERNS):
                return True
    return False


def decisive_forbidden(
    choice: str, probabilities: Mapping[str, float], *, margin: float = 0.40
) -> bool:
    """Whether forbidden is the runaway winner, whatever its absolute number.

    JEV spreads mass then under-reports confidence on forbids (0.43 top choice
    at 0.22 confidence is routine). A 0.40 lead means the model compared the
    options and picked one — that comparison IS the signal. One-sided by
    design: over-rejecting is silent (no card), over-accepting runs the
    action. The accept side keeps its absolute line plus grounding; only
    refusal gets the margin rule. Calibrated against the hil-judge journal:
    0.30 caught temporary-boundary asks (bg-boundary, r-hold); 0.40 keeps
    those asking while still catching true forbids.
    """
    if choice != "forbidden":
        return False
    runner_up = max(
        (prob for label, prob in probabilities.items() if label != "forbidden"),
        default=0.0,
    )
    return probabilities.get("forbidden", 0.0) - runner_up >= margin


async def ask_jev_forbid(
    *,
    user_messages: list[str],
    call: JudgedCall,
) -> tuple[str, float, int, int]:
    """Focused forbid check: does any earlier turn forbid THIS action?

    Second, decomposed opinion — runs only when the main verdict would accept
    but the tripwire found forbid language. Returns (choice, confidence, in,
    out) tokens; transport failure raises (caller degrades to ask, never to
    a run).
    """
    key = settings.OPENROUTER_API_KEY
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY unset; cannot reach the JEV judge")
    state = {
        "earlier_turns": user_messages[:-1],
        "latest_turns": user_messages[-1:],
        "pending_action": {"tool": call.tool_name, "args": call.args},
    }
    async with httpx.AsyncClient(timeout=HIL_JEV_TIMEOUT_SECONDS) as client:
        resp = await client.post(
            HIL_JEV_URL,
            json={
                "model": HIL_JEV_MODEL_NAME,
                "state": state,
                "questions": {"forbid": JEV_FORBID_QUESTION},
            },
            headers={"Authorization": f"Bearer {key}"},
        )
        resp.raise_for_status()
        body = resp.json()
    answer = body["answers"]["forbid"]
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise ValueError(f"non-choice JEV forbid answer: {answer!r}"[:300])
    choice = str(answer["choice"])
    if choice not in ("forbidden", "permitted"):
        raise ValueError(f"unknown JEV forbid choice: {choice!r}")
    usage = body.get("usage") or {}
    return (
        choice,
        float(answer.get("confidence") or 0.0),
        int(usage.get("input_tokens") or 0),
        int(usage.get("output_tokens") or 0),
    )


def needs_forbid_check(mapped_outcome: AutoOutcome, user_messages: list[str]) -> bool:
    """Whether an accept needs the focused forbid double-check.

    Pure predicate shared by prod and the eval transport: the main verdict
    says accept, but earlier turns carry forbid language with no later lift.
    """
    return mapped_outcome == "accept" and forbid_tripwire(user_messages)


async def ask_jev(
    *,
    user_messages: list[str],
    call: JudgedCall,
    prior_calls: list[PriorCall],
    history: AutoHistory,
    assistant_turns: list[str] | None = None,
) -> tuple[str, float, dict[str, float], int, int]:
    """One Decisions call: (choice, confidence, probabilities, in, out) tokens.

    Raises on transport failure or a malformed answer — the caller falls back
    to the LLM judge, never to an allow.
    """
    key = settings.OPENROUTER_API_KEY
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY unset; cannot reach the JEV judge")
    pending_action: dict[str, object] = {
        "tool": call.tool_name,
        "description": call.description,
        "summary": call.summary,
        "args": call.args,
    }
    # Enrichment rides only when it exists: empty evidence reads as missing
    # evidence and costs confidence, so an old-shape call posts the old-shape
    # state byte for byte.
    if call.tool_schema:
        pending_action["tool_schema"] = call.tool_schema
    prior_actions = []
    for prior in prior_calls:
        item: dict[str, object] = {"tool": prior.name, "args": prior.args}
        if prior.output.strip():
            item["output"] = prior.output
        prior_actions.append(item)
    state = {
        "user_messages": user_messages,
        "pending_action": pending_action,
        "prior_actions": prior_actions,
        "recent_history": history_line(history, call.tool_name),
        "now": datetime.now(UTC).isoformat(),
    }
    if assistant_turns:
        state["assistant_turns"] = list(assistant_turns)
    async with httpx.AsyncClient(timeout=HIL_JEV_TIMEOUT_SECONDS) as client:
        resp = await client.post(
            HIL_JEV_URL,
            json={
                "model": HIL_JEV_MODEL_NAME,
                "state": state,
                "questions": {"decision": JEV_QUESTION},
            },
            headers={"Authorization": f"Bearer {key}"},
        )
        resp.raise_for_status()
        body = resp.json()
    answer = body["answers"]["decision"]
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise ValueError(f"non-choice JEV answer: {answer!r}"[:300])
    choice = str(answer["choice"])
    if choice not in ("authorized", "forbidden", "unclear"):
        raise ValueError(f"unknown JEV choice: {choice!r}")
    usage = body.get("usage") or {}
    return (
        choice,
        float(answer.get("confidence") or 0.0),
        {str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()},
        int(usage.get("input_tokens") or 0),
        int(usage.get("output_tokens") or 0),
    )


class JevIntentJudge:
    """Auto mode's classifier: JEV decides, code applies the vetoes.

    ``fallback`` runs when the Decisions call fails or misbehaves — same
    verdict shape, today's LLM behavior. ``unclear`` never escalates: a
    confused classifier asks the user (the card carries why), it does not buy
    a second opinion that has its own dangerous accepts.
    """

    def __init__(self, fallback: IntentJudge | None = None) -> None:
        self._fallback = fallback or _LLMIntentJudge()

    async def decide(
        self,
        *,
        user_id: str,
        user_messages: list[str],
        call: JudgedCall,
        prior_calls: list[PriorCall],
        history: AutoHistory,
        assistant_turns: list[str] | None = None,
    ) -> IntentDecision:
        try:
            choice, confidence, probs, _in, _out = await ask_jev(
                user_messages=user_messages,
                call=call,
                prior_calls=prior_calls,
                history=history,
                assistant_turns=assistant_turns,
            )
        except Exception as e:
            log.warning(
                f"{LogTag.HIL} JEV judge failed; falling back to the LLM judge",
                tool_name=call.tool_name,
                error=str(e),
                error_type=type(e).__name__,
            )
            return await self._fallback.decide(
                user_id=user_id,
                user_messages=user_messages,
                call=call,
                prior_calls=prior_calls,
                history=history,
                assistant_turns=assistant_turns,
            )
        return decide_from_verdict(
            choice=choice,
            confidence=confidence,
            probabilities=probs,
            user_messages=user_messages,
            call=call,
            prior_calls=prior_calls,
            history=history,
            forbid=await self._forbid_verdict(choice, confidence, probs, user_messages, call),
        )

    async def _forbid_verdict(
        self,
        choice: str,
        confidence: float,
        probs: dict[str, float],
        user_messages: list[str],
        call: JudgedCall,
    ) -> str | None:
        """Focused forbid verdict, or None when no double-check is warranted.

        Runs only on the accept path with tripped forbid language. A failed
        focused check degrades to ask (the tripwire already downgraded the
        accept) — never back to a run.
        """
        outcome = map_jev_choice(
            choice,
            confidence,
            accept_line=HIL_JEV_ACCEPT_LINE,
            reject_floor=HIL_JEV_REJECT_FLOOR,
        )
        if outcome == "ask" and decisive_forbidden(choice, probs):
            outcome = "reject"
        if not needs_forbid_check(outcome, user_messages):
            return None
        try:
            forbid_choice, forbid_conf, _in, _out = await ask_jev_forbid(
                user_messages=user_messages, call=call
            )
        except Exception as e:
            log.warning(
                f"{LogTag.HIL} forbid double-check failed; asking",
                tool_name=call.tool_name,
                error=str(e),
                error_type=type(e).__name__,
            )
            return "unclear-forbid"
        log.info(
            f"{LogTag.HIL} forbid double-check : {forbid_choice}",
            tool_name=call.tool_name,
            hil={"choice": forbid_choice, "confidence": round(forbid_conf, 3)},
        )
        if forbid_choice == "forbidden" and forbid_conf >= HIL_JEV_REJECT_FLOOR:
            return "forbidden"
        return "permitted"


def decide_from_verdict(
    *,
    choice: str,
    confidence: float,
    probabilities: Mapping[str, float] | None,
    user_messages: list[str],
    call: JudgedCall,
    prior_calls: list[PriorCall],
    history: AutoHistory,
    accept_line: float = HIL_JEV_ACCEPT_LINE,
    reject_floor: float = HIL_JEV_REJECT_FLOOR,
    forbid: str | None = None,
) -> IntentDecision:
    """Map one JEV verdict through the code vetoes. Pure — shared by prod and eval.

    ``forbid`` carries the focused double-check ("forbidden" / "permitted" /
    "unclear-forbid" / None when not run): a forbidden double-check rejects;
    anything else leaves the mapped outcome (plus grounding/history) standing.
    """
    log.info(
        f"{LogTag.HIL} JEV judge : {choice}",
        tool_name=call.tool_name,
        hil={
            "choice": choice,
            "confidence": round(confidence, 3),
            "version": JEV_QUESTIONS_VERSION,
        },
    )
    outcome = map_jev_choice(
        choice,
        confidence,
        accept_line=accept_line,
        reject_floor=reject_floor,
    )
    if outcome == "ask" and decisive_forbidden(choice, probabilities or {}):
        outcome = "reject"
    if outcome == "accept" and forbid == "forbidden":
        return IntentDecision(
            "reject",
            "Auto mode held this off: an earlier message forbids this "
            f"{call.tool_name} call and nothing since lifted it.",
        )
    if outcome == "accept" and forbid is not None:
        # The tripwire found forbid language but the double-check would not
        # refuse: contradictory turns still need the human, so the accept
        # floors to a card rather than running.
        return IntentDecision(
            "ask",
            "Auto mode wasn't sure: an earlier message argues against this "
            f"{call.tool_name} call — your call.",
        )
    if outcome == "reject":
        return IntentDecision(
            "reject",
            f"Auto mode held this off: your messages argue against this {call.tool_name} call.",
        )
    if outcome == "accept":
        if _history_blocks(history):
            return IntentDecision(
                "ask",
                f"You denied {history.denied_recent} recent "
                f"{call.tool_name} call(s), so this one needs your "
                "go-ahead even though it looks authorized.",
            )
        missing = ungrounded_targets(
            call.args,
            "\n".join(user_messages),
            prior_calls,
            frozenset(history.known_targets),
        )
        if missing:
            return IntentDecision(
                "ask",
                f"the target ({', '.join(missing[:3])}) doesn't trace to your words.",
            )
        return IntentDecision(
            "accept",
            f"Auto mode matched this to your request ({choice} {confidence:.2f}).",
        )
    return IntentDecision(
        "ask",
        f"this {call.tool_name} call may not match your request ({choice} {confidence:.2f}).",
    )
