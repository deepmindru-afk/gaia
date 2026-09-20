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

import httpx

from datetime import UTC, datetime

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
from app.services.hil.prompts import JEV_QUESTION, JEV_QUESTIONS_VERSION
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


async def ask_jev(
    *,
    user_messages: list[str],
    call: JudgedCall,
    prior_calls: list[PriorCall],
    history: AutoHistory,
) -> tuple[str, float, dict[str, float], int, int]:
    """One Decisions call: (choice, confidence, probabilities, in, out) tokens.

    Raises on transport failure or a malformed answer — the caller falls back
    to the LLM judge, never to an allow.
    """
    key = settings.OPENROUTER_API_KEY
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY unset; cannot reach the JEV judge")
    state = {
        "user_messages": user_messages,
        "pending_action": {
            "tool": call.tool_name,
            "description": call.description,
            "summary": call.summary,
            "args": call.args,
        },
        "prior_actions": [
            {"tool": prior.name, "args": prior.args} for prior in prior_calls
        ],
        "recent_history": history_line(history, call.tool_name),
        "now": datetime.now(UTC).isoformat(),
    }
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
    ) -> IntentDecision:
        try:
            choice, confidence, _probs, _in, _out = await ask_jev(
                user_messages=user_messages,
                call=call,
                prior_calls=prior_calls,
                history=history,
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
            )
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
            accept_line=HIL_JEV_ACCEPT_LINE,
            reject_floor=HIL_JEV_REJECT_FLOOR,
        )
        if outcome == "reject":
            return IntentDecision(
                "reject",
                "Auto mode held this off: your messages argue against this "
                f"{call.tool_name} call.",
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
                call.args, "\n".join(user_messages), prior_calls
            )
            if missing:
                return IntentDecision(
                    "ask",
                    f"the target ({', '.join(missing[:3])}) doesn't trace to "
                    "your words.",
                )
            return IntentDecision(
                "accept",
                f"Auto mode matched this to your request ({choice} {confidence:.2f}).",
            )
        return IntentDecision(
            "ask",
            f"this {call.tool_name} call may not match your request "
            f"({choice} {confidence:.2f}).",
        )
