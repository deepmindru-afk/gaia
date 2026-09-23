"""JEV reply classifier: what a bot user's chat reply means for each pending approval.

One Decisions call asks one question per numbered pending action. Code, not the
model, owns the safety rules: an approve under the approve line leaves the
action pending, and a reply is unrelated only when every action reads it so.
Question text lives in prompts.py, the line in constants/hil.py — both tuned
through the hil-reply calibration suite, never by hand here.
"""

from app.constants.hil import (
    HIL_CLASSIFIER_MAX_ARG_CHARS,
    HIL_JEV_REPLY_APPROVE_LINE,
    HIL_JEV_REPLY_QUESTION_PREFIX,
    ReplyChoice,
)
from app.constants.log_tags import LogTag
from app.models.jev_models import JevReplyVerdict
from app.models.message_models import MessageDict
from app.services.hil.jev_client import post_decisions
from app.services.hil.prompts import (
    JEV_REPLY_CRITERIA,
    JEV_REPLY_INSTRUCTIONS,
    JEV_REPLY_QUESTIONS_VERSION,
)
from app.utils.general_utils import clip_text
from shared.py.wide_events import log


def _question(number: int) -> dict[str, object]:
    return {
        "type": "choice",
        "instructions": JEV_REPLY_INSTRUCTIONS.format(number=number),
        "criteria": {
            choice.value: text.format(number=number) for choice, text in JEV_REPLY_CRITERIA.items()
        },
    }


def reply_state(
    message: str, action_details: list[str], history: list[MessageDict] | None
) -> dict[str, object]:
    """Build the Decisions state: the reply, the numbered actions, and any recent turns."""
    state: dict[str, object] = {
        "reply": message,
        "pending_actions": [
            {"number": number, "action": detail}
            for number, detail in enumerate(action_details, start=1)
        ],
    }
    # Absent rather than empty: an empty list reads as missing evidence.
    if history:
        state["recent_conversation"] = [
            {
                "role": turn.get("role", ""),
                "content": clip_text(turn.get("content") or "", HIL_CLASSIFIER_MAX_ARG_CHARS),
            }
            for turn in history
        ]
    return state


async def ask_jev_reply(
    message: str, action_details: list[str], history: list[MessageDict] | None
) -> tuple[list[JevReplyVerdict], int, int]:
    """One Decisions call: a verdict per pending action, plus (in, out) tokens.

    Raises on transport failure or a malformed answer; the caller falls back.
    """
    names = [f"{HIL_JEV_REPLY_QUESTION_PREFIX}{n}" for n in range(1, len(action_details) + 1)]
    body = await post_decisions(
        reply_state(message, action_details, history),
        {name: _question(number) for number, name in enumerate(names, start=1)},
    )
    verdicts = []
    for name in names:
        answer = body.choice(name, tuple(ReplyChoice), label="reply ")
        verdicts.append(
            JevReplyVerdict(
                choice=ReplyChoice(answer.choice),
                confidence=float(answer.confidence or 0.0),
                probabilities=dict(answer.probabilities or {}),
            )
        )
    log.info(
        f"{LogTag.HIL} JEV reply classifier",
        hil={
            "choices": [v.choice.value for v in verdicts],
            "confidences": [round(v.confidence, 3) for v in verdicts],
            "version": JEV_REPLY_QUESTIONS_VERSION,
        },
    )
    return verdicts, *body.tokens()


def settle_reply(
    verdicts: list[JevReplyVerdict], *, approve_line: float = HIL_JEV_REPLY_APPROVE_LINE
) -> list[ReplyChoice]:
    """Apply the code rules, one choice per action. Pure, shared by prod and eval.

    Unrelated survives only when unanimous; an approve under the line is a leave.
    """
    if all(v.choice is ReplyChoice.UNRELATED for v in verdicts):
        return [ReplyChoice.UNRELATED] * len(verdicts)
    settled = []
    for verdict in verdicts:
        if verdict.choice is ReplyChoice.UNRELATED or (
            verdict.choice is ReplyChoice.APPROVE and verdict.confidence < approve_line
        ):
            settled.append(ReplyChoice.LEAVE)
        else:
            settled.append(verdict.choice)
    return settled
