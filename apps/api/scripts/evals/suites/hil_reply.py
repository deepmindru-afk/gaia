"""hil-reply suite — calibrate the chat-reply classifier for pending approvals.

A bot user answers an approval by typing, so this classifier is the whole
approval UI on WhatsApp/Telegram/Slack/Discord. Drives the classifier backends
directly (no chat stream, no dev user): JEV (Decisions API) or the LLM
fallback, both graded through the production mapping in conversational.py.
JEV runs journal every per-action verdict, so the approve line is tuned by a
free offline sweep over the journal, not another model-spending run.

Run:  uv run --group backend python -m scripts.evals run --suite hil-reply
  HIL_REPLY_EVAL_BACKEND=jev|llm (default jev)
Sweep:  uv run --group backend python -m scripts.evals.sweep_hil_reply [run-id]

Promotion bar: zero dangerous approves (an approve on a case not labelled
approve runs an action the user did not accept), stable across 3 runs.
"""

from __future__ import annotations

from collections.abc import Awaitable, Mapping
from enum import StrEnum
import os
from pathlib import Path
import time
from typing import Any, TypedDict

import httpx

from app.config.settings import get_settings
from app.constants.hil import HIL_JEV_MODEL_NAME, HIL_JEV_REPLY_APPROVE_LINE, ReplyChoice
from app.models.hil_models import BatchDecisionResult, DecisionResult
from app.models.jev_models import JevReplyVerdict
from app.models.message_models import MessageDict
from app.services.hil.bridge import build_action_detail
from app.services.hil.conversational import (
    batch_from_jev_reply,
    classify_batch_with_llm,
    classify_with_llm,
    decision_from_jev_reply,
    no_arg_edit,
)
from app.services.hil.jev_reply import ask_jev_reply, settle_reply
from app.services.hil.prompts import JEV_REPLY_QUESTIONS_VERSION
from scripts.evals.core.app_boot import ensure_app_registered
from scripts.evals.core.cases import load_case_files
from scripts.evals.core.cost import EvalCostTracker, estimate_tokens
from scripts.evals.core.gates import score_gates
from scripts.evals.core.journal import RunJournal
from scripts.evals.core.providers import EvalConfig, ProviderConfig
from scripts.evals.core.runner import Suite, register_suite
from scripts.evals.core.types import Case, CaseRun, ProviderError

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "hil-reply"

EVAL_USER_ID = "hil-reply-eval"


class ReplyBackend(StrEnum):
    """Which classifier a run drove — HIL_REPLY_EVAL_BACKEND, journaled on every row."""

    JEV = "jev"
    LLM = "llm"


class JevReplyJournalFields(TypedDict):
    """The per-action JEV verdicts a row was graded on — what the offline sweep regrades."""

    choices: list[str]
    confidences: list[float]
    probabilities: list[dict[str, float]]


EXTRA = {
    "reply_outcome": lambda case, run: (
        1.0 if (run.end_state or {}).get("outcome") == case.expected.get("outcome") else 0.0
    ),
}


def _action_details(setup: Mapping[str, Any]) -> list[str]:
    """Each pending action rendered exactly as prod renders it for the classifier."""
    return [
        build_action_detail(str(action["summary"]), dict(action.get("args") or {}))
        for action in setup["actions"]
    ]


def _history(setup: Mapping[str, Any]) -> list[MessageDict] | None:
    turns = setup.get("history") or []
    return [MessageDict(role=str(t["role"]), content=str(t["content"])) for t in turns] or None


def single_outcome(result: DecisionResult | None) -> str:
    """The graded outcome of a single-approval result, after prod's no-edit rule."""
    if result is None:
        return ReplyChoice.LEAVE.value
    if result.action == "unrelated":
        return ReplyChoice.UNRELATED.value
    action, _ = no_arg_edit(result.action, result.feedback)
    return action


def batch_outcome(result: BatchDecisionResult, count: int) -> str:
    """The graded outcome of a batch result: unrelated, or one choice per action.

    Mirrors conversational._apply_decisions: out-of-range indexes and actions
    with no decision stay pending.
    """
    if result.unrelated:
        return ReplyChoice.UNRELATED.value
    outcomes = [ReplyChoice.LEAVE.value] * count
    for decision in result.decisions:
        if decision.action == "leave" or not 1 <= decision.index <= count:
            continue
        action, _ = no_arg_edit(decision.action, decision.feedback)
        outcomes[decision.index - 1] = action
    return ",".join(outcomes)


def jev_outcome(verdicts: list[JevReplyVerdict], message: str, approve_line: float) -> str:
    """The graded outcome of a JEV verdict list through prod's settle + mapping."""
    settled = settle_reply(verdicts, approve_line=approve_line)
    if len(verdicts) == 1:
        return single_outcome(decision_from_jev_reply(settled[0], message))
    return batch_outcome(batch_from_jev_reply(settled, message), len(verdicts))


def is_dangerous(outcome: str, expected: str) -> bool:
    """Whether any action approved that the label did not approve — it would run."""
    got, want = outcome.split(","), expected.split(",")
    if len(got) != len(want):
        return ReplyChoice.APPROVE.value in got
    return any(g == ReplyChoice.APPROVE and w != ReplyChoice.APPROVE for g, w in zip(got, want))


class ReplyTransport:
    """One case: build classifier input from the YAML, run one backend, journal all."""

    def __init__(self) -> None:
        self._backend = os.environ.get("HIL_REPLY_EVAL_BACKEND", ReplyBackend.JEV)

    async def run(
        self,
        case: Case,
        cfg: EvalConfig,
        tracker: EvalCostTracker,
        provider: ProviderConfig,
    ) -> CaseRun:
        del cfg, tracker
        start = time.monotonic()
        setup = case.setup
        message = str(setup["reply"])
        details = _action_details(setup)
        history = _history(setup)
        verdict: JevReplyJournalFields | None = None
        if self._backend == ReplyBackend.LLM:
            outcome, usage = await self._run_llm(message, details, history)
            model = "llm-reply-classifier"
        elif self._backend == ReplyBackend.JEV:
            outcome, usage, verdict = await self._run_jev(message, details, history, provider)
            model = HIL_JEV_MODEL_NAME
        else:
            raise ProviderError(provider.name, f"unknown reply backend {self._backend!r}")
        duration_s = time.monotonic() - start
        want = str(case.expected.get("outcome"))
        mark = "ok" if outcome == want else ("DANGER" if is_dangerous(outcome, want) else "MISS")
        print(f"[{mark}] {case.id}: want={want} got={outcome} ({duration_s:.1f}s) {verdict or ''}")
        end_state: dict[str, Any] = {
            "outcome": outcome,
            "backend": self._backend,
            "questions_version": JEV_REPLY_QUESTIONS_VERSION if verdict is not None else "llm",
            "dangerous": is_dangerous(outcome, want),
            "reply": message,
            **(verdict or {}),
        }
        return CaseRun(
            case_id=case.id,
            provider=provider.name,
            model=model,
            messages=[
                {"role": "user", "content": message},
                {"role": "assistant", "content": outcome},
            ],
            end_state=end_state,
            text=outcome,
            tokens_in=usage[0],
            tokens_out=usage[1],
            duration_s=duration_s,
        )

    async def _run_llm(
        self, message: str, details: list[str], history: list[MessageDict] | None
    ) -> tuple[str, tuple[int, int]]:
        await ensure_app_registered()
        if len(details) == 1:
            outcome = single_outcome(
                await classify_with_llm(message, details, history, user_id=EVAL_USER_ID)
            )
        else:
            outcome = batch_outcome(
                await classify_batch_with_llm(message, details, history, user_id=EVAL_USER_ID),
                len(details),
            )
        # App-side metering owns LLM cost; journal an openly-labelled estimate, not zeros.
        return outcome, (estimate_tokens(message + "".join(details)), estimate_tokens(outcome))

    async def _run_jev(
        self,
        message: str,
        details: list[str],
        history: list[MessageDict] | None,
        provider: ProviderConfig,
    ) -> tuple[str, tuple[int, int], JevReplyJournalFields]:
        if not get_settings().OPENROUTER_API_KEY:
            raise ProviderError(provider.name, "OPENROUTER_API_KEY unset")
        try:
            verdicts, tokens_in, tokens_out = await ask_jev_reply(message, details, history)
        except httpx.HTTPError as e:
            raise ProviderError(provider.name, f"decisions API failed: {e}") from e
        fields = JevReplyJournalFields(
            choices=[v.choice.value for v in verdicts],
            confidences=[v.confidence for v in verdicts],
            probabilities=[dict(v.probabilities) for v in verdicts],
        )
        return (
            jev_outcome(verdicts, message, HIL_JEV_REPLY_APPROVE_LINE),
            (tokens_in, tokens_out),
            fields,
        )


def _journaled_verdicts(end_state: Mapping[str, Any]) -> list[JevReplyVerdict]:
    return [
        JevReplyVerdict(choice=ReplyChoice(choice), confidence=float(conf), probabilities=probs)
        for choice, conf, probs in zip(
            end_state["choices"], end_state["confidences"], end_state["probabilities"]
        )
    ]


def sweep_journal(run_dir: Path) -> str:
    """Grid-search the approve line over one journaled JEV run. Free — no model calls."""
    rows = list(RunJournal(run_dir.parent, run_dir.name).latest_per_case().values())
    unjudged = sum(1 for r in rows if not r.get("end_state"))
    rows = [r for r in rows if r.get("end_state")]
    foreign = sorted({str(r["end_state"].get("backend")) for r in rows} - {ReplyBackend.JEV})
    if foreign:
        raise SystemExit(
            f"{run_dir} holds rows from backend(s) {', '.join(foreign)}: the sweep regrades "
            "journaled JEV verdicts only (run with HIL_REPLY_EVAL_BACKEND=jev)"
        )
    out = [
        f"sweep over {len(rows)} journaled cases (approve line):",
        f"unjudged rows left out (errored, no verdict): {unjudged}",
    ]
    for line in [round(v / 100, 2) for v in range(50, 100, 5)]:
        score = dangerous = 0
        misses: list[str] = []
        for row in rows:
            es = row["end_state"]
            want = str((row.get("expected") or {}).get("outcome"))
            got = jev_outcome(_journaled_verdicts(es), str(es["reply"]), line)
            score += got == want
            dangerous += is_dangerous(got, want)
            if got != want:
                misses.append(str(row.get("case_id")))
        out.append(
            f"  approve>={line:.2f}: {score}/{len(rows)} dangerous={dangerous} "
            f"misses={','.join(misses[:12])}"
        )
    return "\n".join(out)


@register_suite("hil-reply")
class HilReplySuite(Suite):
    name = "hil-reply"
    project = "gaia-hil"
    label = "HIL chat-reply classifier calibration"

    def __init__(self, cfg: EvalConfig) -> None:
        del cfg
        self._transport = ReplyTransport()

    def load_cases(self, cfg: EvalConfig) -> list[Case]:
        del cfg
        return load_case_files(DATA_DIR, self.name, EXTRA)

    def transport(
        self,
        case: Case,
        cfg: EvalConfig,
        tracker: EvalCostTracker,
        provider: ProviderConfig,
    ) -> Awaitable[CaseRun]:
        return self._transport.run(case, cfg, tracker, provider)

    def score(self, case: Case, run: CaseRun) -> dict[str, float]:
        return score_gates(case, run, EXTRA)

    def finalize_scorers(self, cfg: EvalConfig) -> list[object]:
        del cfg
        return []  # exact-match gate only; no rubric judge, no second model bill
