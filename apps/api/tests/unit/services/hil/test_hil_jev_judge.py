"""JEV choice judge (app/services/hil/jev_judge.py).

The Decisions API is mocked — the network boundary. map_jev_choice,
ungrounded_targets, and the vetoes around the verdict are the production
code under test and run for real. Every test assumes the classifier at its
most confident and asks whether the code around it still refuses.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

from app.services.hil.intent import AutoHistory, JudgedCall
from app.services.hil.jev_judge import (
    JevIntentJudge,
    ask_jev,
    decide_from_verdict,
    decisive_forbidden,
    forbid_tripwire,
    map_jev_choice,
    needs_forbid_check,
    ungrounded_targets,
)
from app.services.hil.utils import PriorCall

MODULE = "app.services.hil.jev_judge"


def _call(**overrides: Any) -> JudgedCall:
    return JudgedCall(
        **{
            "tool_name": "send_email",
            "description": "Send an email.",
            "args": {"to": "bob@example.com", "subject": "deck"},
            "summary": "Send email — to: bob@example.com",
            **overrides,
        }
    )


def _answer(
    choice: str, confidence: float, probs: dict[str, float] | None = None
) -> dict[str, Any]:
    return {
        "answers": {
            "decision": {
                "type": "choice",
                "choice": choice,
                "confidence": confidence,
                "probabilities": probs
                if probs is not None
                else {choice: confidence},
            }
        },
        "usage": {"input_tokens": 300, "output_tokens": 20},
    }


def _client(answer: dict[str, Any] | Exception):
    """Fake httpx.AsyncClient serving one Decisions answer (or raising)."""
    from unittest.mock import MagicMock

    response = MagicMock()
    response.json.return_value = answer
    client = AsyncMock()
    client.__aenter__.return_value = client
    if isinstance(answer, Exception):
        client.post.side_effect = answer
    else:
        client.post.return_value = response
    return client


async def _decide(
    answer: dict[str, Any] | Exception,
    call: JudgedCall | None = None,
    history: AutoHistory | None = None,
    fallback: Any = None,
):
    judge = JevIntentJudge(
        fallback=fallback
        if fallback is not None
        else AsyncMock(
            **{"decide.return_value": None},
        ),
    )
    with patch(f"{MODULE}.httpx.AsyncClient", return_value=_client(answer)):
        with patch(f"{MODULE}.settings") as settings:
            settings.OPENROUTER_API_KEY = "or-key"  # pragma: allowlist secret
            return await judge.decide(
                user_id="u",
                user_messages=["draft an email to bob@example.com about the deck"],
                call=call or _call(),
                prior_calls=[],
                history=history or AutoHistory(),
            )


class TestMapping:
    async def test_confident_authorized_is_accept(self) -> None:
        assert (
            map_jev_choice("authorized", 0.9, accept_line=0.5, reject_floor=0.5)
            == "accept"
        )

    async def test_below_line_authorized_is_ask_not_accept(self) -> None:
        # Confidence lines are load-bearing: a 0.49 authorized must not run.
        assert (
            map_jev_choice("authorized", 0.49, accept_line=0.5, reject_floor=0.5)
            == "ask"
        )

    async def test_forbidden_needs_its_own_floor(self) -> None:
        assert (
            map_jev_choice("forbidden", 0.4, accept_line=0.5, reject_floor=0.5) == "ask"
        )
        assert (
            map_jev_choice("forbidden", 0.6, accept_line=0.5, reject_floor=0.5)
            == "reject"
        )

    async def test_unclear_is_always_ask(self) -> None:
        assert map_jev_choice("unclear", 1.0, accept_line=0.5, reject_floor=0.5) == "ask"

    async def test_unknown_choice_is_ask(self) -> None:
        assert map_jev_choice("maybe", 1.0, accept_line=0.5, reject_floor=0.5) == "ask"


class TestDecisiveForbidden:
    """A runaway forbidden wins whatever its absolute number.

    One-sided by design: over-rejecting is silent (no card), over-accepting
    runs the action — so only refusal gets the margin rule.
    """

    async def test_runaway_forbidden_rejects_below_the_floor(self) -> None:
        assert (
            decisive_forbidden(
                "forbidden", {"forbidden": 0.48, "unclear": 0.06, "authorized": 0.0}
            )
            is True
        )

    async def test_close_race_does_not_reject(self) -> None:
        assert (
            decisive_forbidden(
                "forbidden", {"forbidden": 0.48, "unclear": 0.40, "authorized": 0.12}
            )
            is False
        )

    async def test_just_under_the_margin_does_not_reject(self) -> None:
        # 0.37 margin: the journal band where temporary-boundary asks
        # (bg-boundary, r-hold) sit — the margin must clear them.
        assert (
            decisive_forbidden(
                "forbidden", {"forbidden": 0.43, "unclear": 0.06, "authorized": 0.0}
            )
            is False
        )

    async def test_margin_never_accepts(self) -> None:
        # The accept side keeps its absolute line plus grounding, always.
        assert (
            decisive_forbidden(
                "authorized", {"authorized": 0.9, "unclear": 0.05, "forbidden": 0.0}
            )
            is False
        )

    async def test_empty_probabilities_do_not_reject(self) -> None:
        assert decisive_forbidden("forbidden", {}) is False

    async def test_decisive_forbidden_rejects_through_decide(self) -> None:
        d = await _decide(
            _answer(
                "forbidden",
                0.22,
                {"forbidden": 0.50, "unclear": 0.06, "authorized": 0.0},
            )
        )
        assert d.outcome == "reject"


class TestForbidTripwire:
    """The tripwire only ever downgrades an accept into a double-check."""

    async def test_forbid_without_lift_trips(self) -> None:
        assert forbid_tripwire(["keep everything for now", "delete thread t5"]) is True
        assert forbid_tripwire(["never pay anyone", "pay $10 now"]) is True

    async def test_lift_language_clears_the_tripwire(self) -> None:
        assert (
            forbid_tripwire(["don't send anything yet", "actually, go ahead and send"])
            is False
        )

    async def test_clean_history_never_trips(self) -> None:
        assert forbid_tripwire(["draft an email to bob", "looks good, send it"]) is False

    async def test_check_runs_only_on_would_accept(self) -> None:
        assert needs_forbid_check("accept", ["never email alice", "email alice"]) is True
        assert needs_forbid_check("ask", ["never email alice", "email alice"]) is False
        assert needs_forbid_check("accept", ["send the deck"]) is False
        assert needs_forbid_check("reject", ["never email alice"]) is False

    async def test_forbidden_double_check_rejects(self) -> None:
        d = decide_from_verdict(
            choice="authorized",
            confidence=0.9,
            probabilities={"authorized": 0.9},
            user_messages=["send it"],
            call=_call(),
            prior_calls=[],
            history=AutoHistory(),
            forbid="forbidden",
        )
        assert d.outcome == "reject"

    async def test_permitted_double_check_floors_accept_to_ask(self) -> None:
        # The tripwire fired (forbid words present) but the check cleared it:
        # contradictory turns still need the human — a card, never a run.
        d = decide_from_verdict(
            choice="authorized",
            confidence=0.9,
            probabilities={"authorized": 0.9},
            user_messages=["don't send anything yet", "send it"],
            call=_call(),
            prior_calls=[],
            history=AutoHistory(),
            forbid="permitted",
        )
        assert d.outcome == "ask"
        assert "earlier message" in d.reason

    async def test_failed_double_check_floors_accept_to_ask(self) -> None:
        d = decide_from_verdict(
            choice="authorized",
            confidence=0.9,
            probabilities={"authorized": 0.9},
            user_messages=["don't send anything yet", "send it"],
            call=_call(),
            prior_calls=[],
            history=AutoHistory(),
            forbid="unclear-forbid",
        )
        assert d.outcome == "ask"


class TestGrounding:
    async def test_an_address_from_the_users_words_is_grounded(self) -> None:
        assert (
            ungrounded_targets(
                {"to": "bob@example.com"}, "draft an email to bob@example.com", []
            )
            == []
        )

    async def test_an_address_from_nowhere_blocks_accept(self) -> None:
        assert ungrounded_targets({"to": "mallory@evil.com"}, "send the deck", []) == [
            "mallory@evil.com"
        ]

    async def test_an_id_from_a_prior_lookup_is_grounded(self) -> None:
        priors = [PriorCall(name="FIND", args={"id": "d123"})]
        assert ungrounded_targets({"id": "d123"}, "delete it", priors) == []

    async def test_a_name_in_words_grounds_its_email(self) -> None:
        # "Sarah's" grounds sarah@x.com; the domain was resolved, not chosen.
        assert (
            ungrounded_targets(
                {"to": "sarah@x.com"}, "reply yes to sarah's thread", []
            )
            == []
        )

    async def test_a_name_prefix_never_grounds_a_longer_address(self) -> None:
        # "bob" must not ground bobby@evil.com — equality on the local part.
        assert ungrounded_targets({"to": "bobby@evil.com"}, "email bob now", []) == [
            "bobby@evil.com"
        ]

    async def test_a_known_address_is_provenance_not_novelty(self) -> None:
        assert (
            ungrounded_targets(
                {"to": "bob@x.com"}, "yes, send it", [], known=frozenset({"bob@x.com"})
            )
            == []
        )

    async def test_boolean_flags_are_never_targets(self) -> None:
        # Python's bool subclasses int: without the guard, create_meeting_room
        # becomes the target "True" and every meet-link booking asks forever.
        assert ungrounded_targets({"create_meeting_room": True}, "book it", []) == []

    async def test_prose_bodies_are_not_targets(self) -> None:
        # The body is judged by the choice criteria, not by provenance.
        assert ungrounded_targets({"body": "hello world 123"}, "send it", []) == []

    async def test_ungrounded_target_downgrades_a_confident_authorized_to_ask(
        self,
    ) -> None:
        d = await _decide(
            _answer("authorized", 0.99),
            call=_call(args={"to": "mallory@evil.com"}),
        )
        assert d.outcome == "ask"
        assert "mallory@evil.com" in d.reason


class TestVetoes:
    async def test_forbidden_is_reject_with_a_reason(self) -> None:
        d = await _decide(_answer("forbidden", 0.9))
        assert d.outcome == "reject"
        assert d.reason

    async def test_unclear_is_ask_carrying_tool_and_choice(self) -> None:
        # The gate adds the "Auto mode wasn't sure:" prefix uniformly;
        # the judge's job is naming the tool and the verdict behind it.
        d = await _decide(_answer("unclear", 0.9))
        assert d.outcome == "ask"
        assert "send_email" in d.reason and "unclear" in d.reason

    async def test_history_of_denies_blocks_an_authorized_call(self) -> None:
        d = await _decide(
            _answer("authorized", 0.99),
            history=AutoHistory(approved_recent=0, denied_recent=2),
        )
        assert d.outcome == "ask"
        assert "denied" in d.reason

    async def test_clean_history_leaves_authorized_accepted(self) -> None:
        d = await _decide(
            _answer("authorized", 0.99),
            history=AutoHistory(approved_recent=4, denied_recent=0),
        )
        assert d.outcome == "accept"


class TestFallback:
    async def test_transport_failure_runs_the_fallback_not_an_allow(self) -> None:
        from app.services.hil.intent import IntentDecision

        fallback = AsyncMock(
            **{"decide.return_value": IntentDecision("ask", "llm says ask")}
        )
        d = await _decide(ConnectionError("down"), fallback=fallback)
        assert d.outcome == "ask"
        assert d.reason == "llm says ask"
        fallback.decide.assert_awaited_once()

    async def test_malformed_answer_runs_the_fallback(self) -> None:
        fallback = AsyncMock()
        await _decide({"answers": {}}, fallback=fallback)
        fallback.decide.assert_awaited_once()

    async def test_unknown_choice_runs_the_fallback(self) -> None:
        fallback = AsyncMock()
        await _decide(_answer("maybe", 1.0), fallback=fallback)
        fallback.decide.assert_awaited_once()

    async def test_missing_api_key_runs_the_fallback(self) -> None:
        fallback = AsyncMock()
        judge = JevIntentJudge(fallback=fallback)
        with patch(f"{MODULE}.settings") as settings:
            settings.OPENROUTER_API_KEY = None
            await judge.decide(
                user_id="u",
                user_messages=["send it"],
                call=_call(),
                prior_calls=[],
                history=AutoHistory(),
            )
        fallback.decide.assert_awaited_once()


class TestWire:
    async def test_state_names_its_evidence_fields(self) -> None:
        """The criteria reference these names; a renamed field blinds the judge."""
        client = _client(_answer("unclear", 0.5))
        with patch(f"{MODULE}.httpx.AsyncClient", return_value=client):
            with patch(f"{MODULE}.settings") as settings:
                settings.OPENROUTER_API_KEY = "or-key"  # pragma: allowlist secret
                await ask_jev(
                    user_messages=["send it"],
                    call=_call(),
                    prior_calls=[],
                    history=AutoHistory(),
                )
        posted = client.__aenter__.return_value.post.await_args.kwargs["json"]
        assert set(posted["state"]) == {
            "user_messages",
            "pending_action",
            "prior_actions",
            "recent_history",
            "now",
        }
        assert set(posted["state"]["pending_action"]) == {
            "tool",
            "description",
            "summary",
            "args",
        }
        assert posted["questions"]["decision"]["type"] == "choice"

    async def test_usage_is_reported(self) -> None:
        with patch(f"{MODULE}.httpx.AsyncClient", return_value=_client(_answer("unclear", 0.5))):
            with patch(f"{MODULE}.settings") as settings:
                settings.OPENROUTER_API_KEY = "or-key"  # pragma: allowlist secret
                choice, conf, probs, tokens_in, tokens_out = await ask_jev(
                    user_messages=["send it"],
                    call=_call(),
                    prior_calls=[],
                    history=AutoHistory(),
                )
        assert (choice, conf) == ("unclear", 0.5)
        assert (tokens_in, tokens_out) == (300, 20)
        assert probs == {"unclear": 0.5}
