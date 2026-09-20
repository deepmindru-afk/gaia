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
    map_jev_choice,
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


def _answer(choice: str, confidence: float) -> dict[str, Any]:
    return {
        "answers": {
            "decision": {
                "type": "choice",
                "choice": choice,
                "confidence": confidence,
                "probabilities": {choice: confidence},
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
