"""The loop's state machine, driven by a fake browser and canned decisions."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import pytest

from app.services.browser.jev.ultrafast.agent import JevRunStopped, JevUltrafastAgent
from app.services.browser.jev.ultrafast.browser import StalePage
from app.services.browser.jev.ultrafast.model import JevUltrafastDecision
from tests.unit.services.browser.jev.ultrafast.conftest import (
    FakeBrowser,
    completion,
    distribution,
    make_gateway,
    make_text_helper,
)

pytestmark = pytest.mark.unit


def decide(choice: str = "e1", operation: str = "TYPE_TEXT") -> JevUltrafastDecision:
    return JevUltrafastDecision(
        choice=choice,
        operation=operation,
        target="1",
        confidence=0.9,
        probabilities={choice: 1.0},
        operation_probabilities={operation: 1.0},
        latency_ms=10,
    )


def make_agent(page, *, decisions: Any = None, text: str = '{"text":"book"}'):
    """An agent already past its first observation, with canned model answers."""
    browser = FakeBrowser(page)
    client, gateway_calls = make_gateway(decisions or (lambda _body: {"answers": {}}))
    helper, text_calls = make_text_helper(lambda _body: completion(text))
    agent = JevUltrafastAgent(
        browser,  # type: ignore[arg-type]
        page,
        goal="Find a book",
        client=client,
        text_helper=helper,
    )
    agent.started_at = perf_counter()
    agent.status = "predicted"
    return agent, browser, gateway_calls, text_calls


# --------------------------------------------------------------------------
# consume before mutate
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_stale_page_consumes_the_decision_without_touching_the_browser(page) -> None:
    agent, browser, _, _ = make_agent(page)
    agent.decision = decide("e3", "CLICK")
    browser.is_fresh = False

    with pytest.raises(StalePage):
        await agent.act(page["fingerprint"])

    assert agent.decision is None
    assert browser.acted == []


@pytest.mark.asyncio
async def test_acting_twice_on_one_decision_cannot_double_click(page) -> None:
    agent, browser, _, _ = make_agent(page)
    agent.decision = decide("e3", "CLICK")

    await agent.act(page["fingerprint"])
    with pytest.raises(JevRunStopped, match="Observe and choose"):
        await agent.act(page["fingerprint"])

    assert browser.acted == [("e3", None)]


@pytest.mark.asyncio
async def test_a_decision_taken_on_another_page_is_refused(page) -> None:
    agent, browser, _, _ = make_agent(page)
    agent.decision = decide("e3", "CLICK")

    with pytest.raises(JevRunStopped, match="Observe and choose"):
        await agent.act("a-different-fingerprint")

    assert browser.acted == []


# --------------------------------------------------------------------------
# the text helper and its one-retry cache
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_generated_value_is_reused_for_an_identical_retry(page) -> None:
    agent, browser, _, text_calls = make_agent(page)
    agent.decision = decide("e1")
    browser.act_error = StalePage("Changed before input")

    with pytest.raises(StalePage):
        await agent.act(page["fingerprint"])
    agent.decision = decide("e1")
    await agent.act(page["fingerprint"])

    assert len(text_calls.bodies) == 1
    assert browser.acted == [("e1", "book")]
    assert agent.pending_text is None
    assert [c["value"] for c in agent.text_calls] == ["book"]


@pytest.mark.asyncio
async def test_a_changed_helper_context_regenerates_the_value(page) -> None:
    agent, browser, _, text_calls = make_agent(page)
    agent.decision = decide("e1")
    browser.act_error = StalePage("Changed before input")

    with pytest.raises(StalePage):
        await agent.act(page["fingerprint"])
    page["text"] = "A different page context"
    agent.decision = decide("e1")
    await agent.act(page["fingerprint"])

    assert len(text_calls.bodies) == 2


@pytest.mark.asyncio
async def test_no_text_is_generated_for_a_click(page) -> None:
    agent, browser, _, text_calls = make_agent(page)
    agent.decision = decide("e3", "CLICK")

    await agent.act(page["fingerprint"])

    assert text_calls.bodies == []
    assert browser.acted == [("e3", None)]


@pytest.mark.asyncio
async def test_a_page_that_moved_before_generation_types_nothing(page) -> None:
    agent, browser, _, text_calls = make_agent(page)
    agent.decision = decide("e1")
    browser.is_fresh = False

    with pytest.raises(StalePage, match="before text generation"):
        await agent.act(page["fingerprint"])

    assert text_calls.bodies == []
    assert browser.acted == []


# --------------------------------------------------------------------------
# history and the no-progress stop
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_executed_action_survives_a_stale_observation(page) -> None:
    agent, browser, _, _ = make_agent(page)
    agent.decision = decide("e3", "CLICK")
    browser.observe_error = StalePage("changed")

    with pytest.raises(StalePage):
        await agent.act(page["fingerprint"])

    assert [h["action"] for h in agent.history] == ["Go"]
    assert browser.acted == [("e3", None)]


@pytest.mark.asyncio
async def test_three_unchanged_actions_stop_the_run(page) -> None:
    agent, browser, _, _ = make_agent(page)

    for _ in range(3):
        agent.decision = decide("e3", "CLICK")
        await agent.act(page["fingerprint"])

    assert len(agent.history) == 3
    assert all(h["page_changed"] is False for h in agent.history)
    assert agent.status == "blocked"


@pytest.mark.asyncio
async def test_loading_waits_are_not_a_no_progress_stop(page) -> None:
    agent, _, _, _ = make_agent(page)

    for _ in range(5):
        agent.decision = decide("wait", "WAIT")
        await agent.act(page["fingerprint"])

    assert len(agent.history) == 5
    assert agent.status == "ready"


@pytest.mark.asyncio
async def test_the_action_budget_stops_the_run(page) -> None:
    agent, browser, _, _ = make_agent(page)
    agent.history = [{"step": i} for i in range(60)]
    agent.decision = decide("e3", "CLICK")

    with pytest.raises(JevRunStopped, match="60-action budget"):
        await agent.act(page["fingerprint"])

    assert agent.status == "blocked"
    assert browser.acted == []


# --------------------------------------------------------------------------
# finishing
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("choice", "status"), [("DONE", "done"), ("BLOCKED", "blocked")])
async def test_a_final_choice_ends_the_run_without_acting(page, choice, status) -> None:
    agent, browser, _, _ = make_agent(page)
    agent.decision = decide(choice, choice)

    await agent.act(page["fingerprint"])

    assert agent.status == status
    assert browser.acted == []


@pytest.mark.asyncio
async def test_done_on_a_page_that_moved_is_refused(page) -> None:
    agent, browser, _, _ = make_agent(page)
    agent.decision = decide("DONE", "DONE")
    browser.is_fresh = False

    with pytest.raises(StalePage):
        await agent.act(page["fingerprint"])

    assert agent.status == "ready"


# --------------------------------------------------------------------------
# tick and run
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_navigation_during_prediction_reobserves_instead_of_acting(page) -> None:
    agent, browser, gateway_calls, _ = make_agent(page)
    browser.fresh_error = StalePage("Document navigating")

    await agent.tick()

    assert agent.status == "ready"
    assert agent.decision is None
    assert browser.acted == []
    assert gateway_calls.bodies == []


@pytest.mark.asyncio
async def test_a_run_ends_at_done_and_reports_every_step(page) -> None:
    answers = iter(["CLICK", "DONE"])

    def responder(body: dict[str, Any]) -> dict[str, Any]:
        operation = next(answers)
        result = {"operation": distribution(body["questions"]["operation"]["criteria"], operation)}
        if operation == "CLICK":
            result["click_target"] = distribution(
                body["questions"]["click_target"]["criteria"], "2"
            )
        return {"answers": result}

    agent, browser, gateway_calls, _ = make_agent(page, decisions=responder)
    agent.status = "ready"

    result = await agent.run()

    assert result.status == "done"
    assert result.succeeded
    assert len(gateway_calls.bodies) == 2
    assert browser.acted == [("e3", None)]
    assert [h["action"] for h in result.history] == ["Go"]
    assert result.history[0]["operation"] == "CLICK"
    assert result.url == "https://example.test/"


@pytest.mark.asyncio
async def test_the_decision_budget_stops_the_run(page) -> None:
    agent, _, _, _ = make_agent(page)
    agent.decisions = [{"choice": "e3"} for _ in range(120)]
    agent.status = "ready"

    with pytest.raises(JevRunStopped, match="model-call budget"):
        await agent.predict()
