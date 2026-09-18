"""The pure parts: the action space, the questions, and answer validation."""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from app.constants.browser import JevOperation, SensitiveCategory
from app.services.browser.jev.gateway import JevChoiceAnswer
from app.services.browser.jev.prompts import (
    CAPTCHA_CHALLENGE,
    HUMAN_RULES,
    NAVIGATE_RULE,
    NEXT_ACTION,
    REQUEST_HUMAN_CRITERION,
    SOLVE_CAPTCHA_CRITERION,
    TAKEOVER_REASON,
    TARGET,
    TEXT_VALUE,
)
from app.services.browser.jev.ultrafast.model import (
    JevTextHelperError,
    JevUltrafastDecisionError,
    action_space,
    build_request,
    choose,
    field_context,
    validate_choice,
)
from tests.unit.services.browser.jev.ultrafast.conftest import (
    completion,
    distribution,
    make_gateway,
    make_text_helper,
)

pytestmark = pytest.mark.unit


def _answer(**overrides: Any) -> JevChoiceAnswer:
    base: dict[str, Any] = {
        "type": "choice",
        "choice": "a",
        "confidence": 0.8,
        "probabilities": {"a": 0.7, "b": 0.3},
    }
    base.update(overrides)
    return JevChoiceAnswer.model_validate(base)


# --------------------------------------------------------------------------
# action_space
# --------------------------------------------------------------------------


def test_one_index_per_node_with_operation_specific_targets(page) -> None:
    elements, targets, controls = action_space(page["actions"])

    assert len(elements) == 2
    assert elements[0]["index"] == "1"
    assert elements[0]["operations"] == ["TYPE_TEXT", "CLICK"]
    assert elements[0]["label"] == "Search"
    assert targets["TYPE_TEXT"]["1"]["id"] == "e1"
    assert targets["CLICK"]["1"]["id"] == "e2"
    assert targets["CLICK"]["2"]["id"] == "e3"
    assert set(controls) == {"WAIT"}


def test_a_dropdown_option_carries_a_code_owned_option_index(page) -> None:
    page["actions"][0:3] = [
        {
            "id": "e1",
            "kind": "select",
            "label": "Class → Economy",
            "node": 30,
            "value": "economy",
            "current_value": "Business",
        },
        {
            "id": "e2",
            "kind": "select",
            "label": "Class → First",
            "node": 30,
            "value": "first",
            "current_value": "Business",
        },
    ]

    elements, targets, _ = action_space(page["actions"])

    assert len(elements) == 1
    assert elements[0]["label"] == "Class"
    assert elements[0]["value"] == "Business"
    assert [o["index"] for o in elements[0]["options"]] == ["1:1", "1:2"]
    assert targets["SELECT"]["1:1"]["value"] == "economy"
    assert targets["SELECT"]["1:2"]["value"] == "first"


def test_a_truncated_action_table_offers_only_what_it_kept(page) -> None:
    """snapshot.js caps the table; only the rows that survived are selectable."""
    kept = [a for a in page["actions"] if a["id"] != "e3"]

    _, targets, _ = action_space(kept)

    assert "2" not in targets["CLICK"]


# --------------------------------------------------------------------------
# question construction
# --------------------------------------------------------------------------


def test_one_request_carries_the_operation_head_and_a_head_per_element_operation(page) -> None:
    history = [{"action": "CLICK [2] Go", "kind": "click", "text": None, "page_changed": True}]

    request, operations, targets, controls = build_request(page, "Find a book", history)

    assert set(request.questions) == {"operation", "click_target", "type_text_target"}
    assert set(operations) == {"CLICK", "TYPE_TEXT", "WAIT", "NAVIGATE", "DONE", "BLOCKED"}
    assert operations["WAIT"] == "Wait for the page to update"
    assert set(controls) == {"WAIT"}
    assert set(targets) == {"CLICK", "TYPE_TEXT"}


def test_the_state_is_the_page_the_element_table_and_recent_actions(page) -> None:
    history = [{"action": f"step {i}", "kind": "click", "text": None} for i in range(12)]

    request, _, _, _ = build_request(page, "Find a book", history)

    assert request.state["page"] == {
        "url": "https://example.test/",
        "title": "Search",
        "text": "Search",
    }
    assert [e["index"] for e in request.state["elements"]] == ["1", "2"]
    recent = request.state["recent_actions"]
    assert len(recent) == 10
    assert recent[0] == {"action": "step 2", "kind": "click", "text": None, "page_changed": None}


def test_a_target_head_states_the_operation_it_assumes_and_the_full_next_step_rules(page) -> None:
    request, _, _, _ = build_request(page, "Find a book", [])

    operation = request.questions["operation"]
    target = request.questions["click_target"]
    assert operation.instructions == {"goal": "Find a book", "rules": [NEXT_ACTION, NAVIGATE_RULE]}
    assert target.instructions == {
        "goal": "Find a book",
        "operation": "CLICK",
        "rules": [NEXT_ACTION, TARGET],
    }
    assert target.criteria["2"] == {"element": "[2] Go", "current_value": "", "role": "button"}


def test_the_target_criteria_carry_control_state(page) -> None:
    page["actions"].insert(
        0,
        {
            "id": "toggle",
            "kind": "click",
            "label": "Free cancellation",
            "node": 30,
            "role": "checkbox",
            "checked": "true",
        },
    )

    request, _, _, _ = build_request(page, "Search with free cancellation", [])

    assert request.questions["click_target"].criteria["1"]["checked"] == "true"


# --------------------------------------------------------------------------
# validate_choice
# --------------------------------------------------------------------------


def test_a_well_formed_answer_is_accepted() -> None:
    answer = _answer()

    assert validate_choice(answer, {"a", "b"}) is answer


@pytest.mark.parametrize(
    ("name", "overrides"),
    [
        ("unoffered_choice", {"choice": "invented"}),
        ("not_the_argmax", {"choice": "b"}),
        ("missing_key", {"probabilities": {"a": 1.0}}),
        ("extra_key", {"probabilities": {"a": 0.5, "b": 0.3, "c": 0.2}}),
        ("no_probabilities", {"probabilities": {}}),
        ("negative", {"probabilities": {"a": 1.3, "b": -0.3}}),
        ("above_one", {"probabilities": {"a": 1.4, "b": -0.4}}),
        ("nan", {"probabilities": {"a": float("nan"), "b": 0.3}}),
        ("infinite", {"probabilities": {"a": float("inf"), "b": 0.3}}),
        ("mass_not_one", {"probabilities": {"a": 0.7, "b": 0.1}}),
        ("confidence_out_of_range", {"confidence": 5}),
        ("confidence_missing", {"confidence": None}),
    ],
)
def test_a_malformed_answer_executes_nothing(name: str, overrides: dict[str, Any]) -> None:
    with pytest.raises(JevUltrafastDecisionError, match="Invalid Jev response"):
        validate_choice(_answer(**overrides), {"a", "b"})


def test_a_missing_answer_executes_nothing() -> None:
    with pytest.raises(JevUltrafastDecisionError):
        validate_choice(None, {"a", "b"})


def test_rounding_inside_the_tolerance_is_still_valid() -> None:
    answer = _answer(probabilities={"a": 0.7, "b": 0.31})

    assert math.isclose(sum(validate_choice(answer, {"a", "b"}).probabilities.values()), 1.01)


# --------------------------------------------------------------------------
# choose: one round trip, only the matching head executes
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_head_answers_in_one_request_and_only_the_matching_one_executes(page) -> None:
    def responder(body: dict[str, Any]) -> dict[str, Any]:
        questions = body["questions"]
        return {
            "answers": {
                "operation": distribution(questions["operation"]["criteria"], "TYPE_TEXT"),
                "type_text_target": distribution(["1"], "1"),
                "click_target": {"type": "choice", "choice": "invented"},
            },
            "usage": {"inputTokens": 410, "outputTokens": 38},
        }

    client, recorder = make_gateway(responder)

    decision = await choose(client, page, "Find a book", [])

    assert len(recorder.bodies) == 1
    assert recorder.bodies[0]["model"] == "~typesafe/jev-latest"
    assert decision.operation == "TYPE_TEXT"
    assert decision.target == "1"
    assert decision.choice == "e1"
    assert decision.probabilities == {"e1": 1.0}
    assert decision.usage is not None
    assert decision.usage.input_tokens == 410


@pytest.mark.asyncio
async def test_a_click_cannot_consume_a_target_from_another_head(page) -> None:
    def responder(body: dict[str, Any]) -> dict[str, Any]:
        questions = body["questions"]
        return {
            "answers": {
                "operation": distribution(questions["operation"]["criteria"], "CLICK"),
                "type_text_target": distribution(["1"], "1"),
                "click_target": distribution(["1", "2", "999"], "999"),
            }
        }

    client, _ = make_gateway(responder)

    with pytest.raises(JevUltrafastDecisionError):
        await choose(client, page, "Find a book", [])


@pytest.mark.asyncio
async def test_a_control_operation_resolves_to_its_observed_action_id(page) -> None:
    def responder(body: dict[str, Any]) -> dict[str, Any]:
        return {
            "answers": {
                "operation": distribution(body["questions"]["operation"]["criteria"], "WAIT")
            }
        }

    client, _ = make_gateway(responder)

    decision = await choose(client, page, "Find a book", [])

    assert decision.choice == "wait"
    assert decision.target is None


@pytest.mark.asyncio
async def test_done_carries_no_target(page) -> None:
    def responder(body: dict[str, Any]) -> dict[str, Any]:
        return {
            "answers": {
                "operation": distribution(body["questions"]["operation"]["criteria"], "DONE")
            }
        }

    client, _ = make_gateway(responder)

    decision = await choose(client, page, "Find a book", [])

    assert decision.choice == "DONE"
    assert decision.target_probabilities == {}


# --------------------------------------------------------------------------
# the human-in-the-loop operations
# --------------------------------------------------------------------------

_HANDOFFS = (JevOperation.REQUEST_HUMAN, JevOperation.SOLVE_CAPTCHA)


def test_no_handler_means_neither_human_operation_is_offered(page) -> None:
    request, operations, _, _ = build_request(page, "Sign in", [])

    assert "REQUEST_HUMAN" not in operations
    assert "SOLVE_CAPTCHA" not in operations
    assert set(request.questions["operation"].criteria) == set(operations)
    assert request.questions["operation"].instructions["rules"] == [NEXT_ACTION, NAVIGATE_RULE]


@pytest.mark.parametrize(
    ("handoffs", "offered", "absent"),
    [
        ((JevOperation.REQUEST_HUMAN,), "REQUEST_HUMAN", "SOLVE_CAPTCHA"),
        ((JevOperation.SOLVE_CAPTCHA,), "SOLVE_CAPTCHA", "REQUEST_HUMAN"),
    ],
)
def test_only_the_handed_operation_is_offered(page, handoffs, offered, absent) -> None:
    _, operations, _, _ = build_request(page, "Sign in", [], handoffs)

    assert offered in operations
    assert absent not in operations


def test_the_human_operations_are_controls_with_no_target_head(page) -> None:
    request, operations, targets, controls = build_request(page, "Sign in", [], _HANDOFFS)

    assert operations["REQUEST_HUMAN"] == REQUEST_HUMAN_CRITERION
    assert operations["SOLVE_CAPTCHA"] == SOLVE_CAPTCHA_CRITERION
    assert set(request.questions) == {"operation", "click_target", "type_text_target"}
    assert set(targets) == {"CLICK", "TYPE_TEXT"}
    assert set(controls) == {"WAIT"}


def test_the_human_rules_ride_in_the_same_instruction_block(page) -> None:
    request, _, _, _ = build_request(page, "Sign in", [], _HANDOFFS)

    assert request.questions["operation"].instructions["rules"] == [
        NEXT_ACTION,
        NAVIGATE_RULE,
        HUMAN_RULES,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["REQUEST_HUMAN", "SOLVE_CAPTCHA"])
async def test_a_chosen_human_operation_carries_no_target(page, operation: str) -> None:
    def responder(body: dict[str, Any]) -> dict[str, Any]:
        return {
            "answers": {
                "operation": distribution(body["questions"]["operation"]["criteria"], operation)
            }
        }

    client, _ = make_gateway(responder)

    decision = await choose(client, page, "Sign in", [], _HANDOFFS)

    assert decision.choice == operation
    assert decision.operation == operation
    assert decision.target is None
    assert decision.probabilities == {operation: pytest.approx(0.9)}


@pytest.mark.asyncio
async def test_an_unoffered_human_operation_executes_nothing(page) -> None:
    """Jev naming REQUEST_HUMAN when no handler exists is not an offered choice."""

    def responder(body: dict[str, Any]) -> dict[str, Any]:
        criteria = [*body["questions"]["operation"]["criteria"], "REQUEST_HUMAN"]
        return {"answers": {"operation": distribution(criteria, "REQUEST_HUMAN")}}

    client, _ = make_gateway(responder)

    with pytest.raises(JevUltrafastDecisionError):
        await choose(client, page, "Sign in", [])


# --------------------------------------------------------------------------
# the text helper
# --------------------------------------------------------------------------


def test_the_field_context_is_the_goal_the_field_and_visible_page_text(page) -> None:
    context = field_context('Fly from "Zurich" to London', page["actions"][0], page, [])

    assert context["goal"] == 'Fly from "Zurich" to London'
    assert context["field"] == {"label": "Search", "role": "textbox", "value": ""}
    assert context["page"] == {"title": "Search", "text": "Search"}


@pytest.mark.asyncio
async def test_quoted_goal_text_still_goes_through_the_helper(page) -> None:
    helper, recorder = make_text_helper(lambda _body: completion('{"text":"Zurich"}'))
    context = field_context('Fly from "Zurich" to London', page["actions"][0], page, [])

    value, call = await helper.field_text(context)

    assert value == "Zurich"
    assert call.model == "inception/mercury-2.5"
    body = recorder.bodies[0]
    assert body["model"] == "inception/mercury-2.5"
    assert body["reasoning"] == {"enabled": False}
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][0] == {"role": "system", "content": TEXT_VALUE}
    assert json.loads(body["messages"][1]["content"])["goal"] == 'Fly from "Zurich" to London'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        "Thinking: Zurich",
        '{"text":null}',
        '{"text":""}',
        '{"text":"   "}',
        '{"text":"Zurich","extra":true}',
        '{"text":123}',
        "{}",
        '["Zurich"]',
    ],
)
async def test_an_unusable_helper_answer_types_nothing(content: str) -> None:
    helper, _ = make_text_helper(lambda _body: completion(content))

    with pytest.raises(JevTextHelperError, match="nothing typed"):
        await helper.field_text({"goal": "Find a flight"})


@pytest.mark.asyncio
async def test_an_over_budget_value_types_nothing() -> None:
    helper, _ = make_text_helper(lambda _body: completion(json.dumps({"text": "x" * 2001})))

    with pytest.raises(JevTextHelperError, match="nothing typed"):
        await helper.field_text({"goal": "Find a flight"})


@pytest.mark.asyncio
async def test_the_takeover_reason_carries_its_category(page) -> None:
    helper, calls = make_text_helper(
        lambda _body: completion(
            json.dumps({"text": "Enter your password", "category": "credentials"})
        )
    )

    reason, category, call = await helper.takeover_reason(field_context("Sign in", None, page, []))

    assert reason == "Enter your password"
    assert category is SensitiveCategory.CREDENTIALS
    assert call.model == "inception/mercury-2.5"
    assert calls.bodies[0]["messages"][0]["content"] == TAKEOVER_REASON
    assert json.loads(calls.bodies[0]["messages"][1]["content"])["field"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        '{"text":"Enter your password"}',
        '{"text":"Enter your password","category":"none"}',
        '{"text":"Enter your password","category":"invented"}',
        '{"text":"","category":"payment"}',
        '{"text":"Enter your password","category":"payment","extra":1}',
    ],
)
async def test_an_unusable_takeover_reason_hands_off_nothing(content: str) -> None:
    helper, _ = make_text_helper(lambda _body: completion(content))

    with pytest.raises(JevTextHelperError, match="nothing typed"):
        await helper.takeover_reason({"goal": "Sign in"})


@pytest.mark.asyncio
async def test_the_captcha_challenge_uses_its_own_prompt_and_the_same_validation() -> None:
    helper, calls = make_text_helper(lambda _body: completion('{"text":"Solve the CAPTCHA"}'))

    challenge, _ = await helper.field_text({"goal": "Sign in"}, instructions=CAPTCHA_CHALLENGE)

    assert challenge == "Solve the CAPTCHA"
    assert calls.bodies[0]["messages"][0]["content"] == CAPTCHA_CHALLENGE


def test_a_page_with_nothing_to_act_on_can_still_be_left(page) -> None:
    """A run starts on a blank page, so without NAVIGATE its only honest answer is BLOCKED."""
    blank = {**page, "url": "about:blank", "text": "", "actions": []}

    _, operations, targets, _ = build_request(blank, "Open Wikipedia", [])

    assert JevOperation.NAVIGATE.value in operations
    assert not targets


def test_navigate_is_a_control_so_it_never_asks_for_an_element(page) -> None:
    request, operations, _, controls = build_request(page, "Open Wikipedia", [])

    assert JevOperation.NAVIGATE.value in operations
    assert JevOperation.NAVIGATE.value not in controls
    assert "navigate_target" not in request.questions


@pytest.mark.asyncio
async def test_one_off_contract_answer_is_retried_rather_than_ending_the_run() -> None:
    """The helper is a nondeterministic model: one malformed answer must not be the
    reason a whole browser task stops."""
    answers = iter([completion("sorry, I cannot"), completion('{"text":"dune"}')])
    helper, calls = make_text_helper(lambda _body: next(answers))

    value, _ = await helper.field_text({"goal": "Find a book"})

    assert value == "dune"
    assert len(calls.bodies) == 2


@pytest.mark.asyncio
async def test_a_helper_that_never_answers_in_contract_still_fails_loud() -> None:
    helper, calls = make_text_helper(lambda _body: completion("sorry, I cannot"))

    with pytest.raises(JevTextHelperError, match="nothing typed"):
        await helper.field_text({"goal": "Find a book"})

    assert len(calls.bodies) == 3


@pytest.mark.asyncio
async def test_a_refused_value_says_the_task_never_gave_one() -> None:
    """The prompt asks for null when the goal names no value, and the user reads this
    message — "no valid field value" would blame the model for the task's gap."""
    helper, _ = make_text_helper(lambda _body: completion('{"text":null}'))

    with pytest.raises(JevTextHelperError, match="does not say what this field needs"):
        await helper.field_text({"goal": "Fill in the form"})
