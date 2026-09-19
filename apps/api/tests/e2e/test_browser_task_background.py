"""The user journeys of a browser task that runs in a worker instead of inside the turn.

The real compiled executor graph runs against a scripted model; the real worker
task body runs in this process, on a fake Redis, over a scripted Browser-Use
agent. What that costs in fidelity: there is no process boundary and no ARQ
serialization, and the browser is scripted rather than real, so nothing here
proves a run survives an API restart or that Browser-Use behaves as scripted.
"""

import asyncio
from typing import Any

import pytest

from app.constants.browser import BrowserSessionStatus
from app.constants.chat import SourceCategory
from app.models.chat_models import ConversationSource
from app.services.browser.jobs import get_conversation_slot
from tests.e2e._harness.browser_job import (
    LIVE_VIEW_LINK,
    SHOT_URL_TEMPLATE,
    JevScript,
    JobWorld,
    ScriptedStep,
    browser_job_world,
)
from tests.e2e._harness.graph_run import call, executor_graph, run_graph

pytestmark = pytest.mark.e2e

CONVERSATION = "conv-browser-e2e"
STREAM = "stream-browser-e2e"
USER = "user-e2e"

RETRIEVE = call("retrieve_tools", {"exact_tool_names": ["browser_task"]}, "r1")
START = call("browser_task", {"task": "book a table for two at 7pm"}, "b1")
JOIN = call("wait_for_browser_task", {}, "w1")

TWO_STEPS = [
    ScriptedStep(
        actions=[("go_to_url", {"url": "https://example.test/book"})],
        outputs=["opened the booking page"],
    ),
    ScriptedStep(
        actions=[("click", {"index": 4})],
        url="https://example.test/confirm",
        outputs=["confirmed the table"],
    ),
]


def _configurable(**extra: Any) -> dict[str, Any]:
    return {"conversation_id": CONVERSATION, "stream_id": STREAM, **extra}


async def _drive(graph: Any, world: JobWorld, prompt: str = "book me a table") -> Any:
    run = await run_graph(
        graph,
        prompt,
        thread_id=CONVERSATION,
        user_id=USER,
        **_configurable(),
    )
    await world.settle()
    return run


async def test_the_run_answers_the_question_and_the_executor_reports_that_answer() -> None:
    """The tool call no longer carries the result, so the turn only has an answer if the join really collects one out of the worker's terminal state."""
    async with browser_job_world(STREAM, steps=TWO_STEPS) as world:
        async with executor_graph([RETRIEVE, START, JOIN, "Booked."]) as graph:
            run = await _drive(graph, world)

    assert run.ran("browser_task")
    assert "started in the background" in (run.result_for("browser_task") or "")
    joined = run.result_for("wait_for_browser_task") or ""
    assert joined.startswith("The table is booked for 7pm on Friday.")
    assert len(world.enqueued) == 1


async def test_the_join_is_bound_from_the_first_model_call() -> None:
    """A started run with no reachable join is a run whose answer nobody can collect, so the join must never depend on a retrieval hit."""
    async with browser_job_world(STREAM, steps=TWO_STEPS) as world:
        async with executor_graph([RETRIEVE, START, JOIN, "Booked."]) as graph:
            run = await _drive(graph, world)

    assert "wait_for_browser_task" in run.bound[0]
    assert "wait_for_browser_task" not in run.bound_tools()


async def test_every_step_card_reaches_the_turns_stream_and_its_message() -> None:
    """The cards are produced in a worker with no stream of its own; without the relay the user watches nothing and reloads to nothing."""
    from app.agents.core.background.executor_capture import drain_executor_tool_data

    async with browser_job_world(STREAM, steps=TWO_STEPS) as world:
        async with executor_graph([RETRIEVE, START, JOIN, "Booked."]) as graph:
            await _drive(graph, world)

        cards = world.cards()
        assert [card["kind"] for card in cards] == ["session", "step", "step", "result"]
        assert [card["index"] for card in cards if card["kind"] == "step"] == [1, 2]
        assert cards[3]["status"] == BrowserSessionStatus.COMPLETED.value

        persisted = drain_executor_tool_data(STREAM)
        assert [
            entry["data"]["kind"]
            for entry in persisted
            if entry["tool_name"] == "browser_task_data"
        ] == ["session", "step", "step", "result"]


async def test_a_bot_conversation_gets_one_photo_per_step_and_the_result_line() -> None:
    """Bots never see the SSE stream: a run the worker does not mirror to the platform is a run a Discord user watches in silence."""
    async with browser_job_world(STREAM, steps=TWO_STEPS) as world:
        async with executor_graph([RETRIEVE, START, JOIN, "Booked."]) as graph:
            await run_graph(
                graph,
                "book me a table",
                thread_id=CONVERSATION,
                user_id=USER,
                **_configurable(
                    source_category=SourceCategory.BOT.value,
                    conversation_source=ConversationSource.DISCORD.value,
                ),
            )
            await world.settle()

    assert world.bot_photos == [
        SHOT_URL_TEMPLATE.format(index=1),
        SHOT_URL_TEMPLATE.format(index=2),
    ]
    assert any(LIVE_VIEW_LINK in message for message in world.bot_messages)
    assert world.bot_messages[-1].startswith("✅")


async def test_a_turn_that_ends_without_joining_has_the_result_delivered_to_the_user() -> None:
    """Nothing is holding the tool call open any more, so a run that outlives its turn has to speak for itself or the user is left with a started notice and no outcome."""
    async with browser_job_world(STREAM, steps=TWO_STEPS) as world:
        async with executor_graph([RETRIEVE, START, "I've started on it."]) as graph:
            await _drive(graph, world)

    assert len(world.deliveries) == 1
    delivery = world.deliveries[0]
    assert "The table is booked for 7pm on Friday." in delivery["text"]
    assert delivery["conversation_id"] == CONVERSATION
    assert [
        entry["data"]["kind"]
        for entry in delivery["tool_data"]
        if entry["tool_name"] == "browser_task_data"
    ] == ["session", "step", "step", "result"]


async def test_a_second_browser_task_in_one_turn_is_refused_by_name() -> None:
    """One browser per conversation: a second run would fight the first for the same live view, and the model would narrate whichever finished last."""
    second = call("browser_task", {"task": "also check the menu"}, "b2")
    async with browser_job_world(STREAM, steps=TWO_STEPS) as world:
        async with executor_graph([RETRIEVE, START, second, JOIN, "Booked."]) as graph:
            run = await _drive(graph, world)

    assert len(world.enqueued) == 1
    refusal = run.results_from("tools")
    assert any("already running in this conversation" in text for text in refusal)
    assert any(world.enqueued[0].job_id in text for text in refusal)


async def test_a_stop_reaches_the_browser_and_releases_the_conversation() -> None:
    """The run outlives the turn that started it, so a stop typed later has to reach the job itself rather than a stream nobody is on."""
    from app.agents.tools.executor_tool import cancel_executor

    waiting = [ScriptedStep(actions=[], await_stop=True)]
    async with browser_job_world(STREAM, steps=waiting) as world:
        async with executor_graph([RETRIEVE, START, "Started."]) as graph:
            # No settle here on purpose: the run is still on the page, which is
            # the only state a stop can reach and the whole point of this journey.
            await run_graph(
                graph,
                "book me a table",
                thread_id=CONVERSATION,
                user_id=USER,
                **_configurable(),
            )
            assert world.enqueued, "the job never reached the worker"

            await cancel_executor.ainvoke(
                {"task_ids": []}, config={"configurable": {"thread_id": CONVERSATION}}
            )
            await world.settle()

    assert world.browser.stop_observed
    assert world.cards()[-1]["status"] == BrowserSessionStatus.CANCELLED.value
    assert await get_conversation_slot(CONVERSATION) is None


async def test_a_worker_crash_still_reports_a_failure_and_frees_the_conversation() -> None:
    """A job that dies silently wedges the conversation's one browser slot and leaves the user watching a run that already stopped."""
    async with browser_job_world(
        STREAM, host_error=RuntimeError("the browser host fell over")
    ) as world:
        async with executor_graph([RETRIEVE, START, "Started."]) as graph:
            await _drive(graph, world)

    assert len(world.deliveries) == 1
    assert "DID NOT COMPLETE" in world.deliveries[0]["text"]
    assert await get_conversation_slot(CONVERSATION) is None


async def test_a_handoff_note_reaches_the_run_and_the_policy_deciding_it() -> None:
    """The user takes over mid-run and leaves an instruction; a note that stopped at the tool result would leave the policy still working the goal the user just changed."""
    from app.constants.browser import HandoffDecision, HandoffStatus
    from app.services.browser.handoff import resolve_handoff

    note = "skip the login, just tell me the opening hours"
    script = JevScript(
        decisions=[("REQUEST_HUMAN", None), ("TYPE_TEXT", "1")],
        texts=[{"text": "Sign in and come back", "category": "credentials"}, {"text": "hours"}],
    )
    steps = [ScriptedStep(actions=[], decide=True), ScriptedStep(actions=[], decide=True)]

    async with browser_job_world(STREAM, steps=steps, jev=script) as world:
        async with executor_graph([RETRIEVE, START, JOIN, "Done."]) as graph:
            run_task = asyncio.create_task(
                run_graph(
                    graph,
                    "book me a table",
                    thread_id=CONVERSATION,
                    user_id=USER,
                    **_configurable(),
                )
            )
            handoff_id = await _wait_for_pending_handoff(world)
            assert await resolve_handoff(handoff_id, HandoffDecision.CONTINUE, USER, note) == (
                HandoffStatus.COMPLETED
            )
            run = await run_task
            await world.settle()

    assert world.browser.takeover_notes == [note]
    assert world.jev is not None
    assert note in world.jev.goal(1)
    assert "book a table for two at 7pm" in world.jev.goal(1)
    handoffs = [card for card in world.cards() if card["kind"] == "handoff"]
    assert [card["status"] for card in handoffs] == ["pending", "completed"]
    joined = run.result_for("wait_for_browser_task") or ""
    assert joined.startswith("The table is booked for 7pm on Friday.")
    # The closing reply is written against the original booking otherwise, and
    # confirms a table nobody booked.
    assert note in joined


async def _wait_for_pending_handoff(world: JobWorld) -> str:
    """Return the handoff id off the card the run put on the turn's stream."""
    for _ in range(500):
        pending = [
            card
            for card in world.cards()
            if card["kind"] == "handoff" and card["status"] == "pending"
        ]
        if pending:
            return str(pending[0]["handoff_id"])
        await asyncio.sleep(0.01)
    raise AssertionError("the run never asked the user to take over")


# ---------------------------------------------------------------------------
# A blocked run asks the executor that started it
# ---------------------------------------------------------------------------

GUIDE = call(
    "guide_browser_task", {"instruction": "open the Contact tab, the hours are there"}, "g1"
)
JOIN_AGAIN = call("wait_for_browser_task", {}, "w2")

BLOCK_THEN_ACT = [
    ScriptedStep(actions=[], decide=True, await_joiner=True),
    ScriptedStep(actions=[], decide=True),
]


def _blocked_script(blocks: int = 1) -> JevScript:
    """Return a Jev script that gives up the given number of times, then types once."""
    return JevScript(
        decisions=[("BLOCKED", None)] * blocks + [("TYPE_TEXT", "1")],
        texts=[{"text": "The opening hours are not on this page."}] * blocks + [{"text": "hours"}],
    )


async def test_a_blocked_run_is_unstuck_by_the_executor_that_started_it() -> None:
    """A run that ends on BLOCKED throws away everything the executor knows; the whole point is that the instruction reaches the policy deciding the next step."""
    async with browser_job_world(STREAM, steps=BLOCK_THEN_ACT, jev=_blocked_script()) as world:
        async with executor_graph([RETRIEVE, START, JOIN, GUIDE, JOIN_AGAIN, "Done."]) as graph:
            run = await _drive(graph, world)

    asked = run.result_for("wait_for_browser_task") or ""
    assert "THE BROWSER TASK IS STUCK" in asked
    assert "The opening hours are not on this page." in asked
    assert "guide_browser_task" in asked
    assert world.browser.guidance_notes == ["open the Contact tab, the hours are there"]
    assert world.jev is not None
    assert "open the Contact tab, the hours are there" in world.jev.goal(1)
    assert "book a table for two at 7pm" in world.jev.goal(1)
    assert (run.results_from("tools") or [])[-1].startswith(
        "The table is booked for 7pm on Friday."
    )


async def test_the_user_is_never_shown_a_handoff_for_a_question_asked_of_the_executor() -> None:
    """A handoff card and the conversation's pending key would ask the user to answer something they were never told about, and swallow their next chat message."""
    from app.services.browser.handoff import get_conversation_pending_handoff

    async with browser_job_world(STREAM, steps=BLOCK_THEN_ACT, jev=_blocked_script()) as world:
        async with executor_graph([RETRIEVE, START, JOIN, GUIDE, JOIN_AGAIN, "Done."]) as graph:
            await _drive(graph, world)

        assert [card["kind"] for card in world.cards() if card["kind"] == "handoff"] == []
        assert await get_conversation_pending_handoff(CONVERSATION) is None


async def test_the_join_keeps_its_claim_on_the_result_while_the_executor_answers() -> None:
    """Dropping the lease on the way out would tell the worker nobody is joined, and it would narrate the run to the user behind the executor's back."""
    from app.services.browser.jobs import joiner_lease_held

    held: list[bool] = []
    guide_and_check = call("guide_browser_task", {"instruction": "click Search"}, "g1")

    async with browser_job_world(STREAM, steps=BLOCK_THEN_ACT, jev=_blocked_script()) as world:
        async with executor_graph(
            [RETRIEVE, START, JOIN, guide_and_check, JOIN_AGAIN, "Done."]
        ) as graph:
            run_task = asyncio.create_task(
                run_graph(
                    graph,
                    "book me a table",
                    thread_id=CONVERSATION,
                    user_id=USER,
                    **_configurable(),
                )
            )
            for _ in range(500):
                if world.browser.guidance_reasons:
                    held.append(await joiner_lease_held(world.enqueued[0].job_id))
                    break
                await asyncio.sleep(0.01)
            await run_task
            await world.settle()

    assert held == [True]


async def test_a_blocked_run_with_no_executor_joined_ends_blocked_as_before() -> None:
    """Asking when nobody is listening would stall the run for the whole guidance timeout and answer nothing."""
    from app.constants.browser import BROWSER_RUN_BLOCKED_SUMMARY

    steps = [ScriptedStep(actions=[], decide=True)]
    script = JevScript(decisions=[("BLOCKED", None)], texts=[{"text": "nothing here"}])

    async with browser_job_world(
        STREAM, steps=steps, jev=script, successful=False, summary=BROWSER_RUN_BLOCKED_SUMMARY
    ) as world:
        async with executor_graph([RETRIEVE, START, "I've started on it."]) as graph:
            await _drive(graph, world)

    assert world.browser.guidance_reasons == []
    assert _step_actions(world) == [
        {"done": {"text": BROWSER_RUN_BLOCKED_SUMMARY, "success": False}}
    ]
    results = [card for card in world.cards() if card["kind"] == "result"]
    assert [card["status"] for card in results] == [BrowserSessionStatus.FAILED.value]
    assert results[0]["success"] is False


def _step_actions(world: JobWorld) -> list[dict[str, Any]]:
    """Return the action each step card carries, which is what the policy actually decided."""
    return [
        {action["name"]: action["inputs"] for action in card["actions"]}
        for card in world.cards()
        if card["kind"] == "step" and card["actions"]
    ]


async def test_the_fourth_blocked_step_ends_the_run_instead_of_asking_again() -> None:
    """Without a budget a run that cannot be unstuck bounces off the executor forever, burning a model call each time."""
    from app.constants.browser import BROWSER_AGENT_GUIDANCE_MAX, BROWSER_RUN_BLOCKED_SUMMARY

    rounds = BROWSER_AGENT_GUIDANCE_MAX
    guides = [
        call("guide_browser_task", {"instruction": f"try route {n}"}, f"g{n}")
        for n in range(1, rounds + 1)
    ]
    joins = [call("wait_for_browser_task", {}, f"w{n}") for n in range(2, rounds + 2)]
    # Every guidance is followed by one real attempt: BLOCKED is off the table on
    # the step right after an answer, so a run can only give up again after trying.
    script = JevScript(
        decisions=[("BLOCKED", None), ("CLICK", "1")] * rounds + [("BLOCKED", None)],
        texts=[{"text": "still stuck"}] * rounds,
    )
    steps = [ScriptedStep(actions=[], decide=True, await_joiner=True)] + [
        ScriptedStep(actions=[], decide=True) for _ in range(rounds * 2)
    ]
    plan: list[Any] = [RETRIEVE, START, JOIN]
    for guide, join in zip(guides, joins, strict=True):
        plan += [guide, join]
    plan.append("Could not do it.")

    async with browser_job_world(
        STREAM, steps=steps, jev=script, successful=False, summary=BROWSER_RUN_BLOCKED_SUMMARY
    ) as world:
        async with executor_graph(plan) as graph:
            run = await _drive(graph, world)

    assert len(world.browser.guidance_reasons) == BROWSER_AGENT_GUIDANCE_MAX
    assert _step_actions(world)[-1] == {
        "done": {"text": BROWSER_RUN_BLOCKED_SUMMARY, "success": False}
    }
    assert "DID NOT COMPLETE" in ((run.results_from("tools") or [])[-1])
    assert await get_conversation_slot(CONVERSATION) is None


async def test_giving_up_ends_the_run_failed_and_frees_the_conversation() -> None:
    """An executor that honestly cannot answer must end the run, not leave it waiting out a two-minute timeout on a browser nobody is watching."""
    give_up = call(
        "guide_browser_task",
        {"give_up": True, "reason": "the site needs an account the user does not have"},
        "g1",
    )
    async with browser_job_world(STREAM, steps=BLOCK_THEN_ACT, jev=_blocked_script()) as world:
        async with executor_graph([RETRIEVE, START, JOIN, give_up, JOIN_AGAIN, "Sorry."]) as graph:
            run = await _drive(graph, world)

    results = [card for card in world.cards() if card["kind"] == "result"]
    assert [card["status"] for card in results] == [BrowserSessionStatus.FAILED.value]
    assert "the site needs an account the user does not have" in results[0]["summary"]
    assert "DID NOT COMPLETE" in ((run.results_from("tools") or [])[-1])
    assert await get_conversation_slot(CONVERSATION) is None
