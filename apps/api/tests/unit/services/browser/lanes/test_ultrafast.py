"""The ultrafast lane, driven through the real loop.

Nothing under test is mocked: the real :class:`JevUltrafastAgent`, the real
gateway client and the real text helper run — only the CDP browser (the ultrafast
suite's ``FakeBrowser``) and the network (``httpx.MockTransport``) are faked. So a
step snapshot here is built from a history entry the loop really appended.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.constants.browser import (
    BrowserAgentLoop,
    BrowserEventKind,
    BrowserSessionStatus,
    HandoffStatus,
    SensitiveCategory,
)
from app.schemas.browser import HandoffOutcome
from app.services.browser.exceptions import BrowserHandoffCancelled, BrowserUnavailableError
from app.services.browser.jev.ultrafast import agent as agent_mod, clients as clients_mod
from app.services.browser.lanes import (
    BrowserRunConfig,
    LaneHooks,
    UltrafastLane,
)
from app.services.browser.runner import BrowserRunnerCallbacks, BrowserTaskRunner
from app.services.browser.session import BrowserHostSession
from tests.unit.services.browser.jev.ultrafast.conftest import (
    FakeBrowser,
    distribution,
    make_page,
)

pytestmark = pytest.mark.unit

JEV_MODEL = "~typesafe/jev-latest"
TEXT_MODEL = "inception/mercury-2.5"


def _session() -> BrowserHostSession:
    return BrowserHostSession(
        session_id="s1",
        cdp_url="ws://x",  # NOSONAR
        live_view_url="http://v",  # NOSONAR
        context_id="ctx-1",
    )


def _config(**overrides: Any) -> BrowserRunConfig:
    fields: dict[str, Any] = {
        "max_steps": 10,
        "max_actions_per_step": 5,
        "task_timeout_seconds": 30,
        "step_timeout_seconds": 5,
        "handoff_timeout_seconds": 0,
        "stream_screenshots": True,
        "use_vision": False,
        "solve_captcha": True,
        "agent_loop": BrowserAgentLoop.JEV_ULTRAFAST,
    }
    fields.update(overrides)
    return BrowserRunConfig(**fields)


def decisions(*operations: str, usage: dict[str, int] | None = None) -> Any:
    """One canned Jev answer per call, targeting the page's "Go" button."""
    answers = iter(operations)

    def responder(body: dict[str, Any]) -> dict[str, Any]:
        operation = next(answers)
        result = {"operation": distribution(body["questions"]["operation"]["criteria"], operation)}
        if operation == "CLICK":
            result["click_target"] = distribution(
                body["questions"]["click_target"]["criteria"], "2"
            )
        if operation == "TYPE_TEXT":
            result["type_text_target"] = distribution(
                body["questions"]["type_text_target"]["criteria"], "1"
            )
        return {"answers": result, "usage": usage or {"inputTokens": 120, "outputTokens": 4}}

    return responder


def text(value: str = "dune", usage: dict[str, int] | None = None) -> Any:
    def responder(_body: dict[str, Any]) -> dict[str, Any]:
        return {
            "choices": [{"message": {"content": f'{{"text":"{value}"}}'}}],
            "usage": usage or {"prompt_tokens": 40, "completion_tokens": 3},
        }

    return responder


class _RecordingBrowser(FakeBrowser):
    """The ultrafast suite's CDP seam, plus what each observation was asked for."""

    def __init__(self, page) -> None:
        super().__init__(page)
        self.screenshots: list[bool] = []

    async def observe(self, *, screenshot: bool = False):
        self.screenshots.append(screenshot)
        return await super().observe(screenshot=screenshot)


async def _unused_handoff(request) -> HandoffOutcome:
    return HandoffOutcome(status=HandoffStatus.COMPLETED)


async def _never_cancelled() -> bool:
    return False


@pytest.fixture
def wire(monkeypatch):
    """Attach the loop to a fake page and both models to canned HTTP."""
    from tests.unit.services.browser.jev.ultrafast.conftest import make_gateway, make_text_helper

    page = make_page()
    browser = _RecordingBrowser(page)

    class _Attach:
        @staticmethod
        async def attach(cdp_url: str, *, start_url: str | None = None) -> _RecordingBrowser:
            return browser

    monkeypatch.setattr(agent_mod, "UltrafastBrowser", _Attach)
    monkeypatch.setattr(clients_mod.settings, "OPENROUTER_API_KEY", "test-key")

    def _install(decision_responder: Any, text_responder: Any) -> None:
        client, _ = make_gateway(decision_responder)
        helper, _ = make_text_helper(text_responder)
        monkeypatch.setattr(clients_mod, "JevGatewayClient", lambda **_kwargs: client)
        monkeypatch.setattr(clients_mod, "JevTextHelper", lambda **_kwargs: helper)

    return page, browser, _install


def _lane(hooks: LaneHooks, **config: Any) -> UltrafastLane:
    return UltrafastLane(session=_session(), config=_config(**config), hooks=hooks, step_timeout=5)


def _hooks(*, takeover: Any = None, should_stop: Any = None) -> tuple[list, LaneHooks]:
    frames: list = []

    async def _never() -> bool:
        return False

    async def _no_takeover(reason: str, category: str) -> str:
        raise AssertionError("the run was not supposed to need the human")

    return frames, LaneHooks(
        step=frames.append,
        takeover=takeover or _no_takeover,
        should_stop=should_stop or _never,
    )


# ---------------------------------------------------------------------------
# per-step progress
# ---------------------------------------------------------------------------


async def test_every_executed_step_becomes_a_frame_naming_what_jev_did(wire) -> None:
    page, browser, install = wire
    install(decisions("TYPE_TEXT", "CLICK", "DONE"), text("dune"))
    frames, hooks = _hooks()

    outcome = await _lane(hooks).execute("Find a book")

    assert outcome.success is True
    # The canned helper answers every call with the same word, closing summary included.
    assert outcome.summary == "dune"
    assert [(f.index, f.goal) for f in frames] == [
        (1, 'Typing "dune" into "Search"'),
        (2, 'Clicking "Go"'),
    ]
    # The actions are what really executed, in the loop's own order.
    assert browser.acted == [("e1", "dune"), ("e3", None)]
    assert [(a.name, a.inputs, a.target) for f in frames for a in f.actions] == [
        ("input", {"text": "dune"}, "Search"),
        ("click", {}, "Go"),
    ]


async def test_a_navigation_frame_names_the_site_it_opened(wire) -> None:
    """A task names its site in prose, so the first step is usually the navigation."""
    _, browser, install = wire
    install(decisions("NAVIGATE", "DONE"), text("https://en.wikipedia.org/"))
    frames, hooks = _hooks()

    outcome = await _lane(hooks).execute("Look up Ada Lovelace on Wikipedia")

    assert outcome.success is True
    assert browser.navigated == ["https://en.wikipedia.org/"]
    assert [(f.index, f.goal) for f in frames] == [(1, "Opening en.wikipedia.org")]
    assert [(a.name, a.inputs, a.target) for f in frames for a in f.actions] == [
        ("navigate", {"url": "https://en.wikipedia.org/"}, None)
    ]


async def test_a_frame_carries_the_observed_page_and_its_screenshot(wire) -> None:
    page, _, install = wire
    page["screenshot"] = "am9lZw=="
    install(decisions("CLICK", "DONE"), text())
    frames, hooks = _hooks()

    await _lane(hooks).execute("Find a book")

    [frame] = frames
    assert frame.url == page["url"]
    assert frame.title == page["title"]
    assert frame.raw_screenshot == "am9lZw=="
    # The loop captures JPEG, so the frame must say so — not inherit Browser-Use's PNG.
    assert frame.screenshot_media_type == "image/jpeg"


@pytest.mark.parametrize("streaming", [True, False])
async def test_screenshots_follow_the_runs_streaming_setting(wire, streaming: bool) -> None:
    """A frame the card will never show is a screenshot the loop should not take."""
    _, browser, install = wire
    install(decisions("CLICK", "DONE"), text())
    _, hooks = _hooks()

    await _lane(hooks, stream_screenshots=streaming).execute("Find a book")

    assert browser.screenshots and all(taken is streaming for taken in browser.screenshots)


# ---------------------------------------------------------------------------
# the human
# ---------------------------------------------------------------------------


async def test_a_takeover_routes_to_the_runners_handoff_and_the_run_resumes(wire) -> None:
    _, browser, install = wire
    install(
        decisions("REQUEST_HUMAN", "CLICK", "DONE"),
        # The takeover reason is generated under the same helper as a typed value.
        lambda _body: {
            "choices": [
                {"message": {"content": '{"text":"Enter your password","category":"credentials"}'}}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
    )
    handoffs: list[tuple[str, str]] = []

    async def takeover(reason: str, category: str) -> str:
        handoffs.append((reason, category))
        return "continue"

    frames, hooks = _hooks(takeover=takeover)

    outcome = await _lane(hooks).execute("Log in and search")

    assert handoffs == [("Enter your password", SensitiveCategory.CREDENTIALS.value)]
    assert outcome.success is True
    # The human step is a step of its own, and the run carried on after it.
    assert [f.goal for f in frames] == ["Handing this step to you", 'Clicking "Go"']
    assert browser.acted == [("e3", None)]


async def test_a_cancelled_takeover_ends_the_run_without_acting(wire) -> None:
    _, browser, install = wire
    install(
        decisions("REQUEST_HUMAN", "CLICK", "DONE"),
        lambda _body: {
            "choices": [{"message": {"content": '{"text":"Pay now","category":"payment"}'}}],
            "usage": {},
        },
    )

    async def takeover(reason: str, category: str) -> str:
        raise BrowserHandoffCancelled("cancelled")

    _, hooks = _hooks(takeover=takeover)

    outcome = await _lane(hooks).execute("Buy the book")

    assert outcome.success is False
    assert browser.acted == []


async def test_captcha_is_only_offered_when_the_run_can_service_it(wire) -> None:
    """An operation with no handler is never put in front of Jev."""
    _, _, install = wire
    offered: list[list[str]] = []

    def responder(body: dict[str, Any]) -> dict[str, Any]:
        offered.append(list(body["questions"]["operation"]["criteria"]))
        return {
            "answers": {
                "operation": distribution(body["questions"]["operation"]["criteria"], "DONE")
            }
        }

    install(responder, text())
    _, hooks = _hooks()

    await _lane(hooks, solve_captcha=False).execute("Find a book")

    assert "SOLVE_CAPTCHA" not in offered[0]
    assert "REQUEST_HUMAN" in offered[0]


# ---------------------------------------------------------------------------
# cancellation
# ---------------------------------------------------------------------------


async def test_cancellation_stops_the_loop_between_steps(wire) -> None:
    _, browser, install = wire
    install(decisions("CLICK", "CLICK", "DONE"), text())
    cancel_after = iter([False, True, True])

    async def should_stop() -> bool:
        return next(cancel_after)

    frames, hooks = _hooks(should_stop=should_stop)

    outcome = await _lane(hooks).execute("Find a book")

    assert outcome.success is False
    assert outcome.summary == "Browser task stopped."
    # One step ran before the cancel landed; the second decision was never asked for.
    assert browser.acted == [("e3", None)]
    assert len(frames) == 1


async def test_a_cancelled_run_reports_cancelled_not_failed(wire) -> None:
    """End to end through the runner: a cancelled ultrafast run is CANCELLED."""
    _, _, install = wire
    install(decisions("CLICK", "DONE"), text())
    events: list = []

    async def emit(snapshot) -> None:
        events.append(snapshot)

    cancelled = iter([False, True, True, True])

    async def is_cancelled() -> bool:
        return next(cancelled)

    runner = BrowserTaskRunner(
        session=_session(),
        llm=None,
        callbacks=BrowserRunnerCallbacks(
            emit=emit,
            request_handoff=_unused_handoff,
            is_cancelled=is_cancelled,
        ),
        config=_config(),
    )
    result = await runner.run("Find a book")

    assert result.status == BrowserSessionStatus.CANCELLED
    assert result.success is False
    assert events[-1].kind == BrowserEventKind.RESULT


# ---------------------------------------------------------------------------
# metering
# ---------------------------------------------------------------------------


async def test_both_models_are_billed_under_their_real_names(wire) -> None:
    """Jev is billed per decision — including the terminal one that executes
    nothing — and the text helper per generated value, the closing summary included."""
    _, _, install = wire
    install(
        decisions("TYPE_TEXT", "DONE", usage={"inputTokens": 100, "outputTokens": 5}),
        text("dune", usage={"prompt_tokens": 40, "completion_tokens": 3}),
    )
    _, hooks = _hooks()

    outcome = await _lane(hooks).execute("Find a book")

    billed = {u.model_name: (u.input_tokens, u.output_tokens) for u in outcome.usage}
    # Two Jev decisions, and two helper calls: the typed value and the answer
    # written when Jev chose DONE.
    assert billed == {JEV_MODEL: (200, 10), TEXT_MODEL: (80, 6)}


async def test_the_runner_charges_every_model_the_lane_used(wire, monkeypatch) -> None:
    from unittest.mock import AsyncMock

    from app.services.browser import runner as runner_mod

    _, _, install = wire
    install(decisions("TYPE_TEXT", "DONE"), text("dune"))
    record = AsyncMock()
    monkeypatch.setattr(runner_mod, "record_llm_call", record)

    async def emit(_snapshot) -> None:
        return None

    runner = BrowserTaskRunner(
        session=_session(),
        llm=None,
        callbacks=BrowserRunnerCallbacks(
            emit=emit,
            request_handoff=_unused_handoff,
            is_cancelled=_never_cancelled,
        ),
        config=_config(),
        user_id="u1",
    )
    await runner.run("Find a book")

    charged = {call.kwargs["model_name"] for call in record.await_args_list}
    assert charged == {JEV_MODEL, TEXT_MODEL}
    assert all(call.kwargs["user_id"] == "u1" for call in record.await_args_list)


# ---------------------------------------------------------------------------
# credentials
# ---------------------------------------------------------------------------


async def test_the_lane_refuses_to_run_without_its_credential(monkeypatch) -> None:
    monkeypatch.setattr(clients_mod.settings, "OPENROUTER_API_KEY", None)
    _, hooks = _hooks()

    with pytest.raises(BrowserUnavailableError, match="OPENROUTER_API_KEY"):
        await _lane(hooks).execute("Find a book")


async def test_the_runs_answer_is_what_the_lane_reports_back(wire) -> None:
    """The assistant only sees the outcome, so a fixed "Completed the browser task."
    would throw away the answer the user actually asked for."""
    _, _, install = wire
    install(decisions("CLICK", "DONE"), text("The top story is Bend 2, posted by liam."))
    _, hooks = _hooks()

    outcome = await _lane(hooks).execute("What is the top story and who posted it?")

    assert outcome.success is True
    assert outcome.summary == "The top story is Bend 2, posted by liam."
