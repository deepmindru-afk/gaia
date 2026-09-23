"""The step frames a browser run hands the user: numbered by what was emitted, never silent on an errored step.

Browser-Use's own counter advances for steps that never reach a frame (an action
that errors, an observation the watchdog times out), so the user saw photos 2, 3,
4 and then 100s of nothing until 7.
"""

from collections.abc import Awaitable
from types import SimpleNamespace
from typing import Any

import pytest

from app.services.browser.agent_run import STEP_ERROR_CAPTION, BrowserAgentRun
from app.services.browser.run_contract import BrowserRunConfig, RunHooks, StepFrame

CONFIG = BrowserRunConfig(
    max_steps=20,
    max_actions_per_step=2,
    task_timeout_seconds=300,
    step_timeout_seconds=30,
    handoff_timeout_seconds=60,
    stream_screenshots=True,
    solve_captcha=False,
)


class _Action:
    """One of Browser-Use's own action models, as agent_output.action holds them."""

    def __init__(self, name: str, params: dict[str, Any]) -> None:
        self._dump = {name: params}

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        return self._dump


class _Result:
    """One executed action's ActionResult, errored or not."""

    def __init__(self, *, error: str | None = None, content: str = "") -> None:
        self.error = error
        self.extracted_content = content
        self.long_term_memory = None


def _page(url: str = "https://example.test/page") -> Any:
    return SimpleNamespace(
        dom_state=SimpleNamespace(selector_map={}),
        url=url,
        title="Example",
        screenshot="c2hvdA==",
    )


def _output(name: str, params: dict[str, Any]) -> Any:
    return SimpleNamespace(action=[_Action(name, params)])


async def _never_stop() -> bool:
    return False


async def _no_takeover(reason: str, category: str) -> str | None:
    return None


class _Harness:
    """A run wired to record what it emitted, plus the two Browser-Use callbacks."""

    def __init__(self) -> None:
        self.frames: list[StepFrame] = []
        self.outputs: list[tuple[int, list[str]]] = []
        self.run = BrowserAgentRun(
            session=SimpleNamespace(cdp_url="ws://browser.test/cdp", session_id="sess-1"),
            llm=None,
            config=CONFIG,
            hooks=RunHooks(
                step=self.frames.append,
                takeover=_no_takeover,
                should_stop=_never_stop,
                action_results=self._record_outputs,
            ),
            step_timeout=30.0,
        )

    async def _record_outputs(self, step_index: int, outputs: list[Any]) -> None:
        self.outputs.append((step_index, [out.output for out in outputs]))

    async def step(self, n_steps: int, name: str = "click", **params: Any) -> None:
        await self.run._on_step(_page(), _output(name, params), n_steps)

    def end(self, *results: _Result) -> Awaitable[None]:
        return self.run._on_step_end(SimpleNamespace(state=SimpleNamespace(last_result=results)))


@pytest.fixture
def harness() -> _Harness:
    return _Harness()


@pytest.mark.unit
class TestFrameNumbering:
    async def test_frames_are_numbered_by_what_the_user_saw_not_by_browser_uses_counter(
        self, harness: _Harness
    ) -> None:
        await harness.step(2, index=4)
        await harness.end(_Result(content="clicked"))
        await harness.end(_Result(error="Step 3 timed out after 30 seconds"))
        await harness.step(7, index=9)
        await harness.end(_Result(content="clicked"))

        assert [frame.index for frame in harness.frames] == [1, 2, 3]

    async def test_a_step_that_never_reached_a_frame_still_gets_one_saying_so(
        self, harness: _Harness
    ) -> None:
        await harness.step(2, index=4)
        await harness.end(_Result(content="clicked"))
        await harness.end(_Result(error="Step 3 timed out after 30 seconds"))

        assert [frame.goal for frame in harness.frames][-1] == STEP_ERROR_CAPTION
        assert harness.frames[-1].actions == []
        assert harness.frames[-1].raw_screenshot is None

    async def test_a_step_that_errored_after_its_frame_is_not_framed_twice(
        self, harness: _Harness
    ) -> None:
        await harness.step(2, index=4)
        await harness.end(_Result(error="element is not clickable"))

        assert len(harness.frames) == 1
        assert harness.frames[0].goal != STEP_ERROR_CAPTION

    async def test_an_action_result_lands_on_the_frame_number_the_user_can_see(
        self, harness: _Harness
    ) -> None:
        await harness.step(2, index=4)
        await harness.end(_Result(content="clicked"))
        await harness.end(_Result(error="Step 3 timed out after 30 seconds"))
        await harness.step(7, index=9)
        await harness.end(_Result(content="confirmed"))

        assert harness.outputs[0] == (1, ["clicked"])
        assert harness.outputs[-1] == (3, ["confirmed"])


@pytest.mark.unit
class TestStepCaption:
    async def test_a_finishing_step_is_named_after_the_part_it_finished(
        self, harness: _Harness
    ) -> None:
        """Regression: the only step of a one-step run read "Step 1 · Finished"."""
        output = SimpleNamespace(
            next_goal="Read the top story on news.ycombinator.com",
            action=[_Action("done", {"text": "The top story is X.", "success": True})],
        )

        await harness.run._on_step(_page(), output, 1)

        assert harness.frames[-1].goal == "Read the top story on news.ycombinator.com"

    async def test_a_finish_that_did_not_achieve_the_goal_says_so_not_the_part(
        self, harness: _Harness
    ) -> None:
        output = SimpleNamespace(
            next_goal="Buy the item on shop.test",
            action=[_Action("done", {"text": "There is no Buy button.", "success": False})],
        )

        await harness.run._on_step(_page(), output, 1)

        assert harness.frames[-1].goal == "Could not find a way forward on this page"
