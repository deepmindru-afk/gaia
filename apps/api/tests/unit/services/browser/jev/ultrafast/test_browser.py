"""Freshness, target validation and execution, over a scripted CDP socket.

The websocket is faked; everything above it — the guards, the one-read
observation, the select-all-then-insert typing — is the real code.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Any

import pytest

from app.services.browser.jev.ultrafast.browser import (
    JevExecutionError,
    JevUltrafastNodeError,
    StalePage,
    UltrafastBrowser,
    fingerprint,
)
from tests.unit.services.browser.jev.ultrafast.conftest import make_page

pytestmark = pytest.mark.unit


class FakeCDP:
    """A scripted CDP socket: replies in order, records every call."""

    def __init__(self, *responses: Any) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.responses = list(responses)
        self.stopped = False

    async def send_raw(
        self, method: str, params: dict[str, Any] | None = None, session_id: str | None = None
    ) -> dict[str, Any]:
        self.calls.append((method, params or {}))
        reply = self.responses.pop(0) if self.responses else {}
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def stop(self) -> None:
        self.stopped = True

    @property
    def methods(self) -> list[str]:
        return [method for method, _ in self.calls]


def value(payload: Any) -> dict[str, Any]:
    return {"result": {"value": payload}}


def make_browser(*responses: Any) -> tuple[UltrafastBrowser, FakeCDP]:
    cdp = FakeCDP(*responses)
    return UltrafastBrowser(cdp, "session-1"), cdp  # type: ignore[arg-type]


def guard_reply(page: dict[str, Any], node: str) -> dict[str, Any]:
    return value([page["page_key"], page["guards"][node]])


# --------------------------------------------------------------------------
# observation
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_observation_is_one_browser_read() -> None:
    page = make_page()
    browser, cdp = make_browser(value(deepcopy(page)))

    observed = await browser.observe()

    assert observed["actions"] == page["actions"]
    assert observed["fingerprint"] == page["fingerprint"]
    assert cdp.methods == ["Runtime.evaluate"]


@pytest.mark.asyncio
async def test_no_screenshot_is_taken_unless_asked_for() -> None:
    page = make_page()
    browser, cdp = make_browser(value(deepcopy(page)), {"data": "jpeg-bytes"})

    await browser.observe()
    assert "Page.captureScreenshot" not in cdp.methods

    browser2, cdp2 = make_browser(value(deepcopy(page)), {"data": "jpeg-bytes"})
    observed = await browser2.observe(screenshot=True)
    assert observed["screenshot"] == "jpeg-bytes"
    assert cdp2.methods == ["Runtime.evaluate", "Page.captureScreenshot"]


@pytest.mark.asyncio
async def test_a_navigating_document_is_retried_and_then_refused(monkeypatch) -> None:
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    browser, cdp = make_browser(*[value(None)] * 10)

    with pytest.raises(StalePage, match="navigating"):
        await browser.observe()

    assert len(cdp.calls) == 10


@pytest.mark.asyncio
async def test_a_document_that_settles_is_observed(monkeypatch) -> None:
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    page = make_page()
    browser, _ = make_browser(value(None), value(None), value(deepcopy(page)))

    assert (await browser.observe())["url"] == page["url"]


# --------------------------------------------------------------------------
# freshness
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_click_is_fresh_only_while_its_target_and_document_are_unchanged() -> None:
    page = make_page()
    action = page["actions"][2]
    browser, _ = make_browser(guard_reply(page, "20"))
    assert await browser.fresh(page, action) is True

    moved, _ = make_browser(value([["other-page-key"], page["guards"]["20"]]))
    assert await moved.fresh(page, action) is False


@pytest.mark.asyncio
async def test_a_non_element_action_compares_the_whole_marker() -> None:
    page = make_page()
    browser, _ = make_browser(value(page["marker"]))
    assert await browser.fresh(page) is True

    changed, _ = make_browser(value(["something-else"]))
    assert await changed.fresh(page) is False


@pytest.mark.asyncio
async def test_a_node_that_is_not_an_observed_id_is_never_fresh() -> None:
    page = make_page()
    action = {**page["actions"][2], "node": "document.body"}
    browser, cdp = make_browser()

    assert await browser.fresh(page, action) is False
    assert cdp.calls == []


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_executor_refuses_a_stale_page_before_any_input() -> None:
    page = make_page()
    browser, cdp = make_browser(value([["other-page-key"], None]))

    with pytest.raises(StalePage):
        await browser.act(page["actions"][2], page)

    assert cdp.methods == ["Runtime.evaluate"]


@pytest.mark.asyncio
async def test_a_covered_or_disconnected_target_is_not_clicked() -> None:
    page = make_page()
    browser, cdp = make_browser(guard_reply(page, "20"), value(None))

    with pytest.raises(StalePage, match="covered"):
        await browser.act(page["actions"][2], page)

    assert "Input.dispatchMouseEvent" not in cdp.methods


@pytest.mark.asyncio
async def test_a_click_dispatches_press_and_release_at_the_resolved_point() -> None:
    page = make_page()
    browser, cdp = make_browser(guard_reply(page, "20"), value({"x": 12.0, "y": 34.0}), {}, {})

    result = await browser.act(page["actions"][2], page)

    assert result == {"executed": "e3"}
    assert cdp.methods == [
        "Runtime.evaluate",
        "Runtime.evaluate",
        "Input.dispatchMouseEvent",
        "Input.dispatchMouseEvent",
    ]
    assert [c[1]["type"] for c in cdp.calls[2:]] == ["mousePressed", "mouseReleased"]
    assert cdp.calls[2][1]["x"] == 12.0


@pytest.mark.asyncio
async def test_typing_focuses_and_selects_all_before_inserting() -> None:
    page = make_page()
    browser, cdp = make_browser(
        value(page["marker"]), value({"x": 1.0, "y": 2.0}), {}, {}, value(True), {}, {}, {}
    )

    await browser.act(page["actions"][0], page, text="Zurich")

    assert cdp.methods == [
        "Runtime.evaluate",
        "Runtime.evaluate",
        "Input.dispatchMouseEvent",
        "Input.dispatchMouseEvent",
        "Runtime.evaluate",
        "Input.dispatchKeyEvent",
        "Input.dispatchKeyEvent",
        "Input.insertText",
    ]
    assert cdp.calls[-3][1]["commands"] == ["selectAll"]
    assert cdp.calls[-1][1]["text"] == "Zurich"


@pytest.mark.asyncio
async def test_a_field_that_refuses_focus_is_never_typed_into() -> None:
    page = make_page()
    browser, cdp = make_browser(
        value(page["marker"]), value({"x": 1.0, "y": 2.0}), {}, {}, value(False)
    )

    with pytest.raises(JevExecutionError, match="did not take focus"):
        await browser.act(page["actions"][0], page, text="Zurich")

    assert "Input.insertText" not in cdp.methods


@pytest.mark.asyncio
async def test_a_wait_sleeps_without_dispatching_input(monkeypatch) -> None:
    page = make_page()
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    browser, cdp = make_browser(value(page["marker"]))

    await browser.act(page["actions"][3], page)

    assert cdp.methods == ["Runtime.evaluate"]
    assert browser.after_input is None


@pytest.mark.asyncio
async def test_a_scroll_dispatches_one_wheel_event() -> None:
    page = make_page()
    scroll = {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560}
    browser, cdp = make_browser(value(page["marker"]), {})

    await browser.act(scroll, page)

    assert cdp.methods == ["Runtime.evaluate", "Input.dispatchMouseEvent"]
    assert cdp.calls[1][1]["deltaY"] == 560


@pytest.mark.asyncio
async def test_a_non_integer_node_never_reaches_the_browser() -> None:
    page = make_page()
    action = {"id": "e9", "kind": "click", "label": "x", "node": "document.body"}
    browser, cdp = make_browser(value(page["marker"]))

    with pytest.raises(JevUltrafastNodeError):
        await browser._execute(action, None)

    assert cdp.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply", [{"exceptionDetails": {"text": "Execution context destroyed"}}, {"result": {}}]
)
async def test_an_interrupted_dropdown_is_never_retried_as_a_stale_page(reply) -> None:
    page = make_page()
    select = {"id": "e9", "kind": "select", "label": "Class → First", "node": 20, "value": "first"}
    browser, _ = make_browser(guard_reply(page, "20"), reply)

    with pytest.raises(JevExecutionError, match="Dropdown execution"):
        await browser.act(select, page)


@pytest.mark.asyncio
async def test_a_confirmed_dropdown_dispatches_no_mouse_input() -> None:
    page = make_page()
    select = {"id": "e9", "kind": "select", "label": "Class → First", "node": 20, "value": "first"}
    browser, cdp = make_browser(guard_reply(page, "20"), value({"x": 1.0, "y": 2.0}))

    assert await browser.act(select, page) == {"executed": "e9"}
    assert cdp.methods == ["Runtime.evaluate", "Runtime.evaluate"]


# --------------------------------------------------------------------------
# fingerprint
# --------------------------------------------------------------------------


def test_the_fingerprint_tracks_meaning_and_identity_not_screenshots() -> None:
    page = make_page()
    other = deepcopy(page)
    other["screenshot"] = "changed"

    assert fingerprint(page) == fingerprint(other)

    other["actions"][0]["node"] = 99
    assert fingerprint(page) != fingerprint(other)


def test_the_fingerprint_tracks_field_values() -> None:
    page = make_page()
    other = deepcopy(page)
    other["actions"][0]["value"] = "typed"

    assert fingerprint(page) != fingerprint(other)


async def _no_sleep(_seconds: float) -> None:
    return None
