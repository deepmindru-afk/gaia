"""Clicks must land on the element's real rect, through the one dispatch Obscura honours.

Browser-Use dispatches Input mouse events at the point it computes for an
element. On the Obscura host those events are accepted and dropped -- nothing
reaches the page -- so the patch clicks the element handle in JavaScript
instead and reports the centre of the rect the page itself measured.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from browser_use.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog
import pytest

import app.patches.browser_use_click_patch as patch_module

pytestmark = pytest.mark.unit

# What the page measures for the search button, and what the snapshot fabricates
# for it -- a click computed from the snapshot lands a whole page down.
_REAL_RECT = {"x": 670.0, "y": 17.0, "width": 70.0, "height": 32.0}
_REAL_CENTRE = {"click_x": 705.0, "click_y": 33.0}


class _Input:
    """The Input domain is dead on Obscura; touching it at all is the bug."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the patched click must not call Input.{name}")


class _FakeCdp:
    def __init__(self, *, rect: dict[str, float] | None = _REAL_RECT) -> None:
        self.rect = rect
        self.scrolled: list[int] = []
        self.resolved: list[int] = []
        self.functions: list[str] = []

        class _DOM:
            @staticmethod
            async def scrollIntoViewIfNeeded(params: dict[str, Any], session_id: str) -> dict:
                self.scrolled.append(params["backendNodeId"])
                return {}

            @staticmethod
            async def resolveNode(params: dict[str, Any], session_id: str) -> dict:
                self.resolved.append(params["backendNodeId"])
                return {"object": {"objectId": f"obj-{params['backendNodeId']}"}}

        class _Runtime:
            @staticmethod
            async def callFunctionOn(params: dict[str, Any], session_id: str) -> dict:
                self.functions.append(params["functionDeclaration"])
                if self.rect is None:
                    return {"result": {"value": None}}
                return {
                    "result": {
                        "value": {
                            "click_x": self.rect["x"] + self.rect["width"] / 2,
                            "click_y": self.rect["y"] + self.rect["height"] / 2,
                        }
                    }
                }

        self.send = SimpleNamespace(DOM=_DOM(), Runtime=_Runtime(), Input=_Input())


def _watchdog(cdp: _FakeCdp) -> SimpleNamespace:
    session = SimpleNamespace(session_id="sess", cdp_client=cdp)

    async def cdp_client_for_node(node: object) -> SimpleNamespace:
        return session

    return SimpleNamespace(browser_session=SimpleNamespace(cdp_client_for_node=cdp_client_for_node))


def _node(tag: str = "button", attributes: dict[str, str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        tag_name=tag,
        attributes=attributes or {},
        backend_node_id=11,
        # The snapshot's own geometry, fabricated on this engine.
        absolute_position=SimpleNamespace(x=0.0, y=4482.0, width=1280.0, height=18.0),
    )


async def test_the_click_point_is_the_centre_of_the_rect_the_page_measured() -> None:
    cdp = _FakeCdp()

    point = await patch_module._click_element_node_impl(_watchdog(cdp), _node())

    assert point == _REAL_CENTRE
    assert cdp.resolved == [11]
    assert "getBoundingClientRect" in cdp.functions[0]


async def test_the_element_is_clicked_in_the_page_and_no_mouse_event_is_dispatched() -> None:
    cdp = _FakeCdp()

    await patch_module._click_element_node_impl(_watchdog(cdp), _node())

    # _Input raises on any attribute, so reaching here proves nothing was dispatched.
    assert ".click()" in cdp.functions[0]


async def test_browser_uses_own_scroll_into_view_still_runs_first() -> None:
    cdp = _FakeCdp()

    await patch_module._click_element_node_impl(_watchdog(cdp), _node())

    assert cdp.scrolled == [11]


async def test_a_select_element_keeps_browser_uses_own_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[object] = []

    async def original(watchdog: object, node: object) -> dict[str, str]:
        seen.append(node)
        return {"validation_error": "Cannot click on <select> elements."}

    monkeypatch.setattr(patch_module, "_original_click_element_node_impl", original)
    cdp = _FakeCdp()

    result = await patch_module._click_element_node_impl(_watchdog(cdp), _node("select"))

    assert result == {"validation_error": "Cannot click on <select> elements."}
    assert len(seen) == 1
    assert cdp.resolved == []


async def test_a_file_input_keeps_browser_uses_own_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def original(watchdog: object, node: object) -> dict[str, str]:
        return {"validation_error": "File uploads must be handled using upload_file_to_element."}

    monkeypatch.setattr(patch_module, "_original_click_element_node_impl", original)
    cdp = _FakeCdp()

    result = await patch_module._click_element_node_impl(
        _watchdog(cdp), _node("input", {"type": "file"})
    )

    assert result == {
        "validation_error": "File uploads must be handled using upload_file_to_element."
    }
    assert cdp.resolved == []


async def test_an_element_the_page_cannot_measure_falls_back_to_browser_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero-sized or detached node has no rect to click; Browser-Use's own path reports it."""
    called: list[object] = []

    async def original(watchdog: object, node: object) -> None:
        called.append(node)

    monkeypatch.setattr(patch_module, "_original_click_element_node_impl", original)

    result = await patch_module._click_element_node_impl(_watchdog(_FakeCdp(rect=None)), _node())

    assert result is None
    assert len(called) == 1


async def test_apply_rebinds_the_watchdog_method() -> None:
    assert DefaultActionWatchdog._click_element_node_impl is patch_module._click_element_node_impl
