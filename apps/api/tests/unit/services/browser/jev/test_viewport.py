"""Reading on-screen truth from the page itself, one CDP evaluate per step."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TypedDict, cast
from unittest.mock import MagicMock

from browser_use.browser.session import BrowserSession
from browser_use.dom.views import EnhancedDOMTreeNode
import pytest

from app.constants.browser import JEV_PAGE_TEXT_MAX_CHARS
from app.constants.log_tags import LogTag
from app.services.browser.jev import viewport as viewport_mod
from app.services.browser.jev.viewport import ViewportBox, ViewportRead, read_viewport

pytestmark = pytest.mark.unit


def _node(
    xpath: str,
    *,
    parent: EnhancedDOMTreeNode | None = None,
    node_name: str = "A",
    backend_node_id: int | None = None,
) -> EnhancedDOMTreeNode:
    return cast(
        EnhancedDOMTreeNode,
        SimpleNamespace(
            xpath=xpath,
            parent_node=parent,
            node_name=node_name,
            frame_id=None,
            backend_node_id=backend_node_id,
        ),
    )


class _Call(TypedDict):
    params: dict[str, object]
    session_id: str


def _browser(result: object, seen: list[_Call] | None = None) -> BrowserSession:
    class _Runtime:
        async def evaluate(self, params, session_id):
            if seen is not None:
                seen.append(_Call(params=params, session_id=session_id))
            if isinstance(result, Exception):
                raise result
            return result

    session = SimpleNamespace(
        session_id="sess", cdp_client=SimpleNamespace(send=SimpleNamespace(Runtime=_Runtime()))
    )

    async def get_or_create_cdp_session():
        return session

    return cast(
        BrowserSession, SimpleNamespace(get_or_create_cdp_session=get_or_create_cdp_session)
    )


def _pairs_sent(seen: list[_Call]) -> list[list[object]]:
    expression = str(seen[0]["params"]["expression"])
    payload = expression[expression.rindex("([") + 1 : expression.rindex("])") + 1]
    return json.loads(payload)


async def test_every_index_and_xpath_pair_reaches_the_page_and_comes_back_as_a_box() -> None:
    seen: list[_Call] = []
    browser = _browser(
        {
            "result": {
                "value": {
                    "7": {"on_screen": True, "cx": 0.25, "cy": 0.5},
                    "9": {"on_screen": False, "cx": 1.4, "cy": 0.1},
                }
            }
        },
        seen,
    )
    selector_map = {7: _node("html/body/a"), 9: _node("html/body/div/button")}

    boxes = (await read_viewport(browser, selector_map)).boxes

    assert _pairs_sent(seen) == [[7, "html/body/a"], [9, "html/body/div/button"]]
    assert seen[0]["session_id"] == "sess"
    assert seen[0]["params"]["returnByValue"] is True
    assert boxes == {
        7: ViewportBox(on_screen=True, cx=0.25, cy=0.5),
        9: ViewportBox(on_screen=False, cx=1.4, cy=0.1),
    }


async def test_an_xpath_that_resolved_nothing_is_simply_absent() -> None:
    browser = _browser({"result": {"value": {"7": {"on_screen": True, "cx": 0.1, "cy": 0.2}}}})

    boxes = (await read_viewport(browser, {7: _node("html/body/a"), 9: _node("html/body/b")})).boxes

    assert set(boxes) == {7}


async def test_a_cdp_failure_degrades_to_an_empty_map_and_warns(monkeypatch) -> None:
    logger = MagicMock()
    monkeypatch.setattr(viewport_mod, "log", logger)

    screen = await read_viewport(_browser(ConnectionError("gone")), {1: _node("html/body/a")})

    assert screen == ViewportRead()
    logger.warning.assert_any_call(
        f"{LogTag.BROWSER} Jev viewport read failed", error_type="ConnectionError"
    )


async def test_a_javascript_exception_degrades_to_an_empty_map_and_warns(monkeypatch) -> None:
    logger = MagicMock()
    monkeypatch.setattr(viewport_mod, "log", logger)

    browser = _browser({"exceptionDetails": {"text": "boom"}, "result": {}})

    assert (await read_viewport(browser, {1: _node("html/body/a")})).boxes == {}
    logger.warning.assert_any_call(
        f"{LogTag.BROWSER} Jev viewport read raised in the page", error_type="JSError"
    )


async def test_nodes_inside_an_iframe_are_never_asked_about_and_are_counted() -> None:
    """Browser-Use's xpath stops at the iframe, so it cannot be resolved from the top document."""
    logger = MagicMock()
    seen: list[_Call] = []
    browser = _browser({"result": {"value": {}}}, seen)
    frame = _node("html/body/iframe", node_name="IFRAME")
    selector_map = {1: _node("html/body/a"), 2: _node("div/button", parent=frame)}

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(viewport_mod, "log", logger)
        boxes = (await read_viewport(browser, selector_map)).boxes

    assert boxes == {}
    assert _pairs_sent(seen) == [[1, "html/body/a"]]
    logger.debug.assert_called_once()


async def test_no_resolvable_node_skips_the_element_measure_entirely() -> None:
    """Nothing to measure still reads the screen's text, which needs no element."""
    seen: list[_Call] = []
    browser = _browser({"result": {"value": ""}}, seen)

    assert (await read_viewport(browser, {})).boxes == {}
    assert [call["params"]["expression"] for call in seen] == [
        f"({viewport_mod._TEXT_JS})({JEV_PAGE_TEXT_MAX_CHARS})"
    ]


class _FallbackClient:
    """A CDP engine whose xpaths resolve nothing, so only backend node ids work."""

    def __init__(self) -> None:
        self.resolved: list[int] = []
        self.batched: list[list[str]] = []

        class _Runtime:
            @staticmethod
            async def evaluate(params, session_id):
                return {"result": {"value": {}}}

            @staticmethod
            async def callFunctionOn(params, session_id):
                ids = [argument["objectId"] for argument in params["arguments"]]
                self.batched.append(ids)
                return {
                    "result": {
                        "value": [
                            {"on_screen": object_id == "obj-11", "cx": 0.5, "cy": 0.25}
                            for object_id in ids
                        ]
                    }
                }

        class _DOM:
            @staticmethod
            async def resolveNode(params, session_id):
                self.resolved.append(params["backendNodeId"])
                return {"object": {"objectId": f"obj-{params['backendNodeId']}"}}

        self.send = SimpleNamespace(Runtime=_Runtime(), DOM=_DOM())


async def test_an_engine_without_usable_xpaths_is_measured_node_by_node() -> None:
    """Some engines serialise no parent chain, so every xpath collapses to a bare tag name."""
    client = _FallbackClient()
    session = SimpleNamespace(session_id="sess", cdp_client=client)

    async def get_or_create_cdp_session():
        return session

    browser = cast(
        BrowserSession, SimpleNamespace(get_or_create_cdp_session=get_or_create_cdp_session)
    )
    selector_map = {
        3: _node("a", backend_node_id=11),
        4: _node("a", backend_node_id=12),
    }

    boxes = (await read_viewport(browser, selector_map)).boxes

    assert sorted(client.resolved) == [11, 12]
    assert client.batched == [["obj-11", "obj-12"]]
    assert boxes == {
        3: ViewportBox(on_screen=True, cx=0.5, cy=0.25),
        4: ViewportBox(on_screen=False, cx=0.5, cy=0.25),
    }


class _ScreenClient:
    """A CDP engine that answers the element measure and the viewport-text read."""

    def __init__(self, text: object = "Visible line\nSecond line") -> None:
        self.text = text
        self.expressions: list[str] = []

        class _Runtime:
            @staticmethod
            async def evaluate(params, session_id):
                self.expressions.append(params["expression"])
                if "createTreeWalker" in params["expression"]:
                    if isinstance(self.text, Exception):
                        raise self.text
                    return {"result": {"value": self.text}}
                return {"result": {"value": {"7": {"on_screen": True, "cx": 0.5, "cy": 0.5}}}}

        self.send = SimpleNamespace(Runtime=_Runtime())


def _screen_browser(client: _ScreenClient) -> BrowserSession:
    session = SimpleNamespace(session_id="sess", cdp_client=client)

    async def get_or_create_cdp_session():
        return session

    return cast(
        BrowserSession, SimpleNamespace(get_or_create_cdp_session=get_or_create_cdp_session)
    )


async def test_the_screens_own_text_comes_back_with_the_boxes() -> None:
    client = _ScreenClient()

    screen = await read_viewport(_screen_browser(client), {7: _node("html/body/a")})

    assert screen.text == "Visible line\nSecond line"
    assert screen.boxes == {7: ViewportBox(on_screen=True, cx=0.5, cy=0.5)}


async def test_the_viewport_text_is_capped(monkeypatch) -> None:
    monkeypatch.setattr(viewport_mod, "JEV_PAGE_TEXT_MAX_CHARS", 7)

    screen = await read_viewport(_screen_browser(_ScreenClient()), {7: _node("html/body/a")})

    assert screen.text == "Visible"


async def test_a_text_read_that_fails_leaves_the_text_unknown(monkeypatch) -> None:
    logger = MagicMock()
    monkeypatch.setattr(viewport_mod, "log", logger)
    client = _ScreenClient(text=ConnectionError("gone"))

    screen = await read_viewport(_screen_browser(client), {7: _node("html/body/a")})

    assert screen.text is None
    # The element table survives a text failure; only the text is lost.
    assert screen.boxes == {7: ViewportBox(on_screen=True, cx=0.5, cy=0.5)}
    logger.warning.assert_called_once()
