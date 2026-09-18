"""Picking a dropdown option must not write option.selected, which Obscura ignores."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from browser_use.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog
import pytest

import app.patches.browser_use_select_patch as patch_module

pytestmark = pytest.mark.unit


class _Input:
    """A dropdown is set in JavaScript; reaching Input at all is the bug."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"selecting an option must not call Input.{name}")


class _FakeCdp:
    """A fake CDP client that runs the patch's JS against a tiny <select> model."""

    def __init__(
        self,
        options: list[tuple[str, str]],
        *,
        object_id: str | None = "obj-1",
        raises: bool = False,
        honour_selection: bool = True,
    ) -> None:
        self.options = options
        self.selected_index = 0
        self.calls: list[str] = []
        self.honour_selection = honour_selection
        outer = self

        class _DOM:
            @staticmethod
            async def resolveNode(params: dict[str, Any], session_id: str) -> dict[str, Any]:
                outer.calls.append("DOM.resolveNode")
                return {"object": {"objectId": object_id}} if object_id else {"object": {}}

        class _Runtime:
            @staticmethod
            async def callFunctionOn(params: dict[str, Any], session_id: str) -> dict[str, Any]:
                outer.calls.append("Runtime.callFunctionOn")
                source = params["functionDeclaration"]
                assert "option.selected" not in source, (
                    "Obscura ignores `option.selected = true` and it corrupts the select"
                )
                if raises:
                    return {"exceptionDetails": {"text": "boom"}}
                wanted = str(params["arguments"][0]["value"]).strip().lower()
                index = next(
                    (
                        i
                        for i, (text, value) in enumerate(outer.options)
                        if text.strip().lower() == wanted or value.strip().lower() == wanted
                    ),
                    -1,
                )
                if index < 0:
                    payload = {
                        "error": "not-found",
                        "options": [{"text": t, "value": v} for t, v in outer.options],
                    }
                else:
                    if outer.honour_selection:
                        outer.selected_index = index
                    landed = outer.options[outer.selected_index][0]
                    payload = {
                        "selected": landed == outer.options[index][0],
                        "text": outer.options[index][0],
                        "value": outer.options[index][1],
                        "index": outer.selected_index,
                        "landed_on": landed,
                    }
                return {"result": {"value": json.dumps(payload)}}

        self.send = SimpleNamespace(DOM=_DOM(), Runtime=_Runtime(), Input=_Input())


def _watchdog(cdp: _FakeCdp) -> SimpleNamespace:
    session = SimpleNamespace(session_id="sess", cdp_client=cdp)

    async def cdp_client_for_node(node: object) -> SimpleNamespace:
        return session

    return SimpleNamespace(browser_session=SimpleNamespace(cdp_client_for_node=cdp_client_for_node))


def _event(text: str, tag: str = "select") -> SimpleNamespace:
    return SimpleNamespace(node=SimpleNamespace(backend_node_id=42, tag_name=tag), text=text)


_OPTIONS = [("Open this select menu", "Open this select menu"), ("One", "1"), ("Two", "2")]


async def test_the_option_matching_the_text_is_selected() -> None:
    cdp = _FakeCdp(_OPTIONS)

    result = await patch_module.on_SelectDropdownOptionEvent(_watchdog(cdp), _event("Two"))

    assert result["success"] == "true"
    assert "Two" in result["message"]
    assert cdp.selected_index == 2


async def test_an_option_is_matched_by_its_value_too() -> None:
    cdp = _FakeCdp(_OPTIONS)

    assert (await patch_module.on_SelectDropdownOptionEvent(_watchdog(cdp), _event("1")))[
        "success"
    ] == "true"
    assert cdp.selected_index == 1


async def test_matching_ignores_case_and_surrounding_space() -> None:
    cdp = _FakeCdp(_OPTIONS)

    assert (await patch_module.on_SelectDropdownOptionEvent(_watchdog(cdp), _event("  tWo  ")))[
        "success"
    ] == "true"
    assert cdp.selected_index == 2


async def test_a_missing_option_reports_the_ones_that_exist() -> None:
    result = await patch_module.on_SelectDropdownOptionEvent(
        _watchdog(_FakeCdp(_OPTIONS)), _event("Four")
    )

    assert result["success"] == "false"
    assert "One" in result["short_term_memory"]
    assert "Two" in result["short_term_memory"]


async def test_a_dropdown_that_reverts_the_pick_is_reported_as_a_failure() -> None:
    # A page framework really resetting the value must still fail loudly -- the
    # patch only stops Obscura's stale `value` from faking that reversion.
    result = await patch_module.on_SelectDropdownOptionEvent(
        _watchdog(_FakeCdp(_OPTIONS, honour_selection=False)), _event("Two")
    )

    assert result["success"] == "false"
    assert "Open this select menu" in result["short_term_memory"]


async def test_an_unresolvable_node_fails_without_running_the_script() -> None:
    cdp = _FakeCdp(_OPTIONS, object_id=None)

    result = await patch_module.on_SelectDropdownOptionEvent(_watchdog(cdp), _event("Two"))

    assert result["success"] == "false"
    assert "Runtime.callFunctionOn" not in cdp.calls


async def test_a_script_that_throws_fails_instead_of_claiming_success() -> None:
    result = await patch_module.on_SelectDropdownOptionEvent(
        _watchdog(_FakeCdp(_OPTIONS, raises=True)), _event("Two")
    )

    assert result["success"] == "false"


async def test_an_aria_dropdown_goes_back_to_browser_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # role=menu/listbox widgets have no options to assign; only <select> is broken.
    called: list[str] = []

    async def original(watchdog: object, event: object) -> dict[str, str]:
        called.append(getattr(event.node, "tag_name", ""))
        return {"success": "true", "message": "aria"}

    monkeypatch.setattr(patch_module, "_original_on_select", original)
    cdp = _FakeCdp(_OPTIONS)

    result = await patch_module.on_SelectDropdownOptionEvent(
        _watchdog(cdp), _event("Two", tag="div")
    )

    assert result == {"success": "true", "message": "aria"}
    assert called == ["div"]
    assert cdp.calls == []


async def test_apply_rebinds_the_watchdog_handler() -> None:
    assert (
        DefaultActionWatchdog.on_SelectDropdownOptionEvent
        is patch_module.on_SelectDropdownOptionEvent
    )
