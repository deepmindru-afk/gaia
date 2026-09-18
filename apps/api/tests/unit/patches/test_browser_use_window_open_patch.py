"""A page's window.open must reach somewhere, because Obscura opens no window for it."""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from browser_use.browser.session import BrowserSession
import pytest

import app.patches.browser_use_window_open_patch as patch_module

pytestmark = pytest.mark.unit


class _FakeCdp:
    def __init__(self, *, fails: bool = False) -> None:
        self.sources: list[str] = []
        self.evaluated: list[str] = []
        outer = self

        class _Page:
            @staticmethod
            async def addScriptToEvaluateOnNewDocument(
                params: dict[str, Any], session_id: str
            ) -> dict[str, Any]:
                if fails:
                    raise RuntimeError("no such target")
                outer.sources.append(params["source"])
                return {"identifier": "1"}

        class _Runtime:
            @staticmethod
            async def evaluate(params: dict[str, Any], session_id: str) -> dict[str, Any]:
                outer.evaluated.append(params["expression"])
                return {"result": {}}

        self.send = SimpleNamespace(Page=_Page(), Runtime=_Runtime())


def _session(target_id: str, cdp: _FakeCdp) -> SimpleNamespace:
    return SimpleNamespace(target_id=target_id, session_id=f"{target_id}-s", cdp_client=cdp)


def _browser_session(*targets: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> object:
    queue = list(targets)

    async def original(self: object, target_id: str | None = None, focus: bool = True) -> object:
        return queue.pop(0)

    monkeypatch.setattr(patch_module, "_original_get_or_create_cdp_session", original)
    return SimpleNamespace()


async def test_the_shim_is_installed_on_a_fresh_target(monkeypatch: pytest.MonkeyPatch) -> None:
    cdp = _FakeCdp()
    session = _browser_session(_session("t1", cdp), monkeypatch=monkeypatch)

    await patch_module._get_or_create_cdp_session(session)

    assert cdp.sources == [patch_module.WINDOW_OPEN_SHIM]


async def test_the_shim_also_runs_on_the_document_already_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Obscura accepts runImmediately and ignores it, arming only the next load.
    cdp = _FakeCdp()
    session = _browser_session(_session("t1", cdp), monkeypatch=monkeypatch)

    await patch_module._get_or_create_cdp_session(session)

    assert cdp.evaluated == [patch_module.WINDOW_OPEN_SHIM]


async def test_a_target_is_only_shimmed_once(monkeypatch: pytest.MonkeyPatch) -> None:
    cdp = _FakeCdp()
    session = _browser_session(_session("t1", cdp), _session("t1", cdp), monkeypatch=monkeypatch)

    await patch_module._get_or_create_cdp_session(session)
    await patch_module._get_or_create_cdp_session(session)

    assert len(cdp.sources) == 1


async def test_every_new_target_gets_its_own_shim(monkeypatch: pytest.MonkeyPatch) -> None:
    cdp = _FakeCdp()
    session = _browser_session(_session("t1", cdp), _session("t2", cdp), monkeypatch=monkeypatch)

    await patch_module._get_or_create_cdp_session(session)
    await patch_module._get_or_create_cdp_session(session)

    assert len(cdp.sources) == 2


async def test_a_failed_injection_is_retried_on_the_next_visit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(patch_module, "log", MagicMock())
    failing, working = _FakeCdp(fails=True), _FakeCdp()
    session = _browser_session(
        _session("t1", failing), _session("t1", working), monkeypatch=monkeypatch
    )

    await patch_module._get_or_create_cdp_session(session)
    await patch_module._get_or_create_cdp_session(session)

    assert working.sources == [patch_module.WINDOW_OPEN_SHIM]


async def test_a_failed_injection_never_fails_the_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = MagicMock()
    monkeypatch.setattr(patch_module, "log", logger)
    target = _session("t1", _FakeCdp(fails=True))
    session = _browser_session(target, monkeypatch=monkeypatch)

    assert await patch_module._get_or_create_cdp_session(session) is target
    logger.warning.assert_called_once()


@pytest.mark.parametrize(
    "url",
    ["", "javascript:alert(1)", "about:blank", "data:text/html,<p>x"],
    ids=["empty", "javascript", "about", "data"],
)
def test_the_shim_refuses_to_navigate_anywhere_but_http(url: str) -> None:
    # The shim runs in the page, so the guard is read off its source: only
    # http(s) may take over the tab Jev is working in.
    assert "'http:'" in patch_module.WINDOW_OPEN_SHIM
    assert "'https:'" in patch_module.WINDOW_OPEN_SHIM
    assert re.search(r"if\s*\(!url\)\s*return null", patch_module.WINDOW_OPEN_SHIM)


def test_the_returned_stub_cannot_close_the_tab() -> None:
    # A page calling w.close() on the real window would close Jev's own tab.
    assert "close() {}" in patch_module.WINDOW_OPEN_SHIM
    assert "return stub" in patch_module.WINDOW_OPEN_SHIM


async def test_apply_rebinds_the_session_accessor() -> None:
    # Applied explicitly: the stealth patch wraps the same funnel, so which
    # wrapper sits outermost depends on import order, and this asserts only that
    # applying ours puts ours there.
    patch_module.apply()

    assert BrowserSession.get_or_create_cdp_session is patch_module._get_or_create_cdp_session
