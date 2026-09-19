"""A session that takes its own step photo gets state reads without a screenshot."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

from browser_use.browser.session import BrowserSession
import pytest

from app.patches import browser_use_deferred_screenshot_patch as patch_mod

pytestmark = pytest.mark.unit


async def _state_read(monkeypatch: pytest.MonkeyPatch, session: BrowserSession) -> dict[str, Any]:
    original = AsyncMock(return_value="state")
    monkeypatch.setattr(patch_mod, "_original_get_browser_state_summary", original)
    await patch_mod._get_browser_state_summary(session, include_screenshot=True, cached=True)
    return original.await_args.kwargs


async def test_a_registered_session_reads_state_without_a_screenshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = BrowserSession()
    patch_mod.defer_screenshots_for(session)

    kwargs = await _state_read(monkeypatch, session)

    assert kwargs["include_screenshot"] is False
    assert kwargs["cached"] is True


async def test_an_unregistered_session_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs = await _state_read(monkeypatch, BrowserSession())

    assert kwargs["include_screenshot"] is True
