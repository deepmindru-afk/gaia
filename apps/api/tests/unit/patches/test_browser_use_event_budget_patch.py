"""Browser-Use's screenshot and state-read budgets fit what the engine measures."""

from __future__ import annotations

from browser_use.browser.events import BrowserStateRequestEvent, ScreenshotEvent
import pytest

from app.patches import browser_use_event_budget_patch as patch_mod

pytestmark = pytest.mark.unit

# Slowest first screenshot measured on a very long page, in seconds.
_SLOWEST_MEASURED_SCREENSHOT = 35.0


def test_a_screenshot_is_given_longer_than_the_slowest_one_measured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TIMEOUT_ScreenshotEvent", raising=False)
    patch_mod.apply()

    event = ScreenshotEvent()

    assert event.event_timeout is not None
    assert event.event_timeout > _SLOWEST_MEASURED_SCREENSHOT


def test_the_state_read_outlasts_the_screenshot_it_contains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TIMEOUT_ScreenshotEvent", raising=False)
    monkeypatch.delenv("TIMEOUT_BrowserStateRequestEvent", raising=False)
    patch_mod.apply()

    screenshot, state_read = ScreenshotEvent(), BrowserStateRequestEvent()

    assert screenshot.event_timeout is not None
    assert state_read.event_timeout is not None
    assert state_read.event_timeout > screenshot.event_timeout


def test_an_operators_own_budget_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TIMEOUT_ScreenshotEvent", "5")
    patch_mod.apply()

    assert ScreenshotEvent().event_timeout == 5.0
