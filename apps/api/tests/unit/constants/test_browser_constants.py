"""The handoff-action enum against the prompt text the model actually reads.

The preamble tells the model to call the two actions by name. If the enum and
the prose ever disagree, the model is told to call something that isn't
registered — so the prose is pinned to the enum here.
"""

import pytest

from app.constants.browser import BROWSER_TAKEOVER_PREAMBLE, BrowserHandoffAction


@pytest.mark.unit
def test_the_takeover_preamble_names_the_registered_actions() -> None:
    assert BrowserHandoffAction.REQUEST_HUMAN_TAKEOVER.value in BROWSER_TAKEOVER_PREAMBLE
    assert BrowserHandoffAction.SOLVE_CAPTCHA_WITH_HELP.value in BROWSER_TAKEOVER_PREAMBLE


@pytest.mark.unit
def test_handoff_action_members_render_as_their_value_in_prompts() -> None:
    """StrEnum, not (str, Enum): interpolating a member yields the bare action name, never BrowserHandoffAction.REQUEST_HUMAN_TAKEOVER."""
    assert f"{BrowserHandoffAction.REQUEST_HUMAN_TAKEOVER}" == "request_human_takeover"
    assert f"{BrowserHandoffAction.SOLVE_CAPTCHA_WITH_HELP}" == "solve_captcha_with_help"
