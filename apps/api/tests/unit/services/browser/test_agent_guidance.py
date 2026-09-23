"""What a blocked run shows the executor that has to unstick it."""

from __future__ import annotations

import pytest

from app.schemas.browser import AgentGuidanceRequest
from app.services.browser.agent_guidance import guidance_message

pytestmark = pytest.mark.unit


def _request(**extra: object) -> AgentGuidanceRequest:
    return AgentGuidanceRequest(
        reason="The upvote control is not on this screen.",
        task="Upvote the top post on r/python",
        url="https://www.reddit.com/r/python",
        title="r/python",
        **extra,  # type: ignore[arg-type]  # the test varies one optional field per case
    )


def test_the_changed_instruction_is_stated_before_the_task_it_overrides() -> None:
    """Regression: guidance was written against the original task, so it sent the run back to the login the user had cancelled."""
    note = "skip the upvote, just tell me the title of the top post"

    message = guidance_message(_request(user_notes=[note]))

    assert note in message
    assert message.index(note) < message.index("Upvote the top post on r/python")


def test_the_guidance_is_told_never_to_send_the_run_back_to_a_declined_step() -> None:
    message = guidance_message(_request(user_notes=["skip the upvote, just tell me the title"]))

    assert "declined" in message.lower()


def test_a_run_nobody_redirected_is_told_nothing_about_a_changed_instruction() -> None:
    message = guidance_message(_request())

    assert "changed the instruction" not in message.lower()
    assert "Upvote the top post on r/python" in message


def test_every_instruction_the_user_sent_reaches_the_guidance() -> None:
    message = guidance_message(_request(user_notes=["skip the upvote", "just read the title"]))

    assert "skip the upvote" in message
    assert "just read the title" in message
