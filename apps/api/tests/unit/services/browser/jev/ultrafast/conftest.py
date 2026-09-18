"""Seams for the ultrafast loop: a fake page, a fake browser, canned HTTP.

Nothing under test is mocked. The gateway and the text helper are real clients
over ``httpx.MockTransport``, so the real request bodies are built and the real
responses are parsed; only the browser (CDP) and the network are faked.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.services.browser.jev.gateway import JevGatewayClient
from app.services.browser.jev.ultrafast.browser import StalePage, fingerprint
from app.services.browser.jev.ultrafast.model import Action, JevTextHelper, PageState

DECISIONS_URL = "https://openrouter.test/api/alpha/decisions"
TEXT_URL = "https://openrouter.test/api/v1/chat/completions"


def make_page() -> PageState:
    """One snapshot: a search box (fill + click on one node) and a submit button."""
    state: PageState = {
        "url": "https://example.test/",
        "title": "Search",
        "text": "Search",
        "scroll": {"y": 0, "height": 800},
        "marker": ["marker"],
        "page_key": ["page-key"],
        "guards": {"10": ["guard-10"], "20": ["guard-20"]},
        "actions": [
            {
                "id": "e1",
                "kind": "fill",
                "label": "Search",
                "role": "textbox",
                "value": "",
                "node": 10,
            },
            {
                "id": "e2",
                "kind": "click",
                "label": "Open Search",
                "role": "textbox",
                "value": "",
                "node": 10,
            },
            {
                "id": "e3",
                "kind": "click",
                "label": "Go",
                "role": "button",
                "value": "",
                "node": 20,
            },
            {"id": "wait", "kind": "wait", "label": "Wait for the page to update"},
        ],
    }
    state["fingerprint"] = fingerprint(state)
    return state


@pytest.fixture
def page() -> PageState:
    return make_page()


def distribution(keys: Any, choice: str, confidence: float = 0.9) -> dict[str, Any]:
    """A well-formed choice answer: argmax on ``choice``, mass summing to 1."""
    keys = list(keys)
    rest = (1 - 0.9) / (len(keys) - 1) if len(keys) > 1 else 0.0
    top = 1 - rest * (len(keys) - 1)
    return {
        "type": "choice",
        "choice": choice,
        "confidence": confidence,
        "probabilities": {k: (top if k == choice else rest) for k in keys},
    }


class FakeBrowser:
    """The CDP seam. ``act`` keeps the real executor's freshness gate so the
    agent's consume-before-mutate ordering is genuinely exercised."""

    def __init__(self, page: PageState) -> None:
        self.page = page
        self.acted: list[tuple[str, str | None]] = []
        self.observations = 0
        self.is_fresh = True
        self.fresh_error: Exception | None = None
        self.act_error: Exception | None = None
        self.observe_error: Exception | None = None
        self.closed = False

    async def fresh(self, page: PageState, action: Action | None = None) -> bool:
        if self.fresh_error is not None:
            raise self.fresh_error
        return self.is_fresh

    async def act(self, action: Action, page: PageState, text: str | None = None) -> dict[str, str]:
        if not await self.fresh(page, action):
            raise StalePage("Page changed since this decision. Observe again.")
        if self.act_error is not None:
            error, self.act_error = self.act_error, None
            raise error
        self.acted.append((action["id"], text))
        return {"executed": action["id"]}

    async def observe(self, *, screenshot: bool = False) -> PageState:
        self.observations += 1
        if self.observe_error is not None:
            raise self.observe_error
        return self.page

    async def close(self) -> None:
        self.closed = True


class RecordingTransport:
    """Canned HTTP that keeps every request body it was sent."""

    def __init__(self, responder: Any) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.responder = responder

    def transport(self) -> httpx.MockTransport:
        def handle(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            self.bodies.append(body)
            return httpx.Response(200, json=self.responder(body))

        return httpx.MockTransport(handle)


def make_gateway(responder: Any) -> tuple[JevGatewayClient, RecordingTransport]:
    recorder = RecordingTransport(responder)
    client = JevGatewayClient(
        api_key="test-key",
        model="~typesafe/jev-latest",
        url=DECISIONS_URL,
        client=httpx.AsyncClient(transport=recorder.transport()),
    )
    return client, recorder


def make_text_helper(responder: Any) -> tuple[JevTextHelper, RecordingTransport]:
    recorder = RecordingTransport(responder)
    helper = JevTextHelper(
        api_key="test-key",
        model="inception/mercury-2.5",
        url=TEXT_URL,
        client=httpx.AsyncClient(transport=recorder.transport()),
    )
    return helper, recorder


def completion(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}}], "usage": {"total_tokens": 7}}
