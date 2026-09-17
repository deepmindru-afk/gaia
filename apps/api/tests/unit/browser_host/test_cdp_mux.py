"""The one engine connection behind a session, driven against a fake websocket.

Obscura isolates every CDP connection and numbers each one's contexts from 1, so
the host, the CDP proxy and the screencast must share a session's single socket.
Sharing it only works if every reply reaches the consumer that asked — including
when a client's own frame id collides with one the mux allocated, which is the
exact class of bug this module exists to prevent.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from app.browser_host import cdp_mux
from app.browser_host.cdp_mux import CdpConnectionClosed, CdpMux

_URL = "ws://127.0.0.1:9222/devtools/browser/fake"


class _FakeWebSocket:
    """A websockets client connection whose inbound frames the test feeds by hand."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.closed = False
        self._inbound: asyncio.Queue[str | None] = asyncio.Queue()
        # Set each time the mux writes, so a test can await "the send landed"
        # without a sleep or a poll loop.
        self.wrote = asyncio.Event()

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))
        self.wrote.set()

    async def close(self) -> None:
        self.closed = True
        await self._inbound.put(None)

    def deliver(self, *frames: dict[str, Any] | str) -> None:
        """Queue frames for the read loop; a str is delivered verbatim (malformed payloads)."""
        for frame in frames:
            self._inbound.put_nowait(frame if isinstance(frame, str) else json.dumps(frame))

    def end(self) -> None:
        """Let the engine hang up: the async-for finishes and the loop exits normally."""
        self._inbound.put_nowait(None)

    def __aiter__(self) -> _FakeWebSocket:
        return self

    async def __anext__(self) -> str:
        raw = await self._inbound.get()
        if raw is None:
            raise StopAsyncIteration
        return raw


async def _started_mux(ws: _FakeWebSocket) -> CdpMux:
    mux = CdpMux(_URL)
    with patch.object(cdp_mux.websockets, "connect", AsyncMock(return_value=ws)):
        await mux.start()
    return mux


async def _next_write(ws: _FakeWebSocket) -> dict[str, Any]:
    await asyncio.wait_for(ws.wrote.wait(), timeout=1.0)
    ws.wrote.clear()
    return ws.sent[-1]


@pytest.mark.unit
async def test_start_dials_the_session_url_and_a_call_gets_its_own_reply() -> None:
    ws = _FakeWebSocket()
    mux = CdpMux(_URL)
    connect = AsyncMock(return_value=ws)

    with patch.object(cdp_mux.websockets, "connect", connect):
        await mux.start()
    call = asyncio.create_task(mux.send_raw("Target.getTargets", {"a": 1}))
    sent = await _next_write(ws)
    ws.deliver({"id": sent["id"], "result": {"targetInfos": []}})

    assert await asyncio.wait_for(call, timeout=1.0) == {"targetInfos": []}
    assert connect.await_args is not None
    assert connect.await_args.args == (_URL,)
    assert sent["method"] == "Target.getTargets"
    assert sent["params"] == {"a": 1}
    await mux.close()


@pytest.mark.unit
async def test_a_forwarded_frame_goes_out_on_the_muxs_id_and_comes_back_on_the_clients() -> None:
    """The client routes on the id it chose, so the reply must wear that id again."""
    ws = _FakeWebSocket()
    mux = await _started_mux(ws)
    seen: list[dict[str, Any]] = []
    mux.subscribe(seen.append)

    await mux.forward(
        {"id": 77, "method": "Page.navigate", "params": {"url": "https://a.test"}}, seen.append
    )
    outbound = ws.sent[-1]
    ws.deliver({"id": outbound["id"], "result": {"frameId": "f1"}})
    await asyncio.sleep(0)

    assert outbound["id"] != 77  # rewritten onto the mux's own allocator
    assert outbound["method"] == "Page.navigate"
    assert seen == [{"id": 77, "result": {"frameId": "f1"}}]
    await mux.close()


@pytest.mark.unit
async def test_an_id_shared_by_a_control_call_and_a_client_frame_still_routes_to_each() -> None:
    """Two independent id counters is the collision this mux exists to prevent."""
    ws = _FakeWebSocket()
    mux = await _started_mux(ws)
    seen: list[dict[str, Any]] = []
    mux.subscribe(seen.append)

    call = asyncio.create_task(mux.send_raw("Target.getTargets"))
    control = await _next_write(ws)
    # The client picks the very id the mux just used for its own call.
    await mux.forward({"id": control["id"], "method": "Page.enable"}, seen.append)
    forwarded = ws.sent[-1]
    ws.deliver(
        {"id": control["id"], "result": {"targetInfos": [{"targetId": "t1"}]}},
        {"id": forwarded["id"], "result": {"enabled": True}},
    )

    assert await asyncio.wait_for(call, timeout=1.0) == {"targetInfos": [{"targetId": "t1"}]}
    await asyncio.sleep(0)
    # The control reply went to the caller alone; the client's reply went back to
    # the sink that forwarded it, wearing the colliding id the client chose.
    assert seen == [{"id": control["id"], "result": {"enabled": True}}]
    assert forwarded["id"] != control["id"]
    await mux.close()


@pytest.mark.unit
async def test_a_reply_to_nobody_and_every_event_reach_all_subscribers() -> None:
    ws = _FakeWebSocket()
    mux = await _started_mux(ws)
    first: list[dict[str, Any]] = []
    second: list[dict[str, Any]] = []
    mux.subscribe(first.append)
    mux.subscribe(second.append)

    event = {"method": "Page.loadEventFired", "params": {"timestamp": 1.0}}
    ws.deliver(event)
    await asyncio.sleep(0)

    assert first == [event]
    assert second == [event]
    await mux.close()


@pytest.mark.unit
async def test_a_sink_that_raises_neither_kills_the_loop_nor_starves_the_others() -> None:
    ws = _FakeWebSocket()
    mux = await _started_mux(ws)
    survivor: list[dict[str, Any]] = []

    def explode(_frame: dict[str, Any]) -> None:
        raise RuntimeError("bad sink")

    mux.subscribe(explode)
    mux.subscribe(survivor.append)

    ws.deliver({"method": "Page.loadEventFired"}, {"method": "Page.frameNavigated"})
    await asyncio.sleep(0)

    assert survivor == [{"method": "Page.loadEventFired"}, {"method": "Page.frameNavigated"}]
    # Still reading: a later call would hang forever if the loop had died.
    call = asyncio.create_task(mux.send_raw("Page.enable"))
    sent = await _next_write(ws)
    ws.deliver({"id": sent["id"], "result": {}})
    assert await asyncio.wait_for(call, timeout=1.0) == {}
    await mux.close()


@pytest.mark.unit
async def test_unsubscribing_stops_delivery_to_that_sink_only() -> None:
    ws = _FakeWebSocket()
    mux = await _started_mux(ws)
    leaving: list[dict[str, Any]] = []
    staying: list[dict[str, Any]] = []
    remove = mux.subscribe(leaving.append)
    mux.subscribe(staying.append)

    ws.deliver({"method": "Page.loadEventFired"})
    await asyncio.sleep(0)
    remove()
    ws.deliver({"method": "Page.frameNavigated"})
    await asyncio.sleep(0)

    assert leaving == [{"method": "Page.loadEventFired"}]
    assert staying == [{"method": "Page.loadEventFired"}, {"method": "Page.frameNavigated"}]
    await mux.close()


@pytest.mark.unit
async def test_close_fails_a_pending_caller_instead_of_leaving_it_hanging() -> None:
    """A caller parked on a dead socket never wakes up, which is how the host wedged."""
    ws = _FakeWebSocket()
    mux = await _started_mux(ws)
    call = asyncio.create_task(mux.send_raw("Target.getTargets"))
    await _next_write(ws)

    await mux.close()

    with pytest.raises(CdpConnectionClosed):
        await asyncio.wait_for(call, timeout=1.0)
    assert mux.closed is True
    assert ws.closed is True


@pytest.mark.unit
async def test_a_call_after_close_raises_rather_than_hanging() -> None:
    ws = _FakeWebSocket()
    mux = await _started_mux(ws)
    await mux.close()

    with pytest.raises(CdpConnectionClosed):
        await asyncio.wait_for(mux.send_raw("Target.getTargets"), timeout=1.0)


@pytest.mark.unit
async def test_wait_closed_releases_only_once_the_connection_ends() -> None:
    """An idle consumer has nothing to fail on, so this is how it learns to tear down."""
    ws = _FakeWebSocket()
    mux = await _started_mux(ws)
    waiter = asyncio.create_task(mux.wait_closed())
    await asyncio.sleep(0)

    assert not waiter.done()  # a live connection never releases it

    await mux.close()
    await asyncio.wait_for(waiter, timeout=1.0)


@pytest.mark.unit
async def test_the_engine_hanging_up_fails_pending_callers_too() -> None:
    ws = _FakeWebSocket()
    mux = await _started_mux(ws)
    call = asyncio.create_task(mux.send_raw("Target.getTargets"))
    await _next_write(ws)

    ws.end()  # the engine drops the socket; nobody called close()

    with pytest.raises(CdpConnectionClosed):
        await asyncio.wait_for(call, timeout=1.0)
    await asyncio.wait_for(mux.wait_closed(), timeout=1.0)


@pytest.mark.unit
async def test_an_unparsable_frame_is_dropped_without_killing_the_reader() -> None:
    ws = _FakeWebSocket()
    mux = await _started_mux(ws)
    seen: list[dict[str, Any]] = []
    mux.subscribe(seen.append)

    ws.deliver("{not json", {"method": "Page.loadEventFired"})
    await asyncio.sleep(0)

    assert seen == [{"method": "Page.loadEventFired"}]
    call = asyncio.create_task(mux.send_raw("Page.enable"))
    sent = await _next_write(ws)
    ws.deliver({"id": sent["id"], "result": {}})
    assert await asyncio.wait_for(call, timeout=1.0) == {}
    await mux.close()


@pytest.mark.unit
async def test_a_cdp_error_reply_raises_carrying_the_engines_own_error() -> None:
    ws = _FakeWebSocket()
    mux = await _started_mux(ws)
    call = asyncio.create_task(mux.send_raw("Target.createTarget"))
    sent = await _next_write(ws)

    ws.deliver(
        {"id": sent["id"], "error": {"code": -32000, "message": "Browser context not found"}}
    )

    with pytest.raises(RuntimeError) as failure:
        await asyncio.wait_for(call, timeout=1.0)
    assert failure.value.args[0] == {"code": -32000, "message": "Browser context not found"}
    await mux.close()


@pytest.mark.unit
async def test_a_session_scoped_call_carries_its_session_id_and_a_bare_one_does_not() -> None:
    ws = _FakeWebSocket()
    mux = await _started_mux(ws)

    scoped = asyncio.create_task(mux.send_raw("Runtime.evaluate", {"e": "1"}, session_id="sess-1"))
    scoped_frame = await _next_write(ws)
    ws.deliver({"id": scoped_frame["id"], "result": {}})
    await asyncio.wait_for(scoped, timeout=1.0)
    bare = asyncio.create_task(mux.send_raw("Target.getTargets"))
    bare_frame = await _next_write(ws)
    ws.deliver({"id": bare_frame["id"], "result": {}})
    await asyncio.wait_for(bare, timeout=1.0)

    assert scoped_frame["sessionId"] == "sess-1"
    assert "sessionId" not in bare_frame
    await mux.close()


@pytest.mark.unit
async def test_a_claimed_sessions_frames_reach_its_owner_and_nobody_else() -> None:
    """Regression: the live view's own page stream was fanned out to the agent's client."""
    ws = _FakeWebSocket()
    mux = await _started_mux(ws)
    viewer: list[dict[str, Any]] = []
    agent: list[dict[str, Any]] = []
    mux.subscribe(viewer.append, owns_session="page-1-session-1")
    mux.subscribe(agent.append)

    screenshot = {
        "method": "Page.screencastFrame",
        "sessionId": "page-1-session-1",
        "params": {"data": "<jpeg>"},
    }
    loaded = {"method": "Page.loadEventFired", "sessionId": "page-1-session-1", "params": {}}
    ws.deliver(screenshot, loaded)
    await asyncio.sleep(0)

    assert viewer == [screenshot, loaded]
    assert agent == []
    await mux.close()


@pytest.mark.unit
async def test_an_unclaimed_session_reaches_the_open_stream_but_not_a_claiming_sink() -> None:
    ws = _FakeWebSocket()
    mux = await _started_mux(ws)
    viewer: list[dict[str, Any]] = []
    agent: list[dict[str, Any]] = []
    mux.subscribe(viewer.append, owns_session="page-1-session-1")
    mux.subscribe(agent.append)

    # The agent attaches its own page sessions through the proxy, and browser-level
    # events carry no session at all; both belong to whoever claimed nothing.
    other_page = {"method": "Page.loadEventFired", "sessionId": "page-2-session-1", "params": {}}
    browser_level = {"method": "Target.targetCreated", "params": {"targetInfo": {}}}
    ws.deliver(other_page, browser_level)
    await asyncio.sleep(0)

    assert agent == [other_page, browser_level]
    assert viewer == []
    await mux.close()


@pytest.mark.unit
async def test_a_session_returns_to_the_open_stream_once_its_owner_unsubscribes() -> None:
    ws = _FakeWebSocket()
    mux = await _started_mux(ws)
    agent: list[dict[str, Any]] = []
    remove = mux.subscribe(lambda _frame: None, owns_session="page-1-session-1")
    mux.subscribe(agent.append)

    claimed = {"method": "Page.loadEventFired", "sessionId": "page-1-session-1", "params": {}}
    ws.deliver(claimed)
    await asyncio.sleep(0)
    remove()
    ws.deliver(claimed)
    await asyncio.sleep(0)

    assert agent == [claimed]
    await mux.close()


@pytest.mark.unit
async def test_the_engine_hanging_up_marks_the_mux_closed_and_fails_later_calls() -> None:
    """Teardown reads closed to decide whether to talk to the engine, so it must not lie."""
    ws = _FakeWebSocket()
    mux = await _started_mux(ws)

    ws.end()  # the engine drops the socket; nobody called close()
    await asyncio.wait_for(mux.wait_closed(), timeout=1.0)

    assert mux.closed is True
    with pytest.raises(CdpConnectionClosed):
        await asyncio.wait_for(mux.send_raw("Target.getTargets"), timeout=1.0)
    with pytest.raises(CdpConnectionClosed):
        await asyncio.wait_for(
            mux.forward({"id": 1, "method": "Page.enable"}, lambda _frame: None), timeout=1.0
        )


@pytest.mark.unit
async def test_a_forwarded_reply_goes_to_its_sender_even_when_another_sink_owns_the_session() -> (
    None
):
    """A reply belongs to whoever asked, not to whoever owns the session it names."""
    ws = _FakeWebSocket()
    mux = await _started_mux(ws)
    agent: list[dict[str, Any]] = []
    viewer: list[dict[str, Any]] = []
    mux.subscribe(agent.append)
    mux.subscribe(viewer.append, owns_session="page-1-session-1")

    await mux.forward(
        {"id": 5, "method": "Page.enable", "sessionId": "page-1-session-1"}, agent.append
    )
    outbound = ws.sent[-1]
    ws.deliver({"id": outbound["id"], "sessionId": "page-1-session-1", "result": {}})
    await asyncio.sleep(0)

    assert agent == [{"id": 5, "sessionId": "page-1-session-1", "result": {}}]
    assert viewer == []
    await mux.close()
