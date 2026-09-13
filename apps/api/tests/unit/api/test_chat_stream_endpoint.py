"""GET /stream/{stream_id} — replay semantics of the executor-stream endpoint.

Regression: a completed stream whose Redis event log still exists must replay
that log, not short-circuit to a bare [DONE]. A HIL resume publishes its
frames (second approval card included) and closes within ~100ms — faster than
the client's websocket-to-fetch round trip — so the short-circuit dropped
every frame of nearly every resumed run.
"""

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

STREAM_ID = "queued_regression-replay"

FRAMES = [
    'data: {"tool_data": {"tool_name": "approval_request", "data": {"approval_id": "a2"}}}\n\n',
    "data: [DONE]\n\n",
]


def _fake_subscribe(
    stream_id: str, keepalive_interval: float = 15, last_event_id: str | None = None
) -> AsyncGenerator[str, None]:
    async def _gen() -> AsyncGenerator[str, None]:
        for frame in FRAMES:
            yield frame

    return _gen()


class TestSubscribeExecutorStreamReplay:
    @pytest.mark.regression
    async def test_completed_stream_with_live_log_replays_frames(self, client) -> None:
        with (
            patch(
                "app.api.v1.endpoints.chat.stream_manager.get_progress",
                new=AsyncMock(
                    return_value={
                        "user_id": "507f1f77bcf86cd799439011",
                        "conversation_id": "conv-1",
                        "is_complete": True,
                    }
                ),
            ),
            patch(
                "app.api.v1.endpoints.chat.stream_manager.has_events",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "app.api.v1.endpoints.chat.stream_manager.subscribe_stream",
                new=_fake_subscribe,
            ),
            # _stream_from_redis checks this singleton before subscribe_stream;
            # an xdist-shared client state could otherwise flip this test
            # between replay and [STREAM_ERROR]. Pinning isolates the assertion.
            patch("app.api.v1.endpoints.chat.redis_cache.redis", new=MagicMock()),
        ):
            async with client.stream("GET", f"/api/v1/stream/{STREAM_ID}") as response:
                assert response.status_code == 200
                body = "".join([chunk async for chunk in response.aiter_text()])

        assert "approval_request" in body
        assert "[DONE]" in body

    async def test_no_redis_client_reports_a_stream_error(self, client) -> None:
        """Without a Redis client there is no event log to follow, and the client must be told so rather than handed a bare [DONE]."""
        with (
            patch(
                "app.api.v1.endpoints.chat.stream_manager.get_progress",
                new=AsyncMock(
                    return_value={
                        "user_id": "507f1f77bcf86cd799439011",
                        "conversation_id": "conv-1",
                        "is_complete": True,
                    }
                ),
            ),
            patch(
                "app.api.v1.endpoints.chat.stream_manager.has_events",
                new=AsyncMock(return_value=True),
            ),
            patch("app.api.v1.endpoints.chat.redis_cache.redis", new=None),
        ):
            async with client.stream("GET", f"/api/v1/stream/{STREAM_ID}") as response:
                assert response.status_code == 200
                body = "".join([chunk async for chunk in response.aiter_text()])

        assert body == "data: [STREAM_ERROR]\n\n"

    @pytest.mark.regression
    async def test_the_log_lookup_names_the_requested_stream(self, client) -> None:
        """The expired-log check must ask about this stream; asking about another id answers for the wrong stream and reads a live log as expired."""
        events_by_stream = {STREAM_ID: True}

        async def _has_events(stream_id: str) -> bool:
            return events_by_stream.get(stream_id, False)

        with (
            patch(
                "app.api.v1.endpoints.chat.stream_manager.get_progress",
                new=AsyncMock(
                    return_value={
                        "user_id": "507f1f77bcf86cd799439011",
                        "conversation_id": "conv-1",
                        "is_complete": True,
                    }
                ),
            ),
            patch("app.api.v1.endpoints.chat.stream_manager.has_events", new=_has_events),
            patch(
                "app.api.v1.endpoints.chat.stream_manager.subscribe_stream",
                new=_fake_subscribe,
            ),
            patch("app.api.v1.endpoints.chat.redis_cache.redis", new=MagicMock()),
        ):
            async with client.stream("GET", f"/api/v1/stream/{STREAM_ID}") as response:
                assert response.status_code == 200
                body = "".join([chunk async for chunk in response.aiter_text()])

        assert body == "".join(FRAMES)

    async def test_completed_stream_with_expired_log_returns_done_only(self, client) -> None:
        # Both are pinned even though the short-circuit means neither should
        # be reached — if the guard stops short-circuiting, this fails fast
        # on unexpected frames instead of hanging on real keepalives.
        with (
            patch(
                "app.api.v1.endpoints.chat.stream_manager.get_progress",
                new=AsyncMock(
                    return_value={
                        "user_id": "507f1f77bcf86cd799439011",
                        "conversation_id": "conv-1",
                        "is_complete": True,
                    }
                ),
            ),
            patch(
                "app.api.v1.endpoints.chat.stream_manager.has_events",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "app.api.v1.endpoints.chat.stream_manager.subscribe_stream",
                new=_fake_subscribe,
            ),
            patch("app.api.v1.endpoints.chat.redis_cache.redis", new=MagicMock()),
        ):
            async with client.stream("GET", f"/api/v1/stream/{STREAM_ID}") as response:
                assert response.status_code == 200
                body = "".join([chunk async for chunk in response.aiter_text()])

        assert body == "data: [DONE]\n\n"

    @pytest.mark.parametrize("has_events", [True, False])
    async def test_a_live_stream_always_replays_whatever_the_log_says(
        self, client, has_events: bool
    ) -> None:
        """The expired-log short-circuit is gated on completion first; a still-running stream has more frames coming and must be followed regardless."""
        with (
            patch(
                "app.api.v1.endpoints.chat.stream_manager.get_progress",
                new=AsyncMock(
                    return_value={
                        "user_id": "507f1f77bcf86cd799439011",
                        "conversation_id": "conv-1",
                        "is_complete": False,
                    }
                ),
            ),
            patch(
                "app.api.v1.endpoints.chat.stream_manager.has_events",
                new=AsyncMock(return_value=has_events),
            ),
            patch(
                "app.api.v1.endpoints.chat.stream_manager.subscribe_stream",
                new=_fake_subscribe,
            ),
            patch("app.api.v1.endpoints.chat.redis_cache.redis", new=MagicMock()),
        ):
            async with client.stream("GET", f"/api/v1/stream/{STREAM_ID}") as response:
                assert response.status_code == 200
                body = "".join([chunk async for chunk in response.aiter_text()])

        assert body == "".join(FRAMES)
