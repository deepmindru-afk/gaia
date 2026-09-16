"""The crash catch-all that runs inside CORS.

Starlette answers an uncaught exception in ServerErrorMiddleware, which wraps
the whole app including CORSMiddleware — so that 500 envelope carries no
Access-Control-Allow-Origin and no browser can read it. This middleware sits
innermost and converts the crash there instead.
"""

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

from httpx import ASGITransport, AsyncClient
import pytest
from starlette.responses import PlainTextResponse
from starlette.types import Receive, Scope, Send

from app.api.v1.middleware.unhandled_exception import UnhandledExceptionMiddleware

pytestmark = pytest.mark.unit


async def _boom_app(scope: Scope, receive: Receive, send: Send) -> None:
    raise RuntimeError("db down")


async def _ok_app(scope: Scope, receive: Receive, send: Send) -> None:
    await PlainTextResponse("fine")(scope, receive, send)


async def _crash_midstream_app(scope: Scope, receive: Receive, send: Send) -> None:
    await send({"type": "http.response.start", "status": 200, "headers": []})
    raise RuntimeError("died after the status line")


def _client(app: Any) -> AsyncClient:
    return AsyncClient(
        transport=ASGITransport(app=UnhandledExceptionMiddleware(app), raise_app_exceptions=False),
        base_url="http://test",
    )


async def test_a_crash_becomes_the_500_envelope() -> None:
    async with _client(_boom_app) as client:
        response = await client.get("/x")

    assert response.status_code == 500
    assert response.json() == {"message": "Internal server error", "code": "internal_server_error"}


async def test_a_successful_response_passes_through_untouched() -> None:
    async with _client(_ok_app) as client:
        response = await client.get("/x")

    assert response.status_code == 200
    assert response.text == "fine"


async def test_a_crash_after_the_status_line_propagates_instead_of_double_sending() -> None:
    """There is no envelope to send once the status is on the wire."""
    transport = ASGITransport(
        app=UnhandledExceptionMiddleware(_crash_midstream_app), raise_app_exceptions=True
    )
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with pytest.raises(RuntimeError, match="died after the status line"):
            await client.get("/x")


async def test_a_websocket_scope_is_not_intercepted() -> None:
    seen: list[str] = []

    async def _ws_app(scope: Scope, receive: Receive, send: Send) -> None:
        seen.append(scope["type"])

    await UnhandledExceptionMiddleware(_ws_app)({"type": "websocket"}, MagicMock(), MagicMock())

    assert seen == ["websocket"]


async def test_the_crash_is_recorded_on_the_wide_event() -> None:
    """LoggingMiddleware's except path never sees a crash this converts, so the
    failure would vanish from the canonical event without this line."""
    with patch("app.api.v1.middleware.unhandled_exception.log") as mock_log:
        async with _client(_boom_app) as client:
            await client.get("/boom-path")

    mock_log.error.assert_called_once_with(
        "unhandled_exception",
        error_type="RuntimeError",
        error="db down",
        path="/boom-path",
    )


async def test_the_crash_is_attributed_in_posthog(
    posthog_provider: Callable[..., None],
) -> None:
    posthog_client = MagicMock()
    posthog_provider(available=True, client=posthog_client)

    async with _client(_boom_app) as client:
        await client.get("/x")

    posthog_client.capture_exception.assert_called_once()
    assert isinstance(posthog_client.capture_exception.call_args.args[0], RuntimeError)
