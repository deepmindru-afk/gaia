"""Middleware registration order — the invariant PostHog identity rests on.

Starlette runs ``app.user_middleware`` outermost-first, so a middleware's index
IS its execution order. Two orderings here are load-bearing and neither was
asserted anywhere; the module had no unit test at all.
"""

import inspect
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from limits import parse
from limits.errors import StorageError
import pytest
from slowapi.errors import RateLimitExceeded
from slowapi.wrappers import Limit
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.testclient import TestClient

from app.api.v1.middleware.auth import PostHogRequestContextMiddleware
from app.core.middleware import configure_middleware, rate_limit_handler


@pytest.fixture
def middleware_names() -> list[str]:
    app = FastAPI()
    configure_middleware(app)
    return [m.cls.__name__ for m in app.user_middleware]


def test_posthog_context_is_registered(middleware_names: list[str]) -> None:
    """Without it no authenticated request is identified and every capture in
    a route handler lands on an anonymous profile."""
    assert "PostHogRequestContextMiddleware" in middleware_names


def test_posthog_context_runs_inside_workos_auth(middleware_names: list[str]) -> None:
    """It reads ``request.state.user``, which WorkOSAuthMiddleware populates.

    Registered the other way round it would run first, see no user, and
    silently identify nobody — the events still send, just unattributed.
    """
    assert middleware_names.index("WorkOSAuthMiddleware") < middleware_names.index(
        "PostHogRequestContextMiddleware"
    )


def test_bot_auth_runs_inside_posthog_context(middleware_names: list[str]) -> None:
    """The documented reason bot routes must attribute explicitly.

    BotAuthMiddleware populates ``request.state.user`` for bot API-key traffic,
    but it runs INSIDE the PostHog context, which has already decided there is
    nobody to identify. Hence ``capture_event(user_id, ...)`` rather than
    ``capture_context_event`` on every bot route (see apps/api/CLAUDE.md).
    Should this order ever flip, that guidance becomes wrong.
    """
    assert middleware_names.index("PostHogRequestContextMiddleware") < (
        middleware_names.index("BotAuthMiddleware")
    )


def test_the_crash_catch_all_runs_inside_cors(middleware_names: list[str]) -> None:
    """Outside CORS its 500 envelope carries no Access-Control-Allow-Origin and
    no browser can read it — which is exactly what ServerErrorMiddleware does."""
    assert middleware_names.index("CORSMiddleware") < middleware_names.index(
        "UnhandledExceptionMiddleware"
    )


def test_the_crash_catch_all_covers_every_layer_inside_cors(
    middleware_names: list[str],
) -> None:
    """A crash in the paywall gate, the timeout or the limiter is a 500 too."""
    catch_all = middleware_names.index("UnhandledExceptionMiddleware")
    for inner in (
        "EntitlementMiddleware",
        "RequestTimeoutMiddleware",
        "RouterAwareSlowAPIMiddleware",
    ):
        assert catch_all < middleware_names.index(inner)


class TestPostHogContextDoesNotSwallowExceptions:
    """The context must not become the thing that reports the error.

    ``new_context`` autocaptures escaping exceptions by default, through the
    MODULE-level posthog client — which this codebase never configures, since
    it builds a ``Posthog()`` instance via the lazy provider. That autocapture
    raises ``ValueError("API key is required")`` on the way out and REPLACES the
    real exception, so every authenticated 500 reaches the error handler, the
    wide event and Sentry as the same bogus ValueError.

    Order assertions above cannot catch this; only driving a request can.
    """

    @staticmethod
    def _app_that_raises() -> FastAPI:
        app = FastAPI()

        class _AuthenticateEveryone(BaseHTTPMiddleware):
            async def dispatch(
                self, request: Request, call_next: RequestResponseEndpoint
            ) -> Response:
                request.state.user = {"user_id": "user-123"}
                return await call_next(request)

        app.add_middleware(PostHogRequestContextMiddleware)
        app.add_middleware(_AuthenticateEveryone)

        @app.get("/boom")
        async def boom() -> None:
            raise RuntimeError("the real bug in the handler")

        return app

    def test_the_handlers_own_exception_is_what_propagates(self) -> None:
        with patch("app.api.v1.middleware.auth.providers") as providers:
            providers.is_available.return_value = True
            providers.get.return_value = MagicMock()
            client = TestClient(self._app_that_raises())

            with pytest.raises(RuntimeError, match="the real bug in the handler"):
                client.get("/boom")


class TestRateLimitHandler:
    """slowapi's 429 answers before any route runs, so it must be the envelope too."""

    def test_configure_middleware_registers_it_for_slowapi(self) -> None:
        app = FastAPI()
        configure_middleware(app)
        assert app.exception_handlers[RateLimitExceeded] is rate_limit_handler

    def test_the_handler_is_sync_so_slowapi_cannot_discard_it(self) -> None:
        """``sync_check_limits`` drops a coroutine handler and emits slowapi's
        own ``{"error": ...}`` body, taking every default-limit 429 off the envelope."""
        assert not inspect.iscoroutinefunction(rate_limit_handler)

    @staticmethod
    def _app_that_is_rate_limited(*, record_window: bool) -> FastAPI:
        app = FastAPI()
        app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
        item = parse("5/minute")
        limit = Limit(
            item,
            key_func=lambda: "client",
            scope=None,
            per_method=False,
            methods=None,
            error_message="Too many requests, slow down",
            exempt_when=None,
            cost=1,
            override_defaults=False,
        )

        @app.get("/limited")
        async def limited(request: Request) -> None:
            if record_window:
                # What Limiter.__evaluate_limits stamps just before it raises.
                request.state.view_rate_limit = (item, ["client", "/limited"])
            raise RateLimitExceeded(limit)

        return app

    def test_the_429_body_is_the_envelope_with_a_real_retry_after_header(self) -> None:
        resp = TestClient(self._app_that_is_rate_limited(record_window=True)).get("/limited")
        assert resp.status_code == 429
        assert resp.json() == {
            "message": "Too many requests, slow down",
            "code": "rate_limit_exceeded",
        }
        assert 0 < int(resp.headers["retry-after"]) <= 60

    def test_unreadable_storage_falls_back_to_the_whole_window(self) -> None:
        """Redis can die between the hit that refused the request and this read;
        the window length is the correct upper bound, and the 429 must still land."""
        with patch(
            "app.core.middleware.limiter.limiter.get_window_stats",
            side_effect=StorageError(RuntimeError("redis gone")),
        ):
            resp = TestClient(self._app_that_is_rate_limited(record_window=True)).get("/limited")

        assert resp.status_code == 429
        assert resp.headers["retry-after"] == "60"

    def test_an_unknown_window_omits_the_header_instead_of_shipping_a_null(self) -> None:
        resp = TestClient(self._app_that_is_rate_limited(record_window=False)).get("/limited")
        assert resp.status_code == 429
        assert resp.json() == {
            "message": "Too many requests, slow down",
            "code": "rate_limit_exceeded",
        }
        assert "retry-after" not in resp.headers
