"""
Middleware configuration for the GAIA FastAPI application.

This module provides functions to configure middleware for the FastAPI application.
"""

import math
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import UJSONResponse
from limits import RateLimitItem
from limits.errors import StorageError
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from workos import AsyncWorkOSClient

from app.api.v1.middleware.auth import PostHogRequestContextMiddleware, WorkOSAuthMiddleware
from app.api.v1.middleware.entitlement import EntitlementMiddleware
from app.api.v1.middleware.logging import LoggingMiddleware
from app.api.v1.middleware.profiling import ProfilingMiddleware
from app.api.v1.middleware.rate_limiter import RouterAwareSlowAPIMiddleware, limiter
from app.api.v1.middleware.timeout import RequestTimeoutMiddleware
from app.api.v1.middleware.unhandled_exception import UnhandledExceptionMiddleware
from app.api.v1.middleware.websocket_wide_event import WebSocketWideEventMiddleware
from app.config.settings import settings
from app.constants.http import RETRY_AFTER_HEADER
from app.core.bot_auth_middleware import BotAuthMiddleware
from app.schemas.errors import ErrorEnvelope, error_response
from shared.py.wide_events import log as wide_log

RATE_LIMIT_EXCEEDED_CODE = "rate_limit_exceeded"


def _retry_after_seconds(request: Request) -> int | None:
    """Whole seconds until the window that refused this request reopens.

    ``RateLimitExceeded`` carries no retry hint of its own — the old handler's
    ``getattr(exc, "retry_after", None)`` was always ``None`` and shipped that
    null as a contract. The limiter records the window it hit on
    ``request.state`` just before raising, so ask that; if the storage cannot
    answer, the full window length is the correct upper bound.

    The limiter comes off ``request.app.state`` like slowapi's own handler reads
    it, not from this module: the app is what owns the counters, and asking the
    module-level one queries a storage that never saw the request.
    """
    current_limit: tuple[RateLimitItem, list[str]] | None = getattr(
        request.state, "view_rate_limit", None
    )
    if current_limit is None:
        return None
    request_limiter: Limiter = request.app.state.limiter
    window, identifiers = current_limit
    try:
        reset_time, _remaining = request_limiter.limiter.get_window_stats(window, *identifiers)
    except StorageError:
        return window.get_expiry()
    return max(1, math.ceil(reset_time - time.time()))


def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> UJSONResponse:
    """Render a rate-limit refusal as the envelope, with a real ``Retry-After``.

    Deliberately sync. ``SlowAPIMiddleware`` answers the default limit through
    ``sync_check_limits``, which discards a coroutine handler and falls back to
    slowapi's own ``{"error": ...}`` body — so an async handler here took every
    default-limit 429 off the envelope without any signal that it had.
    """
    retry_after = _retry_after_seconds(request)
    wide_log.warning(
        RATE_LIMIT_EXCEEDED_CODE,
        client_ip=request.client.host
        if request.client
        else request.headers.get("x-forwarded-for", "unknown"),
        path=request.url.path,
        method=request.method,
        retry_after=retry_after,
    )
    return error_response(
        429,
        ErrorEnvelope(message=str(exc.detail), code=RATE_LIMIT_EXCEEDED_CODE),
        headers=None if retry_after is None else {RETRY_AFTER_HEADER: str(retry_after)},
    )


def configure_middleware(app: FastAPI) -> None:
    """
    Configure middleware for the FastAPI application.

    Args:
        app (FastAPI): FastAPI application instance
    """

    # Attach limiter to app state
    app.state.limiter = limiter

    # Exception handler for rate limiting
    # The decorator form, as in app_factory: add_exception_handler is typed
    # Callable[[Request, Exception], ...], which rejects a handler that names
    # the exception it is registered for.
    app.exception_handler(RateLimitExceeded)(rate_limit_handler)

    # Middleware stack, innermost → outermost (add order == inner first).
    # LoggingMiddleware is deliberately the OUTERMOST app middleware: it owns
    # the per-request wide event, and everything that runs inside its
    # dispatch — auth, CORS, timeout, rate limiting, the handler — both gets
    # recorded (an auth 401, a timeout 504 and a rate-limit 429 all emit an
    # http_request line) and can attach context with log.set(). Any response
    # produced outside the boundary is invisible in Loki, which is how
    # timed-out requests used to produce zero telemetry.

    # Rate limiting (innermost — a 429 flows up through the boundary)
    app.add_middleware(RouterAwareSlowAPIMiddleware)

    # Pyinstrument profiling for detailed call stack analysis
    app.add_middleware(ProfilingMiddleware)

    # Request timeout — INSIDE Logging on purpose: its anyio cancel scope is
    # contained in its own __call__, so the 504 it synthesizes travels up to
    # LoggingMiddleware as a normal response and gets emitted. With timeout
    # outside Logging, the cancellation killed the emit and the slowest
    # requests were the only ones with no canonical event.
    app.add_middleware(RequestTimeoutMiddleware)

    # Paid-only gate — INSIDE CORS, and that is the whole point of its position.
    # It runs late enough that WorkOSAuthMiddleware has already put the caller on
    # request.state (which rides scope["state"], so it survives every layer in
    # between), and early enough that no handler can spend money first. It must
    # stay inside CORS: a 402 it returns short-circuits everything further out,
    # so with the gate outside CORS the paywall response would carry no
    # Access-Control-Allow-Origin and the browser would refuse to read the
    # checkout link out of it — the paywall modal would never open.
    app.add_middleware(EntitlementMiddleware)

    # Crash catch-all — the innermost thing inside CORS, and that is the whole
    # point of its position. Starlette answers an uncaught exception in
    # ServerErrorMiddleware, which wraps everything INCLUDING CORS, so the 500
    # envelope it returns carries no Access-Control-Allow-Origin and a browser
    # refuses to read it — the web app shows a generic network failure instead
    # of the error. Converting the crash here covers every layer inside CORS
    # (the gate, the timeout, rate limiting, the router and the handler); the
    # handler registered on the app stays as the last resort for a failure in
    # the middlewares outside this one.
    app.add_middleware(UnhandledExceptionMiddleware)

    # CORS (inside Logging so preflight rejections are visible in Loki)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_allowed_origins(),
        allow_origin_regex=get_allowed_origin_regex(),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["*"],
    )

    # Bot authentication (before WorkOS to allow bot auth to take precedence)
    app.add_middleware(BotAuthMiddleware)

    # PostHog's context must run after WorkOS authentication has populated
    # request.state.user, while still wrapping every downstream handler.
    app.add_middleware(PostHogRequestContextMiddleware)

    # WorkOS authentication — inside the logging boundary, so its rejections
    # are logged and its log.set()/log.error() calls reach the wide event.
    workos_client = AsyncWorkOSClient(
        api_key=settings.WORKOS_API_KEY, client_id=settings.WORKOS_CLIENT_ID
    )
    app.add_middleware(WorkOSAuthMiddleware, workos_client=workos_client)

    # Wide-event boundary — outermost (see block comment above).
    app.add_middleware(LoggingMiddleware)

    # WebSocket wide-event boundary — outermost, after Logging. This is a pure
    # ASGI middleware (not BaseHTTPMiddleware, which drops websocket scope), so
    # add_middleware still works and the app keeps its FastAPI type. It wraps
    # every websocket connection in a log_context() boundary so a handler just
    # calls log.set() like an HTTP handler — see the middleware's docstring.
    app.add_middleware(WebSocketWideEventMiddleware)


def get_allowed_origins() -> list[str]:
    """
    Get allowed origins for CORS based on environment.

    Returns:
        list[str]: List of allowed origins
    """
    # Always include configured frontend URL
    allowed_origins = [settings.FRONTEND_URL]

    # Desktop app embedded Next.js server ports (5174 is preferred; 5175-5180
    # are fallbacks when the preferred port is already in use)
    desktop_origins = [f"http://localhost:{port}" for port in range(5174, 5181)]

    # Add additional origins based on environment
    if settings.ENV == "production":
        # Only allow trusted HTTPS origins in production
        allowed_origins.extend(
            [
                "https://heygaia.io",
                "https://www.heygaia.io",
                "https://heygaia.app",
                # Cloudflare/OpenNext deployment of the web app (migration target).
                "https://cf.heygaia.io",
                *desktop_origins,
            ]
        )
    else:
        # Allow development origins
        allowed_origins.extend(
            [
                "http://localhost:3000",
                "http://192.168.138.215:5173",
                "https://192.168.13.215:5173",
                *desktop_origins,
            ]
        )

    return allowed_origins


def get_allowed_origin_regex() -> str | None:
    """Regex of additional allowed origins (dev-only).

    Matches any localhost origin on any port over http or https, with or
    without a subdomain — covers dev servers on arbitrary ports (e.g. worktree
    ports) and `*.localhost` tunnels alike.
    """
    if settings.ENV == "production":
        return None
    return r"^https?://([a-z0-9-]+\.)?localhost(?::\d+)?$"
