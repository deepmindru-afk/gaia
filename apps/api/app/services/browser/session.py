"""Browser-host session lifecycle — create, live-view URL, guaranteed release.

The *infrastructure* layer: it talks to gaia-browser-host (via host_client)
and knows nothing about Browser-Use or the agent. The session is always released
on exit — success, error, or cancellation — so no browser context is ever
orphaned. It seeds the user's saved login for the target domain before handing
the session to the agent, persists the returned login back when the session
ends, and exposes a live-view URL served from our own authenticated API.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from urllib.parse import urlsplit

from app.constants.browser import (
    BROWSER_HANDOFF_KEEPALIVE_SECONDS,
    HANDOFF_AUTORESOLVE_POLL_SECONDS,
    HANDOFF_AUTORESOLVE_STABLE_POLLS,
    HandoffDecision,
)
from app.constants.log_tags import LogTag
from app.services.browser import host_client
from app.services.browser.exceptions import BrowserUnavailableError
from app.services.browser.handoff import resolve_handoff
from app.services.browser.live_view import live_view_url
from app.services.browser.registry import register_session, unregister_session
from app.services.browser.storage_persistence import (
    domain_of,
    load_storage_state,
    save_storage_state,
)
from shared.py.wide_events import log


@dataclass(frozen=True, slots=True)
class BrowserHostSession:
    """Client-side handle to one browser-host context (CDP + live endpoints)."""

    session_id: str
    cdp_url: str
    live_view_url: str
    context_id: str


async def keep_session_alive(session_id: str) -> None:
    """Periodically reset the host's idle clock while a handoff is pending.

    A paused session has no traffic and the idle TTL is shorter than the
    handoff timeout. Best-effort: a failed touch is logged. Run under
    spawn_background_task and cancel when the handoff resolves.
    """
    while True:
        await asyncio.sleep(BROWSER_HANDOFF_KEEPALIVE_SECONDS)
        try:
            await host_client.touch_session(session_id)
        except BrowserUnavailableError as exc:
            log.warning(
                f"{LogTag.BROWSER} Browser handoff keepalive failed",
                error_type=type(exc).__name__,
                browser={"session_id": session_id, "operation": "handoff_keepalive"},
            )


# Path fragments that mean "still inside the auth flow": a login commonly walks
# /login -> /sessions/two-factor -> /verify, each hop a real navigation, so
# navigation alone must not end the handoff.
_AUTH_PATH_MARKERS = (
    "login",
    "signin",
    "sign-in",
    "auth",
    "session",
    "two-factor",
    "two_factor",
    "2fa",
    "mfa",
    "otp",
    "verify",
    "verification",
    "challenge",
    "password",
    "consent",
    "oauth",
)


def _is_auth_url(url: str | None) -> bool:
    """Whether the URL still looks like part of a sign-in flow."""
    if not url:
        return False
    path = urlsplit(url).path.lower()
    return any(marker in path for marker in _AUTH_PATH_MARKERS)


def _navigated_away(start: str | None, current: str | None) -> bool:
    """Return whether the sign-in is visibly finished.

    True when the page moved to a different scheme+host+path (query/fragment
    ignored) that is no longer part of the auth flow; staying inside auth
    (2FA, OTP, verification) is NOT done — see _AUTH_PATH_MARKERS.
    """
    if not start or not current:
        return False
    a, b = urlsplit(start), urlsplit(current)
    if (a.scheme, a.netloc, a.path) == (b.scheme, b.netloc, b.path):
        return False
    return not _is_auth_url(current)


async def auto_resolve_handoff_on_navigation(
    handoff_id: str, session_id: str, user_id: str
) -> None:
    """Auto-complete a login handoff once the page navigates off the sign-in URL.

    Best-effort: the manual "I'm done" races this through resolve_handoff
    (first write wins). Debounced so a transient mid-login redirect does not
    resolve early. Run under spawn_background_task and cancel on resolve.
    """
    try:
        start = (await host_client.get_session(session_id)).url
    except BrowserUnavailableError:
        return
    stable = 0
    while True:
        await asyncio.sleep(HANDOFF_AUTORESOLVE_POLL_SECONDS)
        try:
            current = (await host_client.get_session(session_id)).url
        except BrowserUnavailableError:
            return
        if not _navigated_away(start, current):
            stable = 0
            continue
        stable += 1
        if stable >= HANDOFF_AUTORESOLVE_STABLE_POLLS:
            await resolve_handoff(
                handoff_id,
                HandoffDecision.CONTINUE,
                user_id,
                "Looks like you're done here, resuming.",
            )
            return


@asynccontextmanager
async def browser_session(
    *,
    user_id: str,
    start_url: str | None = None,
) -> AsyncIterator[BrowserHostSession]:
    """Create a browser-host session, yield it, and always release it.

    Seeds the user's saved storage_state for start_url's domain, registers
    live-view ownership, and on exit persists storage_state and unregisters.
    Raises BrowserUnavailableError or BrowserConcurrencyLimit from the host.
    """
    domain = domain_of(start_url)
    storage_state = await load_storage_state(user_id, domain)

    host = await host_client.create_session(storage_state)
    session = BrowserHostSession(
        session_id=host.session_id,
        cdp_url=host.cdp_ws,
        live_view_url=live_view_url(host.session_id),
        context_id=host.context_id,
    )
    log.set(browser={"session_id": session.session_id, "operation": "create"})
    log.info(f"{LogTag.BROWSER} Browser session created")

    try:
        registered = await register_session(session.session_id, user_id, live_ws=host.live_ws)
        if not registered:
            # Without the ownership entry the live-view link we hand the user can
            # never authorize; fail the session (release runs in the finally below)
            # instead of stranding them in a handoff they can't open.
            raise BrowserUnavailableError(
                "Could not register the browser session (storage unavailable)."
            )
        yield session
    finally:
        try:
            returned_state = await host_client.delete_session(session.session_id)
            await save_storage_state(user_id, domain, returned_state)
            log.info(f"{LogTag.BROWSER} Browser session released")
        except Exception as exc:
            log.warning(
                f"{LogTag.BROWSER} Failed to release browser session",
                error_type=type(exc).__name__,
                browser={"session_id": session.session_id, "operation": "release_failed"},
            )
        await unregister_session(session.session_id)
