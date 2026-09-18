"""The entry point: a CDP websocket and a goal in, a finished run out."""

from __future__ import annotations

from app.constants.log_tags import LogTag
from app.services.browser.jev.ultrafast.agent import (
    JevHandoffHandler,
    JevRunResult,
    JevUltrafastAgent,
)
from app.services.browser.jev.ultrafast.clients import build_jev_clients
from shared.py.wide_events import log


async def run_jev_ultrafast(
    cdp_url: str,
    goal: str,
    *,
    start_url: str | None = None,
    screenshots: bool = False,
    on_takeover: JevHandoffHandler | None = None,
    on_captcha: JevHandoffHandler | None = None,
) -> JevRunResult:
    """Run the loop on an existing browser session until it is done or blocked.

    ``cdp_url`` is the session's CDP websocket (``cdp_ws`` from the browser
    host). The page it attaches to is left open; disposing the session is the
    caller's job.

    ``on_takeover(reason, category)`` and ``on_captcha(challenge, NONE)`` are the
    handoffs to the human. Each is offered to Jev only when it is supplied, and
    the handler owns the live-view link, the message to the user, keeping the
    session alive and the wait; raising
    :class:`~app.services.browser.exceptions.BrowserHandoffCancelled` from it (the
    user cancelled, or the wait timed out) ends the run the way BLOCKED does.
    """
    client, text_helper = build_jev_clients()
    log.set(
        browser={
            "operation": "jev_ultrafast_run",
            "jev_model": client.model,
            "text_model": text_helper.model,
        }
    )
    agent = await JevUltrafastAgent.start(
        cdp_url,
        goal,
        client=client,
        text_helper=text_helper,
        start_url=start_url,
        screenshots=screenshots,
        on_takeover=on_takeover,
        on_captcha=on_captcha,
    )
    try:
        result = await agent.run()
    finally:
        await agent.close()
        await client.aclose()
        await text_helper.aclose()
    log.set(
        browser={
            "jev_status": result.status,
            "jev_steps": len(result.history),
            "jev_decisions": len(result.decisions),
            "jev_text_calls": len(result.text_calls),
            "jev_elapsed_ms": result.elapsed_ms,
        }
    )
    log.info(f"{LogTag.BROWSER} jev ultrafast run finished")
    return result
