"""The entry point: a CDP websocket and a goal in, a finished run out."""

from __future__ import annotations

from app.config.settings import settings
from app.constants.log_tags import LogTag
from app.services.browser.exceptions import BrowserUnavailableError
from app.services.browser.jev.gateway import JevGatewayClient
from app.services.browser.jev.ultrafast.agent import JevRunResult, JevUltrafastAgent
from app.services.browser.jev.ultrafast.model import JevTextHelper
from shared.py.wide_events import log


async def run_jev_ultrafast(
    cdp_url: str,
    goal: str,
    *,
    start_url: str | None = None,
    screenshots: bool = False,
) -> JevRunResult:
    """Run the loop on an existing browser session until it is done or blocked.

    ``cdp_url`` is the session's CDP websocket (``cdp_ws`` from the browser
    host). The page it attaches to is left open; disposing the session is the
    caller's job.
    """
    key = settings.OPENROUTER_API_KEY
    if not key:
        raise BrowserUnavailableError(
            "OPENROUTER_API_KEY is not set; Jev decisions and the text helper both need it."
        )
    client = JevGatewayClient(
        api_key=key,
        model=settings.BROWSER_USE_JEV_MODEL,
        url=settings.BROWSER_USE_JEV_DECISIONS_URL,
    )
    text_helper = JevTextHelper(
        api_key=key,
        model=settings.BROWSER_USE_JEV_TEXT_MODEL,
        url=settings.BROWSER_USE_JEV_TEXT_URL,
    )
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
