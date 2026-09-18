"""The two models the loop talks to, built from one credential.

Jev decides and the text helper writes the values it types, and both ride the
same OpenRouter key, so a deployment either has the lane or it does not. Built
in one place because a second copy is a second answer to "is this configured".
"""

from __future__ import annotations

from app.config.settings import settings
from app.services.browser.exceptions import BrowserUnavailableError
from app.services.browser.jev.gateway import JevGatewayClient
from app.services.browser.jev.ultrafast.model import JevTextHelper


def build_jev_clients() -> tuple[JevGatewayClient, JevTextHelper]:
    """Return the decision client and the text helper, or refuse without a key."""
    key = settings.OPENROUTER_API_KEY
    if not key:
        raise BrowserUnavailableError(
            "OPENROUTER_API_KEY is not set; Jev decisions and the text helper both need it."
        )
    return (
        JevGatewayClient(
            api_key=key,
            model=settings.BROWSER_USE_JEV_MODEL,
            url=settings.BROWSER_USE_JEV_DECISIONS_URL,
        ),
        JevTextHelper(
            api_key=key,
            model=settings.BROWSER_USE_JEV_TEXT_MODEL,
            url=settings.BROWSER_USE_JEV_TEXT_URL,
        ),
    )
