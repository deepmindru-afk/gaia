"""Jev decision policy for the browser agent — a port of browser-use/jev-ultrafast.

One Jev evaluation per step picks the operation and the target element from the
page's indexed element table; a small chat model writes text only when needed.
Wired in behind ``BROWSER_USE_JEV_ENABLED`` by ``services/browser/llm.py``; the
runner binds the live browser session so the policy reads the same observation
Browser-Use just took (``chat_model.py``).
"""

from app.services.browser.jev.chat_model import JevChatModel, build_jev_chat_model
from app.services.browser.jev.gateway import JevGatewayClient, JevGatewayError

__all__ = ["JevChatModel", "JevGatewayClient", "JevGatewayError", "build_jev_chat_model"]
