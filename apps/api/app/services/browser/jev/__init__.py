"""Jev decision policy for the browser agent — a port of browser-use/jev-ultrafast.

One Jev evaluation per step picks the operation and the target element from the
page's indexed element table; a small chat model writes text only when needed.
Built by ``services/browser/llm.py`` as the browser's only model; the
runner binds the live browser session so the policy reads the same observation
Browser-Use just took (``chat_model.py``).
"""

from app.services.browser.jev.chat_model import JevChatModel, build_jev_chat_model
from app.services.browser.jev.gateway import JevGatewayClient, JevGatewayError

__all__ = ["JevChatModel", "JevGatewayClient", "JevGatewayError", "build_jev_chat_model"]
