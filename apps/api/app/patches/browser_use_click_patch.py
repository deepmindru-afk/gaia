"""Click through the element itself, because Obscura drops synthetic mouse events.

Measured on the Obscura host: Input.dispatchMouseEvent is accepted and then
discarded -- dispatched at an element's true centre it fires no listener in
the page and costs ~1.8s a call, so Browser-Use reports "Clicked" on a page
that never changed, twenty times in one run. Runtime and DOM work normally
there, and the element's own rect is honest (its snapshot absolute_position is
not, see jev/viewport.py).

So this keeps Browser-Use's scroll-into-view and clicks in one
Runtime.callFunctionOn on the element handle: read getBoundingClientRect, call
element.click(), report that centre as the click point the step card draws.
A node the page cannot measure goes back to Browser-Use's own path.

Pinned to browser-use==0.11.13; the import fails loudly if the method moves.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from browser_use.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog

from app.constants.log_tags import LogTag
from shared.py.wide_events import log

if TYPE_CHECKING:
    from browser_use.dom.views import EnhancedDOMTreeNode

# One round trip: measure where the element really is, then press it there.
_CLICK_JS = """function() {
  const rect = this.getBoundingClientRect();
  if (!rect.width && !rect.height) return null;
  this.click();
  return {click_x: rect.left + rect.width / 2, click_y: rect.top + rect.height / 2};
}"""

# Browser-Use's own wait after its JavaScript click, for a dialog or a
# navigation the click starts to reach its watchdogs.
_SETTLE_SECONDS = 0.05

_original_click_element_node_impl = DefaultActionWatchdog._click_element_node_impl


def _needs_browser_use(node: EnhancedDOMTreeNode) -> bool:
    """Say whether this is a <select> or a file input, which Browser-Use rejects with its own message."""
    tag = (getattr(node, "tag_name", "") or "").lower()
    attributes: dict[str, str] = getattr(node, "attributes", None) or {}
    return tag == "select" or (tag == "input" and attributes.get("type", "").lower() == "file")


async def _click_element_node_impl(
    self: DefaultActionWatchdog, element_node: EnhancedDOMTreeNode
) -> dict[str, Any] | None:
    """Click the element in the page and return the real centre Browser-Use records."""
    if _needs_browser_use(element_node):
        return await _original_click_element_node_impl(self, element_node)

    cdp_session = await self.browser_session.cdp_client_for_node(element_node)
    backend_node_id = element_node.backend_node_id
    try:
        await cdp_session.cdp_client.send.DOM.scrollIntoViewIfNeeded(
            params={"backendNodeId": backend_node_id}, session_id=cdp_session.session_id
        )
    except Exception as exc:
        # Already-visible elements on some engines answer this with an error;
        # the click below still measures wherever the element ended up.
        log.debug(
            f"{LogTag.BROWSER} Could not scroll an element into view before clicking",
            error_type=type(exc).__name__,
        )

    resolved: dict[str, Any] = dict(
        await cdp_session.cdp_client.send.DOM.resolveNode(
            params={"backendNodeId": backend_node_id}, session_id=cdp_session.session_id
        )
    )
    object_id = (resolved.get("object") or {}).get("objectId")
    if not object_id:
        return await _original_click_element_node_impl(self, element_node)

    response: dict[str, Any] = dict(
        await cdp_session.cdp_client.send.Runtime.callFunctionOn(
            params={
                "functionDeclaration": _CLICK_JS,
                "objectId": object_id,
                "returnByValue": True,
            },
            session_id=cdp_session.session_id,
        )
    )
    point = (response.get("result") or {}).get("value")
    if response.get("exceptionDetails") or not isinstance(point, dict):
        return await _original_click_element_node_impl(self, element_node)

    await asyncio.sleep(_SETTLE_SECONDS)
    return {"click_x": float(point["click_x"]), "click_y": float(point["click_y"])}


def apply() -> None:
    """Route every element click through the page's own rect and click handler."""
    # type.__setattr__ mirrors the stealth patch: an honest rebind of a private
    # coroutine method that keeps mypy satisfied without an ignore.
    type.__setattr__(DefaultActionWatchdog, "_click_element_node_impl", _click_element_node_impl)


apply()
