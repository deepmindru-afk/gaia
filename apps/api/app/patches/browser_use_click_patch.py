"""Click through the element itself: a dispatched press is slow and aims blind.

Re-measured on Obscura 2026-09-19, correcting an earlier reading taken against
occluded elements. Input.dispatchMouseEvent does work: the event lands on the
exact x/y given (slope 1.0000, intercept 0 over 28 points), isTrusted is true,
and in 32 of 32 probes it hit the page's own elementFromPoint. It is still the
wrong tool: a pressed button costs 450ms to 5.6s, against 0.4ms for a move, and
aiming at a rect centre needs occlusion data Browser-Use takes from the DOM
snapshot, whose geometry this engine fabricates (see jev/viewport.py).

So scroll-into-view and the click go in one Runtime.callFunctionOn on the
element handle, reporting the measured centre as the click point. The cost is
isTrusted, which a page gating on it will refuse.

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
