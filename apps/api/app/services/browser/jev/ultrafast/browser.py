"""Observation and execution over one attached CDP session.

Port of jev-ultrafast's ``browser.py``. One snapshot is one ``Runtime.evaluate``
of ``snapshot.js``: visible controls, their names and current values, the page's
visible text, and a freshness marker, read atomically so nothing can shift
between reads. Execution resolves the target's geometry again, hit-tests it, and
refuses a covered or disconnected control.

The reference drives Chrome through Browser Harness's synchronous ``cdp()``
helper; GAIA's browser host exposes a per-session CDP websocket instead, so this
attaches to that socket with ``cdp_use.CDPClient`` and awaits every call. The
protocol traffic is otherwise the same, including the device-metrics override,
focus emulation, and the navigation wait.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
from pathlib import Path
import sys
from time import monotonic
from typing import Any, cast

from cdp_use.client import CDPClient

from app.constants.browser import (
    JEV_ULTRAFAST_CDP_TIMEOUT_SECONDS,
    JEV_ULTRAFAST_NAVIGATION_TIMEOUT_SECONDS,
    JEV_ULTRAFAST_OBSERVE_ATTEMPTS,
    JEV_ULTRAFAST_POLL_SECONDS,
    JEV_ULTRAFAST_VIEWPORT_HEIGHT,
    JEV_ULTRAFAST_VIEWPORT_WIDTH,
    JEV_ULTRAFAST_WAIT_SECONDS,
)
from app.services.browser.exceptions import BrowserAutomationError, BrowserUnavailableError
from app.services.browser.jev.ultrafast.model import Action, PageState

# Atomically read visible content and controls, preserving actual DOM node identity.
READ_STATE = Path(__file__).with_name("snapshot.js").read_text()
MARKER = f"(() => {{ const state={READ_STATE}; return state?.marker ?? null; }})()"

# Wait for the page to be useful again after input: two animation frames or 50ms,
# or — for an editable combobox — visible autocomplete options, capped at 200ms.
SETTLE = """(action => new Promise(resolve => {
  const field=window.__jevFast?.nodes.get(action.node);
  const autocomplete=action.kind==='fill' && field?.getAttribute('role')==='combobox';
  let frames=0, stopped=false;
  const finish=()=>{stopped=true;resolve()};
  setTimeout(finish,autocomplete ? 200 : 50);
  const ready=()=>{
    if (stopped) return;
    const ids=(field?.getAttribute('aria-controls')||field?.getAttribute('aria-owns')||'')
      .split(/\\s+/).filter(Boolean);
    const roots=ids.length ? ids.map(id=>document.getElementById(id)).filter(Boolean) : [document];
    const options=roots.flatMap(root=>[...root.querySelectorAll('[role="option"]')]);
    if (++frames>=2 && (!autocomplete || options.some(e=>{
      const r=e.getBoundingClientRect();
      return r.width && r.height && r.bottom>0 && r.top<innerHeight &&
        e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true});
    }))) finish();
    else requestAnimationFrame(ready);
  };
  requestAnimationFrame(ready);
}))"""

# Code-owned node IDs refer to actual observed elements, never model-generated
# selectors. The target must still be connected, enabled, visible, inside the
# viewport and not covered; a native select additionally needs the option to
# exist and be selectable, and is mutated here so the change event fires once.
RESOLVE_TARGET = """(action => {
  const e=window.__jevFast?.nodes.get(action.node);
  if (!e?.isConnected || e.matches(':disabled') || e.closest('[aria-disabled="true"],[inert]') ||
      !e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true})) return null;
  if (action.kind==='fill' && (e.readOnly || e.getAttribute('aria-readonly')==='true')) return null;
  const r=e.getBoundingClientRect(), x=r.x+r.width/2, y=r.y+r.height/2;
  if (!r.width || !r.height || x<0 || y<0 || x>=innerWidth || y>=innerHeight) return null;
  const hit=document.elementFromPoint(x,y);
  // The reference writes this as e.contains(hit). Obscura (BROWSER_ENGINE=obscura,
  // the default host engine) implements Node.contains as a strict-descendant test,
  // so contains(self) is false there and every click on its own hit target would be
  // rejected as covered. Naming the self case keeps the same meaning on both engines.
  if (!(hit===e || e.contains(hit))) return null;
  if (action.kind==='select') {
    if (e.tagName!=='SELECT' || ![...e.options].some(o=>o.value===action.value &&
        !o.disabled && !o.closest('optgroup[disabled]'))) return null;
    e.value=action.value;
    e.dispatchEvent(new Event('input',{bubbles:true}));
    e.dispatchEvent(new Event('change',{bubbles:true}));
  }
  return {x,y};
})"""

# The reference types straight after its click, because a synthetic CDP click
# focuses an input in Chrome. The host's engine (BROWSER_ENGINE=obscura) dispatches
# the click — handlers run, links follow — but leaves document.activeElement on
# <body>, so Input.insertText would go nowhere and the field would silently stay
# empty. Focusing after the click, not instead of it, keeps the reference's
# behaviour (menus and autocomplete still open on the click) on both engines.
FOCUS_TARGET = """(action => {
  const e=window.__jevFast?.nodes.get(action.node);
  if (!e?.isConnected) return false;
  if (document.activeElement!==e) e.focus();
  return document.activeElement===e;
})"""


# Chrome's select-all accelerator: Cmd+A on macOS, Ctrl+A elsewhere.
_SELECT_ALL_MODIFIER = 4 if sys.platform == "darwin" else 2
_SCROLL_ORIGIN = (550, 650)


class StalePage(BrowserAutomationError):
    """A decision no longer refers to the observed page."""


class JevExecutionError(BrowserAutomationError):
    """A mutation may have half-applied; it must be inspected, never retried."""


class JevUltrafastNodeError(BrowserAutomationError):
    """An action carried something other than an observed node id."""


def fingerprint(state: PageState) -> str:
    """What the page means, not what it looks like — screenshots are excluded."""
    content = {k: state[k] for k in ("url", "text", "actions", "scroll")}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


class UltrafastBrowser:
    """One attached CDP session: observe, check freshness, execute."""

    def __init__(self, cdp: CDPClient, session: str) -> None:
        self._cdp = cdp
        self._session: str | None = session
        self.after_input: Action | None = None

    @classmethod
    async def attach(cls, cdp_url: str, *, start_url: str | None = None) -> UltrafastBrowser:
        """Attach to the session's page, size it, and navigate when asked to."""
        cdp = CDPClient(cdp_url)
        await cdp.start()
        try:
            target_id = await _page_target_id(cdp)
            attached = await _call(cdp, "Target.attachToTarget", targetId=target_id, flatten=True)
            browser = cls(cdp, attached["sessionId"])
            await browser.call(
                "Emulation.setDeviceMetricsOverride",
                width=JEV_ULTRAFAST_VIEWPORT_WIDTH,
                height=JEV_ULTRAFAST_VIEWPORT_HEIGHT,
                deviceScaleFactor=1,
                mobile=False,
            )
            # Keep rAF/menus rendering in an owned background tab, without
            # activating a tab the user is looking at.
            await browser.call("Emulation.setFocusEmulationEnabled", enabled=True)
            if start_url:
                await browser.navigate(start_url)
        except BaseException:
            await cdp.stop()
            raise
        return browser

    async def navigate(self, url: str) -> None:
        await self.call("Page.navigate", url=url)
        deadline = monotonic() + JEV_ULTRAFAST_NAVIGATION_TIMEOUT_SECONDS
        while monotonic() < deadline:
            if await self.evaluate("document.readyState") == "complete":
                return
            await asyncio.sleep(JEV_ULTRAFAST_POLL_SECONDS)

    async def call(self, method: str, **params: object) -> dict[str, Any]:
        if self._session is None:
            raise BrowserUnavailableError("The Jev browser session is closed.")
        return await _call(self._cdp, method, session_id=self._session, **params)

    async def evaluate(self, expression: str) -> object:
        response = await self.call("Runtime.evaluate", expression=expression, returnByValue=True)
        if response.get("exceptionDetails"):
            raise StalePage("Document changed during evaluation")
        return response.get("result", {}).get("value")

    async def observe(self, *, screenshot: bool = False) -> PageState:
        """One snapshot, after letting the previous input settle."""
        if self.after_input is not None:
            action, self.after_input = self.after_input, None
            # Read-only, and it runs after execution was already logged — a
            # navigation that destroys the context here loses nothing.
            with contextlib.suppress(RuntimeError, BrowserAutomationError):
                await self.call(
                    "Runtime.evaluate",
                    expression=f"{SETTLE}({json.dumps(action)})",
                    awaitPromise=True,
                    returnByValue=True,
                )
        for attempt in range(JEV_ULTRAFAST_OBSERVE_ATTEMPTS):
            try:
                return await self._snapshot(screenshot=screenshot)
            except StalePage:
                if attempt == JEV_ULTRAFAST_OBSERVE_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(JEV_ULTRAFAST_POLL_SECONDS)
        raise StalePage("Page did not settle")

    async def fresh(self, page: PageState, action: Action | None = None) -> bool:
        """Scoped for a click/select — the document, the target and its context —
        and a full semantic comparison for everything else."""
        if action is not None and action["kind"] in {"click", "select"}:
            node = action["node"]
            if type(node) is not int:
                return False
            current = await self.evaluate(
                "(() => { const c=window.__jevFast; "
                f"return c ? [c.pageKey(),c.guard(c.nodes.get({node}))] : null; }})()"
            )
            return bool(current == [page["page_key"], page["guards"].get(str(node))])
        return bool(await self.evaluate(MARKER) == page["marker"])

    async def act(self, action: Action, page: PageState, text: str | None = None) -> dict[str, str]:
        if not await self.fresh(page, action):
            raise StalePage("Page changed since this decision. Observe again.")
        if action["kind"] == "wait":
            await asyncio.sleep(JEV_ULTRAFAST_WAIT_SECONDS)
        result = await self._execute(action, text)
        self.after_input = action if action["kind"] != "wait" else None
        return result

    async def close(self) -> None:
        if self._session is None:
            return
        self._session = None
        await self._cdp.stop()

    async def _snapshot(self, *, screenshot: bool) -> PageState:
        info = cast("PageState | None", await self.evaluate(READ_STATE))
        if info is None:
            raise StalePage("Document is navigating")
        info["fingerprint"] = fingerprint(info)
        if screenshot:
            captured = await self.call("Page.captureScreenshot", format="jpeg", quality=72)
            info["screenshot"] = captured["data"]
        return info

    async def _execute(self, action: Action, text: str | None) -> dict[str, str]:
        kind = action["kind"]
        if kind == "scroll":
            x, y = _SCROLL_ORIGIN
            await self.call(
                "Input.dispatchMouseEvent",
                type="mouseWheel",
                x=x,
                y=y,
                deltaX=0,
                deltaY=action["delta"],
            )
        elif kind != "wait":
            await self._input(action, text)
        return {"executed": action["id"]}

    async def _input(self, action: Action, text: str | None) -> None:
        if type(action["node"]) is not int:
            raise JevUltrafastNodeError("Invalid observed node")
        target = await self._resolve(action)
        if target is None:
            if action["kind"] == "select":
                raise JevExecutionError(
                    "Dropdown execution was not confirmed; inspect before retrying."
                )
            raise StalePage("Target changed or is covered. Observe again.")
        if action["kind"] == "select":
            return
        x, y = target["x"], target["y"]
        for event in ("mousePressed", "mouseReleased"):
            await self.call(
                "Input.dispatchMouseEvent", type=event, x=x, y=y, button="left", clickCount=1
            )
        if action["kind"] == "fill":
            await self._focus(action)
            await self.call(
                "Input.dispatchKeyEvent",
                type="keyDown",
                key="a",
                code="KeyA",
                modifiers=_SELECT_ALL_MODIFIER,
                commands=["selectAll"],
            )
            await self.call(
                "Input.dispatchKeyEvent",
                type="keyUp",
                key="a",
                code="KeyA",
                modifiers=_SELECT_ALL_MODIFIER,
            )
            await self.call("Input.insertText", text=text)

    async def _focus(self, action: Action) -> None:
        """Typing into an unfocused field is a silent no-op, so refuse instead."""
        if not await self.evaluate(f"{FOCUS_TARGET}({json.dumps(action)})"):
            raise JevExecutionError("The field did not take focus; nothing typed.")

    async def _resolve(self, action: Action) -> dict[str, float] | None:
        """Geometry and hit-test immediately before input; a select also mutates here."""
        response = await self.call(
            "Runtime.evaluate",
            expression=f"{RESOLVE_TARGET}({json.dumps(action)})",
            returnByValue=True,
        )
        if response.get("exceptionDetails"):
            # A navigation can destroy the result after the change event already fired.
            if action["kind"] == "select":
                raise JevExecutionError(
                    "Dropdown execution was interrupted; inspect before retrying."
                )
            raise StalePage("Document changed during evaluation")
        resolved: dict[str, float] | None = response.get("result", {}).get("value")
        return resolved


async def _call(
    cdp: CDPClient, method: str, *, session_id: str | None = None, **params: object
) -> dict[str, Any]:
    """One CDP round-trip, bounded — a wedged renderer fails one call, not the run."""
    try:
        return await asyncio.wait_for(
            cdp.send_raw(method, params, session_id=session_id),
            timeout=JEV_ULTRAFAST_CDP_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        raise BrowserUnavailableError(f"CDP call {method} timed out") from exc


async def _page_target_id(cdp: CDPClient) -> str:
    """The session's page. The host's proxy already trims this to one context."""
    targets = await _call(cdp, "Target.getTargets")
    for info in targets.get("targetInfos", []):
        if info.get("type") == "page":
            target_id: str = info["targetId"]
            return target_id
    raise BrowserUnavailableError("The browser session has no page target to attach to.")
