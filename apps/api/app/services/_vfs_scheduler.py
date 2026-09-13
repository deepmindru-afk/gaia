"""Shared orchestration for VFS sync glue modules.

The glue modules (gaia_tasks_fs, user_todos_fs) share two patterns:
hash-gated sync (bail on missing mount, hash Mongo docs against the on-disk
marker, materialize off-thread only on mismatch — run_hashed_sync) and
fire-and-forget scheduling (wrap an async sync fn into a schedule(user_id)
that spawns a background task and never raises — make_scheduler).

Both helpers are deliberately small so the glue modules read identically; a
third VFS area slots in by providing the same five callbacks.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
import contextlib
from pathlib import Path
from typing import Any, TypeVar

from app.services.storage._vfs_common import (
    catalog_signature,
    read_marker,
    write_marker,
)
from app.services.storage.juicefs import _is_mounted, user_workspace_path
from app.services.storage.metrics import fs_timer
from app.utils.background_tasks import spawn_background_task
from shared.py.wide_events import UserContext, log, wide_task

# Generic over each module's projection TypedDict. Mapping[str, Any] is the
# right bound: TypedDicts structurally satisfy Mapping, so callers pass their
# concrete TypedDict here without a cast, while the helper still gets d["id"].
ProjectionT = TypeVar("ProjectionT", bound=Mapping[str, Any])


async def run_hashed_sync(
    user_id: str,
    *,
    fs_op: str,
    fetch_fn: Callable[[str], Awaitable[list[ProjectionT]]],
    per_doc_sig_fn: Callable[[ProjectionT], str],
    materialize_fn: Callable[[Path, list[ProjectionT], str], int],
    guide_md: str,
    catalog_marker_path_fn: Callable[[Path], Path],
    log_name: str,
) -> int:
    """Run a hash-gated VFS sync for user_id; return the number of doc bodies rewritten.

    0 means either the mount was missing or the on-disk signature already
    matched Mongo — both are no-ops from the caller's POV. fs_timer wraps even
    the no-op path so dashboards see the call and can spot a runaway caller.
    """
    if not _is_mounted():
        return 0
    async with fs_timer(fs_op):
        docs = await fetch_fn(user_id)
        per_doc = {d["id"]: per_doc_sig_fn(d) for d in docs}
        expected = catalog_signature(per_doc)
        u_root = user_workspace_path(user_id)
        marker_path = catalog_marker_path_fn(u_root)
        if read_marker(marker_path) == expected:
            return 0
        written = await asyncio.to_thread(materialize_fn, u_root, docs, guide_md)
        write_marker(marker_path, expected)
        log.set(vfs_sync={"name": log_name, "written": written, "total": len(docs)})
        return written


def make_scheduler(
    sync_fn: Callable[[str], Awaitable[int]],
    *,
    log_name: str,
) -> Callable[[str], None]:
    """Build a schedule(user_id) wrapper around sync_fn.

    No-ops when unmounted or when no asyncio loop is running (e.g. workers
    calling tools synchronously at startup). Logs but never raises — a
    fire-and-forget call must not crash its caller — and spawns via
    spawn_background_task so the task isn't garbage-collected mid-flight.
    """

    async def _safe(user_id: str) -> None:
        # Own wide_task scope: no request middleware runs in this fire-and-forget
        # task, so this is what makes the result and failure emit a queryable
        # wide event. wide_task already records failure, so suppress the re-raise.
        with contextlib.suppress(Exception):
            async with wide_task(log_name, user=UserContext(id=user_id)):
                await sync_fn(user_id)

    def schedule(user_id: str) -> None:
        if not _is_mounted():
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        spawn_background_task(_safe(user_id))

    return schedule
