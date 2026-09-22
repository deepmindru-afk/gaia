"""ARQ cron task: resolve expired HIL approvals and re-dispatch crashed resumes."""

from collections.abc import Mapping

from app.services.hil.resolution import SweepCounts, sweep_approvals
from shared.py.wide_events import log


async def sweep_hil_approvals(ctx: Mapping[str, object]) -> str:  # noqa: ARG001 -- contract
    """Every-minute sweep. See sweep_approvals for the two passes."""
    counts: SweepCounts = await sweep_approvals()
    log.set(
        expired_count=counts["expired"],
        redispatched_count=counts["redispatched"],
        deferred_subagent_count=counts["deferred_subagent"],
    )
    return f"expired={counts['expired']} redispatched={counts['redispatched']} deferred_subagent={counts['deferred_subagent']}"
