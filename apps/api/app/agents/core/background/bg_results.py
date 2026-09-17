"""Durable background-handoff idempotency, keyed by conversation (Redis).

Results themselves go straight to the executor inbox on completion — no
collect call, no bucket. What stays here is the dispatch claim that keeps a
node replay from spawning the same background subagent twice.
"""

from app.constants.hil import HIL_BG_RESULTS_KEY_PREFIX, HIL_BG_RESULTS_TTL_SECONDS
from app.db.redis import redis_cache


async def try_claim_bg_dispatch(conversation_id: str, tool_call_id: str) -> bool:
    """One background dispatch per handoff tool call, durable across node replays.

    A ``handoff`` sharing its node run with a pause re-runs when the pause resumes;
    ``tool_call_id`` lives in the checkpointed AI message,
    so this SETNX makes the side effect (spawning the subagent) idempotent as the
    pre-interrupt code must be. ``True`` = first dispatch, proceed.
    """
    key = f"{HIL_BG_RESULTS_KEY_PREFIX}dispatch:{conversation_id}:{tool_call_id}"
    return bool(await redis_cache.client.set(key, "1", nx=True, ex=HIL_BG_RESULTS_TTL_SECONDS))
