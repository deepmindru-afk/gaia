"""Types for driving one LangGraph agent run: the config, the user it is built
from, and the middleware stack it runs under."""

from dataclasses import dataclass
from typing import Any, TypedDict, cast

from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_config

from app.constants.hil import HIL_RESUME_CONFIG_KEY

# The configurable bag's shape, and the typed read of it, live in
# ``app.models.agent_config`` — a leaf with no langchain import — so
# ``app.utils.timezone`` can read a run's home timezone without pulling the agent
# middleware stack in. Re-exported here under the same names because this module
# is the import site every consumer already uses.
from app.models.agent_config import (
    AgentConfigurable,
    AgentRunConfig,
    ExecutionMode,
    agent_configurable,
)

__all__ = [
    "CONFIGURABLE_OWNED_KEYS",
    "CONFIGURABLE_RUN_SCOPED_KEYS",
    "AgentConfigurable",
    "AgentMiddlewareStack",
    "AgentRunConfig",
    "AgentRunnableConfig",
    "AgentUserContext",
    "AnyAgentMiddleware",
    "ExecutionMode",
    "SilentRunResult",
    "agent_configurable",
    "config_agent_name",
    "current_run_config",
    "runtime_configurable",
]

#: One entry of an agent's middleware stack.
#:
#: ``AgentMiddleware``'s ``StateT`` is erased here because a stack is genuinely
#: heterogeneous — ``SubagentMiddleware`` is typed over ``SubagentState``, the
#: rest over the base ``AgentState`` — and ``StateT`` is invariant, so no
#: narrower common element type exists. This still checks that every entry IS an
#: ``AgentMiddleware``, which the ``Any`` it replaces did not.
#:
#: It lives here, rather than beside the factory that builds stacks, because
#: both that factory and the spawn-graph builder that consumes one need it, and
#: those two must not import each other (see ``spawn_agent``'s module docstring).
AnyAgentMiddleware = AgentMiddleware[Any, Any, Any]

#: An agent's middleware stack, in execution order.
AgentMiddlewareStack = list[AnyAgentMiddleware]


class AgentUserContext(TypedDict, total=False):
    """The user fields ``build_agent_config`` reads — nothing more.

    Deliberately narrower than :class:`~app.models.user_models.AuthenticatedUser`,
    which is assignable to it: only the top-level entries (chat, background
    narration) hold a real request auth context. Every child agent — executor,
    handoff subagents, spawn, the workflow author — reconstructs a bare identity
    bag from its parent's ``configurable``, and typing those as
    ``AuthenticatedUser`` would claim they carry auth-path flags and the whole
    user document, which they do not.

    ``total=False`` because those child bags omit ``timezone`` (they inherit the
    resolved zone from the parent configurable instead).
    """

    user_id: str
    email: str | None
    name: str | None
    timezone: str | None


def current_run_config() -> RunnableConfig:
    """The active ``RunnableConfig`` for the current graph run.

    LangChain's middleware hooks are called as ``(state, runtime)`` and
    ``(request, handler)`` — neither hands the config in as a parameter.
    ``get_config()`` reads it from LangGraph's context-var, the same mechanism
    nodes use. Returns an empty config outside a runnable context, so callers on
    a sync fallback path never have to guard.
    """
    try:
        return get_config()
    except RuntimeError:
        return RunnableConfig()


def config_agent_name(config: RunnableConfig | None) -> str:
    """Which agent a run belongs to, for metric labels (``"unknown"`` when unstamped).

    ``build_agent_config`` stamps ``agent_name`` at the top level, but LangGraph's
    ``ensure_config`` folds every non-standard top-level key into ``configurable``
    before a node sees the config — so inside a graph the key only exists there.
    """
    bag = cast(dict[str, Any], config or {})
    name = (bag.get("configurable") or {}).get("agent_name") or bag.get("agent_name")
    return str(name) if name else "unknown"


def runtime_configurable(request: ToolCallRequest) -> AgentConfigurable:
    """The same view as :func:`agent_configurable`, reached through a middleware
    ``ToolCallRequest``.

    A tool intercepted by middleware gets its config off ``request.runtime``
    rather than as an injected ``RunnableConfig``, and that attribute is typed
    loosely enough that it may not be a mapping at all — hence the guard.
    Returns an empty view outside a graph.
    """
    runtime = getattr(request, "runtime", None)
    config = getattr(runtime, "config", None)
    if not isinstance(config, dict):
        return {}
    return agent_configurable(cast(RunnableConfig, config))


class AgentRunnableConfig(RunnableConfig):
    """What ``build_agent_config`` returns: a ``RunnableConfig`` plus ``agent_name``.

    ``agent_name`` is GAIA's own key, not LangGraph's — the graph drivers gate
    text accumulation on ``config["agent_name"] == "comms_agent"`` so only the
    user-facing agent's tokens reach the client. Subclassing rather than a
    parallel type keeps the value directly passable to ``graph.astream(config=...)``.

    ``configurable`` keeps LangGraph's ``dict[str, Any]`` annotation because a
    TypedDict field cannot be narrowed in a subclass without making the result
    unassignable to ``RunnableConfig`` — which is the whole point of
    subclassing. :class:`AgentConfigurable` names what is inside it, and
    :func:`agent_configurable` is how you read it.
    """

    agent_name: str


@dataclass(frozen=True)
class SilentRunResult:
    """What one ``call_agent_silent`` turn produced.

    ``queued_task_id`` is set when the turn's comms agent delegated to the
    executor and that dispatch was QUEUED behind an in-flight run for the same
    conversation instead of running. The ``message`` is then an acknowledgement
    of work that has not started, so a caller must not record the turn as work
    done. It is ``None`` whenever an executor actually ran.
    """

    message: str
    tool_data: dict[str, Any]
    queued_task_id: str | None = None
    #: The executor this turn delegated to ended in an error. ``message`` is
    #: then comms' account of that error, not a result; ``executor_failure``
    #: is the error itself, or why the wait for it gave up.
    executor_failed: bool = False
    executor_failure: str | None = None


# What survives a queue hop / HIL resume. Every GAIA-owned configurable key is
# safe to carry by construction, so AgentConfigurable IS the allowlist: the
# hand-maintained list this replaces had fallen behind it and was dropping the
# OpenRouter provider pin, plan_type, root_request_id, langfuse_trace_id and the
# HIL intent judge's user_messages — a queued run was not a smaller run, it was a
# different one. What must still be filtered is LangGraph's own runtime keys
# (checkpoint_ns, checkpoint_id, __pregel_*, Runtime objects), which are exactly
# the keys NOT declared on AgentConfigurable. Derived from the TypedDict at
# import time, so it belongs beside it: app/constants/ is a leaf, and reaching
# agent_models from there dragged langchain into every constants importer.
CONFIGURABLE_OWNED_KEYS: frozenset[str] = frozenset(AgentConfigurable.__annotations__)

# Owned keys that are nonetheless scoped to ONE dispatch and must not ride along
# to the next: hil_resume_replay means "this exact call is a replay", so carrying
# it would make a fresh run probe its subagent threads for interrupts it cannot have.
CONFIGURABLE_RUN_SCOPED_KEYS: frozenset[str] = frozenset({HIL_RESUME_CONFIG_KEY})
