"""The shape of one agent run's ``config["configurable"]`` bag, and the typed
way to read it.

A leaf on purpose: it imports nothing from langchain, langgraph or the rest of
``app.models``, so ``app.utils.timezone`` — which every user model, and so every
test worker at collection time, pulls in — can read a run's home timezone
without dragging langchain's agent middleware (and the ~1.5 s ``transformers``
import behind it) along. ``app.models.agent_models`` re-exports everything here
under the same names, so consumers keep importing from there.
"""

from collections.abc import Mapping
from typing import Any, Literal, TypedDict, cast

#: All home_timezone_from_config needs of a run config: a string-keyed mapping
#: it reads configurable out of. Naming RunnableConfig here pulled langchain_core
#: into every importer. RunnableConfig is a TypedDict and satisfies this.
AgentRunConfig = Mapping[str, object]


#: The execution mode a run is in. ``background`` runs have no user waiting on
#: them, which is what suppresses HIL pauses and the executor wake-up path.
ExecutionMode = Literal["interactive", "background"]


class AgentConfigurable(TypedDict, total=False):
    """Every key GAIA puts in a run's ``config["configurable"]`` — the whole agreement.

    This is the one place the bag is described. Anything an agent, tool,
    middleware or node expects to find in ``configurable`` is declared here, so
    a reader can answer "what is actually in this thing" without grepping, and
    mypy rejects a subscript for a key nobody writes.

    ``total=False`` is the honest shape, not a shortcut. ``build_agent_config``
    is the main producer and fills nearly all of it, but several paths
    legitimately construct a partial bag — a bare ``{"thread_id": ...}`` to
    address a checkpoint, ``{"user_id": ...}`` for a one-off silent run, the
    queue's serializable subset — and every consumer already reads through
    ``.get()`` with a fallback.

    It stays a ``TypedDict`` over a plain dict (not a Pydantic model) because
    LangGraph owns the object at runtime: it merges its own keys in
    (``checkpoint_ns``, ``checkpoint_id``, ``__pregel_*``), passes it to
    ``llm.with_config(configurable=...)``, and checkpoints it. Declaring the
    GAIA-owned keys describes that bag without trying to own it — read
    :func:`agent_configurable` for how consumers get here from a
    ``RunnableConfig``.
    """

    # --- identity: who and which conversation ------------------------------
    #: LangGraph's checkpoint thread. For child agents this is the WRAPPED
    #: thread (``<integration>_executor_<conv>``), never the conversation id.
    thread_id: str
    #: The TRUE conversation id, established once by comms and inherited
    #: parent-overrides. HIL approvals, notifications and the executor queue
    #: key on this, never on ``thread_id``.
    conversation_id: str
    user_id: str | None
    email: str | None
    user_name: str
    #: IANA home zone, DST-aware. Read via ``home_timezone_from_config``.
    user_timezone: str
    #: The user's own recent turns, verbatim and oldest first. The HIL intent
    #: judge grounds gated calls against these, so they are never an agent's
    #: paraphrase of the request.
    user_messages: list[str] | None
    #: The live turn's request as typed, NOT clipped: user_messages is capped at
    #: HIL_JUDGE_MAX_TURN_CHARS so it cannot be the verbatim copy. Set once by
    #: comms; absent for non-chat roots, which have no literal user turn.
    user_request: str | None
    user_message_id: str
    #: The comms turn's own bot message id. Threaded into call_executor so a HIL
    #: pause that resumes reconciles onto this SAME message instead of minting a
    #: rival one. See ExecutorRun.bot_message_id and _record_pause.
    bot_message_id: str
    #: Onboarding preferences and writing style, set once at a run tree's root
    #: and inherited unchanged by every child. Absent when the root had none; the
    #: context sections that read them degrade to no section rather than guess.
    user_preferences: dict[str, Any] | None
    writing_style: dict[str, Any] | None
    #: One id for the WHOLE user turn, minted at the top-level build_agent_config
    #: and inherited by every child. The accounting middleware keys the tree's
    #: aggregate token ceiling on it, so it binds across the tree, not per graph.
    root_request_id: str

    # --- model selection ----------------------------------------------------

    #: THE model selection: a serialized ModelLane, resolved once per turn and
    #: inherited verbatim by every child, queue hop and HIL resume. The only
    #: model key GAIA code reads.

    #: Absent on a bag written before lanes existed; ModelLane.from_configurable
    #: returns None for those and the caller resolves a fresh lane.
    lane: dict[str, Any]
    #: LangChain's own binding keys, written from the lane and read ONLY by its
    #: field resolution. Never read these in GAIA code: they are the expansion,
    #: not the decision. Read lane.
    provider: str
    model: str
    model_kwargs: dict[str, Any]
    reasoning: dict[str, Any]

    # --- run scope ----------------------------------------------------------
    selected_tool: str | None
    tool_category: str | None
    subagent_id: str | None
    #: Shared VFS session, held constant across the executor and the handoff
    #: subagents it spawns so all resolve paths against one workspace.
    vfs_session_id: str | None
    #: OpenRouter sticky-routing key — the conversation id, pinned on every
    #: request so OpenRouter routes the whole conversation tree to the
    #: provider holding the warm prompt cache (see build_agent_config).
    session_id: str | None
    stream_id: str | None
    active_todo_id: str | None
    execution_mode: ExecutionMode
    #: The specific channel (``ConversationSource`` value) and its generalized
    #: category (``SourceCategory`` value).
    conversation_source: str | None
    source_category: str
    #: The resolved plan tier, stamped by resolve_lane and inherited by children.
    #: The budget wall reads it to avoid a Redis plan lookup on the hot path;
    #: absent, the wall derives the tier from the cached plan itself.
    plan_type: str

    # --- workflow context (must survive queueing) ---------------------------
    workflow_id: str
    workflow_title: str
    workflow_notify_on_completion: bool
    #: A replay stopped partway in THIS fire: its own record of what already ran.
    #: Carried to the executor verbatim because comms cannot be trusted to
    #: transcribe "do not repeat these" into its task.
    playbook_fallback: str | None
    #: The calls a stopped replay made this fire, as ``RecordedCall`` dumps, so a
    #: rewrite may freeze them. See ``PLAYBOOK_REPLAYED_CALLS_KEY``.
    playbook_replayed_calls: list[dict[str, Any]] | None

    # --- tracing ------------------------------------------------------------
    #: Stashed here so child agents spawned via ``asyncio.create_task`` re-emit
    #: the same trace from their own ``build_agent_config`` call.
    langfuse_trace_id: str
    langfuse_tags: list[str]

    # --- internal flags -----------------------------------------------------
    #: Set only on a HIL resume re-dispatch; the handoff tool probes it to tell
    #: a replayed call from a fresh one. Keyed by ``HIL_RESUME_CONFIG_KEY``.
    hil_resume_replay: bool
    #: DEV-ONLY: the DEV_MODEL_OPTIONS key picked for the executor in the dev
    #: model switcher. The executor builds its own configurable rather than
    #: inheriting comms's lane wholesale, so the choice rides down here.
    dev_executor_model: str


def agent_configurable(config: AgentRunConfig | None) -> AgentConfigurable:
    """The GAIA-owned keys of a run's ``configurable``, typed.

    The single way to READ a ``configurable``. Every consumer used to inline
    ``config.get("configurable", {}).get(key)``, which yields ``Any`` and so
    checks neither the key nor the value type.

    A ``cast`` rather than a validation step: the dict is built by
    ``build_agent_config`` as an :class:`AgentConfigurable` and is correct by
    construction (Type Safety item 12). LangGraph's own runtime keys ride along
    in the same dict and are simply not part of this view.

    Reads only. The ``or {}`` means a config with no ``configurable`` yields a
    throwaway dict, so a write through it would be silently dropped — the few
    sites that mutate a live bag index ``config["configurable"]`` directly and
    keep today's ``KeyError`` when it is absent.
    """
    return cast(AgentConfigurable, (config or {}).get("configurable") or {})
