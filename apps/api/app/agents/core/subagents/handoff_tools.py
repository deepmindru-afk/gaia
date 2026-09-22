"""Subagent tools — per-user MCP delegation.

The handoff tool delegates ONLY to per-user MCP integrations (custom/auth-required
MCP) that run in their own per-user graph. Provider and built-in integrations
activate in-context instead and are redirected at resolve time. Subagent
identity/metadata comes from agents/core/subagents/registry.py.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
import re
import time
from typing import Annotated
from uuid import uuid4

from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.config import get_stream_writer
from langgraph.errors import GraphBubbleUp
from langgraph.store.base import BaseStore, PutOp
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict

from app.agents.context.tiers import AgentTier
from app.agents.core.background.bg_results import try_claim_bg_dispatch
from app.agents.core.background.running_registry import RunningSubagents
from app.agents.core.background.session import (
    claim_bg_integration,
    has_bg_integration,
    release_bg_integration,
)
from app.agents.core.background.subagent_runner import BackgroundHandoff, run_subagent_background
from app.agents.core.graph_manager import CompiledAgentGraph
from app.agents.core.subagents.call_record import append_call_record
from app.agents.core.subagents.provider_subagents import (
    SubagentUnavailableError,
    create_subagent_for_user,
)
from app.agents.core.subagents.registry import (
    all_subagents,
    foreign_provider_named_in,
    get_subagent_by_id,
)
from app.agents.core.subagents.subagent_helpers import (
    create_subagent_system_message,
)
from app.agents.core.subagents.subagent_runner import (
    SubagentExecutionContext,
    ThreadSeed,
    build_initial_messages,
    execute_subagent_stream,
    recover_from_checkpoint,
    resume_for_gate,
    subagent_row_id,
)
from app.agents.tools.core.retrieval import preloaded_startup_docs
from app.constants.cache import SUBAGENT_CACHE_PREFIX, SUBAGENT_CACHE_TTL
from app.constants.hil import HIL_RESUME_CONFIG_KEY
from app.constants.log_tags import LogTag
from app.db.redis import get_cache, set_cache
from app.db.repositories.integrations import integration_repository
from app.helpers.agent_helpers import AgentIdentity, AgentThread, build_agent_config
from app.helpers.namespace_utils import derive_integration_namespace
from app.models.agent_models import (
    AgentConfigurable,
    AgentUserContext,
    RunningSubagent,
    agent_configurable,
)
from app.models.subagent_models import Subagent
from app.services.hil.approvals_store import list_parked_subagents_for_conversation
from app.services.integrations.integration_resolver import IntegrationResolver
from app.services.mcp.mcp_token_store import MCPTokenStore
from app.services.oauth.oauth_service import check_integration_status
from app.services.provider_metadata_service import get_provider_metadata
from app.utils.agent_utils import (
    IntegrationMetadata,
    SubagentStartDetails,
    format_subagent_end_event,
    format_subagent_start_event,
    parse_subagent_id,
)
from app.utils.background_tasks import spawn_background_task
from app.utils.integration_checker import request_integration_connection
from shared.py.wide_events import log

SUBAGENTS_NAMESPACE = ("subagents",)


@dataclass(frozen=True)
class CustomMcpIndexRequest:
    """The custom-MCP identity/description cluster indexed for handoff discovery.

    Args:
        integration_id: Unique ID of the custom integration (12-char hex)
        name: Display name of the integration
        description: Description of what the integration does
        server_url: MCP server URL for namespace derivation
        tools: The MCP's tools — their names/summaries are embedded so the
            subagent ranks for what it can actually do (e.g. a "get_meetings"
            tool surfaces on a "meetings" query) instead of generic boilerplate.
    """

    integration_id: str
    name: str
    description: str
    server_url: str | None = None
    tools: list[BaseTool] | None = None


@dataclass(frozen=True)
class _HandoffDispatch:
    """Everything besides the execution context that one handoff dispatch needs."""

    metadata: IntegrationMetadata | None
    agent_name: str
    integration_id: str
    tool_call_id: str
    probe_parked: bool = False
    record_calls: bool = False


class CustomMcpSubagent(BaseModel):
    """A custom MCP resolved as a handoff target, as cached under ``SUBAGENT_CACHE_PREFIX``.

    Fields default because the cache holds whatever an earlier resolution wrote;
    ``_resolve_custom_mcp_subagent`` refuses one with no ``id``.
    """

    model_config = ConfigDict(extra="ignore")

    id: str = ""
    name: str | None = None
    source: str | None = None
    managed_by: str | None = None
    mcp_config: dict[str, object] | None = None
    icon_url: str | None = None
    subagent_config: None = None


class _CustomIntegrationDoc(BaseModel):
    """The ``IntegrationResolver.custom_doc`` keys a custom-MCP handoff target copies.

    ``mcp_config`` stays the stored mapping: it is cached verbatim, never read here.
    """

    model_config = ConfigDict(extra="ignore")

    integration_id: str = ""
    name: str | None = None
    mcp_config: dict[str, object] | None = None
    icon_url: str | None = None


class _RunMetadata(BaseModel):
    """The ``metadata`` key of a ``RunnableConfig`` the handoff falls back on."""

    model_config = ConfigDict(extra="ignore")

    user_id: str | None = None


def _extract_service_username(metadata: dict[str, str] | None) -> str | None:
    if not metadata:
        return None
    for key in ("username", "login", "handle"):
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def _sanitize_task_user_reference(
    task: str,
    gaia_name: str | None,
    provider_hint: str,
    service_username: str | None,
) -> str:
    if not gaia_name:
        return task

    lowered = task.lower()
    if provider_hint.lower() not in lowered:
        return task

    replacement = service_username or "authenticated user"
    patterns = [
        rf"(user\s*[:=]?\s*['\"]?)({re.escape(gaia_name)})(['\"]?)",
        rf"(username\s*[:=]?\s*['\"]?)({re.escape(gaia_name)})(['\"]?)",
        rf"(account\s*[:=]?\s*['\"]?)({re.escape(gaia_name)})(['\"]?)",
    ]

    updated = task
    for pattern in patterns:
        updated = re.sub(pattern, rf"\1{replacement}\3", updated, flags=re.IGNORECASE)
    return updated


async def check_integration_connection(
    integration_id: str,
    user_id: str,
) -> str | None:
    """Return the connect prompt when the integration isn't connected, else None.

    Lives here rather than in integration_checker: it needs check_integration_status
    from oauth_service, which reaches integration_checker through composio, so
    moving it down closes a real import cycle.
    """
    subagent = get_subagent_by_id(integration_id)
    if not subagent:
        return None

    if await check_integration_status(integration_id, user_id):
        return None

    return await request_integration_connection(subagent.id, subagent.name, user_id)


async def _get_subagent_by_id(subagent_id: str) -> Subagent | CustomMcpSubagent | None:
    """Get subagent by ID or short_name.

    Checks both platform/builtin subagents (via registry) and custom MCPs
    from MongoDB, with Redis caching for the latter.
    """
    search_id = subagent_id.lower().strip()

    # Check platform/builtin subagents first (no caching needed - in-memory)
    subagent = get_subagent_by_id(search_id)
    if subagent:
        return subagent

    # Check Redis cache for custom integrations
    cache_key = f"{SUBAGENT_CACHE_PREFIX}:{search_id}"
    cached = await get_cache(cache_key)
    if cached is not None:
        # Return cached result (could be empty dict for negative cache)
        return CustomMcpSubagent.model_validate(cached) if cached else None

    # Search by integration_id (case-insensitive) or exact name
    custom = await integration_repository.find_by_id_prefix_or_name(search_id)

    if custom:
        result = CustomMcpSubagent(
            id=custom.integration_id,
            name=custom.name,
            source=custom.source,
            managed_by=custom.managed_by,
            mcp_config=custom.mcp_config.model_dump() if custom.mcp_config else None,
            icon_url=custom.icon_url,
        )
        await set_cache(cache_key, result.model_dump(), ttl=SUBAGENT_CACHE_TTL)
        return result

    # Fallback: Try IntegrationResolver which checks multiple sources
    # This handles cases where integration is in user_integrations but not integrations

    resolved = await IntegrationResolver.resolve(search_id)
    if resolved and resolved.custom_doc:
        doc = _CustomIntegrationDoc.model_validate(resolved.custom_doc)
        result = CustomMcpSubagent(
            id=doc.integration_id,
            name=doc.name,
            source=resolved.source,
            managed_by="mcp",
            mcp_config=doc.mcp_config,
            icon_url=doc.icon_url,
        )
        await set_cache(cache_key, result.model_dump(), ttl=SUBAGENT_CACHE_TTL)
        return result

    # Cache negative result to avoid repeated DB queries
    await set_cache(cache_key, {}, ttl=SUBAGENT_CACHE_TTL)
    return None


async def index_custom_mcp_as_subagent(
    store: BaseStore,
    request: CustomMcpIndexRequest,
) -> None:
    """Index a custom MCP as a subagent for handoff discovery.

    Called when user connects a custom MCP to make it immediately available
    for semantic search and handoff.
    """
    integration_id = request.integration_id
    name = request.name
    description = request.description
    server_url = request.server_url
    tools = request.tools

    parts = [f"{name}."]
    if description:
        parts.append(f"{description}.")
    if tools:
        summaries = []
        for t in tools:
            first_line = (t.description or "").strip().splitlines()
            summary = first_line[0][:120] if first_line else ""
            summaries.append(f"{t.name}: {summary}" if summary else t.name)
        parts.append(f"Available tools: {'; '.join(summaries)}.")
    rich_description = " ".join(parts)

    tool_namespace = derive_integration_namespace(integration_id, server_url, is_custom=True)

    put_op = PutOp(
        namespace=SUBAGENTS_NAMESPACE,
        key=integration_id,
        value={
            "id": integration_id,
            "name": name,
            "description": rich_description,
            "source": "custom",
            "tool_namespace": tool_namespace,
        },
        index=["description"],
    )

    await store.abatch([put_op])
    log.info(
        f"{LogTag.AGENT} Indexed custom MCP as subagent",
        integration_name=name,
        integration_id=integration_id,
        tool_count=len(tools or []),
    )


async def _resolve_custom_mcp_subagent(
    resolved: CustomMcpSubagent,
    user_id: str | None,
) -> tuple[CompiledAgentGraph | None, str | None, str | None, bool]:
    """Resolve a custom MCP into the _resolve_subagent tuple."""
    integration_id = resolved.id
    integration_name = resolved.name if resolved.name is not None else integration_id

    if not integration_id:
        return None, None, "Error: Custom integration has no ID", False

    if not user_id:
        return (
            None,
            None,
            f"Error: {integration_name} requires authentication. Please sign in first.",
            False,
        )

    # Create subagent for custom MCP
    try:
        subagent_graph = await create_subagent_for_user(integration_id, user_id)
    except SubagentUnavailableError as e:
        return (
            None,
            None,
            f"Error: {integration_name} is unavailable: {e.reason}",
            False,
        )

    agent_name = f"custom_mcp_{integration_id}"
    return subagent_graph, agent_name, integration_id, True


async def _resolve_auth_mcp_graph(
    subagent: Subagent,
    agent_name: str,
    integration_id: str,
    user_id: str | None,
) -> tuple[CompiledAgentGraph | None, str | None]:
    """Graph for an auth-required MCP integration, or (None, error_message)."""
    if not user_id:
        return None, f"Error: {agent_name} requires authentication. Please sign in first."

    # Check if user has connected this MCP integration
    token_store = MCPTokenStore(user_id=user_id)
    is_connected = await token_store.is_connected(integration_id)
    if not is_connected:
        return None, await request_integration_connection(integration_id, subagent.name, user_id)

    # Create subagent on-the-fly with user's tokens
    try:
        return await create_subagent_for_user(integration_id, user_id), None
    except SubagentUnavailableError as e:
        return None, f"Error: {agent_name} is unavailable: {e.reason}"


async def _resolve_subagent(
    subagent_id: str,
    user_id: str | None,
) -> tuple[CompiledAgentGraph | None, str | None, str | None, bool]:
    """Resolve subagent from ID and get the graph.

    Accepts 'subagent:gmail', 'subagent:fb9dfd7e05f8 (Semantic Scholar)', or
    a bare id. Returns (None, None, error_message, False) on failure.
    """
    clean_id, _ = parse_subagent_id(subagent_id)

    resolved = await _get_subagent_by_id(clean_id)

    if not resolved:
        available = [s.id for s in all_subagents()][:5]
        known = get_subagent_by_id(clean_id)
        if known is not None:
            error = (
                f"'{subagent_id}' is not a handoff target. Use "
                f"activate_integration(integration_id='{known.id}') to load it "
                "in-context, then act on it yourself."
            )
        else:
            error = (
                f"Unknown integration '{subagent_id}'. "
                f"Examples: {', '.join(available)}{'...' if len(available) == 5 else ''}"
            )
        return None, None, error, False

    # Handle custom MCPs (resolved from MongoDB)
    if isinstance(resolved, CustomMcpSubagent):
        return await _resolve_custom_mcp_subagent(resolved, user_id)

    # Platform/builtin subagent (Subagent object)
    subagent = resolved
    agent_name = subagent.config.agent_name
    integration_id = subagent.id

    # Handle auth-required MCP integrations specially
    if subagent.managed_by == "mcp" and subagent.mcp_config and subagent.mcp_config.requires_auth:
        subagent_graph, graph_error = await _resolve_auth_mcp_graph(
            subagent, agent_name, integration_id, user_id
        )
    else:
        # Only per-user MCP integrations run as subagents (their tools are issued
        # per user and never load in-context). Provider and built-in integrations
        # activate in-context instead — handing one off resurrects the old model.
        log.set(handoff={"integration": integration_id, "routed_to_activation": True})
        return (
            None,
            None,
            f"'{integration_id}' is not a handoff target. Load it in-context with "
            f"activate_integration(integration_id='{integration_id}'), then act on "
            "it yourself with its tools.",
            False,
        )
    if graph_error is not None:
        return None, None, graph_error, False

    return subagent_graph, agent_name, integration_id, False


async def _build_integration_metadata(
    is_custom: bool, integration_id: str
) -> IntegrationMetadata | None:
    """Build display metadata for a resolved subagent integration."""
    if is_custom:
        integration = await _get_subagent_by_id(integration_id)
        if isinstance(integration, CustomMcpSubagent):
            return IntegrationMetadata(
                icon_url=integration.icon_url,
                integration_id=integration_id,
                name=integration.name or integration_id,
            )
        return None
    platform_integ = get_subagent_by_id(integration_id)
    if platform_integ:
        return IntegrationMetadata(
            icon_url=getattr(platform_integ, "icon_url", None),
            integration_id=integration_id,
            name=platform_integ.name,
        )
    return None


def _resolve_display_metadata(
    metadata: IntegrationMetadata | None,
    fallback_name: str,
    fallback_category: str,
) -> tuple[str, str | None, str]:
    """Extract display name, icon URL, and tool category from integration metadata."""
    if not metadata:
        return fallback_name, None, fallback_category
    return (
        str(metadata.name or fallback_name),
        metadata.icon_url,
        str(metadata.integration_id or fallback_category),
    )


async def prepare_subagent_execution(
    subagent_id: str,
    task: str,
    configurable: AgentConfigurable,
    stream_id: str | None = None,
) -> tuple[SubagentExecutionContext | None, IntegrationMetadata | None, str | None]:
    """Resolve a subagent and build everything needed to execute it.

    The single preparation path for running one subagent — used by the
    executor's handoff tool and the dev direct-invocation endpoint.
    Returns (ctx, integration_metadata, None) on success or
    (None, None, error_message) when the subagent can't be resolved.
    """
    user_id = configurable.get("user_id")

    (
        subagent_graph,
        resolved_agent_name,
        int_id_or_error,
        is_custom,
    ) = await _resolve_subagent(subagent_id, user_id)

    if subagent_graph is None or resolved_agent_name is None or int_id_or_error is None:
        return None, None, int_id_or_error or "Unknown error resolving subagent"

    agent_name: str = resolved_agent_name
    integration_id: str = int_id_or_error
    log.set(
        subagent={
            "name": agent_name,
            "provider": integration_id,
            "is_custom": is_custom,
            "task_length": len(task),
        }
    )

    thread_id = configurable.get("thread_id", "")
    subagent_thread_id = f"{integration_id}_{thread_id}"

    user: AgentUserContext = {
        "user_id": user_id,
        "email": configurable.get("email"),
        "name": configurable.get("user_name"),
    }

    subagent_config = await build_agent_config(
        identity=AgentIdentity(
            conversation_id=thread_id,
            user=user,
            agent_name=agent_name,
        ),
        thread=AgentThread(
            thread_id=subagent_thread_id,
            base_configurable=configurable,
            subagent_id=agent_name,
        ),
    )
    new_configurable = agent_configurable(subagent_config)

    system_message = await create_subagent_system_message(integration_id=integration_id)

    # Declared startup tools load as schema docs appended to the static prompt, so
    # the slot stays singleton and cache-stable. Degrades to no docs (with a
    # warning) rather than failing: the agent can still retrieve_tools + execute.
    try:
        preload_block = await preloaded_startup_docs(user_id, integration_id)
    except Exception as e:
        log.warning(
            f"{LogTag.AGENT} Startup tool docs unavailable; continuing without them",
            integration_id=integration_id,
            error_type=type(e).__name__,
        )
        preload_block = ""
    if preload_block:
        system_message = SystemMessage(content=f"{system_message.content}\n\n{preload_block}")

    # Avoid passing Gaia display name as a service username
    provider_meta = None
    provider_name = None
    platform_subagent = get_subagent_by_id(integration_id)
    if platform_subagent and platform_subagent.provider and user_id:
        provider_name = platform_subagent.provider
        provider_meta = await get_provider_metadata(user_id, platform_subagent.provider)
    service_username = _extract_service_username(provider_meta)
    integration_usernames: dict[str, str] = {}
    if provider_name and service_username:
        integration_usernames[provider_name] = service_username
    sanitized_task = _sanitize_task_user_reference(
        task=task,
        gaia_name=user.get("name"),
        provider_hint=(provider_name or integration_id),
        service_username=service_username,
    )

    messages = await build_initial_messages(
        system_message=system_message,
        agent_name=agent_name,
        task=sanitized_task,
        seed=ThreadSeed(
            tier=AgentTier.PROVIDER_SUBAGENT,
            configurable=new_configurable,
            user_id=user_id,
            subagent_id=agent_name,
            # Without this the custom-instructions/provider-metadata lookup falls
            # back to agent_name ("gmail_agent"), which never matches the stored
            # integration id ("gmail"), so the user's instructions are dropped.
            integration_id=integration_id,
        ),
    )

    ctx = SubagentExecutionContext(
        subagent_graph=subagent_graph,
        agent_name=agent_name,
        config=subagent_config,
        configurable=new_configurable,
        integration_id=integration_id,
        initial_state={
            "messages": messages,
            "todos": [],
            "intent": sanitized_task,
            "integration_usernames": integration_usernames,
        },
        user_id=user_id,
        stream_id=stream_id,
    )

    integration_metadata = await _build_integration_metadata(is_custom, integration_id)
    return ctx, integration_metadata, None


async def _run_blocking_handoff(
    ctx: SubagentExecutionContext,
    dispatch: _HandoffDispatch,
) -> str:
    """Run a handoff subagent synchronously, emitting lifecycle SSE events.

    dispatch.record_calls (workflow runs only) appends the subagent's successful
    tool calls to the result so the executor can transcribe them into a playbook.
    """
    metadata = dispatch.metadata
    agent_name = dispatch.agent_name
    integration_id = dispatch.integration_id
    probe_parked = dispatch.probe_parked
    record_calls = dispatch.record_calls

    writer = get_stream_writer()
    # Stable across replays so an approval pause reuses the same UI row instead of
    # orphaning the paused one and opening a duplicate on resume.
    sa_id = subagent_row_id(dispatch.tool_call_id)
    display, icon_url, tool_category = _resolve_display_metadata(
        metadata, agent_name, integration_id
    )

    # Propagate this subagent's UUID into config so nested spawned subagents
    # can reference it as parent_subagent_id.
    ctx.configurable["subagent_id"] = sa_id
    ctx.config.setdefault("configurable", {})["subagent_id"] = sa_id

    writer(
        {
            "subagent_start": format_subagent_start_event(
                subagent_name=display,
                agent_type="handoff",
                subagent_id=sa_id,
                subagent=integration_id,
                details=SubagentStartDetails(icon_url=icon_url, tool_category=tool_category),
            )
        }
    )
    start_time = time.monotonic()

    # Blocking runs register like background ones, or list_running_subagents shows
    # only background work. Deregistered in `finally`, so pauses (which raise) read
    # as parked, not running — the same line the background path draws.
    blocking_conversation_id = str(ctx.configurable.get("conversation_id") or "")
    blocking_thread_id = str(agent_configurable(ctx.config).get("thread_id", ""))
    blocking_record = (
        RunningSubagent(
            subagent_id=sa_id,
            subagent_thread_id=blocking_thread_id,
            integration_id=integration_id,
            agent_name=agent_name,
            task_summary=str(ctx.initial_state.get("intent", ""))[:200],
            started_at=datetime.now(UTC).isoformat(),
        )
        if blocking_conversation_id and blocking_thread_id
        else None
    )
    if blocking_record is not None:
        await RunningSubagents(blocking_conversation_id).register(blocking_record)
    try:
        # On resume, THIS node re-runs over a subagent thread that already holds
        # work, so an existing checkpoint means "recover, don't rerun". Such a
        # checkpoint exists only on a replay, so fresh runs skip the probe entirely.
        recovered = await recover_from_checkpoint(ctx) if probe_parked else None
        if recovered is not None:
            outcome = recovered
        else:
            outcome = await execute_subagent_stream(
                ctx=ctx,
                stream_writer=writer,
                integration_metadata=metadata,
                subagent_id=sa_id,
            )

        # The subagent runs imperatively, so its GraphInterrupt never reaches the
        # executor — bubble each pause up explicitly. A LOOP because one task can gate
        # several calls in sequence; earlier decided gates are matched out by approval_id.
        run_messages: list[AnyMessage] = list(outcome.run_messages)
        while outcome.paused:
            decision = resume_for_gate(outcome.interrupt)
            outcome = await execute_subagent_stream(
                ctx=ctx,
                stream_writer=writer,
                integration_metadata=metadata,
                subagent_id=sa_id,
                resume=Command(resume=decision),
            )
            run_messages.extend(outcome.run_messages)
    finally:
        if blocking_record is not None:
            await RunningSubagents(blocking_conversation_id).deregister(sa_id)

    writer(
        {
            "subagent_end": format_subagent_end_event(
                subagent_id=sa_id,
                duration_ms=int((time.monotonic() - start_time) * 1000),
            )
        }
    )
    if record_calls:
        return append_call_record(outcome.text, run_messages)
    return outcome.text


async def _has_parked_subagent(ctx: SubagentExecutionContext) -> bool:
    """Whether an uncollected HIL-parked subagent owns this ctx's checkpoint thread.

    Durable check (Mongo), so it holds across executor pause/resume and process
    restarts — the session slot only tracks live tasks in this invocation.
    """
    configurable: AgentConfigurable = ctx.configurable
    conversation_id = str(configurable.get("conversation_id") or "")
    thread_id = str(configurable.get("thread_id") or "")
    if not conversation_id or not thread_id:
        return False
    records = await list_parked_subagents_for_conversation(conversation_id)
    return any(record.subagent_thread_id == thread_id for record in records)


async def _handoff_rejection(
    ctx: SubagentExecutionContext,
    task: str,
    background: bool,
    stream_id: str | None,
) -> str | None:
    """Return the pre-dispatch refusal for one handoff, or None when it may run."""
    agent_name: str = ctx.agent_name
    integration_id: str = ctx.integration_id

    # A task naming one provider while routed to another tells the user their data
    # went somewhere it never did. Refuse before dispatch — the executor can
    # re-route or drop the name, but once said it cannot un-say it.
    foreign = foreign_provider_named_in(task, integration_id)
    if foreign is not None:
        return (
            f"HANDOFF REJECTED: this task is routed to the {agent_name} subagent "
            f"({integration_id}) but its text names {foreign.name}. {foreign.name} is a "
            f"separate integration with its own tools, and nothing you hand to "
            f"{integration_id} touches it: leaving the name in makes the result claim "
            f"{foreign.name} did work it never did. Either activate {foreign.id} with "
            f"activate_integration and do that part yourself, or send this handoff "
            f"again with every mention of {foreign.name} removed from the task."
        )

    # A parked subagent owns this integration's checkpoint thread; feeding it any
    # new handoff makes LangGraph discard the pending interrupt and orphan the
    # user's approval card. Refuse until the join collects it.
    if await _has_parked_subagent(ctx):
        return (
            f"The {agent_name} subagent is paused waiting for the user's approval. "
            "Its outcome will be delivered on resolution; send it nothing meanwhile."
        )

    # Same collision for a BLOCKING handoff while a background task holds
    # this thread (the background branch guards via the session slot claim).
    if not background and has_bg_integration(str(stream_id or ""), integration_id):
        return (
            f"A background {agent_name} subagent is already running on this "
            "integration. Steer it with message_subagent or wait for its result to arrive."
        )

    return None


async def _dispatch_background_handoff(
    ctx: SubagentExecutionContext,
    dispatch: _HandoffDispatch,
    sid: str,
) -> str:
    """Spawn the subagent as a detached task, returning the collect-later notice."""
    agent_name = dispatch.agent_name
    integration_id = dispatch.integration_id

    # execution_mode is inherited, NOT forced to "background": a live-conversation
    # detached subagent keeps a pause path, a headless run stays fail-closed. One
    # per integration — a duplicate would corrupt the shared checkpoint thread id.
    if not claim_bg_integration(sid, integration_id):
        return (
            f"A background {agent_name} subagent is already running. Call "
            "Its result arrives on its own; send it new tasks after it lands."
        )
    # Idempotent across node replays: when this handoff shares its node run with a
    # pause, the node re-runs on resume and must not spawn twice. tool_call_id is
    # stable (checkpointed AI message); the claim is durable in Redis.
    conversation_id = str(ctx.configurable.get("conversation_id") or "")
    if (
        dispatch.tool_call_id
        and conversation_id
        and not await try_claim_bg_dispatch(conversation_id, dispatch.tool_call_id)
    ):
        release_bg_integration(sid, integration_id)
        return (
            f"Subagent {agent_name} started in background. "
            "Steer it with message_subagent/cancel_subagent; its result arrives automatically."
        )
    bg_sa_id = str(uuid4())
    bg_display, bg_icon, bg_cat = _resolve_display_metadata(
        dispatch.metadata, agent_name, integration_id
    )
    spawn_background_task(
        run_subagent_background(
            ctx=ctx,
            stream_id=sid,
            handoff=BackgroundHandoff(
                integration_metadata=dispatch.metadata,
                subagent_id=bg_sa_id,
                display_name=bg_display,
                tool_category=bg_cat,
                icon_url=bg_icon,
                integration_id=integration_id,
                record_calls=dispatch.record_calls,
            ),
        )
    )
    log.info(
        f"{LogTag.AGENT} Subagent dispatched to background",
        agent_name=agent_name,
        stream_id=sid,
    )
    return (
        f"Subagent {agent_name} started in background. "
        "Steer it with message_subagent/cancel_subagent; its result arrives automatically."
    )


@tool
async def handoff(
    subagent_id: Annotated[
        str,
        "The ID of a per-user MCP integration to delegate to (custom MCP "
        "connections only — e.g. 'my-notion-mcp'). Discovered via "
        "retrieve_tools as subagent: entries (only per-user MCP integrations "
        "appear there). Provider and built-in integrations (gmail, github, "
        "todos, ...) are NOT handoff targets: load those with "
        "activate_integration instead.",
    ],
    task: Annotated[
        str,
        "Detailed description of the task for the subagent, including all relevant context.",
    ],
    config: RunnableConfig,
    background: Annotated[
        bool,
        "If True, run the subagent in the background and return immediately. "
        "Use for parallel subagent dispatch. Steer mid-run with "
        "message_subagent/cancel_subagent; results arrive automatically. "
        "Runs that carry a stream_id always go to the background even when "
        "this is False, so the executor stays alive to steer them; without a "
        "stream_id results cannot be routed back, so that case stays blocking. "
        "Default False (blocking unless a stream_id is present).",
    ] = False,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> str:
    """Delegate a task to a per-user MCP integration's own subagent.

    This is the ONLY path for custom/auth-required MCP integrations, whose
    tools are issued per user and can never load in-context (activate_integration
    tells you to come here for exactly those). Every other integration is
    activated in-context with activate_integration and acted on directly —
    never handed off.

    The subagent will:
    1. Process the task using its specialized tools
    2. Return the result of the completed task

    For parallel execution, set background=True on multiple handoff calls, then
    results arrive automatically; steer meanwhile.

    Args:
        subagent_id: ID of the per-user MCP integration (NOT a provider id)
        task: Complete task description with all necessary context
        background: If True, run non-blocking and return immediately. Runs
            that carry a stream_id run non-blocking regardless.
    """
    try:
        configurable: AgentConfigurable = agent_configurable(config)
        user_id = configurable.get("user_id")

        # Fallback: try to get user_id from metadata if not in configurable
        if not user_id:
            user_id = _RunMetadata.model_validate(config.get("metadata") or {}).user_id
            if user_id:
                configurable["user_id"] = user_id

        stream_id = configurable.get("stream_id")  # Extract stream_id for cancellation

        # Only a resume replay can have left a parked subagent behind, so the
        # per-handoff checkpoint probe in _run_blocking_handoff is gated on this.
        probe_parked = bool(configurable.get(HIL_RESUME_CONFIG_KEY))

        # Workflow runs get the subagent's actual calls appended to the result
        # so write_playbook transcribes real tool names/args instead of guessing
        # them. Chat runs are untouched — no extra text, no extra tokens.
        record_calls = bool(configurable.get("workflow_id"))

        ctx, integration_metadata, error = await prepare_subagent_execution(
            subagent_id=subagent_id,
            task=task,
            configurable=configurable,
            stream_id=stream_id,
        )
        if ctx is None:
            return error or "Unknown error resolving subagent"

        agent_name: str = ctx.agent_name
        integration_id: str = ctx.integration_id

        # handoff only reaches per-user MCP subagents; run those in the background
        # by default so the executor stays alive to steer or cancel them (results
        # arrive via the inbox). Without a stream_id results can't route, so blocking.
        if stream_id and not background:
            background = True

        rejection = await _handoff_rejection(ctx, task, background, stream_id)
        if rejection is not None:
            return rejection

        dispatch = _HandoffDispatch(
            metadata=integration_metadata,
            agent_name=agent_name,
            integration_id=integration_id,
            tool_call_id=tool_call_id,
            probe_parked=probe_parked,
            record_calls=record_calls,
        )

        # Background mode: spawn as an asyncio task and return immediately; results
        # arrive via the executor inbox. Requires stream_id in the configurable so
        # the result can route back to this conversation.
        if background:
            if not stream_id:
                log.warning(
                    f"{LogTag.AGENT} handoff background=True but stream_id is missing — "
                    "falling back to blocking execution"
                )
                blocking_result = await _run_blocking_handoff(ctx, dispatch)
                return (
                    "[WARNING: background handoff fell back to blocking: "
                    "stream_id not propagated into executor configurable] "
                    f"{blocking_result}"
                )
            sid: str = str(stream_id)
            return await _dispatch_background_handoff(ctx, dispatch, sid)

        # Blocking (default): execute synchronously and return result.
        return await _run_blocking_handoff(ctx, dispatch)

    except GraphBubbleUp:
        # The HIL gate's interrupt bubbling up from _run_blocking_handoff. Control
        # flow, not a failure — swallowing it here would convert the approval pause
        # into a tool error and run the executor on without ever pausing.
        raise
    except Exception as e:
        log.error(
            f"{LogTag.AGENT} handoff_failed",
            subagent_id=subagent_id,
            user_id=user_id,
            error_type=type(e).__name__,
            error=str(e)[:500],
            exc_info=True,
        )
        return f"Error executing task: {e!s}"
