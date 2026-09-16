"""Startup tool preload: integration tools load as schema docs, never as bindings.

Regression contract for the execute-proxy cutover: a subagent's declared
startup tools (``auto_bind_tools`` + ``extra_initial_tools``) split by kind —
internal tools bind into ``initial_tool_ids`` as before, while integration
tools (Composio ``require_integration`` categories + per-user MCP) are
excluded from binding and instead render as schema docs injected into the
run's context, executed via ``execute``.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _category(*, require_integration: bool, tool_names: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        require_integration=require_integration,
        tools=[SimpleNamespace(name=n, tool=MagicMock(name=n)) for n in tool_names],
    )


def _registry(
    *,
    integration_names: set[str],
    internal_names: set[str],
) -> MagicMock:
    """A registry where ALLCAPS names are integration tools, the rest internal."""
    registry = MagicMock()
    registry.get_category_of_tool.side_effect = (
        lambda n: "int_cat" if n in integration_names else "general"
    )

    def _get_category(name: str) -> SimpleNamespace | None:
        if name == "int_cat":
            return _category(require_integration=True, tool_names=sorted(integration_names))
        if name == "general":
            return _category(require_integration=False, tool_names=sorted(internal_names))
        return None

    registry.get_category.side_effect = _get_category
    registry.get_category_by_space.side_effect = lambda space: (
        _category(require_integration=True, tool_names=sorted(integration_names))
        if space == "gmail"
        else None
    )
    return registry


@pytest.mark.unit
class TestSplitStartupTools:
    async def test_integration_tools_preload_while_internal_tools_bind(self) -> None:
        from app.agents.tools.core import retrieval

        with patch.object(
            retrieval, "get_tool_registry", new=AsyncMock(return_value=_registry(
                integration_names={"GMAIL_FETCH_MESSAGES"},
                internal_names={"query_json"},
            )),
        ):
            bind, preload = await retrieval.split_startup_tools(
                None, ["GMAIL_FETCH_MESSAGES", "query_json"]
            )
        assert bind == ["query_json"]
        assert preload == ["GMAIL_FETCH_MESSAGES"]

    async def test_mcp_tool_names_preload(self) -> None:
        from app.agents.tools.core import retrieval

        registry = _registry(integration_names=set(), internal_names=set())
        with patch.object(
            retrieval, "get_tool_registry", new=AsyncMock(return_value=registry)
        ):
            bind, preload = await retrieval.split_startup_tools(
                "u1", ["MY_MCP_TOOL"], mcp_tool_names={"MY_MCP_TOOL"}
            )
        assert bind == []
        assert preload == ["MY_MCP_TOOL"]

    async def test_empty_declaration_splits_empty(self) -> None:
        from app.agents.tools.core import retrieval

        with patch.object(
            retrieval, "get_tool_registry", new=AsyncMock(return_value=_registry(
                integration_names=set(), internal_names=set()
            )),
        ):
            assert await retrieval.split_startup_tools(None, []) == ([], [])
            assert await retrieval.split_startup_tools(None, None) == ([], [])

    async def test_order_is_stable_and_duplicates_collapse(self) -> None:
        from app.agents.tools.core import retrieval

        with patch.object(
            retrieval, "get_tool_registry", new=AsyncMock(return_value=_registry(
                integration_names={"GMAIL_A"},
                internal_names={"query_json"},
            )),
        ):
            bind, preload = await retrieval.split_startup_tools(
                None, ["GMAIL_A", "query_json", "GMAIL_A"]
            )
        assert bind == ["query_json"]
        assert preload == ["GMAIL_A"]


@pytest.mark.unit
class TestRenderPreloadBlock:
    async def test_renders_docs_with_execute_guidance(self) -> None:
        from langchain_core.tools import tool as langchain_tool

        from app.agents.tools.core import retrieval
        from app.agents.tools.execute.resolver import ResolvedTool

        @langchain_tool
        def GMAIL_FETCH_MESSAGES(thread_id: str) -> str:
            """Fetch one Gmail thread by id."""
            return thread_id

        resolved = ResolvedTool(
            name="GMAIL_FETCH_MESSAGES",
            tool=GMAIL_FETCH_MESSAGES,
            is_integration=True,
            in_registry=True,
        )
        with patch.object(
            retrieval, "_resolve_for_retrieval", new=AsyncMock(return_value=resolved)
        ):
            block = await retrieval.render_preload_block("u1", ["GMAIL_FETCH_MESSAGES"])
        assert "## GMAIL_FETCH_MESSAGES" in block
        assert "execute(" in block
        assert "NOT bound" in block

    async def test_empty_preload_renders_empty(self) -> None:
        from app.agents.tools.core import retrieval

        assert await retrieval.render_preload_block("u1", []) == ""

    async def test_unresolvable_tools_degrade_to_empty_with_warning(self) -> None:
        from app.agents.tools.core import retrieval

        with patch.object(
            retrieval, "_resolve_for_retrieval", new=AsyncMock(return_value=None)
        ):
            assert await retrieval.render_preload_block("u1", ["GMAIL_GHOST"]) == ""


@pytest.mark.unit
class TestFactoryDoesNotBindIntegrationTools:
    """The bug pin: gmail-style auto_bind integration tools must not reach
    ``initial_tool_ids`` (provider ``bind_tools``). Internal extras still do."""

    async def test_integration_auto_bind_excluded_from_initial_ids(self) -> None:
        from app.agents.core.subagents.base_subagent import (
            SubAgentFactory,
            SubAgentToolConfig,
        )

        scoped = {
            name: MagicMock(name=name)
            for name in (
                "GMAIL_FETCH_MESSAGES",
                "query_json",
                "search_memory",
                "read",
                "bash",
                "execute",
                "get_tool_schema",
                "finish_task",
            )
        }
        for name, tool in scoped.items():
            tool.name = name

        captured: dict = {}

        def _fake_create_agent(**kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)
            builder = MagicMock()
            builder.compile.return_value = MagicMock()
            return builder

        with (
            patch(
                "app.agents.core.subagents.base_subagent.get_tools_store",
                new=AsyncMock(),
            ),
            patch(
                "app.agents.core.subagents.base_subagent.get_tool_registry",
                new=AsyncMock(return_value=_registry(
                    integration_names={"GMAIL_FETCH_MESSAGES"},
                    internal_names={"query_json"},
                )),
            ),
            patch(
                "app.agents.core.subagents.base_subagent.build_scoped_tool_dict",
                return_value=(scoped, ["GMAIL_FETCH_MESSAGES", "query_json"]),
            ),
            patch(
                "app.agents.core.subagents.base_subagent.create_subagent_middleware",
                return_value=[],
            ),
            patch(
                "app.agents.core.subagents.base_subagent.create_todo_tools",
                return_value=[],
            ),
            patch(
                "app.agents.core.subagents.base_subagent.create_todo_pre_model_hook",
                return_value=None,
            ),
            patch(
                "app.agents.core.subagents.base_subagent.worker_pre_model_hooks",
                return_value=[],
            ),
            patch(
                "app.agents.core.subagents.base_subagent.create_agent",
                side_effect=_fake_create_agent,
            ),
            patch(
                "app.agents.core.subagents.base_subagent.get_checkpointer_manager",
                new=AsyncMock(
                    return_value=MagicMock(
                        get_checkpointer=MagicMock(return_value=MagicMock())
                    )
                ),
            ),
        ):
            await SubAgentFactory.create_provider_subagent(
                provider="gmail",
                name="gmail_agent",
                llm=MagicMock(),
                config=SubAgentToolConfig(
                    tool_space="gmail",
                    auto_bind_tools=["GMAIL_FETCH_MESSAGES"],
                    extra_initial_tools=["query_json"],
                ),
            )

        initial_ids = list(captured["tools_config"].initial_tool_ids)
        assert "query_json" in initial_ids
        assert "GMAIL_FETCH_MESSAGES" not in initial_ids


@pytest.mark.unit
class TestPrepareInjectsPreloadDocs:
    """`prepare_subagent_execution` opens the run with the integration's
    startup schemas inside the static system message — no binding, no
    retrieve_tools round trip."""

    async def test_system_message_carries_preloaded_schemas(self) -> None:
        from app.agents.core.subagents import handoff_tools
        from app.agents.tools.core import retrieval
        from app.agents.tools.execute.resolver import ResolvedTool
        from langchain_core.tools import tool as langchain_tool

        captured: dict = {}

        async def _fake_build_initial_messages(**kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)
            return [kwargs["system_message"]]

        def _fake_resolved(name: str) -> ResolvedTool:
            @langchain_tool
            def _fake_tool(query: str) -> str:
                """Run a fake query."""
                return query

            _fake_tool.name = name
            return ResolvedTool(name=name, tool=_fake_tool, is_integration=True)

        split_registry = MagicMock()
        split_registry.get_category_of_tool.side_effect = (
            lambda n: "gmail" if n.startswith("GMAIL_") else "general"
        )

        def _get_category(name: str) -> MagicMock:
            category = MagicMock()
            category.require_integration = name == "gmail"
            return category

        split_registry.get_category.side_effect = _get_category

        with (
            patch.object(
                handoff_tools,
                "_resolve_subagent",
                new=AsyncMock(return_value=(MagicMock(), "gmail_agent", "gmail", False)),
            ),
            patch.object(
                handoff_tools,
                "build_agent_config",
                new=AsyncMock(
                    return_value={
                        "configurable": {
                            "thread_id": "t1",
                            "user_id": "u1",
                            "conversation_id": "c1",
                        }
                    }
                ),
            ),
            patch.object(
                handoff_tools, "get_provider_metadata", new=AsyncMock(return_value=None)
            ),
            patch.object(
                handoff_tools,
                "build_initial_messages",
                side_effect=_fake_build_initial_messages,
            ),
            patch.object(
                retrieval, "get_tool_registry", new=AsyncMock(return_value=split_registry)
            ),
            patch.object(
                retrieval, "_user_mcp_tool_names", new=AsyncMock(return_value=set())
            ),
            patch.object(
                retrieval,
                "_resolve_for_retrieval",
                new=AsyncMock(side_effect=lambda _u, n: _fake_resolved(n)),
            ),
        ):
            from app.agents.core.subagents.handoff_tools import prepare_subagent_execution

            ctx, _, error = await prepare_subagent_execution(
                "gmail", "List my inbox", {"user_id": "u1", "thread_id": "t1"}
            )

        assert error is None
        system_message = captured["system_message"]
        assert system_message.type == "system"
        # The real static prompt is intact and the preloaded schemas follow it
        # in the same singleton STATIC message.
        assert "## GMAIL_FETCH_MESSAGES" in system_message.content
        assert "execute(" in system_message.content
        assert "NOT bound" in system_message.content
        assert ctx is not None

    async def test_no_declared_tools_leaves_system_message_untouched(self) -> None:
        from app.agents.core.subagents import handoff_tools

        captured: dict = {}

        async def _fake_build_initial_messages(**kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)
            return [kwargs["system_message"]]

        with (
            patch.object(
                handoff_tools,
                "_resolve_subagent",
                new=AsyncMock(return_value=(MagicMock(), "docgen_agent", "docgen", False)),
            ),
            patch.object(
                handoff_tools,
                "build_agent_config",
                new=AsyncMock(
                    return_value={
                        "configurable": {
                            "thread_id": "t1",
                            "user_id": "u1",
                            "conversation_id": "c1",
                        }
                    }
                ),
            ),
            patch.object(
                handoff_tools, "get_provider_metadata", new=AsyncMock(return_value=None)
            ),
            patch.object(
                handoff_tools,
                "build_initial_messages",
                side_effect=_fake_build_initial_messages,
            ),
        ):
            from app.agents.core.subagents.handoff_tools import prepare_subagent_execution

            _, _, error = await prepare_subagent_execution(
                "docgen", "Make a PDF", {"user_id": "u1", "thread_id": "t1"}
            )

        assert error is None
        # docgen declares no startup tools and binds direct — no docs appended.
        assert "preloaded below" not in captured["system_message"].content
