"""Tests for app/agents/tools/core/retrieval.py — tool retrieval functions."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_subagent_mock(
    subagent_id: str, managed_by: str = "internal", name: str | None = None
) -> MagicMock:
    """Build a Subagent-shaped MagicMock for retrieval tests."""
    sa = MagicMock()
    sa.id = subagent_id
    sa.name = name or subagent_id.title()
    sa.managed_by = managed_by
    return sa


# ---------------------------------------------------------------------------
# _get_user_context
# ---------------------------------------------------------------------------


class TestGetUserContext:
    @pytest.mark.asyncio
    async def test_no_user_id_returns_defaults(self):
        from app.agents.tools.core.retrieval import _get_user_context

        ns, connected = await _get_user_context(None, "general", include_subagents=True)
        assert "general" in ns
        assert connected == {}

    @pytest.mark.asyncio
    async def test_with_user_id_gets_namespaces(self):
        from app.agents.tools.core.retrieval import _get_user_context

        with (
            patch(
                "app.agents.tools.core.retrieval.get_user_available_tool_namespaces",
                new_callable=AsyncMock,
                return_value=["general", "gmail", "subagents"],
            ),
            patch(
                "app.agents.tools.core.retrieval._resolve_connected_subagents",
                new_callable=AsyncMock,
                # gmail is not a platform subagent → custom integration in connected map
                return_value={"gmail": None},
            ),
        ):
            ns, connected = await _get_user_context(
                "user1", "general", include_subagents=True
            )
        assert "general" in ns
        assert "gmail" in ns
        # gmail is not a platform integration -> treated as custom -> connected
        assert "gmail" in connected

    @pytest.mark.asyncio
    async def test_user_context_exception_returns_defaults(self):
        from app.agents.tools.core.retrieval import _get_user_context

        with (
            patch(
                "app.agents.tools.core.retrieval.get_user_available_tool_namespaces",
                new_callable=AsyncMock,
                side_effect=RuntimeError("db fail"),
            ),
        ):
            ns, connected = await _get_user_context(
                "user1", "myspace", include_subagents=True
            )
        # Falls back to seeded defaults — non-platform tool spaces are NOT seeded
        assert "general" in ns
        assert "myspace" not in ns

    @pytest.mark.asyncio
    async def test_connected_integrations_with_subagent_config(self):
        from app.agents.tools.core.retrieval import _get_user_context

        slack_subagent = _make_subagent_mock("slack", "composio")

        with (
            patch(
                "app.agents.tools.core.retrieval.get_user_available_tool_namespaces",
                new_callable=AsyncMock,
                return_value=["general", "slack", "subagents"],
            ),
            patch(
                "app.agents.tools.core.retrieval._resolve_connected_subagents",
                new_callable=AsyncMock,
                return_value={"slack": slack_subagent.name},
            ),
        ):
            ns, connected = await _get_user_context(
                "user1", "general", include_subagents=True
            )
        assert "slack" in connected


# ---------------------------------------------------------------------------
# _build_search_tasks
# ---------------------------------------------------------------------------


class TestBuildSearchTasks:
    def test_tool_space_in_namespaces(self):
        from app.agents.tools.core.retrieval import _build_search_tasks

        store = MagicMock()
        store.asearch = AsyncMock(return_value=[])
        tasks = _build_search_tasks(
            store, "email", "gmail", {"gmail", "general"}, include_subagents=False, limit=10
        )
        # gmail search + general search (limited)
        assert len(tasks) == 2
        for t in tasks:
            if asyncio.iscoroutine(t):
                t.close()

    def test_general_space_no_duplicate(self):
        from app.agents.tools.core.retrieval import _build_search_tasks

        store = MagicMock()
        store.asearch = AsyncMock(return_value=[])
        tasks = _build_search_tasks(
            store, "email", "general", {"general"}, include_subagents=False, limit=10
        )
        # Only one search for general (tool_space == general, skip second general search)
        assert len(tasks) == 1
        for t in tasks:
            if asyncio.iscoroutine(t):
                t.close()

    def test_no_subagent_searches_when_disabled(self):
        from app.agents.tools.core.retrieval import _build_search_tasks

        store = MagicMock()
        store.asearch = AsyncMock(return_value=[])
        tasks = _build_search_tasks(
            store, "email", "general", {"general"}, include_subagents=False, limit=10
        )
        # Only the general search — scoped agents never look for subagents.
        assert len(tasks) == 1
        for t in tasks:
            if asyncio.iscoroutine(t):
                t.close()

    def test_subagent_and_public_searches_when_enabled(self):
        from app.agents.tools.core.retrieval import _build_search_tasks

        store = MagicMock()
        store.asearch = AsyncMock(return_value=[])
        tasks = _build_search_tasks(
            store, "email", "general", {"general"}, include_subagents=True, limit=10
        )
        # general + subagents namespace (MCP pointers) + public integrations
        assert len(tasks) == 3
        for t in tasks:
            if asyncio.iscoroutine(t):
                t.close()

    def test_tool_space_not_in_namespaces(self):
        from app.agents.tools.core.retrieval import _build_search_tasks

        store = MagicMock()
        store.asearch = AsyncMock(return_value=[])
        tasks = _build_search_tasks(
            store, "email", "slack", {"general"}, include_subagents=False, limit=10
        )
        # Only general search (slack not in namespaces)
        assert len(tasks) == 1
        for t in tasks:
            if asyncio.iscoroutine(t):
                t.close()


# ---------------------------------------------------------------------------
# _process_chroma_search_result
# ---------------------------------------------------------------------------


class TestProcessChromaSearchResult:
    def _make_item(self, key: str, score: float = 0.8, namespace=None, value=None):
        item = MagicMock()
        item.key = key
        item.score = score
        item.namespace = namespace
        if value is not None:
            item.value = value
        return item

    def test_regular_tool_in_available(self):
        from app.agents.tools.core.retrieval import _process_chroma_search_result

        item = self._make_item("GMAIL_SEND", namespace=("gmail",))
        registry = MagicMock()
        registry.get_category_of_tool.return_value = None

        result = _process_chroma_search_result(
            [item], {"GMAIL_SEND"}, registry, include_subagents=True
        )
        assert len(result) == 1
        assert result[0]["id"] == "GMAIL_SEND"

    def test_regular_tool_not_in_available(self):
        from app.agents.tools.core.retrieval import _process_chroma_search_result

        item = self._make_item("UNKNOWN_TOOL", namespace=("gmail",))
        registry = MagicMock()
        registry.get_category_of_tool.return_value = None

        result = _process_chroma_search_result(
            [item], {"GMAIL_SEND"}, registry, include_subagents=True
        )
        assert len(result) == 0

    def test_general_namespace_filters_non_webpage_tools_for_subagent(self):
        from app.agents.tools.core.retrieval import _process_chroma_search_result

        # tool_space != "general" -> general namespace should filter non-webpage tools
        item = self._make_item("create_todo", namespace=("general",))
        registry = MagicMock()
        registry.get_category_of_tool.return_value = None

        result = _process_chroma_search_result(
            [item],
            {"create_todo"},
            registry,
            include_subagents=False,
            tool_space="gmail",
        )
        assert len(result) == 0

    def test_general_namespace_allows_webpage_tools(self):
        from app.agents.tools.core.retrieval import (
            WEBPAGE_TOOLS,
            _process_chroma_search_result,
        )

        webpage_tool = WEBPAGE_TOOLS[0]
        item = self._make_item(webpage_tool, namespace=("general",))
        registry = MagicMock()
        registry.get_category_of_tool.return_value = None

        result = _process_chroma_search_result(
            [item],
            {webpage_tool},
            registry,
            include_subagents=False,
            tool_space="gmail",
        )
        assert len(result) == 1

    def test_delegated_tools_filtered_when_subagents_included(self):
        from app.agents.tools.core.retrieval import _process_chroma_search_result

        item = self._make_item("GMAIL_SEND", namespace=("gmail",))
        registry = MagicMock()
        registry.get_category_of_tool.return_value = "email_category"
        category = MagicMock()
        category.is_delegated = True
        registry.get_category.return_value = category

        result = _process_chroma_search_result(
            [item], {"GMAIL_SEND"}, registry, include_subagents=True
        )
        assert len(result) == 0


# ---------------------------------------------------------------------------
# _process_search_results
# ---------------------------------------------------------------------------


class TestProcessSearchResults:
    @pytest.mark.asyncio
    async def test_handles_exceptions_in_results(self):
        from app.agents.tools.core.retrieval import _process_search_results

        registry = MagicMock()
        results = [RuntimeError("search fail"), []]
        processed = await _process_search_results(results, set(), registry, include_subagents=False)
        assert processed == []

    @pytest.mark.asyncio
    async def test_handles_empty_results(self):
        from app.agents.tools.core.retrieval import _process_search_results

        registry = MagicMock()
        processed = await _process_search_results(
            [[], None], set(), registry, include_subagents=False
        )
        assert processed == []

    @pytest.mark.asyncio
    async def test_dict_results_render_as_integration_entries(self):
        from app.agents.tools.core.retrieval import _process_search_results

        registry = MagicMock()
        public_results = [{"integration_id": "abc", "name": "App", "relevance_score": 0.9}]
        processed = await _process_search_results(
            [public_results], set(), registry, include_subagents=True
        )
        assert [r["id"] for r in processed] == ["integration:abc (App)"]


# ---------------------------------------------------------------------------
# _deduplicate_and_sort
# ---------------------------------------------------------------------------


class TestDeduplicateAndSort:
    def test_deduplicates(self):
        from app.agents.tools.core.retrieval import _deduplicate_and_sort

        results = [
            {"id": "a", "score": 0.9},
            {"id": "a", "score": 0.8},
            {"id": "b", "score": 0.7},
        ]
        out = _deduplicate_and_sort(results, 10)
        assert out == ["a", "b"]

    def test_respects_limit(self):
        from app.agents.tools.core.retrieval import _deduplicate_and_sort

        results = [
            {"id": "a", "score": 0.9},
            {"id": "b", "score": 0.8},
            {"id": "c", "score": 0.7},
        ]
        out = _deduplicate_and_sort(results, 2)
        assert len(out) == 2

    def test_sorts_by_score_descending(self):
        from app.agents.tools.core.retrieval import _deduplicate_and_sort

        results = [
            {"id": "c", "score": 0.3},
            {"id": "a", "score": 0.9},
            {"id": "b", "score": 0.6},
        ]
        out = _deduplicate_and_sort(results, 10)
        assert out == ["a", "b", "c"]

    def test_handles_none_score(self):
        from app.agents.tools.core.retrieval import _deduplicate_and_sort

        results = [
            {"id": "a", "score": None},
            {"id": "b", "score": 0.5},
        ]
        out = _deduplicate_and_sort(results, 10)
        assert out == ["b", "a"]


# get_retrieve_tools_function / retrieve_tools
# ---------------------------------------------------------------------------


class TestGetRetrieveToolsFunction:
    def test_returns_callable(self):
        from app.agents.tools.core.retrieval import get_retrieve_tools_function

        fn = get_retrieve_tools_function()
        assert callable(fn)

    def test_docstring_has_no_subagent_section(self):
        from app.agents.tools.core.retrieval import get_retrieve_tools_function

        for fn in (
            get_retrieve_tools_function(),
            get_retrieve_tools_function(include_subagents=False),
        ):
            assert "SUBAGENT TOOLS" not in (fn.__doc__ or "")
            assert "subagent:" not in (fn.__doc__ or "")
            assert "handoff" not in (fn.__doc__ or "")


class TestRetrieveToolsBinding:
    @pytest.mark.asyncio
    async def test_binding_mode_stamps_the_wide_event(self):
        """The binding counts are how an operator tells "model asked for the wrong names" from "registry lost tools" in production events."""
        from app.agents.tools.core.retrieval import get_retrieve_tools_function
        from shared.py.wide_events import log

        log.reset()
        fn = get_retrieve_tools_function(include_subagents=True)
        store = MagicMock()
        config: dict = {"configurable": {"user_id": "u1"}}

        mock_registry = MagicMock()
        mock_registry.get_tool_names.return_value = ["TOOL_A", "TOOL_B"]
        mock_registry.get_category.return_value = SimpleNamespace(require_integration=False)

        with (
            patch(
                "app.agents.tools.core.retrieval.get_tool_registry",
                new_callable=AsyncMock,
                return_value=mock_registry,
            ),
            patch(
                "app.agents.tools.core.retrieval.resolve_tool",
                new=AsyncMock(return_value=None),
            ),
        ):
            await fn(
                store=store,
                config=config,
                exact_tool_names=["TOOL_A", "TOOL_C"],
            )

        assert log.get()["tool_retrieval"] == {
            "mode": "binding",
            "tools_requested": 2,
            "tools_bound": 1,
            "tools_proxied": 0,
            "tools_filtered": 1,
        }

    @pytest.mark.asyncio
    async def test_returns_corrective_when_no_args(self):
        from app.agents.tools.core.retrieval import get_retrieve_tools_function

        fn = get_retrieve_tools_function()
        store = MagicMock()
        config: dict = {"configurable": {"user_id": "u1"}}

        with patch(
            "app.agents.tools.core.retrieval.get_tool_registry",
            new_callable=AsyncMock,
        ):
            # The real no-usable-argument call is an empty list (a plain-array schema can't send
            # null); it returns corrective guidance instead of raising, so a recoverable model
            # slip doesn't abort the executor turn.
            result = await fn(store=store, config=config, exact_tool_names=[])

        assert result["tools_to_bind"] == []
        assert any("no usable argument" in r for r in result["response"])

    @pytest.mark.asyncio
    async def test_binding_mode_validates_tools(self):
        from app.agents.tools.core.retrieval import get_retrieve_tools_function

        fn = get_retrieve_tools_function(include_subagents=True)
        store = MagicMock()
        config: dict = {"configurable": {"user_id": "u1"}}

        mock_registry = MagicMock()
        mock_registry.get_tool_names.return_value = ["TOOL_A", "TOOL_B"]
        mock_registry.get_category.return_value = SimpleNamespace(require_integration=False)

        with (
            patch(
                "app.agents.tools.core.retrieval.get_tool_registry",
                new_callable=AsyncMock,
                return_value=mock_registry,
            ),
            patch(
                "app.agents.tools.core.retrieval.resolve_tool",
                new=AsyncMock(return_value=None),
            ),
        ):
            result = await fn(
                store=store,
                config=config,
                exact_tool_names=["TOOL_A", "TOOL_C"],
            )

        assert "TOOL_A" in result["tools_to_bind"]
        assert "TOOL_C" not in result["tools_to_bind"]

    @pytest.mark.asyncio
    async def test_binding_mode_reports_subagent_names_as_unknown(self):
        from app.agents.tools.core.retrieval import get_retrieve_tools_function

        fn = get_retrieve_tools_function(include_subagents=True)
        store = MagicMock()
        config: dict = {"configurable": {"user_id": "u1"}}

        mock_registry = MagicMock()
        mock_registry.get_tool_names.return_value = ["TOOL_A"]
        mock_registry.get_category.return_value = SimpleNamespace(require_integration=False)

        with (
            patch(
                "app.agents.tools.core.retrieval.get_tool_registry",
                new_callable=AsyncMock,
                return_value=mock_registry,
            ),
            patch(
                "app.agents.tools.core.retrieval.resolve_tool",
                new=AsyncMock(return_value=None),
            ),
        ):
            result = await fn(
                store=store,
                config=config,
                exact_tool_names=["subagent:gmail"],
            )

        # Discovery never returns subagents, so a subagent: name is reported
        # unknown — never as a bind, and with no handoff pointer.
        assert result["tools_to_bind"] == []
        assert "subagent:gmail" in result["response_text"]
        assert "handoff" not in result["response_text"]

    @pytest.mark.asyncio
    async def test_binding_mode_filters_subagents_when_disabled(self):
        from app.agents.tools.core.retrieval import get_retrieve_tools_function

        fn = get_retrieve_tools_function(include_subagents=False)
        store = MagicMock()
        config: dict = {"configurable": {"user_id": "u1"}}

        mock_registry = MagicMock()
        mock_registry.get_tool_names.return_value = ["TOOL_A"]
        mock_registry.get_category.return_value = SimpleNamespace(require_integration=False)

        with (
            patch(
                "app.agents.tools.core.retrieval.get_tool_registry",
                new_callable=AsyncMock,
                return_value=mock_registry,
            ),
            patch(
                "app.agents.tools.core.retrieval.resolve_tool",
                new=AsyncMock(return_value=None),
            ),
        ):
            result = await fn(
                store=store,
                config=config,
                exact_tool_names=["TOOL_A", "subagent:gmail"],
            )

        assert "TOOL_A" in result["tools_to_bind"]
        assert "subagent:gmail" not in result["tools_to_bind"]


class TestRetrieveToolsDiscovery:
    @pytest.mark.asyncio
    async def test_discovery_mode_stamps_the_wide_event(self):
        """Discovery telemetry answers "did the index have anything?" vs "did the filter drop it?" — pinned field by field, counts included."""
        from types import SimpleNamespace

        from app.agents.tools.core.retrieval import get_retrieve_tools_function
        from shared.py.wide_events import log

        log.reset()
        fn = get_retrieve_tools_function(include_subagents=False, limit=5)
        store = MagicMock()
        store.asearch = AsyncMock(
            return_value=[
                SimpleNamespace(key="TOOL_A", score=0.9, namespace=("general",), value={}),
                SimpleNamespace(key="TOOL_B", score=0.8, namespace=("general",), value={}),
            ]
        )
        config: dict = {"configurable": {"user_id": "u1"}}

        mock_registry = MagicMock()
        # Only TOOL_A is a known tool — TOOL_B is filtered out downstream.
        mock_registry.get_tool_names.return_value = ["TOOL_A"]

        with (
            patch(
                "app.agents.tools.core.retrieval.get_tool_registry",
                new_callable=AsyncMock,
                return_value=mock_registry,
            ),
            patch(
                "app.agents.tools.core.retrieval._get_user_context",
                new_callable=AsyncMock,
                return_value=({"general"}, {}),
            ),
            patch(
                "app.agents.tools.core.retrieval._user_mcp_tool_names",
                new_callable=AsyncMock,
                return_value=set(),
            ),
        ):
            await fn(store=store, config=config, query="send email", exact_tool_names=[])

        assert log.get()["tool_retrieval"] == {
            "mode": "discovery",
            "query": "send email",
            "tool_space": "general",
            "user_id": "u1",
            "namespaces_searched": ["general"],
            "tools_discovered": 1,
            "chroma_hits": 2,
            "public_hits": 0,
            "per_namespace_hits": {"general": 2},
            "candidates_after_filter": 1,
            "chroma_preview": ["('general',)::TOOL_A", "('general',)::TOOL_B"],
            "active_namespaces": [],
        }

    @pytest.mark.asyncio
    async def test_discovery_mode(self):
        from app.agents.tools.core.retrieval import get_retrieve_tools_function

        fn = get_retrieve_tools_function(include_subagents=False, limit=5)
        store = MagicMock()
        store.asearch = AsyncMock(return_value=[])
        config: dict = {"configurable": {"user_id": "u1"}}

        mock_registry = MagicMock()
        mock_registry.get_tool_names.return_value = ["TOOL_A"]

        with (
            patch(
                "app.agents.tools.core.retrieval.get_tool_registry",
                new_callable=AsyncMock,
                return_value=mock_registry,
            ),
            patch(
                "app.agents.tools.core.retrieval._get_user_context",
                new_callable=AsyncMock,
                # connected_integrations is now dict[str, str | None], not set
                return_value=({"general"}, {}),
            ),
        ):
            # Pass exact_tool_names=[] explicitly — the Field() default is a FieldInfo
            # object (truthy) when called directly, not an empty list.
            result = await fn(store=store, config=config, query="send email", exact_tool_names=[])

        assert result["tools_to_bind"] == []
        assert isinstance(result["response"], list)

    @pytest.mark.asyncio
    async def test_discovery_uses_metadata_fallback_for_user_id(self):
        from app.agents.tools.core.retrieval import get_retrieve_tools_function

        fn = get_retrieve_tools_function(include_subagents=False)
        store = MagicMock()
        store.asearch = AsyncMock(return_value=[])
        config: dict = {
            "configurable": {},
            "metadata": {"user_id": "from_metadata"},
        }

        mock_registry = MagicMock()
        mock_registry.get_tool_names.return_value = []

        with (
            patch(
                "app.agents.tools.core.retrieval.get_tool_registry",
                new_callable=AsyncMock,
                return_value=mock_registry,
            ),
            patch(
                "app.agents.tools.core.retrieval._get_user_context",
                new_callable=AsyncMock,
                # connected_integrations is now dict[str, str | None], not set
                return_value=({"general"}, {}),
            ) as mock_ctx,
        ):
            # Pass exact_tool_names=[] explicitly — the Field() default is a FieldInfo
            # object (truthy) when called directly, not an empty list.
            await fn(store=store, config=config, query="test", exact_tool_names=[])

        # user_id should have been resolved from metadata
        mock_ctx.assert_called_once()
        assert mock_ctx.call_args[0][0] == "from_metadata"


# ---------------------------------------------------------------------------
# Activated-namespace discovery
# ---------------------------------------------------------------------------


class TestActivatedNamespaceDiscovery:
    def _github_hit(self):
        item = MagicMock()
        item.key = "GITHUB_LIST_PULL_REQUESTS"
        item.score = 0.9
        item.namespace = ("github",)
        return item

    @pytest.mark.asyncio
    async def test_active_integration_namespace_is_searched(self):
        """An activated integration's tools must be discoverable from the
        activating run: discovery searches its namespace, so the "use
        retrieve_tools for the rest" the activation reply promises works."""
        from app.agents.tools.core import retrieval

        seen: list = []

        async def fake_asearch(namespace, query="", limit=25):
            seen.append(namespace)
            if namespace == ("github",):
                return [self._github_hit()]
            return []

        store = MagicMock()
        store.asearch = fake_asearch
        config: dict = {"configurable": {"user_id": "u1", "conversation_id": "c1"}}

        registry = MagicMock()
        registry.get_tool_names.return_value = ["GITHUB_LIST_PULL_REQUESTS"]
        category = MagicMock()
        # Genuinely delegated, like the real GITHUB category: the hit must
        # still survive because the namespace was activated in-conversation.
        category.is_delegated = True
        registry.get_category_of_tool.return_value = "github_cat"
        registry.get_category.return_value = category

        with (
            patch(
                "app.agents.tools.core.retrieval.get_tool_registry",
                new_callable=AsyncMock,
                return_value=registry,
            ),
            patch(
                "app.agents.tools.core.retrieval._get_user_context",
                new_callable=AsyncMock,
                return_value=({"general"}, {}),
            ),
            patch(
                "app.agents.tools.core.retrieval.get_active",
                new_callable=AsyncMock,
                return_value={"github"},
            ),
        ):
            fn = retrieval.get_retrieve_tools_function(tool_space="general")
            result = await fn(store=store, config=config, query="list pull requests", exact_tool_names=[])

        assert ("github",) in seen
        assert "GITHUB_LIST_PULL_REQUESTS" in result["response"]

    @pytest.mark.asyncio
    async def test_no_active_integrations_searches_no_extra_namespace(self):
        from app.agents.tools.core import retrieval

        seen: list = []

        async def fake_asearch(namespace, query="", limit=25):
            seen.append(namespace)
            return []

        store = MagicMock()
        store.asearch = fake_asearch
        config: dict = {"configurable": {"user_id": "u1", "conversation_id": "c1"}}

        registry = MagicMock()
        registry.get_tool_names.return_value = []

        with (
            patch(
                "app.agents.tools.core.retrieval.get_tool_registry",
                new_callable=AsyncMock,
                return_value=registry,
            ),
            patch(
                "app.agents.tools.core.retrieval._get_user_context",
                new_callable=AsyncMock,
                return_value=({"general"}, {}),
            ),
            patch(
                "app.agents.tools.core.retrieval.get_active",
                new_callable=AsyncMock,
                return_value=set(),
            ),
        ):
            fn = retrieval.get_retrieve_tools_function(tool_space="general")
            await fn(store=store, config=config, query="list pull requests", exact_tool_names=[])

        assert ("github",) not in seen
        assert set(seen) <= {("general",), ("subagents",)}

    @pytest.mark.asyncio
    async def test_unknown_active_id_is_dropped_with_warning(self):
        """A stamped id the registry no longer knows never reaches the store."""
        from app.agents.tools.core import retrieval

        with patch(
            "app.agents.tools.core.retrieval.get_active",
            new_callable=AsyncMock,
            return_value={"deleted_integration"},
        ):
            assert await retrieval._active_tool_spaces("c1") == set()


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Activated-namespace exemption: activated tools surface as real tools
# ---------------------------------------------------------------------------


class TestActivatedNamespaceTools:
    """An activated integration's tools must reach the model as tools (run via
    execute), never as subagent pointers — and only that namespace is exempt:
    every other delegated tool stays hidden."""

    def _delegated_hit(self, key="GITHUB_LIST_PULL_REQUESTS", namespace=("github",)):
        item = MagicMock()
        item.key = key
        item.score = 0.9
        item.namespace = namespace
        return item

    def _delegated_registry(self):
        registry = MagicMock()
        registry.get_category_of_tool.return_value = "GITHUB"
        category = MagicMock()
        category.is_delegated = True
        registry.get_category.return_value = category
        return registry

    def test_activated_namespace_delegated_hit_survives(self):
        from app.agents.tools.core.retrieval import _process_chroma_search_result

        result = _process_chroma_search_result(
            [self._delegated_hit()],
            {"GITHUB_LIST_PULL_REQUESTS"},
            self._delegated_registry(),
            include_subagents=True,
            active_namespaces={"github"},
        )
        assert [r["id"] for r in result] == ["GITHUB_LIST_PULL_REQUESTS"]

    def test_non_activated_delegated_hit_still_dropped(self):
        from app.agents.tools.core.retrieval import _process_chroma_search_result

        result = _process_chroma_search_result(
            [self._delegated_hit()],
            {"GITHUB_LIST_PULL_REQUESTS"},
            self._delegated_registry(),
            include_subagents=True,
            active_namespaces={"gmail"},
        )
        assert result == []

    def test_no_stamp_still_drops(self):
        from app.agents.tools.core.retrieval import _process_chroma_search_result

        result = _process_chroma_search_result(
            [self._delegated_hit()],
            {"GITHUB_LIST_PULL_REQUESTS"},
            self._delegated_registry(),
            include_subagents=True,
            active_namespaces=set(),
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_discovery_end_to_end_no_provider_subagent_surface(self):
        """The reported bug: a github-namespace Chroma hit reached the model as
        nothing (delegated filter) next to a subagent:github pointer. Now the
        real tool is in the response and no provider subagent surface exists."""
        import json

        from app.agents.tools.core import retrieval
        from shared.py.wide_events import log

        item = MagicMock()
        item.key = "GITHUB_LIST_PULL_REQUESTS"
        item.score = 0.9
        item.namespace = ("github",)

        async def fake_asearch(namespace, query="", limit=25):
            if namespace == ("github",):
                return [item]
            return []

        store = MagicMock()
        store.asearch = fake_asearch
        config: dict = {"configurable": {"user_id": "u1", "conversation_id": "c1"}}

        registry = self._delegated_registry()
        registry.get_tool_names.return_value = ["GITHUB_LIST_PULL_REQUESTS"]
        registry.get_tool_meta.return_value = None

        log.reset()
        with (
            patch(
                "app.agents.tools.core.retrieval.get_tool_registry",
                new_callable=AsyncMock,
                return_value=registry,
            ),
            patch(
                "app.agents.tools.core.retrieval._get_user_context",
                new_callable=AsyncMock,
                return_value=({"general", "github"}, {"github": "GitHub"}),
            ),
            patch(
                "app.agents.tools.core.retrieval.get_active",
                new_callable=AsyncMock,
                return_value={"github"},
            ),
            patch(
                "app.agents.tools.core.retrieval._user_mcp_tool_names",
                new_callable=AsyncMock,
                return_value=set(),
            ),
            patch(
                "app.agents.tools.core.retrieval.search_public_integrations",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            fn = retrieval.get_retrieve_tools_function(tool_space="general")
            result = await fn(
                store=store, config=config, query="list pull requests", exact_tool_names=[]
            )

        assert "GITHUB_LIST_PULL_REQUESTS" in result["response"]
        assert not [e for e in result["response"] if e.startswith("subagent:")]
        body = json.loads(result["response_text"])
        assert body["mcp_subagents"] == []
        assert body["new_integrations"] == []
        assert "handoff" not in result["response_text"]

    def test_mcp_pointer_in_subagents_namespace_survives(self):
        """Custom MCP pointers are the one subagent surface: source=='custom'
        entries render as subagent: entries for handoff."""
        from app.agents.tools.core.retrieval import _process_chroma_search_result

        item = self._delegated_hit("my-mcp", namespace=("subagents",))
        item.value = {"name": "My MCP", "source": "custom"}
        registry = MagicMock()
        result = _process_chroma_search_result(
            [item], set(), registry, include_subagents=True
        )
        assert [r["id"] for r in result] == ["subagent:my-mcp (My MCP)"]

    def test_stale_provider_doc_in_subagents_namespace_dropped(self):
        """Docs without source=='custom' (pre-removal provider entries) never
        surface, even though the namespace is searched."""
        from app.agents.tools.core.retrieval import _process_chroma_search_result

        item = self._delegated_hit("github", namespace=("subagents",))
        item.value = {"name": "GitHub"}
        registry = MagicMock()
        result = _process_chroma_search_result(
            [item], {"github"}, registry, include_subagents=True
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_public_hits_render_as_integration_entries(self):
        """Marketplace hits render as integration: entries — activatable, never
        handed off."""
        from app.agents.tools.core.retrieval import _process_search_results

        registry = MagicMock()
        public = [{"integration_id": "linear", "name": "Linear", "relevance_score": 0.7}]
        processed = await _process_search_results(
            [public], set(), registry, include_subagents=True
        )
        assert [r["id"] for r in processed] == ["integration:linear (Linear)"]
