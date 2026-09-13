"""Unit tests for the Linear custom tools (linear_tool.py).

Strategy: register_linear_custom_tools decorates inner functions with
@composio.tools.custom_tool(); a capturing Composio mock records them so
they can be called directly. The only seam faked is the Composio proxy under
linear_utils.graphql_request, so every response runs through the real
GraphQL envelope parsing and the typed Linear models, fed with fixtures shaped
like Linear's schema (every field the operation selects).
"""

from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

from pydantic import ValidationError
import pytest

from app.agents.tools.integrations.linear_tool import register_linear_custom_tools
from app.models.common_models import GatherContextInput
from app.models.linear_models import (
    BulkUpdateIssuesInput,
    CreateIssueInput,
    CreateIssueRelationInput,
    CreateSubIssuesInput,
    GetActiveSprintInput,
    GetIssueActivityInput,
    GetIssueFullContextInput,
    GetMyTasksInput,
    GetNotificationsInput,
    GetWorkspaceContextInput,
    ResolveContextInput,
    SearchIssuesInput,
    SubIssueItem,
)
from app.utils.linear_utils import (
    MUTATION_CREATE_ISSUE,
    MUTATION_UPDATE_ISSUES,
    QUERY_ISSUE_BY_ID,
    QUERY_LABELS,
    QUERY_LABELS_ALL,
)

LINEAR_MODULE = "app.agents.tools.integrations.linear_tool"
PROXY = "app.utils.linear_utils.proxy_request_sync"
AUTH_CREDS: dict[str, Any] = {"user_id": "user-123", "version": "v1"}
EXECUTE_REQUEST = MagicMock()


def _capture_tools() -> dict[str, Any]:
    composio = MagicMock()
    captured: dict[str, Any] = {}

    def capturing_custom_tool(**_kwargs: Any) -> Any:
        def wrapper(fn: Any) -> Any:
            captured[fn.__name__] = fn
            return fn

        return wrapper

    composio.tools.custom_tool = capturing_custom_tool
    register_linear_custom_tools(composio)
    return captured


@pytest.fixture
def tools() -> dict[str, Any]:
    return _capture_tools()


@pytest.fixture
def proxy() -> Iterator[MagicMock]:
    with patch(PROXY) as mock:
        yield mock


def _answers(proxy: MagicMock, *datas: dict[str, Any]) -> None:
    """Answer successive GraphQL calls with {"data": ...} in order."""
    proxy.side_effect = [{"data": d} for d in datas]


def _bodies(proxy: MagicMock) -> list[dict[str, Any]]:
    return [c.args[0].body for c in proxy.call_args_list]


VIEWER = {
    "viewer": {
        "id": "u1",
        "name": "Alice",
        "email": "a@b.com",
        "assignedIssues": {"nodes": [{"id": "x"}]},
    }
}
TEAM = {"id": "t1", "key": "ENG", "name": "Eng"}


def _summary(issue_id: str, **overrides: Any) -> dict[str, Any]:
    """Build an issue as QUERY_MY_ISSUES / QUERY_SEARCH_ISSUES select it."""
    node: dict[str, Any] = {
        "id": issue_id,
        "identifier": f"ENG-{issue_id}",
        "title": f"Task {issue_id}",
        "priority": 3,
        "state": {"id": "s1", "name": "In Progress", "type": "started"},
        "dueDate": None,
        "team": TEAM,
        "cycle": None,
        "parent": None,
        "assignee": None,
    }
    node.update(overrides)
    return node


def _full_issue(**overrides: Any) -> dict[str, Any]:
    """Build an issue as QUERY_ISSUE_BY_ID selects it."""
    node: dict[str, Any] = {
        "id": "i1",
        "identifier": "ENG-1",
        "title": "Bug",
        "description": "desc",
        "priority": 1,
        "state": {"id": "s1", "name": "In Progress", "type": "started"},
        "dueDate": None,
        "estimate": 3,
        "team": TEAM,
        "cycle": None,
        "project": {"id": "p1", "name": "GAIA"},
        "assignee": {"id": "u1", "name": "Alice", "email": "a@b.com"},
        "creator": {"id": "u2", "name": "Bob"},
        "parent": None,
        "children": {"nodes": []},
        "relations": {"nodes": []},
        "comments": {"nodes": []},
        "history": {"nodes": []},
        "attachments": {"nodes": []},
    }
    node.update(overrides)
    return node


def _history(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": "h1",
        "createdAt": "2024-01-01",
        "actor": {"id": "u1", "name": "Alice"},
        "fromState": None,
        "toState": None,
        "fromAssignee": None,
        "toAssignee": None,
        "fromPriority": None,
        "toPriority": None,
        "addedLabels": None,
        "removedLabels": None,
    }
    entry.update(overrides)
    return entry


# =============================================================================
# CUSTOM_RESOLVE_CONTEXT
# =============================================================================


class TestLinearResolveContext:
    def test_resolve_context_basic(self, tools, proxy) -> None:
        _answers(proxy, VIEWER)

        result = tools["CUSTOM_RESOLVE_CONTEXT"](ResolveContextInput(), EXECUTE_REQUEST, AUTH_CREDS)

        assert result == {
            "data": {"current_user": {"id": "u1", "name": "Alice", "email": "a@b.com"}}
        }
        assert proxy.call_count == 1

    def test_resolve_context_with_team_name(self, tools, proxy) -> None:
        eng = {**TEAM, "activeCycle": {"id": "c1", "name": "Sprint 5", "progress": 0.5}}
        design = {"id": "t2", "key": "DES", "name": "Design", "activeCycle": None}
        _answers(proxy, VIEWER, {"teams": {"nodes": [design, eng]}})

        result = tools["CUSTOM_RESOLVE_CONTEXT"](
            ResolveContextInput(team_name="eng"), EXECUTE_REQUEST, AUTH_CREDS
        )

        assert result["data"]["teams"] == [
            {
                "id": "t1",
                "name": "Eng",
                "key": "ENG",
                "activeCycle": {"id": "c1", "name": "Sprint 5", "progress": 0.5},
            },
            # "eng" is within SequenceMatcher's 0.4 threshold of "design".
            {"id": "t2", "name": "Design", "key": "DES", "activeCycle": None},
        ]

    def test_resolve_context_with_user_name(self, tools, proxy) -> None:
        _answers(
            proxy,
            VIEWER,
            {
                "users": {
                    "nodes": [
                        {"id": "u2", "name": "Bob", "email": "b@b.com", "active": True},
                        {"id": "u3", "name": "Bobby", "email": "c@b.com", "active": False},
                    ]
                }
            },
        )

        result = tools["CUSTOM_RESOLVE_CONTEXT"](
            ResolveContextInput(user_name="bob"), EXECUTE_REQUEST, AUTH_CREDS
        )

        assert result["data"]["users"] == [
            {"id": "u2", "name": "Bob", "email": "b@b.com", "active": True}
        ]

    def test_resolve_context_labels_with_team_id(self, tools, proxy) -> None:
        _answers(
            proxy,
            VIEWER,
            {"issueLabels": {"nodes": [{"id": "l1", "name": "Bug", "color": "#f00"}]}},
        )

        result = tools["CUSTOM_RESOLVE_CONTEXT"](
            ResolveContextInput(label_names=["bug"], team_id="t1"), EXECUTE_REQUEST, AUTH_CREDS
        )

        assert result["data"]["labels"] == [{"id": "l1", "name": "Bug", "color": "#f00"}]
        assert _bodies(proxy)[1] == {"query": QUERY_LABELS, "variables": {"teamId": "t1"}}

    def test_resolve_context_labels_without_team_id(self, tools, proxy) -> None:
        _answers(
            proxy,
            VIEWER,
            {
                "issueLabels": {
                    "nodes": [
                        {"id": "l1", "name": "Bug", "color": "#f00"},
                        {"id": "l2", "name": "Feature", "color": "#0f0"},
                    ]
                }
            },
        )

        result = tools["CUSTOM_RESOLVE_CONTEXT"](
            ResolveContextInput(label_names=["bug", "feat"]), EXECUTE_REQUEST, AUTH_CREDS
        )

        assert [label["id"] for label in result["data"]["labels"]] == ["l1", "l2"]
        assert _bodies(proxy)[1] == {"query": QUERY_LABELS_ALL}

    def test_resolve_context_with_project_name(self, tools, proxy) -> None:
        _answers(
            proxy,
            VIEWER,
            {
                "projects": {
                    "nodes": [{"id": "p1", "name": "GAIA", "state": "started", "progress": 0.2}]
                }
            },
        )

        result = tools["CUSTOM_RESOLVE_CONTEXT"](
            ResolveContextInput(project_name="gaia"), EXECUTE_REQUEST, AUTH_CREDS
        )

        assert result["data"]["projects"] == [
            {"id": "p1", "name": "GAIA", "state": "started", "progress": 0.2}
        ]

    def test_resolve_context_with_state_and_team(self, tools, proxy) -> None:
        _answers(
            proxy,
            VIEWER,
            {
                "workflowStates": {
                    "nodes": [
                        {"id": "s1", "name": "In Progress", "type": "started", "position": 2.0}
                    ]
                }
            },
        )

        result = tools["CUSTOM_RESOLVE_CONTEXT"](
            ResolveContextInput(state_name="progress", team_id="t1"), EXECUTE_REQUEST, AUTH_CREDS
        )

        assert result["data"]["states"] == [
            {"id": "s1", "name": "In Progress", "type": "started", "position": 2.0}
        ]


# =============================================================================
# CUSTOM_GET_MY_TASKS
# =============================================================================


class TestLinearGetMyTasks:
    def _run(self, tools, proxy, request: GetMyTasksInput, *nodes: dict[str, Any]) -> Any:
        _answers(proxy, VIEWER, {"issues": {"nodes": list(nodes)}})
        with patch(f"{LINEAR_MODULE}._user_local_today", return_value=datetime.now().date()):
            return tools["CUSTOM_GET_MY_TASKS"](request, EXECUTE_REQUEST, AUTH_CREDS)

    def test_get_my_tasks_all_filter(self, tools, proxy) -> None:
        result = self._run(
            tools,
            proxy,
            GetMyTasksInput(filter="all", limit=10),
            _summary("2", priority=3),
            _summary("1", priority=1, assignee={"id": "u1", "name": "Alice"}),
        )

        assert result["filter"] == "all"
        assert result["count"] == 2
        assert result["issues"][0] == {
            "id": "1",
            "identifier": "ENG-1",
            "title": "Task 1",
            "state": "In Progress",
            "priority": "urgent",
            "assignee": "Alice",
            "dueDate": None,
            "team": "ENG",
            "cycle": None,
            "parent": None,
        }
        assert _bodies(proxy)[1]["variables"] == {
            "assigneeId": "u1",
            "includeCompleted": True,
            "first": 20,
        }

    def test_get_my_tasks_no_viewer(self, tools, proxy) -> None:
        """Linear's schema makes viewer non-null: a body without it is a provider fault."""
        _answers(proxy, {})

        with pytest.raises(ValidationError):
            tools["CUSTOM_GET_MY_TASKS"](GetMyTasksInput(), EXECUTE_REQUEST, AUTH_CREDS)

    def test_get_my_tasks_high_priority_filter(self, tools, proxy) -> None:
        result = self._run(
            tools,
            proxy,
            GetMyTasksInput(filter="high_priority"),
            _summary("1", priority=1),
            _summary("2", priority=4),
        )
        assert [i["id"] for i in result["issues"]] == ["1"]

    def test_get_my_tasks_overdue_filter(self, tools, proxy) -> None:
        yesterday = (datetime.now().date() - timedelta(days=1)).isoformat()
        result = self._run(
            tools,
            proxy,
            GetMyTasksInput(filter="overdue"),
            _summary("1", dueDate=yesterday),
            _summary("2"),
        )
        assert [i["id"] for i in result["issues"]] == ["1"]

    def test_get_my_tasks_excludes_completed(self, tools, proxy) -> None:
        result = self._run(
            tools,
            proxy,
            GetMyTasksInput(filter="all", include_completed=False),
            _summary("1", state={"id": "s9", "name": "Done", "type": "completed"}),
            _summary("2"),
        )
        assert [i["id"] for i in result["issues"]] == ["2"]

    def test_get_my_tasks_today_filter(self, tools, proxy) -> None:
        today = datetime.now().date().isoformat()
        result = self._run(
            tools,
            proxy,
            GetMyTasksInput(filter="today"),
            _summary("1", dueDate=today),
            _summary("2"),
        )
        assert [i["id"] for i in result["issues"]] == ["1"]

    def test_get_my_tasks_this_week_filter(self, tools, proxy) -> None:
        tomorrow = (datetime.now().date() + timedelta(days=1)).isoformat()
        far = (datetime.now().date() + timedelta(days=30)).isoformat()
        result = self._run(
            tools,
            proxy,
            GetMyTasksInput(filter="this_week"),
            _summary("1", dueDate=tomorrow),
            _summary("2", dueDate=far),
        )
        assert [i["id"] for i in result["issues"]] == ["1"]


# =============================================================================
# CUSTOM_SEARCH_ISSUES
# =============================================================================


def _search_node(issue_id: str, **overrides: Any) -> dict[str, Any]:
    return _summary(issue_id, **{"createdAt": "2024-01-01", **overrides})


class TestLinearSearchIssues:
    def _run(self, tools, proxy, request: SearchIssuesInput, *nodes: dict[str, Any]) -> Any:
        _answers(proxy, {"searchIssues": {"nodes": list(nodes)}})
        return tools["CUSTOM_SEARCH_ISSUES"](request, EXECUTE_REQUEST, AUTH_CREDS)

    def test_search_issues_basic(self, tools, proxy) -> None:
        result = self._run(tools, proxy, SearchIssuesInput(query="bug"), _search_node("1"))

        assert result["query"] == "bug"
        assert result["count"] == 1
        assert _bodies(proxy)[0]["variables"] == {"query": "bug", "first": 40}

    def test_search_issues_with_team_filter(self, tools, proxy) -> None:
        other = {"id": "t2", "key": "DES", "name": "Design"}
        result = self._run(
            tools,
            proxy,
            SearchIssuesInput(query="test", team_id="t1"),
            _search_node("1"),
            _search_node("2", team=other),
        )
        assert [i["id"] for i in result["issues"]] == ["1"]

    def test_search_issues_with_state_filter(self, tools, proxy) -> None:
        result = self._run(
            tools,
            proxy,
            SearchIssuesInput(query="test", state_filter="completed"),
            _search_node("1", state={"id": "s9", "name": "Done", "type": "completed"}),
            _search_node("2"),
        )
        assert [i["id"] for i in result["issues"]] == ["1"]

    def test_search_issues_with_assignee_filter(self, tools, proxy) -> None:
        result = self._run(
            tools,
            proxy,
            SearchIssuesInput(query="test", assignee_id="u1"),
            _search_node("1", assignee={"id": "u1", "name": "Alice"}),
            _search_node("2", assignee={"id": "u2", "name": "Bob"}),
            _search_node("3"),
        )
        assert [i["id"] for i in result["issues"]] == ["1"]

    def test_search_issues_with_priority_filter(self, tools, proxy) -> None:
        result = self._run(
            tools,
            proxy,
            SearchIssuesInput(query="test", priority_filter="urgent"),
            _search_node("1", priority=1),
            _search_node("2", priority=3),
        )
        assert [i["id"] for i in result["issues"]] == ["1"]

    def test_search_issues_with_created_after(self, tools, proxy) -> None:
        result = self._run(
            tools,
            proxy,
            SearchIssuesInput(query="test", created_after="2024-03-01"),
            _search_node("1", createdAt="2024-06-01"),
            _search_node("2", createdAt="2024-01-01"),
        )
        assert [i["id"] for i in result["issues"]] == ["1"]


# =============================================================================
# CUSTOM_GET_ISSUE_FULL_CONTEXT
# =============================================================================


class TestLinearGetIssueFullContext:
    def test_get_issue_by_id(self, tools, proxy) -> None:
        _answers(proxy, {"issue": _full_issue()})

        result = tools["CUSTOM_GET_ISSUE_FULL_CONTEXT"](
            GetIssueFullContextInput(issue_id="i1"), EXECUTE_REQUEST, AUTH_CREDS
        )

        assert result == {
            "issue": {
                "id": "i1",
                "identifier": "ENG-1",
                "title": "Bug",
                "description": "desc",
                "priority": "urgent",
                "state": "In Progress",
                "dueDate": None,
                "estimate": 3,
                "team": "Eng",
                "project": "GAIA",
                "cycle": None,
                "assignee": "Alice",
                "creator": "Bob",
            }
        }
        assert _bodies(proxy) == [{"query": QUERY_ISSUE_BY_ID, "variables": {"id": "i1"}}]

    def test_get_issue_by_identifier(self, tools, proxy) -> None:
        _answers(proxy, {"issue": _full_issue(identifier="ENG-123")})

        result = tools["CUSTOM_GET_ISSUE_FULL_CONTEXT"](
            GetIssueFullContextInput(issue_identifier="ENG-123"), EXECUTE_REQUEST, AUTH_CREDS
        )

        assert result["issue"]["identifier"] == "ENG-123"
        assert _bodies(proxy)[0]["variables"] == {"id": "ENG-123"}

    def test_get_issue_no_id_or_identifier(self, tools) -> None:
        with pytest.raises(ValueError, match="Provide either"):
            tools["CUSTOM_GET_ISSUE_FULL_CONTEXT"](
                GetIssueFullContextInput(), EXECUTE_REQUEST, AUTH_CREDS
            )

    def test_get_issue_invalid_identifier_format(self, tools, proxy) -> None:
        with pytest.raises(ValueError, match="Invalid identifier format"):
            tools["CUSTOM_GET_ISSUE_FULL_CONTEXT"](
                GetIssueFullContextInput(issue_identifier="BADFORMAT"), EXECUTE_REQUEST, AUTH_CREDS
            )
        proxy.assert_not_called()

    def test_get_issue_invalid_number_in_identifier(self, tools, proxy) -> None:
        with pytest.raises(ValueError, match="Invalid issue number"):
            tools["CUSTOM_GET_ISSUE_FULL_CONTEXT"](
                GetIssueFullContextInput(issue_identifier="ENG-abc"), EXECUTE_REQUEST, AUTH_CREDS
            )
        proxy.assert_not_called()

    def test_get_issue_not_found(self, tools, proxy) -> None:
        """issue(id:) is non-null: Linear answers an unknown id with a GraphQL error."""
        proxy.return_value = {"data": None, "errors": [{"message": "Entity not found: Issue"}]}

        with pytest.raises(Exception, match="GraphQL errors: Entity not found: Issue"):
            tools["CUSTOM_GET_ISSUE_FULL_CONTEXT"](
                GetIssueFullContextInput(issue_id="nonexistent"), EXECUTE_REQUEST, AUTH_CREDS
            )

    def test_get_issue_with_children_and_relations(self, tools, proxy) -> None:
        _answers(
            proxy,
            {
                "issue": _full_issue(
                    parent={"id": "i0", "identifier": "ENG-0", "title": "Grand"},
                    children={
                        "nodes": [
                            {
                                "id": "i2",
                                "identifier": "ENG-2",
                                "title": "Sub",
                                "state": {"name": "Done"},
                            }
                        ]
                    },
                    relations={
                        "nodes": [
                            {
                                "id": "r1",
                                "type": "blocks",
                                "relatedIssue": {"id": "i3", "identifier": "ENG-3", "title": "Dep"},
                            }
                        ]
                    },
                    comments={
                        "nodes": [
                            {
                                "id": "cm1",
                                "body": "comment",
                                "createdAt": "2024-01-01",
                                "user": {"id": "u1", "name": "Alice"},
                            },
                            {"id": "cm2", "body": "bot", "createdAt": "2024-01-02", "user": None},
                        ]
                    },
                    history={
                        "nodes": [
                            _history(
                                fromState={"id": "s0", "name": "Todo", "type": "unstarted"},
                                toState={"id": "s9", "name": "Done", "type": "completed"},
                            ),
                            _history(
                                id="h2", actor=None, removedLabels=[{"id": "l1", "name": "Bug"}]
                            ),
                            _history(id="h3"),
                        ]
                    },
                    attachments={
                        "nodes": [
                            {"id": "a1", "title": "file.pdf", "url": "https://example.com/file.pdf"}
                        ]
                    },
                )
            },
        )

        issue = tools["CUSTOM_GET_ISSUE_FULL_CONTEXT"](
            GetIssueFullContextInput(issue_id="i1"), EXECUTE_REQUEST, AUTH_CREDS
        )["issue"]

        assert issue["parent"] == {"identifier": "ENG-0", "title": "Grand"}
        assert issue["sub_issues"] == [{"identifier": "ENG-2", "title": "Sub", "state": "Done"}]
        assert issue["relations"] == [
            {"type": "blocks", "issue": {"identifier": "ENG-3", "title": "Dep"}}
        ]
        assert issue["comments"] == [
            {"author": "Alice", "body": "comment", "createdAt": "2024-01-01"},
            {"author": None, "body": "bot", "createdAt": "2024-01-02"},
        ]
        assert issue["activity"] == [
            {
                "timestamp": "2024-01-01",
                "actor": "Alice",
                "change": "state",
                "from": "Todo",
                "to": "Done",
            },
            {
                "timestamp": "2024-01-01",
                "actor": None,
                "change": "labels_removed",
                "labels": ["Bug"],
            },
        ]
        assert issue["attachments"] == [
            {"title": "file.pdf", "url": "https://example.com/file.pdf"}
        ]


# =============================================================================
# CUSTOM_CREATE_ISSUE / CUSTOM_CREATE_SUB_ISSUES / CUSTOM_CREATE_ISSUE_RELATION
# =============================================================================


def _created(issue_id: str, title: str) -> dict[str, Any]:
    return {
        "issueCreate": {
            "success": True,
            "issue": {
                "id": issue_id,
                "identifier": f"ENG-{issue_id}",
                "title": title,
                "url": f"https://linear.app/eng-{issue_id}",
            },
        }
    }


class TestLinearCreateIssue:
    def test_create_issue_basic(self, tools, proxy) -> None:
        _answers(proxy, _created("1", "New Bug"))

        result = tools["CUSTOM_CREATE_ISSUE"](
            CreateIssueInput(team_id="t1", title="New Bug"), EXECUTE_REQUEST, AUTH_CREDS
        )

        assert result == {
            "issue": {
                "id": "1",
                "identifier": "ENG-1",
                "title": "New Bug",
                "url": "https://linear.app/eng-1",
            }
        }
        assert _bodies(proxy) == [
            {
                "query": MUTATION_CREATE_ISSUE,
                "variables": {"input": {"teamId": "t1", "title": "New Bug", "priority": 0}},
            }
        ]

    def test_create_issue_failure(self, tools, proxy) -> None:
        _answers(proxy, {"issueCreate": {"success": False, "issue": None}})

        with pytest.raises(RuntimeError, match="Failed to create issue"):
            tools["CUSTOM_CREATE_ISSUE"](
                CreateIssueInput(team_id="t1", title="Fail"), EXECUTE_REQUEST, AUTH_CREDS
            )

    def test_create_issue_with_all_fields(self, tools, proxy) -> None:
        _answers(proxy, _created("1", "Full"))

        tools["CUSTOM_CREATE_ISSUE"](
            CreateIssueInput(
                team_id="t1",
                title="Full",
                description="Desc",
                assignee_id="u1",
                priority=2,
                state_id="s1",
                label_ids=["l1"],
                project_id="p1",
                cycle_id="c1",
                due_date="2024-12-31",
                estimate=5,
                parent_id="parent-1",
            ),
            EXECUTE_REQUEST,
            AUTH_CREDS,
        )

        assert _bodies(proxy)[0]["variables"] == {
            "input": {
                "teamId": "t1",
                "title": "Full",
                "description": "Desc",
                "assigneeId": "u1",
                "priority": 2,
                "stateId": "s1",
                "labelIds": ["l1"],
                "projectId": "p1",
                "cycleId": "c1",
                "dueDate": "2024-12-31",
                "estimate": 5,
                "parentId": "parent-1",
            }
        }


class TestLinearCreateSubIssues:
    def test_create_sub_issues_with_parent_id(self, tools, proxy) -> None:
        _answers(
            proxy,
            {"issue": _full_issue(id="parent-1")},
            _created("2", "Sub 1"),
            {"issueCreate": {"success": False, "issue": None}},
        )

        result = tools["CUSTOM_CREATE_SUB_ISSUES"](
            CreateSubIssuesInput(
                parent_issue_id="parent-1",
                sub_issues=[SubIssueItem(title="Sub 1", priority=2), SubIssueItem(title="Sub 2")],
            ),
            EXECUTE_REQUEST,
            AUTH_CREDS,
        )

        assert result == {
            "parent": "parent-1",
            "created_count": 1,
            "sub_issues": [{"id": "2", "identifier": "ENG-2", "title": "Sub 1"}],
        }
        assert _bodies(proxy)[1]["variables"] == {
            "input": {"teamId": "t1", "title": "Sub 1", "parentId": "parent-1", "priority": 2}
        }

    def test_create_sub_issues_by_parent_identifier_fetches_it_by_identifier(
        self, tools, proxy
    ) -> None:
        _answers(proxy, {"issue": _full_issue(id="parent-1")}, _created("2", "Sub 1"))

        result = tools["CUSTOM_CREATE_SUB_ISSUES"](
            CreateSubIssuesInput(
                parent_identifier="ENG-1", sub_issues=[SubIssueItem(title="Sub 1")]
            ),
            EXECUTE_REQUEST,
            AUTH_CREDS,
        )

        assert result["parent"] == "ENG-1"
        assert _bodies(proxy)[0]["variables"] == {"id": "ENG-1"}
        assert _bodies(proxy)[1]["variables"]["input"]["parentId"] == "parent-1"

    def test_create_sub_issues_no_parent(self, tools, proxy) -> None:
        with pytest.raises(ValueError, match="Could not resolve parent"):
            tools["CUSTOM_CREATE_SUB_ISSUES"](
                CreateSubIssuesInput(sub_issues=[SubIssueItem(title="Sub 1")]),
                EXECUTE_REQUEST,
                AUTH_CREDS,
            )
        proxy.assert_not_called()


class TestLinearCreateIssueRelation:
    def test_create_relation_success(self, tools, proxy) -> None:
        _answers(
            proxy,
            {
                "issueRelationCreate": {
                    "success": True,
                    "issueRelation": {"id": "r1", "type": "blocked_by"},
                }
            },
        )

        result = tools["CUSTOM_CREATE_ISSUE_RELATION"](
            CreateIssueRelationInput(
                issue_id="i1", related_issue_id="i2", relation_type="is_blocked_by"
            ),
            EXECUTE_REQUEST,
            AUTH_CREDS,
        )

        assert result == {
            "relation": {"id": "r1", "type": "is_blocked_by", "from_issue": "i1", "to_issue": "i2"}
        }
        assert _bodies(proxy)[0]["variables"] == {
            "issueId": "i1",
            "relatedIssueId": "i2",
            "type": "blocked_by",
        }

    def test_create_relation_failure(self, tools, proxy) -> None:
        _answers(
            proxy,
            {
                "issueRelationCreate": {
                    "success": False,
                    "issueRelation": {"id": "r1", "type": "blocks"},
                }
            },
        )

        with pytest.raises(RuntimeError, match="Failed to create relation"):
            tools["CUSTOM_CREATE_ISSUE_RELATION"](
                CreateIssueRelationInput(
                    issue_id="i1", related_issue_id="i2", relation_type="blocks"
                ),
                EXECUTE_REQUEST,
                AUTH_CREDS,
            )


# =============================================================================
# CUSTOM_GET_ISSUE_ACTIVITY
# =============================================================================


class TestLinearGetIssueActivity:
    def _run(self, tools, proxy, *entries: dict[str, Any]) -> Any:
        _answers(proxy, {"issue": {"history": {"nodes": list(entries)}}})
        return tools["CUSTOM_GET_ISSUE_ACTIVITY"](
            GetIssueActivityInput(issue_id="i1"), EXECUTE_REQUEST, AUTH_CREDS
        )

    def test_get_activity_by_id(self, tools, proxy) -> None:
        result = self._run(
            tools,
            proxy,
            _history(
                fromState={"id": "s0", "name": "Todo", "type": "unstarted"},
                toState={"id": "s9", "name": "Done", "type": "completed"},
            ),
            _history(id="h2"),
        )

        assert result == {
            "issue": "i1",
            "activity_count": 1,
            "activities": [
                {
                    "timestamp": "2024-01-01",
                    "actor": "Alice",
                    "change_type": "state",
                    "from": "Todo",
                    "to": "Done",
                }
            ],
        }
        assert _bodies(proxy)[0]["variables"] == {"issueId": "i1", "first": 10}

    def test_get_activity_by_identifier(self, tools, proxy) -> None:
        _answers(proxy, {"issue": _full_issue(id="i9")}, {"issue": {"history": {"nodes": []}}})

        result = tools["CUSTOM_GET_ISSUE_ACTIVITY"](
            GetIssueActivityInput(issue_identifier="ENG-123"), EXECUTE_REQUEST, AUTH_CREDS
        )

        assert result == {"issue": "ENG-123", "activity_count": 0, "activities": []}
        assert _bodies(proxy)[1]["variables"]["issueId"] == "i9"

    def test_get_activity_no_issue(self, tools) -> None:
        with pytest.raises(ValueError, match="Could not resolve issue"):
            tools["CUSTOM_GET_ISSUE_ACTIVITY"](GetIssueActivityInput(), EXECUTE_REQUEST, AUTH_CREDS)

    def test_get_activity_priority_change(self, tools, proxy) -> None:
        result = self._run(tools, proxy, _history(actor=None, fromPriority=0, toPriority=1))

        assert result["activities"] == [
            {
                "timestamp": "2024-01-01",
                "actor": "System",
                "change_type": "priority",
                "from": "none",
                "to": "urgent",
            }
        ]

    def test_get_activity_assignee_change(self, tools, proxy) -> None:
        result = self._run(tools, proxy, _history(toAssignee={"id": "u2", "name": "Bob"}))

        assert result["activities"] == [
            {
                "timestamp": "2024-01-01",
                "actor": "Alice",
                "change_type": "assignee",
                "from": None,
                "to": "Bob",
            }
        ]

    def test_get_activity_labels_added(self, tools, proxy) -> None:
        """Linear's schema types addedLabels as [IssueLabel!], a plain list, not a connection."""
        result = self._run(tools, proxy, _history(addedLabels=[{"id": "l1", "name": "Bug"}]))

        assert result["activities"] == [
            {
                "timestamp": "2024-01-01",
                "actor": "Alice",
                "change_type": "labels_added",
                "labels": ["Bug"],
            }
        ]


# =============================================================================
# CUSTOM_GET_ACTIVE_SPRINT
# =============================================================================


def _cycle(cycle_id: str, team: dict[str, Any], *issues: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": cycle_id,
        "name": "Sprint 5",
        "number": 5,
        "startsAt": "2024-01-01",
        "endsAt": "2024-01-15",
        "progress": 0.5,
        "team": team,
        "issues": {"nodes": list(issues)},
    }


class TestLinearGetActiveSprint:
    def test_get_active_sprint(self, tools, proxy) -> None:
        started = {
            "id": "i1",
            "identifier": "ENG-1",
            "title": "Task",
            "state": {"name": "In Progress", "type": "started"},
            "priority": 1,
            "assignee": {"name": "Alice"},
        }
        todo = {
            "id": "i2",
            "identifier": "ENG-2",
            "title": "Next",
            "state": {"name": "Todo", "type": "unstarted"},
            "priority": 0,
            "assignee": None,
        }
        triage = {
            **todo,
            "id": "i3",
            "identifier": "ENG-3",
            "state": {"name": "Triage", "type": "triage"},
        }
        _answers(proxy, {"cycles": {"nodes": [_cycle("c1", TEAM, started, todo, triage)]}})

        result = tools["CUSTOM_GET_ACTIVE_SPRINT"](
            GetActiveSprintInput(), EXECUTE_REQUEST, AUTH_CREDS
        )

        assert result == {
            "sprint_count": 1,
            "sprints": [
                {
                    "id": "c1",
                    "name": "Sprint 5",
                    "number": 5,
                    "team": "Eng",
                    "team_key": "ENG",
                    "starts_at": "2024-01-01",
                    "ends_at": "2024-01-15",
                    "progress": 50.0,
                    "total_issues": 3,
                    "issues_by_state": {"backlog": 0, "unstarted": 1, "started": 1, "completed": 0},
                    "in_progress": [
                        {
                            "identifier": "ENG-1",
                            "title": "Task",
                            "priority": "urgent",
                            "assignee": "Alice",
                        }
                    ],
                    "todo": [
                        {
                            "identifier": "ENG-2",
                            "title": "Next",
                            "priority": "none",
                            "assignee": None,
                        }
                    ],
                }
            ],
        }

    def test_get_active_sprint_filtered_by_team(self, tools, proxy) -> None:
        design = {"id": "t2", "key": "DES", "name": "Design"}
        _answers(proxy, {"cycles": {"nodes": [_cycle("c1", TEAM), _cycle("c2", design)]}})

        result = tools["CUSTOM_GET_ACTIVE_SPRINT"](
            GetActiveSprintInput(team_id="t1"), EXECUTE_REQUEST, AUTH_CREDS
        )

        assert [s["id"] for s in result["sprints"]] == ["c1"]


# =============================================================================
# CUSTOM_BULK_UPDATE_ISSUES
# =============================================================================


class TestLinearBulkUpdateIssues:
    def test_bulk_update_success(self, tools, proxy) -> None:
        _answers(
            proxy,
            {
                "issueBatchUpdate": {
                    "success": True,
                    "issues": [
                        {"id": "i1", "identifier": "ENG-1", "title": "A"},
                        {"id": "i2", "identifier": "ENG-2", "title": "B"},
                    ],
                }
            },
        )

        result = tools["CUSTOM_BULK_UPDATE_ISSUES"](
            BulkUpdateIssuesInput(issue_ids=["i1", "i2"], state_id="s1", assignee_id=""),
            EXECUTE_REQUEST,
            AUTH_CREDS,
        )

        assert result == {
            "updated_count": 2,
            "updated_issues": [
                {"id": "i1", "identifier": "ENG-1"},
                {"id": "i2", "identifier": "ENG-2"},
            ],
        }
        assert _bodies(proxy) == [
            {
                "query": MUTATION_UPDATE_ISSUES,
                "variables": {
                    "issueIds": ["i1", "i2"],
                    "input": {"stateId": "s1", "assigneeId": None},
                },
            }
        ]

    def test_bulk_update_no_ids(self, tools) -> None:
        with pytest.raises(ValueError, match="No issue IDs"):
            tools["CUSTOM_BULK_UPDATE_ISSUES"](
                BulkUpdateIssuesInput(issue_ids=[], state_id="s1"), EXECUTE_REQUEST, AUTH_CREDS
            )

    def test_bulk_update_no_updates(self, tools) -> None:
        with pytest.raises(ValueError, match="No updates specified"):
            tools["CUSTOM_BULK_UPDATE_ISSUES"](
                BulkUpdateIssuesInput(issue_ids=["i1"]), EXECUTE_REQUEST, AUTH_CREDS
            )

    def test_bulk_update_failure(self, tools, proxy) -> None:
        _answers(proxy, {"issueBatchUpdate": {"success": False, "issues": []}})

        with pytest.raises(RuntimeError, match="Batch update failed"):
            tools["CUSTOM_BULK_UPDATE_ISSUES"](
                BulkUpdateIssuesInput(issue_ids=["i1"], state_id="s1"), EXECUTE_REQUEST, AUTH_CREDS
            )


# =============================================================================
# CUSTOM_GET_NOTIFICATIONS
# =============================================================================

_NOTIFICATIONS = {
    "notifications": {
        "nodes": [
            {
                "id": "n1",
                "type": "issueAssignedToYou",
                "createdAt": "2024-01-01",
                "readAt": None,
                "issue": {"id": "i1", "identifier": "ENG-1", "title": "Bug"},
                "actor": {"id": "u1", "name": "Alice"},
            },
            {
                "id": "n2",
                "type": "projectUpdate",
                "createdAt": "2024-01-02",
                "readAt": "2024-01-02",
                "actor": None,
            },
        ]
    }
}


class TestLinearGetNotifications:
    def test_get_notifications_unread(self, tools, proxy) -> None:
        _answers(proxy, _NOTIFICATIONS)

        result = tools["CUSTOM_GET_NOTIFICATIONS"](
            GetNotificationsInput(include_read=False), EXECUTE_REQUEST, AUTH_CREDS
        )

        assert result == {
            "count": 1,
            "notifications": [
                {
                    "id": "n1",
                    "type": "issueAssignedToYou",
                    "created_at": "2024-01-01",
                    "read": False,
                    "issue": {"identifier": "ENG-1", "title": "Bug"},
                    "actor": "Alice",
                }
            ],
        }

    def test_get_notifications_include_read(self, tools, proxy) -> None:
        _answers(proxy, _NOTIFICATIONS)

        result = tools["CUSTOM_GET_NOTIFICATIONS"](
            GetNotificationsInput(include_read=True), EXECUTE_REQUEST, AUTH_CREDS
        )

        assert result["count"] == 2
        assert result["notifications"][1] == {
            "id": "n2",
            "type": "projectUpdate",
            "created_at": "2024-01-02",
            "read": True,
            "issue": None,
            "actor": None,
        }


# =============================================================================
# CUSTOM_GET_WORKSPACE_CONTEXT / CUSTOM_GATHER_CONTEXT
# =============================================================================


class TestLinearGetWorkspaceContext:
    def test_get_workspace_context(self, tools, proxy) -> None:
        local_today = datetime.now().date()
        yesterday = (local_today - timedelta(days=1)).isoformat()
        _answers(
            proxy,
            VIEWER,
            {
                "teams": {
                    "nodes": [
                        {**TEAM, "activeCycle": {"id": "c1", "name": "Sprint 5", "progress": 0.5}},
                        {"id": "t2", "key": "DES", "name": "Design", "activeCycle": None},
                    ]
                }
            },
            {
                "issues": {
                    "nodes": [
                        _summary("1", priority=1, dueDate=yesterday, slaBreachesAt="2024-01-01"),
                        _summary(
                            "2",
                            priority=1,
                            dueDate=yesterday,
                            state={"id": "s9", "name": "Done", "type": "completed"},
                        ),
                    ]
                }
            },
        )

        with patch(f"{LINEAR_MODULE}._user_local_today", return_value=local_today):
            result = tools["CUSTOM_GET_WORKSPACE_CONTEXT"](
                GetWorkspaceContextInput(), EXECUTE_REQUEST, AUTH_CREDS
            )

        assert result["user"] == {
            "id": "u1",
            "name": "Alice",
            "email": "a@b.com",
            "assigned_issue_count": 1,
        }
        assert result["teams"] == [
            {
                "id": "t1",
                "name": "Eng",
                "key": "ENG",
                "active_cycle": "Sprint 5",
                "cycle_progress": 50.0,
            },
            {
                "id": "t2",
                "name": "Design",
                "key": "DES",
                "active_cycle": None,
                "cycle_progress": None,
            },
        ]
        assert [[i["id"] for i in v] for v in result["urgent_items"].values()] == [
            ["1"],
            ["1"],
            ["1"],
        ]
        assert _bodies(proxy)[2]["variables"] == {
            "assigneeId": "u1",
            "includeCompleted": True,
            "first": 50,
        }


class TestLinearGatherContext:
    def test_gather_context(self, tools, proxy) -> None:
        local_today = datetime.now().date()
        yesterday = (local_today - timedelta(days=1)).isoformat()
        _answers(
            proxy,
            VIEWER,
            {"teams": {"nodes": [{**TEAM, "activeCycle": None}]}},
            {"issues": {"nodes": [_summary("1", dueDate=yesterday), _summary("2", priority=2)]}},
        )

        with patch(f"{LINEAR_MODULE}._user_local_today", return_value=local_today):
            result = tools["CUSTOM_GATHER_CONTEXT"](
                GatherContextInput(), EXECUTE_REQUEST, AUTH_CREDS
            )

        assert result["user"] == {"id": "u1", "name": "Alice", "email": "a@b.com"}
        assert result["teams"] == [{"id": "t1", "name": "Eng", "key": "ENG"}]
        assert [i["id"] for i in result["urgent_items"]["overdue"]] == ["1"]
        assert [i["id"] for i in result["urgent_items"]["high_priority"]] == ["2"]
