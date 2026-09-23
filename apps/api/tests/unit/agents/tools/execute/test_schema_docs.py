"""render_tool_doc — the discovery contract: compact, budgeted, never invented."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

from pydantic import BaseModel, Field, JsonValue
import pytest

from app.agents.tools.execute import schema_docs
from app.agents.tools.execute.schema_docs import (
    render_compact_type,
    render_compact_type_budgeted,
    render_tool_doc,
)
from app.constants.execute import SCHEMA_DOC_MAX_CHARS
from app.utils.general_utils import clip_text


class _Args(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = 25


def _tool(
    name: str = "GMAIL_FETCH_EMAILS",
    description: str = "Fetch emails.",
    metadata: dict | None = None,
) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.args_schema = _Args
    tool.metadata = metadata
    return tool


def _deep_response_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "properties": {
                    f"field_{i}": {
                        "type": "object",
                        "properties": {"leaf": {"type": "string", "description": "y" * 200}},
                    }
                    for i in range(50)
                },
            }
        },
    }


@pytest.mark.unit
class TestRenderToolDoc:
    def test_doc_carries_name_description_and_args_schema(self) -> None:
        doc = render_tool_doc(_tool())
        assert "## GMAIL_FETCH_EMAILS" in doc
        assert "Fetch emails." in doc
        assert '"query"' in doc and '"max_results"' in doc
        assert 'tool_name="GMAIL_FETCH_EMAILS"' in doc

    def test_generator_noise_and_internal_params_are_stripped(self) -> None:
        class _WithInternal(BaseModel):
            query: str

        tool = _tool()
        schema = _WithInternal.model_json_schema()
        schema["properties"]["__runnable_config__"] = {"type": "string"}
        tool.args_schema = schema
        doc = render_tool_doc(tool)
        assert "__runnable_config__" not in doc
        assert '"title"' not in doc

    def test_returns_never_render_in_discovery_docs(self) -> None:
        # Shapes are explored on demand (get_tool_schema / gaia.schema), never
        # paid for in every discovery doc - even when the provider supplies one.
        with_schema = _tool(metadata={"output_parameters": _deep_response_schema()})
        assert "Returns" not in render_tool_doc(with_schema)
        assert "Returns" not in render_tool_doc(_tool(metadata=None))

    def test_huge_args_schema_never_starves_the_rest_of_the_doc(self) -> None:
        # Real case: GOOGLECALENDAR_EVENTS_LIST's args schema alone exceeded the
        # doc cap, clipping away Returns and the usage line mid-JSON.
        deep_args = {
            "type": "object",
            "properties": {
                f"arg_{i}": {
                    "type": "object",
                    "properties": {"nested": {"type": "string", "description": "z" * 200}},
                }
                for i in range(60)
            },
        }
        tool = _tool()
        tool.args_schema = deep_args
        doc = render_tool_doc(tool)
        assert 'tool_name="GMAIL_FETCH_EMAILS"' in doc  # usage line survives
        assert '"..."' in doc  # args pruned visibly, not clipped mid-JSON

    def test_compact_type_notation_core_shapes(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "count": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "status": {"type": "string", "enum": ["open", "closed"]},
                "parent": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "meta": {"type": "object"},
            },
            "required": ["id", "count"],
        }
        rendered = render_compact_type(schema)
        assert rendered == (
            '{id:str, count:int, tags?:str[], status?:"open"|"closed", parent?:null|str, meta?:obj}'
        )

    def test_compact_type_renders_a_map_as_an_index_signature(self) -> None:
        """Observed shapes store data-keyed maps as additionalProperties; the notation must show the value shape, not degrade the map to bare obj."""
        schema = {
            "type": "object",
            "properties": {
                "per_user": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {"n": {"type": "integer"}},
                        "required": ["n"],
                    },
                }
            },
            "required": ["per_user"],
        }
        assert render_compact_type(schema) == "{per_user:{[key]:{n:int}}}"

    def test_compact_type_union_array_items_are_grouped(self) -> None:
        schema = {
            "type": "array",
            "items": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
        }
        # Without grouping, {a}|{b}[] misreads as a union with an array arm.
        assert render_compact_type(schema) == "(int|str)[]"

    def test_budgeted_compact_type_depth_collapses_when_oversized(self) -> None:
        rendered = render_compact_type_budgeted(_deep_response_schema(), 800)
        assert len(rendered.splitlines()[0]) <= 800
        assert "obj" in rendered  # collapsed depth is visible as bare obj
        assert "omitted for size" in rendered

    def test_huge_schema_is_capped(self) -> None:
        huge = {
            "type": "object",
            "properties": {
                f"field_{i}": {"type": "string", "description": "x" * 80} for i in range(400)
            },
        }
        tool = _tool(metadata={"output_parameters": huge})
        tool.args_schema = huge
        doc = render_tool_doc(tool)
        assert len(doc) <= SCHEMA_DOC_MAX_CHARS + 50  # clip marker allowance


def _nested(depth: int) -> dict[str, JsonValue]:
    """Build an object nested depth levels deep, named a, b, c, ... down to a string leaf."""
    node: dict[str, JsonValue] = {"type": "string"}
    for name in reversed("abcdefgh"[:depth]):
        node = {"type": "object", "properties": {name: node}}
    return node


_SHAPES: dict[str, JsonValue] = {
    "type": "object",
    "required": ["a"],
    "properties": {
        "a": {
            "type": "array",
            "items": {"type": "object", "properties": {"b": {"type": "string"}}},
        },
        "m": {"type": "object", "additionalProperties": {"type": "integer"}},
    },
}


@pytest.mark.unit
class TestRenderToolDocLayout:
    def test_a_tool_with_no_description_or_schema_still_documents_its_call(self) -> None:
        tool = _tool(name="PING", description="")
        tool.args_schema = None
        assert render_tool_doc(tool).split("\n") == [
            "## PING",
            "Args schema for execute(tool_name=..., data={...}):",
            '{"type":"object","properties":{}}',
            'Run it with: execute(task_description="...", tool_name="PING", data={...})',
        ]

    def test_internal_params_leave_required_and_titles_leave_nested_variants(self) -> None:
        tool = _tool()
        tool.args_schema = {
            "type": "object",
            "properties": {
                "q": {"anyOf": [{"type": "string", "title": "Q"}]},
                "__runnable_config__": {"type": "object"},
            },
            "required": ["q", "__runnable_config__"],
        }
        assert render_tool_doc(tool).split("\n")[3] == (
            '{"type":"object","properties":{"q":{"anyOf":[{"type":"string"}]}},"required":["q"]}'
        )

    def test_a_schema_carrying_a_python_value_still_renders(self) -> None:
        """Python-built dict schemas can carry non-JSON defaults and enum members; a doc must render them, not crash retrieval."""
        when = datetime(2026, 1, 1, tzinfo=UTC)
        tool = _tool()
        tool.args_schema = {
            "type": "object",
            "properties": {"at": {"type": "string", "default": when}},
        }
        assert '"default":"2026-01-01 00:00:00+00:00"' in render_tool_doc(tool)
        assert render_compact_type({"enum": [when]}) == '"2026-01-01 00:00:00+00:00"'


@pytest.mark.unit
class TestCompactTypeEdges:
    @pytest.mark.parametrize(
        ("schema", "rendered"),
        [
            ({"type": "object", "properties": {"x": True}}, "{x?:any}"),
            ({"type": "date"}, "date"),
            ({}, "any"),
            ({"oneOf": [{"type": "string"}, {"type": "integer"}]}, "int|str"),
            ({"anyOf": [], "type": "string"}, "str"),
            ({"anyOf": {"type": "string"}, "type": "integer"}, "int"),
            ({"type": ["string", "null"]}, "null|str"),
            ({"properties": {"a": {"type": "string"}}}, "{a?:str}"),
            ({"type": "string", "enum": ["only"]}, '"only"'),
            ({"type": "string", "enum": []}, "str"),
            ({"type": "integer", "enum": [1, 2, 3, 4, 5, 6]}, "1|2|3|4|5|6"),
            ({"type": "integer", "enum": [1, 2, 3, 4, 5, 6, 7]}, "int"),
        ],
        ids=[
            "boolean_subschema",
            "unknown_type",
            "no_type",
            "one_of",
            "empty_any_of",
            "malformed_any_of",
            "type_list",
            "untyped_object",
            "single_member_enum",
            "empty_enum",
            "enum_at_the_cap",
            "enum_over_the_cap",
        ],
    )
    def test_shape(self, schema: dict[str, JsonValue], rendered: str) -> None:
        assert render_compact_type(schema) == rendered


@pytest.mark.unit
class TestBudgetedCompactType:
    def test_a_type_that_exactly_fits_is_returned_whole(self) -> None:
        assert render_compact_type_budgeted(_nested(1), len("{a?:str}")) == "{a?:str}"

    def test_a_depth_collapse_that_exactly_fits_is_taken(self) -> None:
        assert render_compact_type_budgeted(_nested(4), len("{a?:{b?:{c?:obj}}}")) == (
            "{a?:{b?:{c?:obj}}}\n(deeper fields omitted for size; the real data has them)"
        )

    def test_nothing_fitting_clips_the_full_rendering(self) -> None:
        full = render_compact_type(_nested(4))
        assert render_compact_type_budgeted(_nested(4), 3) == clip_text(full, 3)


@pytest.mark.unit
class TestPruneToLevels:
    @pytest.mark.parametrize(
        ("levels", "expected"),
        [
            (0, {"type": "object", "required": ["a"], "properties": "..."}),
            (
                1,
                {
                    "type": "object",
                    "required": ["a"],
                    "properties": {
                        "a": {"type": "array", "items": "..."},
                        "m": {"type": "object", "additionalProperties": "..."},
                    },
                },
            ),
            (
                2,
                {
                    "type": "object",
                    "required": ["a"],
                    "properties": {
                        "a": {"type": "array", "items": {"type": "object", "properties": "..."}},
                        "m": {"type": "object", "additionalProperties": {"type": "integer"}},
                    },
                },
            ),
            (3, _SHAPES),
        ],
    )
    def test_each_level_keeps_exactly_that_much_nesting(
        self, levels: int, expected: dict[str, JsonValue]
    ) -> None:
        assert schema_docs._prune_to_levels(_SHAPES, levels) == expected

    def test_union_variants_are_pruned_at_the_unions_own_level(self) -> None:
        union = {"anyOf": [{"type": "object", "properties": {"x": {"type": "string"}}}]}
        assert schema_docs._prune_to_levels(union, 0) == {
            "anyOf": [{"type": "object", "properties": "..."}]
        }


@pytest.mark.unit
class TestBudgetedSchema:
    NOTE = schema_docs._SCHEMA_TRUNCATION_NOTE

    def test_a_schema_that_exactly_fits_is_returned_whole(self) -> None:
        full = '{"type":"string"}'
        assert schema_docs._render_budgeted_schema({"type": "string"}, len(full)) == full

    def test_a_depth_prune_that_exactly_fits_is_taken_with_the_note(self) -> None:
        pruned = (
            '{"type":"object","properties":{"a":{"type":"object","properties":{"b":'
            '{"type":"object","properties":{"c":{"type":"object","properties":"..."}}}}}}}'
        )
        rendered = schema_docs._render_budgeted_schema(_nested(4), len(pruned))
        assert rendered == f"{pruned}\n{self.NOTE}"

    @pytest.mark.parametrize(
        ("schema_type", "floor"),
        [
            ("object", '{"type":"object","fields":["a","z"]}'),
            (["object", "null"], '{"type":["object","null"],"fields":["a","z"]}'),
            (None, '{"type":"object","fields":["a","z"]}'),
        ],
        ids=["object", "nullable_object", "untyped"],
    )
    def test_when_no_depth_fits_only_the_field_names_remain(
        self, schema_type: JsonValue, floor: str
    ) -> None:
        schema: dict[str, JsonValue] = {
            "properties": {
                "z": {"type": "string", "description": "x" * 500},
                "a": {"type": "string", "description": "y" * 500},
            }
        }
        if schema_type is not None:
            schema["type"] = schema_type
        assert schema_docs._render_budgeted_schema(schema, 60) == f"{floor}\n{self.NOTE}"
