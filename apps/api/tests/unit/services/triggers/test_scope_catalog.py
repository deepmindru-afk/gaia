"""The registration-scope catalog: which fields a trigger accepts and needs.

The required set is what stops a per-resource watch registering against nothing,
so it is checked field by field, and a drift guard proves every required name is
a real field on its config class.
"""

import pytest

from app.services.triggers.scope_catalog import (
    REQUIRED_SCOPE,
    TRIGGER_CONFIG_CLASSES,
    ScopeField,
    scope_fields_for,
)

pytestmark = pytest.mark.unit


class TestScopeFieldsFor:
    def test_a_per_resource_trigger_returns_its_required_field(self) -> None:
        assert scope_fields_for("github_pr_event") == (
            ScopeField(
                name="repos",
                type="list of text",
                description="List of repositories in owner/repo format",
                required=True,
            ),
        )

    def test_an_account_level_trigger_has_no_scope_fields(self) -> None:
        assert scope_fields_for("gmail_new_message") == ()

    def test_an_unknown_trigger_has_no_scope_fields(self) -> None:
        assert scope_fields_for("not_a_trigger") == ()

    def test_scalar_and_boolean_and_list_fields_get_distinct_type_labels(self) -> None:
        # calendar carries all three shapes, so it exercises every _scope_type_label arm.
        by_name = {f.name: f for f in scope_fields_for("calendar_event_starting_soon")}

        assert by_name["calendar_ids"].type == "list of text"
        assert by_name["minutes_before_start"].type == "integer"
        assert by_name["include_all_day"].type == "true/false"

    def test_a_string_field_is_labelled_text(self) -> None:
        (team_id,) = scope_fields_for("linear_issue_created")

        assert team_id.type == "text"

    def test_only_the_required_fields_are_marked_required(self) -> None:
        # A trigger with an optional field alongside a required one must not mark
        # the optional one (sheets: spreadsheet_ids required, sheet_names not).
        by_name = {f.name: f for f in scope_fields_for("google_sheets_new_row")}

        assert by_name["spreadsheet_ids"].required is True
        assert by_name["sheet_names"].required is False


class TestRequiredScopeIntegrity:
    def test_every_required_field_exists_on_its_config_class(self) -> None:
        # A required name that is not a real field would silently never be enforced.
        for trigger_name, required in REQUIRED_SCOPE.items():
            config_class = TRIGGER_CONFIG_CLASSES[trigger_name]
            fields = set(config_class.model_fields) - {"trigger_name"}
            assert required <= fields, f"{trigger_name}: {required - fields} not on the config"

    def test_the_config_class_map_covers_a_known_trigger(self) -> None:
        assert "github_pr_event" in TRIGGER_CONFIG_CLASSES
        assert TRIGGER_CONFIG_CLASSES["github_pr_event"].__name__ == "GitHubPrEventConfig"
