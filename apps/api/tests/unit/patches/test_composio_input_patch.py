"""CustomTool input coercion: LangChain-side lookalike instances must validate.

Composio builds the LangChain tool's args_schema by regenerating Pydantic
models from JSON schema (``json_schema_to_pydantic``), so the instances
LangChain hands to ``invoke_trusted`` are NOT instances of our real input
models. The real model's ``isinstance`` check then rejects them with a
``model_type`` error that shows a perfectly good instance being refused.
Coercing to plain data before the real validation fixes every nested-model
custom tool at the one boundary where the identities collide.
"""

from types import SimpleNamespace
from typing import Any

import pytest


def _lookalike_of(model: type) -> type:
    """A same-shaped model built the way Composio rebuilds ours from schema."""
    from composio.utils import shared

    rebuilt = shared.json_schema_to_pydantic_type(model.model_json_schema())
    assert isinstance(rebuilt, type), "expected a rebuilt model class"
    return rebuilt


def _event_dict() -> dict[str, Any]:
    return {"events": [{"summary": "s", "start_datetime": "2026-09-26T11:00:00"}]}


@pytest.mark.unit
class TestLookalikeCollision:
    def test_rebuilt_instance_fails_real_validation_without_coercion(self) -> None:
        """Pins the production failure: same shape, different class identity."""
        from pydantic import ValidationError

        from app.models.calendar_models import CreateEventInput

        lookalike = _lookalike_of(CreateEventInput)
        assert lookalike is not CreateEventInput
        instance = lookalike.model_validate(_event_dict())
        with pytest.raises(ValidationError, match="model_type"):
            CreateEventInput.model_validate(instance)

    def test_coerced_plain_data_passes_real_validation(self) -> None:
        from app.models.calendar_models import CreateEventInput
        from app.patches.composio_custom_tool_input_patch import to_plain_data

        lookalike = _lookalike_of(CreateEventInput)
        instance = lookalike.model_validate(_event_dict())
        coerced = to_plain_data({"request": instance, "flag": True})
        assert coerced == {
            "request": {
                "events": [
                    {
                        "summary": "s",
                        "start_datetime": "2026-09-26T11:00:00",
                        "duration_hours": 0,
                        "duration_minutes": 30,
                        "calendar_id": "primary",
                        "description": None,
                        "location": None,
                        "attendees": None,
                        "is_all_day": False,
                        "create_meeting_room": False,
                    }
                ],
                "confirm_immediately": False,
            },
            "flag": True,
        }
        validated = CreateEventInput.model_validate(coerced["request"])
        assert validated.events[0].summary == "s"

    def test_plain_data_passes_through_untouched(self) -> None:
        from app.patches.composio_custom_tool_input_patch import to_plain_data

        plain = {"a": [1, {"b": None}], "c": "x"}
        assert to_plain_data(plain) == plain


@pytest.mark.unit
class TestInvokeTrustedWrapper:
    def _stub_tool(self) -> Any:
        from app.models.calendar_models import CreateEventInput

        return SimpleNamespace(
            request_model=CreateEventInput,
            toolkit=None,
            f=lambda request: {"summary": request.events[0].summary},
        )

    def test_lookalike_kwargs_execute_end_to_end(self) -> None:
        """The wrapper coerces before the real validation: a rebuilt instance
        nested in kwargs runs the function instead of raising model_type."""
        from pydantic import ValidationError

        from app.models.calendar_models import CreateEventInput, SingleEventInput
        from app.patches import composio_custom_tool_input_patch as patch_mod

        lookalike_item = _lookalike_of(SingleEventInput)
        assert lookalike_item is not SingleEventInput
        nested = lookalike_item.model_validate(
            {"summary": "s", "start_datetime": "2026-09-26T11:00:00"}
        )
        kwargs = {"events": [nested], "confirm_immediately": True}

        def fake_original(self: Any, user_id: str, request_kwargs: Any) -> Any:
            validated = CreateEventInput.model_validate(request_kwargs)
            return self.f(request=validated)

        # Without coercion this is the production failure, verbatim.
        with pytest.raises(ValidationError, match="model_type"):
            fake_original(self._stub_tool(), "u1", kwargs)

        previous, patch_mod._original_invoke_trusted = (
            patch_mod._original_invoke_trusted,
            fake_original,
        )
        try:
            bound = patch_mod._coercing_invoke_trusted.__get__(
                self._stub_tool(), object
            )
            result = bound("u1", kwargs)
        finally:
            patch_mod._original_invoke_trusted = previous
        assert result == {"summary": "s"}

    def test_apply_wraps_once_and_marks(self) -> None:
        from composio.core.models.custom_tools import CustomTool

        from app.patches import composio_custom_tool_input_patch as patch_mod

        patch_mod.apply()
        first = CustomTool.invoke_trusted
        assert getattr(first, "__gaia_coercing__", False) is True
        patch_mod.apply()
        assert CustomTool.invoke_trusted is first
