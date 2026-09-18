"""The element table Jev decides over, built from Browser-Use's selector map."""

from __future__ import annotations

import pytest

from app.constants.browser import JEV_MAX_TARGETS_PER_OPERATION, JevOperation
from app.services.browser.jev.observation import observe

from .conftest import (
    FakeAXNode,
    FakeAXProperty,
    FakeNode,
    FakeRect,
    make_page_info,
    make_state,
)

pytestmark = pytest.mark.unit


def test_elements_are_renumbered_in_order_and_keep_their_browser_index(flights_state) -> None:
    observation = observe(flights_state)

    assert [e.index for e in observation.elements] == [1, 2, 3, 4, 5]
    assert [e.browser_index for e in observation.elements] == [17, 23, 31, 40, 41]
    assert observation.url == "https://x"
    assert observation.title == "X"


def test_a_text_field_offers_click_and_type_and_carries_its_value(flights_state) -> None:
    origin, destination = observe(flights_state).elements[:2]

    assert origin.label == "Zurich"  # the current value is what Browser-Use shows too
    assert origin.value == "Zurich"
    assert origin.operations == (JevOperation.CLICK, JevOperation.TYPE_TEXT)
    assert destination.label == "Where to?"
    assert destination.value == ""
    assert destination.role == "combobox"


def test_a_native_select_offers_select_with_its_enabled_options(flights_state) -> None:
    cabin = observe(flights_state).elements[2]

    assert cabin.operations == (JevOperation.CLICK, JevOperation.SELECT)
    assert [(o.target, o.label, o.value) for o in cabin.options] == [
        ("3:1", "Economy", "economy"),
        ("3:2", "Business", "business"),
    ]
    assert cabin.value == "Economy"
    assert cabin.state_entry()["options"] == [
        {"index": "3:1", "label": "Economy", "value": "economy"},
        {"index": "3:2", "label": "Business", "value": "business"},
    ]


def test_a_button_is_click_only_and_named_by_its_accessible_name(flights_state) -> None:
    search = observe(flights_state).elements[3]

    assert search.operations == (JevOperation.CLICK,)
    assert search.label == "Search"
    assert search.role == "button"
    assert "value" not in search.state_entry()


def test_a_checkbox_reports_its_checked_state(flights_state) -> None:
    nonstop = observe(flights_state).elements[4]

    assert nonstop.operations == (JevOperation.CLICK,)
    assert nonstop.checked is True
    assert nonstop.criterion() == {
        "element": "[5] Nonstop only",
        "current_value": "",
        "role": "checkbox",
        "checked": True,
    }


def test_targets_group_elements_per_operation(flights_state) -> None:
    observation = observe(flights_state)

    assert set(observation.targets(JevOperation.CLICK)) == {"1", "2", "3", "4", "5"}
    assert set(observation.targets(JevOperation.TYPE_TEXT)) == {"1", "2"}
    assert set(observation.targets(JevOperation.SELECT)) == {"3:1", "3:2"}
    element, option = observation.targets(JevOperation.SELECT)["3:2"]
    assert (element.index, option.label) == (3, "Business")


@pytest.mark.parametrize(
    "attributes",
    [
        {"type": "submit", "value": "Go"},
        {"type": "checkbox", "aria-label": "x"},
        {"readonly": "", "placeholder": "Date"},
        {"aria-readonly": "true", "placeholder": "Date"},
    ],
)
def test_inputs_that_cannot_take_text_do_not_offer_type_text(attributes) -> None:
    (element,) = observe(make_state({1: FakeNode("INPUT", attributes)})).elements

    assert JevOperation.TYPE_TEXT not in element.operations


def test_contenteditable_and_textarea_offer_type_text() -> None:
    state = make_state(
        {
            1: FakeNode("DIV", {"contenteditable": "true", "aria-label": "Message"}),
            2: FakeNode("TEXTAREA", {"placeholder": "Notes"}),
        }
    )

    assert all(JevOperation.TYPE_TEXT in e.operations for e in observe(state).elements)


def test_an_unlabelled_non_field_element_is_dropped() -> None:
    state = make_state({1: FakeNode("DIV"), 2: FakeNode("INPUT", {"placeholder": "q"})})

    assert [e.browser_index for e in observe(state).elements] == [2]


def test_a_disabled_ax_field_does_not_offer_type_text() -> None:
    node = FakeNode(
        "INPUT",
        {"placeholder": "q"},
        ax_node=FakeAXNode(role="textbox", properties=[FakeAXProperty("disabled", True)]),
    )

    (element,) = observe(make_state({1: node})).elements
    assert element.operations == (JevOperation.CLICK,)


def test_page_text_is_browser_uses_own_rendering_capped(monkeypatch, flights_state) -> None:
    monkeypatch.setattr("app.services.browser.jev.observation.JEV_PAGE_TEXT_MAX_CHARS", 12)

    assert observe(flights_state).text == "[17]<input>\n"[:12]


def test_fingerprint_changes_with_a_value_but_not_with_an_unrelated_title(flights_state) -> None:
    before = observe(flights_state).fingerprint
    flights_state.title = "Other"
    assert observe(flights_state).fingerprint == before

    flights_state.dom_state.selector_map[23].attributes["value"] = "London"
    assert observe(flights_state).fingerprint != before


def test_a_node_that_raises_loses_only_its_row_and_is_logged(flights_state, monkeypatch) -> None:
    from unittest.mock import MagicMock

    from app.constants.log_tags import LogTag
    from app.services.browser.jev import observation as observation_mod

    logger = MagicMock()
    monkeypatch.setattr(observation_mod, "log", logger)

    class Broken:
        node_name = "BUTTON"
        attributes = None

        def get_meaningful_text_for_llm(self):
            raise RuntimeError("shape")

    flights_state.dom_state.selector_map[99] = Broken()

    assert len(observe(flights_state).elements) == 5
    logger.warning.assert_called_once_with(
        f"{LogTag.BROWSER} Jev could not read a DOM node; row skipped",
        error_type="RuntimeError",
        browser_index=99,
    )


def test_a_serializer_failure_yields_empty_page_text_and_is_logged(
    flights_state, monkeypatch
) -> None:
    from unittest.mock import MagicMock

    from app.constants.log_tags import LogTag
    from app.services.browser.jev import observation as observation_mod

    logger = MagicMock()
    monkeypatch.setattr(observation_mod, "log", logger)

    def boom():
        raise RuntimeError("serializer")

    flights_state.dom_state.llm_representation = boom

    assert observe(flights_state).text == ""
    logger.warning.assert_called_once_with(
        f"{LogTag.BROWSER} Jev page text unavailable for this step", error_type="RuntimeError"
    )


def test_live_values_override_attributes_for_text_selects_and_checkboxes(flights_state) -> None:
    from app.services.browser.jev.live_values import LiveValues

    for index, node in flights_state.dom_state.selector_map.items():
        node.backend_node_id = index
    business = flights_state.dom_state.selector_map[31].children_nodes[1]
    business.backend_node_id = 310
    live = LiveValues(
        values={23: "London", 17: ""}, checked=frozenset({40}), selected_options=frozenset({310})
    )

    elements = {e.label: e for e in observe(flights_state, live).elements}

    assert elements["Where to?"].value == "London"
    assert elements["Zurich"].value == ""  # cleared live, whatever the attribute says
    assert elements["Cabin class"].value == "Business"
    assert elements["Search"].checked is True


# ---------------------------------------------------------------------------
# The gateway refuses a question with more than 255 choices
# ---------------------------------------------------------------------------


def _big_page(count: int = 300, *, in_viewport_from: int = 0):
    """``count`` buttons; those from ``in_viewport_from`` on sit inside the viewport."""
    return make_state(
        {
            i: FakeNode(
                "BUTTON",
                text=f"Button {i}",
                ax_node=FakeAXNode(role="button", name=f"Button {i}"),
                absolute_position=FakeRect(y=10.0 if i >= in_viewport_from else 5000.0),
            )
            for i in range(count)
        },
        page_info=make_page_info(),
    )


def test_a_huge_page_is_capped_to_the_gateways_choice_limit() -> None:
    observation = observe(_big_page())

    assert len(observation.targets(JevOperation.CLICK)) == JEV_MAX_TARGETS_PER_OPERATION


def test_an_in_viewport_element_outranks_an_earlier_one_below_the_fold() -> None:
    # 241 in-viewport elements (document indices 60..300) crowd out every element
    # above the fold, so the cap is decided by the viewport and not by position.
    observation = observe(_big_page(in_viewport_from=59))

    targets = observation.targets(JevOperation.CLICK)

    assert "299" in targets  # in the viewport, near the end of the document
    assert "10" not in targets  # below the fold, near the start
    assert len(targets) == JEV_MAX_TARGETS_PER_OPERATION


def test_the_options_of_one_select_are_capped_in_document_order() -> None:
    state = make_state(
        {
            1: FakeNode(
                "SELECT",
                ax_node=FakeAXNode(role="combobox", name="Year"),
                children_nodes=[
                    FakeNode("OPTION", {"value": str(i)}, text=str(i)) for i in range(300)
                ],
            )
        },
        page_info=make_page_info(),
    )

    targets = observe(state).targets(JevOperation.SELECT)

    assert len(targets) == JEV_MAX_TARGETS_PER_OPERATION
    assert list(targets)[:2] == ["1:1", "1:2"]
    assert f"1:{JEV_MAX_TARGETS_PER_OPERATION}" in targets
    assert f"1:{JEV_MAX_TARGETS_PER_OPERATION + 1}" not in targets


def test_a_cap_that_cuts_choices_is_never_silent(monkeypatch) -> None:
    from unittest.mock import MagicMock

    from app.constants.log_tags import LogTag
    from app.services.browser.jev import observation as observation_mod

    logger = MagicMock()
    monkeypatch.setattr(observation_mod, "log", logger)

    observe(_big_page()).targets(JevOperation.CLICK)

    logger.warning.assert_called_once_with(
        f"{LogTag.BROWSER} Jev targets capped",
        browser={"operation": "CLICK", "dropped": 60, "url": "https://x"},
    )


def test_a_cap_that_cuts_nothing_logs_nothing(monkeypatch, flights_state) -> None:
    from unittest.mock import MagicMock

    from app.services.browser.jev import observation as observation_mod

    logger = MagicMock()
    monkeypatch.setattr(observation_mod, "log", logger)

    observe(flights_state).targets(JevOperation.CLICK)

    logger.warning.assert_not_called()
