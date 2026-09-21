"""Tests for app.services.browser.captions — caption_from_action_list, describe_action, et al.

Covers every action name branch in describe_action with exact expected strings
from source, plus the two caption builders and _dedupe_join.
"""

from __future__ import annotations

import pytest

from app.constants.browser import BrowserHandoffAction
from app.schemas.browser import BrowserAction
from app.services.browser.captions import (
    _dedupe_join,
    _shorten,
    caption_from_action_list,
    describe_action,
)

# ---------------------------------------------------------------------------
# describe_action — every named branch, exact strings from source
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDescribeAction:
    def test_navigate_without_www(self):
        assert (
            describe_action("navigate", {"url": "https://example.com/search"})
            == "Opening example.com"
        )

    def test_navigate_subdomain(self):
        assert (
            describe_action("navigate", {"url": "https://sub.example.com/"})
            == "Opening sub.example.com"
        )

    def test_navigate_with_port_and_path(self):
        assert (
            describe_action("navigate", {"url": "https://sub.example.com:8080/foo?x=1"})
            == "Opening sub.example.com"
        )

    def test_navigate_invalid_url_no_hostname(self):
        assert describe_action("navigate", {"url": "not-a-url"}) == "Opening the page"

    @pytest.mark.parametrize("name", ["search", "search_page"])
    def test_search_with_text_fallback(self, name):
        assert describe_action(name, {"text": "fallback query"}) == 'Searching "fallback query"'

    @pytest.mark.parametrize("name", ["search", "search_page"])
    def test_search_query_precedence_over_text(self, name):
        assert describe_action(name, {"query": "q", "text": "t"}) == 'Searching "q"'

    def test_select_dropdown_without_text_names_the_field(self):
        assert describe_action("select_dropdown", {}, target="Country") == 'Choosing in "Country"'

    def test_input_without_text_names_the_field(self):
        assert describe_action("input", {}, target="Full name") == 'Typing into "Full name"'

    @pytest.mark.parametrize(
        "params",
        [
            {"coordinate_x": 412},
            {"coordinate_y": 680},
            {"coordinate_x": 412, "coordinate_y": None},
            {"coordinate_x": None, "coordinate_y": 680},
        ],
        ids=["x-only", "y-only", "y-none", "x-none"],
    )
    def test_click_needs_both_coordinates_to_name_a_point(self, params):
        """Half a coordinate pair names no point on the page -- the caption has to fall back to the bare verb rather than print "Clicking at 412, None"."""
        assert describe_action("click", params) == "Clicking"

    def test_a_failed_done_says_what_happened_not_BLOCKED(self):
        """Jev ends a stuck run with success=False; the card must not show a raw label."""
        assert (
            describe_action("done", {"text": "", "success": False})
            == "Could not find a way forward on this page"
        )

    def test_done_without_an_explicit_success_reads_as_finished(self):
        # Browser-Use's DoneAction defaults success to True.
        assert describe_action("done", {}) == "Finished"

    def test_done_with_an_empty_text_reads_as_finished(self):
        assert describe_action("done", {"text": "  ", "success": True}) == "Finished"

    def test_a_finished_done_captions_with_its_own_summary(self):
        """The terminal step photo should say what was found, not the DONE verb."""
        assert (
            describe_action("done", {"text": "Top HN post: 227 points.", "success": True})
            == "Top HN post: 227 points."
        )

    def test_done_summary_collapses_internal_whitespace(self):
        assert (
            describe_action("done", {"text": "line one\nline two", "success": True})
            == "line one line two"
        )

    def test_done_summary_is_never_truncated(self):
        text = "x" * 300
        result = describe_action("done", {"text": text, "success": True})
        assert result == text

    def test_fallback_replaces_underscores(self):
        assert describe_action("my_custom_action", {}) == "my custom action"


# ---------------------------------------------------------------------------
# _dedupe_join
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDedupeJoin:
    def test_consecutive_duplicates(self):
        assert _dedupe_join(["Clicking", "Clicking"]) == "Clicking"

    def test_non_consecutive_duplicates_still_deduped(self):
        assert _dedupe_join(["Clicking", "Scrolling", "Clicking"]) == "Clicking, Scrolling"

    def test_filters_empty_strings(self):
        assert _dedupe_join(["", "Clicking", ""]) == "Clicking"


# ---------------------------------------------------------------------------
# caption_from_action_list — a step snapshot's structured actions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCaptionFromActionList:
    def test_navigate_names_the_host(self):
        """The whole point of structured actions: params survive, so the caption says which site was opened instead of a generic phrase."""
        actions = [BrowserAction(name="navigate", inputs={"url": "https://www.github.com/x"})]
        assert caption_from_action_list(actions) == "Opening github.com"

    def test_click_names_the_element_it_hit(self):
        """Ground the element name in the page, not in the model's claim about its intent."""
        actions = [BrowserAction(name="click", inputs={"index": 9}, target="Add to cart")]
        assert caption_from_action_list(actions) == 'Clicking "Add to cart"'

    def test_click_without_a_target_names_the_coordinates(self):
        actions = [BrowserAction(name="click", inputs={"coordinate_x": 412, "coordinate_y": 680})]
        assert caption_from_action_list(actions) == "Clicking at 412, 680"

    def test_typing_names_the_field(self):
        actions = [BrowserAction(name="input", inputs={"text": "Aryan"}, target="Full name")]
        assert caption_from_action_list(actions) == 'Typing "Aryan" into "Full name"'

    def test_long_target_is_kept_whole(self):
        actions = [BrowserAction(name="click", inputs={}, target="x" * 80)]
        caption = caption_from_action_list(actions)
        assert caption == 'Clicking "' + "x" * 80 + '"'


# ---------------------------------------------------------------------------
# _shorten — whitespace collapsing only; captions never clip
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestShorten:
    def test_collapses_internal_whitespace(self):
        assert _shorten("a  \n b\tc") == "a b c"

    def test_long_text_is_kept_whole(self):
        """No length cap: a long caption survives untouched."""
        text = "y" * 300
        assert _shorten(text) == text

    def test_internal_whitespace_collapses_without_clipping(self):
        text = "a" * 178 + " " + "b" * 20
        assert _shorten(text) == text


@pytest.mark.unit
def test_every_handoff_action_has_a_caption() -> None:
    """A handoff with no caption entry falls through to the underscore fallback ("request human takeover"), so the caption table has to cover the enum."""
    for member in BrowserHandoffAction:
        actions = [BrowserAction(name=member.value, inputs={}, target=None)]
        assert caption_from_action_list(actions) != member.value.replace("_", " ")
