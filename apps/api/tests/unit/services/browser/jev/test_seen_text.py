"""The page memory the closing summary answers list questions from."""

from __future__ import annotations

import pytest

from app.constants.browser import JEV_SEEN_TEXT_MAX_CHARS
from app.services.browser.jev.seen_text import SeenText

pytestmark = pytest.mark.unit


def test_lines_from_every_screen_of_one_page_are_kept_in_reading_order() -> None:
    memory = SeenText()

    memory.record("https://books.test/travel", "It's Only the Himalayas 45.17\nFull Moon 49.43")
    memory.record("https://books.test/travel", "The Road to Little Dribbling 23.21")

    assert memory.text.splitlines() == [
        "It's Only the Himalayas 45.17",
        "Full Moon 49.43",
        "The Road to Little Dribbling 23.21",
    ]


def test_an_overlapping_scroll_does_not_repeat_the_lines_it_shares() -> None:
    memory = SeenText()

    memory.record("https://books.test/travel", "Full Moon 49.43\nSee America 48.87")
    memory.record("https://books.test/travel", "See America 48.87\nUnder the Tuscan Sun 37.33")

    assert memory.text.splitlines() == [
        "Full Moon 49.43",
        "See America 48.87",
        "Under the Tuscan Sun 37.33",
    ]


def test_a_different_page_starts_the_memory_over() -> None:
    memory = SeenText()
    memory.record("https://books.test/travel", "Full Moon 49.43")

    memory.record("https://books.test/mystery", "Sharp Objects 47.82")

    assert memory.text == "Sharp Objects 47.82"


def test_scrolling_to_an_anchor_on_the_same_page_keeps_what_was_seen() -> None:
    memory = SeenText()
    memory.record("https://books.test/travel", "Full Moon 49.43")

    memory.record("https://books.test/travel#bottom", "See America 48.87")

    assert memory.text.splitlines() == ["Full Moon 49.43", "See America 48.87"]


def test_the_memory_stops_growing_at_the_cap(monkeypatch) -> None:
    monkeypatch.setattr("app.services.browser.jev.seen_text.JEV_SEEN_TEXT_MAX_CHARS", 20)
    memory = SeenText()

    memory.record("https://books.test/travel", "a" * 15)
    memory.record("https://books.test/travel", "b" * 15)

    assert memory.text == "a" * 15


def test_a_page_counts_as_read_to_the_end_only_once_its_bottom_was_on_screen() -> None:
    memory = SeenText()

    memory.record("https://news.test/", "1. First story", "News")
    assert memory.pages == [{"url": "https://news.test/", "title": "News", "read": "top part only"}]

    memory.record("https://news.test/", "30. Last story", at_bottom=True)
    assert memory.pages[0]["read"] == "to the end"


@pytest.mark.regression
def test_a_new_document_on_the_same_url_is_not_read_to_the_end_until_its_own_bottom_shows() -> None:
    """Regression: a wall's bottom counted as the bottom of the list that replaced it."""
    memory = SeenText()
    memory.record("https://q.test/questions", "Ray ID: a3f9", "Just a moment...", at_bottom=True)

    memory.record("https://q.test/questions", "1. First question", "Newest Questions")

    assert memory.pages == [
        {"url": "https://q.test/questions", "title": "Newest Questions", "read": "top part only"}
    ]


@pytest.mark.regression
def test_the_last_page_of_a_long_research_run_still_reaches_the_closing_answer() -> None:
    """Regression: HN pages read first spent the budget, so the final Wikipedia article was dropped."""
    memory = SeenText()
    memory.record("https://news.ycombinator.com/", _lines("story", 60, 80))
    for article in (
        "https://blog.google/tts",
        "https://drivingbench.com/",
        "https://stripe.dev/kai",
    ):
        memory.record(article, _lines(article, 40, 90))
    memory.record(
        "https://en.wikipedia.org/wiki/Transformer_(deep_learning)",
        "The transformer was introduced in 2017 by Vaswani et al. at Google.",
    )

    closing_input = memory.all_text

    assert "introduced in 2017 by Vaswani" in closing_input
    assert len(closing_input) <= JEV_SEEN_TEXT_MAX_CHARS + 500  # headers ride on top
    for page in memory.pages:
        assert f"## {page['url']}\n" in closing_input


def test_a_page_read_alone_keeps_the_whole_budget() -> None:
    memory = SeenText()
    memory.record("https://news.ycombinator.com/", _lines("story", 100, 120))

    assert JEV_SEEN_TEXT_MAX_CHARS - 120 <= len(memory.all_text) <= JEV_SEEN_TEXT_MAX_CHARS + 50


def _lines(prefix: str, count: int, width: int) -> str:
    return "\n".join(f"{prefix} {i}: ".ljust(width, "x") for i in range(count))
