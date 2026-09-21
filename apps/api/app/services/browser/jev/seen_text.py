"""What the run has read, page by page, across screens.

The closing summary is written from the current screen, so a list scrolled past
its first screenful, or a task that reads several pages, would otherwise be
answered from whatever happens to be showing at the end. This keeps every line
read on each page, in the order it was read and without repeats, and keeps the
pages in the order the run opened them.
"""

from __future__ import annotations

from app.constants.browser import JEV_SEEN_TEXT_MAX_CHARS


class SeenText:
    """Every distinct line read on each page the run has opened, oldest first."""

    def __init__(self) -> None:
        self._page = ""
        self._lines: dict[str, list[str]] = {}
        self._titles: dict[str, str] = {}
        self._seen: dict[str, set[str]] = {}
        self._length = 0

    def record(self, url: str, text: str, title: str = "") -> None:
        """Add this screen's lines to its page's memory; a page returned to keeps what it had."""
        self._page = url.split("#", 1)[0]
        if title:
            self._titles[self._page] = title
        lines = self._lines.setdefault(self._page, [])
        seen = self._seen.setdefault(self._page, set())
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped in seen:
                continue
            if self._length + len(stripped) + 1 > JEV_SEEN_TEXT_MAX_CHARS:
                return
            seen.add(stripped)
            lines.append(stripped)
            self._length += len(stripped) + 1

    @property
    def text(self) -> str:
        """What was read on the current page."""
        return "\n".join(self._lines.get(self._page, []))

    @property
    def pages(self) -> list[dict[str, str]]:
        """The pages read, oldest first, each as its url and title."""
        return [{"url": page, "title": self._titles.get(page, "")} for page in self._lines]

    @property
    def all_text(self) -> str:
        """What was read on every page, each under its URL, for the closing answer."""
        return "\n\n".join(
            f"## {page}\n" + "\n".join(lines) for page, lines in self._lines.items() if lines
        )


__all__ = ["SeenText"]
