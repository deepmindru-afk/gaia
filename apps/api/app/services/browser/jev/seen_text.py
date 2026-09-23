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
        self._to_the_end: set[str] = set()
        self._length = 0

    def record(self, url: str, text: str, title: str = "", *, at_bottom: bool = False) -> None:
        """Add this screen's lines to its page's memory; a page returned to keeps what it had."""
        self._page = url.split("#", 1)[0]
        if self._page.startswith("about:"):
            # The blank tab before the first navigate is no page read: judging it
            # spent a writer call on nothing, and a blocked run "reported" it.
            return
        if title and self._titles.get(self._page, title) != title:
            # A new document on the same url (a "Just a moment..." wall that cleared
            # into the list): the wall's bottom is not the list's.
            self._to_the_end.discard(self._page)
        if title:
            self._titles[self._page] = title
        if at_bottom:
            self._to_the_end.add(self._page)
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
        """The pages read, oldest first, each as its url, title and how far down it was read.

        A page counts as read to the end once a screen of it showed its bottom;
        until then only its top part has been read, which a judgement of "every
        item counted" or "the whole list seen" has to know.
        """
        return [
            {
                "url": page,
                "title": self._titles.get(page, ""),
                "read": "to the end" if page in self._to_the_end else "top part only",
            }
            for page in self._lines
        ]

    @property
    def all_text(self) -> str:
        """What was read on every page, each under its URL, for the closing answer."""
        return "\n\n".join(
            f"## {page}\n" + "\n".join(lines) for page, lines in self._lines.items() if lines
        )


__all__ = ["SeenText"]
