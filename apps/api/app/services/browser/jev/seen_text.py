"""What the run has read on the page it is on, across screens.

The closing summary is written from the current screen, so a list scrolled past
its first screenful is answered from whatever happens to be showing at the end.
This keeps every line read on one page, in the order it was read and without
repeats, and starts over when the run moves to a different page.
"""

from __future__ import annotations

from app.constants.browser import JEV_SEEN_TEXT_MAX_CHARS


class SeenText:
    """Every distinct line read on the current page, oldest screen first."""

    def __init__(self) -> None:
        self._page = ""
        self._lines: list[str] = []
        self._seen: set[str] = set()
        self._length = 0

    def record(self, url: str, text: str) -> None:
        """Add this screen's lines to the page's memory, starting over on a different page."""
        page = url.split("#", 1)[0]
        if page != self._page:
            self._page, self._lines, self._seen, self._length = page, [], set(), 0
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped in self._seen:
                continue
            if self._length + len(stripped) + 1 > JEV_SEEN_TEXT_MAX_CHARS:
                return
            self._seen.add(stripped)
            self._lines.append(stripped)
            self._length += len(stripped) + 1

    @property
    def text(self) -> str:
        return "\n".join(self._lines)


__all__ = ["SeenText"]
