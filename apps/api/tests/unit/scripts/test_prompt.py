"""scripts._prompt.ainput against real stdin sources: a redirected file, an empty stdin, a terminal."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
import os
from pathlib import Path
import sys
from typing import TextIO

import pytest
from scripts._prompt import ainput


@pytest.fixture
def use_stdin(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[TextIO], None]]:
    """Point sys.stdin at a real stream the test opened, and close it afterwards."""
    opened: list[TextIO] = []

    def use(stream: TextIO) -> None:
        opened.append(stream)
        monkeypatch.setattr(sys, "stdin", stream)

    yield use
    for stream in opened:
        stream.close()


async def test_answer_comes_from_a_redirected_file(
    tmp_path: Path, use_stdin: Callable[[TextIO], None]
) -> None:
    """Regression: epoll refuses regular files, so `script < answers.txt` died with PermissionError on Linux."""
    answers = tmp_path / "answers.txt"
    answers.write_text("yes\nignored\n")
    use_stdin(answers.open())

    assert await ainput("Delete? ") == "yes"


async def test_closed_stdin_raises_eof_like_input(use_stdin: Callable[[TextIO], None]) -> None:
    use_stdin(open(os.devnull))  # noqa: SIM115 -- closed by the use_stdin fixture

    with pytest.raises(EOFError):
        await ainput("Delete? ")


@pytest.mark.timeout(10)
async def test_terminal_answer_arrives_without_blocking_the_loop(
    use_stdin: Callable[[TextIO], None],
) -> None:
    """The typist answers from another task: a loop-blocking read would never let that task run."""
    controller, terminal = os.openpty()
    use_stdin(os.fdopen(terminal, "r"))

    async def type_answer() -> None:
        await asyncio.sleep(0.05)
        os.write(controller, b"no\n")

    try:
        answer, _ = await asyncio.gather(ainput("Delete? "), type_answer())
    finally:
        os.close(controller)
    assert answer == "no"
