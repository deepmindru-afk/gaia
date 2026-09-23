"""Operator prompts for scripts that already run inside an event loop.

input() blocks the loop, and asyncio.to_thread(input) leaves a thread parked on
stdin that asyncio.run waits for after Ctrl+C, so the script hangs until Enter.
"""

from __future__ import annotations

import asyncio
import sys


async def ainput(prompt: str) -> str:
    """Read one stdin line without blocking the loop; Ctrl+C cancels the wait at once."""
    print(prompt, end="", flush=True)
    loop = asyncio.get_running_loop()
    line: asyncio.Future[str] = loop.create_future()
    fd = sys.stdin.fileno()

    def _on_readable() -> None:
        if not line.done():
            line.set_result(sys.stdin.readline())

    loop.add_reader(fd, _on_readable)
    try:
        return (await line).rstrip("\n")
    finally:
        loop.remove_reader(fd)
