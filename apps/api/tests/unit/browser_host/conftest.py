"""Shared setup for the browser-host unit tests.

Default every test to ample memory headroom and no backpressure wait so
admission stays hermetic instead of depending on the machine's live memory;
memory-gate tests override memory_usage_mb themselves to simulate pressure.
"""

from __future__ import annotations

import pytest

from app.browser_host import chromium
from app.config.settings import settings


@pytest.fixture(autouse=True)
def _ample_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chromium, "memory_usage_mb", lambda: (100.0, 100_000.0))
    monkeypatch.setattr(settings, "BROWSER_HOST_ADMISSION_WAIT_SECONDS", 0.0)
