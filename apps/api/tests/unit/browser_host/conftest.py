"""Shared setup for the browser-host unit tests.

Admission is memory-based (ChromiumHost._reserve_slot reads the real cgroup /
system memory), which would make every create test depend on the machine's live
memory. Default every test to ample headroom and no backpressure wait so the
existing behaviour tests stay hermetic and fast; the memory-gate tests override
chromium.memory_usage_mb themselves to simulate pressure.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.browser_host import chromium, proxy
from app.config.settings import settings


@pytest.fixture(autouse=True)
def _no_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the proxy's DNS-resolving guard so no unit test touches real DNS."""
    monkeypatch.setattr(proxy, "assert_public_http_url", AsyncMock())


@pytest.fixture(autouse=True)
def _ample_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chromium, "memory_usage_mb", lambda: (100.0, 100_000.0))
    monkeypatch.setattr(settings, "BROWSER_HOST_ADMISSION_WAIT_SECONDS", 0.0)
