"""Per-conversation record of in-context activated integrations."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.core.subagents import active_integrations
from app.db.redis import RedisCache


def _client(members: object = None) -> MagicMock:
    client = MagicMock()
    client.sadd = AsyncMock()
    client.expire = AsyncMock()
    client.smembers = AsyncMock(return_value=members if members is not None else set())
    return client


@pytest.mark.unit
class TestMarkActive:
    async def test_stamp_writes_set_and_refreshes_ttl(self, monkeypatch) -> None:
        from app.constants.cache import ACTIVATION_ACTIVE_PREFIX, ACTIVATION_ACTIVE_TTL

        client = _client()
        monkeypatch.setattr(RedisCache, "client", client)

        await active_integrations.mark_active("c1", "github")

        client.sadd.assert_awaited_once_with(f"{ACTIVATION_ACTIVE_PREFIX}c1", "github")
        client.expire.assert_awaited_once_with(
            f"{ACTIVATION_ACTIVE_PREFIX}c1", ACTIVATION_ACTIVE_TTL
        )

    async def test_no_redis_client_degrades_without_raising(self, monkeypatch) -> None:
        monkeypatch.setattr(RedisCache, "client", None)

        await active_integrations.mark_active("c1", "github")


@pytest.mark.unit
class TestGetActive:
    async def test_returns_stamped_ids(self, monkeypatch) -> None:
        monkeypatch.setattr(RedisCache, "client", _client({"github", "gmail"}))

        assert await active_integrations.get_active("c1") == {"github", "gmail"}

    async def test_no_client_means_nothing_active(self, monkeypatch) -> None:
        monkeypatch.setattr(RedisCache, "client", None)

        assert await active_integrations.get_active("c1") == set()

    async def test_missing_conversation_id_never_touches_redis(self) -> None:
        with patch.object(RedisCache, "client", new=AsyncMock()) as client:
            assert await active_integrations.get_active(None) == set()
            client.smembers.assert_not_awaited()

    async def test_redis_failure_degrades_to_empty(self, monkeypatch) -> None:
        client = _client()
        client.smembers = AsyncMock(side_effect=ConnectionError("redis down"))
        monkeypatch.setattr(RedisCache, "client", client)

        assert await active_integrations.get_active("c1") == set()
