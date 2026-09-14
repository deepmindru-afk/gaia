"""Unit tests for CheckpointerManager.setup's serialization of langgraph's migrations."""

from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from app.agents.core.graph_builder import checkpointer_manager as cm


def _patched_setup(order: MagicMock, *, store_error: Exception | None = None):
    conn = AsyncMock()
    conn.execute.side_effect = lambda sql, params: order.execute(sql, params)
    pool = MagicMock()
    pool.open = AsyncMock()
    pool.connection.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.connection.return_value.__aexit__ = AsyncMock(return_value=None)

    saver = MagicMock()
    saver.setup = AsyncMock(side_effect=lambda: order.saver_setup())
    store = MagicMock()
    store.setup = AsyncMock(side_effect=store_error or (lambda: order.store_setup()))
    store_ctx = MagicMock()
    store_ctx.__aenter__ = AsyncMock(return_value=store)
    store_ctx.__aexit__ = AsyncMock(return_value=None)

    return (
        patch.object(cm, "AsyncConnectionPool", return_value=pool),
        patch.object(cm, "AsyncPostgresSaver", return_value=saver),
        patch.object(cm.AsyncPostgresStore, "from_conn_string", return_value=store_ctx),
    )


async def test_langgraph_migrations_run_under_the_advisory_lock() -> None:
    """Concurrent starters must not race langgraph's CREATE TYPE checkpoint_migrations."""
    order = MagicMock()
    pool_patch, saver_patch, store_patch = _patched_setup(order)

    with pool_patch, saver_patch, store_patch:
        await cm.CheckpointerManager("postgresql://localhost/test").setup()

    lock = (cm.LANGGRAPH_SETUP_LOCK_ID,)
    assert order.mock_calls == [
        call.execute("SELECT pg_advisory_lock(%s)", lock),
        call.saver_setup(),
        call.store_setup(),
        call.execute("SELECT pg_advisory_unlock(%s)", lock),
    ]


async def test_a_failed_migration_still_releases_the_lock() -> None:
    order = MagicMock()
    pool_patch, saver_patch, store_patch = _patched_setup(
        order, store_error=RuntimeError("store DDL failed")
    )

    with pool_patch, saver_patch, store_patch, pytest.raises(RuntimeError, match="store DDL"):
        await cm.CheckpointerManager("postgresql://localhost/test").setup()

    assert order.mock_calls[-1] == call.execute(
        "SELECT pg_advisory_unlock(%s)", (cm.LANGGRAPH_SETUP_LOCK_ID,)
    )
