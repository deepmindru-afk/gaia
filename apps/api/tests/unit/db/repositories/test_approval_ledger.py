"""Hermetic unit tests for ``ApprovalLedgerRepository``.

Real-Mongo proof belongs in the contracts tier; this tier pins the exact
filters and update documents handed to the driver — especially the CAS filter
(``state == expected``), which is the whole safety story, and the live-only
dedup scope (terminal fingerprints must re-register fresh).
"""

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.db.repositories.approval_ledger import (
    ApprovalLedgerRepository,
    approval_ledger_repository,
)
from app.models.hil_models import LedgerState


def _doc(state: str = "pending", approval_id: str = "ap_1", fingerprint: str = "fp") -> dict[str, Any]:
    return {
        "_id": "oid",
        "approval_id": approval_id,
        "conversation_id": "c1",
        "fingerprint": fingerprint,
        "tool_name": "T",
        "args": {},
        "summary": "s",
        "rationale": "",
        "preview": "",
        "owner_agent": "executor",
        "blocked_by": [],
        "state": state,
        "v": 0,
    }


@pytest.fixture
def collection() -> Iterator[MagicMock]:
    mock = MagicMock()
    mock.insert_one = AsyncMock(return_value=MagicMock(inserted_id="oid"))
    mock.update_one = AsyncMock(return_value=MagicMock(modified_count=1))
    mock.find_one = AsyncMock(return_value=_doc())
    cursor = MagicMock()
    cursor.sort = MagicMock(return_value=cursor)
    cursor.to_list = AsyncMock(return_value=[])
    mock.find = MagicMock(return_value=cursor)
    with patch("app.db.repositories.base.get_async_collection", return_value=mock):
        yield mock


@pytest.fixture
def repo() -> ApprovalLedgerRepository:
    return ApprovalLedgerRepository()


@pytest.mark.unit
class TestRegister:
    async def test_inserts_pending_and_returns_new_id(
        self, repo: ApprovalLedgerRepository, collection: MagicMock
    ) -> None:
        ap_id = await repo.register(
            conversation_id="c1",
            fingerprint="fp",
            tool_name="T",
            args={},
            summary="s",
        )

        assert ap_id.startswith("ap_")
        inserted = collection.insert_one.await_args.args[0]
        assert inserted["state"] == "pending"
        assert inserted["approval_id"] == ap_id

    async def test_live_duplicate_returns_existing_id_without_insert(
        self, repo: ApprovalLedgerRepository, collection: MagicMock
    ) -> None:
        cursor = MagicMock()
        cursor.sort = MagicMock(return_value=cursor)
        cursor.to_list = AsyncMock(return_value=[_doc(state="approved")])
        collection.find = MagicMock(return_value=cursor)

        ap_id = await repo.register(
            conversation_id="c1", fingerprint="fp", tool_name="T", args={}, summary="s"
        )

        assert ap_id == "ap_1"
        collection.insert_one.assert_not_awaited()

    async def test_terminal_fingerprint_registers_fresh(
        self, repo: ApprovalLedgerRepository, collection: MagicMock
    ) -> None:
        cursor = MagicMock()
        cursor.sort = MagicMock(return_value=cursor)
        cursor.to_list = AsyncMock(return_value=[_doc(state="revoked")])
        collection.find = MagicMock(return_value=cursor)

        ap_id = await repo.register(
            conversation_id="c1", fingerprint="fp", tool_name="T", args={}, summary="s"
        )

        assert ap_id != "ap_1"
        collection.insert_one.assert_awaited_once()


@pytest.mark.unit
class TestTransition:
    async def test_matching_state_transitions_and_bumps_version(
        self, repo: ApprovalLedgerRepository, collection: MagicMock
    ) -> None:
        assert await repo.transition("ap_1", LedgerState.PENDING, LedgerState.APPROVED) is True

        flt, update = collection.update_one.await_args.args
        assert flt == {"approval_id": "ap_1", "state": "pending"}
        assert update["$set"] == {"state": "approved"}
        assert update["$inc"] == {"v": 1}

    async def test_mismatched_state_is_a_quiet_false(
        self, repo: ApprovalLedgerRepository, collection: MagicMock
    ) -> None:
        collection.update_one = AsyncMock(return_value=MagicMock(modified_count=0))

        assert await repo.transition("ap_1", LedgerState.PENDING, LedgerState.DENIED) is False

    async def test_decision_stamp_rides_in_the_same_write(
        self, repo: ApprovalLedgerRepository, collection: MagicMock
    ) -> None:
        assert (
            await repo.transition(
                "ap_1",
                LedgerState.PENDING,
                LedgerState.DENIED,
                decided_by="u1",
                feedback="nope",
            )
            is True
        )

        _, update = collection.update_one.await_args.args
        assert update["$set"]["state"] == "denied"
        assert update["$set"]["decided_by"] == "u1"
        assert update["$set"]["feedback"] == "nope"
        assert "decided_at" in update["$set"]


@pytest.mark.unit
class TestReads:
    async def test_list_open_filters_live_and_sorts_oldest_first(
        self, repo: ApprovalLedgerRepository, collection: MagicMock
    ) -> None:
        cursor = MagicMock()
        cursor.sort = MagicMock(return_value=cursor)
        cursor.to_list = AsyncMock(return_value=[_doc()])
        collection.find = MagicMock(return_value=cursor)

        docs = await repo.list_open("c1")

        flt = collection.find.call_args.args[0]
        assert flt["conversation_id"] == "c1"
        assert set(flt["state"]["$in"]) == {"pending", "approved"}
        cursor.sort.assert_called_once_with("created_at", 1)
        assert len(docs) == 1 and docs[0].approval_id == "ap_1"

    async def test_find_latest_denied_queries_denied_newest_first(
        self, repo: ApprovalLedgerRepository, collection: MagicMock
    ) -> None:
        collection.find_one = AsyncMock(return_value=_doc(state="denied"))

        doc = await repo.find_latest_denied("fp", "c1")

        flt = collection.find_one.await_args.args[0]
        assert flt == {"fingerprint": "fp", "conversation_id": "c1", "state": "denied"}
        assert doc is not None and doc.state == LedgerState.DENIED

    async def test_singleton_exists(self) -> None:
        assert isinstance(approval_ledger_repository, ApprovalLedgerRepository)

    async def test_list_live_by_owners_batches_one_query(
        self, repo: ApprovalLedgerRepository, collection: MagicMock
    ) -> None:
        """List views read one page, not one row per todo."""
        cursor = MagicMock()
        cursor.sort = MagicMock(return_value=cursor)
        cursor.to_list = AsyncMock(return_value=[_doc()])
        collection.find = MagicMock(return_value=cursor)

        docs = await repo.list_live_by_owners("todo", ["todo-1", "todo-2"])

        flt = collection.find.call_args.args[0]
        assert flt["owner_run_type"] == "todo"
        assert flt["owner_id"] == {"$in": ["todo-1", "todo-2"]}
        assert set(flt["state"]["$in"]) == {"pending", "approved"}
        assert len(docs) == 1

    async def test_list_live_by_owners_empty_means_no_query(
        self, repo: ApprovalLedgerRepository, collection: MagicMock
    ) -> None:
        assert await repo.list_live_by_owners("todo", []) == []
        assert await repo.list_live_by_owners("", ["todo-1"]) == []
        collection.find.assert_not_called()
