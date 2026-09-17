"""Repository for the executor-free approval ledger (``approval_ledger``).

One row per gated call envelope. Uncached (``cache_policy = None``): rows are
decision state read at low volume, and every transition is a conditional write
the entity cache could only misrepresent. Rows never expire — a pending row
leaves only by user decision or agent revoke.

Dedup is query-first on (conversation_id, fingerprint) over live states. An
exact-race duplicate insert is possible but harmless: each id stays consistent
under CAS, and nothing auto-executes in Phase 1.
"""

from typing import Any
from uuid import uuid4

from app.db.repositories.base import MongoRepository
from app.models.hil_models import (
    LIVE_LEDGER_STATES,
    ApprovalLedgerDocument,
    LedgerState,
)


class ApprovalLedgerRepository(MongoRepository[ApprovalLedgerDocument, ApprovalLedgerDocument]):
    collection_name = "approval_ledger"
    document_model = ApprovalLedgerDocument
    update_model = ApprovalLedgerDocument
    uses_object_id = True
    cache_policy = None

    async def register(
        self,
        *,
        conversation_id: str,
        fingerprint: str,
        tool_name: str,
        args: dict[str, Any],
        summary: str,
        rationale: str = "",
        preview: str = "",
        owner_agent: str = "",
        blocked_by: list[str] | None = None,
        proposing_run_id: str | None = None,
    ) -> str:
        """Insert a PENDING row, or return the live duplicate's id.

        Dedup consults live states only: a terminal fingerprint re-registers
        fresh (revoked-then-reproposed must not collapse into a dead id).
        """
        existing = await self.find_live(fingerprint, conversation_id)
        if existing is not None:
            return existing.approval_id
        approval_id = f"ap_{uuid4().hex[:12]}"
        await self.create(
            ApprovalLedgerDocument(
                approval_id=approval_id,
                conversation_id=conversation_id,
                fingerprint=fingerprint,
                tool_name=tool_name,
                args=args,
                summary=summary,
                rationale=rationale,
                preview=preview,
                owner_agent=owner_agent,
                blocked_by=list(blocked_by or []),
                proposing_run_id=proposing_run_id,
            )
        )
        return approval_id

    async def transition(
        self, approval_id: str, expected: LedgerState, nxt: LedgerState
    ) -> bool:
        """Move one row ``expected -> nxt``, exactly once.

        The CAS behind every ledger edge: concurrent contenders race on the
        filter and exactly one wins. Bumps the row version for stale clients.
        """
        result = await self._raw_collection().update_one(
            {"approval_id": approval_id, "state": str(expected)},
            {"$set": {"state": str(nxt)}, "$inc": {"v": 1}},
        )
        return bool(result.modified_count)

    async def get_by_approval_id(self, approval_id: str) -> ApprovalLedgerDocument | None:
        """One row by its ``ap_`` id, or ``None``."""
        raw = await self._raw_collection().find_one({"approval_id": approval_id})
        return ApprovalLedgerDocument.model_validate(raw) if raw else None

    async def find_live(
        self, fingerprint: str, conversation_id: str
    ) -> ApprovalLedgerDocument | None:
        """Newest live (PENDING/APPROVED) row for a fingerprint, or ``None``."""
        cursor = (
            self._raw_collection()
            .find({"fingerprint": fingerprint, "conversation_id": conversation_id})
            .sort("created_at", -1)
        )
        raws = await cursor.to_list(length=50)
        for raw in raws:
            doc = ApprovalLedgerDocument.model_validate(raw)
            if doc.state in LIVE_LEDGER_STATES:
                return doc
        return None

    async def list_open(self, conversation_id: str) -> list[ApprovalLedgerDocument]:
        """Live rows for a conversation, oldest first (registration order)."""
        cursor = (
            self._raw_collection()
            .find(
                {
                    "conversation_id": conversation_id,
                    "state": {"$in": sorted(str(s) for s in LIVE_LEDGER_STATES)},
                }
            )
            .sort("created_at", 1)
        )
        return [ApprovalLedgerDocument.model_validate(raw) for raw in await cursor.to_list(length=200)]

    async def find_latest_denied(
        self, fingerprint: str, conversation_id: str
    ) -> ApprovalLedgerDocument | None:
        """Newest DENIED row for reject-memory context, or ``None``."""
        raw = await self._raw_collection().find_one(
            {
                "fingerprint": fingerprint,
                "conversation_id": conversation_id,
                "state": str(LedgerState.DENIED),
            },
            sort=[("created_at", -1)],
        )
        return ApprovalLedgerDocument.model_validate(raw) if raw else None


approval_ledger_repository = ApprovalLedgerRepository()
