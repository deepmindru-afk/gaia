"""Unit tests for gaia_knowledge_service (GAIA self-knowledge in ChromaDB).

Search serves from an in-memory snapshot of a corpus that production never writes
(the only writers are the offline populate script and an explicit clear), so the
tests pin three things: the ranking is local cosine over the snapshot, the
snapshot is loaded once and reused, and every writer clears it.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import ValidationError
import pytest

from app.services.gaia_knowledge_service import (
    KnowledgeItem,
    gaia_knowledge_service,
)

_MOD = "app.services.gaia_knowledge_service"


@pytest.fixture
def chroma():
    """The raw Chroma client + the langchain client ``add_knowledge_batch`` uses."""
    collection = MagicMock()
    collection.get = AsyncMock(return_value={"documents": [], "metadatas": []})
    client = MagicMock()
    client.get_or_create_collection = AsyncMock(return_value=collection)
    client.delete_collection = AsyncMock()
    client.create_collection = AsyncMock()
    langchain = MagicMock()
    langchain.aadd_texts = AsyncMock()
    with patch(f"{_MOD}.ChromaClient") as m_cls:
        m_cls.get_client = AsyncMock(return_value=client)
        m_cls.get_langchain_client = AsyncMock(return_value=langchain)
        yield SimpleNamespace(collection=collection, client=client, langchain=langchain)


@pytest.fixture
def embeddings():
    """The google_embeddings provider, returning whatever the test declares."""
    fake = SimpleNamespace(
        aembed_documents=AsyncMock(return_value=[]),
        aembed_query=AsyncMock(return_value=[1.0, 0.0]),
    )
    with patch(f"{_MOD}.providers") as providers:
        providers.aget = AsyncMock(return_value=fake)
        yield fake


@pytest.fixture(autouse=True)
def clean_snapshot():
    """The service is a process singleton — never let a snapshot cross tests."""
    service = gaia_knowledge_service
    saved = (service._snapshot, service._loaded_at)
    service._snapshot = None
    service._loaded_at = 0.0
    yield
    service._snapshot, service._loaded_at = saved


def _corpus(
    chroma: SimpleNamespace,
    embeddings: SimpleNamespace,
    documents: list[str],
    doc_vectors: list[list[float]],
    *,
    query_vector: list[float] | None = None,
) -> None:
    """Declare the collection's documents and the vectors the embedder returns."""
    chroma.collection.get.return_value = {
        "documents": documents,
        "metadatas": [{"i": i} for i in range(len(documents))],
    }
    embeddings.aembed_documents.return_value = doc_vectors
    embeddings.aembed_query.return_value = query_vector or [1.0, 0.0]


class TestKnowledgeItemValidation:
    def test_accepts_non_empty_content(self):
        item = KnowledgeItem(content="GAIA can send emails")

        assert item.content == "GAIA can send emails"

    def test_rejects_empty_content(self):
        with pytest.raises(ValidationError):
            KnowledgeItem(content="")

    def test_rejects_whitespace_only_content(self):
        with pytest.raises(ValidationError):
            KnowledgeItem(content="   \n\t  ")

    def test_strips_whitespace(self):
        item = KnowledgeItem(content="  padded  ")

        assert item.content == "padded"


class TestSearchKnowledge:
    async def test_ranks_by_local_cosine_similarity(self, chroma, embeddings):
        """Query [1,0] against an aligned doc ([1,0]) and an orthogonal one
        ([0,1]) — nearest first, cosine distance (1 - similarity) as the score."""
        _corpus(
            chroma,
            embeddings,
            ["Doc aligned", "Doc orthogonal"],
            [[1.0, 0.0], [0.0, 1.0]],
        )

        results = await gaia_knowledge_service.search_knowledge("q", limit=2)

        assert [r.content for r in results] == ["Doc aligned", "Doc orthogonal"]
        assert results[0].relevance_score == pytest.approx(0.0)
        assert results[0].metadata == {"i": 0}
        assert results[1].relevance_score == pytest.approx(1.0)
        assert results[1].metadata == {"i": 1}

    async def test_limit_caps_the_results(self, chroma, embeddings):
        _corpus(
            chroma,
            embeddings,
            ["a", "b", "c"],
            [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]],
        )

        results = await gaia_knowledge_service.search_knowledge("q", limit=2)

        assert [r.content for r in results] == ["a", "b"]

    async def test_a_second_search_reuses_the_snapshot(self, chroma, embeddings):
        """The point of the snapshot: the per-turn cost is the query embedding
        alone — no Chroma read, no re-embedding of the corpus."""
        _corpus(chroma, embeddings, ["Doc"], [[1.0, 0.0]])

        first = await gaia_knowledge_service.search_knowledge("q", limit=5)
        second = await gaia_knowledge_service.search_knowledge("q", limit=5)

        assert first == second
        assert chroma.collection.get.await_count == 1
        assert embeddings.aembed_documents.await_count == 1
        assert embeddings.aembed_query.await_count == 2

    async def test_an_expired_snapshot_reloads(self, chroma, embeddings):
        _corpus(chroma, embeddings, ["Doc"], [[1.0, 0.0]])

        await gaia_knowledge_service.search_knowledge("q")
        gaia_knowledge_service._loaded_at = 0.0  # past any TTL
        await gaia_knowledge_service.search_knowledge("q")

        assert chroma.collection.get.await_count == 2

    async def test_a_failed_refresh_serves_the_previous_snapshot(self, chroma, embeddings):
        """A corpus for a section is enrichment: a refresh blip must degrade to
        the last good snapshot, not to an empty knowledge block."""
        _corpus(chroma, embeddings, ["Doc"], [[1.0, 0.0]])
        first = await gaia_knowledge_service.search_knowledge("q")

        chroma.collection.get.side_effect = RuntimeError("chroma down")
        gaia_knowledge_service._loaded_at = 0.0
        second = await gaia_knowledge_service.search_knowledge("q")

        assert second == first

    async def test_an_empty_corpus_yields_no_results(self, chroma, embeddings):
        _corpus(chroma, embeddings, [], [])

        assert await gaia_knowledge_service.search_knowledge("anything") == []

    async def test_a_first_load_failure_degrades_to_empty(self, chroma, embeddings):
        chroma.collection.get.side_effect = RuntimeError("chroma down")

        assert await gaia_knowledge_service.search_knowledge("anything") == []


class TestSnapshotInvalidation:
    """The writers are the only way the corpus changes; each must drop the
    snapshot so the next search reflects it rather than serving an hour stale."""

    async def test_add_knowledge_batch_invalidates(self, chroma, embeddings):
        _corpus(chroma, embeddings, ["Doc"], [[1.0, 0.0]])
        await gaia_knowledge_service.search_knowledge("q")

        await gaia_knowledge_service.add_knowledge_batch([KnowledgeItem(content="New")])
        await gaia_knowledge_service.search_knowledge("q")

        assert chroma.collection.get.await_count == 2

    async def test_clear_knowledge_invalidates(self, chroma, embeddings):
        _corpus(chroma, embeddings, ["Doc"], [[1.0, 0.0]])
        await gaia_knowledge_service.search_knowledge("q")

        await gaia_knowledge_service.clear_knowledge()
        await gaia_knowledge_service.search_knowledge("q")

        assert chroma.collection.get.await_count == 2


class TestAddKnowledgeBatch:
    async def test_empty_batch_returns_zero_without_touching_chroma(self, chroma, embeddings):
        assert await gaia_knowledge_service.add_knowledge_batch([]) == 0
        chroma.langchain.aadd_texts.assert_not_awaited()

    async def test_adds_texts_and_metadatas(self, chroma, embeddings):
        items = [
            KnowledgeItem(content="A", metadata={"x": 1}),
            KnowledgeItem(content="B"),
        ]

        count = await gaia_knowledge_service.add_knowledge_batch(items)

        assert count == 2
        chroma.langchain.aadd_texts.assert_awaited_once()
        assert chroma.langchain.aadd_texts.await_args.kwargs["texts"] == ["A", "B"]
        assert chroma.langchain.aadd_texts.await_args.kwargs["metadatas"] == [{"x": 1}, {}]

    async def test_failure_degrades_to_zero(self, chroma, embeddings):
        chroma.langchain.aadd_texts.side_effect = RuntimeError("chroma down")

        assert await gaia_knowledge_service.add_knowledge_batch([KnowledgeItem(content="A")]) == 0


class TestClearKnowledge:
    async def test_deletes_and_recreates_collection(self, chroma, embeddings):
        ok = await gaia_knowledge_service.clear_knowledge()

        assert ok is True
        chroma.client.delete_collection.assert_awaited_once_with(name="gaia_knowledge")
        chroma.client.create_collection.assert_awaited_once()
        assert chroma.client.create_collection.await_args.kwargs["name"] == "gaia_knowledge"
        assert chroma.client.create_collection.await_args.kwargs["metadata"] == {
            "hnsw:space": "cosine"
        }

    async def test_failure_degrades_to_false(self, chroma, embeddings):
        chroma.client.delete_collection.side_effect = RuntimeError("chroma down")

        assert await gaia_knowledge_service.clear_knowledge() is False
