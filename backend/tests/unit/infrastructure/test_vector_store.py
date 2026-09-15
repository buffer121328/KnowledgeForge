"""VectorStoreService 单元测试

覆盖:
  - add_chunks: 写入 tenant_id 到 metadata
  - search: tenant_id 过滤（chroma / pgvector 双后端）
  - delete_by_doc_id: 带 tenant_id 的删除（防跨租户删除）
  - get_stats: 后端统计
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from domain.documents import DocType, DocumentChunk
from infrastructure.retrieval.vector_store import VectorStoreService


def _make_chunk(
    content: str,
    doc_id: str,
    tenant_id: str = "",
    department_id: str = "",
) -> DocumentChunk:
    return DocumentChunk(
        content=content,
        doc_id=doc_id,
        chunk_index=0,
        doc_type=DocType.TEXT,
        metadata={"source": "test.txt", "department_id": department_id},
        tenant_id=tenant_id,
    )


@pytest.fixture
def chroma_store():
    """Chroma 后端 VectorStoreService（mock 掉 embeddings 与 chromadb）"""
    with patch("infrastructure.retrieval.vector_store.OpenAIEmbeddings"):
        store = VectorStoreService()
        store._backend = "chroma"
        store._store = MagicMock()
        store.embeddings = AsyncMock()
        return store


@pytest.fixture
def pgvector_store():
    """PGVector 后端 VectorStoreService"""
    with patch("infrastructure.retrieval.vector_store.OpenAIEmbeddings"):
        store = VectorStoreService()
        store._backend = "pgvector"
        store._store = AsyncMock()
        store.embeddings = AsyncMock()
        return store


class TestAddChunks:
    @pytest.mark.asyncio
    async def test_add_chunks_chroma_writes_tenant_and_department_scope(self, chroma_store):
        chunk = _make_chunk(
            "内容",
            "doc_001",
            tenant_id="org_001",
            department_id="finance",
        )
        chroma_store.embeddings.aembed_documents.return_value = [[0.1, 0.2]]

        count = await chroma_store.add_chunks([chunk])
        assert count == 1
        chroma_store._store.upsert.assert_called_once()
        call_kwargs = chroma_store._store.upsert.call_args[1]
        assert call_kwargs["metadatas"][0]["chunk_id"] == "doc_001#chunk-0"
        assert call_kwargs["metadatas"][0]["source_document_id"] == "doc_001"
        assert call_kwargs["metadatas"][0]["tenant_id"] == "org_001"
        assert call_kwargs["metadatas"][0]["department_id"] == "finance"

    @pytest.mark.asyncio
    async def test_add_chunks_empty_returns_zero(self, chroma_store):
        count = await chroma_store.add_chunks([])
        assert count == 0
        chroma_store._store.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_chunks_default_tenant_id_empty_string(self, chroma_store):
        chunk = _make_chunk("内容", "doc_002", tenant_id="")
        chroma_store.embeddings.aembed_documents.return_value = [[0.1]]

        await chroma_store.add_chunks([chunk])
        call_kwargs = chroma_store._store.upsert.call_args[1]
        assert call_kwargs["metadatas"][0]["tenant_id"] == ""

    @pytest.mark.asyncio
    async def test_add_chunks_pgvector(self, pgvector_store):
        chunk = _make_chunk("内容", "doc_001", tenant_id="org_001")
        await pgvector_store.add_chunks([chunk])
        pgvector_store._store.aadd_texts.assert_called_once()
        call_kwargs = pgvector_store._store.aadd_texts.call_args[1]
        assert call_kwargs["metadatas"][0]["tenant_id"] == "org_001"


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_chroma_with_tenant_filter(self, chroma_store):
        chroma_store.embeddings.aembed_query.return_value = [0.1, 0.2]
        chroma_store._store.query.return_value = {
            "ids": [["doc-1#chunk-3"]],
            "documents": [["doc content"]],
            "metadatas": [[{
                "source": "s.pdf",
                "doc_id": "doc-1",
                "chunk_index": 3,
                "tenant_id": "org_001",
                "department_id": "finance",
            }]],
            "distances": [[0.15]],
        }

        results = await chroma_store.search("query", top_k=5, tenant_id="org_001")
        assert len(results) == 1
        doc, score = results[0]
        assert doc["content"] == "doc content"
        assert doc["metadata"]["chunk_id"] == "doc-1#chunk-3"
        assert doc["metadata"]["source_document_id"] == "doc-1"
        assert doc["metadata"]["chunk_index"] == 3
        assert doc["metadata"]["tenant_id"] == "org_001"
        assert doc["metadata"]["department_id"] == "finance"
        assert score == 0.85  # 1.0 - 0.15

        call_kwargs = chroma_store._store.query.call_args[1]
        assert call_kwargs["where"] == {"tenant_id": "org_001"}

    @pytest.mark.asyncio
    async def test_search_chroma_without_tenant_filter(self, chroma_store):
        chroma_store.embeddings.aembed_query.return_value = [0.1]
        chroma_store._store.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        await chroma_store.search("query", top_k=5, tenant_id=None)
        call_kwargs = chroma_store._store.query.call_args[1]
        assert call_kwargs["where"] is None

    @pytest.mark.asyncio
    async def test_search_chroma_missing_or_mismatched_id_fails_closed(self, chroma_store):
        chroma_store.embeddings.aembed_query.return_value = [0.1]
        chroma_store._store.query.return_value = {
            "ids": [["doc-1#chunk-9"]],
            "documents": [["content"]],
            "metadatas": [[{
                "doc_id": "doc-1",
                "chunk_id": "doc-1#chunk-1",
                "chunk_index": 1,
            }]],
            "distances": [[0.1]],
        }

        assert await chroma_store.search("query") == []

        chroma_store._store.query.return_value = {
            "documents": [["content"]],
            "metadatas": [[{"doc_id": "doc-1", "chunk_index": 1}]],
            "distances": [[0.1]],
        }
        assert await chroma_store.search("query") == []

    @pytest.mark.asyncio
    async def test_search_pgvector_with_tenant_filter(self, pgvector_store):
        mock_doc = MagicMock()
        mock_doc.page_content = "content"
        mock_doc.metadata = {
            "source": "s",
            "doc_id": "doc-1",
            "source_document_id": "doc-1",
            "chunk_id": "doc-1#chunk-2",
            "chunk_index": 2,
            "tenant_id": "org_001",
            "department_id": "finance",
        }
        pgvector_store._store.asimilarity_search_with_score.return_value = [(mock_doc, 0.5)]

        results = await pgvector_store.search("query", top_k=5, tenant_id="org_001")
        assert len(results) == 1
        assert results[0][0]["content"] == "content"
        assert results[0][0]["metadata"]["chunk_id"] == "doc-1#chunk-2"

        call_kwargs = pgvector_store._store.asimilarity_search_with_score.call_args[1]
        assert call_kwargs["filter"] == {"tenant_id": "org_001"}

    @pytest.mark.asyncio
    async def test_search_pgvector_without_tenant_filter(self, pgvector_store):
        mock_doc = MagicMock()
        mock_doc.page_content = "content"
        mock_doc.metadata = {
            "doc_id": "doc-1",
            "chunk_id": "doc-1#chunk-0",
            "chunk_index": 0,
        }
        pgvector_store._store.asimilarity_search_with_score.return_value = [(mock_doc, 0.5)]

        await pgvector_store.search("query", top_k=5, tenant_id=None)
        call_kwargs = pgvector_store._store.asimilarity_search_with_score.call_args[1]
        assert "filter" not in call_kwargs

    @pytest.mark.asyncio
    async def test_search_pgvector_missing_chunk_identity_fails_closed(self, pgvector_store):
        mock_doc = MagicMock()
        mock_doc.page_content = "content"
        mock_doc.metadata = {"doc_id": "doc-1", "chunk_index": 0}
        pgvector_store._store.asimilarity_search_with_score.return_value = [(mock_doc, 0.5)]

        assert await pgvector_store.search("query") == []


class TestDeleteByDocId:
    @pytest.mark.asyncio
    async def test_delete_chroma_without_tenant_id(self, chroma_store):
        chroma_store._store.get.return_value = {"ids": ["id1", "id2"]}

        count = await chroma_store.delete_by_doc_id("doc_001", tenant_id=None)
        assert count == 2
        chroma_store._store.delete.assert_called_once_with(ids=["id1", "id2"])

        get_call = chroma_store._store.get.call_args[1]
        assert get_call["where"] == {"doc_id": "doc_001"}

    @pytest.mark.asyncio
    async def test_delete_chroma_with_tenant_id_prevents_cross_tenant(self, chroma_store):
        """带 tenant_id 时使用 $and 过滤，防跨租户删除"""
        chroma_store._store.get.return_value = {"ids": ["id1"]}

        count = await chroma_store.delete_by_doc_id("doc_001", tenant_id="org_001")
        assert count == 1

        get_call = chroma_store._store.get.call_args[1]
        assert get_call["where"] == {
            "$and": [{"doc_id": "doc_001"}, {"tenant_id": "org_001"}]
        }

    @pytest.mark.asyncio
    async def test_delete_chroma_no_matching_ids(self, chroma_store):
        chroma_store._store.get.return_value = {"ids": []}

        count = await chroma_store.delete_by_doc_id("doc_001")
        assert count == 0
        chroma_store._store.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_pgvector_returns_zero(self, pgvector_store):
        count = await pgvector_store.delete_by_doc_id("doc_001")
        assert count == 0


class TestGetStats:
    @pytest.mark.asyncio
    async def test_get_stats_chroma(self, chroma_store):
        chroma_store._store.count.return_value = 42
        stats = await chroma_store.get_stats()
        assert stats["backend"] == "chroma"
        assert stats["total_vectors"] == 42
        assert stats["collection"] == "knowledge_chunks"

    @pytest.mark.asyncio
    async def test_get_stats_pgvector(self, pgvector_store):
        stats = await pgvector_store.get_stats()
        assert stats["backend"] == "pgvector"
        assert stats["collection"] == "knowledge_chunks"


class TestInit:
    @pytest.mark.asyncio
    async def test_init_chroma(self):
        with patch("infrastructure.retrieval.vector_store.OpenAIEmbeddings"), \
             patch("chromadb.HttpClient") as mock_client:
            mock_collection = MagicMock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection

            store = VectorStoreService()
            store._backend = "chroma"
            await store.init()
            assert store._store is mock_collection

    @pytest.mark.asyncio
    async def test_init_pgvector(self):
        with patch("infrastructure.retrieval.vector_store.OpenAIEmbeddings"), \
             patch("langchain_community.vectorstores.PGVector") as mock_pg:
            store = VectorStoreService()
            store._backend = "pgvector"
            await store.init()
            mock_pg.assert_called_once()
