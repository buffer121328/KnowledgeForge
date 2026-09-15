"""
向量存储服务 — 支持 ChromaDB / PGVector 双后端

职责:
  1. 文档块向量化 (Embedding)
  2. 向量存储 & 检索
  3. 按 doc_id 删除（支持增量更新）
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from langchain_openai import OpenAIEmbeddings

from domain.documents import DocumentChunk
from shared.config import settings


class VectorStoreService:
    """向量库统一接口，底层可切换 ChromaDB / PGVector"""

    COLLECTION_NAME = "knowledge_chunks"

    def __init__(self) -> None:
        # DashScope 等兼容接口不支持 OpenAI 的 token 切分预检，开启会把 input 打坏导致 400
        """Initialize the vector store service."""
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
            check_embedding_ctx_length=False,
            chunk_size=settings.embedding_batch_size,
        )
        self._store: Any = None
        self._backend = settings.vector_store_type

    # ── initialization ───────────────────────────────────────

    async def init(self) -> None:
        """Initialize the vector store service resources."""
        if self._backend == "chroma":
            await self._init_chroma()
        else:
            await self._init_pgvector()

    async def _init_chroma(self) -> None:
        """Initialize the Chroma vector-store backend."""
        import chromadb
        client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        self._store = client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    async def _init_pgvector(self) -> None:
        """Initialize the pgvector vector-store backend."""
        from langchain_community.vectorstores import PGVector
        self._store = PGVector(
            connection_string=settings.pgvector_dsn,
            collection_name=self.COLLECTION_NAME,
            embedding_function=self.embeddings,
        )

    async def health_check(self) -> bool:
        """Return whether the initialized vector backend accepts a lightweight probe."""
        if self._store is None:
            return False
        if self._backend == "chroma":
            import asyncio

            await asyncio.to_thread(self._store.count)
        return True

    # ── CRUD ─────────────────────────────────────────────────

    async def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """向量化并批量存储文档块，返回成功写入的块数。

        Args:
            chunks: 待写入的文档块列表；为空时直接返回 0。
        """
        if not chunks:
            return 0

        texts = [c.content for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "doc_id": c.doc_id,
                "source_document_id": c.doc_id,
                "chunk_id": c.chunk_id,
                "doc_type": c.doc_type.value,
                "source": c.metadata.get("source", ""),
                "file_name": c.metadata.get("file_name")
                or os.path.basename(str(c.metadata.get("source", "")))
                or c.doc_id,
                "chunk_index": c.chunk_index,
                "tenant_id": c.tenant_id or "",
                # 部门 ID 冗余存入向量元数据，供检索期按可见部门过滤。
                "department_id": str(c.metadata.get("department_id") or ""),
            }
            for c in chunks
        ]

        if self._backend == "chroma":
            vectors = await self.embeddings.aembed_documents(texts)
            self._store.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
        else:
            await self._store.aadd_texts(texts=texts, metadatas=metadatas, ids=ids)

        return len(chunks)

    async def fetch_chunks(self, chunk_ids: list[str]) -> list[tuple[dict, float]]:
        """按 chunk_id 精确取回已入库分块（用于授权上下文的目录区扩展）。

        Args:
            chunk_ids: 需要取回的 chunk_id 列表；不存在的 id 会被底层忽略。

        Returns:
            (文档, 相关性分数 1.0) 列表；分数为占位值，调用方应自行处理排序。
        """
        if not chunk_ids or self._backend != "chroma":
            return []
        got = await asyncio.to_thread(
            self._store.get,
            ids=list(dict.fromkeys(chunk_ids)),
            include=["documents", "metadatas"],
        )
        results: list[tuple[dict, float]] = []
        for chunk_id, document, metadata in zip(
            got.get("ids") or [],
            got.get("documents") or [],
            got.get("metadatas") or [],
            strict=True,
        ):
            if not isinstance(document, str) or not document.strip():
                continue
            safe_metadata = dict(metadata) if isinstance(metadata, dict) else {}
            safe_metadata["chunk_id"] = chunk_id
            results.append(({"content": document, "metadata": safe_metadata}, 1.0))
        return results

    async def search(
        self,
        query: str,
        top_k: int = 5,
        tenant_id: str | None = None,
        visible_department_ids: tuple[str, ...] | list[str] | None = None,
    ) -> list[tuple[dict, float]]:
        """语义搜索，返回 (文档, 分数) 列表

        Args:
            query: 查询文本。
            top_k: 返回的最相似结果数量。
            tenant_id: 若提供则只搜索该租户的数据；为 None 表示不过滤（管理员场景）。
            visible_department_ids: 用户可见的部门 ID 集合；None 表示不过滤部门，非 None 时空集合以哨兵值收紧为不可见任何部门（fail-closed）。
        """
        if self._backend == "chroma":
            q_vec = await self.embeddings.aembed_query(query)
            where_filter: dict | None = None
            # ① 逐维度收集过滤条件：租户与可见部门相互独立、可叠加。
            filters: list[dict[str, Any]] = []
            if tenant_id is not None:
                filters.append({"tenant_id": tenant_id})
            if visible_department_ids is not None:
                # ② 关键安全关卡：空集合替换为不可能命中的哨兵值，防止“无可见部门”退化为全量可见。
                departments = [value for value in visible_department_ids if value]
                filters.append({"department_id": {"$in": departments}} if departments else {"department_id": "__no_visible_department__"})
            # ③ 单条件直接下发，多条件用 $and 组合。
            if len(filters) == 1:
                where_filter = filters[0]
            elif filters:
                where_filter = {"$and": filters}
            results = self._store.query(
                query_embeddings=[q_vec],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
            out: list[tuple[dict, float]] = []
            ids = results.get("ids", [[]])[0]
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]
            if not (len(ids) == len(docs) == len(metas) == len(dists)):
                return []
            for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
                meta = dict(meta or {})
                canonical_id = str(chunk_id or "").strip()
                stored_id = str(meta.get("chunk_id") or "").strip()
                if not canonical_id or (stored_id and stored_id != canonical_id):
                    continue
                doc_id = str(meta.get("source_document_id") or meta.get("doc_id") or "").strip()
                if not doc_id:
                    continue
                meta["chunk_id"] = canonical_id
                meta["doc_id"] = doc_id
                meta["source_document_id"] = doc_id
                score = 1.0 - dist  # cosine distance → similarity
                out.append(({"content": doc, "source": meta.get("source", ""), "metadata": meta}, score))
            return out
        else:
            kwargs: dict[str, Any] = {"k": top_k}
            # PGVector 分支：与 Chroma 分支相同的租户 + 可见部门过滤语义（含空集合 fail-closed 哨兵）。
            filters = []
            if tenant_id is not None:
                filters.append({"tenant_id": tenant_id})
            if visible_department_ids is not None:
                # 关键安全关卡：空集合替换为不可能命中的哨兵值，防止“无可见部门”退化为全量可见。
                departments = [value for value in visible_department_ids if value]
                filters.append({"department_id": {"$in": departments}} if departments else {"department_id": "__no_visible_department__"})
            if len(filters) == 1:
                kwargs["filter"] = filters[0]
            elif filters:
                kwargs["filter"] = {"$and": filters}
            results = await self._store.asimilarity_search_with_score(query, **kwargs)
            out: list[tuple[dict, float]] = []
            for doc, score in results:
                meta = dict(doc.metadata or {})
                chunk_id = str(meta.get("chunk_id") or "").strip()
                doc_id = str(meta.get("source_document_id") or meta.get("doc_id") or "").strip()
                if not chunk_id or not doc_id:
                    continue
                meta["chunk_id"] = chunk_id
                meta["doc_id"] = doc_id
                meta["source_document_id"] = doc_id
                out.append(({
                    "content": doc.page_content,
                    "source": meta.get("source", ""),
                    "metadata": meta,
                }, score))
            return out

    async def delete_by_doc_id(self, doc_id: str, tenant_id: str | None = None) -> int:
        """按 doc_id 删除所有相关向量

        Args:
            tenant_id: 若提供则同时按租户过滤，防止跨租户删除
        """
        if self._backend == "chroma":
            where_filter: dict = {"doc_id": doc_id}
            if tenant_id is not None:
                where_filter = {"$and": [{"doc_id": doc_id}, {"tenant_id": tenant_id}]}
            existing = self._store.get(where=where_filter, include=[])
            ids = existing.get("ids", [])
            if ids:
                self._store.delete(ids=ids)
            return len(ids)
        return 0

    async def list_documents(self, tenant_id: str | None = None) -> list[dict]:
        """从向量库聚合文档列表（按 doc_id）"""
        if not self._store or self._backend != "chroma":
            return []
        where_filter: dict | None = None
        if tenant_id is not None:
            where_filter = {"tenant_id": tenant_id}
        kwargs: dict[str, Any] = {"include": ["metadatas"]}
        if where_filter is not None:
            kwargs["where"] = where_filter
        data = self._store.get(**kwargs)
        ids = data.get("ids") or []
        metas = data.get("metadatas") or []
        docs: dict[str, dict] = {}
        for _id, meta in zip(ids, metas):
            meta = meta or {}
            doc_id = str(meta.get("doc_id") or "")
            if not doc_id:
                continue
            item = docs.get(doc_id)
            if item is None:
                source = str(meta.get("source") or "")
                file_name = str(meta.get("file_name") or "") or (
                    source.rsplit("/", 1)[-1] if source else doc_id
                )
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "source": source,
                    "file_name": file_name,
                    "doc_type": str(meta.get("doc_type") or "unknown"),
                    "tenant_id": str(meta.get("tenant_id") or ""),
                    "chunks_count": 1,
                }
            else:
                item["chunks_count"] += 1
        return sorted(docs.values(), key=lambda d: d["file_name"])

    async def get_chunks_by_doc_id(
        self,
        doc_id: str,
        tenant_id: str | None = None,
    ) -> list[dict]:
        """按 doc_id 取回分块原文与元数据"""
        if not self._store or self._backend != "chroma":
            return []
        where_filter: dict = {"doc_id": doc_id}
        if tenant_id is not None:
            where_filter = {"$and": [{"doc_id": doc_id}, {"tenant_id": tenant_id}]}
        data = self._store.get(
            where=where_filter,
            include=["documents", "metadatas"],
        )
        ids = data.get("ids") or []
        documents = data.get("documents") or []
        metas = data.get("metadatas") or []
        chunks: list[dict] = []
        for chunk_id, content, meta in zip(ids, documents, metas):
            meta = meta or {}
            chunks.append({
                "chunk_id": chunk_id,
                "content": content or "",
                "chunk_index": int(meta.get("chunk_index") or 0),
                "doc_id": str(meta.get("doc_id") or doc_id),
                "doc_type": str(meta.get("doc_type") or ""),
                "source": str(meta.get("source") or ""),
                "tenant_id": str(meta.get("tenant_id") or ""),
            })
        chunks.sort(key=lambda c: c["chunk_index"])
        return chunks

    async def get_stats(self) -> dict:
        """获取向量库统计信息"""
        disconnected = {
            "backend": self._backend,
            "total_vectors": 0,
            "collection": self.COLLECTION_NAME,
            "status": "disconnected",
        }
        if not self._store:
            return disconnected
        try:
            if self._backend == "chroma":
                count = self._store.count()
                return {"backend": "chroma", "total_vectors": count, "collection": self.COLLECTION_NAME}
            return {"backend": "pgvector", "collection": self.COLLECTION_NAME}
        except Exception:
            return disconnected
