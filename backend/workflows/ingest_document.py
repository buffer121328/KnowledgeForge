"""LangGraph document-ingestion workflow."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Awaitable, Callable, TypedDict

from langgraph.graph import END, StateGraph
from langchain_openai import OpenAIEmbeddings

from agents.document_parser import DocParserAgent
from agents.knowledge_extractor import KnowledgeExtractAgent
from domain.documents import DocumentChunk, IngestDocumentInput
from domain.knowledge import ExtractionResult
from infrastructure.graph.neo4j_graph import KnowledgeGraphService
from infrastructure.retrieval.vector_store import VectorStoreService
from shared.config import settings
from workflows.ingestion_persistence import persist_extractions_to_graph


class IngestState(TypedDict, total=False):
    """Track ingest state."""
    file_paths: list[str]
    document_inputs: list[IngestDocumentInput]
    tenant_id: str
    progress_callback: Callable[[str, int, int, str], Awaitable[None] | None]
    chunks: list[DocumentChunk]
    extractions: list[ExtractionResult]
    vectors_stored: int
    entities_stored: int
    relations_stored: int
    sparse_stored: int


@dataclass
class MultimodalSearchResult:
    """Represent the result of multimodal search processing."""
    content: str
    modality: str
    score: float
    metadata: dict[str, Any]


class MultimodalService:
    """Document-ingestion helper for modality-aware embedding and reranking."""

    MODALITY_WEIGHTS = {
        "text": 1.0,
        "markdown": 1.0,
        "pdf": 0.95,
        "word": 0.95,
        "table": 0.9,
        "image": 0.85,
    }

    def __init__(self) -> None:
        """Initialize the multimodal service."""
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
            check_embedding_ctx_length=False,
            chunk_size=settings.embedding_batch_size,
        )

    async def embed_chunks(self, chunks: list[DocumentChunk]) -> list[list[float]]:
        """Handle embed chunks for the multimodal service."""
        return await self.embeddings.aembed_documents([chunk.content for chunk in chunks])

    async def embed_query(self, query: str) -> list[float]:
        """Handle embed query for the multimodal service."""
        return await self.embeddings.aembed_query(query)

    def weighted_rerank(self, results: list[tuple[DocumentChunk, float]]) -> list[MultimodalSearchResult]:
        """Handle weighted rerank for the multimodal service."""
        reranked = [
            MultimodalSearchResult(
                content=chunk.content,
                modality=chunk.doc_type.value,
                score=score * self.MODALITY_WEIGHTS.get(chunk.doc_type.value, 1.0),
                metadata=chunk.metadata,
            )
            for chunk, score in results
        ]
        return sorted(reranked, key=lambda result: result.score, reverse=True)




async def _emit_progress(state: IngestState, step: str, index: int, total: int, label: str) -> None:
    """Invoke the optional ingest progress callback if one is supplied."""

    callback = state.get("progress_callback")
    if callback is None:
        return
    result = callback(step, index, total, label)
    if inspect.isawaitable(result):
        await result

def build_ingest_document_workflow(
    doc_parser: DocParserAgent,
    extractor: KnowledgeExtractAgent,
    vector_store: VectorStoreService | None,
    knowledge_graph: KnowledgeGraphService | None,
    revision_store: Any = None,
    sparse_index: Any = None,
) -> StateGraph:
    """Build the ingest document workflow."""
    async def parse_documents(state: IngestState) -> dict:
        """Parse documents with aligned stable catalog context when supplied."""

        await _emit_progress(state, "parse", 1, 6, "解析文件")
        document_inputs = state.get("document_inputs")
        file_paths = state.get("file_paths", [])
        if document_inputs:
            file_paths = [item.file_path for item in document_inputs]
        parse_kwargs: dict[str, Any] = {"tenant_id": state.get("tenant_id", "")}
        if document_inputs is not None:
            parse_kwargs["document_inputs"] = document_inputs
        return {"chunks": await doc_parser.parse_batch(file_paths, **parse_kwargs)}

    async def extract_knowledge(state: IngestState) -> dict:
        """Extract the knowledge."""
        await _emit_progress(state, "extract", 2, 6, "知识抽取")
        return {"extractions": await extractor.extract(state.get("chunks", []))}

    async def store_vectors(state: IngestState) -> dict:
        """Store the vectors."""
        await _emit_progress(state, "store_vectors", 3, 6, "写向量库")
        chunks = state.get("chunks", [])
        return {"vectors_stored": await vector_store.add_chunks(chunks) if vector_store and chunks else 0}

    async def store_sparse(state: IngestState) -> dict:
        """Store canonical chunks in the tenant-partitioned native sparse index."""
        await _emit_progress(state, "store_sparse", 5, 6, "写稀疏索引")
        chunks = state.get("chunks", [])
        return {
            "sparse_stored":
                await sparse_index.add_chunks(chunks) if sparse_index and chunks else 0
        }

    async def store_graph(state: IngestState) -> dict:
        """Store extracted graph records with the same tenant/source contract as workers."""
        await _emit_progress(state, "store_graph", 4, 6, "写知识图谱")
        if not knowledge_graph:
            return {"entities_stored": 0, "relations_stored": 0}
        entities_stored, relations_stored = await persist_extractions_to_graph(
            chunks=state.get("chunks", []),
            extractions=state.get("extractions", []),
            knowledge_graph=knowledge_graph,
            tenant_id=state.get("tenant_id", ""),
        )
        return {
            "entities_stored": entities_stored,
            "relations_stored": relations_stored,
        }

    async def advance_revision(state: IngestState) -> dict:
        """Advance the tenant cache boundary only after both stores succeed."""
        await _emit_progress(state, "advance_revision", 6, 6, "收尾提交")
        tenant_id = state.get("tenant_id", "")
        if revision_store is not None and tenant_id:
            await revision_store.advance(tenant_id)
        return {}

    graph = StateGraph(IngestState)
    graph.add_node("parse", parse_documents)
    graph.add_node("extract", extract_knowledge)
    graph.add_node("store_vectors", store_vectors)
    graph.add_node("store_graph", store_graph)
    graph.add_node("store_sparse", store_sparse)
    graph.add_node("advance_revision", advance_revision)
    graph.set_entry_point("parse")
    graph.add_edge("parse", "extract")
    graph.add_edge("extract", "store_vectors")
    graph.add_edge("extract", "store_graph")
    graph.add_edge("extract", "store_sparse")
    graph.add_edge(["store_vectors", "store_graph", "store_sparse"], "advance_revision")
    graph.add_edge("advance_revision", END)
    return graph.compile()
